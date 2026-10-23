"""
Lambda function for model drift detection.
Scheduled to run periodically to detect model performance degradation.
"""

import json
import boto3
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List
import logging
from boto3.dynamodb.conditions import Key
from decimal import Decimal

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
dynamodb = boto3.resource('dynamodb')
sns = boto3.client('sns')
cloudwatch = boto3.client('cloudwatch')

# Environment variables
DYNAMODB_TABLE = os.environ['DYNAMODB_TABLE']
ALERT_TOPIC = os.environ['ALERT_TOPIC']
CONFIDENCE_THRESHOLD = float(os.environ.get('CONFIDENCE_THRESHOLD', '0.7'))
ENVIRONMENT = os.environ['ENVIRONMENT']

# Initialize DynamoDB table
table = dynamodb.Table(DYNAMODB_TABLE)


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Main Lambda handler for model drift detection.

    This function is triggered by CloudWatch Events on a schedule.
    """
    try:
        logger.info("Starting model drift detection analysis")

        # Analyze recent predictions for drift
        drift_analysis = analyze_model_drift()

        # Send alerts if drift is detected
        if drift_analysis['alerts']:
            send_drift_alerts(drift_analysis)

        # Send custom metrics to CloudWatch
        send_drift_metrics(drift_analysis)

        logger.info("Model drift detection completed successfully")

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Drift detection completed',
                'analysis': drift_analysis
            }, default=str)
        }

    except Exception as e:
        logger.error(f"Error in drift detection: {str(e)}", exc_info=True)

        # Send error alert
        send_error_alert(str(e))

        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': f"Drift detection failed: {str(e)}"
            })
        }


def analyze_model_drift() -> Dict[str, Any]:
    """Analyze recent predictions for model drift indicators."""

    # Get predictions from the last 24 hours and previous 24 hours for comparison
    now = datetime.utcnow()
    recent_end = now
    recent_start = now - timedelta(hours=24)
    previous_end = recent_start
    previous_start = previous_end - timedelta(hours=24)

    # Query recent and previous predictions
    recent_predictions = query_predictions_by_time_range(recent_start, recent_end)
    previous_predictions = query_predictions_by_time_range(previous_start, previous_end)

    logger.info(f"Analyzing {len(recent_predictions)} recent predictions vs {len(previous_predictions)} previous predictions")

    # Calculate drift metrics
    drift_analysis = {
        'timestamp': now.isoformat(),
        'period': {
            'recent': {'start': recent_start.isoformat(), 'end': recent_end.isoformat()},
            'previous': {'start': previous_start.isoformat(), 'end': previous_end.isoformat()}
        },
        'recent_count': len(recent_predictions),
        'previous_count': len(previous_predictions),
        'alerts': [],
        'metrics': {}
    }

    if not recent_predictions:
        logger.warning("No recent predictions found for drift analysis")
        return drift_analysis

    # Analyze confidence drift
    confidence_drift = analyze_confidence_drift(recent_predictions, previous_predictions)
    drift_analysis['metrics']['confidence'] = confidence_drift

    if confidence_drift['drift_detected']:
        drift_analysis['alerts'].append({
            'type': 'confidence_drift',
            'severity': 'high' if confidence_drift['recent_avg'] < CONFIDENCE_THRESHOLD else 'medium',
            'message': f"Confidence drift detected: {confidence_drift['change']:.1%} decrease",
            'details': confidence_drift
        })

    # Analyze distribution drift
    distribution_drift = analyze_distribution_drift(recent_predictions, previous_predictions)
    drift_analysis['metrics']['distribution'] = distribution_drift

    if distribution_drift['significant_changes']:
        drift_analysis['alerts'].append({
            'type': 'distribution_drift',
            'severity': 'medium',
            'message': f"Classification distribution drift detected in {len(distribution_drift['significant_changes'])} classes",
            'details': distribution_drift
        })

    # Analyze volume drift
    volume_drift = analyze_volume_drift(recent_predictions, previous_predictions)
    drift_analysis['metrics']['volume'] = volume_drift

    if volume_drift['significant_change']:
        drift_analysis['alerts'].append({
            'type': 'volume_drift',
            'severity': 'low' if abs(volume_drift['change']) < 0.5 else 'medium',
            'message': f"Prediction volume changed by {volume_drift['change']:.1%}",
            'details': volume_drift
        })

    # Analyze error patterns
    error_patterns = analyze_error_patterns(recent_predictions)
    drift_analysis['metrics']['errors'] = error_patterns

    if error_patterns['high_error_rate']:
        drift_analysis['alerts'].append({
            'type': 'error_pattern',
            'severity': 'high',
            'message': f"High error rate detected: {error_patterns['error_rate']:.1%}",
            'details': error_patterns
        })

    return drift_analysis


def query_predictions_by_time_range(start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
    """Query predictions by time range using date partitions."""
    try:
        all_predictions = []

        # Generate date partitions to query
        current_date = start_date.date()
        end_date_only = end_date.date()

        while current_date <= end_date_only:
            date_partition = current_date.strftime('%Y-%m-%d')

            try:
                response = table.query(
                    IndexName='TimestampIndex',
                    KeyConditionExpression=Key('date_partition').eq(date_partition) & \
                                         Key('timestamp').between(start_date.isoformat(), end_date.isoformat()),
                    ScanIndexForward=False
                )

                all_predictions.extend(response['Items'])

                # Handle pagination
                while 'LastEvaluatedKey' in response:
                    response = table.query(
                        IndexName='TimestampIndex',
                        KeyConditionExpression=Key('date_partition').eq(date_partition) & \
                                             Key('timestamp').between(start_date.isoformat(), end_date.isoformat()),
                        ScanIndexForward=False,
                        ExclusiveStartKey=response['LastEvaluatedKey']
                    )
                    all_predictions.extend(response['Items'])

            except Exception as e:
                logger.warning(f"Error querying date partition {date_partition}: {str(e)}")
                continue

            current_date += timedelta(days=1)

        # Convert Decimal types to float for calculations
        for prediction in all_predictions:
            for key, value in prediction.items():
                if isinstance(value, Decimal):
                    prediction[key] = float(value)

        return all_predictions

    except Exception as e:
        logger.error(f"Error querying predictions by time range: {str(e)}")
        return []


def analyze_confidence_drift(recent_predictions: List[Dict], previous_predictions: List[Dict]) -> Dict[str, Any]:
    """Analyze confidence score drift between periods."""

    recent_confidences = [float(p.get('confidence', 0)) for p in recent_predictions]
    previous_confidences = [float(p.get('confidence', 0)) for p in previous_predictions] if previous_predictions else []

    recent_avg = sum(recent_confidences) / len(recent_confidences) if recent_confidences else 0
    previous_avg = sum(previous_confidences) / len(previous_confidences) if previous_confidences else recent_avg

    change = recent_avg - previous_avg
    drift_detected = False

    # Detect drift conditions
    if recent_avg < CONFIDENCE_THRESHOLD:
        drift_detected = True
    elif previous_confidences and abs(change) > 0.1:  # 10% change
        drift_detected = True
    elif len(recent_confidences) >= 10:
        # Check for sustained decline trend within recent period
        mid_point = len(recent_confidences) // 2
        early_recent = recent_confidences[:mid_point]
        late_recent = recent_confidences[mid_point:]

        if early_recent and late_recent:
            early_avg = sum(early_recent) / len(early_recent)
            late_avg = sum(late_recent) / len(late_recent)

            if late_avg < early_avg - 0.05:  # 5% decline within recent period
                drift_detected = True

    return {
        'recent_avg': recent_avg,
        'previous_avg': previous_avg,
        'change': change,
        'drift_detected': drift_detected,
        'recent_count': len(recent_confidences),
        'previous_count': len(previous_confidences)
    }


def analyze_distribution_drift(recent_predictions: List[Dict], previous_predictions: List[Dict]) -> Dict[str, Any]:
    """Analyze classification distribution drift between periods."""

    # Calculate distributions
    recent_dist = {}
    for pred in recent_predictions:
        class_name = pred.get('predicted_class', 'Unknown')
        recent_dist[class_name] = recent_dist.get(class_name, 0) + 1

    previous_dist = {}
    for pred in previous_predictions:
        class_name = pred.get('predicted_class', 'Unknown')
        previous_dist[class_name] = previous_dist.get(class_name, 0) + 1

    # Normalize to percentages
    recent_total = sum(recent_dist.values()) or 1
    previous_total = sum(previous_dist.values()) or 1

    recent_pct = {k: v / recent_total for k, v in recent_dist.items()}
    previous_pct = {k: v / previous_total for k, v in previous_dist.items()}

    # Calculate changes
    all_classes = set(list(recent_pct.keys()) + list(previous_pct.keys()))
    changes = {}
    significant_changes = []

    for class_name in all_classes:
        recent_rate = recent_pct.get(class_name, 0)
        previous_rate = previous_pct.get(class_name, 0)
        change = recent_rate - previous_rate
        changes[class_name] = change

        # Consider changes > 10 percentage points as significant
        if abs(change) > 0.1:
            significant_changes.append({
                'class': class_name,
                'change': change,
                'recent_rate': recent_rate,
                'previous_rate': previous_rate
            })

    return {
        'recent_distribution': recent_pct,
        'previous_distribution': previous_pct,
        'changes': changes,
        'significant_changes': significant_changes,
        'drift_detected': len(significant_changes) > 0
    }


def analyze_volume_drift(recent_predictions: List[Dict], previous_predictions: List[Dict]) -> Dict[str, Any]:
    """Analyze prediction volume changes."""

    recent_count = len(recent_predictions)
    previous_count = len(previous_predictions)

    if previous_count == 0:
        change = 0.0
        significant_change = False
    else:
        change = (recent_count - previous_count) / previous_count
        significant_change = abs(change) > 0.3  # 30% change threshold

    return {
        'recent_count': recent_count,
        'previous_count': previous_count,
        'change': change,
        'significant_change': significant_change
    }


def analyze_error_patterns(predictions: List[Dict]) -> Dict[str, Any]:
    """Analyze error patterns in recent predictions."""

    if not predictions:
        return {
            'error_rate': 0.0,
            'high_error_rate': False,
            'error_patterns': {}
        }

    # Mock error analysis - in practice, you'd track actual prediction errors
    # This could be enhanced by comparing with review decisions or ground truth

    # For now, consider very low confidence predictions as potential errors
    low_confidence_predictions = [p for p in predictions if float(p.get('confidence', 1)) < 0.5]
    error_rate = len(low_confidence_predictions) / len(predictions)

    high_error_rate = error_rate > 0.1  # 10% error rate threshold

    # Analyze error patterns by class
    error_patterns = {}
    for pred in low_confidence_predictions:
        class_name = pred.get('predicted_class', 'Unknown')
        error_patterns[class_name] = error_patterns.get(class_name, 0) + 1

    return {
        'error_rate': error_rate,
        'high_error_rate': high_error_rate,
        'error_patterns': error_patterns,
        'low_confidence_count': len(low_confidence_predictions)
    }


def send_drift_alerts(drift_analysis: Dict[str, Any]) -> None:
    """Send SNS alerts for detected drift."""
    try:
        for alert in drift_analysis['alerts']:
            message = {
                'alert_type': 'model_drift',
                'severity': alert['severity'],
                'environment': ENVIRONMENT,
                'timestamp': drift_analysis['timestamp'],
                'drift_type': alert['type'],
                'message': alert['message'],
                'details': alert['details'],
                'recommendation': get_drift_recommendation(alert)
            }

            sns.publish(
                TopicArn=ALERT_TOPIC,
                Message=json.dumps(message, default=str),
                Subject=f"Medical Triage Model Drift Alert - {alert['type']} ({alert['severity']})",
                MessageAttributes={
                    'environment': {
                        'DataType': 'String',
                        'StringValue': ENVIRONMENT
                    },
                    'severity': {
                        'DataType': 'String',
                        'StringValue': alert['severity']
                    },
                    'drift_type': {
                        'DataType': 'String',
                        'StringValue': alert['type']
                    }
                }
            )

            logger.info(f"Sent drift alert: {alert['type']} - {alert['severity']}")

    except Exception as e:
        logger.error(f"Failed to send drift alerts: {str(e)}")


def get_drift_recommendation(alert: Dict[str, Any]) -> str:
    """Get recommendation based on drift alert type."""

    recommendations = {
        'confidence_drift': [
            "Investigate recent data quality changes",
            "Consider model retraining with recent data",
            "Review input preprocessing pipeline",
            "Check for data distribution shifts"
        ],
        'distribution_drift': [
            "Analyze patient population changes",
            "Review data collection procedures",
            "Consider updating class balance in training data",
            "Validate data labeling consistency"
        ],
        'volume_drift': [
            "Check system capacity and scaling",
            "Review data ingestion pipeline",
            "Investigate workflow changes",
            "Monitor system performance metrics"
        ],
        'error_pattern': [
            "Immediate model performance review required",
            "Consider temporary increase in human review threshold",
            "Investigate specific error-prone classes",
            "Review recent model deployment changes"
        ]
    }

    return ". ".join(recommendations.get(alert['type'], ["Review model performance"]))


def send_drift_metrics(drift_analysis: Dict[str, Any]) -> None:
    """Send drift metrics to CloudWatch."""
    try:
        timestamp = datetime.utcnow()

        # Confidence metrics
        confidence_metrics = drift_analysis['metrics'].get('confidence', {})
        if confidence_metrics:
            cloudwatch.put_metric_data(
                Namespace='MedicalTriage/Drift',
                MetricData=[
                    {
                        'MetricName': 'AverageConfidence',
                        'Value': confidence_metrics.get('recent_avg', 0),
                        'Unit': 'None',
                        'Timestamp': timestamp,
                        'Dimensions': [
                            {'Name': 'Environment', 'Value': ENVIRONMENT},
                            {'Name': 'Period', 'Value': 'Recent24h'}
                        ]
                    },
                    {
                        'MetricName': 'ConfidenceChange',
                        'Value': confidence_metrics.get('change', 0),
                        'Unit': 'None',
                        'Timestamp': timestamp,
                        'Dimensions': [
                            {'Name': 'Environment', 'Value': ENVIRONMENT}
                        ]
                    },
                    {
                        'MetricName': 'ConfidenceDriftDetected',
                        'Value': 1 if confidence_metrics.get('drift_detected') else 0,
                        'Unit': 'Count',
                        'Timestamp': timestamp,
                        'Dimensions': [
                            {'Name': 'Environment', 'Value': ENVIRONMENT}
                        ]
                    }
                ]
            )

        # Volume metrics
        volume_metrics = drift_analysis['metrics'].get('volume', {})
        if volume_metrics:
            cloudwatch.put_metric_data(
                Namespace='MedicalTriage/Drift',
                MetricData=[
                    {
                        'MetricName': 'PredictionVolumeChange',
                        'Value': volume_metrics.get('change', 0),
                        'Unit': 'Percent',
                        'Timestamp': timestamp,
                        'Dimensions': [
                            {'Name': 'Environment', 'Value': ENVIRONMENT}
                        ]
                    },
                    {
                        'MetricName': 'RecentPredictionCount',
                        'Value': volume_metrics.get('recent_count', 0),
                        'Unit': 'Count',
                        'Timestamp': timestamp,
                        'Dimensions': [
                            {'Name': 'Environment', 'Value': ENVIRONMENT}
                        ]
                    }
                ]
            )

        # Alert count
        cloudwatch.put_metric_data(
            Namespace='MedicalTriage/Drift',
            MetricData=[
                {
                    'MetricName': 'DriftAlertsCount',
                    'Value': len(drift_analysis['alerts']),
                    'Unit': 'Count',
                    'Timestamp': timestamp,
                    'Dimensions': [
                        {'Name': 'Environment', 'Value': ENVIRONMENT}
                    ]
                }
            ]
        )

        logger.info("Sent drift metrics to CloudWatch")

    except Exception as e:
        logger.error(f"Failed to send drift metrics: {str(e)}")


def send_error_alert(error_message: str) -> None:
    """Send error alert for drift detection failures."""
    try:
        message = {
            'alert_type': 'drift_detection_error',
            'severity': 'high',
            'environment': ENVIRONMENT,
            'timestamp': datetime.utcnow().isoformat(),
            'error': error_message,
            'recommendation': 'Investigate drift detection system immediately'
        }

        sns.publish(
            TopicArn=ALERT_TOPIC,
            Message=json.dumps(message, default=str),
            Subject=f"Medical Triage Drift Detection Error - {ENVIRONMENT}",
            MessageAttributes={
                'environment': {
                    'DataType': 'String',
                    'StringValue': ENVIRONMENT
                },
                'severity': {
                    'DataType': 'String',
                    'StringValue': 'high'
                }
            }
        )

        logger.info("Sent drift detection error alert")

    except Exception as e:
        logger.error(f"Failed to send error alert: {str(e)}")