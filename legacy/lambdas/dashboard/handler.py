"""
Lambda function for medical image triage dashboard metrics.
Provides real-time analytics and model drift detection.
"""

import json
import boto3
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List
import logging
from boto3.dynamodb.conditions import Key, Attr
from decimal import Decimal

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
dynamodb = boto3.resource('dynamodb')
cloudwatch = boto3.client('cloudwatch')

# Environment variables
DYNAMODB_TABLE = os.environ['DYNAMODB_TABLE']
ENVIRONMENT = os.environ['ENVIRONMENT']

# Initialize DynamoDB table
table = dynamodb.Table(DYNAMODB_TABLE)


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Main Lambda handler for dashboard metrics.

    Handles API Gateway requests for dashboard data.
    """
    try:
        logger.info(f"Received event: {json.dumps(event, default=str)}")

        # Parse request
        http_method = event.get('httpMethod')
        path_parameters = event.get('pathParameters', {})
        query_parameters = event.get('queryStringParameters') or {}

        # Extract user info from Cognito authorizer
        user_context = event.get('requestContext', {}).get('authorizer', {})
        user_groups = user_context.get('claims', {}).get('cognito:groups', '')

        # Check authorization - only physicians and admins can access dashboard
        if not any(group in user_groups for group in ['physicians', 'administrators']):
            return create_error_response(403, "Access denied: insufficient permissions")

        if http_method == 'GET':
            # Determine which metrics to return based on path
            resource_path = event.get('resource', '')

            if 'metrics' in resource_path:
                # General dashboard metrics
                days_lookback = int(query_parameters.get('days', 30))
                return get_dashboard_metrics(days_lookback)

            elif 'drift' in resource_path:
                # Model drift metrics
                days_lookback = int(query_parameters.get('days', 7))
                return get_drift_metrics(days_lookback)

            else:
                return create_error_response(404, "Metrics endpoint not found")

        else:
            return create_error_response(405, f"Method {http_method} not allowed")

    except Exception as e:
        logger.error(f"Error processing dashboard request: {str(e)}", exc_info=True)
        return create_error_response(500, f"Internal server error: {str(e)}")


def get_dashboard_metrics(days_lookback: int) -> Dict[str, Any]:
    """Calculate comprehensive dashboard metrics."""
    try:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_lookback)
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

        # Query predictions in the time range
        predictions = query_predictions_by_time_range(start_date, end_date)
        predictions_today = query_predictions_by_time_range(today_start, end_date)

        # Calculate metrics
        total_predictions = len(predictions)
        predictions_today_count = len(predictions_today)

        # Confidence distribution
        confidence_distribution = calculate_confidence_distribution(predictions)

        # Classification distribution
        classification_distribution = calculate_classification_distribution(predictions)

        # Triage distribution
        triage_distribution = calculate_triage_distribution(predictions)

        # Average metrics
        avg_confidence = calculate_average_confidence(predictions)
        avg_processing_time = calculate_average_processing_time(predictions)

        # Review metrics (mocked for now - would be calculated from review data)
        pending_reviews = sum(triage_distribution.get(decision, 0)
                            for decision in ['expedited_review', 'senior_review'])
        completed_reviews = len([p for p in predictions if p.get('reviewer_id')])
        avg_review_time = calculate_average_review_time(predictions)

        # Accuracy rate (approximate from review data)
        accuracy_rate = calculate_accuracy_rate(predictions)

        dashboard_metrics = {
            'total_predictions': total_predictions,
            'predictions_today': predictions_today_count,
            'accuracy_rate': accuracy_rate,
            'average_confidence': avg_confidence,
            'confidence_distribution': confidence_distribution,
            'classification_distribution': classification_distribution,
            'triage_distribution': triage_distribution,
            'average_processing_time_ms': avg_processing_time,
            'pending_reviews': pending_reviews,
            'completed_reviews': completed_reviews,
            'average_review_time_hours': avg_review_time,
            'period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'days': days_lookback
            }
        }

        return create_success_response(dashboard_metrics)

    except Exception as e:
        logger.error(f"Error calculating dashboard metrics: {str(e)}")
        return create_error_response(500, f"Dashboard metrics calculation failed: {str(e)}")


def get_drift_metrics(days_lookback: int) -> Dict[str, Any]:
    """Calculate model drift metrics."""
    try:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_lookback)

        # Query predictions for the period
        predictions = query_predictions_by_time_range(start_date, end_date)

        # Calculate confidence trend over time
        confidence_trend = calculate_confidence_trend(predictions, days_lookback)

        # Calculate distribution changes
        distribution_change = calculate_distribution_change(predictions)

        # Detect confidence decline alerts
        decline_alerts = detect_confidence_decline(confidence_trend)

        # Generate recommendations
        recommendations = generate_drift_recommendations(
            confidence_trend, distribution_change, decline_alerts
        )

        drift_metrics = {
            'period_start': start_date.isoformat(),
            'period_end': end_date.isoformat(),
            'average_confidence_trend': confidence_trend,
            'classification_distribution_change': distribution_change,
            'confidence_decline_alerts': decline_alerts,
            'recommendations': recommendations
        }

        return create_success_response(drift_metrics)

    except Exception as e:
        logger.error(f"Error calculating drift metrics: {str(e)}")
        return create_error_response(500, f"Drift metrics calculation failed: {str(e)}")


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

        # Convert Decimal types to float for JSON serialization
        for prediction in all_predictions:
            for key, value in prediction.items():
                if isinstance(value, Decimal):
                    prediction[key] = float(value)

        return all_predictions

    except Exception as e:
        logger.error(f"Error querying predictions by time range: {str(e)}")
        return []


def calculate_confidence_distribution(predictions: List[Dict[str, Any]]) -> Dict[str, int]:
    """Calculate confidence level distribution."""
    distribution = {'high': 0, 'medium': 0, 'low': 0}

    for pred in predictions:
        confidence_level = pred.get('confidence_level', 'low')
        if confidence_level in distribution:
            distribution[confidence_level] += 1

    return distribution


def calculate_classification_distribution(predictions: List[Dict[str, Any]]) -> Dict[str, int]:
    """Calculate predicted class distribution."""
    distribution = {}

    for pred in predictions:
        predicted_class = pred.get('predicted_class', 'Unknown')
        distribution[predicted_class] = distribution.get(predicted_class, 0) + 1

    return distribution


def calculate_triage_distribution(predictions: List[Dict[str, Any]]) -> Dict[str, int]:
    """Calculate triage decision distribution."""
    distribution = {}

    for pred in predictions:
        triage_decision = pred.get('routing_decision', 'unknown')
        distribution[triage_decision] = distribution.get(triage_decision, 0) + 1

    return distribution


def calculate_average_confidence(predictions: List[Dict[str, Any]]) -> float:
    """Calculate average confidence score."""
    if not predictions:
        return 0.0

    total_confidence = sum(float(pred.get('confidence', 0)) for pred in predictions)
    return total_confidence / len(predictions)


def calculate_average_processing_time(predictions: List[Dict[str, Any]]) -> float:
    """Calculate average processing time."""
    if not predictions:
        return 0.0

    total_time = sum(float(pred.get('processing_time_ms', 0)) for pred in predictions)
    return total_time / len(predictions)


def calculate_average_review_time(predictions: List[Dict[str, Any]]) -> float:
    """Calculate average review time in hours."""
    reviewed_predictions = [p for p in predictions if p.get('review_timestamp')]

    if not reviewed_predictions:
        return None

    total_review_time_hours = 0
    for pred in reviewed_predictions:
        try:
            created_time = datetime.fromisoformat(pred.get('created_at', '').replace('Z', '+00:00'))
            review_time = datetime.fromisoformat(pred.get('review_timestamp', '').replace('Z', '+00:00'))
            review_duration_hours = (review_time - created_time).total_seconds() / 3600
            total_review_time_hours += review_duration_hours
        except:
            continue

    return total_review_time_hours / len(reviewed_predictions) if reviewed_predictions else None


def calculate_accuracy_rate(predictions: List[Dict[str, Any]]) -> float:
    """Calculate accuracy rate from reviewed predictions."""
    reviewed_predictions = [p for p in predictions if p.get('review_decision')]

    if not reviewed_predictions:
        return 0.0

    correct_predictions = sum(
        1 for pred in reviewed_predictions
        if pred.get('review_decision') in ['approved', 'confirmed']
    )

    return correct_predictions / len(reviewed_predictions)


def calculate_confidence_trend(predictions: List[Dict[str, Any]], days: int) -> List[Dict[str, Any]]:
    """Calculate confidence trend over time."""
    try:
        # Group predictions by day
        daily_data = {}

        for pred in predictions:
            try:
                timestamp = datetime.fromisoformat(pred.get('timestamp', '').replace('Z', '+00:00'))
                date_key = timestamp.date().isoformat()

                if date_key not in daily_data:
                    daily_data[date_key] = []

                daily_data[date_key].append(float(pred.get('confidence', 0)))
            except:
                continue

        # Calculate daily averages
        trend_data = []
        for date_key in sorted(daily_data.keys()):
            confidences = daily_data[date_key]
            avg_confidence = sum(confidences) / len(confidences)

            trend_data.append({
                'date': date_key,
                'average_confidence': avg_confidence,
                'prediction_count': len(confidences)
            })

        return trend_data

    except Exception as e:
        logger.error(f"Error calculating confidence trend: {str(e)}")
        return []


def calculate_distribution_change(predictions: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate change in classification distribution over time."""
    try:
        if len(predictions) < 10:  # Need sufficient data
            return {}

        # Split predictions in half by time
        predictions.sort(key=lambda x: x.get('timestamp', ''))
        mid_point = len(predictions) // 2

        first_half = predictions[:mid_point]
        second_half = predictions[mid_point:]

        # Calculate distributions
        first_dist = calculate_classification_distribution(first_half)
        second_dist = calculate_classification_distribution(second_half)

        # Calculate changes
        changes = {}
        all_classes = set(list(first_dist.keys()) + list(second_dist.keys()))

        for class_name in all_classes:
            first_count = first_dist.get(class_name, 0)
            second_count = second_dist.get(class_name, 0)

            first_total = sum(first_dist.values()) or 1
            second_total = sum(second_dist.values()) or 1

            first_rate = first_count / first_total
            second_rate = second_count / second_total

            change = second_rate - first_rate
            changes[class_name] = change

        return changes

    except Exception as e:
        logger.error(f"Error calculating distribution change: {str(e)}")
        return {}


def detect_confidence_decline(confidence_trend: List[Dict[str, Any]]) -> List[str]:
    """Detect significant confidence decline patterns."""
    if len(confidence_trend) < 3:
        return []

    alerts = []
    confidences = [point["average_confidence"] for point in confidence_trend]

    # Check for sustained decline
    if len(confidences) >= 3:
        recent_avg = sum(confidences[-3:]) / 3
        early_avg = sum(confidences[:3]) / 3

        if recent_avg < early_avg - 0.1:  # 10% decline
            alerts.append(f"Sustained confidence decline detected: {early_avg:.3f} → {recent_avg:.3f}")

    # Check for sudden drop
    for i in range(1, len(confidences)):
        if confidences[i] < confidences[i-1] - 0.15:  # 15% sudden drop
            alerts.append(f"Sudden confidence drop on {confidence_trend[i]['date']}: {confidences[i-1]:.3f} → {confidences[i]:.3f}")

    # Check if recent confidence is below threshold
    if confidences and confidences[-1] < 0.7:
        alerts.append(f"Current average confidence below threshold: {confidences[-1]:.3f}")

    return alerts


def generate_drift_recommendations(
    confidence_trend: List[Dict[str, Any]],
    distribution_change: Dict[str, float],
    decline_alerts: List[str]
) -> List[str]:
    """Generate recommendations based on drift analysis."""
    recommendations = []

    # Confidence-based recommendations
    if decline_alerts:
        recommendations.append("Model performance monitoring required - confidence levels declining")
        recommendations.append("Consider retraining model with recent data")

    # Distribution change recommendations
    for class_name, change in distribution_change.items():
        if abs(change) > 0.1:  # 10% change
            if change > 0:
                recommendations.append(f"Increase in {class_name} predictions (+{change:.1%}) - verify data quality")
            else:
                recommendations.append(f"Decrease in {class_name} predictions ({change:.1%}) - check for missing cases")

    # Data quality recommendations
    if confidence_trend:
        recent_avg = confidence_trend[-1]["average_confidence"] if confidence_trend else 0
        if recent_avg < 0.75:
            recommendations.append("Review input data quality - low confidence may indicate data shift")

    # General recommendations
    if not recommendations:
        recommendations.append("Model performance appears stable - continue monitoring")

    return recommendations


def create_success_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create successful API response."""
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type,Authorization',
            'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
        },
        'body': json.dumps(data, default=str)
    }


def create_error_response(status_code: int, message: str) -> Dict[str, Any]:
    """Create error API response."""
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type,Authorization',
            'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
        },
        'body': json.dumps({
            'error': message,
            'timestamp': datetime.utcnow().isoformat()
        })
    }