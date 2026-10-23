"""
Lambda function for medical image triage audit queries.
Handles audit trail queries and compliance reporting.
"""

import json
import boto3
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging
from boto3.dynamodb.conditions import Key, Attr

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
dynamodb = boto3.resource('dynamodb')

# Environment variables
DYNAMODB_TABLE = os.environ['DYNAMODB_TABLE']
ENVIRONMENT = os.environ['ENVIRONMENT']

# Initialize DynamoDB table
table = dynamodb.Table(DYNAMODB_TABLE)


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Main Lambda handler for audit queries.

    Handles API Gateway requests for audit trail queries.
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

        # Check authorization - only physicians and admins can access audit data
        if not any(group in user_groups for group in ['physicians', 'administrators']):
            return create_error_response(403, "Access denied: insufficient permissions")

        if http_method == 'GET':
            if path_parameters.get('image_hash'):
                # Get specific audit record by image hash
                return get_audit_by_image_hash(path_parameters['image_hash'])
            else:
                # Query audit records with filters
                return query_audit_records(query_parameters)
        else:
            return create_error_response(405, f"Method {http_method} not allowed")

    except Exception as e:
        logger.error(f"Error processing audit request: {str(e)}", exc_info=True)
        return create_error_response(500, f"Internal server error: {str(e)}")


def get_audit_by_image_hash(image_hash: str) -> Dict[str, Any]:
    """Get audit record by image hash."""
    try:
        # Query by image_hash (partition key)
        response = table.query(
            KeyConditionExpression=Key('image_hash').eq(image_hash),
            ScanIndexForward=False,  # Most recent first
            Limit=1
        )

        if not response['Items']:
            return create_error_response(404, f"No audit record found for image hash: {image_hash}")

        audit_record = response['Items'][0]

        # Format response
        formatted_record = format_audit_record(audit_record)

        return create_success_response(formatted_record)

    except Exception as e:
        logger.error(f"Error querying audit by image hash: {str(e)}")
        return create_error_response(500, f"Query failed: {str(e)}")


def query_audit_records(query_params: Dict[str, str]) -> Dict[str, Any]:
    """Query audit records with filters."""
    try:
        # Parse query parameters
        limit = min(int(query_params.get('limit', 100)), 1000)  # Max 1000 records
        start_date = query_params.get('start_date')
        end_date = query_params.get('end_date')
        routing_decision = query_params.get('routing_decision')
        predicted_class = query_params.get('predicted_class')
        patient_id = query_params.get('patient_id')  # Will be hashed
        study_id = query_params.get('study_id')

        # Determine query strategy
        if start_date and end_date:
            # Time-range query using TimestampIndex GSI
            return query_by_time_range(start_date, end_date, limit, routing_decision, predicted_class)
        elif routing_decision:
            # Query by routing decision using RoutingDecisionIndex GSI
            return query_by_routing_decision(routing_decision, limit, start_date, end_date)
        elif patient_id:
            # Query by patient using PatientIndex GSI
            return query_by_patient(patient_id, limit)
        else:
            # General scan with filters
            return scan_with_filters(query_params, limit)

    except Exception as e:
        logger.error(f"Error querying audit records: {str(e)}")
        return create_error_response(500, f"Query failed: {str(e)}")


def query_by_time_range(
    start_date: str,
    end_date: str,
    limit: int,
    routing_decision: Optional[str] = None,
    predicted_class: Optional[str] = None
) -> Dict[str, Any]:
    """Query audit records by time range using TimestampIndex GSI."""
    try:
        # Parse dates
        start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))

        # Generate date partitions to query
        date_partitions = []
        current_date = start_dt.date()
        end_date_only = end_dt.date()

        while current_date <= end_date_only:
            date_partitions.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=1)

        all_items = []

        # Query each date partition
        for date_partition in date_partitions:
            key_condition = Key('date_partition').eq(date_partition) & \
                          Key('timestamp').between(start_dt.isoformat(), end_dt.isoformat())

            response = table.query(
                IndexName='TimestampIndex',
                KeyConditionExpression=key_condition,
                ScanIndexForward=False,  # Most recent first
                Limit=limit
            )

            items = response['Items']

            # Apply additional filters
            if routing_decision:
                items = [item for item in items if item.get('routing_decision') == routing_decision]

            if predicted_class:
                items = [item for item in items if item.get('predicted_class') == predicted_class]

            all_items.extend(items)

            # Break if we have enough items
            if len(all_items) >= limit:
                break

        # Sort by timestamp (most recent first) and limit
        all_items.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        all_items = all_items[:limit]

        # Format results
        formatted_records = [format_audit_record(record) for record in all_items]

        return create_success_response({
            'records': formatted_records,
            'count': len(formatted_records),
            'query_info': {
                'start_date': start_date,
                'end_date': end_date,
                'routing_decision': routing_decision,
                'predicted_class': predicted_class,
                'limit': limit
            }
        })

    except Exception as e:
        logger.error(f"Error in time range query: {str(e)}")
        return create_error_response(500, f"Time range query failed: {str(e)}")


def query_by_routing_decision(
    routing_decision: str,
    limit: int,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, Any]:
    """Query audit records by routing decision using RoutingDecisionIndex GSI."""
    try:
        key_condition = Key('routing_decision').eq(routing_decision)

        # Add time range if provided
        if start_date and end_date:
            start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            key_condition = key_condition & Key('timestamp').between(start_dt.isoformat(), end_dt.isoformat())

        response = table.query(
            IndexName='RoutingDecisionIndex',
            KeyConditionExpression=key_condition,
            ScanIndexForward=False,  # Most recent first
            Limit=limit
        )

        items = response['Items']

        # Format results
        formatted_records = [format_audit_record(record) for record in items]

        return create_success_response({
            'records': formatted_records,
            'count': len(formatted_records),
            'query_info': {
                'routing_decision': routing_decision,
                'start_date': start_date,
                'end_date': end_date,
                'limit': limit
            }
        })

    except Exception as e:
        logger.error(f"Error in routing decision query: {str(e)}")
        return create_error_response(500, f"Routing decision query failed: {str(e)}")


def query_by_patient(patient_id: str, limit: int) -> Dict[str, Any]:
    """Query audit records by patient using PatientIndex GSI."""
    try:
        import hashlib

        # Hash patient ID for HIPAA compliance
        patient_id_hash = hashlib.sha256(patient_id.encode()).hexdigest()

        response = table.query(
            IndexName='PatientIndex',
            KeyConditionExpression=Key('patient_id_hash').eq(patient_id_hash),
            ScanIndexForward=False,  # Most recent first
            Limit=limit
        )

        items = response['Items']

        # Format results
        formatted_records = [format_audit_record(record) for record in items]

        return create_success_response({
            'records': formatted_records,
            'count': len(formatted_records),
            'query_info': {
                'patient_id_hash': patient_id_hash,
                'limit': limit
            }
        })

    except Exception as e:
        logger.error(f"Error in patient query: {str(e)}")
        return create_error_response(500, f"Patient query failed: {str(e)}")


def scan_with_filters(query_params: Dict[str, str], limit: int) -> Dict[str, Any]:
    """Scan table with filters for general queries."""
    try:
        # Build filter expression
        filter_expr = None

        if query_params.get('predicted_class'):
            filter_expr = Attr('predicted_class').eq(query_params['predicted_class'])

        if query_params.get('confidence_min'):
            confidence_filter = Attr('confidence').gte(float(query_params['confidence_min']))
            filter_expr = filter_expr & confidence_filter if filter_expr else confidence_filter

        if query_params.get('confidence_max'):
            confidence_filter = Attr('confidence').lte(float(query_params['confidence_max']))
            filter_expr = filter_expr & confidence_filter if filter_expr else confidence_filter

        # Scan with filter
        scan_kwargs = {
            'Limit': limit,
            'Select': 'ALL_ATTRIBUTES'
        }

        if filter_expr:
            scan_kwargs['FilterExpression'] = filter_expr

        response = table.scan(**scan_kwargs)
        items = response['Items']

        # Sort by timestamp (most recent first)
        items.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

        # Format results
        formatted_records = [format_audit_record(record) for record in items]

        return create_success_response({
            'records': formatted_records,
            'count': len(formatted_records),
            'query_info': {
                'scan_with_filters': True,
                'filters': query_params,
                'limit': limit
            }
        })

    except Exception as e:
        logger.error(f"Error in scan with filters: {str(e)}")
        return create_error_response(500, f"Scan query failed: {str(e)}")


def format_audit_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Format audit record for API response."""
    try:
        # Parse all_scores if it's a JSON string
        all_scores = record.get('all_scores')
        if isinstance(all_scores, str):
            try:
                all_scores = json.loads(all_scores)
            except:
                all_scores = {}

        formatted = {
            'prediction_id': record.get('prediction_id'),
            'image_hash': record.get('image_hash'),
            'patient_id_hash': record.get('patient_id_hash'),  # Already hashed
            'study_id': record.get('study_id'),
            'timestamp': record.get('timestamp'),
            'predicted_class': record.get('predicted_class'),
            'confidence': float(record.get('confidence', 0)),
            'confidence_level': record.get('confidence_level'),
            'all_scores': all_scores,
            'routing_decision': record.get('routing_decision'),
            'priority_level': int(record.get('priority_level', 0)),
            'assigned_reviewer_type': record.get('assigned_reviewer_type'),
            'reasoning': record.get('reasoning'),
            'estimated_review_time': record.get('estimated_review_time'),
            'processing_time_ms': float(record.get('processing_time_ms', 0)),
            'uploaded_by_hash': record.get('uploaded_by_hash'),  # Already hashed
            'created_at': record.get('created_at')
        }

        # Remove None values
        return {k: v for k, v in formatted.items() if v is not None}

    except Exception as e:
        logger.error(f"Error formatting audit record: {str(e)}")
        return record


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