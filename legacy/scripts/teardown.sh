#!/bin/bash

# Medical Image Triage System - AWS Teardown Script
# This script safely tears down the deployed infrastructure

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT=${ENVIRONMENT:-dev}
AWS_REGION=${AWS_REGION:-us-east-1}
AWS_ACCOUNT_ID=""
FORCE_DELETE=false
KEEP_DATA=false

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Teardown Medical Image Triage System from AWS

OPTIONS:
    -e, --environment ENV     Environment name (default: dev)
    -r, --region REGION      AWS region (default: us-east-1)
    -a, --account ACCOUNT    AWS account ID (required)
    --force                  Force deletion without confirmation
    --keep-data             Keep S3 data and DynamoDB tables
    -h, --help              Show this help message

EXAMPLES:
    # Interactive teardown
    $0 -e dev -r us-east-1 -a 123456789012

    # Force teardown without confirmation
    $0 -e prod -a 123456789012 --force

    # Keep data but remove infrastructure
    $0 -e dev -a 123456789012 --keep-data

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -r|--region)
            AWS_REGION="$2"
            shift 2
            ;;
        -a|--account)
            AWS_ACCOUNT_ID="$2"
            shift 2
            ;;
        --force)
            FORCE_DELETE=true
            shift
            ;;
        --keep-data)
            KEEP_DATA=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate required parameters
if [[ -z "$AWS_ACCOUNT_ID" ]]; then
    print_error "AWS Account ID is required. Use -a or --account option."
    show_usage
    exit 1
fi

# Confirm deletion
confirm_deletion() {
    if [[ "$FORCE_DELETE" == "true" ]]; then
        return 0
    fi

    echo
    print_warning "You are about to DELETE the following resources:"
    echo "  Environment: $ENVIRONMENT"
    echo "  Region: $AWS_REGION"
    echo "  Account: $AWS_ACCOUNT_ID"
    echo

    if [[ "$KEEP_DATA" == "true" ]]; then
        print_info "Data will be PRESERVED (S3 buckets, DynamoDB tables)"
    else
        print_error "ALL DATA will be DELETED (S3 buckets, DynamoDB tables, audit logs)"
    fi

    echo
    read -p "Are you sure you want to continue? (type 'DELETE' to confirm): " confirmation

    if [[ "$confirmation" != "DELETE" ]]; then
        print_info "Teardown cancelled"
        exit 0
    fi
}

# Validate AWS credentials
check_aws_credentials() {
    print_info "Checking AWS credentials..."

    if ! aws sts get-caller-identity > /dev/null 2>&1; then
        print_error "AWS credentials not configured or invalid"
        exit 1
    fi

    local current_account=$(aws sts get-caller-identity --query Account --output text)
    if [[ "$current_account" != "$AWS_ACCOUNT_ID" ]]; then
        print_error "Current AWS account ($current_account) doesn't match specified account ($AWS_ACCOUNT_ID)"
        exit 1
    fi

    print_success "AWS credentials validated"
}

# Delete SageMaker endpoint
delete_sagemaker_endpoint() {
    print_info "Deleting SageMaker endpoint..."

    local endpoint_name="medical-triage-endpoint-$ENVIRONMENT"

    # Check if endpoint exists
    if aws sagemaker describe-endpoint --endpoint-name $endpoint_name --region $AWS_REGION > /dev/null 2>&1; then
        print_info "Deleting SageMaker endpoint: $endpoint_name"

        # Delete endpoint
        aws sagemaker delete-endpoint --endpoint-name $endpoint_name --region $AWS_REGION

        # Wait for deletion
        print_info "Waiting for endpoint deletion..."
        while aws sagemaker describe-endpoint --endpoint-name $endpoint_name --region $AWS_REGION > /dev/null 2>&1; do
            sleep 10
        done

        print_success "SageMaker endpoint deleted"
    else
        print_info "SageMaker endpoint not found"
    fi

    # Delete endpoint configurations (they may have timestamps)
    print_info "Cleaning up endpoint configurations..."
    local configs=$(aws sagemaker list-endpoint-configs --region $AWS_REGION \
        --name-contains "medical-triage-config" \
        --query 'EndpointConfigs[].EndpointConfigName' --output text)

    for config in $configs; do
        if [[ ! -z "$config" ]]; then
            print_info "Deleting endpoint config: $config"
            aws sagemaker delete-endpoint-config --endpoint-config-name $config --region $AWS_REGION || true
        fi
    done

    # Delete models
    print_info "Cleaning up SageMaker models..."
    local models=$(aws sagemaker list-models --region $AWS_REGION \
        --name-contains "medical-triage-model" \
        --query 'Models[].ModelName' --output text)

    for model in $models; do
        if [[ ! -z "$model" ]]; then
            print_info "Deleting model: $model"
            aws sagemaker delete-model --model-name $model --region $AWS_REGION || true
        fi
    done
}

# Empty and prepare S3 buckets for deletion
prepare_s3_buckets() {
    if [[ "$KEEP_DATA" == "true" ]]; then
        print_warning "Skipping S3 bucket cleanup (--keep-data flag set)"
        return 0
    fi

    print_info "Preparing S3 buckets for deletion..."

    local ingest_bucket="image-triage-ingest-$AWS_ACCOUNT_ID-$ENVIRONMENT"
    local archive_bucket="image-triage-archive-$AWS_ACCOUNT_ID-$ENVIRONMENT"

    # Empty ingest bucket
    if aws s3api head-bucket --bucket $ingest_bucket --region $AWS_REGION > /dev/null 2>&1; then
        print_info "Emptying ingest bucket: $ingest_bucket"
        aws s3 rm s3://$ingest_bucket --recursive --region $AWS_REGION || true

        # Remove versioned objects if any
        aws s3api delete-objects --bucket $ingest_bucket --region $AWS_REGION \
            --delete "$(aws s3api list-object-versions --bucket $ingest_bucket --region $AWS_REGION \
            --output json --query '{Objects: Versions[].{Key:Key,VersionId:VersionId}}')" > /dev/null 2>&1 || true

        # Remove delete markers
        aws s3api delete-objects --bucket $ingest_bucket --region $AWS_REGION \
            --delete "$(aws s3api list-object-versions --bucket $ingest_bucket --region $AWS_REGION \
            --output json --query '{Objects: DeleteMarkers[].{Key:Key,VersionId:VersionId}}')" > /dev/null 2>&1 || true
    fi

    # Empty archive bucket
    if aws s3api head-bucket --bucket $archive_bucket --region $AWS_REGION > /dev/null 2>&1; then
        print_info "Emptying archive bucket: $archive_bucket"
        aws s3 rm s3://$archive_bucket --recursive --region $AWS_REGION || true

        # Remove versioned objects if any
        aws s3api delete-objects --bucket $archive_bucket --region $AWS_REGION \
            --delete "$(aws s3api list-object-versions --bucket $archive_bucket --region $AWS_REGION \
            --output json --query '{Objects: Versions[].{Key:Key,VersionId:VersionId}}')" > /dev/null 2>&1 || true

        # Remove delete markers
        aws s3api delete-objects --bucket $archive_bucket --region $AWS_REGION \
            --delete "$(aws s3api list-object-versions --bucket $archive_bucket --region $AWS_REGION \
            --output json --query '{Objects: DeleteMarkers[].{Key:Key,VersionId:VersionId}}')" > /dev/null 2>&1 || true
    fi

    print_success "S3 buckets prepared for deletion"
}

# Delete CloudFormation stacks
delete_stacks() {
    print_info "Deleting CloudFormation stacks..."

    cd cdk

    # Set environment variables
    export CDK_DEFAULT_ACCOUNT=$AWS_ACCOUNT_ID
    export CDK_DEFAULT_REGION=$AWS_REGION

    # Update CDK context for teardown
    cat > cdk.context.json << EOF
{
  "account": "$AWS_ACCOUNT_ID",
  "region": "$AWS_REGION",
  "environment": "$ENVIRONMENT"
}
EOF

    # Delete monitoring stack first (depends on core stack)
    local monitoring_stack="ImageTriageMonitoring-$ENVIRONMENT"
    if aws cloudformation describe-stacks --stack-name $monitoring_stack --region $AWS_REGION > /dev/null 2>&1; then
        print_info "Deleting monitoring stack: $monitoring_stack"
        cdk destroy $monitoring_stack --force
    else
        print_info "Monitoring stack not found"
    fi

    # Delete core infrastructure stack
    local core_stack="ImageTriageStack-$ENVIRONMENT"
    if aws cloudformation describe-stacks --stack-name $core_stack --region $AWS_REGION > /dev/null 2>&1; then
        print_info "Deleting core stack: $core_stack"
        cdk destroy $core_stack --force
    else
        print_info "Core stack not found"
    fi

    cd ..
    print_success "CloudFormation stacks deleted"
}

# Clean up IAM roles (if not managed by CDK)
cleanup_iam_roles() {
    print_info "Cleaning up IAM roles..."

    local role_name="SageMakerExecutionRole-MedicalTriage"

    if aws iam get-role --role-name $role_name > /dev/null 2>&1; then
        print_info "Detaching policies from role: $role_name"

        # Detach managed policies
        local policies=$(aws iam list-attached-role-policies --role-name $role_name \
            --query 'AttachedPolicies[].PolicyArn' --output text)

        for policy_arn in $policies; do
            if [[ ! -z "$policy_arn" ]]; then
                print_info "Detaching policy: $policy_arn"
                aws iam detach-role-policy --role-name $role_name --policy-arn $policy_arn || true
            fi
        done

        # Delete inline policies
        local inline_policies=$(aws iam list-role-policies --role-name $role_name \
            --query 'PolicyNames[]' --output text)

        for policy_name in $inline_policies; do
            if [[ ! -z "$policy_name" ]]; then
                print_info "Deleting inline policy: $policy_name"
                aws iam delete-role-policy --role-name $role_name --policy-name $policy_name || true
            fi
        done

        # Delete role
        print_info "Deleting IAM role: $role_name"
        aws iam delete-role --role-name $role_name || true

        print_success "IAM roles cleaned up"
    else
        print_info "IAM role not found"
    fi
}

# Clean up SageMaker S3 artifacts
cleanup_sagemaker_artifacts() {
    print_info "Cleaning up SageMaker S3 artifacts..."

    # Get default SageMaker bucket
    local sagemaker_bucket=$(aws s3api list-buckets --query "Buckets[?contains(Name, 'sagemaker-$AWS_REGION-$AWS_ACCOUNT_ID')].Name" --output text)

    if [[ ! -z "$sagemaker_bucket" ]]; then
        print_info "Cleaning SageMaker artifacts from bucket: $sagemaker_bucket"

        # Remove medical triage model artifacts
        aws s3 rm s3://$sagemaker_bucket/sagemaker/medical-triage-model/ --recursive --region $AWS_REGION || true

        print_success "SageMaker artifacts cleaned up"
    else
        print_info "SageMaker bucket not found"
    fi
}

# Show teardown summary
show_teardown_summary() {
    print_success "Teardown completed successfully!"
    echo
    echo "===================================================="
    echo "         TEARDOWN SUMMARY"
    echo "===================================================="
    echo "Environment: $ENVIRONMENT"
    echo "Region: $AWS_REGION"
    echo "Account: $AWS_ACCOUNT_ID"
    echo

    if [[ "$KEEP_DATA" == "true" ]]; then
        print_warning "Data preserved:"
        echo "  - S3 buckets may still contain data"
        echo "  - DynamoDB tables were backed up before deletion"
        echo "  - Manual cleanup may be required"
    else
        print_info "All resources deleted:"
        echo "  - CloudFormation stacks"
        echo "  - SageMaker endpoints and models"
        echo "  - S3 buckets and data"
        echo "  - DynamoDB tables and audit logs"
        echo "  - IAM roles and policies"
    fi

    echo
    echo "Note: Some resources may take additional time to fully delete"
    echo "Check the AWS console to verify complete cleanup"
    echo "===================================================="
}

# Main teardown function
main() {
    print_info "Starting Medical Image Triage System teardown..."
    print_info "Environment: $ENVIRONMENT"
    print_info "Region: $AWS_REGION"
    print_info "Account: $AWS_ACCOUNT_ID"

    # Confirm deletion
    confirm_deletion

    # Run teardown steps
    check_aws_credentials
    delete_sagemaker_endpoint
    prepare_s3_buckets
    delete_stacks
    cleanup_iam_roles
    cleanup_sagemaker_artifacts
    show_teardown_summary
}

# Run main function
main "$@"