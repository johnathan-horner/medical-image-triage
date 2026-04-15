#!/bin/bash

# Medical Image Triage System - AWS Deployment Script
# This script handles the complete deployment of the system to AWS

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
SKIP_MODEL_DEPLOY=false
SKIP_INFRASTRUCTURE=false

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

Deploy Medical Image Triage System to AWS

OPTIONS:
    -e, --environment ENV     Environment name (default: dev)
    -r, --region REGION      AWS region (default: us-east-1)
    -a, --account ACCOUNT    AWS account ID (required)
    --skip-model            Skip SageMaker model deployment
    --skip-infra            Skip infrastructure deployment
    -h, --help              Show this help message

EXAMPLES:
    # Full deployment to dev environment
    $0 -e dev -r us-east-1 -a 123456789012

    # Deploy only infrastructure (skip model)
    $0 -e prod -r us-west-2 -a 123456789012 --skip-model

    # Deploy only model (infrastructure exists)
    $0 -e dev -a 123456789012 --skip-infra

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
        --skip-model)
            SKIP_MODEL_DEPLOY=true
            shift
            ;;
        --skip-infra)
            SKIP_INFRASTRUCTURE=true
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

# Validate AWS credentials
check_aws_credentials() {
    print_info "Checking AWS credentials..."

    if ! aws sts get-caller-identity > /dev/null 2>&1; then
        print_error "AWS credentials not configured or invalid"
        print_error "Please run 'aws configure' or set AWS environment variables"
        exit 1
    fi

    local current_account=$(aws sts get-caller-identity --query Account --output text)
    if [[ "$current_account" != "$AWS_ACCOUNT_ID" ]]; then
        print_error "Current AWS account ($current_account) doesn't match specified account ($AWS_ACCOUNT_ID)"
        exit 1
    fi

    print_success "AWS credentials validated for account: $AWS_ACCOUNT_ID"
}

# Check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."

    # Check required tools
    local tools=("aws" "cdk" "python3" "pip" "docker")
    for tool in "${tools[@]}"; do
        if ! command -v $tool &> /dev/null; then
            print_error "$tool is required but not installed"
            exit 1
        fi
    done

    # Check CDK version
    local cdk_version=$(cdk --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
    print_info "CDK version: $cdk_version"

    # Check Docker is running
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker Desktop."
        exit 1
    fi

    print_success "All prerequisites satisfied"
}

# Setup environment
setup_environment() {
    print_info "Setting up environment..."

    # Set environment variables
    export CDK_DEFAULT_ACCOUNT=$AWS_ACCOUNT_ID
    export CDK_DEFAULT_REGION=$AWS_REGION

    # Update CDK context
    cd cdk
    cat > cdk.context.json << EOF
{
  "account": "$AWS_ACCOUNT_ID",
  "region": "$AWS_REGION",
  "environment": "$ENVIRONMENT",
  "project_name": "medical-image-triage",
  "sagemaker": {
    "instance_type": "ml.t2.medium",
    "instance_count": 1,
    "max_concurrent_transforms": 10
  },
  "s3": {
    "image_retention_days": 1,
    "archive_retention_years": 7
  },
  "dynamodb": {
    "billing_mode": "PAY_PER_REQUEST",
    "point_in_time_recovery": true
  },
  "lambda": {
    "timeout_seconds": 300,
    "memory_mb": 1024,
    "python_version": "3.11"
  },
  "monitoring": {
    "confidence_threshold": 0.7,
    "error_rate_threshold": 0.05,
    "latency_threshold_ms": 5000
  },
  "cognito": {
    "password_policy": {
      "min_length": 12,
      "require_uppercase": true,
      "require_lowercase": true,
      "require_numbers": true,
      "require_symbols": true
    }
  }
}
EOF

    # Install CDK dependencies
    print_info "Installing CDK dependencies..."
    pip install -r requirements.txt

    cd ..
    print_success "Environment setup complete"
}

# Bootstrap CDK
bootstrap_cdk() {
    print_info "Bootstrapping CDK environment..."

    cd cdk

    # Check if already bootstrapped
    if aws cloudformation describe-stacks --stack-name CDKToolkit-MedicalImageTriage --region $AWS_REGION > /dev/null 2>&1; then
        print_info "CDK already bootstrapped"
    else
        print_info "Bootstrapping CDK for the first time..."
        cdk bootstrap aws://$AWS_ACCOUNT_ID/$AWS_REGION \
            --toolkit-stack-name CDKToolkit-MedicalImageTriage \
            --qualifier medicaltriage
    fi

    cd ..
    print_success "CDK bootstrap complete"
}

# Prepare Lambda layers
prepare_lambda_layers() {
    print_info "Preparing Lambda layers..."

    cd lambdas/layer

    # Create layer directory structure
    mkdir -p python/lib/python3.11/site-packages

    # Install dependencies
    pip install -r requirements.txt -t python/lib/python3.11/site-packages

    cd ../..
    print_success "Lambda layers prepared"
}

# Deploy infrastructure
deploy_infrastructure() {
    if [[ "$SKIP_INFRASTRUCTURE" == "true" ]]; then
        print_warning "Skipping infrastructure deployment"
        return 0
    fi

    print_info "Deploying infrastructure stacks..."

    cd cdk

    # Synthesize CDK app first to check for errors
    print_info "Synthesizing CDK application..."
    cdk synth

    # Deploy core infrastructure stack
    print_info "Deploying core infrastructure..."
    cdk deploy ImageTriageStack-$ENVIRONMENT \
        --require-approval never \
        --outputs-file ../outputs/core-stack-outputs.json

    # Deploy monitoring stack
    print_info "Deploying monitoring stack..."
    cdk deploy ImageTriageMonitoring-$ENVIRONMENT \
        --require-approval never \
        --outputs-file ../outputs/monitoring-stack-outputs.json

    cd ..
    print_success "Infrastructure deployment complete"
}

# Deploy SageMaker model
deploy_sagemaker_model() {
    if [[ "$SKIP_MODEL_DEPLOY" == "true" ]]; then
        print_warning "Skipping SageMaker model deployment"
        return 0
    fi

    print_info "Deploying SageMaker model..."

    # Check if trained model exists
    if [[ ! -d "models/medical_model/saved_model" ]]; then
        print_warning "Trained model not found. Training model first..."
        python model/train.py
    fi

    # Deploy model to SageMaker
    python scripts/deploy_sagemaker_model.py \
        --model-path models/medical_model/saved_model \
        --endpoint-name medical-triage-endpoint-$ENVIRONMENT \
        --instance-type ml.t2.medium \
        --region $AWS_REGION

    print_success "SageMaker model deployment complete"
}

# Create Cognito users
create_cognito_users() {
    print_info "Creating Cognito test users..."

    # Get user pool ID from stack outputs
    local user_pool_id=$(cat outputs/core-stack-outputs.json | python3 -c "
import json, sys
outputs = json.load(sys.stdin)
print(outputs.get('ImageTriageStack-$ENVIRONMENT', {}).get('UserPoolId', ''))
")

    if [[ -z "$user_pool_id" ]]; then
        print_error "Could not retrieve User Pool ID from stack outputs"
        return 1
    fi

    # Create test physician user
    print_info "Creating test physician user..."
    aws cognito-idp admin-create-user \
        --user-pool-id $user_pool_id \
        --username test-physician \
        --user-attributes Name=email,Value=physician@example.com Name=email_verified,Value=true \
        --temporary-password TempPass123! \
        --message-action SUPPRESS \
        --region $AWS_REGION || true  # Continue if user already exists

    # Add user to physicians group
    aws cognito-idp admin-add-user-to-group \
        --user-pool-id $user_pool_id \
        --username test-physician \
        --group-name physicians \
        --region $AWS_REGION || true

    # Create test admin user
    print_info "Creating test admin user..."
    aws cognito-idp admin-create-user \
        --user-pool-id $user_pool_id \
        --username test-admin \
        --user-attributes Name=email,Value=admin@example.com Name=email_verified,Value=true \
        --temporary-password TempPass123! \
        --message-action SUPPRESS \
        --region $AWS_REGION || true

    # Add user to administrators group
    aws cognito-idp admin-add-user-to-group \
        --user-pool-id $user_pool_id \
        --username test-admin \
        --group-name administrators \
        --region $AWS_REGION || true

    print_success "Test users created (physician/admin)"
    print_info "Default password: TempPass123! (users will be prompted to change)"
}

# Run integration tests
run_integration_tests() {
    print_info "Running integration tests..."

    # Get API Gateway URL from stack outputs
    local api_url=$(cat outputs/core-stack-outputs.json | python3 -c "
import json, sys
outputs = json.load(sys.stdin)
print(outputs.get('ImageTriageStack-$ENVIRONMENT', {}).get('APIGatewayURL', ''))
")

    if [[ -z "$api_url" ]]; then
        print_error "Could not retrieve API Gateway URL from stack outputs"
        return 1
    fi

    export API_BASE_URL=$api_url
    export ENVIRONMENT=$ENVIRONMENT

    # Run integration tests
    python scripts/integration_tests.py

    print_success "Integration tests completed"
}

# Display deployment summary
show_deployment_summary() {
    print_success "Deployment completed successfully!"
    echo
    echo "===================================================="
    echo "         DEPLOYMENT SUMMARY"
    echo "===================================================="
    echo "Environment: $ENVIRONMENT"
    echo "Region: $AWS_REGION"
    echo "Account: $AWS_ACCOUNT_ID"
    echo

    if [[ -f "outputs/core-stack-outputs.json" ]]; then
        local api_url=$(cat outputs/core-stack-outputs.json | python3 -c "
import json, sys
outputs = json.load(sys.stdin)
print(outputs.get('ImageTriageStack-$ENVIRONMENT', {}).get('APIGatewayURL', 'N/A'))
")

        local user_pool_id=$(cat outputs/core-stack-outputs.json | python3 -c "
import json, sys
outputs = json.load(sys.stdin)
print(outputs.get('ImageTriageStack-$ENVIRONMENT', {}).get('UserPoolId', 'N/A'))
")

        echo "API Gateway URL: $api_url"
        echo "User Pool ID: $user_pool_id"
        echo
    fi

    if [[ -f "outputs/monitoring-stack-outputs.json" ]]; then
        local dashboard_url=$(cat outputs/monitoring-stack-outputs.json | python3 -c "
import json, sys
outputs = json.load(sys.stdin)
print(outputs.get('ImageTriageMonitoring-$ENVIRONMENT', {}).get('DashboardURL', 'N/A'))
")

        echo "CloudWatch Dashboard: $dashboard_url"
        echo
    fi

    echo "Test Users:"
    echo "  Physician: test-physician (password: TempPass123!)"
    echo "  Admin: test-admin (password: TempPass123!)"
    echo
    echo "Next Steps:"
    echo "  1. Access the API documentation at: ${api_url}docs"
    echo "  2. Monitor the system via CloudWatch Dashboard"
    echo "  3. Test the system using the integration test scripts"
    echo
    echo "To tear down the deployment, run:"
    echo "  ./scripts/teardown.sh -e $ENVIRONMENT -a $AWS_ACCOUNT_ID"
    echo "===================================================="
}

# Main deployment function
main() {
    print_info "Starting Medical Image Triage System deployment..."
    print_info "Environment: $ENVIRONMENT"
    print_info "Region: $AWS_REGION"
    print_info "Account: $AWS_ACCOUNT_ID"

    # Create outputs directory
    mkdir -p outputs

    # Run deployment steps
    check_aws_credentials
    check_prerequisites
    setup_environment
    bootstrap_cdk
    prepare_lambda_layers
    deploy_infrastructure
    deploy_sagemaker_model
    create_cognito_users
    run_integration_tests
    show_deployment_summary
}

# Run main function
main "$@"