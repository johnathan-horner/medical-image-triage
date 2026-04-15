#!/usr/bin/env python3
"""
Deploy TensorFlow SavedModel to SageMaker endpoint.
Handles model packaging, upload to S3, and endpoint creation.
"""

import argparse
import boto3
import tarfile
import os
import time
import logging
from datetime import datetime
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SageMakerModelDeployer:
    """Handles deployment of TensorFlow models to SageMaker."""

    def __init__(self, region: str):
        self.region = region
        self.sagemaker = boto3.client('sagemaker', region_name=region)
        self.s3 = boto3.client('s3', region_name=region)
        self.session = boto3.Session(region_name=region)
        self.account_id = boto3.client('sts').get_caller_identity()['Account']

        # Use SageMaker execution role
        self.role_arn = f"arn:aws:iam::{self.account_id}:role/SageMakerExecutionRole-MedicalTriage"

    def package_model(self, model_path: str, output_path: str) -> str:
        """Package TensorFlow SavedModel for SageMaker deployment."""
        logger.info(f"Packaging model from {model_path}")

        # Create model.tar.gz
        model_archive = os.path.join(output_path, "model.tar.gz")

        with tarfile.open(model_archive, "w:gz") as tar:
            # Add the SavedModel files
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Archive path should be relative to model directory
                    archive_path = os.path.relpath(file_path, model_path)
                    tar.add(file_path, arcname=archive_path)

        logger.info(f"Model packaged to {model_archive}")
        return model_archive

    def upload_model_to_s3(self, model_archive: str, bucket_name: str) -> str:
        """Upload model archive to S3."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        s3_key = f"sagemaker/medical-triage-model/{timestamp}/model.tar.gz"

        logger.info(f"Uploading model to s3://{bucket_name}/{s3_key}")

        self.s3.upload_file(model_archive, bucket_name, s3_key)

        s3_model_uri = f"s3://{bucket_name}/{s3_key}"
        logger.info(f"Model uploaded to {s3_model_uri}")
        return s3_model_uri

    def create_sagemaker_model(self, model_name: str, model_data_url: str, execution_role: str) -> str:
        """Create SageMaker model."""
        logger.info(f"Creating SageMaker model: {model_name}")

        # TensorFlow serving image URI for the region
        image_uri = self.get_tensorflow_serving_image_uri()

        try:
            response = self.sagemaker.create_model(
                ModelName=model_name,
                PrimaryContainer={
                    'Image': image_uri,
                    'ModelDataUrl': model_data_url,
                    'Environment': {
                        'SAGEMAKER_PROGRAM': 'inference.py',
                        'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/code',
                        'SAGEMAKER_CONTAINER_LOG_LEVEL': '20',
                        'SAGEMAKER_REGION': self.region,
                        'TF_CPP_MIN_LOG_LEVEL': '2'
                    }
                },
                ExecutionRoleArn=execution_role
            )

            logger.info(f"SageMaker model created: {response['ModelArn']}")
            return response['ModelArn']

        except Exception as e:
            logger.error(f"Failed to create SageMaker model: {str(e)}")
            raise

    def create_endpoint_config(self, config_name: str, model_name: str, instance_type: str) -> str:
        """Create SageMaker endpoint configuration."""
        logger.info(f"Creating endpoint configuration: {config_name}")

        try:
            response = self.sagemaker.create_endpoint_config(
                EndpointConfigName=config_name,
                ProductionVariants=[
                    {
                        'VariantName': 'primary',
                        'ModelName': model_name,
                        'InitialInstanceCount': 1,
                        'InstanceType': instance_type,
                        'InitialVariantWeight': 1.0
                    }
                ]
            )

            logger.info(f"Endpoint configuration created: {response['EndpointConfigArn']}")
            return response['EndpointConfigArn']

        except Exception as e:
            logger.error(f"Failed to create endpoint configuration: {str(e)}")
            raise

    def create_or_update_endpoint(self, endpoint_name: str, config_name: str) -> str:
        """Create or update SageMaker endpoint."""
        logger.info(f"Creating/updating endpoint: {endpoint_name}")

        try:
            # Check if endpoint already exists
            try:
                self.sagemaker.describe_endpoint(EndpointName=endpoint_name)
                logger.info(f"Endpoint {endpoint_name} exists, updating...")

                response = self.sagemaker.update_endpoint(
                    EndpointName=endpoint_name,
                    EndpointConfigName=config_name
                )
                action = "updated"

            except self.sagemaker.exceptions.ClientError as e:
                if e.response['Error']['Code'] == 'ValidationException':
                    logger.info(f"Endpoint {endpoint_name} does not exist, creating...")

                    response = self.sagemaker.create_endpoint(
                        EndpointName=endpoint_name,
                        EndpointConfigName=config_name
                    )
                    action = "created"
                else:
                    raise

            logger.info(f"Endpoint {action}: {response['EndpointArn']}")
            return response['EndpointArn']

        except Exception as e:
            logger.error(f"Failed to create/update endpoint: {str(e)}")
            raise

    def wait_for_endpoint(self, endpoint_name: str, timeout_minutes: int = 30) -> bool:
        """Wait for endpoint to be in service."""
        logger.info(f"Waiting for endpoint {endpoint_name} to be in service...")

        start_time = time.time()
        timeout_seconds = timeout_minutes * 60

        while time.time() - start_time < timeout_seconds:
            try:
                response = self.sagemaker.describe_endpoint(EndpointName=endpoint_name)
                status = response['EndpointStatus']

                logger.info(f"Endpoint status: {status}")

                if status == 'InService':
                    logger.info(f"Endpoint {endpoint_name} is now in service!")
                    return True
                elif status == 'Failed':
                    logger.error(f"Endpoint {endpoint_name} failed to deploy")
                    logger.error(f"Failure reason: {response.get('FailureReason', 'Unknown')}")
                    return False

                time.sleep(30)  # Wait 30 seconds before checking again

            except Exception as e:
                logger.error(f"Error checking endpoint status: {str(e)}")
                return False

        logger.error(f"Timeout waiting for endpoint {endpoint_name} to be in service")
        return False

    def get_tensorflow_serving_image_uri(self) -> str:
        """Get TensorFlow Serving image URI for the current region."""
        # TensorFlow Serving images by region
        region_images = {
            'us-east-1': '763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.11.0-cpu',
            'us-west-2': '763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-inference:2.11.0-cpu',
            'eu-west-1': '763104351884.dkr.ecr.eu-west-1.amazonaws.com/tensorflow-inference:2.11.0-cpu',
            'ap-southeast-1': '763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/tensorflow-inference:2.11.0-cpu',
        }

        image_uri = region_images.get(
            self.region,
            '763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.11.0-cpu'
        )

        return image_uri

    def create_execution_role(self) -> str:
        """Create SageMaker execution role if it doesn't exist."""
        iam = boto3.client('iam')
        role_name = "SageMakerExecutionRole-MedicalTriage"

        try:
            # Check if role exists
            response = iam.get_role(RoleName=role_name)
            logger.info(f"Using existing execution role: {response['Role']['Arn']}")
            return response['Role']['Arn']

        except iam.exceptions.NoSuchEntityException:
            logger.info(f"Creating execution role: {role_name}")

            # Create trust policy
            trust_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"Service": "sagemaker.amazonaws.com"},
                        "Action": "sts:AssumeRole"
                    }
                ]
            }

            # Create role
            response = iam.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description="Execution role for Medical Triage SageMaker endpoints"
            )

            role_arn = response['Role']['Arn']

            # Attach required policies
            policies = [
                "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
                "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess"
            ]

            for policy_arn in policies:
                iam.attach_role_policy(
                    RoleName=role_name,
                    PolicyArn=policy_arn
                )

            # Wait for role to be available
            time.sleep(10)

            logger.info(f"Created execution role: {role_arn}")
            return role_arn

    def get_sagemaker_bucket(self) -> str:
        """Get or create SageMaker default bucket."""
        session = boto3.Session(region_name=self.region)
        bucket = session.default_bucket()

        # Ensure bucket exists
        try:
            self.s3.head_bucket(Bucket=bucket)
        except:
            # Create bucket if it doesn't exist
            if self.region == 'us-east-1':
                self.s3.create_bucket(Bucket=bucket)
            else:
                self.s3.create_bucket(
                    Bucket=bucket,
                    CreateBucketConfiguration={'LocationConstraint': self.region}
                )

        logger.info(f"Using SageMaker bucket: {bucket}")
        return bucket

    def deploy_model(
        self,
        model_path: str,
        endpoint_name: str,
        instance_type: str = "ml.t2.medium"
    ) -> dict:
        """Deploy model to SageMaker endpoint."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        # Create temporary directory for packaging
        temp_dir = f"/tmp/sagemaker-deploy-{timestamp}"
        os.makedirs(temp_dir, exist_ok=True)

        try:
            # Package model
            model_archive = self.package_model(model_path, temp_dir)

            # Upload to S3
            bucket = self.get_sagemaker_bucket()
            model_data_url = self.upload_model_to_s3(model_archive, bucket)

            # Create execution role
            execution_role = self.create_execution_role()

            # Create SageMaker resources
            model_name = f"medical-triage-model-{timestamp}"
            config_name = f"medical-triage-config-{timestamp}"

            model_arn = self.create_sagemaker_model(model_name, model_data_url, execution_role)
            config_arn = self.create_endpoint_config(config_name, model_name, instance_type)
            endpoint_arn = self.create_or_update_endpoint(endpoint_name, config_name)

            # Wait for deployment
            success = self.wait_for_endpoint(endpoint_name)

            if success:
                return {
                    'status': 'success',
                    'endpoint_name': endpoint_name,
                    'endpoint_arn': endpoint_arn,
                    'model_name': model_name,
                    'model_data_url': model_data_url
                }
            else:
                return {
                    'status': 'failed',
                    'endpoint_name': endpoint_name,
                    'error': 'Endpoint failed to deploy'
                }

        finally:
            # Clean up temporary files
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(description='Deploy TensorFlow model to SageMaker')
    parser.add_argument('--model-path', required=True, help='Path to SavedModel directory')
    parser.add_argument('--endpoint-name', required=True, help='SageMaker endpoint name')
    parser.add_argument('--instance-type', default='ml.t2.medium', help='SageMaker instance type')
    parser.add_argument('--region', default='us-east-1', help='AWS region')

    args = parser.parse_args()

    # Validate model path
    if not os.path.exists(args.model_path):
        logger.error(f"Model path does not exist: {args.model_path}")
        return 1

    # Deploy model
    deployer = SageMakerModelDeployer(args.region)

    try:
        result = deployer.deploy_model(
            model_path=args.model_path,
            endpoint_name=args.endpoint_name,
            instance_type=args.instance_type
        )

        if result['status'] == 'success':
            logger.info("Deployment completed successfully!")
            logger.info(f"Endpoint: {result['endpoint_name']}")
            logger.info(f"Model: {result['model_name']}")
            return 0
        else:
            logger.error(f"Deployment failed: {result.get('error', 'Unknown error')}")
            return 1

    except Exception as e:
        logger.error(f"Deployment failed with exception: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())