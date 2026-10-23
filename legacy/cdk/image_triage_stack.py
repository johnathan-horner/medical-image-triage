"""
Core infrastructure stack for Medical Image Triage System on AWS.
Deploys S3, DynamoDB, Lambda, API Gateway, SageMaker, and supporting services.
"""

from aws_cdk import (
    Stack,
    Duration,
    RemovalPolicy,
    aws_s3 as s3,
    aws_dynamodb as dynamodb,
    aws_lambda as lambda_,
    aws_apigateway as apigateway,
    aws_sagemaker as sagemaker,
    aws_sns as sns,
    aws_sns_subscriptions as subscriptions,
    aws_iam as iam,
    aws_cognito as cognito,
    aws_kms as kms,
    aws_logs as logs,
    aws_s3_notifications as s3n,
    CfnOutput
)
from constructs import Construct
import json


class ImageTriageStack(Stack):
    """Core infrastructure stack for Medical Image Triage System."""

    def __init__(self, scope: Construct, construct_id: str, environment_name: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.environment_name = environment_name
        self.account_id = Stack.of(self).account

        # Get configuration from context
        self.config = self.node.try_get_context("config") or {}

        # Create KMS key for encryption
        self.kms_key = self._create_kms_key()

        # Create S3 buckets
        self.ingest_bucket, self.archive_bucket = self._create_s3_buckets()

        # Create DynamoDB table
        self.dynamodb_table = self._create_dynamodb_table()

        # Create Cognito user pool
        self.user_pool, self.user_pool_client = self._create_cognito_user_pool()

        # Create SNS topics for physician queues
        self.sns_topics = self._create_sns_topics()

        # Create Lambda layers
        self.lambda_layer = self._create_lambda_layers()

        # Create Lambda functions
        self.lambda_functions = self._create_lambda_functions()

        # Create SageMaker endpoint (placeholder - will be deployed separately)
        self.sagemaker_endpoint = self._create_sagemaker_endpoint()

        # Create API Gateway
        self.api_gateway = self._create_api_gateway()

        # Set up S3 event notifications
        self._setup_s3_notifications()

        # Create outputs
        self._create_outputs()

    def _create_kms_key(self) -> kms.Key:
        """Create KMS key for encryption at rest."""
        return kms.Key(
            self, "TriageKMSKey",
            description="KMS key for Medical Image Triage System encryption",
            enable_key_rotation=True,
            removal_policy=RemovalPolicy.DESTROY,  # For demo - use RETAIN in production
            alias=f"medical-triage-{self.environment_name}"
        )

    def _create_s3_buckets(self) -> tuple[s3.Bucket, s3.Bucket]:
        """Create S3 buckets for image ingestion and archive."""

        # Ingest bucket - temporary storage for uploaded images
        ingest_bucket = s3.Bucket(
            self, "ImageIngestBucket",
            bucket_name=f"image-triage-ingest-{self.account_id}-{self.environment_name}",
            encryption=s3.BucketEncryption.KMS,
            encryption_key=self.kms_key,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            versioned=False,
            removal_policy=RemovalPolicy.DESTROY,  # For demo
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="DeleteImagesAfterProcessing",
                    enabled=True,
                    expiration=Duration.days(1),  # HIPAA compliance - no raw image retention
                    abort_incomplete_multipart_upload_after=Duration.days(1)
                )
            ],
            cors=[
                s3.CorsRule(
                    allowed_methods=[s3.HttpMethods.POST, s3.HttpMethods.PUT],
                    allowed_origins=["*"],  # Restrict in production
                    allowed_headers=["*"],
                    max_age=3000
                )
            ]
        )

        # Archive bucket - long-term storage for metadata only
        archive_bucket = s3.Bucket(
            self, "ImageArchiveBucket",
            bucket_name=f"image-triage-archive-{self.account_id}-{self.environment_name}",
            encryption=s3.BucketEncryption.KMS,
            encryption_key=self.kms_key,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            versioned=True,
            removal_policy=RemovalPolicy.DESTROY,  # For demo
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="ArchiveMetadata",
                    enabled=True,
                    transitions=[
                        s3.Transition(
                            storage_class=s3.StorageClass.INTELLIGENT_TIERING,
                            transition_after=Duration.days(30)
                        ),
                        s3.Transition(
                            storage_class=s3.StorageClass.GLACIER,
                            transition_after=Duration.days(365)
                        )
                    ],
                    expiration=Duration.days(2555)  # 7 years for HIPAA compliance
                )
            ]
        )

        return ingest_bucket, archive_bucket

    def _create_dynamodb_table(self) -> dynamodb.Table:
        """Create DynamoDB table for audit trails."""

        table = dynamodb.Table(
            self, "TriagePredictionsTable",
            table_name=f"triage-predictions-{self.environment_name}",
            partition_key=dynamodb.Attribute(
                name="image_hash",
                type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="timestamp",
                type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            encryption=dynamodb.TableEncryption.CUSTOMER_MANAGED,
            encryption_key=self.kms_key,
            point_in_time_recovery=True,
            removal_policy=RemovalPolicy.DESTROY,  # For demo
            stream=dynamodb.StreamViewType.NEW_AND_OLD_IMAGES
        )

        # GSI for routing decision queries
        table.add_global_secondary_index(
            index_name="RoutingDecisionIndex",
            partition_key=dynamodb.Attribute(
                name="routing_decision",
                type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="timestamp",
                type=dynamodb.AttributeType.STRING
            )
        )

        # GSI for time-range dashboard queries
        table.add_global_secondary_index(
            index_name="TimestampIndex",
            partition_key=dynamodb.Attribute(
                name="date_partition",  # YYYY-MM-DD format
                type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="timestamp",
                type=dynamodb.AttributeType.STRING
            )
        )

        # GSI for patient queries (for audit trail)
        table.add_global_secondary_index(
            index_name="PatientIndex",
            partition_key=dynamodb.Attribute(
                name="patient_id_hash",
                type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="timestamp",
                type=dynamodb.AttributeType.STRING
            )
        )

        return table

    def _create_cognito_user_pool(self) -> tuple[cognito.UserPool, cognito.UserPoolClient]:
        """Create Cognito user pool for authentication."""

        user_pool = cognito.UserPool(
            self, "TriageUserPool",
            user_pool_name=f"medical-triage-{self.environment_name}",
            self_sign_up_enabled=False,  # Admin-only user creation
            sign_in_aliases=cognito.SignInAliases(email=True, username=True),
            password_policy=cognito.PasswordPolicy(
                min_length=12,
                require_uppercase=True,
                require_lowercase=True,
                require_digits=True,
                require_symbols=True
            ),
            mfa=cognito.Mfa.REQUIRED,
            mfa_second_factor=cognito.MfaSecondFactor(sms=False, otp=True),
            account_recovery=cognito.AccountRecovery.EMAIL_ONLY,
            removal_policy=RemovalPolicy.DESTROY  # For demo
        )

        # Create user groups
        physician_group = cognito.CfnUserPoolGroup(
            self, "PhysicianGroup",
            group_name="physicians",
            user_pool_id=user_pool.user_pool_id,
            description="Physicians who can review medical images"
        )

        admin_group = cognito.CfnUserPoolGroup(
            self, "AdminGroup",
            group_name="administrators",
            user_pool_id=user_pool.user_pool_id,
            description="System administrators with full access"
        )

        # Create user pool client
        user_pool_client = cognito.UserPoolClient(
            self, "TriageUserPoolClient",
            user_pool=user_pool,
            auth_flows=cognito.AuthFlow(
                user_password=True,
                user_srp=True
            ),
            generate_secret=False,  # For web applications
            token_validity=cognito.TokenValidity(
                access_token=Duration.hours(1),
                id_token=Duration.hours(1),
                refresh_token=Duration.days(30)
            )
        )

        return user_pool, user_pool_client

    def _create_sns_topics(self) -> dict:
        """Create SNS topics for physician queue routing."""

        topics = {}

        # Auto-triage complete topic
        topics['auto_triage'] = sns.Topic(
            self, "AutoTriageCompleteTopic",
            topic_name=f"auto-triage-complete-{self.environment_name}",
            display_name="Auto Triage Complete Notifications",
            kms_master_key=self.kms_key
        )

        # Expedited physician review topic
        topics['expedited_review'] = sns.Topic(
            self, "ExpeditedReviewTopic",
            topic_name=f"expedited-physician-review-{self.environment_name}",
            display_name="Expedited Physician Review Queue",
            kms_master_key=self.kms_key
        )

        # Senior physician review topic
        topics['senior_review'] = sns.Topic(
            self, "SeniorReviewTopic",
            topic_name=f"senior-physician-review-{self.environment_name}",
            display_name="Senior Physician Review Queue",
            kms_master_key=self.kms_key
        )

        return topics

    def _create_lambda_layers(self) -> lambda_.LayerVersion:
        """Create Lambda layers with shared dependencies."""

        return lambda_.LayerVersion(
            self, "TriageDependenciesLayer",
            code=lambda_.Code.from_asset("../lambdas/layer"),
            compatible_runtimes=[lambda_.Runtime.PYTHON_3_11],
            description="Shared dependencies for Medical Triage Lambda functions"
        )

    def _create_lambda_functions(self) -> dict:
        """Create Lambda functions for the triage system."""

        functions = {}

        # Common Lambda configuration
        common_config = {
            "runtime": lambda_.Runtime.PYTHON_3_11,
            "timeout": Duration.minutes(5),
            "memory_size": 1024,
            "layers": [self.lambda_layer],
            "environment": {
                "DYNAMODB_TABLE": self.dynamodb_table.table_name,
                "INGEST_BUCKET": self.ingest_bucket.bucket_name,
                "ARCHIVE_BUCKET": self.archive_bucket.bucket_name,
                "KMS_KEY_ID": self.kms_key.key_id,
                "ENVIRONMENT": self.environment_name,
                "AUTO_TRIAGE_TOPIC": self.sns_topics['auto_triage'].topic_arn,
                "EXPEDITED_TOPIC": self.sns_topics['expedited_review'].topic_arn,
                "SENIOR_TOPIC": self.sns_topics['senior_review'].topic_arn
            },
            "log_retention": logs.RetentionDays.ONE_MONTH
        }

        # Inference Lambda
        functions['inference'] = lambda_.Function(
            self, "TriageInferenceFunction",
            function_name=f"triage-inference-{self.environment_name}",
            code=lambda_.Code.from_asset("../lambdas/inference"),
            handler="handler.lambda_handler",
            description="Handles image upload, SageMaker inference, and triage routing",
            **common_config
        )

        # Audit Lambda
        functions['audit'] = lambda_.Function(
            self, "TriageAuditFunction",
            function_name=f"triage-audit-{self.environment_name}",
            code=lambda_.Code.from_asset("../lambdas/audit"),
            handler="handler.lambda_handler",
            description="Handles audit trail queries and compliance reporting",
            **common_config
        )

        # Dashboard Lambda
        functions['dashboard'] = lambda_.Function(
            self, "TriageDashboardFunction",
            function_name=f"triage-dashboard-{self.environment_name}",
            code=lambda_.Code.from_asset("../lambdas/dashboard"),
            handler="handler.lambda_handler",
            description="Provides dashboard metrics and model drift detection",
            **common_config
        )

        # Grant permissions to Lambda functions
        self._grant_lambda_permissions(functions)

        return functions

    def _grant_lambda_permissions(self, functions: dict) -> None:
        """Grant necessary permissions to Lambda functions."""

        for function in functions.values():
            # DynamoDB permissions
            self.dynamodb_table.grant_read_write_data(function)

            # S3 permissions
            self.ingest_bucket.grant_read_write(function)
            self.archive_bucket.grant_read_write(function)

            # KMS permissions
            self.kms_key.grant_encrypt_decrypt(function)

            # SNS permissions
            for topic in self.sns_topics.values():
                topic.grant_publish(function)

        # Additional SageMaker permissions for inference function
        functions['inference'].add_to_role_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "sagemaker:InvokeEndpoint",
                    "sagemaker:DescribeEndpoint"
                ],
                resources=[
                    f"arn:aws:sagemaker:{self.region}:{self.account}:endpoint/*"
                ]
            )
        )

    def _create_sagemaker_endpoint(self) -> str:
        """Create placeholder for SageMaker endpoint (deployed separately)."""

        # This will be created by the deployment script
        endpoint_name = f"medical-triage-endpoint-{self.environment_name}"

        return endpoint_name

    def _create_api_gateway(self) -> apigateway.RestApi:
        """Create API Gateway with Cognito authorization."""

        # Cognito authorizer
        auth = apigateway.CognitoUserPoolsAuthorizer(
            self, "TriageAuthorizer",
            cognito_user_pools=[self.user_pool],
            authorizer_name=f"triage-authorizer-{self.environment_name}"
        )

        # API Gateway
        api = apigateway.RestApi(
            self, "TriageAPI",
            rest_api_name=f"medical-triage-api-{self.environment_name}",
            description="Medical Image Triage System API",
            endpoint_configuration=apigateway.EndpointConfiguration(
                types=[apigateway.EndpointType.REGIONAL]
            ),
            default_cors_preflight_options=apigateway.CorsOptions(
                allow_origins=apigateway.Cors.ALL_ORIGINS,  # Restrict in production
                allow_methods=apigateway.Cors.ALL_METHODS,
                allow_headers=["*"]
            ),
            cloud_watch_role=True,
            deploy_options=apigateway.StageOptions(
                stage_name=self.environment_name,
                throttling_rate_limit=100,
                throttling_burst_limit=200,
                logging_level=apigateway.MethodLoggingLevel.INFO,
                data_trace_enabled=False,  # Disable for HIPAA compliance
                metrics_enabled=True
            )
        )

        # Request/Response models
        error_model = api.add_model(
            "ErrorModel",
            content_type="application/json",
            schema=apigateway.JsonSchema(
                type=apigateway.JsonSchemaType.OBJECT,
                properties={
                    "error": apigateway.JsonSchema(type=apigateway.JsonSchemaType.STRING),
                    "message": apigateway.JsonSchema(type=apigateway.JsonSchemaType.STRING)
                }
            )
        )

        # Triage endpoint - POST /triage
        triage_resource = api.root.add_resource("triage")
        triage_resource.add_method(
            "POST",
            apigateway.LambdaIntegration(
                self.lambda_functions['inference'],
                proxy=False,
                integration_responses=[
                    apigateway.IntegrationResponse(
                        status_code="200",
                        response_templates={
                            "application/json": ""
                        }
                    ),
                    apigateway.IntegrationResponse(
                        status_code="400",
                        selection_pattern="4\\d{2}",
                        response_templates={
                            "application/json": '{"error": "Bad Request", "message": $input.json(\'$.errorMessage\')}'
                        }
                    ),
                    apigateway.IntegrationResponse(
                        status_code="500",
                        selection_pattern="5\\d{2}",
                        response_templates={
                            "application/json": '{"error": "Internal Server Error", "message": $input.json(\'$.errorMessage\')}'
                        }
                    )
                ],
                request_templates={
                    "application/json": json.dumps({
                        "body": "$input.json('$')",
                        "headers": {
                            "Authorization": "$input.params('Authorization')",
                            "Content-Type": "$input.params('Content-Type')"
                        },
                        "requestContext": {
                            "requestId": "$context.requestId",
                            "authorizer": {
                                "claims": {
                                    "sub": "$context.authorizer.claims.sub",
                                    "email": "$context.authorizer.claims.email",
                                    "cognito:groups": "$context.authorizer.claims['cognito:groups']"
                                }
                            }
                        }
                    })
                }
            ),
            authorizer=auth,
            method_responses=[
                apigateway.MethodResponse(status_code="200"),
                apigateway.MethodResponse(
                    status_code="400",
                    response_models={"application/json": error_model}
                ),
                apigateway.MethodResponse(
                    status_code="500",
                    response_models={"application/json": error_model}
                )
            ],
            request_validator=apigateway.RequestValidator(
                self, "TriageRequestValidator",
                rest_api=api,
                validate_request_body=True,
                validate_request_parameters=True
            )
        )

        # Audit endpoint - GET /audit/{image_hash}
        audit_resource = api.root.add_resource("audit")
        audit_hash_resource = audit_resource.add_resource("{image_hash}")
        audit_hash_resource.add_method(
            "GET",
            apigateway.LambdaIntegration(self.lambda_functions['audit']),
            authorizer=auth,
            request_parameters={
                "method.request.path.image_hash": True
            }
        )

        # Dashboard endpoints - GET /dashboard/metrics
        dashboard_resource = api.root.add_resource("dashboard")
        metrics_resource = dashboard_resource.add_resource("metrics")
        metrics_resource.add_method(
            "GET",
            apigateway.LambdaIntegration(self.lambda_functions['dashboard']),
            authorizer=auth
        )

        return api

    def _setup_s3_notifications(self) -> None:
        """Set up S3 event notifications for image processing."""

        # Trigger inference Lambda when images are uploaded
        self.ingest_bucket.add_event_notification(
            s3.EventType.OBJECT_CREATED,
            s3n.LambdaDestination(self.lambda_functions['inference']),
            s3.NotificationKeyFilter(suffix=".jpg")
        )

        self.ingest_bucket.add_event_notification(
            s3.EventType.OBJECT_CREATED,
            s3n.LambdaDestination(self.lambda_functions['inference']),
            s3.NotificationKeyFilter(suffix=".png")
        )

    def _create_outputs(self) -> None:
        """Create CloudFormation outputs."""

        CfnOutput(
            self, "APIGatewayURL",
            value=self.api_gateway.url,
            description="API Gateway URL for Medical Triage System"
        )

        CfnOutput(
            self, "UserPoolId",
            value=self.user_pool.user_pool_id,
            description="Cognito User Pool ID"
        )

        CfnOutput(
            self, "UserPoolClientId",
            value=self.user_pool_client.user_pool_client_id,
            description="Cognito User Pool Client ID"
        )

        CfnOutput(
            self, "IngestBucketName",
            value=self.ingest_bucket.bucket_name,
            description="S3 bucket for image ingestion"
        )

        CfnOutput(
            self, "DynamoDBTableName",
            value=self.dynamodb_table.table_name,
            description="DynamoDB table for audit trails"
        )

        CfnOutput(
            self, "SageMakerEndpointName",
            value=self.sagemaker_endpoint,
            description="SageMaker endpoint name (deployed separately)"
        )