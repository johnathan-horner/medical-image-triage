"""
Monitoring and alerting stack for Medical Image Triage System.
Implements CloudWatch dashboards, alarms, and model drift detection.
"""

from aws_cdk import (
    Stack,
    Duration,
    aws_cloudwatch as cloudwatch,
    aws_cloudwatch_actions as cw_actions,
    aws_sns as sns,
    aws_logs as logs,
    aws_lambda as lambda_,
    aws_events as events,
    aws_events_targets as targets,
    CfnOutput
)
from constructs import Construct


class MonitoringStack(Stack):
    """Monitoring and alerting infrastructure for Medical Triage System."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        environment_name: str,
        core_stack,
        **kwargs
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.environment_name = environment_name
        self.core_stack = core_stack

        # Get configuration from context
        self.config = self.node.try_get_context("monitoring") or {}

        # Create SNS topic for alerts
        self.alert_topic = self._create_alert_topic()

        # Create CloudWatch dashboard
        self.dashboard = self._create_dashboard()

        # Create CloudWatch alarms
        self.alarms = self._create_alarms()

        # Create model drift detection
        self.drift_detector = self._create_drift_detector()

        # Create outputs
        self._create_outputs()

    def _create_alert_topic(self) -> sns.Topic:
        """Create SNS topic for system alerts."""

        return sns.Topic(
            self, "SystemAlertTopic",
            topic_name=f"medical-triage-alerts-{self.environment_name}",
            display_name="Medical Triage System Alerts",
            kms_master_key=self.core_stack.kms_key
        )

    def _create_dashboard(self) -> cloudwatch.Dashboard:
        """Create CloudWatch dashboard for monitoring."""

        dashboard = cloudwatch.Dashboard(
            self, "TriageDashboard",
            dashboard_name=f"MedicalTriageSystem-{self.environment_name}",
            period_override=cloudwatch.PeriodOverride.AUTO
        )

        # System Overview Row
        dashboard.add_widgets(
            cloudwatch.TextWidget(
                markdown="# Medical Image Triage System\n## System Overview",
                width=24,
                height=2
            )
        )

        # API Gateway metrics
        api_widgets = [
            cloudwatch.GraphWidget(
                title="API Gateway Requests",
                left=[
                    cloudwatch.Metric(
                        namespace="AWS/ApiGateway",
                        metric_name="Count",
                        dimensions_map={
                            "ApiName": f"medical-triage-api-{self.environment_name}"
                        },
                        statistic="Sum",
                        period=Duration.minutes(5)
                    )
                ],
                width=12,
                height=6
            ),
            cloudwatch.GraphWidget(
                title="API Gateway Latency",
                left=[
                    cloudwatch.Metric(
                        namespace="AWS/ApiGateway",
                        metric_name="Latency",
                        dimensions_map={
                            "ApiName": f"medical-triage-api-{self.environment_name}"
                        },
                        statistic="Average",
                        period=Duration.minutes(5)
                    )
                ],
                width=12,
                height=6
            )
        ]

        dashboard.add_widgets(*api_widgets)

        # Lambda metrics
        lambda_widgets = []
        for func_name, func in self.core_stack.lambda_functions.items():
            lambda_widgets.append(
                cloudwatch.GraphWidget(
                    title=f"Lambda {func_name.title()} Duration",
                    left=[func.metric_duration(period=Duration.minutes(5))],
                    right=[func.metric_invocations(period=Duration.minutes(5))],
                    width=8,
                    height=6
                )
            )

        dashboard.add_widgets(*lambda_widgets)

        # DynamoDB metrics
        dynamodb_widgets = [
            cloudwatch.GraphWidget(
                title="DynamoDB Read/Write Capacity",
                left=[
                    self.core_stack.dynamodb_table.metric_consumed_read_capacity_units(),
                    self.core_stack.dynamodb_table.metric_consumed_write_capacity_units()
                ],
                width=12,
                height=6
            ),
            cloudwatch.GraphWidget(
                title="DynamoDB Throttles",
                left=[
                    self.core_stack.dynamodb_table.metric_user_errors()
                ],
                width=12,
                height=6
            )
        ]

        dashboard.add_widgets(*dynamodb_widgets)

        # Custom metrics for medical triage
        triage_widgets = [
            cloudwatch.GraphWidget(
                title="Prediction Confidence Distribution",
                left=[
                    cloudwatch.Metric(
                        namespace="MedicalTriage",
                        metric_name="HighConfidencePredictions",
                        period=Duration.minutes(15)
                    ),
                    cloudwatch.Metric(
                        namespace="MedicalTriage",
                        metric_name="MediumConfidencePredictions",
                        period=Duration.minutes(15)
                    ),
                    cloudwatch.Metric(
                        namespace="MedicalTriage",
                        metric_name="LowConfidencePredictions",
                        period=Duration.minutes(15)
                    )
                ],
                width=12,
                height=6
            ),
            cloudwatch.GraphWidget(
                title="Triage Routing Decisions",
                left=[
                    cloudwatch.Metric(
                        namespace="MedicalTriage",
                        metric_name="AutoApproved",
                        period=Duration.minutes(15)
                    ),
                    cloudwatch.Metric(
                        namespace="MedicalTriage",
                        metric_name="ExpeditedReview",
                        period=Duration.minutes(15)
                    ),
                    cloudwatch.Metric(
                        namespace="MedicalTriage",
                        metric_name="SeniorReview",
                        period=Duration.minutes(15)
                    )
                ],
                width=12,
                height=6
            )
        ]

        dashboard.add_widgets(*triage_widgets)

        # Model Performance Row
        dashboard.add_widgets(
            cloudwatch.TextWidget(
                markdown="## Model Performance Monitoring",
                width=24,
                height=1
            )
        )

        model_widgets = [
            cloudwatch.GraphWidget(
                title="Average Prediction Confidence",
                left=[
                    cloudwatch.Metric(
                        namespace="MedicalTriage",
                        metric_name="AverageConfidence",
                        period=Duration.hours(1),
                        statistic="Average"
                    )
                ],
                width=12,
                height=6,
                left_y_axis=cloudwatch.YAxisProps(
                    min=0,
                    max=1
                )
            ),
            cloudwatch.GraphWidget(
                title="SageMaker Endpoint Metrics",
                left=[
                    cloudwatch.Metric(
                        namespace="AWS/SageMaker",
                        metric_name="Invocations",
                        dimensions_map={
                            "EndpointName": self.core_stack.sagemaker_endpoint
                        },
                        period=Duration.minutes(5)
                    )
                ],
                right=[
                    cloudwatch.Metric(
                        namespace="AWS/SageMaker",
                        metric_name="ModelLatency",
                        dimensions_map={
                            "EndpointName": self.core_stack.sagemaker_endpoint
                        },
                        period=Duration.minutes(5)
                    )
                ],
                width=12,
                height=6
            )
        ]

        dashboard.add_widgets(*model_widgets)

        return dashboard

    def _create_alarms(self) -> dict:
        """Create CloudWatch alarms for system monitoring."""

        alarms = {}

        # API Gateway error rate alarm
        alarms['api_errors'] = cloudwatch.Alarm(
            self, "APIErrorRateAlarm",
            alarm_name=f"MedicalTriage-APIErrors-{self.environment_name}",
            metric=cloudwatch.Metric(
                namespace="AWS/ApiGateway",
                metric_name="4XXError",
                dimensions_map={
                    "ApiName": f"medical-triage-api-{self.environment_name}"
                },
                statistic="Sum",
                period=Duration.minutes(5)
            ),
            threshold=10,
            evaluation_periods=2,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
            alarm_description="High API error rate detected"
        )

        # Lambda error rate alarm
        for func_name, func in self.core_stack.lambda_functions.items():
            alarms[f'{func_name}_errors'] = cloudwatch.Alarm(
                self, f"Lambda{func_name.title()}ErrorAlarm",
                alarm_name=f"MedicalTriage-{func_name}-Errors-{self.environment_name}",
                metric=func.metric_errors(period=Duration.minutes(5)),
                threshold=5,
                evaluation_periods=2,
                comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
                alarm_description=f"High error rate in {func_name} Lambda function"
            )

        # DynamoDB throttling alarm
        alarms['dynamodb_throttles'] = cloudwatch.Alarm(
            self, "DynamoDBThrottleAlarm",
            alarm_name=f"MedicalTriage-DynamoDBThrottles-{self.environment_name}",
            metric=self.core_stack.dynamodb_table.metric_user_errors(
                period=Duration.minutes(5)
            ),
            threshold=1,
            evaluation_periods=1,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
            alarm_description="DynamoDB throttling detected"
        )

        # Model confidence drift alarm
        alarms['confidence_drift'] = cloudwatch.Alarm(
            self, "ConfidenceDriftAlarm",
            alarm_name=f"MedicalTriage-ConfidenceDrift-{self.environment_name}",
            metric=cloudwatch.Metric(
                namespace="MedicalTriage",
                metric_name="AverageConfidence",
                period=Duration.hours(1),
                statistic="Average"
            ),
            threshold=self.config.get("confidence_threshold", 0.7),
            evaluation_periods=3,
            comparison_operator=cloudwatch.ComparisonOperator.LESS_THAN_THRESHOLD,
            alarm_description="Model confidence drift detected - average confidence below threshold",
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING
        )

        # SageMaker endpoint alarm
        alarms['sagemaker_errors'] = cloudwatch.Alarm(
            self, "SageMakerErrorAlarm",
            alarm_name=f"MedicalTriage-SageMakerErrors-{self.environment_name}",
            metric=cloudwatch.Metric(
                namespace="AWS/SageMaker",
                metric_name="Invocation4XXErrors",
                dimensions_map={
                    "EndpointName": self.core_stack.sagemaker_endpoint
                },
                period=Duration.minutes(5)
            ),
            threshold=3,
            evaluation_periods=2,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
            alarm_description="SageMaker endpoint errors detected"
        )

        # Add SNS actions to all alarms
        for alarm in alarms.values():
            alarm.add_alarm_action(cw_actions.SnsAction(self.alert_topic))

        return alarms

    def _create_drift_detector(self) -> lambda_.Function:
        """Create Lambda function for model drift detection."""

        drift_function = lambda_.Function(
            self, "DriftDetectorFunction",
            function_name=f"triage-drift-detector-{self.environment_name}",
            runtime=lambda_.Runtime.PYTHON_3_11,
            code=lambda_.Code.from_asset("../lambdas/drift_detector"),
            handler="handler.lambda_handler",
            timeout=Duration.minutes(10),
            memory_size=1024,
            environment={
                "DYNAMODB_TABLE": self.core_stack.dynamodb_table.table_name,
                "ALERT_TOPIC": self.alert_topic.topic_arn,
                "CONFIDENCE_THRESHOLD": str(self.config.get("confidence_threshold", 0.7)),
                "ENVIRONMENT": self.environment_name
            },
            log_retention=logs.RetentionDays.ONE_MONTH
        )

        # Grant permissions
        self.core_stack.dynamodb_table.grant_read_data(drift_function)
        self.alert_topic.grant_publish(drift_function)

        # Grant CloudWatch permissions
        drift_function.add_to_role_policy(
            statement=cloudwatch.PolicyStatement(
                effect=cloudwatch.Effect.ALLOW,
                actions=[
                    "cloudwatch:PutMetricData",
                    "cloudwatch:GetMetricStatistics"
                ],
                resources=["*"]
            )
        )

        # Schedule drift detection to run every hour
        drift_rule = events.Rule(
            self, "DriftDetectionRule",
            rule_name=f"triage-drift-detection-{self.environment_name}",
            schedule=events.Schedule.rate(Duration.hours(1)),
            description="Trigger model drift detection analysis"
        )

        drift_rule.add_target(targets.LambdaFunction(drift_function))

        return drift_function

    def _create_outputs(self) -> None:
        """Create CloudFormation outputs for monitoring."""

        CfnOutput(
            self, "DashboardURL",
            value=f"https://{self.region}.console.aws.amazon.com/cloudwatch/home?region={self.region}#dashboards:name={self.dashboard.dashboard_name}",
            description="CloudWatch Dashboard URL"
        )

        CfnOutput(
            self, "AlertTopicArn",
            value=self.alert_topic.topic_arn,
            description="SNS Topic ARN for system alerts"
        )

        CfnOutput(
            self, "DriftDetectorFunction",
            value=self.drift_detector.function_name,
            description="Model drift detection Lambda function"
        )