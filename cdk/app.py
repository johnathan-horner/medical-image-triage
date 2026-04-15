#!/usr/bin/env python3
"""
AWS CDK app entry point for Medical Image Triage System.
Deploys production-grade infrastructure with HIPAA compliance.
"""

import os
from aws_cdk import App, Environment, Tags

from image_triage_stack import ImageTriageStack
from monitoring_stack import MonitoringStack

app = App()

# Get environment configuration
account = app.node.try_get_context("account") or os.environ.get("CDK_DEFAULT_ACCOUNT")
region = app.node.try_get_context("region") or os.environ.get("CDK_DEFAULT_REGION", "us-east-1")
environment_name = app.node.try_get_context("environment") or "dev"

env = Environment(account=account, region=region)

# Deploy core infrastructure stack
core_stack = ImageTriageStack(
    app,
    f"ImageTriageStack-{environment_name}",
    env=env,
    environment_name=environment_name,
    description="Medical Image Triage System - Core Infrastructure"
)

# Deploy monitoring stack
monitoring_stack = MonitoringStack(
    app,
    f"ImageTriageMonitoring-{environment_name}",
    env=env,
    environment_name=environment_name,
    core_stack=core_stack,
    description="Medical Image Triage System - Monitoring and Alerting"
)

# Add stack dependencies
monitoring_stack.add_dependency(core_stack)

# Add common tags
for stack in [core_stack, monitoring_stack]:
    Tags.of(stack).add("Project", "MedicalImageTriage")
    Tags.of(stack).add("Environment", environment_name)
    Tags.of(stack).add("Owner", "ML-Team")
    Tags.of(stack).add("CostCenter", "Research")
    Tags.of(stack).add("Compliance", "HIPAA")

app.synth()