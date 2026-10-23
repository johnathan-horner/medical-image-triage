"""
Medical Image Triage MCP Server

FastMCP server exposing medical image triage functionality through Model Context Protocol.
Provides tools for image classification, audit trails, dashboard metrics, and drift detection.
"""

import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from PIL import Image
import os
import io
import base64

from fastmcp import FastMCP

# Initialize MCP server
mcp = FastMCP("Medical Image Triage")

# Mock data for demo purposes (in production, this would connect to actual databases)
MOCK_AUDIT_DATA = [
    {
        "timestamp": "2024-04-15T10:30:00Z",
        "image_hash": "abc123normal",
        "prediction_id": "pred-12345-normal",
        "classification": "No Finding",
        "confidence": 0.94,
        "triage_decision": "auto_triage",
        "reviewer_type": "auto_approved",
        "processing_time_ms": 187.3
    },
    {
        "timestamp": "2024-04-15T11:45:00Z",
        "image_hash": "def456pneumonia",
        "prediction_id": "pred-67890-pneumonia",
        "classification": "Pneumonia",
        "confidence": 0.87,
        "triage_decision": "expedited_review",
        "reviewer_type": "radiologist",
        "processing_time_ms": 234.7
    }
]

MOCK_METRICS = {
    "total_images_processed": 2847,
    "avg_confidence": 0.82,
    "drift_status": "normal",
    "psi_score": 0.15,
    "routing_distribution": {
        "auto_triage": 0.65,
        "expedited_review": 0.25,
        "senior_review": 0.10
    },
    "last_updated": "2024-04-15T12:00:00Z"
}

@mcp.tool()
def triage_image(image_path: str) -> Dict[str, Any]:
    """
    Classify a medical image and determine triage routing decision.

    Args:
        image_path: Path to the medical image file

    Returns:
        Dictionary containing:
        - diagnosis_label: Predicted medical condition
        - confidence_score: Model confidence (0.0-1.0)
        - routing_decision: auto_triage/expedited_review/senior_review
        - processing_time_ms: Time taken for classification
        - prediction_id: Unique identifier for this prediction
    """
    try:
        # Simulate image processing (in production, would use actual model)
        if not os.path.exists(image_path):
            return {"error": f"Image file not found: {image_path}"}

        # Generate hash for the image
        with open(image_path, 'rb') as f:
            image_hash = hashlib.md5(f.read()).hexdigest()

        # Mock classification based on filename patterns for demo
        filename = os.path.basename(image_path).lower()
        if 'normal' in filename or 'healthy' in filename:
            result = {
                "diagnosis_label": "No Finding",
                "confidence_score": 0.94,
                "routing_decision": "auto_triage",
                "processing_time_ms": 187.3,
                "prediction_id": f"pred-{image_hash[:8]}-normal"
            }
        elif 'pneumonia' in filename:
            result = {
                "diagnosis_label": "Pneumonia",
                "confidence_score": 0.87,
                "routing_decision": "expedited_review",
                "processing_time_ms": 234.7,
                "prediction_id": f"pred-{image_hash[:8]}-pneumonia"
            }
        else:
            result = {
                "diagnosis_label": "Cardiomegaly",
                "confidence_score": 0.62,
                "routing_decision": "senior_review",
                "processing_time_ms": 298.1,
                "prediction_id": f"pred-{image_hash[:8]}-cardio"
            }

        result["image_hash"] = image_hash
        result["timestamp"] = datetime.now().isoformat()

        return result

    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def get_audit_trail(start_date: Optional[str] = None, end_date: Optional[str] = None, image_hash: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Retrieve audit trail entries for medical image classifications.

    Args:
        start_date: Start date for filtering (ISO format, optional)
        end_date: End date for filtering (ISO format, optional)
        image_hash: Specific image hash to filter by (optional)

    Returns:
        List of audit trail entries with timestamps, predictions, and metadata
    """
    results = MOCK_AUDIT_DATA.copy()

    # Filter by image hash if provided
    if image_hash:
        results = [entry for entry in results if entry.get("image_hash") == image_hash]

    # Filter by date range if provided
    if start_date or end_date:
        filtered_results = []
        for entry in results:
            entry_date = datetime.fromisoformat(entry["timestamp"].replace('Z', '+00:00'))

            if start_date:
                start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                if entry_date < start:
                    continue

            if end_date:
                end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                if entry_date > end:
                    continue

            filtered_results.append(entry)
        results = filtered_results

    return results

@mcp.tool()
def get_dashboard_metrics() -> Dict[str, Any]:
    """
    Get current dashboard metrics for the medical image triage system.

    Returns:
        Dictionary containing:
        - total_images_processed: Total number of images classified
        - avg_confidence: Average model confidence across all predictions
        - drift_status: Current model drift status (normal/warning/critical)
        - routing_distribution: Breakdown of triage decisions
        - psi_score: Population Stability Index score
        - last_updated: Timestamp of last metrics update
    """
    return MOCK_METRICS.copy()

@mcp.tool()
def check_drift_status() -> Dict[str, Any]:
    """
    Check the current model drift status and PSI score.

    Returns:
        Dictionary containing:
        - psi_score: Current Population Stability Index score
        - threshold_exceeded: Boolean indicating if drift threshold is exceeded
        - status: Current drift status (normal/warning/critical)
        - recommendation: Recommended action based on drift level
    """
    psi_score = MOCK_METRICS["psi_score"]

    if psi_score < 0.1:
        status = "normal"
        recommendation = "No action required. Model performance is stable."
    elif psi_score < 0.2:
        status = "warning"
        recommendation = "Monitor closely. Consider model refresh if trend continues."
    else:
        status = "critical"
        recommendation = "Immediate model retraining required. High drift detected."

    return {
        "psi_score": psi_score,
        "threshold_exceeded": psi_score >= 0.1,
        "status": status,
        "recommendation": recommendation,
        "last_checked": datetime.now().isoformat()
    }

@mcp.resource("model://efficientnet")
def get_model_info() -> str:
    """
    Get information about the EfficientNetB0 model used for image classification.

    Returns model details including version, accuracy metrics, and training data summary.
    """
    model_info = {
        "model_name": "EfficientNetB0",
        "version": "1.2.1",
        "architecture": "EfficientNet-B0",
        "input_size": "224x224",
        "num_classes": 14,
        "accuracy_metrics": {
            "overall_accuracy": 0.847,
            "weighted_f1_score": 0.831,
            "auc_roc": 0.892
        },
        "training_data": {
            "dataset": "ChestX-ray14",
            "total_images": 112120,
            "training_split": 0.8,
            "validation_split": 0.1,
            "test_split": 0.1
        },
        "classes": [
            "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
            "Mass", "Nodule", "Pneumonia", "Pneumothorax",
            "Consolidation", "Edema", "Emphysema", "Fibrosis",
            "Pleural_Thickening", "No Finding"
        ],
        "last_updated": "2024-03-15T10:00:00Z"
    }

    return json.dumps(model_info, indent=2)

@mcp.resource("data://routing_rules")
def get_routing_rules() -> str:
    """
    Get the confidence threshold routing configuration for triage decisions.

    Returns routing rules for auto-triage, expedited review, and senior physician review.
    """
    routing_rules = {
        "confidence_thresholds": {
            "auto_triage": {
                "min_confidence": 0.9,
                "description": "High confidence predictions auto-approved",
                "reviewer_type": "auto_approved",
                "max_review_time_minutes": 0
            },
            "expedited_review": {
                "min_confidence": 0.7,
                "max_confidence": 0.89,
                "description": "Medium confidence requires radiologist review",
                "reviewer_type": "radiologist",
                "max_review_time_minutes": 15
            },
            "senior_review": {
                "max_confidence": 0.69,
                "description": "Low confidence requires senior radiologist",
                "reviewer_type": "senior_radiologist",
                "max_review_time_minutes": 45
            }
        },
        "special_conditions": {
            "critical_findings": ["Pneumothorax", "Mass", "Pneumonia"],
            "expedite_regardless_of_confidence": True,
            "description": "Critical conditions always get expedited review"
        },
        "last_updated": "2024-03-01T09:00:00Z"
    }

    return json.dumps(routing_rules, indent=2)

@mcp.prompt()
def triage_review() -> str:
    """
    Pre-built prompt for reviewing a batch of triage decisions with confidence analysis.

    Use this prompt to analyze multiple triage decisions and provide clinical insights.
    """
    return """You are a senior radiologist reviewing AI-assisted medical image triage decisions.

Please analyze the following batch of triage decisions and provide your clinical assessment:

**Review Criteria:**
1. Accuracy of AI classifications vs your clinical judgment
2. Appropriateness of confidence scores
3. Correctness of routing decisions (auto/expedited/senior review)
4. Any patterns suggesting model drift or bias
5. Recommendations for improving triage protocols

**For each case, consider:**
- Clinical context and patient safety implications
- Whether the confidence threshold appropriately matched the routing decision
- Any edge cases that might require protocol adjustments
- Overall system performance and reliability

**Please provide:**
- Individual case assessments
- Overall batch performance summary
- Specific recommendations for protocol improvements
- Any concerns about patient safety or workflow efficiency

Focus on actionable clinical insights that can improve patient outcomes and workflow optimization."""

@mcp.prompt()
def compliance_audit() -> str:
    """
    Pre-built prompt for generating a HIPAA compliance summary of recent triage activity.

    Use this prompt to assess HIPAA compliance across medical image processing activities.
    """
    return """You are conducting a HIPAA compliance audit of the Medical Image Triage System.

Please review the following system activity and generate a compliance assessment:

**HIPAA Requirements to Evaluate:**
1. **Minimum Necessary Standard** - Was only necessary PHI accessed/processed?
2. **Access Controls** - Were appropriate user authentication and authorization used?
3. **Audit Logging** - Are all PHI access events properly logged with required details?
4. **Data Integrity** - Were safeguards in place to prevent unauthorized alteration?
5. **Transmission Security** - Was PHI properly encrypted during transmission?
6. **Data Retention** - Are retention policies being followed correctly?

**Assessment Areas:**
- User access patterns and privilege escalation
- System logs completeness and integrity
- Data encryption at rest and in transit
- Incident response procedures
- Training compliance for system users
- Business Associate Agreement compliance

**Required Deliverables:**
1. Compliance status summary (Compliant/Non-Compliant/Needs Improvement)
2. Detailed findings for each HIPAA requirement
3. Risk assessment of any identified gaps
4. Remediation recommendations with timelines
5. Ongoing monitoring recommendations

Focus on patient privacy protection and regulatory compliance to ensure the system meets all HIPAA requirements."""

if __name__ == "__main__":
    mcp.run()