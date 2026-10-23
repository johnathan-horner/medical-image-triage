"""
Intelligent triage routing logic for medical image classification results.
Routes predictions to appropriate review queues based on confidence and clinical priority.
"""

import logging
from typing import Dict, Optional, Tuple, List
from datetime import datetime, timedelta
from enum import Enum
import json

from api.models import TriageResult, TriageDecision, PredictionConfidence

logger = logging.getLogger(__name__)


class Priority(Enum):
    """Priority levels for medical review."""
    URGENT = 1      # Immediate attention required
    HIGH = 2        # Same day review
    STANDARD = 3    # Within 24-48 hours


class ReviewerType(Enum):
    """Types of medical reviewers."""
    SENIOR_RADIOLOGIST = "senior_radiologist"
    RADIOLOGIST = "radiologist"
    RESIDENT = "resident"
    AI_SYSTEM = "ai_system"


class ClinicalConditionConfig:
    """Configuration for clinical condition-specific routing."""

    def __init__(self):
        # Clinical condition configurations
        self.condition_configs = {
            "Pneumothorax": {
                "urgency_multiplier": 2.0,  # More urgent condition
                "confidence_threshold_adjustment": -0.1,  # Lower threshold for review
                "default_reviewer": ReviewerType.SENIOR_RADIOLOGIST,
                "max_auto_approve_confidence": 0.95,  # Higher bar for auto-approval
                "description": "Collapsed lung - potentially life-threatening"
            },
            "Pneumonia": {
                "urgency_multiplier": 1.5,
                "confidence_threshold_adjustment": 0.0,
                "default_reviewer": ReviewerType.RADIOLOGIST,
                "max_auto_approve_confidence": 0.9,
                "description": "Infection requiring prompt treatment"
            },
            "Mass": {
                "urgency_multiplier": 1.8,  # Potential malignancy
                "confidence_threshold_adjustment": -0.05,
                "default_reviewer": ReviewerType.SENIOR_RADIOLOGIST,
                "max_auto_approve_confidence": 0.92,
                "description": "Possible tumor requiring urgent evaluation"
            },
            "Infiltration": {
                "urgency_multiplier": 1.2,
                "confidence_threshold_adjustment": 0.0,
                "default_reviewer": ReviewerType.RADIOLOGIST,
                "max_auto_approve_confidence": 0.9,
                "description": "Lung infiltrate requiring evaluation"
            },
            "Normal": {
                "urgency_multiplier": 0.8,  # Lower priority
                "confidence_threshold_adjustment": 0.05,  # Higher threshold for review
                "default_reviewer": ReviewerType.RESIDENT,
                "max_auto_approve_confidence": 0.85,  # Lower bar for auto-approval
                "description": "No acute findings"
            }
        }

    def get_config(self, condition: str) -> Dict:
        """Get configuration for a clinical condition."""
        return self.condition_configs.get(condition, self.condition_configs["Normal"])


class TriageRouter:
    """
    Intelligent triage routing system for medical image predictions.

    Routes predictions based on:
    - Confidence scores
    - Clinical condition severity
    - Current queue loads
    - Time of day/availability
    """

    def __init__(self):
        self.clinical_config = ClinicalConditionConfig()
        self.queue_loads = {
            ReviewerType.SENIOR_RADIOLOGIST: 0,
            ReviewerType.RADIOLOGIST: 0,
            ReviewerType.RESIDENT: 0
        }

        # Thresholds for confidence-based routing
        self.confidence_thresholds = {
            "high_confidence": 0.9,
            "medium_confidence": 0.7,
            "low_confidence": 0.5
        }

    def route_prediction(
        self,
        confidence: float,
        predicted_class: str,
        patient_metadata: Optional[Dict] = None
    ) -> TriageResult:
        """
        Route prediction to appropriate review queue.

        Args:
            confidence: Model confidence score (0-1)
            predicted_class: Predicted medical condition
            patient_metadata: Optional patient metadata for context

        Returns:
            TriageResult with routing decision and details
        """
        logger.info(f"Routing prediction: {predicted_class} (confidence: {confidence:.3f})")

        # Get clinical condition configuration
        condition_config = self.clinical_config.get_config(predicted_class)

        # Adjust confidence threshold based on clinical condition
        adjusted_high_threshold = (
            self.confidence_thresholds["high_confidence"] +
            condition_config["confidence_threshold_adjustment"]
        )
        adjusted_medium_threshold = (
            self.confidence_thresholds["medium_confidence"] +
            condition_config["confidence_threshold_adjustment"]
        )

        # Apply clinical urgency multiplier
        urgency_score = confidence * condition_config["urgency_multiplier"]

        # Determine base routing decision
        if confidence >= adjusted_high_threshold and confidence <= condition_config["max_auto_approve_confidence"]:
            base_decision = TriageDecision.AUTO_APPROVE
        elif confidence >= adjusted_medium_threshold:
            base_decision = TriageDecision.EXPEDITED_REVIEW
        else:
            base_decision = TriageDecision.SENIOR_REVIEW

        # Apply clinical condition overrides
        decision = self._apply_clinical_overrides(
            base_decision, predicted_class, confidence, condition_config
        )

        # Determine priority and reviewer assignment
        priority = self._calculate_priority(decision, urgency_score, predicted_class)
        reviewer_type = self._assign_reviewer(decision, predicted_class, condition_config)

        # Estimate review time
        estimated_time = self._estimate_review_time(decision, priority, reviewer_type)

        # Generate reasoning
        reasoning = self._generate_reasoning(
            decision, confidence, predicted_class, condition_config, urgency_score
        )

        return TriageResult(
            decision=decision,
            priority_level=priority.value,
            estimated_review_time=estimated_time,
            assigned_reviewer_type=reviewer_type.value,
            reasoning=reasoning
        )

    def _apply_clinical_overrides(
        self,
        base_decision: TriageDecision,
        predicted_class: str,
        confidence: float,
        condition_config: Dict
    ) -> TriageDecision:
        """Apply clinical condition-specific overrides to base decision."""

        # Critical conditions always require human review
        critical_conditions = ["Pneumothorax", "Mass"]
        if predicted_class in critical_conditions and base_decision == TriageDecision.AUTO_APPROVE:
            logger.info(f"Override: {predicted_class} requires human review despite high confidence")
            return TriageDecision.EXPEDITED_REVIEW

        # Very low confidence always goes to senior review
        if confidence < 0.6:
            logger.info(f"Override: Very low confidence ({confidence:.3f}) requires senior review")
            return TriageDecision.SENIOR_REVIEW

        # Normal findings with very high confidence can be auto-approved
        if predicted_class == "Normal" and confidence > 0.95:
            logger.info(f"Override: Normal finding with very high confidence ({confidence:.3f}) auto-approved")
            return TriageDecision.AUTO_APPROVE

        return base_decision

    def _calculate_priority(
        self,
        decision: TriageDecision,
        urgency_score: float,
        predicted_class: str
    ) -> Priority:
        """Calculate priority level based on decision and clinical factors."""

        # Critical conditions get higher priority
        if predicted_class in ["Pneumothorax", "Mass"]:
            return Priority.URGENT

        # High urgency score increases priority
        if urgency_score > 1.5:
            return Priority.HIGH

        # Decision-based priority
        if decision == TriageDecision.SENIOR_REVIEW:
            return Priority.HIGH
        elif decision == TriageDecision.EXPEDITED_REVIEW:
            return Priority.HIGH
        else:
            return Priority.STANDARD

    def _assign_reviewer(
        self,
        decision: TriageDecision,
        predicted_class: str,
        condition_config: Dict
    ) -> ReviewerType:
        """Assign appropriate reviewer based on decision and condition."""

        if decision == TriageDecision.AUTO_APPROVE:
            return ReviewerType.AI_SYSTEM

        elif decision == TriageDecision.SENIOR_REVIEW:
            return ReviewerType.SENIOR_RADIOLOGIST

        else:  # EXPEDITED_REVIEW
            # Use condition-specific default reviewer
            default_reviewer = condition_config["default_reviewer"]

            # Check queue loads and potentially reassign
            if self.queue_loads[default_reviewer] > 10:
                # If queue is overloaded, escalate or delegate
                if default_reviewer == ReviewerType.RADIOLOGIST:
                    if self.queue_loads[ReviewerType.SENIOR_RADIOLOGIST] < 5:
                        return ReviewerType.SENIOR_RADIOLOGIST
                elif default_reviewer == ReviewerType.RESIDENT:
                    if self.queue_loads[ReviewerType.RADIOLOGIST] < 8:
                        return ReviewerType.RADIOLOGIST

            return default_reviewer

    def _estimate_review_time(
        self,
        decision: TriageDecision,
        priority: Priority,
        reviewer_type: ReviewerType
    ) -> Optional[int]:
        """Estimate review time in minutes based on decision and queue."""

        if decision == TriageDecision.AUTO_APPROVE:
            return None  # No human review required

        # Base review times by reviewer type (in minutes)
        base_times = {
            ReviewerType.SENIOR_RADIOLOGIST: 15,
            ReviewerType.RADIOLOGIST: 10,
            ReviewerType.RESIDENT: 20
        }

        base_time = base_times[reviewer_type]

        # Adjust for queue load
        queue_load = self.queue_loads[reviewer_type]
        queue_multiplier = 1 + (queue_load * 0.1)  # 10% increase per queued item

        # Adjust for priority
        priority_multipliers = {
            Priority.URGENT: 0.5,    # Expedited
            Priority.HIGH: 0.8,      # Faster processing
            Priority.STANDARD: 1.0   # Normal processing
        }

        estimated_time = base_time * queue_multiplier * priority_multipliers[priority]

        return int(estimated_time)

    def _generate_reasoning(
        self,
        decision: TriageDecision,
        confidence: float,
        predicted_class: str,
        condition_config: Dict,
        urgency_score: float
    ) -> str:
        """Generate human-readable reasoning for the triage decision."""

        reasoning_parts = []

        # Add prediction details
        reasoning_parts.append(f"Predicted {predicted_class} with {confidence:.1%} confidence")

        # Add clinical context
        reasoning_parts.append(f"Condition: {condition_config['description']}")

        # Add decision reasoning
        if decision == TriageDecision.AUTO_APPROVE:
            reasoning_parts.append("High confidence allows automatic approval")
        elif decision == TriageDecision.EXPEDITED_REVIEW:
            reasoning_parts.append("Medium confidence requires expedited physician review")
        else:
            reasoning_parts.append("Low confidence or critical findings require senior physician review")

        # Add urgency context
        if urgency_score > 1.5:
            reasoning_parts.append("Clinical urgency factor applied")

        # Add specific overrides if applicable
        if predicted_class in ["Pneumothorax", "Mass"] and confidence > 0.9:
            reasoning_parts.append("Critical condition requires human verification despite high AI confidence")

        return ". ".join(reasoning_parts)

    def update_queue_load(self, reviewer_type: ReviewerType, load: int) -> None:
        """Update queue load for a reviewer type."""
        self.queue_loads[reviewer_type] = load
        logger.info(f"Updated queue load for {reviewer_type.value}: {load}")

    def get_queue_status(self) -> Dict[str, int]:
        """Get current queue loads for all reviewer types."""
        return {reviewer_type.value: load for reviewer_type, load in self.queue_loads.items()}

    def get_configuration_summary(self) -> Dict:
        """Get summary of current triage configuration."""
        return {
            "confidence_thresholds": self.confidence_thresholds,
            "clinical_conditions": {
                condition: {
                    "urgency_multiplier": config["urgency_multiplier"],
                    "default_reviewer": config["default_reviewer"].value,
                    "description": config["description"]
                }
                for condition, config in self.clinical_config.condition_configs.items()
            },
            "current_queue_loads": self.get_queue_status()
        }