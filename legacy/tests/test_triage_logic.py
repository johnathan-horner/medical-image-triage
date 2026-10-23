"""
Unit tests for triage routing logic.
Tests confidence-based routing, clinical overrides, and queue management.
"""

import pytest
from unittest.mock import Mock, patch
from routing.triage_logic import TriageRouter, ReviewerType, Priority
from api.models import TriageDecision, PredictionConfidence


class TestTriageRouter:
    """Test suite for the TriageRouter class."""

    @pytest.fixture
    def router(self):
        """Create a TriageRouter instance for testing."""
        return TriageRouter()

    def test_high_confidence_normal_auto_approve(self, router):
        """Test auto-approval for high confidence normal findings."""
        result = router.route_prediction(
            confidence=0.96,
            predicted_class="Normal"
        )

        assert result.decision == TriageDecision.AUTO_APPROVE
        assert result.assigned_reviewer_type == ReviewerType.AI_SYSTEM.value
        assert result.priority_level == Priority.STANDARD.value
        assert "High confidence allows automatic approval" in result.reasoning

    def test_high_confidence_pneumothorax_requires_review(self, router):
        """Test that pneumothorax requires human review despite high confidence."""
        result = router.route_prediction(
            confidence=0.95,
            predicted_class="Pneumothorax"
        )

        assert result.decision == TriageDecision.EXPEDITED_REVIEW
        assert result.assigned_reviewer_type == ReviewerType.SENIOR_RADIOLOGIST.value
        assert result.priority_level == Priority.URGENT.value
        assert "Critical condition requires human verification" in result.reasoning

    def test_medium_confidence_expedited_review(self, router):
        """Test expedited review for medium confidence predictions."""
        result = router.route_prediction(
            confidence=0.8,
            predicted_class="Pneumonia"
        )

        assert result.decision == TriageDecision.EXPEDITED_REVIEW
        assert result.assigned_reviewer_type == ReviewerType.RADIOLOGIST.value
        assert result.priority_level == Priority.HIGH.value

    def test_low_confidence_senior_review(self, router):
        """Test senior review for low confidence predictions."""
        result = router.route_prediction(
            confidence=0.5,
            predicted_class="Mass"
        )

        assert result.decision == TriageDecision.SENIOR_REVIEW
        assert result.assigned_reviewer_type == ReviewerType.SENIOR_RADIOLOGIST.value
        assert result.priority_level == Priority.HIGH.value

    def test_very_low_confidence_override(self, router):
        """Test override for very low confidence predictions."""
        result = router.route_prediction(
            confidence=0.45,
            predicted_class="Normal"
        )

        assert result.decision == TriageDecision.SENIOR_REVIEW
        assert "Very low confidence" in result.reasoning

    def test_mass_urgent_priority(self, router):
        """Test that mass predictions get urgent priority."""
        result = router.route_prediction(
            confidence=0.85,
            predicted_class="Mass"
        )

        assert result.priority_level == Priority.URGENT.value
        assert result.assigned_reviewer_type == ReviewerType.SENIOR_RADIOLOGIST.value

    def test_normal_very_high_confidence_auto_approve(self, router):
        """Test auto-approval override for very high confidence normal findings."""
        result = router.route_prediction(
            confidence=0.98,
            predicted_class="Normal"
        )

        assert result.decision == TriageDecision.AUTO_APPROVE
        assert "Normal finding with very high confidence" in result.reasoning

    def test_clinical_urgency_multiplier(self, router):
        """Test that clinical urgency affects routing."""
        # Test pneumothorax with urgency multiplier
        result = router.route_prediction(
            confidence=0.75,
            predicted_class="Pneumothorax"
        )

        assert result.priority_level == Priority.URGENT.value
        assert "Clinical urgency factor applied" in result.reasoning

    def test_estimated_review_time_calculation(self, router):
        """Test review time estimation."""
        result = router.route_prediction(
            confidence=0.8,
            predicted_class="Pneumonia"
        )

        # Should have estimated review time for human review
        assert result.estimated_review_time is not None
        assert result.estimated_review_time > 0

        # Auto-approved should have no review time
        auto_result = router.route_prediction(
            confidence=0.96,
            predicted_class="Normal"
        )
        assert auto_result.estimated_review_time is None

    def test_queue_load_management(self, router):
        """Test queue load affects reviewer assignment."""
        # Set high queue load for radiologists
        router.update_queue_load(ReviewerType.RADIOLOGIST, 12)

        result = router.route_prediction(
            confidence=0.8,
            predicted_class="Pneumonia"  # Usually goes to radiologist
        )

        # Should escalate to senior radiologist due to queue load
        assert result.assigned_reviewer_type == ReviewerType.SENIOR_RADIOLOGIST.value

    def test_configuration_summary(self, router):
        """Test configuration summary retrieval."""
        config = router.get_configuration_summary()

        assert "confidence_thresholds" in config
        assert "clinical_conditions" in config
        assert "current_queue_loads" in config
        assert config["confidence_thresholds"]["high_confidence"] == 0.9

    def test_all_conditions_covered(self, router):
        """Test that all medical conditions have appropriate routing."""
        conditions = ["Normal", "Pneumonia", "Pneumothorax", "Infiltration", "Mass"]

        for condition in conditions:
            result = router.route_prediction(
                confidence=0.8,
                predicted_class=condition
            )

            # All should have valid decisions and reasoning
            assert result.decision in [
                TriageDecision.AUTO_APPROVE,
                TriageDecision.EXPEDITED_REVIEW,
                TriageDecision.SENIOR_REVIEW
            ]
            assert len(result.reasoning) > 0
            assert result.priority_level in [1, 2, 3]

    @pytest.mark.parametrize("confidence,expected_decision", [
        (0.95, TriageDecision.AUTO_APPROVE),
        (0.85, TriageDecision.EXPEDITED_REVIEW),
        (0.65, TriageDecision.SENIOR_REVIEW),
        (0.45, TriageDecision.SENIOR_REVIEW),
    ])
    def test_confidence_thresholds(self, router, confidence, expected_decision):
        """Test confidence threshold boundaries."""
        result = router.route_prediction(
            confidence=confidence,
            predicted_class="Infiltration"  # Neutral condition
        )

        assert result.decision == expected_decision

    def test_reasoning_generation(self, router):
        """Test that reasoning is comprehensive and informative."""
        result = router.route_prediction(
            confidence=0.85,
            predicted_class="Pneumonia"
        )

        reasoning = result.reasoning

        # Should include prediction details
        assert "Pneumonia" in reasoning
        assert "85%" in reasoning or "0.85" in reasoning

        # Should include decision reasoning
        assert any(phrase in reasoning.lower() for phrase in [
            "expedited", "review", "confidence"
        ])

    def test_metadata_handling(self, router):
        """Test handling of patient metadata."""
        metadata = {"age": 65, "emergency": True}

        result = router.route_prediction(
            confidence=0.8,
            predicted_class="Pneumonia",
            patient_metadata=metadata
        )

        # Should still work with metadata (currently not used but should not error)
        assert result.decision == TriageDecision.EXPEDITED_REVIEW


class TestClinicalConditionConfig:
    """Test suite for clinical condition configurations."""

    def test_condition_config_completeness(self):
        """Test that all expected conditions have complete configurations."""
        from routing.triage_logic import ClinicalConditionConfig

        config = ClinicalConditionConfig()
        conditions = ["Normal", "Pneumonia", "Pneumothorax", "Infiltration", "Mass"]

        for condition in conditions:
            condition_config = config.get_config(condition)

            # Each condition should have all required fields
            assert "urgency_multiplier" in condition_config
            assert "confidence_threshold_adjustment" in condition_config
            assert "default_reviewer" in condition_config
            assert "max_auto_approve_confidence" in condition_config
            assert "description" in condition_config

    def test_critical_conditions_higher_urgency(self):
        """Test that critical conditions have higher urgency multipliers."""
        from routing.triage_logic import ClinicalConditionConfig

        config = ClinicalConditionConfig()

        # Critical conditions should have higher urgency
        pneumothorax_config = config.get_config("Pneumothorax")
        mass_config = config.get_config("Mass")
        normal_config = config.get_config("Normal")

        assert pneumothorax_config["urgency_multiplier"] > normal_config["urgency_multiplier"]
        assert mass_config["urgency_multiplier"] > normal_config["urgency_multiplier"]

    def test_unknown_condition_default(self):
        """Test handling of unknown medical conditions."""
        from routing.triage_logic import ClinicalConditionConfig

        config = ClinicalConditionConfig()
        unknown_config = config.get_config("UnknownCondition")
        normal_config = config.get_config("Normal")

        # Should return normal configuration as default
        assert unknown_config == normal_config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])