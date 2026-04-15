"""
Unit tests for HIPAA compliance and audit logging functionality.
Tests secure logging, data protection, and audit trail queries.
"""

import pytest
import tempfile
import hashlib
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from audit.compliance import ComplianceLogger
from audit.database import DatabaseManager, PredictionAuditLog, Base
from api.models import AuditQuery, TriageResult, TriageDecision


@pytest.fixture
def test_database():
    """Create a temporary test database."""
    # Use in-memory SQLite for testing
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    return engine, SessionLocal


@pytest.fixture
def compliance_logger(test_database):
    """Create ComplianceLogger with test database."""
    engine, session_factory = test_database

    logger = ComplianceLogger(database_url="sqlite:///:memory:")
    # Replace the session factory with our test one
    logger.db_manager.SessionLocal = session_factory

    return logger


class TestComplianceLogger:
    """Test suite for ComplianceLogger class."""

    def test_log_prediction_success(self, compliance_logger):
        """Test successful prediction logging."""
        # Create test data
        prediction_result = {
            "predicted_class": "Normal",
            "confidence": 0.95,
            "confidence_level": "high",
            "all_scores": {"Normal": 0.95, "Pneumonia": 0.05}
        }

        triage_decision = TriageResult(
            decision=TriageDecision.AUTO_APPROVE,
            priority_level=3,
            estimated_review_time=None,
            assigned_reviewer_type="ai_system",
            reasoning="High confidence allows automatic approval"
        )

        # Log prediction
        success = compliance_logger.log_prediction(
            prediction_id="test-pred-123",
            patient_id="patient-456",
            study_id="study-789",
            image_hash="abc123hash",
            prediction_result=prediction_result,
            triage_decision=triage_decision,
            processing_time_ms=150.5
        )

        assert success is True

        # Verify data was logged correctly
        with compliance_logger.db_manager.get_session() as session:
            logged_entry = session.query(PredictionAuditLog).filter(
                PredictionAuditLog.prediction_id == "test-pred-123"
            ).first()

            assert logged_entry is not None
            assert logged_entry.predicted_class == "Normal"
            assert logged_entry.confidence == 0.95
            assert logged_entry.triage_decision == "auto_approve"

            # Verify patient ID is hashed
            expected_hash = hashlib.sha256("patient-456".encode()).hexdigest()
            assert logged_entry.patient_id == expected_hash

    def test_log_review_decision(self, compliance_logger):
        """Test logging physician review decisions."""
        # First log a prediction
        prediction_result = {
            "predicted_class": "Pneumonia",
            "confidence": 0.8,
            "confidence_level": "medium",
            "all_scores": {"Pneumonia": 0.8, "Normal": 0.2}
        }

        triage_decision = TriageResult(
            decision=TriageDecision.EXPEDITED_REVIEW,
            priority_level=2,
            estimated_review_time=15,
            assigned_reviewer_type="radiologist",
            reasoning="Medium confidence requires review"
        )

        compliance_logger.log_prediction(
            prediction_id="test-review-123",
            patient_id="patient-456",
            study_id="study-789",
            image_hash="def456hash",
            prediction_result=prediction_result,
            triage_decision=triage_decision,
            processing_time_ms=200.0
        )

        # Log review decision
        success = compliance_logger.log_review_decision(
            prediction_id="test-review-123",
            reviewer_id="dr-smith-123",
            review_decision="approved",
            review_notes="Confirmed pneumonia diagnosis",
            review_confidence=0.9
        )

        assert success is True

        # Verify review was logged
        with compliance_logger.db_manager.get_session() as session:
            logged_entry = session.query(PredictionAuditLog).filter(
                PredictionAuditLog.prediction_id == "test-review-123"
            ).first()

            assert logged_entry.review_decision == "approved"
            assert logged_entry.review_notes == "Confirmed pneumonia diagnosis"
            assert logged_entry.review_confidence == 0.9

            # Verify reviewer ID is hashed
            expected_hash = hashlib.sha256("dr-smith-123".encode()).hexdigest()
            assert logged_entry.reviewer_id == expected_hash

    def test_query_audit_log_basic(self, compliance_logger):
        """Test basic audit log querying."""
        # Log some test predictions
        self._create_test_predictions(compliance_logger)

        # Query all predictions
        query = AuditQuery(limit=10)
        results = compliance_logger.query_audit_log(query)

        assert len(results) == 3
        assert all(entry.prediction_id.startswith("test-") for entry in results)

    def test_query_audit_log_with_filters(self, compliance_logger):
        """Test audit log querying with filters."""
        # Log test predictions
        self._create_test_predictions(compliance_logger)

        # Query by predicted class
        query = AuditQuery(predicted_class="Pneumonia", limit=10)
        results = compliance_logger.query_audit_log(query)

        assert len(results) == 1
        assert results[0].predicted_class == "Pneumonia"

        # Query by triage decision
        query = AuditQuery(triage_decision=TriageDecision.AUTO_APPROVE, limit=10)
        results = compliance_logger.query_audit_log(query)

        assert len(results) == 2  # Normal predictions are auto-approved

    def test_query_audit_log_date_range(self, compliance_logger):
        """Test audit log querying with date filters."""
        # Log test predictions
        self._create_test_predictions(compliance_logger)

        # Query with date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(hours=1)

        query = AuditQuery(start_date=start_date, end_date=end_date, limit=10)
        results = compliance_logger.query_audit_log(query)

        assert len(results) == 3  # All recent predictions
        for result in results:
            assert start_date <= result.timestamp <= end_date

    def test_get_prediction_chain(self, compliance_logger):
        """Test retrieving complete prediction decision chain."""
        # Log a prediction with review
        prediction_result = {
            "predicted_class": "Mass",
            "confidence": 0.85,
            "confidence_level": "medium",
            "all_scores": {"Mass": 0.85, "Normal": 0.15}
        }

        triage_decision = TriageResult(
            decision=TriageDecision.SENIOR_REVIEW,
            priority_level=1,
            estimated_review_time=30,
            assigned_reviewer_type="senior_radiologist",
            reasoning="Potential mass requires senior review"
        )

        compliance_logger.log_prediction(
            prediction_id="chain-test-123",
            patient_id="patient-chain",
            study_id="study-chain",
            image_hash="chain123hash",
            prediction_result=prediction_result,
            triage_decision=triage_decision,
            processing_time_ms=300.0
        )

        # Add review
        compliance_logger.log_review_decision(
            prediction_id="chain-test-123",
            reviewer_id="senior-dr-jones",
            review_decision="approved",
            review_notes="Mass confirmed, urgent follow-up required",
            review_confidence=0.95
        )

        # Get complete chain
        chain = compliance_logger.get_prediction_chain("chain-test-123")

        assert chain is not None
        assert chain["prediction_id"] == "chain-test-123"
        assert chain["prediction"]["class"] == "Mass"
        assert chain["triage"]["decision"] == "senior_review"
        assert chain["review"] is not None
        assert chain["review"]["decision"] == "approved"

    def test_generate_compliance_report(self, compliance_logger):
        """Test compliance report generation."""
        # Log test predictions
        self._create_test_predictions(compliance_logger)

        # Add some reviews
        compliance_logger.log_review_decision(
            prediction_id="test-pred-2",
            reviewer_id="dr-test",
            review_decision="approved"
        )

        # Generate report
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=1)

        report = compliance_logger.generate_compliance_report(start_date, end_date)

        assert "period" in report
        assert "metrics" in report
        assert "compliance" in report

        metrics = report["metrics"]
        assert metrics["total_predictions"] == 3
        assert metrics["reviewed_predictions"] == 1
        assert metrics["auto_approved"] == 2

        compliance_info = report["compliance"]
        assert compliance_info["hipaa_compliant"] is True
        assert compliance_info["audit_trail_complete"] is True

    def test_identifier_hashing(self, compliance_logger):
        """Test that identifiers are properly hashed for HIPAA compliance."""
        test_id = "sensitive-patient-id-12345"
        hashed_id = compliance_logger._hash_identifier(test_id)

        # Should be SHA256 hash
        assert len(hashed_id) == 64  # SHA256 produces 64-character hex string
        assert hashed_id != test_id  # Should not be plaintext
        assert all(c in '0123456789abcdef' for c in hashed_id)  # Should be hex

        # Same input should produce same hash
        hashed_id2 = compliance_logger._hash_identifier(test_id)
        assert hashed_id == hashed_id2

    def test_data_retention_policy(self, compliance_logger):
        """Test data retention date setting."""
        prediction_result = {
            "predicted_class": "Normal",
            "confidence": 0.95,
            "confidence_level": "high",
            "all_scores": {"Normal": 0.95}
        }

        triage_decision = TriageResult(
            decision=TriageDecision.AUTO_APPROVE,
            priority_level=3,
            estimated_review_time=None,
            assigned_reviewer_type="ai_system",
            reasoning="Test"
        )

        compliance_logger.log_prediction(
            prediction_id="retention-test",
            patient_id="patient-retention",
            study_id="study-retention",
            image_hash="retention123",
            prediction_result=prediction_result,
            triage_decision=triage_decision,
            processing_time_ms=100.0
        )

        # Check retention date is set (7 years = 2555 days)
        with compliance_logger.db_manager.get_session() as session:
            entry = session.query(PredictionAuditLog).filter(
                PredictionAuditLog.prediction_id == "retention-test"
            ).first()

            assert entry.retention_date is not None
            expected_retention = entry.timestamp + timedelta(days=2555)
            assert abs((entry.retention_date - expected_retention).days) <= 1

    def test_log_system_event(self, compliance_logger):
        """Test system event logging."""
        success = compliance_logger.log_system_event(
            event_type="user_login",
            action_performed="authentication",
            outcome="success",
            user_id="user-123",
            resource_accessed="/dashboard",
            details={"method": "oauth"},
            ip_address="192.168.1.1",
            session_id="sess-abc123"
        )

        assert success is True

        # Verify system event was logged
        with compliance_logger.db_manager.get_session() as session:
            from audit.database import SystemAuditLog
            event = session.query(SystemAuditLog).filter(
                SystemAuditLog.event_type == "user_login"
            ).first()

            assert event is not None
            assert event.action_performed == "authentication"
            assert event.outcome == "success"
            assert event.ip_address == "192.168.1.1"

            # Verify user ID is hashed
            expected_hash = hashlib.sha256("user-123".encode()).hexdigest()
            assert event.user_id == expected_hash

    def test_audit_log_pagination(self, compliance_logger):
        """Test audit log pagination."""
        # Create many test predictions
        for i in range(25):
            prediction_result = {
                "predicted_class": "Normal",
                "confidence": 0.95,
                "confidence_level": "high",
                "all_scores": {"Normal": 0.95}
            }

            triage_decision = TriageResult(
                decision=TriageDecision.AUTO_APPROVE,
                priority_level=3,
                estimated_review_time=None,
                assigned_reviewer_type="ai_system",
                reasoning="Test"
            )

            compliance_logger.log_prediction(
                prediction_id=f"pagination-test-{i}",
                patient_id=f"patient-{i}",
                study_id=f"study-{i}",
                image_hash=f"hash{i}",
                prediction_result=prediction_result,
                triage_decision=triage_decision,
                processing_time_ms=100.0
            )

        # Test pagination
        query1 = AuditQuery(limit=10, offset=0)
        results1 = compliance_logger.query_audit_log(query1)
        assert len(results1) == 10

        query2 = AuditQuery(limit=10, offset=10)
        results2 = compliance_logger.query_audit_log(query2)
        assert len(results2) == 10

        # Ensure different results
        ids1 = {r.prediction_id for r in results1}
        ids2 = {r.prediction_id for r in results2}
        assert len(ids1.intersection(ids2)) == 0

    def _create_test_predictions(self, compliance_logger):
        """Helper method to create test predictions."""
        test_cases = [
            ("test-pred-1", "Normal", 0.95, TriageDecision.AUTO_APPROVE),
            ("test-pred-2", "Pneumonia", 0.8, TriageDecision.EXPEDITED_REVIEW),
            ("test-pred-3", "Normal", 0.92, TriageDecision.AUTO_APPROVE),
        ]

        for pred_id, predicted_class, confidence, decision in test_cases:
            prediction_result = {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "confidence_level": "high" if confidence > 0.9 else "medium",
                "all_scores": {predicted_class: confidence, "Other": 1 - confidence}
            }

            triage_decision = TriageResult(
                decision=decision,
                priority_level=2 if decision != TriageDecision.AUTO_APPROVE else 3,
                estimated_review_time=15 if decision != TriageDecision.AUTO_APPROVE else None,
                assigned_reviewer_type="radiologist" if decision != TriageDecision.AUTO_APPROVE else "ai_system",
                reasoning="Test reasoning"
            )

            compliance_logger.log_prediction(
                prediction_id=pred_id,
                patient_id=f"patient-{pred_id}",
                study_id=f"study-{pred_id}",
                image_hash=f"hash-{pred_id}",
                prediction_result=prediction_result,
                triage_decision=triage_decision,
                processing_time_ms=150.0
            )


class TestDatabaseManager:
    """Test suite for DatabaseManager class."""

    def test_database_initialization(self):
        """Test database initialization."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_url = f"sqlite:///{tmp.name}"
            db_manager = DatabaseManager(db_url)

            assert db_manager.engine is not None
            assert db_manager.SessionLocal is not None

    def test_health_check(self, test_database):
        """Test database health check."""
        engine, session_factory = test_database
        db_manager = DatabaseManager("sqlite:///:memory:")
        db_manager.SessionLocal = session_factory

        assert db_manager.health_check() is True

    def test_session_creation(self, test_database):
        """Test database session creation."""
        engine, session_factory = test_database
        db_manager = DatabaseManager("sqlite:///:memory:")
        db_manager.SessionLocal = session_factory

        with db_manager.get_session() as session:
            assert session is not None
            # Test basic query
            result = session.execute("SELECT 1").scalar()
            assert result == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])