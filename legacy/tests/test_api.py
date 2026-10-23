"""
Unit tests for FastAPI endpoints and API functionality.
Tests prediction endpoints, audit endpoints, and error handling.
"""

import pytest
import tempfile
import io
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import numpy as np

from fastapi.testclient import TestClient
from api.main import app
from api.models import PredictionResponse, DashboardMetrics, TriageDecision


@pytest.fixture
def client():
    """Create FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def mock_model_service():
    """Mock ModelService for testing."""
    mock_service = Mock()
    mock_service.predict.return_value = {
        "predicted_class": "Normal",
        "confidence": 0.95,
        "confidence_level": "high",
        "all_scores": {
            "Normal": 0.95,
            "Pneumonia": 0.02,
            "Pneumothorax": 0.01,
            "Infiltration": 0.01,
            "Mass": 0.01
        },
        "processing_time_ms": 150.5
    }
    return mock_service


@pytest.fixture
def mock_triage_router():
    """Mock TriageRouter for testing."""
    from api.models import TriageResult, TriageDecision

    mock_router = Mock()
    mock_router.route_prediction.return_value = TriageResult(
        decision=TriageDecision.AUTO_APPROVE,
        priority_level=3,
        estimated_review_time=None,
        assigned_reviewer_type="ai_system",
        reasoning="High confidence allows automatic approval. Condition: No acute findings"
    )
    return mock_router


@pytest.fixture
def test_image():
    """Create a test image for upload."""
    # Create a simple test image
    img = Image.new('RGB', (224, 224), color='gray')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes


class TestHealthEndpoint:
    """Test suite for health check endpoint."""

    def test_health_check_success(self, client):
        """Test successful health check."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "uptime_seconds" in data
        assert "version" in data

    def test_health_check_response_structure(self, client):
        """Test health check response structure."""
        response = client.get("/health")
        data = response.json()

        required_fields = [
            "status", "timestamp", "model_loaded",
            "database_connected", "version", "uptime_seconds"
        ]

        for field in required_fields:
            assert field in data


class TestPredictionEndpoint:
    """Test suite for image prediction endpoint."""

    @patch('api.main.model')
    @patch('api.main.triage_router')
    @patch('api.main.compliance_logger')
    def test_prediction_success(self, mock_logger, mock_router, mock_model, client, test_image):
        """Test successful image prediction."""
        # Setup mocks
        mock_model.predict.return_value = {
            "predicted_class": "Normal",
            "confidence": 0.95,
            "confidence_level": "high",
            "all_scores": {"Normal": 0.95, "Pneumonia": 0.05},
            "processing_time_ms": 150.0
        }

        from api.models import TriageResult, TriageDecision
        mock_router.route_prediction.return_value = TriageResult(
            decision=TriageDecision.AUTO_APPROVE,
            priority_level=3,
            estimated_review_time=None,
            assigned_reviewer_type="ai_system",
            reasoning="Test reasoning"
        )

        mock_logger.log_prediction = Mock(return_value=True)

        # Make request
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", test_image, "image/jpeg")},
            data={"patient_id": "test_patient", "study_id": "test_study"}
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "prediction_id" in data
        assert "classification" in data
        assert "triage" in data
        assert "timestamp" in data

        # Verify classification data
        classification = data["classification"]
        assert classification["predicted_class"] == "Normal"
        assert classification["confidence"] == 0.95

        # Verify triage data
        triage = data["triage"]
        assert triage["decision"] == "auto_approve"

    def test_prediction_invalid_file_type(self, client):
        """Test prediction with invalid file type."""
        text_file = io.BytesIO(b"This is not an image")

        response = client.post(
            "/predict",
            files={"file": ("test.txt", text_file, "text/plain")}
        )

        assert response.status_code == 400
        assert "must be an image" in response.json()["detail"]

    @patch('api.main.model', None)
    def test_prediction_model_unavailable(self, client, test_image):
        """Test prediction when model is unavailable."""
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", test_image, "image/jpeg")}
        )

        assert response.status_code == 503
        assert "Model service unavailable" in response.json()["detail"]

    @patch('api.main.model')
    def test_prediction_model_error(self, mock_model, client, test_image):
        """Test prediction when model raises an error."""
        mock_model.predict.side_effect = Exception("Model prediction failed")

        response = client.post(
            "/predict",
            files={"file": ("test.jpg", test_image, "image/jpeg")}
        )

        assert response.status_code == 500
        assert "Prediction failed" in response.json()["detail"]


class TestAuditEndpoint:
    """Test suite for audit trail endpoints."""

    @patch('api.main.compliance_logger')
    def test_audit_log_query(self, mock_logger, client):
        """Test audit log query endpoint."""
        # Mock audit log data
        from api.models import AuditLogEntry
        mock_entries = [
            AuditLogEntry(
                prediction_id="test-123",
                patient_id="hashed_patient",
                study_id="study_123",
                image_hash="image_hash_123",
                predicted_class="Normal",
                confidence=0.95,
                confidence_level="high",
                triage_decision="auto_approve",
                priority_level=3,
                assigned_reviewer_type="ai_system",
                reasoning="Test reasoning",
                timestamp=datetime.utcnow(),
                processing_time_ms=150.0,
                model_version="1.0.0"
            )
        ]

        mock_logger.query_audit_log.return_value = mock_entries

        response = client.get("/audit/predictions")

        assert response.status_code == 200
        data = response.json()

        assert len(data) == 1
        assert data[0]["prediction_id"] == "test-123"
        assert data[0]["predicted_class"] == "Normal"

    @patch('api.main.compliance_logger')
    def test_audit_log_query_with_filters(self, mock_logger, client):
        """Test audit log query with filters."""
        mock_logger.query_audit_log.return_value = []

        response = client.get(
            "/audit/predictions",
            params={
                "patient_id": "test_patient",
                "start_date": "2024-01-01T00:00:00Z",
                "end_date": "2024-01-02T00:00:00Z",
                "limit": 50
            }
        )

        assert response.status_code == 200
        # Verify that query_audit_log was called with correct parameters
        mock_logger.query_audit_log.assert_called_once()

    @patch('api.main.compliance_logger', None)
    def test_audit_log_service_unavailable(self, client):
        """Test audit log when service is unavailable."""
        response = client.get("/audit/predictions")

        assert response.status_code == 503
        assert "Audit service unavailable" in response.json()["detail"]


class TestDashboardEndpoints:
    """Test suite for dashboard metrics endpoints."""

    @patch('api.main.metrics_calculator')
    def test_dashboard_metrics(self, mock_calculator, client):
        """Test dashboard metrics endpoint."""
        # Mock metrics data
        mock_metrics = DashboardMetrics(
            total_predictions=1000,
            predictions_today=50,
            accuracy_rate=0.92,
            average_confidence=0.87,
            confidence_distribution={"high": 400, "medium": 450, "low": 150},
            classification_distribution={"Normal": 600, "Pneumonia": 250, "Mass": 150},
            triage_distribution={"auto_approve": 400, "expedited_review": 450, "senior_review": 150},
            average_processing_time_ms=145.5,
            pending_reviews=25,
            completed_reviews=575,
            average_review_time_hours=2.5
        )

        mock_calculator.calculate_dashboard_metrics.return_value = mock_metrics

        response = client.get("/dashboard/metrics")

        assert response.status_code == 200
        data = response.json()

        assert data["total_predictions"] == 1000
        assert data["accuracy_rate"] == 0.92
        assert "confidence_distribution" in data
        assert "classification_distribution" in data

    @patch('api.main.metrics_calculator')
    def test_drift_metrics(self, mock_calculator, client):
        """Test model drift metrics endpoint."""
        from api.models import ModelDriftMetrics

        mock_drift = ModelDriftMetrics(
            period_start=datetime.utcnow() - timedelta(days=7),
            period_end=datetime.utcnow(),
            average_confidence_trend=[
                {"date": "2024-01-01", "average_confidence": 0.85, "prediction_count": 100}
            ],
            classification_distribution_change={"Normal": 0.05, "Pneumonia": -0.02},
            confidence_decline_alerts=["Sustained confidence decline detected"],
            recommendations=["Consider retraining model with recent data"]
        )

        mock_calculator.calculate_drift_metrics.return_value = mock_drift

        response = client.get("/dashboard/drift", params={"days": 7})

        assert response.status_code == 200
        data = response.json()

        assert "average_confidence_trend" in data
        assert "confidence_decline_alerts" in data
        assert "recommendations" in data

    @patch('api.main.metrics_calculator', None)
    def test_metrics_service_unavailable(self, client):
        """Test dashboard metrics when service is unavailable."""
        response = client.get("/dashboard/metrics")

        assert response.status_code == 503
        assert "Metrics service unavailable" in response.json()["detail"]


class TestErrorHandling:
    """Test suite for error handling and edge cases."""

    def test_invalid_endpoint(self, client):
        """Test request to invalid endpoint."""
        response = client.get("/invalid/endpoint")
        assert response.status_code == 404

    def test_malformed_request_body(self, client):
        """Test handling of malformed request bodies."""
        response = client.post(
            "/predict",
            data="invalid data",
            headers={"Content-Type": "application/json"}
        )

        # Should handle gracefully
        assert response.status_code in [400, 422]

    def test_large_file_upload(self, client):
        """Test handling of oversized file uploads."""
        # Create a large dummy file
        large_data = b"x" * (10 * 1024 * 1024)  # 10MB
        large_file = io.BytesIO(large_data)

        response = client.post(
            "/predict",
            files={"file": ("large.jpg", large_file, "image/jpeg")}
        )

        # Should handle gracefully (exact response depends on server config)
        assert response.status_code in [400, 413, 422, 500]

    def test_concurrent_requests(self, client):
        """Test handling of multiple concurrent requests."""
        import concurrent.futures
        import threading

        def make_request():
            return client.get("/health")

        # Make multiple concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            responses = [f.result() for f in futures]

        # All requests should succeed
        for response in responses:
            assert response.status_code == 200


class TestImageProcessing:
    """Test suite for image processing functionality."""

    def test_various_image_formats(self, client):
        """Test prediction with different image formats."""
        formats = [
            ("test.jpg", "image/jpeg", "JPEG"),
            ("test.png", "image/png", "PNG"),
            ("test.bmp", "image/bmp", "BMP"),
        ]

        for filename, content_type, image_format in formats:
            # Create test image in specific format
            img = Image.new('RGB', (224, 224), color='gray')
            img_bytes = io.BytesIO()
            img.save(img_bytes, format=image_format)
            img_bytes.seek(0)

            with patch('api.main.model') as mock_model, \
                 patch('api.main.triage_router') as mock_router, \
                 patch('api.main.compliance_logger') as mock_logger:

                # Setup mocks
                mock_model.predict.return_value = {
                    "predicted_class": "Normal",
                    "confidence": 0.95,
                    "confidence_level": "high",
                    "all_scores": {"Normal": 0.95},
                    "processing_time_ms": 150.0
                }

                from api.models import TriageResult, TriageDecision
                mock_router.route_prediction.return_value = TriageResult(
                    decision=TriageDecision.AUTO_APPROVE,
                    priority_level=3,
                    estimated_review_time=None,
                    assigned_reviewer_type="ai_system",
                    reasoning="Test"
                )

                mock_logger.log_prediction = Mock(return_value=True)

                response = client.post(
                    "/predict",
                    files={"file": (filename, img_bytes, content_type)}
                )

                assert response.status_code == 200, f"Failed for format {image_format}"

    def test_grayscale_image_conversion(self, client):
        """Test conversion of grayscale images to RGB."""
        # Create grayscale image
        img = Image.new('L', (224, 224), color=128)  # Grayscale
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)

        with patch('api.main.model') as mock_model, \
             patch('api.main.triage_router') as mock_router, \
             patch('api.main.compliance_logger') as mock_logger:

            # Setup mocks
            mock_model.predict.return_value = {
                "predicted_class": "Normal",
                "confidence": 0.95,
                "confidence_level": "high",
                "all_scores": {"Normal": 0.95},
                "processing_time_ms": 150.0
            }

            from api.models import TriageResult, TriageDecision
            mock_router.route_prediction.return_value = TriageResult(
                decision=TriageDecision.AUTO_APPROVE,
                priority_level=3,
                estimated_review_time=None,
                assigned_reviewer_type="ai_system",
                reasoning="Test"
            )

            mock_logger.log_prediction = Mock(return_value=True)

            response = client.post(
                "/predict",
                files={"file": ("gray.jpg", img_bytes, "image/jpeg")}
            )

            assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])