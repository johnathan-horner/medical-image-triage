"""
FastAPI main application for medical image triage system.
Production-grade API with image upload, prediction, and triage routing.
"""

import os
import time
import hashlib
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
import logging

import numpy as np
import tensorflow as tf
from PIL import Image
import io

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
import uvicorn

from .models import (
    PredictionRequest, PredictionResponse, ClassificationResult, TriageResult,
    AuditLogEntry, AuditQuery, DashboardMetrics, ModelDriftMetrics,
    HealthCheckResponse, ErrorResponse, TriageDecision, PredictionConfidence
)
from routing.triage_logic import TriageRouter
from audit.compliance import ComplianceLogger
from dashboard.metrics import MetricsCalculator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# Initialize FastAPI app
app = FastAPI(
    title="Medical Image Triage System",
    description="Production-grade AI system for medical image classification and triage routing",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080"],  # Add your frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and services
model = None
triage_router = None
compliance_logger = None
metrics_calculator = None
model_metadata = None
startup_time = time.time()


class ModelService:
    """Service for handling model predictions."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.class_names = ["Normal", "Pneumonia", "Pneumothorax", "Infiltration", "Mass"]
        self.load_model()

    def load_model(self) -> None:
        """Load the trained model."""
        try:
            self.model = tf.saved_model.load(self.model_path)
            logger.info(f"Model loaded successfully from {self.model_path}")

            # Try to load metadata
            metadata_path = Path(self.model_path) / "metadata.json"
            if metadata_path.exists():
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.class_names = metadata.get("class_names", self.class_names)
                    logger.info(f"Model metadata loaded: {len(self.class_names)} classes")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise HTTPException(status_code=500, detail=f"Model loading failed: {e}")

    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """Preprocess uploaded image for model inference."""
        try:
            # Open image
            image = Image.open(io.BytesIO(image_bytes))

            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Resize to model input size
            image = image.resize((224, 224))

            # Convert to numpy array
            image_array = np.array(image)

            # Normalize pixel values
            image_array = image_array.astype(np.float32) / 255.0

            # Add batch dimension
            image_array = np.expand_dims(image_array, axis=0)

            return image_array

        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

    def predict(self, image_bytes: bytes) -> Dict[str, Any]:
        """Make prediction on uploaded image."""
        if self.model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")

        # Preprocess image
        image_array = self.preprocess_image(image_bytes)

        # Make prediction
        start_time = time.time()
        try:
            # Get the serving function
            infer = self.model.signatures["serving_default"]

            # Convert to tensor
            input_tensor = tf.convert_to_tensor(image_array)

            # Make prediction
            predictions = infer(input_tensor)

            # Extract predictions (adjust key based on your model)
            prediction_key = list(predictions.keys())[0]
            pred_array = predictions[prediction_key].numpy()[0]

        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

        processing_time = (time.time() - start_time) * 1000

        # Process predictions
        predicted_class_idx = np.argmax(pred_array)
        predicted_class = self.class_names[predicted_class_idx]
        confidence = float(pred_array[predicted_class_idx])

        # Create confidence scores dictionary
        all_scores = {
            class_name: float(pred_array[i])
            for i, class_name in enumerate(self.class_names)
        }

        # Determine confidence level
        if confidence >= 0.9:
            confidence_level = PredictionConfidence.HIGH
        elif confidence >= 0.7:
            confidence_level = PredictionConfidence.MEDIUM
        else:
            confidence_level = PredictionConfidence.LOW

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "all_scores": all_scores,
            "processing_time_ms": processing_time
        }


def get_image_hash(image_bytes: bytes) -> str:
    """Generate hash for image (HIPAA compliance - no raw image storage)."""
    return hashlib.sha256(image_bytes).hexdigest()


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global model, triage_router, compliance_logger, metrics_calculator, model_metadata

    try:
        # Initialize model service
        model_path = os.getenv("MODEL_PATH", "models/medical_model/saved_model")
        if not Path(model_path).exists():
            logger.warning(f"Model path {model_path} does not exist. Some endpoints will be unavailable.")
            model = None
        else:
            model = ModelService(model_path)

        # Initialize triage router
        triage_router = TriageRouter()

        # Initialize compliance logger
        database_url = os.getenv("DATABASE_URL", "sqlite:///./medical_triage.db")
        compliance_logger = ComplianceLogger(database_url)

        # Initialize metrics calculator
        metrics_calculator = MetricsCalculator(compliance_logger.get_session_factory())

        logger.info("Medical Triage System started successfully")

    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    uptime = time.time() - startup_time

    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        model_loaded=model is not None,
        database_connected=compliance_logger is not None,
        version="1.0.0",
        uptime_seconds=uptime
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    patient_id: Optional[str] = None,
    study_id: Optional[str] = None,
):
    """
    Upload medical image and get classification with triage routing.

    **HIPAA Compliance**: Raw images are not stored. Only hashed references and metadata are retained.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model service unavailable")

    start_time = time.time()

    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read image bytes
        image_bytes = await file.read()

        # Generate image hash for audit trail (HIPAA compliant)
        image_hash = get_image_hash(image_bytes)

        # Make prediction
        prediction_result = model.predict(image_bytes)

        # Generate unique prediction ID
        prediction_id = str(uuid.uuid4())

        # Get triage decision
        triage_decision = triage_router.route_prediction(
            prediction_result["confidence"],
            prediction_result["predicted_class"]
        )

        # Create response
        classification = ClassificationResult(
            predicted_class=prediction_result["predicted_class"],
            confidence=prediction_result["confidence"],
            confidence_level=prediction_result["confidence_level"],
            all_scores=prediction_result["all_scores"]
        )

        response = PredictionResponse(
            prediction_id=prediction_id,
            classification=classification,
            triage=triage_decision,
            timestamp=datetime.utcnow(),
            processing_time_ms=(time.time() - start_time) * 1000,
            model_version="1.0.0"
        )

        # Log to audit trail (background task for performance)
        background_tasks.add_task(
            compliance_logger.log_prediction,
            prediction_id=prediction_id,
            patient_id=patient_id,
            study_id=study_id,
            image_hash=image_hash,
            prediction_result=prediction_result,
            triage_decision=triage_decision,
            processing_time_ms=response.processing_time_ms
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/audit/predictions", response_model=List[AuditLogEntry])
async def get_audit_log(
    patient_id: Optional[str] = Query(None),
    study_id: Optional[str] = Query(None),
    prediction_id: Optional[str] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    limit: int = Query(100, le=1000),
    offset: int = Query(0, ge=0)
):
    """
    Query audit trail for compliance and review.

    **HIPAA Compliance**: Only metadata and hashed references are returned.
    """
    if compliance_logger is None:
        raise HTTPException(status_code=503, detail="Audit service unavailable")

    try:
        query = AuditQuery(
            patient_id=patient_id,
            study_id=study_id,
            prediction_id=prediction_id,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset
        )

        return compliance_logger.query_audit_log(query)

    except Exception as e:
        logger.error(f"Audit query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Audit query failed: {str(e)}")


@app.get("/dashboard/metrics", response_model=DashboardMetrics)
async def get_dashboard_metrics():
    """Get dashboard metrics for monitoring and analysis."""
    if metrics_calculator is None:
        raise HTTPException(status_code=503, detail="Metrics service unavailable")

    try:
        return metrics_calculator.calculate_dashboard_metrics()
    except Exception as e:
        logger.error(f"Metrics calculation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics calculation failed: {str(e)}")


@app.get("/dashboard/drift", response_model=ModelDriftMetrics)
async def get_drift_metrics(
    days: int = Query(7, ge=1, le=90, description="Number of days to analyze")
):
    """Get model drift metrics for the specified period."""
    if metrics_calculator is None:
        raise HTTPException(status_code=503, detail="Metrics service unavailable")

    try:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        return metrics_calculator.calculate_drift_metrics(start_date, end_date)
    except Exception as e:
        logger.error(f"Drift analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Drift analysis failed: {str(e)}")


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=str(exc),
            request_id=str(uuid.uuid4())
        ).dict()
    )


@app.exception_handler(500)
async def internal_exception_handler(request, exc):
    """Handle internal server errors."""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail="An unexpected error occurred",
            request_id=str(uuid.uuid4())
        ).dict()
    )


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )