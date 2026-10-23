"""
Pydantic models for API request/response schemas.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class TriageDecision(str, Enum):
    """Triage routing decisions."""
    AUTO_APPROVE = "auto_approve"
    EXPEDITED_REVIEW = "expedited_review"
    SENIOR_REVIEW = "senior_review"


class PredictionConfidence(str, Enum):
    """Confidence level categories."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class PredictionRequest(BaseModel):
    """Request model for image prediction."""
    patient_id: Optional[str] = Field(None, description="Patient identifier (hashed)")
    study_id: Optional[str] = Field(None, description="Study identifier")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ClassificationResult(BaseModel):
    """Model prediction result."""
    predicted_class: str = Field(..., description="Predicted medical condition")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    confidence_level: PredictionConfidence = Field(..., description="Confidence category")
    all_scores: Dict[str, float] = Field(..., description="All class probabilities")


class TriageResult(BaseModel):
    """Triage routing decision."""
    decision: TriageDecision = Field(..., description="Triage routing decision")
    priority_level: int = Field(..., ge=1, le=3, description="Priority level (1=highest)")
    estimated_review_time: Optional[int] = Field(None, description="Est. review time in minutes")
    assigned_reviewer_type: str = Field(..., description="Type of reviewer assigned")
    reasoning: str = Field(..., description="Reasoning for triage decision")


class PredictionResponse(BaseModel):
    """Complete prediction and triage response."""
    prediction_id: str = Field(..., description="Unique prediction identifier")
    classification: ClassificationResult
    triage: TriageResult
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    model_version: str = Field(..., description="Model version used")


class AuditLogEntry(BaseModel):
    """Audit log entry model."""
    prediction_id: str
    patient_id: Optional[str]
    study_id: Optional[str]
    image_hash: str
    predicted_class: str
    confidence: float
    confidence_level: str
    triage_decision: str
    priority_level: int
    assigned_reviewer_type: str
    reasoning: str
    timestamp: datetime
    processing_time_ms: float
    model_version: str
    reviewer_id: Optional[str] = None
    review_timestamp: Optional[datetime] = None
    review_decision: Optional[str] = None
    review_notes: Optional[str] = None


class AuditQuery(BaseModel):
    """Query parameters for audit log."""
    patient_id: Optional[str] = None
    study_id: Optional[str] = None
    prediction_id: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    triage_decision: Optional[TriageDecision] = None
    confidence_level: Optional[PredictionConfidence] = None
    predicted_class: Optional[str] = None
    limit: int = Field(default=100, le=1000)
    offset: int = Field(default=0, ge=0)


class DashboardMetrics(BaseModel):
    """Dashboard metrics response."""
    total_predictions: int
    predictions_today: int
    accuracy_rate: float
    average_confidence: float
    confidence_distribution: Dict[str, int]
    classification_distribution: Dict[str, int]
    triage_distribution: Dict[str, int]
    average_processing_time_ms: float
    pending_reviews: int
    completed_reviews: int
    average_review_time_hours: Optional[float]


class ModelDriftMetrics(BaseModel):
    """Model drift monitoring metrics."""
    period_start: datetime
    period_end: datetime
    average_confidence_trend: List[Dict[str, Any]]
    classification_distribution_change: Dict[str, float]
    confidence_decline_alerts: List[str]
    recommendations: List[str]


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    model_loaded: bool
    database_connected: bool
    version: str
    uptime_seconds: float


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None