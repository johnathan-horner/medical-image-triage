"""
Database models and configuration for medical image triage system.
HIPAA-compliant database schema for audit logging and compliance tracking.
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.sqlite import INTEGER
from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)

Base = declarative_base()


class PredictionAuditLog(Base):
    """
    HIPAA-compliant audit log for medical image predictions.

    Note: No raw medical images are stored. Only hashed references and metadata
    are retained for compliance and audit purposes.
    """
    __tablename__ = "prediction_audit_log"

    # Primary identifiers
    id = Column(INTEGER, primary_key=True, autoincrement=True)
    prediction_id = Column(String(36), unique=True, nullable=False, index=True)  # UUID

    # Patient identification (hashed for HIPAA compliance)
    patient_id = Column(String(128), nullable=True, index=True)  # Hashed patient ID
    study_id = Column(String(128), nullable=True, index=True)    # Study/exam identifier

    # Image metadata (HIPAA compliant - no raw image data)
    image_hash = Column(String(64), nullable=False, index=True)  # SHA256 hash of image
    image_metadata = Column(Text, nullable=True)  # JSON metadata about image (size, format, etc.)

    # Prediction results
    predicted_class = Column(String(50), nullable=False, index=True)
    confidence = Column(Float, nullable=False, index=True)
    confidence_level = Column(String(10), nullable=False)  # high, medium, low
    all_class_scores = Column(Text, nullable=True)  # JSON of all class probabilities

    # Triage decision
    triage_decision = Column(String(20), nullable=False, index=True)
    priority_level = Column(Integer, nullable=False, index=True)
    assigned_reviewer_type = Column(String(50), nullable=False)
    reasoning = Column(Text, nullable=False)
    estimated_review_time = Column(Integer, nullable=True)  # minutes

    # System metadata
    model_version = Column(String(20), nullable=False)
    processing_time_ms = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    # Review tracking
    reviewer_id = Column(String(128), nullable=True)  # Assigned reviewer (hashed)
    review_timestamp = Column(DateTime, nullable=True, index=True)
    review_decision = Column(String(50), nullable=True)  # approved, rejected, modified
    review_notes = Column(Text, nullable=True)
    review_confidence = Column(Float, nullable=True)  # Reviewer's confidence in decision

    # Compliance and audit fields
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_deleted = Column(Boolean, nullable=False, default=False)  # Soft delete for compliance
    retention_date = Column(DateTime, nullable=True)  # Data retention compliance

    # Database indexes for performance
    __table_args__ = (
        Index('idx_prediction_search', 'timestamp', 'predicted_class', 'triage_decision'),
        Index('idx_patient_studies', 'patient_id', 'study_id'),
        Index('idx_review_queue', 'triage_decision', 'priority_level', 'assigned_reviewer_type'),
        Index('idx_confidence_analysis', 'confidence_level', 'predicted_class'),
        Index('idx_audit_compliance', 'timestamp', 'is_deleted'),
    )


class ModelPerformanceLog(Base):
    """Track model performance metrics over time for drift detection."""
    __tablename__ = "model_performance_log"

    id = Column(INTEGER, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    model_version = Column(String(20), nullable=False, index=True)

    # Performance metrics
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    auc_score = Column(Float, nullable=True)

    # Confidence metrics
    average_confidence = Column(Float, nullable=False)
    high_confidence_rate = Column(Float, nullable=False)
    medium_confidence_rate = Column(Float, nullable=False)
    low_confidence_rate = Column(Float, nullable=False)

    # Volume metrics
    total_predictions = Column(Integer, nullable=False)
    auto_approved = Column(Integer, nullable=False)
    expedited_review = Column(Integer, nullable=False)
    senior_review = Column(Integer, nullable=False)

    # Class distribution (JSON)
    class_distribution = Column(Text, nullable=True)

    # Calculated for time period
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)

    __table_args__ = (
        Index('idx_performance_tracking', 'timestamp', 'model_version'),
        Index('idx_drift_analysis', 'period_start', 'period_end'),
    )


class SystemAuditLog(Base):
    """System-level audit log for compliance and security monitoring."""
    __tablename__ = "system_audit_log"

    id = Column(INTEGER, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    # Event details
    event_type = Column(String(50), nullable=False, index=True)  # login, prediction, export, etc.
    user_id = Column(String(128), nullable=True, index=True)     # Hashed user identifier
    session_id = Column(String(128), nullable=True)
    ip_address = Column(String(45), nullable=True)  # IPv6 support

    # Action details
    resource_accessed = Column(String(200), nullable=True)
    action_performed = Column(String(100), nullable=False)
    outcome = Column(String(20), nullable=False)  # success, failure, unauthorized

    # Context
    details = Column(Text, nullable=True)  # JSON details
    error_message = Column(Text, nullable=True)

    __table_args__ = (
        Index('idx_security_audit', 'event_type', 'outcome', 'timestamp'),
        Index('idx_user_activity', 'user_id', 'timestamp'),
    )


class DatabaseManager:
    """Database manager for medical image triage system."""

    def __init__(self, database_url: str = "sqlite:///./medical_triage.db"):
        self.database_url = database_url
        self.engine = None
        self.SessionLocal = None
        self._initialize_database()

    def _initialize_database(self) -> None:
        """Initialize database connection and create tables."""
        try:
            # Create engine with appropriate settings
            if "sqlite" in self.database_url:
                # SQLite-specific settings
                self.engine = create_engine(
                    self.database_url,
                    echo=False,  # Set to True for SQL debugging
                    pool_pre_ping=True,
                    connect_args={"check_same_thread": False}
                )
            else:
                # PostgreSQL or other databases
                self.engine = create_engine(
                    self.database_url,
                    echo=False,
                    pool_pre_ping=True
                )

            # Create session factory
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )

            # Create all tables
            Base.metadata.create_all(bind=self.engine)

            logger.info("Database initialized successfully")

        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    def get_session(self) -> Session:
        """Get database session."""
        return self.SessionLocal()

    def get_session_factory(self):
        """Get session factory for dependency injection."""
        return self.SessionLocal

    def health_check(self) -> bool:
        """Check database connectivity."""
        try:
            with self.get_session() as session:
                session.execute("SELECT 1")
                return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

    def backup_database(self, backup_path: str) -> bool:
        """
        Create database backup (SQLite only).
        For production, use proper database backup tools.
        """
        if "sqlite" not in self.database_url:
            logger.warning("Backup only implemented for SQLite")
            return False

        try:
            import shutil
            import sqlite3

            # Extract database file path from URL
            db_path = self.database_url.replace("sqlite:///", "")

            # Create backup
            shutil.copy2(db_path, backup_path)
            logger.info(f"Database backup created: {backup_path}")
            return True

        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            return False

    def cleanup_old_records(self, days_to_keep: int = 90) -> int:
        """
        Clean up old audit records for compliance.
        Returns number of records cleaned up.
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)

            with self.get_session() as session:
                # Soft delete old prediction logs
                updated_count = session.query(PredictionAuditLog).filter(
                    PredictionAuditLog.timestamp < cutoff_date,
                    PredictionAuditLog.is_deleted == False
                ).update({
                    PredictionAuditLog.is_deleted: True,
                    PredictionAuditLog.updated_at: datetime.utcnow()
                })

                session.commit()
                logger.info(f"Cleaned up {updated_count} old prediction records")
                return updated_count

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return 0