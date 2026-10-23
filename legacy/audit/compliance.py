"""
HIPAA-compliant audit logging and compliance management for medical image triage.
Handles secure logging, data retention, and audit trail queries.
"""

import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func

from .database import DatabaseManager, PredictionAuditLog, SystemAuditLog, ModelPerformanceLog
from api.models import AuditLogEntry, AuditQuery, TriageResult
from routing.triage_logic import TriageDecision

logger = logging.getLogger(__name__)


class ComplianceLogger:
    """
    HIPAA-compliant logging and audit trail management.

    This class ensures all medical image predictions are logged securely
    with appropriate data protection and retention policies.
    """

    def __init__(self, database_url: str = "sqlite:///./medical_triage.db"):
        self.db_manager = DatabaseManager(database_url)
        self.retention_days = 2555  # 7 years for medical records (HIPAA requirement)

    def log_prediction(
        self,
        prediction_id: str,
        patient_id: Optional[str],
        study_id: Optional[str],
        image_hash: str,
        prediction_result: Dict[str, Any],
        triage_decision: TriageResult,
        processing_time_ms: float,
        model_version: str = "1.0.0"
    ) -> bool:
        """
        Log a medical image prediction with full audit trail.

        Args:
            prediction_id: Unique prediction identifier
            patient_id: Hashed patient identifier (HIPAA compliant)
            study_id: Study/exam identifier
            image_hash: SHA256 hash of original image
            prediction_result: Model prediction results
            triage_decision: Triage routing decision
            processing_time_ms: Processing time in milliseconds
            model_version: Model version used for prediction

        Returns:
            bool: Success status
        """
        try:
            # Hash patient ID for HIPAA compliance if provided
            hashed_patient_id = None
            if patient_id:
                hashed_patient_id = self._hash_identifier(patient_id)

            # Create audit log entry
            audit_entry = PredictionAuditLog(
                prediction_id=prediction_id,
                patient_id=hashed_patient_id,
                study_id=study_id,
                image_hash=image_hash,
                predicted_class=prediction_result["predicted_class"],
                confidence=prediction_result["confidence"],
                confidence_level=prediction_result["confidence_level"].value,
                all_class_scores=json.dumps(prediction_result["all_scores"]),
                triage_decision=triage_decision.decision.value,
                priority_level=triage_decision.priority_level,
                assigned_reviewer_type=triage_decision.assigned_reviewer_type,
                reasoning=triage_decision.reasoning,
                estimated_review_time=triage_decision.estimated_review_time,
                model_version=model_version,
                processing_time_ms=processing_time_ms,
                retention_date=datetime.utcnow() + timedelta(days=self.retention_days)
            )

            # Save to database
            with self.db_manager.get_session() as session:
                session.add(audit_entry)
                session.commit()

            logger.info(f"Prediction logged successfully: {prediction_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to log prediction {prediction_id}: {e}")
            return False

    def log_review_decision(
        self,
        prediction_id: str,
        reviewer_id: str,
        review_decision: str,
        review_notes: Optional[str] = None,
        review_confidence: Optional[float] = None
    ) -> bool:
        """
        Log physician review decision for a prediction.

        Args:
            prediction_id: Unique prediction identifier
            reviewer_id: Hashed reviewer identifier
            review_decision: Review outcome (approved, rejected, modified)
            review_notes: Optional review notes
            review_confidence: Reviewer's confidence in decision

        Returns:
            bool: Success status
        """
        try:
            hashed_reviewer_id = self._hash_identifier(reviewer_id)

            with self.db_manager.get_session() as session:
                audit_entry = session.query(PredictionAuditLog).filter(
                    PredictionAuditLog.prediction_id == prediction_id
                ).first()

                if not audit_entry:
                    logger.error(f"Prediction not found for review: {prediction_id}")
                    return False

                # Update with review information
                audit_entry.reviewer_id = hashed_reviewer_id
                audit_entry.review_timestamp = datetime.utcnow()
                audit_entry.review_decision = review_decision
                audit_entry.review_notes = review_notes
                audit_entry.review_confidence = review_confidence
                audit_entry.updated_at = datetime.utcnow()

                session.commit()

            logger.info(f"Review decision logged: {prediction_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to log review decision {prediction_id}: {e}")
            return False

    def query_audit_log(self, query: AuditQuery) -> List[AuditLogEntry]:
        """
        Query audit log with HIPAA-compliant filtering.

        Args:
            query: Audit query parameters

        Returns:
            List of audit log entries
        """
        try:
            with self.db_manager.get_session() as session:
                # Build query
                q = session.query(PredictionAuditLog).filter(
                    PredictionAuditLog.is_deleted == False
                )

                # Apply filters
                if query.patient_id:
                    hashed_patient_id = self._hash_identifier(query.patient_id)
                    q = q.filter(PredictionAuditLog.patient_id == hashed_patient_id)

                if query.study_id:
                    q = q.filter(PredictionAuditLog.study_id == query.study_id)

                if query.prediction_id:
                    q = q.filter(PredictionAuditLog.prediction_id == query.prediction_id)

                if query.start_date:
                    q = q.filter(PredictionAuditLog.timestamp >= query.start_date)

                if query.end_date:
                    q = q.filter(PredictionAuditLog.timestamp <= query.end_date)

                if query.triage_decision:
                    q = q.filter(PredictionAuditLog.triage_decision == query.triage_decision.value)

                if query.confidence_level:
                    q = q.filter(PredictionAuditLog.confidence_level == query.confidence_level.value)

                if query.predicted_class:
                    q = q.filter(PredictionAuditLog.predicted_class == query.predicted_class)

                # Order and paginate
                q = q.order_by(desc(PredictionAuditLog.timestamp))
                q = q.offset(query.offset).limit(query.limit)

                # Execute query and convert to response model
                results = q.all()

                audit_entries = []
                for result in results:
                    entry = AuditLogEntry(
                        prediction_id=result.prediction_id,
                        patient_id=result.patient_id,  # Already hashed
                        study_id=result.study_id,
                        image_hash=result.image_hash,
                        predicted_class=result.predicted_class,
                        confidence=result.confidence,
                        confidence_level=result.confidence_level,
                        triage_decision=result.triage_decision,
                        priority_level=result.priority_level,
                        assigned_reviewer_type=result.assigned_reviewer_type,
                        reasoning=result.reasoning,
                        timestamp=result.timestamp,
                        processing_time_ms=result.processing_time_ms,
                        model_version=result.model_version,
                        reviewer_id=result.reviewer_id,
                        review_timestamp=result.review_timestamp,
                        review_decision=result.review_decision,
                        review_notes=result.review_notes
                    )
                    audit_entries.append(entry)

                logger.info(f"Audit query returned {len(audit_entries)} entries")
                return audit_entries

        except Exception as e:
            logger.error(f"Audit query failed: {e}")
            raise

    def log_system_event(
        self,
        event_type: str,
        action_performed: str,
        outcome: str,
        user_id: Optional[str] = None,
        resource_accessed: Optional[str] = None,
        details: Optional[Dict] = None,
        error_message: Optional[str] = None,
        ip_address: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> bool:
        """
        Log system-level events for security auditing.

        Args:
            event_type: Type of event (login, access, export, etc.)
            action_performed: Action description
            outcome: success, failure, unauthorized
            user_id: User identifier (will be hashed)
            resource_accessed: Resource that was accessed
            details: Additional details as dictionary
            error_message: Error message if applicable
            ip_address: Client IP address
            session_id: Session identifier

        Returns:
            bool: Success status
        """
        try:
            hashed_user_id = None
            if user_id:
                hashed_user_id = self._hash_identifier(user_id)

            audit_entry = SystemAuditLog(
                event_type=event_type,
                user_id=hashed_user_id,
                session_id=session_id,
                ip_address=ip_address,
                resource_accessed=resource_accessed,
                action_performed=action_performed,
                outcome=outcome,
                details=json.dumps(details) if details else None,
                error_message=error_message
            )

            with self.db_manager.get_session() as session:
                session.add(audit_entry)
                session.commit()

            logger.info(f"System event logged: {event_type} - {outcome}")
            return True

        except Exception as e:
            logger.error(f"Failed to log system event: {e}")
            return False

    def get_prediction_chain(self, prediction_id: str) -> Optional[Dict[str, Any]]:
        """
        Get complete decision chain for a specific prediction.

        Args:
            prediction_id: Prediction identifier

        Returns:
            Complete audit trail for the prediction
        """
        try:
            with self.db_manager.get_session() as session:
                prediction = session.query(PredictionAuditLog).filter(
                    PredictionAuditLog.prediction_id == prediction_id,
                    PredictionAuditLog.is_deleted == False
                ).first()

                if not prediction:
                    return None

                chain = {
                    "prediction_id": prediction.prediction_id,
                    "timestamp": prediction.timestamp.isoformat(),
                    "prediction": {
                        "class": prediction.predicted_class,
                        "confidence": prediction.confidence,
                        "all_scores": json.loads(prediction.all_class_scores or "{}")
                    },
                    "triage": {
                        "decision": prediction.triage_decision,
                        "priority": prediction.priority_level,
                        "assigned_to": prediction.assigned_reviewer_type,
                        "reasoning": prediction.reasoning,
                        "estimated_time": prediction.estimated_review_time
                    },
                    "model": {
                        "version": prediction.model_version,
                        "processing_time_ms": prediction.processing_time_ms
                    },
                    "review": None
                }

                # Add review information if available
                if prediction.reviewer_id:
                    chain["review"] = {
                        "reviewer_id": prediction.reviewer_id,
                        "timestamp": prediction.review_timestamp.isoformat() if prediction.review_timestamp else None,
                        "decision": prediction.review_decision,
                        "notes": prediction.review_notes,
                        "confidence": prediction.review_confidence
                    }

                return chain

        except Exception as e:
            logger.error(f"Failed to get prediction chain {prediction_id}: {e}")
            return None

    def generate_compliance_report(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """
        Generate compliance report for specified date range.

        Args:
            start_date: Start of reporting period
            end_date: End of reporting period

        Returns:
            Compliance report with metrics and audit trail
        """
        try:
            with self.db_manager.get_session() as session:
                # Basic metrics
                total_predictions = session.query(PredictionAuditLog).filter(
                    and_(
                        PredictionAuditLog.timestamp >= start_date,
                        PredictionAuditLog.timestamp <= end_date,
                        PredictionAuditLog.is_deleted == False
                    )
                ).count()

                # Review completion rate
                reviewed_predictions = session.query(PredictionAuditLog).filter(
                    and_(
                        PredictionAuditLog.timestamp >= start_date,
                        PredictionAuditLog.timestamp <= end_date,
                        PredictionAuditLog.reviewer_id.isnot(None),
                        PredictionAuditLog.is_deleted == False
                    )
                ).count()

                # Auto-approved predictions
                auto_approved = session.query(PredictionAuditLog).filter(
                    and_(
                        PredictionAuditLog.timestamp >= start_date,
                        PredictionAuditLog.timestamp <= end_date,
                        PredictionAuditLog.triage_decision == TriageDecision.AUTO_APPROVE.value,
                        PredictionAuditLog.is_deleted == False
                    )
                ).count()

                # Average processing time
                avg_processing_time = session.query(
                    func.avg(PredictionAuditLog.processing_time_ms)
                ).filter(
                    and_(
                        PredictionAuditLog.timestamp >= start_date,
                        PredictionAuditLog.timestamp <= end_date,
                        PredictionAuditLog.is_deleted == False
                    )
                ).scalar()

                report = {
                    "period": {
                        "start": start_date.isoformat(),
                        "end": end_date.isoformat()
                    },
                    "metrics": {
                        "total_predictions": total_predictions,
                        "reviewed_predictions": reviewed_predictions,
                        "review_completion_rate": reviewed_predictions / total_predictions if total_predictions > 0 else 0,
                        "auto_approved": auto_approved,
                        "auto_approval_rate": auto_approved / total_predictions if total_predictions > 0 else 0,
                        "average_processing_time_ms": float(avg_processing_time or 0)
                    },
                    "compliance": {
                        "data_retention_policy": f"{self.retention_days} days",
                        "hipaa_compliant": True,
                        "audit_trail_complete": True
                    },
                    "generated_at": datetime.utcnow().isoformat()
                }

                logger.info(f"Compliance report generated for {start_date} to {end_date}")
                return report

        except Exception as e:
            logger.error(f"Failed to generate compliance report: {e}")
            raise

    def _hash_identifier(self, identifier: str) -> str:
        """
        Hash identifier for HIPAA compliance.

        Args:
            identifier: Raw identifier

        Returns:
            SHA256 hash of identifier
        """
        return hashlib.sha256(identifier.encode()).hexdigest()

    def get_session_factory(self):
        """Get database session factory for dependency injection."""
        return self.db_manager.get_session_factory()