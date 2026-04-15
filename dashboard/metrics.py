"""
Dashboard metrics calculation for medical image triage system.
Provides real-time analytics, model drift detection, and performance monitoring.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy import func, and_, desc, case
import json
import numpy as np

from audit.database import PredictionAuditLog, ModelPerformanceLog
from api.models import DashboardMetrics, ModelDriftMetrics
from routing.triage_logic import TriageDecision, PredictionConfidence

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Advanced metrics calculator for medical image triage system.

    Provides real-time dashboard metrics, model drift detection,
    and comprehensive performance analytics.
    """

    def __init__(self, session_factory: sessionmaker):
        self.SessionLocal = session_factory

    def calculate_dashboard_metrics(self, days_lookback: int = 30) -> DashboardMetrics:
        """
        Calculate comprehensive dashboard metrics.

        Args:
            days_lookback: Number of days to include in metrics

        Returns:
            DashboardMetrics object with all calculated metrics
        """
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days_lookback)
            today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

            with self.SessionLocal() as session:
                # Total predictions in period
                total_predictions = self._get_prediction_count(
                    session, start_date, end_date
                )

                # Predictions today
                predictions_today = self._get_prediction_count(
                    session, today_start, end_date
                )

                # Accuracy rate (from reviewed predictions)
                accuracy_rate = self._calculate_accuracy_rate(
                    session, start_date, end_date
                )

                # Average confidence
                average_confidence = self._calculate_average_confidence(
                    session, start_date, end_date
                )

                # Confidence distribution
                confidence_distribution = self._get_confidence_distribution(
                    session, start_date, end_date
                )

                # Classification distribution
                classification_distribution = self._get_classification_distribution(
                    session, start_date, end_date
                )

                # Triage distribution
                triage_distribution = self._get_triage_distribution(
                    session, start_date, end_date
                )

                # Average processing time
                avg_processing_time = self._calculate_average_processing_time(
                    session, start_date, end_date
                )

                # Review metrics
                pending_reviews, completed_reviews, avg_review_time = self._get_review_metrics(
                    session, start_date, end_date
                )

                return DashboardMetrics(
                    total_predictions=total_predictions,
                    predictions_today=predictions_today,
                    accuracy_rate=accuracy_rate,
                    average_confidence=average_confidence,
                    confidence_distribution=confidence_distribution,
                    classification_distribution=classification_distribution,
                    triage_distribution=triage_distribution,
                    average_processing_time_ms=avg_processing_time,
                    pending_reviews=pending_reviews,
                    completed_reviews=completed_reviews,
                    average_review_time_hours=avg_review_time
                )

        except Exception as e:
            logger.error(f"Failed to calculate dashboard metrics: {e}")
            raise

    def calculate_drift_metrics(
        self,
        start_date: datetime,
        end_date: datetime,
        window_days: int = 7
    ) -> ModelDriftMetrics:
        """
        Calculate model drift metrics for monitoring model performance over time.

        Args:
            start_date: Start of analysis period
            end_date: End of analysis period
            window_days: Size of sliding window for trend analysis

        Returns:
            ModelDriftMetrics with drift analysis results
        """
        try:
            with self.SessionLocal() as session:
                # Calculate confidence trends over time
                confidence_trend = self._calculate_confidence_trend(
                    session, start_date, end_date, window_days
                )

                # Calculate distribution changes
                distribution_change = self._calculate_distribution_change(
                    session, start_date, end_date
                )

                # Detect confidence decline alerts
                decline_alerts = self._detect_confidence_decline(confidence_trend)

                # Generate recommendations
                recommendations = self._generate_drift_recommendations(
                    confidence_trend, distribution_change, decline_alerts
                )

                return ModelDriftMetrics(
                    period_start=start_date,
                    period_end=end_date,
                    average_confidence_trend=confidence_trend,
                    classification_distribution_change=distribution_change,
                    confidence_decline_alerts=decline_alerts,
                    recommendations=recommendations
                )

        except Exception as e:
            logger.error(f"Failed to calculate drift metrics: {e}")
            raise

    def _get_prediction_count(
        self,
        session: Session,
        start_date: datetime,
        end_date: datetime
    ) -> int:
        """Get count of predictions in date range."""
        return session.query(PredictionAuditLog).filter(
            and_(
                PredictionAuditLog.timestamp >= start_date,
                PredictionAuditLog.timestamp <= end_date,
                PredictionAuditLog.is_deleted == False
            )
        ).count()

    def _calculate_accuracy_rate(
        self,
        session: Session,
        start_date: datetime,
        end_date: datetime
    ) -> float:
        """
        Calculate accuracy rate from reviewed predictions.
        This is approximate since we're using review decisions as ground truth.
        """
        try:
            # Get predictions that have been reviewed
            reviewed_predictions = session.query(PredictionAuditLog).filter(
                and_(
                    PredictionAuditLog.timestamp >= start_date,
                    PredictionAuditLog.timestamp <= end_date,
                    PredictionAuditLog.reviewer_id.isnot(None),
                    PredictionAuditLog.review_decision.isnot(None),
                    PredictionAuditLog.is_deleted == False
                )
            ).all()

            if not reviewed_predictions:
                return 0.0

            # Count correct predictions (approved by reviewer)
            correct_predictions = sum(
                1 for pred in reviewed_predictions
                if pred.review_decision in ['approved', 'confirmed']
            )

            return correct_predictions / len(reviewed_predictions)

        except Exception as e:
            logger.error(f"Failed to calculate accuracy rate: {e}")
            return 0.0

    def _calculate_average_confidence(
        self,
        session: Session,
        start_date: datetime,
        end_date: datetime
    ) -> float:
        """Calculate average confidence score."""
        try:
            avg_confidence = session.query(
                func.avg(PredictionAuditLog.confidence)
            ).filter(
                and_(
                    PredictionAuditLog.timestamp >= start_date,
                    PredictionAuditLog.timestamp <= end_date,
                    PredictionAuditLog.is_deleted == False
                )
            ).scalar()

            return float(avg_confidence or 0.0)

        except Exception as e:
            logger.error(f"Failed to calculate average confidence: {e}")
            return 0.0

    def _get_confidence_distribution(
        self,
        session: Session,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, int]:
        """Get distribution of confidence levels."""
        try:
            distribution = session.query(
                PredictionAuditLog.confidence_level,
                func.count(PredictionAuditLog.id)
            ).filter(
                and_(
                    PredictionAuditLog.timestamp >= start_date,
                    PredictionAuditLog.timestamp <= end_date,
                    PredictionAuditLog.is_deleted == False
                )
            ).group_by(PredictionAuditLog.confidence_level).all()

            return {level: count for level, count in distribution}

        except Exception as e:
            logger.error(f"Failed to get confidence distribution: {e}")
            return {}

    def _get_classification_distribution(
        self,
        session: Session,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, int]:
        """Get distribution of predicted classes."""
        try:
            distribution = session.query(
                PredictionAuditLog.predicted_class,
                func.count(PredictionAuditLog.id)
            ).filter(
                and_(
                    PredictionAuditLog.timestamp >= start_date,
                    PredictionAuditLog.timestamp <= end_date,
                    PredictionAuditLog.is_deleted == False
                )
            ).group_by(PredictionAuditLog.predicted_class).all()

            return {class_name: count for class_name, count in distribution}

        except Exception as e:
            logger.error(f"Failed to get classification distribution: {e}")
            return {}

    def _get_triage_distribution(
        self,
        session: Session,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, int]:
        """Get distribution of triage decisions."""
        try:
            distribution = session.query(
                PredictionAuditLog.triage_decision,
                func.count(PredictionAuditLog.id)
            ).filter(
                and_(
                    PredictionAuditLog.timestamp >= start_date,
                    PredictionAuditLog.timestamp <= end_date,
                    PredictionAuditLog.is_deleted == False
                )
            ).group_by(PredictionAuditLog.triage_decision).all()

            return {decision: count for decision, count in distribution}

        except Exception as e:
            logger.error(f"Failed to get triage distribution: {e}")
            return {}

    def _calculate_average_processing_time(
        self,
        session: Session,
        start_date: datetime,
        end_date: datetime
    ) -> float:
        """Calculate average processing time in milliseconds."""
        try:
            avg_time = session.query(
                func.avg(PredictionAuditLog.processing_time_ms)
            ).filter(
                and_(
                    PredictionAuditLog.timestamp >= start_date,
                    PredictionAuditLog.timestamp <= end_date,
                    PredictionAuditLog.is_deleted == False
                )
            ).scalar()

            return float(avg_time or 0.0)

        except Exception as e:
            logger.error(f"Failed to calculate average processing time: {e}")
            return 0.0

    def _get_review_metrics(
        self,
        session: Session,
        start_date: datetime,
        end_date: datetime
    ) -> Tuple[int, int, Optional[float]]:
        """Get review queue metrics."""
        try:
            # Pending reviews (not auto-approved and no reviewer assigned)
            pending_reviews = session.query(PredictionAuditLog).filter(
                and_(
                    PredictionAuditLog.timestamp >= start_date,
                    PredictionAuditLog.timestamp <= end_date,
                    PredictionAuditLog.triage_decision != TriageDecision.AUTO_APPROVE.value,
                    PredictionAuditLog.reviewer_id.is_(None),
                    PredictionAuditLog.is_deleted == False
                )
            ).count()

            # Completed reviews
            completed_reviews = session.query(PredictionAuditLog).filter(
                and_(
                    PredictionAuditLog.timestamp >= start_date,
                    PredictionAuditLog.timestamp <= end_date,
                    PredictionAuditLog.reviewer_id.isnot(None),
                    PredictionAuditLog.review_timestamp.isnot(None),
                    PredictionAuditLog.is_deleted == False
                )
            ).count()

            # Average review time
            avg_review_time_hours = None
            if completed_reviews > 0:
                # Calculate average time between prediction and review
                reviewed_predictions = session.query(
                    PredictionAuditLog.timestamp,
                    PredictionAuditLog.review_timestamp
                ).filter(
                    and_(
                        PredictionAuditLog.timestamp >= start_date,
                        PredictionAuditLog.timestamp <= end_date,
                        PredictionAuditLog.reviewer_id.isnot(None),
                        PredictionAuditLog.review_timestamp.isnot(None),
                        PredictionAuditLog.is_deleted == False
                    )
                ).all()

                if reviewed_predictions:
                    total_review_time = sum(
                        (review_time - prediction_time).total_seconds() / 3600
                        for prediction_time, review_time in reviewed_predictions
                        if review_time and prediction_time
                    )
                    avg_review_time_hours = total_review_time / len(reviewed_predictions)

            return pending_reviews, completed_reviews, avg_review_time_hours

        except Exception as e:
            logger.error(f"Failed to get review metrics: {e}")
            return 0, 0, None

    def _calculate_confidence_trend(
        self,
        session: Session,
        start_date: datetime,
        end_date: datetime,
        window_days: int
    ) -> List[Dict[str, Any]]:
        """Calculate confidence trend over time using sliding windows."""
        try:
            trend_data = []
            current_date = start_date

            while current_date < end_date:
                window_end = min(current_date + timedelta(days=window_days), end_date)

                # Calculate metrics for this window
                avg_confidence = session.query(
                    func.avg(PredictionAuditLog.confidence)
                ).filter(
                    and_(
                        PredictionAuditLog.timestamp >= current_date,
                        PredictionAuditLog.timestamp < window_end,
                        PredictionAuditLog.is_deleted == False
                    )
                ).scalar()

                prediction_count = session.query(PredictionAuditLog).filter(
                    and_(
                        PredictionAuditLog.timestamp >= current_date,
                        PredictionAuditLog.timestamp < window_end,
                        PredictionAuditLog.is_deleted == False
                    )
                ).count()

                if avg_confidence is not None:
                    trend_data.append({
                        "date": current_date.isoformat(),
                        "average_confidence": float(avg_confidence),
                        "prediction_count": prediction_count
                    })

                current_date += timedelta(days=window_days)

            return trend_data

        except Exception as e:
            logger.error(f"Failed to calculate confidence trend: {e}")
            return []

    def _calculate_distribution_change(
        self,
        session: Session,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, float]:
        """Calculate change in classification distribution."""
        try:
            period_length = (end_date - start_date).days
            if period_length < 14:  # Need at least 2 weeks for comparison
                return {}

            # Split period in half
            mid_point = start_date + timedelta(days=period_length // 2)

            # Get distribution for first half
            first_half_dist = self._get_classification_distribution(
                session, start_date, mid_point
            )

            # Get distribution for second half
            second_half_dist = self._get_classification_distribution(
                session, mid_point, end_date
            )

            # Calculate percentage change
            changes = {}
            all_classes = set(list(first_half_dist.keys()) + list(second_half_dist.keys()))

            for class_name in all_classes:
                first_count = first_half_dist.get(class_name, 0)
                second_count = second_half_dist.get(class_name, 0)

                # Calculate total predictions in each period
                first_total = sum(first_half_dist.values()) or 1
                second_total = sum(second_half_dist.values()) or 1

                first_rate = first_count / first_total
                second_rate = second_count / second_total

                # Calculate percentage point change
                change = second_rate - first_rate
                changes[class_name] = change

            return changes

        except Exception as e:
            logger.error(f"Failed to calculate distribution change: {e}")
            return {}

    def _detect_confidence_decline(self, confidence_trend: List[Dict[str, Any]]) -> List[str]:
        """Detect significant confidence decline patterns."""
        if len(confidence_trend) < 3:
            return []

        alerts = []
        confidences = [point["average_confidence"] for point in confidence_trend]

        # Check for sustained decline
        if len(confidences) >= 3:
            recent_avg = np.mean(confidences[-3:])
            early_avg = np.mean(confidences[:3])

            if recent_avg < early_avg - 0.1:  # 10% decline
                alerts.append(f"Sustained confidence decline detected: {early_avg:.3f} → {recent_avg:.3f}")

        # Check for sudden drop
        for i in range(1, len(confidences)):
            if confidences[i] < confidences[i-1] - 0.15:  # 15% sudden drop
                alerts.append(f"Sudden confidence drop on {confidence_trend[i]['date']}: {confidences[i-1]:.3f} → {confidences[i]:.3f}")

        # Check if recent confidence is below threshold
        if confidences and confidences[-1] < 0.7:
            alerts.append(f"Current average confidence below threshold: {confidences[-1]:.3f}")

        return alerts

    def _generate_drift_recommendations(
        self,
        confidence_trend: List[Dict[str, Any]],
        distribution_change: Dict[str, float],
        decline_alerts: List[str]
    ) -> List[str]:
        """Generate recommendations based on drift analysis."""
        recommendations = []

        # Confidence-based recommendations
        if decline_alerts:
            recommendations.append("Model performance monitoring required - confidence levels declining")
            recommendations.append("Consider retraining model with recent data")

        # Distribution change recommendations
        for class_name, change in distribution_change.items():
            if abs(change) > 0.1:  # 10% change
                if change > 0:
                    recommendations.append(f"Increase in {class_name} predictions (+{change:.1%}) - verify data quality")
                else:
                    recommendations.append(f"Decrease in {class_name} predictions ({change:.1%}) - check for missing cases")

        # Data quality recommendations
        if confidence_trend:
            recent_avg = confidence_trend[-1]["average_confidence"] if confidence_trend else 0
            if recent_avg < 0.75:
                recommendations.append("Review input data quality - low confidence may indicate data shift")

        # General recommendations
        if not recommendations:
            recommendations.append("Model performance appears stable - continue monitoring")

        return recommendations

    def log_performance_metrics(
        self,
        model_version: str,
        accuracy: Optional[float] = None,
        precision: Optional[float] = None,
        recall: Optional[float] = None,
        f1_score: Optional[float] = None,
        auc_score: Optional[float] = None,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None
    ) -> bool:
        """Log model performance metrics to database."""
        try:
            if not period_end:
                period_end = datetime.utcnow()
            if not period_start:
                period_start = period_end - timedelta(days=1)

            with self.SessionLocal() as session:
                # Calculate current metrics
                metrics = self.calculate_dashboard_metrics(days_lookback=1)

                performance_log = ModelPerformanceLog(
                    model_version=model_version,
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    f1_score=f1_score,
                    auc_score=auc_score,
                    average_confidence=metrics.average_confidence,
                    high_confidence_rate=metrics.confidence_distribution.get("high", 0) / metrics.total_predictions if metrics.total_predictions > 0 else 0,
                    medium_confidence_rate=metrics.confidence_distribution.get("medium", 0) / metrics.total_predictions if metrics.total_predictions > 0 else 0,
                    low_confidence_rate=metrics.confidence_distribution.get("low", 0) / metrics.total_predictions if metrics.total_predictions > 0 else 0,
                    total_predictions=metrics.total_predictions,
                    auto_approved=metrics.triage_distribution.get("auto_approve", 0),
                    expedited_review=metrics.triage_distribution.get("expedited_review", 0),
                    senior_review=metrics.triage_distribution.get("senior_review", 0),
                    class_distribution=json.dumps(metrics.classification_distribution),
                    period_start=period_start,
                    period_end=period_end
                )

                session.add(performance_log)
                session.commit()

            logger.info(f"Performance metrics logged for model {model_version}")
            return True

        except Exception as e:
            logger.error(f"Failed to log performance metrics: {e}")
            return False