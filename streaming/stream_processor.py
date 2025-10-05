"""Real-time stream processor for fraud detection."""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Awaitable
from collections import defaultdict, deque
from dataclasses import dataclass, field

from service.models import Transaction, ModelScore, EnrichedTransaction, RiskLevel
from service.core.logging import get_logger
from service.core.config import get_settings
from .kafka_config import get_kafka_manager
from .producer import get_fraud_producer
from .consumer import get_stream_processor

logger = get_logger("fraud_detection.stream_processor")


@dataclass
class ProcessingMetrics:
    """Stream processing metrics."""
    
    transactions_processed: int = 0
    scores_calculated: int = 0
    alerts_generated: int = 0
    processing_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    error_count: int = 0
    last_processed: Optional[datetime] = None
    
    @property
    def avg_processing_time_ms(self) -> float:
        """Average processing time in milliseconds."""
        if not self.processing_times:
            return 0.0
        return sum(self.processing_times) / len(self.processing_times) * 1000
    
    @property
    def throughput_per_second(self) -> float:
        """Transactions per second (last minute)."""
        if not self.last_processed:
            return 0.0
        
        # Simple approximation based on recent activity
        return self.transactions_processed / 60.0 if self.transactions_processed > 0 else 0.0


class TransactionEnricher:
    """Enriches transactions with additional features for fraud detection."""
    
    def __init__(self):
        self.user_transaction_history: Dict[str, List[Transaction]] = defaultdict(list)
        self.merchant_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.velocity_windows = {
            "1min": timedelta(minutes=1),
            "5min": timedelta(minutes=5),
            "1hour": timedelta(hours=1),
            "24hour": timedelta(hours=24)
        }
    
    async def enrich_transaction(self, transaction: Transaction) -> EnrichedTransaction:
        """Enrich transaction with computed features."""
        start_time = datetime.utcnow()
        
        try:
            # Calculate velocity features
            velocity_features = await self._calculate_velocity_features(transaction)
            
            # Calculate user behavior features
            user_features = await self._calculate_user_features(transaction)
            
            # Calculate merchant features
            merchant_features = await self._calculate_merchant_features(transaction)
            
            # Calculate time-based features
            time_features = await self._calculate_time_features(transaction)
            
            # Calculate amount-based features
            amount_features = await self._calculate_amount_features(transaction)
            
            # Create enriched transaction
            enriched = EnrichedTransaction(
                **transaction.dict(),
                velocity_1min=velocity_features["1min"],
                velocity_5min=velocity_features["5min"],
                velocity_1hour=velocity_features["1hour"],
                velocity_24hour=velocity_features["24hour"],
                user_transaction_count_1hour=user_features["count_1hour"],
                user_avg_amount_24hour=user_features["avg_amount_24hour"],
                merchant_risk_score=merchant_features["risk_score"],
                is_weekend=time_features["is_weekend"],
                is_night_time=time_features["is_night_time"],
                hour_of_day=time_features["hour_of_day"],
                amount_zscore=amount_features["zscore"],
                is_round_amount=amount_features["is_round"]
            )
            
            # Update transaction history
            self._update_transaction_history(transaction)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            logger.debug(f"Transaction enriched in {processing_time:.3f}s: {transaction.transaction_id}")
            
            return enriched
            
        except Exception as e:
            logger.error(f"Failed to enrich transaction {transaction.transaction_id}: {e}")
            # Return basic enriched transaction with default values
            return EnrichedTransaction(**transaction.dict())
    
    async def _calculate_velocity_features(self, transaction: Transaction) -> Dict[str, int]:
        """Calculate transaction velocity features."""
        user_history = self.user_transaction_history.get(transaction.user_id, [])
        current_time = transaction.timestamp
        
        velocity = {}
        
        for window_name, window_duration in self.velocity_windows.items():
            window_start = current_time - window_duration
            
            count = sum(
                1 for tx in user_history 
                if tx.timestamp >= window_start and tx.timestamp < current_time
            )
            
            velocity[window_name] = count
        
        return velocity
    
    async def _calculate_user_features(self, transaction: Transaction) -> Dict[str, float]:
        """Calculate user behavior features."""
        user_history = self.user_transaction_history.get(transaction.user_id, [])
        current_time = transaction.timestamp
        
        # Count transactions in last hour
        hour_ago = current_time - timedelta(hours=1)
        count_1hour = sum(
            1 for tx in user_history 
            if tx.timestamp >= hour_ago and tx.timestamp < current_time
        )
        
        # Average amount in last 24 hours
        day_ago = current_time - timedelta(hours=24)
        recent_amounts = [
            float(tx.amount) for tx in user_history 
            if tx.timestamp >= day_ago and tx.timestamp < current_time
        ]
        
        avg_amount_24hour = sum(recent_amounts) / len(recent_amounts) if recent_amounts else 0.0
        
        return {
            "count_1hour": count_1hour,
            "avg_amount_24hour": avg_amount_24hour
        }
    
    async def _calculate_merchant_features(self, transaction: Transaction) -> Dict[str, float]:
        """Calculate merchant-based features."""
        merchant_id = transaction.merchant_id
        
        # Simple risk score based on merchant category and historical data
        # In production, this would use more sophisticated merchant profiling
        risk_score = 0.5  # Default neutral risk
        
        # Adjust based on merchant category (if available)
        high_risk_categories = ["gambling", "adult", "cryptocurrency"]
        if hasattr(transaction, 'merchant_category') and transaction.merchant_category in high_risk_categories:
            risk_score = 0.8
        
        return {
            "risk_score": risk_score
        }
    
    async def _calculate_time_features(self, transaction: Transaction) -> Dict[str, Any]:
        """Calculate time-based features."""
        timestamp = transaction.timestamp
        
        # Weekend detection
        is_weekend = timestamp.weekday() >= 5
        
        # Night time detection (10 PM - 6 AM)
        hour = timestamp.hour
        is_night_time = hour >= 22 or hour <= 6
        
        return {
            "is_weekend": is_weekend,
            "is_night_time": is_night_time,
            "hour_of_day": hour
        }
    
    async def _calculate_amount_features(self, transaction: Transaction) -> Dict[str, Any]:
        """Calculate amount-based features."""
        amount = float(transaction.amount)
        
        # Z-score calculation (simplified - in production use historical data)
        # Using rough estimates for credit card transactions
        mean_amount = 75.0
        std_amount = 150.0
        zscore = (amount - mean_amount) / std_amount if std_amount > 0 else 0.0
        
        # Round amount detection
        is_round = amount == round(amount) and amount % 10 == 0
        
        return {
            "zscore": zscore,
            "is_round": is_round
        }
    
    def _update_transaction_history(self, transaction: Transaction) -> None:
        """Update user transaction history."""
        user_history = self.user_transaction_history[transaction.user_id]
        user_history.append(transaction)
        
        # Keep only recent transactions (last 7 days)
        cutoff_time = transaction.timestamp - timedelta(days=7)
        self.user_transaction_history[transaction.user_id] = [
            tx for tx in user_history if tx.timestamp >= cutoff_time
        ]


class FraudDetectionStreamProcessor:
    """Main stream processor for real-time fraud detection."""
    
    def __init__(self):
        self.enricher = TransactionEnricher()
        self.metrics = ProcessingMetrics()
        self.settings = get_settings()
        self._running = False
        self._ml_service: Optional[Any] = None  # Will be injected
        self._alert_thresholds = {
            RiskLevel.HIGH: 0.8,
            RiskLevel.MEDIUM: 0.6,
            RiskLevel.LOW: 0.4
        }
    
    def set_ml_service(self, ml_service: Any) -> None:
        """Inject ML service for fraud scoring."""
        self._ml_service = ml_service
    
    async def start(self) -> None:
        """Start the stream processor."""
        if self._running:
            logger.warning("Stream processor is already running")
            return
        
        try:
            # Get stream processor and producer
            stream_processor = await get_stream_processor()
            producer = await get_fraud_producer()
            
            # Start transaction processing
            await stream_processor.start_transaction_processor(
                self._process_transaction,
                group_id="fraud_detection_main"
            )
            
            self._running = True
            logger.info("Fraud detection stream processor started")
            
        except Exception as e:
            logger.error(f"Failed to start stream processor: {e}")
            raise
    
    async def _process_transaction(self, transaction: Transaction) -> None:
        """Process a single transaction through the fraud detection pipeline."""
        start_time = datetime.utcnow()
        
        try:
            # Step 1: Enrich transaction
            enriched_transaction = await self.enricher.enrich_transaction(transaction)
            
            # Step 2: Calculate fraud score
            if self._ml_service:
                fraud_score = await self._ml_service.predict_fraud(enriched_transaction)
            else:
                # Fallback simple scoring if ML service not available
                fraud_score = await self._simple_fraud_scoring(enriched_transaction)
            
            # Step 3: Send fraud score to Kafka
            producer = await get_fraud_producer()
            await producer.send_fraud_score(fraud_score, transaction.transaction_id)
            
            # Step 4: Check for alerts
            await self._check_and_send_alerts(fraud_score, transaction)
            
            # Update metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self.metrics.transactions_processed += 1
            self.metrics.scores_calculated += 1
            self.metrics.processing_times.append(processing_time)
            self.metrics.last_processed = datetime.utcnow()
            
            # Log performance warning if processing is slow
            if processing_time > 0.05:  # 50ms threshold
                logger.warning(
                    f"Slow transaction processing: {processing_time:.3f}s for {transaction.transaction_id}"
                )
            
        except Exception as e:
            self.metrics.error_count += 1
            logger.error(f"Failed to process transaction {transaction.transaction_id}: {e}")
            raise
    
    async def _simple_fraud_scoring(self, enriched_transaction: EnrichedTransaction) -> ModelScore:
        """Simple rule-based fraud scoring as fallback."""
        score = 0.0
        features = []
        
        # High velocity scoring
        if enriched_transaction.velocity_1min > 5:
            score += 0.3
            features.append("high_velocity_1min")
        
        if enriched_transaction.velocity_5min > 10:
            score += 0.2
            features.append("high_velocity_5min")
        
        # Amount-based scoring
        if enriched_transaction.amount_zscore > 3:
            score += 0.25
            features.append("unusual_amount")
        
        # Time-based scoring
        if enriched_transaction.is_night_time:
            score += 0.1
            features.append("night_transaction")
        
        # Merchant risk
        if enriched_transaction.merchant_risk_score > 0.7:
            score += 0.15
            features.append("high_risk_merchant")
        
        # Determine risk level
        if score >= 0.8:
            risk_level = RiskLevel.HIGH
            recommended_action = "block"
        elif score >= 0.6:
            risk_level = RiskLevel.MEDIUM
            recommended_action = "review"
        elif score >= 0.4:
            risk_level = RiskLevel.LOW
            recommended_action = "monitor"
        else:
            risk_level = RiskLevel.LOW
            recommended_action = "approve"
        
        return ModelScore(
            fraud_score=min(score, 1.0),
            risk_level=risk_level,
            recommended_action=recommended_action,
            confidence_score=0.7,  # Lower confidence for rule-based scoring
            feature_importance=[
                {"feature": feature, "importance": 0.1}
                for feature in features[:5]  # Top 5 features
            ]
        )
    
    async def _check_and_send_alerts(self, fraud_score: ModelScore, transaction: Transaction) -> None:
        """Check if alerts should be sent based on fraud score."""
        threshold = self._alert_thresholds.get(fraud_score.risk_level, 0.5)
        
        if fraud_score.fraud_score >= threshold:
            producer = await get_fraud_producer()
            
            alert_reason = f"High fraud score: {fraud_score.fraud_score:.3f} (threshold: {threshold})"
            
            await producer.send_fraud_alert(
                fraud_score,
                transaction,
                alert_reason
            )
            
            self.metrics.alerts_generated += 1
            
            logger.warning(
                f"Fraud alert generated for transaction {transaction.transaction_id}: "
                f"score={fraud_score.fraud_score:.3f}, risk={fraud_score.risk_level.value}"
            )
    
    async def stop(self) -> None:
        """Stop the stream processor."""
        if not self._running:
            return
        
        self._running = False
        
        # Stop stream processor
        stream_processor = await get_stream_processor()
        await stream_processor.stop_all()
        
        logger.info("Fraud detection stream processor stopped")
    
    def get_metrics(self) -> ProcessingMetrics:
        """Get current processing metrics."""
        return self.metrics
    
    def reset_metrics(self) -> None:
        """Reset processing metrics."""
        self.metrics = ProcessingMetrics()
        logger.info("Processing metrics reset")


# Global stream processor instance
_fraud_stream_processor: Optional[FraudDetectionStreamProcessor] = None


async def get_fraud_stream_processor() -> FraudDetectionStreamProcessor:
    """Get global fraud detection stream processor."""
    global _fraud_stream_processor
    
    if _fraud_stream_processor is None:
        _fraud_stream_processor = FraudDetectionStreamProcessor()
    
    return _fraud_stream_processor


async def start_fraud_detection_pipeline() -> None:
    """Start the complete fraud detection pipeline."""
    processor = await get_fraud_stream_processor()
    await processor.start()


async def stop_fraud_detection_pipeline() -> None:
    """Stop the fraud detection pipeline."""
    global _fraud_stream_processor
    
    if _fraud_stream_processor:
        await _fraud_stream_processor.stop()
        _fraud_stream_processor = None