"""Kafka producer for fraud detection events."""

import json
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List
from uuid import uuid4

from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaError
from pydantic import BaseModel

from service.models import Transaction, ModelScore, EnrichedTransaction
from service.core.logging import get_logger
from .kafka_config import get_kafka_manager, KafkaManager

logger = get_logger("fraud_detection.producer")


class KafkaMessage(BaseModel):
    """Base Kafka message structure."""
    
    message_id: str
    timestamp: datetime
    event_type: str
    source: str = "fraud_detection_service"
    version: str = "1.0"
    data: Dict[str, Any]
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TransactionMessage(KafkaMessage):
    """Transaction event message."""
    
    event_type: str = "transaction_received"
    
    @classmethod
    def from_transaction(cls, transaction: Transaction) -> "TransactionMessage":
        """Create message from transaction."""
        return cls(
            message_id=str(uuid4()),
            timestamp=datetime.utcnow(),
            data=transaction.dict()
        )


class ScoreMessage(KafkaMessage):
    """Fraud score event message."""
    
    event_type: str = "fraud_score_calculated"
    
    @classmethod
    def from_score(cls, score: ModelScore, transaction_id: str) -> "ScoreMessage":
        """Create message from fraud score."""
        return cls(
            message_id=str(uuid4()),
            timestamp=datetime.utcnow(),
            data={
                "transaction_id": transaction_id,
                **score.dict()
            }
        )


class AlertMessage(KafkaMessage):
    """Fraud alert message."""
    
    event_type: str = "fraud_alert"
    
    @classmethod
    def from_score_and_transaction(
        cls, 
        score: ModelScore, 
        transaction: Transaction,
        alert_reason: str
    ) -> "AlertMessage":
        """Create alert message from score and transaction."""
        return cls(
            message_id=str(uuid4()),
            timestamp=datetime.utcnow(),
            data={
                "transaction_id": transaction.transaction_id,
                "user_id": transaction.user_id,
                "amount": float(transaction.amount),
                "merchant_id": transaction.merchant_id,
                "fraud_score": score.fraud_score,
                "risk_level": score.risk_level.value,
                "recommended_action": score.recommended_action.value,
                "alert_reason": alert_reason,
                "model_version": score.model_version.value,
                "confidence_score": score.confidence_score
            }
        )


class FeedbackMessage(KafkaMessage):
    """Fraud feedback message."""
    
    event_type: str = "fraud_feedback"
    
    @classmethod
    def from_feedback(
        cls,
        transaction_id: str,
        actual_fraud: bool,
        feedback_source: str,
        notes: Optional[str] = None
    ) -> "FeedbackMessage":
        """Create feedback message."""
        return cls(
            message_id=str(uuid4()),
            timestamp=datetime.utcnow(),
            data={
                "transaction_id": transaction_id,
                "actual_fraud": actual_fraud,
                "feedback_source": feedback_source,
                "notes": notes
            }
        )


class FraudDetectionProducer:
    """Kafka producer for fraud detection events."""
    
    def __init__(self, kafka_manager: KafkaManager):
        self.kafka_manager = kafka_manager
        self._producer: Optional[AIOKafkaProducer] = None
    
    async def start(self) -> None:
        """Start the producer."""
        self._producer = await self.kafka_manager.get_producer()
        logger.info("Fraud detection producer started")
    
    async def send_transaction(self, transaction: Transaction) -> None:
        """Send transaction event to Kafka."""
        if not self._producer:
            await self.start()
        
        message = TransactionMessage.from_transaction(transaction)
        
        try:
            await self._producer.send_and_wait(
                topic=self.kafka_manager.config.topic_transactions,
                key=transaction.transaction_id.encode('utf-8'),
                value=message.json().encode('utf-8'),
                headers=[
                    ("event_type", message.event_type.encode('utf-8')),
                    ("source", message.source.encode('utf-8')),
                    ("version", message.version.encode('utf-8'))
                ]
            )
            
            logger.debug(f"Transaction sent to Kafka: {transaction.transaction_id}")
            
        except KafkaError as e:
            logger.error(f"Failed to send transaction to Kafka: {e}")
            raise
    
    async def send_fraud_score(
        self, 
        score: ModelScore, 
        transaction_id: str
    ) -> None:
        """Send fraud score to Kafka."""
        if not self._producer:
            await self.start()
        
        message = ScoreMessage.from_score(score, transaction_id)
        
        try:
            await self._producer.send_and_wait(
                topic=self.kafka_manager.config.topic_fraud_scores,
                key=transaction_id.encode('utf-8'),
                value=message.json().encode('utf-8'),
                headers=[
                    ("event_type", message.event_type.encode('utf-8')),
                    ("source", message.source.encode('utf-8')),
                    ("version", message.version.encode('utf-8')),
                    ("risk_level", score.risk_level.value.encode('utf-8'))
                ]
            )
            
            logger.debug(f"Fraud score sent to Kafka: {transaction_id}")
            
        except KafkaError as e:
            logger.error(f"Failed to send fraud score to Kafka: {e}")
            raise
    
    async def send_fraud_alert(
        self,
        score: ModelScore,
        transaction: Transaction,
        alert_reason: str
    ) -> None:
        """Send fraud alert to Kafka."""
        if not self._producer:
            await self.start()
        
        message = AlertMessage.from_score_and_transaction(
            score, transaction, alert_reason
        )
        
        try:
            await self._producer.send_and_wait(
                topic=self.kafka_manager.config.topic_alerts,
                key=transaction.transaction_id.encode('utf-8'),
                value=message.json().encode('utf-8'),
                headers=[
                    ("event_type", message.event_type.encode('utf-8')),
                    ("source", message.source.encode('utf-8')),
                    ("version", message.version.encode('utf-8')),
                    ("risk_level", score.risk_level.value.encode('utf-8')),
                    ("alert_reason", alert_reason.encode('utf-8'))
                ]
            )
            
            logger.warning(
                f"Fraud alert sent: {transaction.transaction_id} - {alert_reason}"
            )
            
        except KafkaError as e:
            logger.error(f"Failed to send fraud alert to Kafka: {e}")
            raise
    
    async def send_feedback(
        self,
        transaction_id: str,
        actual_fraud: bool,
        feedback_source: str,
        notes: Optional[str] = None
    ) -> None:
        """Send fraud feedback to Kafka."""
        if not self._producer:
            await self.start()
        
        message = FeedbackMessage.from_feedback(
            transaction_id, actual_fraud, feedback_source, notes
        )
        
        try:
            await self._producer.send_and_wait(
                topic=self.kafka_manager.config.topic_feedback,
                key=transaction_id.encode('utf-8'),
                value=message.json().encode('utf-8'),
                headers=[
                    ("event_type", message.event_type.encode('utf-8')),
                    ("source", message.source.encode('utf-8')),
                    ("version", message.version.encode('utf-8')),
                    ("feedback_source", feedback_source.encode('utf-8'))
                ]
            )
            
            logger.info(
                f"Feedback sent: {transaction_id} - fraud: {actual_fraud}"
            )
            
        except KafkaError as e:
            logger.error(f"Failed to send feedback to Kafka: {e}")
            raise
    
    async def send_batch_scores(
        self, 
        scores_and_transactions: List[tuple[ModelScore, str]]
    ) -> None:
        """Send multiple fraud scores in batch."""
        if not self._producer:
            await self.start()
        
        tasks = []
        
        for score, transaction_id in scores_and_transactions:
            message = ScoreMessage.from_score(score, transaction_id)
            
            task = self._producer.send(
                topic=self.kafka_manager.config.topic_fraud_scores,
                key=transaction_id.encode('utf-8'),
                value=message.json().encode('utf-8'),
                headers=[
                    ("event_type", message.event_type.encode('utf-8')),
                    ("source", message.source.encode('utf-8')),
                    ("version", message.version.encode('utf-8')),
                    ("risk_level", score.risk_level.value.encode('utf-8'))
                ]
            )
            tasks.append(task)
        
        try:
            # Wait for all messages to be sent
            await asyncio.gather(*tasks)
            logger.info(f"Batch of {len(scores_and_transactions)} scores sent to Kafka")
            
        except KafkaError as e:
            logger.error(f"Failed to send batch scores to Kafka: {e}")
            raise
    
    async def flush(self) -> None:
        """Flush producer buffer."""
        if self._producer:
            await self._producer.flush()
    
    async def stop(self) -> None:
        """Stop the producer."""
        if self._producer:
            await self._producer.stop()
            self._producer = None
            logger.info("Fraud detection producer stopped")


# Global producer instance
_producer: Optional[FraudDetectionProducer] = None


async def get_fraud_producer() -> FraudDetectionProducer:
    """Get global fraud detection producer instance."""
    global _producer
    
    if _producer is None:
        kafka_manager = await get_kafka_manager()
        _producer = FraudDetectionProducer(kafka_manager)
        await _producer.start()
    
    return _producer


async def close_fraud_producer() -> None:
    """Close global fraud detection producer."""
    global _producer
    
    if _producer:
        await _producer.stop()
        _producer = None