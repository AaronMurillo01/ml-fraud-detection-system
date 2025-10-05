"""Kafka consumer for fraud detection events."""

import json
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, Callable, List, Awaitable
from contextlib import asynccontextmanager

from aiokafka import AIOKafkaConsumer, ConsumerRecord
from aiokafka.errors import KafkaError
from pydantic import BaseModel, ValidationError

from service.models import Transaction, ModelScore
from service.core.logging import get_logger
from .kafka_config import get_kafka_manager, KafkaManager
from .producer import KafkaMessage

logger = get_logger("fraud_detection.consumer")


class MessageHandler:
    """Base message handler interface."""
    
    async def handle(self, message: KafkaMessage, raw_record: ConsumerRecord) -> None:
        """Handle a Kafka message."""
        raise NotImplementedError


class TransactionHandler(MessageHandler):
    """Handler for transaction messages."""
    
    def __init__(self, processor_callback: Callable[[Transaction], Awaitable[None]]):
        self.processor_callback = processor_callback
    
    async def handle(self, message: KafkaMessage, raw_record: ConsumerRecord) -> None:
        """Handle transaction message."""
        try:
            transaction = Transaction(**message.data)
            await self.processor_callback(transaction)
            
            logger.debug(f"Processed transaction: {transaction.transaction_id}")
            
        except ValidationError as e:
            logger.error(f"Invalid transaction data: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to process transaction: {e}")
            raise


class ScoreHandler(MessageHandler):
    """Handler for fraud score messages."""
    
    def __init__(self, processor_callback: Callable[[ModelScore, str], Awaitable[None]]):
        self.processor_callback = processor_callback
    
    async def handle(self, message: KafkaMessage, raw_record: ConsumerRecord) -> None:
        """Handle fraud score message."""
        try:
            transaction_id = message.data.pop("transaction_id")
            score = ModelScore(**message.data)
            
            await self.processor_callback(score, transaction_id)
            
            logger.debug(f"Processed fraud score: {transaction_id}")
            
        except ValidationError as e:
            logger.error(f"Invalid fraud score data: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to process fraud score: {e}")
            raise


class AlertHandler(MessageHandler):
    """Handler for fraud alert messages."""
    
    def __init__(self, processor_callback: Callable[[Dict[str, Any]], Awaitable[None]]):
        self.processor_callback = processor_callback
    
    async def handle(self, message: KafkaMessage, raw_record: ConsumerRecord) -> None:
        """Handle fraud alert message."""
        try:
            await self.processor_callback(message.data)
            
            logger.info(f"Processed fraud alert: {message.data.get('transaction_id')}")
            
        except Exception as e:
            logger.error(f"Failed to process fraud alert: {e}")
            raise


class FeedbackHandler(MessageHandler):
    """Handler for fraud feedback messages."""
    
    def __init__(self, processor_callback: Callable[[Dict[str, Any]], Awaitable[None]]):
        self.processor_callback = processor_callback
    
    async def handle(self, message: KafkaMessage, raw_record: ConsumerRecord) -> None:
        """Handle fraud feedback message."""
        try:
            await self.processor_callback(message.data)
            
            logger.info(f"Processed fraud feedback: {message.data.get('transaction_id')}")
            
        except Exception as e:
            logger.error(f"Failed to process fraud feedback: {e}")
            raise


class FraudDetectionConsumer:
    """Kafka consumer for fraud detection events."""
    
    def __init__(self, kafka_manager: KafkaManager, group_id: Optional[str] = None):
        self.kafka_manager = kafka_manager
        self.group_id = group_id
        self._consumer: Optional[AIOKafkaConsumer] = None
        self._handlers: Dict[str, MessageHandler] = {}
        self._running = False
        self._consumer_task: Optional[asyncio.Task] = None
    
    def register_handler(self, event_type: str, handler: MessageHandler) -> None:
        """Register a message handler for specific event type."""
        self._handlers[event_type] = handler
        logger.info(f"Registered handler for event type: {event_type}")
    
    def register_transaction_handler(
        self, 
        callback: Callable[[Transaction], Awaitable[None]]
    ) -> None:
        """Register transaction handler."""
        handler = TransactionHandler(callback)
        self.register_handler("transaction_received", handler)
    
    def register_score_handler(
        self, 
        callback: Callable[[ModelScore, str], Awaitable[None]]
    ) -> None:
        """Register fraud score handler."""
        handler = ScoreHandler(callback)
        self.register_handler("fraud_score_calculated", handler)
    
    def register_alert_handler(
        self, 
        callback: Callable[[Dict[str, Any]], Awaitable[None]]
    ) -> None:
        """Register fraud alert handler."""
        handler = AlertHandler(callback)
        self.register_handler("fraud_alert", handler)
    
    def register_feedback_handler(
        self, 
        callback: Callable[[Dict[str, Any]], Awaitable[None]]
    ) -> None:
        """Register fraud feedback handler."""
        handler = FeedbackHandler(callback)
        self.register_handler("fraud_feedback", handler)
    
    async def start(self, topics: List[str]) -> None:
        """Start consuming from specified topics."""
        if self._running:
            logger.warning("Consumer is already running")
            return
        
        self._consumer = await self.kafka_manager.get_consumer(topics, self.group_id)
        self._running = True
        
        # Start consumer task
        self._consumer_task = asyncio.create_task(self._consume_loop())
        
        logger.info(f"Fraud detection consumer started for topics: {topics}")
    
    async def _consume_loop(self) -> None:
        """Main consumer loop."""
        try:
            async for record in self._consumer:
                if not self._running:
                    break
                
                await self._process_record(record)
                
        except asyncio.CancelledError:
            logger.info("Consumer loop cancelled")
        except Exception as e:
            logger.error(f"Consumer loop error: {e}")
            raise
    
    async def _process_record(self, record: ConsumerRecord) -> None:
        """Process a single Kafka record."""
        try:
            # Parse message
            message_data = json.loads(record.value.decode('utf-8'))
            message = KafkaMessage(**message_data)
            
            # Get handler for event type
            handler = self._handlers.get(message.event_type)
            if not handler:
                logger.warning(f"No handler for event type: {message.event_type}")
                return
            
            # Process message
            await handler.handle(message, record)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode message JSON: {e}")
        except ValidationError as e:
            logger.error(f"Invalid message format: {e}")
        except Exception as e:
            logger.error(f"Failed to process record: {e}")
            # In production, you might want to send to dead letter queue
            raise
    
    async def stop(self) -> None:
        """Stop the consumer."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel consumer task
        if self._consumer_task:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
        
        # Stop consumer
        if self._consumer:
            await self._consumer.stop()
            self._consumer = None
        
        logger.info("Fraud detection consumer stopped")
    
    @asynccontextmanager
    async def consume_context(self, topics: List[str]):
        """Context manager for consuming messages."""
        await self.start(topics)
        try:
            yield self
        finally:
            await self.stop()


class StreamProcessor:
    """High-level stream processor for fraud detection."""
    
    def __init__(self, kafka_manager: KafkaManager):
        self.kafka_manager = kafka_manager
        self.consumers: Dict[str, FraudDetectionConsumer] = {}
    
    def create_consumer(self, name: str, group_id: Optional[str] = None) -> FraudDetectionConsumer:
        """Create a named consumer."""
        consumer = FraudDetectionConsumer(self.kafka_manager, group_id)
        self.consumers[name] = consumer
        return consumer
    
    async def start_transaction_processor(
        self, 
        callback: Callable[[Transaction], Awaitable[None]],
        group_id: str = "transaction_processor"
    ) -> None:
        """Start processing transaction events."""
        consumer = self.create_consumer("transaction_processor", group_id)
        consumer.register_transaction_handler(callback)
        
        await consumer.start([self.kafka_manager.config.topic_transactions])
    
    async def start_score_processor(
        self, 
        callback: Callable[[ModelScore, str], Awaitable[None]],
        group_id: str = "score_processor"
    ) -> None:
        """Start processing fraud score events."""
        consumer = self.create_consumer("score_processor", group_id)
        consumer.register_score_handler(callback)
        
        await consumer.start([self.kafka_manager.config.topic_fraud_scores])
    
    async def start_alert_processor(
        self, 
        callback: Callable[[Dict[str, Any]], Awaitable[None]],
        group_id: str = "alert_processor"
    ) -> None:
        """Start processing fraud alert events."""
        consumer = self.create_consumer("alert_processor", group_id)
        consumer.register_alert_handler(callback)
        
        await consumer.start([self.kafka_manager.config.topic_alerts])
    
    async def start_feedback_processor(
        self, 
        callback: Callable[[Dict[str, Any]], Awaitable[None]],
        group_id: str = "feedback_processor"
    ) -> None:
        """Start processing fraud feedback events."""
        consumer = self.create_consumer("feedback_processor", group_id)
        consumer.register_feedback_handler(callback)
        
        await consumer.start([self.kafka_manager.config.topic_feedback])
    
    async def stop_all(self) -> None:
        """Stop all consumers."""
        tasks = []
        for consumer in self.consumers.values():
            tasks.append(consumer.stop())
        
        if tasks:
            await asyncio.gather(*tasks)
        
        self.consumers.clear()
        logger.info("All stream processors stopped")


# Global stream processor instance
_stream_processor: Optional[StreamProcessor] = None


async def get_stream_processor() -> StreamProcessor:
    """Get global stream processor instance."""
    global _stream_processor
    
    if _stream_processor is None:
        kafka_manager = await get_kafka_manager()
        _stream_processor = StreamProcessor(kafka_manager)
    
    return _stream_processor


async def close_stream_processor() -> None:
    """Close global stream processor."""
    global _stream_processor
    
    if _stream_processor:
        await _stream_processor.stop_all()
        _stream_processor = None