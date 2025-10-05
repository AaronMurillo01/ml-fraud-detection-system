"""Integration tests for Kafka message processing."""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from typing import Dict, Any, List

from shared.models import Transaction, FraudPrediction, PaymentMethod, TransactionStatus, RiskLevel
from tests.fixtures.test_data import sample_transactions, sample_predictions
from tests.fixtures.mock_objects import MockKafkaProducer, MockKafkaConsumer


class TestKafkaProducerIntegration:
    """Test Kafka producer functionality."""
    
    @pytest.fixture
    def mock_producer(self):
        """Create a mock Kafka producer."""
        return MockKafkaProducer()
    
    @pytest.fixture
    def sample_transaction_message(self):
        """Create a sample transaction message."""
        return {
            "transaction_id": "txn_001",
            "user_id": "user_001",
            "merchant_id": "merchant_001",
            "amount": 25.99,
            "timestamp": datetime.utcnow().isoformat(),
            "payment_method": "CREDIT_CARD",
            "transaction_type": "purchase",
            "status": "APPROVED",
            "currency": "USD",
            "description": "Test transaction",
            "metadata": {
                "ip_address": "192.168.1.100",
                "user_agent": "TestAgent/1.0",
                "device_id": "device_001",
                "location": "New York, NY"
            }
        }
    
    @pytest.fixture
    def sample_prediction_message(self):
        """Create a sample fraud prediction message."""
        return {
            "prediction_id": "pred_001",
            "transaction_id": "txn_001",
            "user_id": "user_001",
            "fraud_probability": 0.15,
            "risk_level": "LOW",
            "decision": "APPROVE",
            "confidence_score": 0.92,
            "model_version": "v1.0.0",
            "model_features": {
                "amount": 25.99,
                "merchant_risk": 0.10,
                "user_risk": 0.25
            },
            "feature_importance": {
                "amount": 0.25,
                "merchant_risk": 0.20,
                "user_risk": 0.15
            },
            "created_at": datetime.utcnow().isoformat()
        }
    
    def test_send_transaction_message(self, mock_producer, sample_transaction_message):
        """Test sending a transaction message to Kafka."""
        topic = "transactions"
        key = sample_transaction_message["transaction_id"]
        
        # Send message
        result = mock_producer.send(topic, key, sample_transaction_message)
        
        # Verify message was sent
        assert result is not None
        assert len(mock_producer.sent_messages) == 1
        
        sent_message = mock_producer.sent_messages[0]
        assert sent_message["topic"] == topic
        assert sent_message["key"] == key
        assert sent_message["value"] == sample_transaction_message
    
    def test_send_prediction_message(self, mock_producer, sample_prediction_message):
        """Test sending a fraud prediction message to Kafka."""
        topic = "fraud_predictions"
        key = sample_prediction_message["prediction_id"]
        
        # Send message
        result = mock_producer.send(topic, key, sample_prediction_message)
        
        # Verify message was sent
        assert result is not None
        assert len(mock_producer.sent_messages) == 1
        
        sent_message = mock_producer.sent_messages[0]
        assert sent_message["topic"] == topic
        assert sent_message["key"] == key
        assert sent_message["value"] == sample_prediction_message
    
    def test_send_batch_messages(self, mock_producer):
        """Test sending multiple messages in batch."""
        topic = "transactions"
        messages = []
        
        # Create batch of messages
        for i, txn_data in enumerate(sample_transactions[:3]):
            message = {
                "transaction_id": txn_data["transaction_id"],
                "user_id": txn_data["user_id"],
                "amount": txn_data["amount"],
                "timestamp": txn_data["timestamp"].isoformat(),
                "payment_method": txn_data["payment_method"].value,
                "status": txn_data["status"].value
            }
            messages.append((txn_data["transaction_id"], message))
        
        # Send batch
        for key, message in messages:
            mock_producer.send(topic, key, message)
        
        # Verify all messages were sent
        assert len(mock_producer.sent_messages) == 3
        
        for i, (key, message) in enumerate(messages):
            sent_message = mock_producer.sent_messages[i]
            assert sent_message["topic"] == topic
            assert sent_message["key"] == key
            assert sent_message["value"] == message
    
    def test_send_message_serialization(self, mock_producer):
        """Test message serialization with complex data types."""
        topic = "test_topic"
        key = "test_key"
        
        # Message with datetime and enum values
        message = {
            "timestamp": datetime.utcnow(),
            "payment_method": PaymentMethod.CREDIT_CARD,
            "status": TransactionStatus.APPROVED,
            "risk_level": RiskLevel.LOW,
            "nested_data": {
                "list_field": [1, 2, 3],
                "dict_field": {"key": "value"}
            }
        }
        
        # Send message (mock producer handles serialization)
        result = mock_producer.send(topic, key, message)
        
        # Verify message was sent and serialized
        assert result is not None
        assert len(mock_producer.sent_messages) == 1
        
        sent_message = mock_producer.sent_messages[0]
        assert sent_message["topic"] == topic
        assert sent_message["key"] == key
        # Mock producer stores the original message
        assert sent_message["value"] == message
    
    def test_producer_error_handling(self, mock_producer):
        """Test producer error handling."""
        topic = "test_topic"
        key = "test_key"
        message = {"test": "data"}
        
        # Simulate producer error
        mock_producer.simulate_error = True
        
        # Attempt to send message
        with pytest.raises(Exception):
            mock_producer.send(topic, key, message)
        
        # Verify no message was sent
        assert len(mock_producer.sent_messages) == 0
    
    def test_producer_flush(self, mock_producer):
        """Test producer flush functionality."""
        topic = "test_topic"
        
        # Send multiple messages
        for i in range(5):
            mock_producer.send(topic, f"key_{i}", {"message": i})
        
        # Flush producer
        mock_producer.flush()
        
        # Verify all messages are flushed
        assert mock_producer.is_flushed
        assert len(mock_producer.sent_messages) == 5


class TestKafkaConsumerIntegration:
    """Test Kafka consumer functionality."""
    
    @pytest.fixture
    def mock_consumer(self):
        """Create a mock Kafka consumer."""
        return MockKafkaConsumer(["transactions", "fraud_predictions"])
    
    @pytest.fixture
    def sample_messages(self):
        """Create sample messages for consumption."""
        return [
            {
                "topic": "transactions",
                "key": "txn_001",
                "value": {
                    "transaction_id": "txn_001",
                    "user_id": "user_001",
                    "amount": 25.99,
                    "timestamp": datetime.utcnow().isoformat()
                },
                "offset": 0,
                "partition": 0
            },
            {
                "topic": "fraud_predictions",
                "key": "pred_001",
                "value": {
                    "prediction_id": "pred_001",
                    "transaction_id": "txn_001",
                    "fraud_probability": 0.15,
                    "risk_level": "LOW"
                },
                "offset": 1,
                "partition": 0
            }
        ]
    
    def test_consume_single_message(self, mock_consumer, sample_messages):
        """Test consuming a single message."""
        # Add message to consumer
        mock_consumer.add_message(sample_messages[0])
        
        # Consume message
        messages = mock_consumer.poll(timeout_ms=1000, max_records=1)
        
        # Verify message was consumed
        assert len(messages) == 1
        message = messages[0]
        assert message["topic"] == "transactions"
        assert message["key"] == "txn_001"
        assert message["value"]["transaction_id"] == "txn_001"
    
    def test_consume_multiple_messages(self, mock_consumer, sample_messages):
        """Test consuming multiple messages."""
        # Add messages to consumer
        for message in sample_messages:
            mock_consumer.add_message(message)
        
        # Consume messages
        messages = mock_consumer.poll(timeout_ms=1000, max_records=10)
        
        # Verify all messages were consumed
        assert len(messages) == 2
        
        # Verify message order and content
        txn_message = next(m for m in messages if m["topic"] == "transactions")
        pred_message = next(m for m in messages if m["topic"] == "fraud_predictions")
        
        assert txn_message["key"] == "txn_001"
        assert pred_message["key"] == "pred_001"
    
    def test_consume_with_timeout(self, mock_consumer):
        """Test consumer timeout when no messages available."""
        # Poll with timeout (no messages available)
        messages = mock_consumer.poll(timeout_ms=100, max_records=1)
        
        # Verify no messages returned
        assert len(messages) == 0
    
    def test_consumer_commit_offset(self, mock_consumer, sample_messages):
        """Test committing consumer offsets."""
        # Add and consume message
        mock_consumer.add_message(sample_messages[0])
        messages = mock_consumer.poll(timeout_ms=1000, max_records=1)
        
        # Commit offset
        mock_consumer.commit()
        
        # Verify offset was committed
        assert mock_consumer.committed_offsets is not None
        assert len(mock_consumer.committed_offsets) > 0
    
    def test_consumer_seek_to_beginning(self, mock_consumer, sample_messages):
        """Test seeking to beginning of topic."""
        # Add messages
        for message in sample_messages:
            mock_consumer.add_message(message)
        
        # Consume some messages
        mock_consumer.poll(timeout_ms=1000, max_records=1)
        
        # Seek to beginning
        mock_consumer.seek_to_beginning()
        
        # Verify position reset
        assert mock_consumer.position == 0
    
    def test_consumer_error_handling(self, mock_consumer):
        """Test consumer error handling."""
        # Simulate consumer error
        mock_consumer.simulate_error = True
        
        # Attempt to poll
        with pytest.raises(Exception):
            mock_consumer.poll(timeout_ms=1000, max_records=1)
    
    def test_consumer_close(self, mock_consumer):
        """Test consumer close functionality."""
        # Close consumer
        mock_consumer.close()
        
        # Verify consumer is closed
        assert mock_consumer.is_closed
        
        # Verify polling after close raises error
        with pytest.raises(Exception):
            mock_consumer.poll(timeout_ms=1000, max_records=1)


class TestKafkaMessageProcessing:
    """Test end-to-end Kafka message processing."""
    
    @pytest.fixture
    def message_processor(self):
        """Create a mock message processor."""
        class MockMessageProcessor:
            def __init__(self):
                self.processed_transactions = []
                self.processed_predictions = []
            
            def process_transaction(self, message: Dict[str, Any]):
                """Process a transaction message."""
                self.processed_transactions.append(message)
                return {"status": "processed", "transaction_id": message["transaction_id"]}
            
            def process_prediction(self, message: Dict[str, Any]):
                """Process a fraud prediction message."""
                self.processed_predictions.append(message)
                return {"status": "processed", "prediction_id": message["prediction_id"]}
        
        return MockMessageProcessor()
    
    def test_transaction_message_flow(self, message_processor):
        """Test complete transaction message processing flow."""
        # Create producer and consumer
        producer = MockKafkaProducer()
        consumer = MockKafkaConsumer(["transactions"])
        
        # Create transaction message
        transaction_data = sample_transactions[0]
        message = {
            "transaction_id": transaction_data["transaction_id"],
            "user_id": transaction_data["user_id"],
            "merchant_id": transaction_data["merchant_id"],
            "amount": transaction_data["amount"],
            "timestamp": transaction_data["timestamp"].isoformat(),
            "payment_method": transaction_data["payment_method"].value,
            "status": transaction_data["status"].value
        }
        
        # Send message
        producer.send("transactions", message["transaction_id"], message)
        
        # Simulate message delivery to consumer
        kafka_message = {
            "topic": "transactions",
            "key": message["transaction_id"],
            "value": message,
            "offset": 0,
            "partition": 0
        }
        consumer.add_message(kafka_message)
        
        # Consume and process message
        messages = consumer.poll(timeout_ms=1000, max_records=1)
        assert len(messages) == 1
        
        consumed_message = messages[0]
        result = message_processor.process_transaction(consumed_message["value"])
        
        # Verify processing
        assert result["status"] == "processed"
        assert result["transaction_id"] == message["transaction_id"]
        assert len(message_processor.processed_transactions) == 1
    
    def test_prediction_message_flow(self, message_processor):
        """Test complete fraud prediction message processing flow."""
        # Create producer and consumer
        producer = MockKafkaProducer()
        consumer = MockKafkaConsumer(["fraud_predictions"])
        
        # Create prediction message
        prediction_data = sample_predictions[0]
        message = {
            "prediction_id": prediction_data["prediction_id"],
            "transaction_id": prediction_data["transaction_id"],
            "user_id": prediction_data["user_id"],
            "fraud_probability": prediction_data["fraud_probability"],
            "risk_level": prediction_data["risk_level"].value,
            "decision": prediction_data["decision"],
            "confidence_score": prediction_data["confidence_score"],
            "model_version": prediction_data["model_version"]
        }
        
        # Send message
        producer.send("fraud_predictions", message["prediction_id"], message)
        
        # Simulate message delivery to consumer
        kafka_message = {
            "topic": "fraud_predictions",
            "key": message["prediction_id"],
            "value": message,
            "offset": 0,
            "partition": 0
        }
        consumer.add_message(kafka_message)
        
        # Consume and process message
        messages = consumer.poll(timeout_ms=1000, max_records=1)
        assert len(messages) == 1
        
        consumed_message = messages[0]
        result = message_processor.process_prediction(consumed_message["value"])
        
        # Verify processing
        assert result["status"] == "processed"
        assert result["prediction_id"] == message["prediction_id"]
        assert len(message_processor.processed_predictions) == 1
    
    def test_batch_message_processing(self, message_processor):
        """Test processing multiple messages in batch."""
        # Create producer and consumer
        producer = MockKafkaProducer()
        consumer = MockKafkaConsumer(["transactions", "fraud_predictions"])
        
        # Send multiple transaction messages
        for i, txn_data in enumerate(sample_transactions[:3]):
            message = {
                "transaction_id": txn_data["transaction_id"],
                "user_id": txn_data["user_id"],
                "amount": txn_data["amount"],
                "timestamp": txn_data["timestamp"].isoformat()
            }
            
            producer.send("transactions", message["transaction_id"], message)
            
            # Add to consumer
            kafka_message = {
                "topic": "transactions",
                "key": message["transaction_id"],
                "value": message,
                "offset": i,
                "partition": 0
            }
            consumer.add_message(kafka_message)
        
        # Consume and process all messages
        messages = consumer.poll(timeout_ms=1000, max_records=10)
        assert len(messages) == 3
        
        for message in messages:
            message_processor.process_transaction(message["value"])
        
        # Verify all messages were processed
        assert len(message_processor.processed_transactions) == 3
    
    def test_message_processing_error_handling(self, message_processor):
        """Test error handling during message processing."""
        # Create consumer with error message
        consumer = MockKafkaConsumer(["transactions"])
        
        # Add invalid message
        invalid_message = {
            "topic": "transactions",
            "key": "invalid_txn",
            "value": {"invalid": "data"},  # Missing required fields
            "offset": 0,
            "partition": 0
        }
        consumer.add_message(invalid_message)
        
        # Consume message
        messages = consumer.poll(timeout_ms=1000, max_records=1)
        assert len(messages) == 1
        
        # Process message (should handle gracefully)
        try:
            message_processor.process_transaction(messages[0]["value"])
            # If no exception, verify it was still recorded
            assert len(message_processor.processed_transactions) == 1
        except Exception as e:
            # Error handling should be implemented in real processor
            assert "invalid" in str(e).lower() or "missing" in str(e).lower()
    
    @pytest.mark.asyncio
    async def test_async_message_processing(self):
        """Test asynchronous message processing."""
        # Create async message processor
        class AsyncMessageProcessor:
            def __init__(self):
                self.processed_messages = []
            
            async def process_message_async(self, message: Dict[str, Any]):
                """Process message asynchronously."""
                # Simulate async processing
                await asyncio.sleep(0.01)
                self.processed_messages.append(message)
                return {"status": "processed", "message_id": message.get("transaction_id", "unknown")}
        
        processor = AsyncMessageProcessor()
        
        # Create test messages
        messages = [
            {"transaction_id": f"txn_{i}", "amount": i * 10}
            for i in range(5)
        ]
        
        # Process messages concurrently
        tasks = [
            processor.process_message_async(message)
            for message in messages
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Verify all messages were processed
        assert len(results) == 5
        assert len(processor.processed_messages) == 5
        
        for result in results:
            assert result["status"] == "processed"
            assert "txn_" in result["message_id"]


if __name__ == "__main__":
    pytest.main([__file__])