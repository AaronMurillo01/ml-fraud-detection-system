"""Streaming components for real-time fraud detection."""

from .kafka_config import (
    KafkaConfig,
    KafkaManager,
    get_kafka_config,
    get_kafka_manager,
    close_kafka_manager
)
from .producer import (
    KafkaMessage,
    TransactionMessage,
    ScoreMessage,
    AlertMessage,
    FeedbackMessage,
    FraudDetectionProducer,
    get_fraud_producer,
    close_fraud_producer
)
from .consumer import (
    MessageHandler,
    TransactionHandler,
    ScoreHandler,
    AlertHandler,
    FeedbackHandler,
    FraudDetectionConsumer,
    StreamProcessor,
    get_stream_processor,
    close_stream_processor
)
from .stream_processor import (
    ProcessingMetrics,
    TransactionEnricher,
    FraudDetectionStreamProcessor,
    get_fraud_stream_processor,
    start_fraud_detection_pipeline,
    stop_fraud_detection_pipeline
)

__all__ = [
    # Configuration
    "KafkaConfig",
    "KafkaManager",
    "get_kafka_config",
    "get_kafka_manager",
    "close_kafka_manager",
    
    # Producer
    "KafkaMessage",
    "TransactionMessage",
    "ScoreMessage",
    "AlertMessage",
    "FeedbackMessage",
    "FraudDetectionProducer",
    "get_fraud_producer",
    "close_fraud_producer",
    
    # Consumer
    "MessageHandler",
    "TransactionHandler",
    "ScoreHandler",
    "AlertHandler",
    "FeedbackHandler",
    "FraudDetectionConsumer",
    "StreamProcessor",
    "get_stream_processor",
    "close_stream_processor",
    
    # Stream Processor
    "ProcessingMetrics",
    "TransactionEnricher",
    "FraudDetectionStreamProcessor",
    "get_fraud_stream_processor",
    "start_fraud_detection_pipeline",
    "stop_fraud_detection_pipeline",
]