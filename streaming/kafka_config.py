"""Kafka configuration and connection management."""

import asyncio
from functools import lru_cache
from typing import Dict, Any, List, Optional

from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from aiokafka.errors import KafkaError
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError
from pydantic import BaseModel, Field

from service.core.config import Settings, get_settings
from service.core.logging import get_logger

logger = get_logger("fraud_detection.kafka")


class KafkaConfig(BaseModel):
    """Kafka configuration settings."""
    
    # Connection settings
    bootstrap_servers: List[str] = Field(default=["localhost:9092"])
    security_protocol: str = Field(default="PLAINTEXT")
    sasl_mechanism: Optional[str] = None
    sasl_username: Optional[str] = None
    sasl_password: Optional[str] = None
    ssl_cafile: Optional[str] = None
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None
    
    # Topic settings
    topic_transactions: str = Field(default="transactions")
    topic_fraud_scores: str = Field(default="fraud_scores")
    topic_alerts: str = Field(default="fraud_alerts")
    topic_feedback: str = Field(default="fraud_feedback")
    
    # Consumer settings
    consumer_group_id: str = Field(default="fraud_detection")
    auto_offset_reset: str = Field(default="latest")
    enable_auto_commit: bool = Field(default=True)
    auto_commit_interval_ms: int = Field(default=1000)
    session_timeout_ms: int = Field(default=30000)
    heartbeat_interval_ms: int = Field(default=10000)
    max_poll_records: int = Field(default=500)
    max_poll_interval_ms: int = Field(default=300000)
    
    # Producer settings
    acks: str = Field(default="all")
    retries: int = Field(default=3)
    batch_size: int = Field(default=16384)
    linger_ms: int = Field(default=10)
    buffer_memory: int = Field(default=33554432)
    compression_type: str = Field(default="gzip")
    max_request_size: int = Field(default=1048576)
    
    # Topic configuration
    topic_num_partitions: int = Field(default=3)
    topic_replication_factor: int = Field(default=1)
    topic_retention_ms: int = Field(default=604800000)  # 7 days
    
    # Performance settings
    fetch_min_bytes: int = Field(default=1)
    fetch_max_wait_ms: int = Field(default=500)
    max_partition_fetch_bytes: int = Field(default=1048576)
    
    @property
    def producer_config(self) -> Dict[str, Any]:
        """Get producer configuration dictionary."""
        config = {
            "bootstrap_servers": self.bootstrap_servers,
            "acks": self.acks,
            "retries": self.retries,
            "batch_size": self.batch_size,
            "linger_ms": self.linger_ms,
            "buffer_memory": self.buffer_memory,
            "compression_type": self.compression_type,
            "max_request_size": self.max_request_size,
            "security_protocol": self.security_protocol,
        }
        
        if self.sasl_mechanism:
            config.update({
                "sasl_mechanism": self.sasl_mechanism,
                "sasl_plain_username": self.sasl_username,
                "sasl_plain_password": self.sasl_password,
            })
        
        if self.security_protocol == "SSL":
            config.update({
                "ssl_cafile": self.ssl_cafile,
                "ssl_certfile": self.ssl_certfile,
                "ssl_keyfile": self.ssl_keyfile,
            })
        
        return config
    
    @property
    def consumer_config(self) -> Dict[str, Any]:
        """Get consumer configuration dictionary."""
        config = {
            "bootstrap_servers": self.bootstrap_servers,
            "group_id": self.consumer_group_id,
            "auto_offset_reset": self.auto_offset_reset,
            "enable_auto_commit": self.enable_auto_commit,
            "auto_commit_interval_ms": self.auto_commit_interval_ms,
            "session_timeout_ms": self.session_timeout_ms,
            "heartbeat_interval_ms": self.heartbeat_interval_ms,
            "max_poll_records": self.max_poll_records,
            "max_poll_interval_ms": self.max_poll_interval_ms,
            "fetch_min_bytes": self.fetch_min_bytes,
            "fetch_max_wait_ms": self.fetch_max_wait_ms,
            "max_partition_fetch_bytes": self.max_partition_fetch_bytes,
            "security_protocol": self.security_protocol,
        }
        
        if self.sasl_mechanism:
            config.update({
                "sasl_mechanism": self.sasl_mechanism,
                "sasl_plain_username": self.sasl_username,
                "sasl_plain_password": self.sasl_password,
            })
        
        if self.security_protocol == "SSL":
            config.update({
                "ssl_cafile": self.ssl_cafile,
                "ssl_certfile": self.ssl_certfile,
                "ssl_keyfile": self.ssl_keyfile,
            })
        
        return config
    
    @property
    def admin_config(self) -> Dict[str, Any]:
        """Get admin client configuration dictionary."""
        config = {
            "bootstrap_servers": self.bootstrap_servers,
            "security_protocol": self.security_protocol,
        }
        
        if self.sasl_mechanism:
            config.update({
                "sasl_mechanism": self.sasl_mechanism,
                "sasl_plain_username": self.sasl_username,
                "sasl_plain_password": self.sasl_password,
            })
        
        if self.security_protocol == "SSL":
            config.update({
                "ssl_cafile": self.ssl_cafile,
                "ssl_certfile": self.ssl_certfile,
                "ssl_keyfile": self.ssl_keyfile,
            })
        
        return config


class KafkaManager:
    """Kafka connection and topic management."""
    
    def __init__(self, config: KafkaConfig):
        self.config = config
        self._producer: Optional[AIOKafkaProducer] = None
        self._consumers: Dict[str, AIOKafkaConsumer] = {}
        self._admin_client: Optional[KafkaAdminClient] = None
    
    async def get_producer(self) -> AIOKafkaProducer:
        """Get or create Kafka producer."""
        if self._producer is None:
            self._producer = AIOKafkaProducer(**self.config.producer_config)
            await self._producer.start()
            logger.info("Kafka producer started")
        
        return self._producer
    
    async def get_consumer(self, topics: List[str], group_id: Optional[str] = None) -> AIOKafkaConsumer:
        """Get or create Kafka consumer for specific topics."""
        consumer_key = f"{group_id or self.config.consumer_group_id}_{','.join(sorted(topics))}"
        
        if consumer_key not in self._consumers:
            consumer_config = self.config.consumer_config.copy()
            if group_id:
                consumer_config["group_id"] = group_id
            
            consumer = AIOKafkaConsumer(*topics, **consumer_config)
            await consumer.start()
            
            self._consumers[consumer_key] = consumer
            logger.info(f"Kafka consumer started for topics: {topics}")
        
        return self._consumers[consumer_key]
    
    def get_admin_client(self) -> KafkaAdminClient:
        """Get or create Kafka admin client."""
        if self._admin_client is None:
            self._admin_client = KafkaAdminClient(**self.config.admin_config)
            logger.info("Kafka admin client created")
        
        return self._admin_client
    
    async def create_topics(self) -> None:
        """Create required Kafka topics if they don't exist."""
        admin_client = self.get_admin_client()
        
        topics_to_create = [
            NewTopic(
                name=self.config.topic_transactions,
                num_partitions=self.config.topic_num_partitions,
                replication_factor=self.config.topic_replication_factor,
                topic_configs={
                    "retention.ms": str(self.config.topic_retention_ms),
                    "compression.type": self.config.compression_type,
                }
            ),
            NewTopic(
                name=self.config.topic_fraud_scores,
                num_partitions=self.config.topic_num_partitions,
                replication_factor=self.config.topic_replication_factor,
                topic_configs={
                    "retention.ms": str(self.config.topic_retention_ms),
                    "compression.type": self.config.compression_type,
                }
            ),
            NewTopic(
                name=self.config.topic_alerts,
                num_partitions=self.config.topic_num_partitions,
                replication_factor=self.config.topic_replication_factor,
                topic_configs={
                    "retention.ms": str(self.config.topic_retention_ms),
                    "compression.type": self.config.compression_type,
                }
            ),
            NewTopic(
                name=self.config.topic_feedback,
                num_partitions=self.config.topic_num_partitions,
                replication_factor=self.config.topic_replication_factor,
                topic_configs={
                    "retention.ms": str(self.config.topic_retention_ms),
                    "compression.type": self.config.compression_type,
                }
            ),
        ]
        
        try:
            result = admin_client.create_topics(topics_to_create, validate_only=False)
            
            # Wait for topic creation
            for topic, future in result.topic_errors.items():
                try:
                    future.result()  # This will raise an exception if creation failed
                    logger.info(f"Topic created successfully: {topic}")
                except TopicAlreadyExistsError:
                    logger.info(f"Topic already exists: {topic}")
                except Exception as e:
                    logger.error(f"Failed to create topic {topic}: {e}")
                    raise
        
        except Exception as e:
            logger.error(f"Failed to create topics: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check Kafka cluster health."""
        try:
            # Try to get cluster metadata
            admin_client = self.get_admin_client()
            metadata = admin_client.describe_cluster()
            
            if metadata:
                logger.debug("Kafka cluster is healthy")
                return True
            
        except Exception as e:
            logger.error(f"Kafka health check failed: {e}")
        
        return False
    
    async def close(self) -> None:
        """Close all Kafka connections."""
        # Close producer
        if self._producer:
            await self._producer.stop()
            self._producer = None
            logger.info("Kafka producer stopped")
        
        # Close consumers
        for consumer_key, consumer in self._consumers.items():
            await consumer.stop()
            logger.info(f"Kafka consumer stopped: {consumer_key}")
        
        self._consumers.clear()
        
        # Close admin client
        if self._admin_client:
            self._admin_client.close()
            self._admin_client = None
            logger.info("Kafka admin client closed")


@lru_cache()
def get_kafka_config() -> KafkaConfig:
    """Get Kafka configuration from application settings."""
    settings = get_settings()
    
    return KafkaConfig(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        topic_transactions=settings.kafka_topic_transactions,
        topic_fraud_scores=settings.kafka_topic_scores,
        consumer_group_id=settings.kafka_consumer_group,
        auto_offset_reset=settings.kafka_auto_offset_reset,
    )


# Global Kafka manager instance
_kafka_manager: Optional[KafkaManager] = None


async def get_kafka_manager() -> KafkaManager:
    """Get global Kafka manager instance."""
    global _kafka_manager
    
    if _kafka_manager is None:
        config = get_kafka_config()
        _kafka_manager = KafkaManager(config)
        
        # Create topics on first initialization
        await _kafka_manager.create_topics()
    
    return _kafka_manager


async def close_kafka_manager() -> None:
    """Close global Kafka manager."""
    global _kafka_manager
    
    if _kafka_manager:
        await _kafka_manager.close()
        _kafka_manager = None