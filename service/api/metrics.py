"""Metrics endpoints for Prometheus monitoring."""

import logging
from typing import Dict, Any

from fastapi import APIRouter, Response, Depends
from prometheus_client import (
    Counter, Histogram, Gauge, Info,
    generate_latest, CONTENT_TYPE_LATEST,
    CollectorRegistry, REGISTRY
)

from service.core.config import get_settings, Settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Application-specific metrics
FRAUD_PREDICTIONS_TOTAL = Counter(
    'fraud_predictions_total',
    'Total number of fraud predictions made',
    ['model_version', 'risk_level', 'action']
)

FRAUD_PREDICTION_DURATION = Histogram(
    'fraud_prediction_duration_seconds',
    'Time spent on fraud prediction',
    ['model_version'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

FEATURE_COMPUTATION_DURATION = Histogram(
    'feature_computation_duration_seconds',
    'Time spent computing transaction features',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
)

ACTIVE_TRANSACTIONS = Gauge(
    'active_transactions_processing',
    'Number of transactions currently being processed'
)

MODEL_ACCURACY = Gauge(
    'model_accuracy_score',
    'Current model accuracy score',
    ['model_version']
)

MODEL_DRIFT_SCORE = Gauge(
    'model_drift_score',
    'Current model drift score',
    ['model_version', 'drift_type']
)

DATA_QUALITY_SCORE = Gauge(
    'data_quality_score',
    'Current data quality score',
    ['check_type']
)

KAFKA_LAG = Gauge(
    'kafka_consumer_lag',
    'Kafka consumer lag by topic and partition',
    ['topic', 'partition', 'consumer_group']
)

DATABASE_CONNECTIONS = Gauge(
    'database_connections_active',
    'Number of active database connections'
)

CACHE_HIT_RATE = Gauge(
    'cache_hit_rate',
    'Cache hit rate percentage',
    ['cache_type']
)

ERROR_RATE = Counter(
    'application_errors_total',
    'Total application errors',
    ['error_type', 'component']
)

# Application info
APP_INFO = Info(
    'fraud_detection_app',
    'Fraud detection application information'
)

# Initialize app info
APP_INFO.info({
    'version': '1.0.0',
    'environment': 'development',  # Will be updated from settings
    'service': 'fraud-detection-api'
})


class MetricsCollector:
    """Custom metrics collector for application-specific metrics."""
    
    def __init__(self):
        self.transaction_count = 0
        self.prediction_count = 0
        self.error_count = 0
    
    def record_transaction_processed(self):
        """Record a processed transaction."""
        self.transaction_count += 1
    
    def record_prediction_made(self, model_version: str, risk_level: str, action: str, duration: float):
        """Record a fraud prediction."""
        FRAUD_PREDICTIONS_TOTAL.labels(
            model_version=model_version,
            risk_level=risk_level,
            action=action
        ).inc()
        
        FRAUD_PREDICTION_DURATION.labels(
            model_version=model_version
        ).observe(duration)
        
        self.prediction_count += 1
    
    def record_feature_computation(self, duration: float):
        """Record feature computation time."""
        FEATURE_COMPUTATION_DURATION.observe(duration)
    
    def set_active_transactions(self, count: int):
        """Set the number of active transactions."""
        ACTIVE_TRANSACTIONS.set(count)
    
    def update_model_metrics(self, model_version: str, accuracy: float, drift_scores: Dict[str, float]):
        """Update model performance metrics."""
        MODEL_ACCURACY.labels(model_version=model_version).set(accuracy)
        
        for drift_type, score in drift_scores.items():
            MODEL_DRIFT_SCORE.labels(
                model_version=model_version,
                drift_type=drift_type
            ).set(score)
    
    def update_data_quality(self, quality_scores: Dict[str, float]):
        """Update data quality metrics."""
        for check_type, score in quality_scores.items():
            DATA_QUALITY_SCORE.labels(check_type=check_type).set(score)
    
    def update_kafka_lag(self, topic: str, partition: int, consumer_group: str, lag: int):
        """Update Kafka consumer lag."""
        KAFKA_LAG.labels(
            topic=topic,
            partition=str(partition),
            consumer_group=consumer_group
        ).set(lag)
    
    def set_database_connections(self, count: int):
        """Set active database connections count."""
        DATABASE_CONNECTIONS.set(count)
    
    def update_cache_hit_rate(self, cache_type: str, hit_rate: float):
        """Update cache hit rate."""
        CACHE_HIT_RATE.labels(cache_type=cache_type).set(hit_rate)
    
    def record_error(self, error_type: str, component: str):
        """Record an application error."""
        ERROR_RATE.labels(
            error_type=error_type,
            component=component
        ).inc()
        self.error_count += 1


# Global metrics collector instance
metrics_collector = MetricsCollector()


@router.get("/prometheus")
async def prometheus_metrics(settings: Settings = Depends(get_settings)):
    """Prometheus metrics endpoint."""
    # Update app info with current settings
    APP_INFO.info({
        'version': '1.0.0',
        'environment': settings.environment,
        'service': 'fraud-detection-api'
    })
    
    # Generate metrics in Prometheus format
    metrics_data = generate_latest(REGISTRY)
    
    return Response(
        content=metrics_data,
        media_type=CONTENT_TYPE_LATEST
    )


@router.get("/health-metrics")
async def health_metrics():
    """Application health metrics in JSON format."""
    return {
        "transactions_processed": metrics_collector.transaction_count,
        "predictions_made": metrics_collector.prediction_count,
        "errors_recorded": metrics_collector.error_count,
        "active_transactions": ACTIVE_TRANSACTIONS._value._value,
        "database_connections": DATABASE_CONNECTIONS._value._value,
    }


@router.get("/model-metrics")
async def model_metrics():
    """Model performance metrics in JSON format."""
    # Collect current model metrics
    model_metrics = {}
    
    # Get accuracy metrics for all model versions
    for sample in MODEL_ACCURACY.collect()[0].samples:
        model_version = sample.labels.get('model_version', 'unknown')
        model_metrics[f"{model_version}_accuracy"] = sample.value
    
    # Get drift metrics
    drift_metrics = {}
    for sample in MODEL_DRIFT_SCORE.collect()[0].samples:
        model_version = sample.labels.get('model_version', 'unknown')
        drift_type = sample.labels.get('drift_type', 'unknown')
        drift_metrics[f"{model_version}_{drift_type}_drift"] = sample.value
    
    return {
        "model_performance": model_metrics,
        "drift_scores": drift_metrics,
        "timestamp": "2024-01-15T14:30:00Z"  # This would be dynamic
    }


@router.get("/system-metrics")
async def system_metrics():
    """System and infrastructure metrics."""
    import psutil
    import os
    
    return {
        "system": {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "process_id": os.getpid(),
            "thread_count": psutil.Process().num_threads(),
        },
        "application": {
            "active_transactions": ACTIVE_TRANSACTIONS._value._value,
            "database_connections": DATABASE_CONNECTIONS._value._value,
        },
        "cache": {
            # This would be populated with actual cache metrics
            "redis_hit_rate": 0.95,
            "feature_cache_hit_rate": 0.88,
        }
    }


@router.post("/reset")
async def reset_metrics():
    """Reset application metrics (for testing/development only)."""
    settings = get_settings()
    
    if settings.environment == "production":
        return {"error": "Metrics reset not allowed in production"}
    
    # Reset counters (note: this doesn't actually reset Prometheus counters)
    metrics_collector.transaction_count = 0
    metrics_collector.prediction_count = 0
    metrics_collector.error_count = 0
    
    # Reset gauges
    ACTIVE_TRANSACTIONS.set(0)
    DATABASE_CONNECTIONS.set(0)
    
    logger.info("Application metrics reset")
    
    return {"message": "Metrics reset successfully"}


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    return metrics_collector