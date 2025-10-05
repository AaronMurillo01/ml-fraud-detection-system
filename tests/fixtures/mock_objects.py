"""Mock objects for testing."""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from unittest.mock import Mock, AsyncMock, MagicMock
import numpy as np
import pandas as pd

from service.ml_inference import (
    MLInferenceService, InferenceRequest, InferenceResponse,
    BatchInferenceRequest, BatchInferenceResponse
)
from service.model_loader import ModelLoader, ModelMetadata, ModelPrediction, LoadedModel
from service.models import (
    EnrichedTransaction, PredictionResult, ModelPerformanceMetrics,
    FeatureImportance, BatchPredictionResult
)
from shared.models import Transaction, User, Merchant, FraudPrediction, RiskLevel
from database.models import TransactionModel, UserModel, MerchantModel, FraudPredictionModel


class MockMLInferenceService:
    """Mock ML inference service for testing."""
    
    def __init__(self):
        self.predict_calls = []
        self.batch_predict_calls = []
        self.model_info_calls = []
        self.service_stats_calls = []
        self._default_prediction = PredictionResult(
            transaction_id="mock_txn_123",
            fraud_probability=0.25,
            risk_level=RiskLevel.LOW,
            decision="APPROVE",
            confidence_score=0.85,
            feature_contributions={
                "amount": 0.15,
                "merchant_risk": 0.30,
                "velocity": 0.25,
                "behavioral": 0.30
            },
            model_version="v1.0.0",
            processing_time_ms=32.5,
            explanation="Mock prediction result"
        )
    
    async def predict(self, request: InferenceRequest) -> InferenceResponse:
        """Mock predict method."""
        self.predict_calls.append(request)
        
        # Simulate different responses based on transaction amount
        if hasattr(request, 'transaction') and request.transaction.amount > 1000:
            # High amount = high risk
            prediction = PredictionResult(
                transaction_id=request.transaction.transaction_id,
                fraud_probability=0.85,
                risk_level=RiskLevel.HIGH,
                decision="DECLINE",
                confidence_score=0.92,
                model_version="v1.0.0",
                processing_time_ms=28.3
            )
        elif hasattr(request, 'transaction') and request.transaction.amount < 10:
            # Very low amount = very low risk
            prediction = PredictionResult(
                transaction_id=request.transaction.transaction_id,
                fraud_probability=0.05,
                risk_level=RiskLevel.VERY_LOW,
                decision="APPROVE",
                confidence_score=0.95,
                model_version="v1.0.0",
                processing_time_ms=15.2
            )
        else:
            # Default prediction
            prediction = self._default_prediction
            if hasattr(request, 'transaction'):
                prediction.transaction_id = request.transaction.transaction_id
        
        return InferenceResponse(
            success=True,
            transaction_id=prediction.transaction_id,
            prediction=prediction.dict(),
            processing_time_ms=prediction.processing_time_ms,
            model_version=prediction.model_version,
            timestamp=datetime.utcnow()
        )
    
    async def predict_batch(self, request: BatchInferenceRequest) -> BatchInferenceResponse:
        """Mock batch predict method."""
        self.batch_predict_calls.append(request)
        
        predictions = []
        total_time = 0.0
        
        for transaction in request.transactions:
            # Create individual inference request
            individual_request = InferenceRequest(
                transaction=transaction,
                model_name=request.model_name,
                model_version=request.model_version,
                include_feature_importance=request.include_feature_importance
            )
            
            # Get prediction
            response = await self.predict(individual_request)
            predictions.append(response)
            total_time += response.processing_time_ms
        
        return BatchInferenceResponse(
            batch_size=len(request.transactions),
            predictions=predictions,
            success_count=len(predictions),
            error_count=0,
            total_processing_time_ms=total_time,
            average_processing_time_ms=total_time / len(predictions) if predictions else 0,
            timestamp=datetime.utcnow()
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Mock get model info method."""
        self.model_info_calls.append(datetime.utcnow())
        
        return {
            "model_name": "mock_fraud_detector",
            "model_version": "v1.0.0",
            "model_type": "xgboost",
            "feature_count": 25,
            "training_date": (datetime.utcnow() - timedelta(days=7)).isoformat(),
            "performance_metrics": {
                "accuracy": 0.95,
                "precision": 0.88,
                "recall": 0.92,
                "f1_score": 0.90,
                "auc_roc": 0.96
            },
            "feature_importance": {
                "amount": 0.25,
                "merchant_risk_score": 0.20,
                "velocity_features": 0.18,
                "behavioral_features": 0.15,
                "contextual_features": 0.12,
                "user_risk_score": 0.10
            }
        }
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Mock get service stats method."""
        self.service_stats_calls.append(datetime.utcnow())
        
        return {
            "total_requests": len(self.predict_calls) + len(self.batch_predict_calls),
            "successful_requests": len(self.predict_calls) + len(self.batch_predict_calls),
            "failed_requests": 0,
            "average_response_time_ms": 32.5,
            "requests_per_minute": 125,
            "error_rate": 0.0,
            "cache_hit_rate": 0.85,
            "active_models": 1,
            "uptime_seconds": 3600
        }
    
    def reset_calls(self):
        """Reset call tracking."""
        self.predict_calls.clear()
        self.batch_predict_calls.clear()
        self.model_info_calls.clear()
        self.service_stats_calls.clear()


class MockModelLoader:
    """Mock model loader for testing."""
    
    def __init__(self):
        self.load_calls = []
        self.cache_calls = []
        self.metadata_calls = []
        self._models = {}
        self._metadata_cache = {}
    
    async def load_model(self, model_name: str, model_version: str = "latest") -> LoadedModel:
        """Mock load model method."""
        self.load_calls.append((model_name, model_version))
        
        # Create mock model
        mock_model = Mock()
        mock_model.predict = Mock(return_value=np.array([0.25]))
        mock_model.predict_proba = Mock(return_value=np.array([[0.75, 0.25]]))
        
        # Create mock preprocessor
        mock_preprocessor = Mock()
        mock_preprocessor.transform = Mock(return_value=np.array([[1, 2, 3, 4, 5]]))
        
        loaded_model = LoadedModel(
            model=mock_model,
            preprocessor=mock_preprocessor,
            metadata=ModelMetadata(
                model_name=model_name,
                model_version=model_version,
                model_type="xgboost",
                feature_names=["amount", "merchant_risk", "velocity", "behavioral", "contextual"],
                model_path=f"/mock/path/{model_name}_{model_version}.pkl",
                created_at=datetime.utcnow(),
                performance_metrics={
                    "accuracy": 0.95,
                    "precision": 0.88,
                    "recall": 0.92,
                    "f1_score": 0.90,
                    "auc_roc": 0.96
                }
            ),
            loaded_at=datetime.utcnow()
        )
        
        self._models[f"{model_name}_{model_version}"] = loaded_model
        return loaded_model
    
    def get_cached_model(self, model_name: str, model_version: str = "latest") -> Optional[LoadedModel]:
        """Mock get cached model method."""
        self.cache_calls.append((model_name, model_version))
        return self._models.get(f"{model_name}_{model_version}")
    
    async def get_model_metadata(self, model_name: str, model_version: str = "latest") -> ModelMetadata:
        """Mock get model metadata method."""
        self.metadata_calls.append((model_name, model_version))
        
        cache_key = f"{model_name}_{model_version}"
        if cache_key in self._metadata_cache:
            return self._metadata_cache[cache_key]
        
        metadata = ModelMetadata(
            model_name=model_name,
            model_version=model_version,
            model_type="xgboost",
            feature_names=["amount", "merchant_risk", "velocity", "behavioral", "contextual"],
            model_path=f"/mock/path/{model_name}_{model_version}.pkl",
            created_at=datetime.utcnow(),
            performance_metrics={
                "accuracy": 0.95,
                "precision": 0.88,
                "recall": 0.92,
                "f1_score": 0.90,
                "auc_roc": 0.96
            }
        )
        
        self._metadata_cache[cache_key] = metadata
        return metadata
    
    def clear_cache(self):
        """Mock clear cache method."""
        self._models.clear()
        self._metadata_cache.clear()
    
    def reset_calls(self):
        """Reset call tracking."""
        self.load_calls.clear()
        self.cache_calls.clear()
        self.metadata_calls.clear()


class MockKafkaProducer:
    """Mock Kafka producer for testing."""
    
    def __init__(self):
        self.sent_messages = []
        self.send_calls = []
        self.flush_calls = []
        self.close_calls = []
    
    async def send(self, topic: str, value: Union[str, bytes, Dict], key: Optional[str] = None):
        """Mock send method."""
        message = {
            "topic": topic,
            "value": value,
            "key": key,
            "timestamp": datetime.utcnow(),
            "partition": 0,
            "offset": len(self.sent_messages)
        }
        
        self.sent_messages.append(message)
        self.send_calls.append((topic, value, key))
        
        # Return mock future
        future = asyncio.Future()
        future.set_result(message)
        return future
    
    async def flush(self):
        """Mock flush method."""
        self.flush_calls.append(datetime.utcnow())
    
    async def close(self):
        """Mock close method."""
        self.close_calls.append(datetime.utcnow())
    
    def reset(self):
        """Reset all tracking."""
        self.sent_messages.clear()
        self.send_calls.clear()
        self.flush_calls.clear()
        self.close_calls.clear()


class MockKafkaConsumer:
    """Mock Kafka consumer for testing."""
    
    def __init__(self, topics: List[str]):
        self.topics = topics
        self.consumed_messages = []
        self.poll_calls = []
        self.commit_calls = []
        self.close_calls = []
        self._message_queue = []
    
    def add_message(self, topic: str, value: Union[str, bytes, Dict], key: Optional[str] = None):
        """Add a message to the mock queue."""
        message = {
            "topic": topic,
            "value": value,
            "key": key,
            "timestamp": datetime.utcnow(),
            "partition": 0,
            "offset": len(self._message_queue)
        }
        self._message_queue.append(message)
    
    async def poll(self, timeout_ms: int = 1000) -> List[Dict]:
        """Mock poll method."""
        self.poll_calls.append((timeout_ms, datetime.utcnow()))
        
        # Return available messages
        messages = self._message_queue.copy()
        self.consumed_messages.extend(messages)
        self._message_queue.clear()
        
        return messages
    
    async def commit(self):
        """Mock commit method."""
        self.commit_calls.append(datetime.utcnow())
    
    async def close(self):
        """Mock close method."""
        self.close_calls.append(datetime.utcnow())
    
    def reset(self):
        """Reset all tracking."""
        self.consumed_messages.clear()
        self.poll_calls.clear()
        self.commit_calls.clear()
        self.close_calls.clear()
        self._message_queue.clear()


class MockDatabase:
    """Mock database for testing."""
    
    def __init__(self):
        self.transactions = {}
        self.users = {}
        self.merchants = {}
        self.predictions = {}
        self.query_calls = []
        self.insert_calls = []
        self.update_calls = []
        self.delete_calls = []
    
    async def get_transaction(self, transaction_id: str) -> Optional[TransactionModel]:
        """Mock get transaction method."""
        self.query_calls.append(("get_transaction", transaction_id))
        return self.transactions.get(transaction_id)
    
    async def create_transaction(self, transaction_data: Dict) -> TransactionModel:
        """Mock create transaction method."""
        self.insert_calls.append(("create_transaction", transaction_data))
        
        transaction = TransactionModel(
            transaction_id=transaction_data["transaction_id"],
            user_id=transaction_data["user_id"],
            merchant_id=transaction_data["merchant_id"],
            amount=transaction_data["amount"],
            timestamp=transaction_data.get("timestamp", datetime.utcnow()),
            payment_method=transaction_data["payment_method"],
            transaction_type=transaction_data["transaction_type"],
            status=transaction_data["status"]
        )
        
        self.transactions[transaction.transaction_id] = transaction
        return transaction
    
    async def get_user(self, user_id: str) -> Optional[UserModel]:
        """Mock get user method."""
        self.query_calls.append(("get_user", user_id))
        return self.users.get(user_id)
    
    async def create_user(self, user_data: Dict) -> UserModel:
        """Mock create user method."""
        self.insert_calls.append(("create_user", user_data))
        
        user = UserModel(
            user_id=user_data["user_id"],
            email=user_data["email"],
            phone_number=user_data.get("phone_number"),
            registration_date=user_data.get("registration_date", datetime.utcnow()),
            risk_score=user_data.get("risk_score", 0.5),
            status=user_data.get("status", "active")
        )
        
        self.users[user.user_id] = user
        return user
    
    async def get_merchant(self, merchant_id: str) -> Optional[MerchantModel]:
        """Mock get merchant method."""
        self.query_calls.append(("get_merchant", merchant_id))
        return self.merchants.get(merchant_id)
    
    async def create_merchant(self, merchant_data: Dict) -> MerchantModel:
        """Mock create merchant method."""
        self.insert_calls.append(("create_merchant", merchant_data))
        
        merchant = MerchantModel(
            merchant_id=merchant_data["merchant_id"],
            merchant_name=merchant_data["merchant_name"],
            category=merchant_data["category"],
            location=merchant_data.get("location"),
            risk_score=merchant_data.get("risk_score", 0.5),
            status=merchant_data.get("status", "active")
        )
        
        self.merchants[merchant.merchant_id] = merchant
        return merchant
    
    async def save_prediction(self, prediction_data: Dict) -> FraudPredictionModel:
        """Mock save prediction method."""
        self.insert_calls.append(("save_prediction", prediction_data))
        
        prediction = FraudPredictionModel(
            prediction_id=f"pred_{len(self.predictions) + 1}",
            transaction_id=prediction_data["transaction_id"],
            user_id=prediction_data["user_id"],
            fraud_probability=prediction_data["fraud_probability"],
            risk_level=prediction_data["risk_level"],
            decision=prediction_data["decision"],
            confidence_score=prediction_data["confidence_score"],
            model_version=prediction_data["model_version"],
            model_features=prediction_data.get("model_features", {}),
            created_at=datetime.utcnow()
        )
        
        self.predictions[prediction.prediction_id] = prediction
        return prediction
    
    def reset(self):
        """Reset all data and tracking."""
        self.transactions.clear()
        self.users.clear()
        self.merchants.clear()
        self.predictions.clear()
        self.query_calls.clear()
        self.insert_calls.clear()
        self.update_calls.clear()
        self.delete_calls.clear()


class MockFeaturePipeline:
    """Mock feature pipeline for testing."""
    
    def __init__(self):
        self.enrich_calls = []
        self.extract_calls = []
    
    async def enrich_transaction(self, transaction: Transaction) -> EnrichedTransaction:
        """Mock enrich transaction method."""
        self.enrich_calls.append(transaction)
        
        # Create enriched transaction with mock features
        enriched = EnrichedTransaction(
            **transaction.dict(),
            velocity_features={
                "txn_count_1h": 3,
                "txn_count_24h": 12,
                "amount_sum_1h": transaction.amount * 2,
                "amount_sum_24h": transaction.amount * 8
            },
            risk_features={
                "merchant_risk_score": 0.3,
                "user_risk_score": 0.2,
                "location_risk_score": 0.1
            },
            behavioral_features={
                "avg_transaction_amount": transaction.amount * 0.8,
                "transaction_frequency": 2.5
            },
            contextual_features={
                "is_weekend": datetime.now().weekday() >= 5,
                "hour_of_day": datetime.now().hour,
                "day_of_week": datetime.now().weekday()
            }
        )
        
        return enriched
    
    def extract_features(self, enriched_transaction: EnrichedTransaction) -> np.ndarray:
        """Mock extract features method."""
        self.extract_calls.append(enriched_transaction)
        
        # Return mock feature vector
        return np.array([
            enriched_transaction.amount,
            enriched_transaction.risk_features.get("merchant_risk_score", 0.5),
            enriched_transaction.velocity_features.get("txn_count_1h", 1),
            enriched_transaction.behavioral_features.get("avg_transaction_amount", 100),
            enriched_transaction.contextual_features.get("hour_of_day", 12)
        ])
    
    def reset_calls(self):
        """Reset call tracking."""
        self.enrich_calls.clear()
        self.extract_calls.clear()


# Factory functions for creating mock objects
def create_mock_inference_service() -> MockMLInferenceService:
    """Create a mock ML inference service."""
    return MockMLInferenceService()


def create_mock_model_loader() -> MockModelLoader:
    """Create a mock model loader."""
    return MockModelLoader()


def create_mock_kafka_producer() -> MockKafkaProducer:
    """Create a mock Kafka producer."""
    return MockKafkaProducer()


def create_mock_kafka_consumer(topics: List[str]) -> MockKafkaConsumer:
    """Create a mock Kafka consumer."""
    return MockKafkaConsumer(topics)


def create_mock_database() -> MockDatabase:
    """Create a mock database."""
    return MockDatabase()


def create_mock_feature_pipeline() -> MockFeaturePipeline:
    """Create a mock feature pipeline."""
    return MockFeaturePipeline()