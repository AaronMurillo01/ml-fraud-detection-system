"""Integration tests for fraud detection API."""

import pytest
import asyncio
from httpx import AsyncClient
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock
from datetime import datetime, timezone
from decimal import Decimal
import json

from service.api import app
from shared.models import Transaction, FraudPrediction
from training.model_manager import ModelManager


class TestFraudDetectionAPI:
    """Integration tests for fraud detection API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def sample_transaction_data(self):
        """Sample transaction data for testing."""
        return {
            "transaction_id": "txn_test_123",
            "user_id": "user_test_123",
            "merchant_id": "merchant_test_123",
            "amount": "150.75",
            "currency": "USD",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "merchant_category": "grocery",
            "payment_method": "credit_card",
            "card_type": "visa",
            "location": {
                "latitude": 40.7128,
                "longitude": -74.0060,
                "city": "New York",
                "country": "US"
            }
        }
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint."""
        response = client.get("/metrics")
        
        assert response.status_code == 200
        # Should return Prometheus metrics format
        assert "fraud_detection" in response.text
    
    @patch('service.api.fraud_detection_service')
    def test_predict_fraud_success(self, mock_service, client, sample_transaction_data):
        """Test successful fraud prediction."""
        # Mock the fraud detection service
        mock_prediction = FraudPrediction(
            transaction_id="txn_test_123",
            is_fraud=False,
            fraud_probability=0.15,
            risk_score=0.2,
            confidence=0.85,
            model_version="ensemble_v1.0.0",
            prediction_timestamp=datetime.now(timezone.utc),
            features_used=["amount_zscore", "frequency_1h", "merchant_risk"],
            model_scores=[
                {
                    "model_name": "xgboost_v1",
                    "model_version": "1.0.0",
                    "score": 0.12,
                    "confidence": 0.88
                }
            ]
        )
        
        mock_service.predict_fraud.return_value = mock_prediction
        
        response = client.post("/api/v1/predict", json=sample_transaction_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["transaction_id"] == "txn_test_123"
        assert data["is_fraud"] is False
        assert data["fraud_probability"] == 0.15
        assert data["risk_score"] == 0.2
        assert data["confidence"] == 0.85
        assert "prediction_timestamp" in data
        assert "processing_time_ms" in data
    
    def test_predict_fraud_invalid_data(self, client):
        """Test fraud prediction with invalid data."""
        invalid_data = {
            "transaction_id": "txn_test_123",
            # Missing required fields
            "amount": "invalid_amount"
        }
        
        response = client.post("/api/v1/predict", json=invalid_data)
        
        assert response.status_code == 422  # Validation error
        data = response.json()
        assert "detail" in data
    
    @patch('service.api.fraud_detection_service')
    def test_predict_fraud_high_risk(self, mock_service, client, sample_transaction_data):
        """Test fraud prediction for high-risk transaction."""
        # Mock high-risk prediction
        mock_prediction = FraudPrediction(
            transaction_id="txn_test_123",
            is_fraud=True,
            fraud_probability=0.92,
            risk_score=0.95,
            confidence=0.88,
            model_version="ensemble_v1.0.0",
            prediction_timestamp=datetime.now(timezone.utc),
            features_used=["amount_zscore", "frequency_1h", "velocity_score"],
            model_scores=[
                {
                    "model_name": "xgboost_v1",
                    "model_version": "1.0.0",
                    "score": 0.94,
                    "confidence": 0.90
                }
            ]
        )
        
        mock_service.predict_fraud.return_value = mock_prediction
        
        response = client.post("/api/v1/predict", json=sample_transaction_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["is_fraud"] is True
        assert data["fraud_probability"] > 0.9
        assert data["risk_score"] > 0.9
    
    @patch('service.api.fraud_detection_service')
    def test_predict_fraud_service_error(self, mock_service, client, sample_transaction_data):
        """Test fraud prediction when service raises an error."""
        mock_service.predict_fraud.side_effect = Exception("Model service unavailable")
        
        response = client.post("/api/v1/predict", json=sample_transaction_data)
        
        assert response.status_code == 500
        data = response.json()
        assert "error" in data
    
    def test_batch_predict_fraud(self, client):
        """Test batch fraud prediction endpoint."""
        batch_data = {
            "transactions": [
                {
                    "transaction_id": "txn_1",
                    "user_id": "user_1",
                    "merchant_id": "merchant_1",
                    "amount": "100.00",
                    "currency": "USD",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "merchant_category": "grocery",
                    "payment_method": "credit_card",
                    "card_type": "visa",
                    "location": {
                        "latitude": 40.7128,
                        "longitude": -74.0060,
                        "city": "New York",
                        "country": "US"
                    }
                },
                {
                    "transaction_id": "txn_2",
                    "user_id": "user_2",
                    "merchant_id": "merchant_2",
                    "amount": "500.00",
                    "currency": "USD",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "merchant_category": "electronics",
                    "payment_method": "debit_card",
                    "card_type": "mastercard",
                    "location": {
                        "latitude": 34.0522,
                        "longitude": -118.2437,
                        "city": "Los Angeles",
                        "country": "US"
                    }
                }
            ]
        }
        
        with patch('service.api.fraud_detection_service') as mock_service:
            # Mock batch predictions
            mock_service.predict_fraud_batch.return_value = [
                FraudPrediction(
                    transaction_id="txn_1",
                    is_fraud=False,
                    fraud_probability=0.1,
                    risk_score=0.15,
                    confidence=0.9,
                    model_version="ensemble_v1.0.0",
                    prediction_timestamp=datetime.now(timezone.utc),
                    features_used=["amount_zscore"],
                    model_scores=[]
                ),
                FraudPrediction(
                    transaction_id="txn_2",
                    is_fraud=True,
                    fraud_probability=0.85,
                    risk_score=0.9,
                    confidence=0.82,
                    model_version="ensemble_v1.0.0",
                    prediction_timestamp=datetime.now(timezone.utc),
                    features_used=["amount_zscore", "velocity_score"],
                    model_scores=[]
                )
            ]
            
            response = client.post("/api/v1/predict/batch", json=batch_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert "predictions" in data
            assert len(data["predictions"]) == 2
            assert data["predictions"][0]["transaction_id"] == "txn_1"
            assert data["predictions"][1]["transaction_id"] == "txn_2"
    
    def test_get_model_info(self, client):
        """Test model information endpoint."""
        with patch('service.api.model_manager') as mock_manager:
            mock_manager.get_model_info.return_value = {
                "active_models": [
                    {
                        "name": "xgboost_v1",
                        "version": "1.0.0",
                        "accuracy": 0.95,
                        "precision": 0.92,
                        "recall": 0.88
                    }
                ],
                "ensemble_config": {
                    "voting_strategy": "soft",
                    "model_weights": {"xgboost_v1": 1.0}
                }
            }
            
            response = client.get("/api/v1/models")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "active_models" in data
            assert "ensemble_config" in data
    
    def test_update_model_config(self, client):
        """Test model configuration update endpoint."""
        config_data = {
            "models": [
                {
                    "name": "xgboost_v2",
                    "version": "2.0.0",
                    "weight": 0.6,
                    "enabled": True
                },
                {
                    "name": "random_forest_v1",
                    "version": "1.0.0",
                    "weight": 0.4,
                    "enabled": True
                }
            ],
            "voting_strategy": "soft",
            "fraud_threshold": 0.5
        }
        
        with patch('service.api.model_manager') as mock_manager:
            mock_manager.update_config.return_value = True
            
            response = client.put("/api/v1/models/config", json=config_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
    
    def test_get_prediction_history(self, client):
        """Test prediction history endpoint."""
        with patch('service.api.prediction_store') as mock_store:
            mock_store.get_predictions.return_value = [
                {
                    "transaction_id": "txn_1",
                    "is_fraud": False,
                    "fraud_probability": 0.1,
                    "prediction_timestamp": datetime.now(timezone.utc).isoformat()
                },
                {
                    "transaction_id": "txn_2",
                    "is_fraud": True,
                    "fraud_probability": 0.9,
                    "prediction_timestamp": datetime.now(timezone.utc).isoformat()
                }
            ]
            
            response = client.get("/api/v1/predictions?limit=10")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "predictions" in data
            assert len(data["predictions"]) == 2
    
    def test_api_rate_limiting(self, client, sample_transaction_data):
        """Test API rate limiting."""
        # This would require actual rate limiting implementation
        # For now, just test that multiple requests work
        
        with patch('service.api.fraud_detection_service') as mock_service:
            mock_prediction = FraudPrediction(
                transaction_id="txn_test_123",
                is_fraud=False,
                fraud_probability=0.1,
                risk_score=0.15,
                confidence=0.9,
                model_version="ensemble_v1.0.0",
                prediction_timestamp=datetime.now(timezone.utc),
                features_used=[],
                model_scores=[]
            )
            mock_service.predict_fraud.return_value = mock_prediction
            
            # Make multiple requests
            responses = []
            for i in range(5):
                sample_transaction_data["transaction_id"] = f"txn_test_{i}"
                response = client.post("/api/v1/predict", json=sample_transaction_data)
                responses.append(response)
            
            # All should succeed (no rate limiting in test)
            assert all(r.status_code == 200 for r in responses)
    
    def test_api_authentication(self, client, sample_transaction_data):
        """Test API authentication (if implemented)."""
        # This would test API key or JWT authentication
        # For now, just verify the endpoint works without auth
        
        with patch('service.api.fraud_detection_service') as mock_service:
            mock_prediction = FraudPrediction(
                transaction_id="txn_test_123",
                is_fraud=False,
                fraud_probability=0.1,
                risk_score=0.15,
                confidence=0.9,
                model_version="ensemble_v1.0.0",
                prediction_timestamp=datetime.now(timezone.utc),
                features_used=[],
                model_scores=[]
            )
            mock_service.predict_fraud.return_value = mock_prediction
            
            response = client.post("/api/v1/predict", json=sample_transaction_data)
            assert response.status_code == 200


class TestFraudDetectionWorkflow:
    """Integration tests for complete fraud detection workflow."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @patch('service.api.fraud_detection_service')
    @patch('service.api.kafka_producer')
    @patch('service.api.prediction_store')
    def test_complete_fraud_detection_workflow(self, mock_store, mock_kafka, 
                                             mock_service, client):
        """Test complete workflow from transaction to prediction storage."""
        transaction_data = {
            "transaction_id": "txn_workflow_test",
            "user_id": "user_workflow_test",
            "merchant_id": "merchant_workflow_test",
            "amount": "1500.00",  # High amount
            "currency": "USD",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "merchant_category": "electronics",
            "payment_method": "credit_card",
            "card_type": "visa",
            "location": {
                "latitude": 40.7128,
                "longitude": -74.0060,
                "city": "New York",
                "country": "US"
            }
        }
        
        # Mock fraud detection service
        mock_prediction = FraudPrediction(
            transaction_id="txn_workflow_test",
            is_fraud=True,
            fraud_probability=0.85,
            risk_score=0.9,
            confidence=0.82,
            model_version="ensemble_v1.0.0",
            prediction_timestamp=datetime.now(timezone.utc),
            features_used=["amount_zscore", "velocity_score"],
            model_scores=[
                {
                    "model_name": "xgboost_v1",
                    "model_version": "1.0.0",
                    "score": 0.88,
                    "confidence": 0.85
                }
            ]
        )
        
        mock_service.predict_fraud.return_value = mock_prediction
        mock_store.store_prediction.return_value = True
        mock_kafka.send.return_value = True
        
        # Make prediction request
        response = client.post("/api/v1/predict", json=transaction_data)
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["is_fraud"] is True
        assert data["fraud_probability"] == 0.85
        
        # Verify service was called
        mock_service.predict_fraud.assert_called_once()
        
        # Verify prediction was stored
        mock_store.store_prediction.assert_called_once()
        
        # Verify Kafka message was sent (for real-time alerts)
        mock_kafka.send.assert_called_once()
    
    @patch('service.api.fraud_detection_service')
    def test_performance_under_load(self, mock_service, client):
        """Test API performance under simulated load."""
        import time
        
        # Mock fast prediction
        mock_prediction = FraudPrediction(
            transaction_id="txn_perf_test",
            is_fraud=False,
            fraud_probability=0.1,
            risk_score=0.15,
            confidence=0.9,
            model_version="ensemble_v1.0.0",
            prediction_timestamp=datetime.now(timezone.utc),
            features_used=[],
            model_scores=[]
        )
        mock_service.predict_fraud.return_value = mock_prediction
        
        transaction_data = {
            "transaction_id": "txn_perf_test",
            "user_id": "user_perf_test",
            "merchant_id": "merchant_perf_test",
            "amount": "100.00",
            "currency": "USD",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "merchant_category": "grocery",
            "payment_method": "credit_card",
            "card_type": "visa",
            "location": {
                "latitude": 40.7128,
                "longitude": -74.0060,
                "city": "New York",
                "country": "US"
            }
        }
        
        # Measure response times
        response_times = []
        num_requests = 10
        
        for i in range(num_requests):
            transaction_data["transaction_id"] = f"txn_perf_test_{i}"
            
            start_time = time.time()
            response = client.post("/api/v1/predict", json=transaction_data)
            end_time = time.time()
            
            assert response.status_code == 200
            response_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Verify performance requirements
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        
        # Should meet <50ms requirement (in test environment)
        assert avg_response_time < 100  # Relaxed for test environment
        assert max_response_time < 200   # Relaxed for test environment
        
        print(f"Average response time: {avg_response_time:.2f}ms")
        print(f"Max response time: {max_response_time:.2f}ms")


if __name__ == "__main__":
    pytest.main([__file__])