"""Integration tests for API endpoints."""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any
from fastapi.testclient import TestClient
from httpx import AsyncClient

from api.fraud import app, get_inference_service
from service.ml_inference import MLInferenceService, InferenceResponse
from service.model_loader import ModelLoader
from shared.models import Transaction, PaymentMethod, TransactionStatus, RiskLevel
from service.models import EnrichedTransaction, PredictionResult


class TestFraudDetectionAPI:
    """Integration tests for fraud detection API endpoints."""
    
    @pytest.fixture
    def mock_inference_service(self):
        """Create a mock inference service."""
        service = Mock(spec=MLInferenceService)
        return service
    
    @pytest.fixture
    def test_client(self, mock_inference_service):
        """Create test client with mocked dependencies."""
        app.dependency_overrides[get_inference_service] = lambda: mock_inference_service
        client = TestClient(app)
        yield client
        app.dependency_overrides.clear()
    
    @pytest.fixture
    def sample_transaction_data(self):
        """Create sample transaction data for testing."""
        return {
            "transaction_id": "txn_test_123",
            "user_id": "user_test_456",
            "merchant_id": "merchant_test_789",
            "amount": 250.75,
            "timestamp": datetime.utcnow().isoformat(),
            "payment_method": "credit_card",
            "transaction_type": "purchase",
            "status": "pending",
            "currency": "USD",
            "description": "Test transaction",
            "metadata": {
                "ip_address": "192.168.1.100",
                "user_agent": "TestAgent/1.0",
                "device_id": "device_123"
            }
        }
    
    @pytest.fixture
    def sample_enriched_transaction(self, sample_transaction_data):
        """Create sample enriched transaction."""
        return EnrichedTransaction(
            **sample_transaction_data,
            velocity_features={
                "txn_count_1h": 2,
                "txn_count_24h": 8,
                "amount_sum_1h": 500.0,
                "amount_sum_24h": 1200.0
            },
            risk_features={
                "merchant_risk_score": 0.3,
                "user_risk_score": 0.2,
                "location_risk_score": 0.1
            },
            behavioral_features={
                "avg_transaction_amount": 180.50,
                "transaction_frequency": 3.2
            },
            contextual_features={
                "is_weekend": False,
                "hour_of_day": 14,
                "day_of_week": 2
            }
        )
    
    @pytest.fixture
    def sample_prediction_result(self):
        """Create sample prediction result."""
        return PredictionResult(
            transaction_id="txn_test_123",
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
            explanation="Low fraud probability based on normal transaction pattern"
        )
    
    def test_health_check_endpoint(self, test_client):
        """Test health check endpoint."""
        response = test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "uptime_seconds" in data
    
    def test_predict_fraud_success(self, test_client, mock_inference_service, 
                                 sample_transaction_data, sample_prediction_result):
        """Test successful fraud prediction."""
        # Setup mock response
        mock_response = InferenceResponse(
            success=True,
            transaction_id="txn_test_123",
            prediction=sample_prediction_result.dict(),
            processing_time_ms=32.5,
            model_version="v1.0.0",
            timestamp=datetime.utcnow()
        )
        mock_inference_service.predict = AsyncMock(return_value=mock_response)
        
        # Make request
        response = test_client.post("/api/v1/predict", json=sample_transaction_data)
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["transaction_id"] == "txn_test_123"
        assert "prediction" in data
        assert data["prediction"]["fraud_probability"] == 0.25
        assert data["prediction"]["risk_level"] == "LOW"
        assert data["prediction"]["decision"] == "APPROVE"
        assert "processing_time_ms" in data
        assert "model_version" in data
        assert "timestamp" in data
        
        # Verify service was called
        mock_inference_service.predict.assert_called_once()
    
    def test_predict_fraud_with_enrichment(self, test_client, mock_inference_service, 
                                         sample_transaction_data, sample_prediction_result):
        """Test fraud prediction with transaction enrichment."""
        # Add enrichment data to request
        enriched_data = {
            **sample_transaction_data,
            "velocity_features": {
                "txn_count_1h": 5,
                "amount_sum_24h": 2000.0
            },
            "risk_features": {
                "merchant_risk_score": 0.7,
                "user_risk_score": 0.4
            }
        }
        
        # Setup mock response for high-risk transaction
        high_risk_result = PredictionResult(
            transaction_id="txn_test_123",
            fraud_probability=0.85,
            risk_level=RiskLevel.HIGH,
            decision="DECLINE",
            confidence_score=0.92,
            model_version="v1.0.0",
            processing_time_ms=28.3
        )
        
        mock_response = InferenceResponse(
            success=True,
            transaction_id="txn_test_123",
            prediction=high_risk_result.dict(),
            processing_time_ms=28.3,
            model_version="v1.0.0",
            timestamp=datetime.utcnow()
        )
        mock_inference_service.predict = AsyncMock(return_value=mock_response)
        
        # Make request
        response = test_client.post("/api/v1/predict", json=enriched_data)
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["prediction"]["fraud_probability"] == 0.85
        assert data["prediction"]["risk_level"] == "HIGH"
        assert data["prediction"]["decision"] == "DECLINE"
    
    def test_predict_fraud_error(self, test_client, mock_inference_service, sample_transaction_data):
        """Test fraud prediction with service error."""
        # Setup mock to return error response
        mock_response = InferenceResponse(
            success=False,
            transaction_id="txn_test_123",
            prediction={
                "fraud_probability": 0.5,
                "risk_level": "UNKNOWN",
                "decision": "REVIEW",
                "confidence_score": 0.0
            },
            error_message="Model loading failed",
            processing_time_ms=5.0,
            model_version="unknown",
            timestamp=datetime.utcnow()
        )
        mock_inference_service.predict = AsyncMock(return_value=mock_response)
        
        # Make request
        response = test_client.post("/api/v1/predict", json=sample_transaction_data)
        
        # Assertions
        assert response.status_code == 200  # API should still return 200 but with error info
        data = response.json()
        assert data["success"] is False
        assert "error_message" in data
        assert data["error_message"] == "Model loading failed"
        assert data["prediction"]["decision"] == "REVIEW"
    
    def test_predict_fraud_invalid_input(self, test_client):
        """Test fraud prediction with invalid input data."""
        invalid_data = {
            "transaction_id": "",  # Empty transaction ID
            "amount": -100,  # Negative amount
            "timestamp": "invalid_date"
        }
        
        response = test_client.post("/api/v1/predict", json=invalid_data)
        
        # Should return validation error
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
    
    def test_predict_fraud_missing_fields(self, test_client):
        """Test fraud prediction with missing required fields."""
        incomplete_data = {
            "transaction_id": "txn_123"
            # Missing required fields like amount, user_id, etc.
        }
        
        response = test_client.post("/api/v1/predict", json=incomplete_data)
        
        # Should return validation error
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
    
    def test_batch_predict_fraud_success(self, test_client, mock_inference_service, 
                                       sample_transaction_data, sample_prediction_result):
        """Test successful batch fraud prediction."""
        # Create batch request data
        batch_data = {
            "transactions": [sample_transaction_data, sample_transaction_data],
            "include_feature_importance": True,
            "include_model_features": False
        }
        
        # Setup mock batch response
        from service.ml_inference import BatchInferenceResponse
        mock_batch_response = BatchInferenceResponse(
            batch_size=2,
            predictions=[
                InferenceResponse(
                    success=True,
                    transaction_id="txn_test_123",
                    prediction=sample_prediction_result.dict(),
                    processing_time_ms=30.0,
                    model_version="v1.0.0",
                    timestamp=datetime.utcnow()
                ),
                InferenceResponse(
                    success=True,
                    transaction_id="txn_test_123",
                    prediction=sample_prediction_result.dict(),
                    processing_time_ms=28.5,
                    model_version="v1.0.0",
                    timestamp=datetime.utcnow()
                )
            ],
            success_count=2,
            error_count=0,
            total_processing_time_ms=58.5,
            average_processing_time_ms=29.25,
            timestamp=datetime.utcnow()
        )
        mock_inference_service.predict_batch = AsyncMock(return_value=mock_batch_response)
        
        # Make request
        response = test_client.post("/api/v1/predict/batch", json=batch_data)
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["batch_size"] == 2
        assert len(data["predictions"]) == 2
        assert data["success_count"] == 2
        assert data["error_count"] == 0
        assert "total_processing_time_ms" in data
        assert "average_processing_time_ms" in data
        
        # Verify service was called
        mock_inference_service.predict_batch.assert_called_once()
    
    def test_batch_predict_fraud_empty_batch(self, test_client):
        """Test batch prediction with empty transaction list."""
        batch_data = {
            "transactions": []
        }
        
        response = test_client.post("/api/v1/predict/batch", json=batch_data)
        
        # Should return validation error
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
    
    def test_get_model_info(self, test_client, mock_inference_service):
        """Test model information endpoint."""
        # Setup mock model info
        mock_model_info = {
            "model_name": "fraud_detector_v1",
            "model_version": "1.0.0",
            "model_type": "xgboost",
            "feature_count": 25,
            "training_date": datetime.utcnow().isoformat(),
            "performance_metrics": {
                "accuracy": 0.95,
                "precision": 0.88,
                "recall": 0.92,
                "f1_score": 0.90,
                "auc_roc": 0.96
            }
        }
        mock_inference_service.get_model_info = Mock(return_value=mock_model_info)
        
        # Make request
        response = test_client.get("/api/v1/model/info")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "fraud_detector_v1"
        assert data["model_version"] == "1.0.0"
        assert "performance_metrics" in data
        assert data["performance_metrics"]["accuracy"] == 0.95
    
    def test_get_service_metrics(self, test_client, mock_inference_service):
        """Test service metrics endpoint."""
        # Setup mock service metrics
        mock_metrics = {
            "total_requests": 10000,
            "successful_requests": 9850,
            "failed_requests": 150,
            "average_response_time_ms": 32.5,
            "requests_per_minute": 125,
            "error_rate": 0.015,
            "cache_hit_rate": 0.85,
            "active_models": 2,
            "uptime_seconds": 86400
        }
        mock_inference_service.get_service_stats = Mock(return_value=mock_metrics)
        
        # Make request
        response = test_client.get("/api/v1/metrics")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["total_requests"] == 10000
        assert data["error_rate"] == 0.015
        assert data["cache_hit_rate"] == 0.85
        assert data["active_models"] == 2
    
    def test_predict_with_custom_model(self, test_client, mock_inference_service, 
                                     sample_transaction_data, sample_prediction_result):
        """Test prediction with custom model specification."""
        # Add model specification to request
        request_data = {
            **sample_transaction_data,
            "model_name": "custom_fraud_model",
            "model_version": "2.0.0"
        }
        
        # Setup mock response
        mock_response = InferenceResponse(
            success=True,
            transaction_id="txn_test_123",
            prediction=sample_prediction_result.dict(),
            processing_time_ms=35.2,
            model_version="2.0.0",
            timestamp=datetime.utcnow()
        )
        mock_inference_service.predict = AsyncMock(return_value=mock_response)
        
        # Make request
        response = test_client.post("/api/v1/predict", json=request_data)
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["model_version"] == "2.0.0"
        
        # Verify the correct model was requested
        call_args = mock_inference_service.predict.call_args[0][0]
        assert call_args.model_name == "custom_fraud_model"
        assert call_args.model_version == "2.0.0"
    
    def test_api_rate_limiting(self, test_client, sample_transaction_data):
        """Test API rate limiting (if implemented)."""
        # This test would verify rate limiting functionality
        # Implementation depends on the actual rate limiting strategy
        
        # Make multiple rapid requests
        responses = []
        for i in range(10):
            response = test_client.post("/api/v1/predict", json=sample_transaction_data)
            responses.append(response)
        
        # All requests should succeed in test environment
        # In production, some might be rate limited
        for response in responses:
            assert response.status_code in [200, 429]  # 429 = Too Many Requests
    
    def test_api_authentication(self, test_client, sample_transaction_data):
        """Test API authentication (if implemented)."""
        # This test would verify authentication requirements
        # Implementation depends on the actual auth strategy
        
        # Test without authentication headers
        response = test_client.post("/api/v1/predict", json=sample_transaction_data)
        
        # In test environment, auth might be disabled
        # In production, this might return 401 Unauthorized
        assert response.status_code in [200, 401]
    
    def test_cors_headers(self, test_client):
        """Test CORS headers are properly set."""
        response = test_client.options("/api/v1/predict")
        
        # Check for CORS headers (if CORS is enabled)
        # This depends on FastAPI CORS middleware configuration
        assert response.status_code in [200, 405]  # 405 = Method Not Allowed if OPTIONS not supported


class TestAPIErrorHandling:
    """Test API error handling scenarios."""
    
    @pytest.fixture
    def test_client(self):
        """Create test client without mocked dependencies to test error handling."""
        return TestClient(app)
    
    def test_internal_server_error_handling(self, test_client):
        """Test handling of internal server errors."""
        # This would test how the API handles unexpected exceptions
        # Implementation depends on error handling middleware
        
        with patch('api.fraud.get_inference_service', side_effect=Exception("Service unavailable")):
            response = test_client.get("/api/v1/model/info")
            
            # Should return 500 Internal Server Error
            assert response.status_code == 500
            data = response.json()
            assert "detail" in data or "error" in data
    
    def test_service_unavailable_handling(self, test_client):
        """Test handling when inference service is unavailable."""
        sample_data = {
            "transaction_id": "txn_123",
            "user_id": "user_456",
            "merchant_id": "merchant_789",
            "amount": 100.0,
            "timestamp": datetime.utcnow().isoformat(),
            "payment_method": "credit_card",
            "transaction_type": "purchase",
            "status": "pending"
        }
        
        with patch('api.fraud.get_inference_service', return_value=None):
            response = test_client.post("/api/v1/predict", json=sample_data)
            
            # Should handle gracefully
            assert response.status_code in [500, 503]  # 503 = Service Unavailable


class TestAPIPerformance:
    """Test API performance characteristics."""
    
    @pytest.fixture
    def test_client(self, mock_inference_service):
        """Create test client with fast mock responses."""
        app.dependency_overrides[get_inference_service] = lambda: mock_inference_service
        client = TestClient(app)
        yield client
        app.dependency_overrides.clear()
    
    def test_prediction_response_time(self, test_client, mock_inference_service):
        """Test that prediction responses are within acceptable time limits."""
        import time
        
        # Setup fast mock response
        mock_response = InferenceResponse(
            success=True,
            transaction_id="txn_perf_test",
            prediction={
                "fraud_probability": 0.3,
                "risk_level": "LOW",
                "decision": "APPROVE",
                "confidence_score": 0.8
            },
            processing_time_ms=15.0,
            model_version="v1.0.0",
            timestamp=datetime.utcnow()
        )
        mock_inference_service.predict = AsyncMock(return_value=mock_response)
        
        sample_data = {
            "transaction_id": "txn_perf_test",
            "user_id": "user_456",
            "merchant_id": "merchant_789",
            "amount": 100.0,
            "timestamp": datetime.utcnow().isoformat(),
            "payment_method": "credit_card",
            "transaction_type": "purchase",
            "status": "pending"
        }
        
        # Measure response time
        start_time = time.time()
        response = test_client.post("/api/v1/predict", json=sample_data)
        end_time = time.time()
        
        response_time_ms = (end_time - start_time) * 1000
        
        # Assertions
        assert response.status_code == 200
        assert response_time_ms < 1000  # Should respond within 1 second
    
    def test_concurrent_requests(self, test_client, mock_inference_service):
        """Test handling of concurrent requests."""
        import concurrent.futures
        import threading
        
        # Setup mock response
        mock_response = InferenceResponse(
            success=True,
            transaction_id="txn_concurrent",
            prediction={
                "fraud_probability": 0.4,
                "risk_level": "MEDIUM",
                "decision": "REVIEW",
                "confidence_score": 0.7
            },
            processing_time_ms=20.0,
            model_version="v1.0.0",
            timestamp=datetime.utcnow()
        )
        mock_inference_service.predict = AsyncMock(return_value=mock_response)
        
        sample_data = {
            "transaction_id": "txn_concurrent",
            "user_id": "user_456",
            "merchant_id": "merchant_789",
            "amount": 100.0,
            "timestamp": datetime.utcnow().isoformat(),
            "payment_method": "credit_card",
            "transaction_type": "purchase",
            "status": "pending"
        }
        
        def make_request():
            return test_client.post("/api/v1/predict", json=sample_data)
        
        # Make concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            responses = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True