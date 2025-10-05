"""Performance tests for fraud detection system latency requirements."""

import pytest
import asyncio
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import patch, Mock
from datetime import datetime, timezone
from decimal import Decimal
import numpy as np

from service.ml_inference import MLInferenceService
from service.xgboost_model import XGBoostModelWrapper
from features.feature_pipeline import FeaturePipeline
from service.api import app
from shared.models import Transaction, FraudPrediction
from fastapi.testclient import TestClient


class TestLatencyPerformance:
    """Performance tests focusing on latency requirements (<50ms)."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def sample_transaction(self):
        """Sample transaction for performance testing."""
        return Transaction(
            transaction_id="txn_perf_001",
            user_id="user_perf_001",
            merchant_id="merchant_perf_001",
            amount=Decimal("125.50"),
            currency="USD",
            timestamp=datetime.now(timezone.utc),
            merchant_category="grocery",
            payment_method="credit_card",
            card_type="visa",
            location={
                "latitude": 40.7128,
                "longitude": -74.0060,
                "city": "New York",
                "country": "US"
            }
        )
    
    @pytest.fixture
    def trained_model(self):
        """Create a trained model for performance testing."""
        model = FraudDetectionModel(
            model_name="perf_test_xgboost",
            model_version="1.0.0"
        )
        
        # Create synthetic training data
        X_train = np.random.rand(1000, 10)
        y_train = np.random.randint(0, 2, 1000)
        
        model.train(X_train, y_train)
        return model
    
    def test_single_prediction_latency(self, trained_model):
        """Test latency of single fraud prediction."""
        # Create test features
        test_features = np.random.rand(1, 10)
        
        # Warm up the model (first prediction might be slower)
        trained_model.predict_proba(test_features)
        
        # Measure prediction latency
        latencies = []
        num_predictions = 100
        
        for _ in range(num_predictions):
            start_time = time.perf_counter()
            trained_model.predict_proba(test_features)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        # Calculate statistics
        avg_latency = statistics.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        max_latency = max(latencies)
        
        print(f"Single Prediction Latency Stats:")
        print(f"Average: {avg_latency:.2f}ms")
        print(f"P95: {p95_latency:.2f}ms")
        print(f"P99: {p99_latency:.2f}ms")
        print(f"Max: {max_latency:.2f}ms")
        
        # Performance assertions
        assert avg_latency < 10, f"Average latency {avg_latency:.2f}ms exceeds 10ms"
        assert p95_latency < 20, f"P95 latency {p95_latency:.2f}ms exceeds 20ms"
        assert p99_latency < 30, f"P99 latency {p99_latency:.2f}ms exceeds 30ms"
    
    def test_ensemble_prediction_latency(self):
        """Test latency of ensemble model predictions."""
        # Create multiple trained models
        models = []
        for i in range(3):
            model = FraudDetectionModel(
                model_name=f"perf_test_model_{i}",
                model_version="1.0.0"
            )
            
            # Train with synthetic data
            X_train = np.random.rand(500, 10)
            y_train = np.random.randint(0, 2, 500)
            model.train(X_train, y_train)
            models.append(model)
        
        # Create ensemble
        ensemble = ModelEnsemble(
            models=models,
            voting_strategy="soft",
            weights=[0.4, 0.4, 0.2]
        )
        
        # Test features
        test_features = np.random.rand(1, 10)
        
        # Warm up
        ensemble.predict(test_features)
        
        # Measure ensemble prediction latency
        latencies = []
        num_predictions = 50
        
        for _ in range(num_predictions):
            start_time = time.perf_counter()
            ensemble.predict(test_features)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        # Calculate statistics
        avg_latency = statistics.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        max_latency = max(latencies)
        
        print(f"Ensemble Prediction Latency Stats:")
        print(f"Average: {avg_latency:.2f}ms")
        print(f"P95: {p95_latency:.2f}ms")
        print(f"Max: {max_latency:.2f}ms")
        
        # Performance assertions (ensemble should still be fast)
        assert avg_latency < 25, f"Average ensemble latency {avg_latency:.2f}ms exceeds 25ms"
        assert p95_latency < 40, f"P95 ensemble latency {p95_latency:.2f}ms exceeds 40ms"
    
    @patch('service.ml_service.get_user_transaction_history')
    @patch('service.ml_service.get_merchant_profile')
    @patch('service.ml_service.get_user_profile')
    def test_feature_engineering_latency(self, mock_user_profile, mock_merchant_profile,
                                       mock_transaction_history, sample_transaction):
        """Test latency of feature engineering process."""
        # Mock external data sources
        mock_user_profile.return_value = {
            'avg_transaction_amount': Decimal('100.00'),
            'transaction_count_30d': 50,
            'risk_score': 0.2
        }
        
        mock_merchant_profile.return_value = {
            'avg_transaction_amount': Decimal('85.00'),
            'fraud_rate_30d': 0.01,
            'risk_score': 0.15
        }
        
        mock_transaction_history.return_value = [
            {'amount': Decimal('120.00'), 'timestamp': datetime.now(timezone.utc)}
            for _ in range(10)
        ]
        
        feature_engineer = FeatureEngineer()
        
        # Warm up
        feature_engineer.engineer_features(sample_transaction)
        
        # Measure feature engineering latency
        latencies = []
        num_iterations = 100
        
        for _ in range(num_iterations):
            start_time = time.perf_counter()
            feature_engineer.engineer_features(sample_transaction)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        # Calculate statistics
        avg_latency = statistics.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        max_latency = max(latencies)
        
        print(f"Feature Engineering Latency Stats:")
        print(f"Average: {avg_latency:.2f}ms")
        print(f"P95: {p95_latency:.2f}ms")
        print(f"Max: {max_latency:.2f}ms")
        
        # Performance assertions
        assert avg_latency < 15, f"Average feature engineering latency {avg_latency:.2f}ms exceeds 15ms"
        assert p95_latency < 25, f"P95 feature engineering latency {p95_latency:.2f}ms exceeds 25ms"
    
    @patch('service.api.fraud_detection_service')
    def test_api_endpoint_latency(self, mock_service, client):
        """Test end-to-end API latency."""
        # Mock fraud detection service
        mock_prediction = FraudPrediction(
            transaction_id="txn_perf_001",
            is_fraud=False,
            fraud_probability=0.15,
            risk_score=0.2,
            confidence=0.85,
            model_version="ensemble_v1.0.0",
            prediction_timestamp=datetime.now(timezone.utc),
            features_used=["amount_zscore", "frequency_1h"],
            model_scores=[]
        )
        mock_service.predict_fraud.return_value = mock_prediction
        
        transaction_data = {
            "transaction_id": "txn_perf_001",
            "user_id": "user_perf_001",
            "merchant_id": "merchant_perf_001",
            "amount": "125.50",
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
        
        # Warm up
        client.post("/api/v1/predict", json=transaction_data)
        
        # Measure API latency
        latencies = []
        num_requests = 100
        
        for i in range(num_requests):
            transaction_data["transaction_id"] = f"txn_perf_{i:03d}"
            
            start_time = time.perf_counter()
            response = client.post("/api/v1/predict", json=transaction_data)
            end_time = time.perf_counter()
            
            assert response.status_code == 200
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        # Calculate statistics
        avg_latency = statistics.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        max_latency = max(latencies)
        
        print(f"API Endpoint Latency Stats:")
        print(f"Average: {avg_latency:.2f}ms")
        print(f"P95: {p95_latency:.2f}ms")
        print(f"P99: {p99_latency:.2f}ms")
        print(f"Max: {max_latency:.2f}ms")
        
        # Critical performance assertions for <50ms requirement
        assert avg_latency < 50, f"Average API latency {avg_latency:.2f}ms exceeds 50ms requirement"
        assert p95_latency < 75, f"P95 API latency {p95_latency:.2f}ms exceeds 75ms"
        assert p99_latency < 100, f"P99 API latency {p99_latency:.2f}ms exceeds 100ms"
    
    @patch('service.api.fraud_detection_service')
    def test_concurrent_request_latency(self, mock_service, client):
        """Test latency under concurrent load."""
        # Mock fraud detection service
        mock_prediction = FraudPrediction(
            transaction_id="txn_concurrent",
            is_fraud=False,
            fraud_probability=0.15,
            risk_score=0.2,
            confidence=0.85,
            model_version="ensemble_v1.0.0",
            prediction_timestamp=datetime.now(timezone.utc),
            features_used=[],
            model_scores=[]
        )
        mock_service.predict_fraud.return_value = mock_prediction
        
        def make_request(request_id):
            """Make a single API request."""
            transaction_data = {
                "transaction_id": f"txn_concurrent_{request_id}",
                "user_id": f"user_concurrent_{request_id}",
                "merchant_id": "merchant_concurrent",
                "amount": "125.50",
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
            
            start_time = time.perf_counter()
            response = client.post("/api/v1/predict", json=transaction_data)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            return latency_ms, response.status_code
        
        # Test with concurrent requests
        num_concurrent = 20
        num_requests_per_thread = 5
        
        all_latencies = []
        
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = []
            
            for thread_id in range(num_concurrent):
                for req_id in range(num_requests_per_thread):
                    request_id = f"{thread_id}_{req_id}"
                    future = executor.submit(make_request, request_id)
                    futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                latency, status_code = future.result()
                assert status_code == 200
                all_latencies.append(latency)
        
        # Calculate statistics
        avg_latency = statistics.mean(all_latencies)
        p95_latency = np.percentile(all_latencies, 95)
        p99_latency = np.percentile(all_latencies, 99)
        max_latency = max(all_latencies)
        
        print(f"Concurrent Request Latency Stats ({num_concurrent} threads):")
        print(f"Total requests: {len(all_latencies)}")
        print(f"Average: {avg_latency:.2f}ms")
        print(f"P95: {p95_latency:.2f}ms")
        print(f"P99: {p99_latency:.2f}ms")
        print(f"Max: {max_latency:.2f}ms")
        
        # Performance assertions under load
        assert avg_latency < 100, f"Average concurrent latency {avg_latency:.2f}ms exceeds 100ms"
        assert p95_latency < 150, f"P95 concurrent latency {p95_latency:.2f}ms exceeds 150ms"
    
    def test_memory_usage_during_predictions(self, trained_model):
        """Test memory usage during continuous predictions."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Measure initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Make many predictions
        test_features = np.random.rand(1, 10)
        num_predictions = 1000
        
        start_time = time.perf_counter()
        
        for _ in range(num_predictions):
            trained_model.predict_proba(test_features)
        
        end_time = time.perf_counter()
        
        # Measure final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        total_time = (end_time - start_time) * 1000  # ms
        avg_time_per_prediction = total_time / num_predictions
        
        print(f"Memory Usage Stats:")
        print(f"Initial memory: {initial_memory:.2f}MB")
        print(f"Final memory: {final_memory:.2f}MB")
        print(f"Memory increase: {memory_increase:.2f}MB")
        print(f"Average time per prediction: {avg_time_per_prediction:.2f}ms")
        
        # Memory should not increase significantly
        assert memory_increase < 50, f"Memory increased by {memory_increase:.2f}MB, indicating potential leak"
        assert avg_time_per_prediction < 1, f"Average prediction time {avg_time_per_prediction:.2f}ms too slow"
    
    def test_batch_prediction_efficiency(self, trained_model):
        """Test efficiency of batch predictions vs individual predictions."""
        batch_sizes = [1, 10, 50, 100]
        
        for batch_size in batch_sizes:
            test_features = np.random.rand(batch_size, 10)
            
            # Warm up
            trained_model.predict_proba(test_features)
            
            # Measure batch prediction time
            start_time = time.perf_counter()
            trained_model.predict_proba(test_features)
            end_time = time.perf_counter()
            
            batch_time_ms = (end_time - start_time) * 1000
            time_per_prediction = batch_time_ms / batch_size
            
            print(f"Batch size {batch_size}: {batch_time_ms:.2f}ms total, {time_per_prediction:.2f}ms per prediction")
            
            # Larger batches should be more efficient per prediction
            if batch_size == 1:
                single_prediction_time = time_per_prediction
            else:
                efficiency_gain = single_prediction_time / time_per_prediction
                assert efficiency_gain > 1, f"Batch size {batch_size} not more efficient than single predictions"


class TestThroughputPerformance:
    """Performance tests focusing on throughput requirements."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @patch('service.api.fraud_detection_service')
    def test_requests_per_second(self, mock_service, client):
        """Test maximum requests per second the API can handle."""
        # Mock fraud detection service
        mock_prediction = FraudPrediction(
            transaction_id="txn_throughput",
            is_fraud=False,
            fraud_probability=0.15,
            risk_score=0.2,
            confidence=0.85,
            model_version="ensemble_v1.0.0",
            prediction_timestamp=datetime.now(timezone.utc),
            features_used=[],
            model_scores=[]
        )
        mock_service.predict_fraud.return_value = mock_prediction
        
        transaction_data = {
            "transaction_id": "txn_throughput",
            "user_id": "user_throughput",
            "merchant_id": "merchant_throughput",
            "amount": "125.50",
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
        
        # Test duration
        test_duration_seconds = 5
        
        # Make requests for the test duration
        start_time = time.time()
        request_count = 0
        
        while time.time() - start_time < test_duration_seconds:
            transaction_data["transaction_id"] = f"txn_throughput_{request_count}"
            response = client.post("/api/v1/predict", json=transaction_data)
            assert response.status_code == 200
            request_count += 1
        
        end_time = time.time()
        actual_duration = end_time - start_time
        
        requests_per_second = request_count / actual_duration
        
        print(f"Throughput Test Results:")
        print(f"Total requests: {request_count}")
        print(f"Test duration: {actual_duration:.2f}s")
        print(f"Requests per second: {requests_per_second:.2f}")
        
        # Should handle at least 100 requests per second
        assert requests_per_second >= 100, f"Throughput {requests_per_second:.2f} RPS below 100 RPS requirement"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])