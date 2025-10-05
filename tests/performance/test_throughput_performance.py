"""Throughput performance tests for fraud detection system."""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

import pytest
from fastapi.testclient import TestClient

from api.main import app
from ml.fraud_detection_model import FraudDetectionModel
from tests.conftest import sample_transaction_data


class ThroughputTester:
    """Throughput testing utilities."""
    
    def __init__(self):
        self.results = []
        self.errors = []
    
    def measure_throughput(self, func, requests_per_second: int, duration_seconds: int) -> Dict[str, Any]:
        """Measure throughput for a given function."""
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        request_count = 0
        success_count = 0
        error_count = 0
        response_times = []
        
        while time.time() < end_time:
            batch_start = time.time()
            
            # Calculate how many requests to send in this batch
            batch_size = min(requests_per_second, int((end_time - time.time()) * requests_per_second))
            
            if batch_size <= 0:
                break
            
            # Send batch of requests
            with ThreadPoolExecutor(max_workers=min(batch_size, 50)) as executor:
                futures = [executor.submit(func) for _ in range(batch_size)]
                
                for future in as_completed(futures):
                    request_start = time.time()
                    try:
                        result = future.result(timeout=5.0)
                        success_count += 1
                        response_times.append(time.time() - request_start)
                    except Exception as e:
                        error_count += 1
                        self.errors.append(str(e))
                    
                    request_count += 1
            
            # Wait for next batch if needed
            batch_duration = time.time() - batch_start
            expected_batch_duration = batch_size / requests_per_second
            
            if batch_duration < expected_batch_duration:
                time.sleep(expected_batch_duration - batch_duration)
        
        actual_duration = time.time() - start_time
        actual_rps = success_count / actual_duration if actual_duration > 0 else 0
        
        return {
            "total_requests": request_count,
            "successful_requests": success_count,
            "failed_requests": error_count,
            "duration_seconds": actual_duration,
            "requests_per_second": actual_rps,
            "success_rate": success_count / request_count if request_count > 0 else 0,
            "avg_response_time": sum(response_times) / len(response_times) if response_times else 0,
            "min_response_time": min(response_times) if response_times else 0,
            "max_response_time": max(response_times) if response_times else 0,
            "p95_response_time": self._percentile(response_times, 95) if response_times else 0,
            "p99_response_time": self._percentile(response_times, 99) if response_times else 0,
        }
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]


@pytest.fixture
def throughput_tester():
    """Throughput tester fixture."""
    return ThroughputTester()


@pytest.fixture
def test_client():
    """Test client fixture."""
    return TestClient(app)


@pytest.mark.performance
class TestAPIThroughput:
    """API throughput performance tests."""
    
    def test_prediction_endpoint_throughput(self, test_client, throughput_tester, sample_transaction_data):
        """Test prediction endpoint throughput."""
        def make_request():
            response = test_client.post("/api/v1/predict", json=sample_transaction_data)
            assert response.status_code == 200
            return response.json()
        
        # Test sustained throughput
        results = throughput_tester.measure_throughput(
            func=make_request,
            requests_per_second=100,
            duration_seconds=30
        )
        
        # Assertions
        assert results["requests_per_second"] >= 80, f"Expected >= 80 RPS, got {results['requests_per_second']:.1f}"
        assert results["success_rate"] >= 0.95, f"Expected >= 95% success rate, got {results['success_rate']:.2%}"
        assert results["avg_response_time"] <= 0.1, f"Expected <= 100ms avg response time, got {results['avg_response_time']:.3f}s"
        assert results["p95_response_time"] <= 0.2, f"Expected <= 200ms P95 response time, got {results['p95_response_time']:.3f}s"
    
    def test_batch_prediction_throughput(self, test_client, throughput_tester, sample_transaction_data):
        """Test batch prediction endpoint throughput."""
        batch_data = [sample_transaction_data for _ in range(10)]
        
        def make_batch_request():
            response = test_client.post("/api/v1/predict/batch", json=batch_data)
            assert response.status_code == 200
            return response.json()
        
        results = throughput_tester.measure_throughput(
            func=make_batch_request,
            requests_per_second=20,
            duration_seconds=15
        )
        
        # Calculate effective transaction throughput (requests * batch_size)
        effective_tps = results["requests_per_second"] * 10
        
        assert effective_tps >= 150, f"Expected >= 150 transactions/sec, got {effective_tps:.1f}"
        assert results["success_rate"] >= 0.95, f"Expected >= 95% success rate, got {results['success_rate']:.2%}"
        assert results["avg_response_time"] <= 0.5, f"Expected <= 500ms avg response time, got {results['avg_response_time']:.3f}s"
    
    def test_health_check_throughput(self, test_client, throughput_tester):
        """Test health check endpoint throughput."""
        def make_health_request():
            response = test_client.get("/health")
            assert response.status_code == 200
            return response.json()
        
        results = throughput_tester.measure_throughput(
            func=make_health_request,
            requests_per_second=500,
            duration_seconds=10
        )
        
        assert results["requests_per_second"] >= 400, f"Expected >= 400 RPS, got {results['requests_per_second']:.1f}"
        assert results["success_rate"] >= 0.99, f"Expected >= 99% success rate, got {results['success_rate']:.2%}"
        assert results["avg_response_time"] <= 0.01, f"Expected <= 10ms avg response time, got {results['avg_response_time']:.3f}s"


@pytest.mark.performance
class TestMLModelThroughput:
    """ML model throughput performance tests."""
    
    def test_model_prediction_throughput(self, trained_xgboost_model, throughput_tester, sample_features):
        """Test ML model prediction throughput."""
        def make_prediction():
            prediction = trained_xgboost_model.predict(sample_features)
            assert 0 <= prediction <= 1
            return prediction
        
        results = throughput_tester.measure_throughput(
            func=make_prediction,
            requests_per_second=1000,
            duration_seconds=10
        )
        
        assert results["requests_per_second"] >= 800, f"Expected >= 800 predictions/sec, got {results['requests_per_second']:.1f}"
        assert results["success_rate"] >= 0.99, f"Expected >= 99% success rate, got {results['success_rate']:.2%}"
        assert results["avg_response_time"] <= 0.002, f"Expected <= 2ms avg response time, got {results['avg_response_time']:.3f}s"
    
    def test_batch_prediction_throughput(self, trained_xgboost_model, throughput_tester, sample_features):
        """Test batch prediction throughput."""
        import numpy as np
        
        # Create batch of features
        batch_features = np.tile(sample_features, (100, 1))
        
        def make_batch_prediction():
            predictions = trained_xgboost_model.predict_batch(batch_features)
            assert len(predictions) == 100
            assert all(0 <= p <= 1 for p in predictions)
            return predictions
        
        results = throughput_tester.measure_throughput(
            func=make_batch_prediction,
            requests_per_second=50,
            duration_seconds=10
        )
        
        # Calculate effective prediction throughput
        effective_pps = results["requests_per_second"] * 100
        
        assert effective_pps >= 4000, f"Expected >= 4000 predictions/sec, got {effective_pps:.1f}"
        assert results["success_rate"] >= 0.99, f"Expected >= 99% success rate, got {results['success_rate']:.2%}"
        assert results["avg_response_time"] <= 0.05, f"Expected <= 50ms avg response time, got {results['avg_response_time']:.3f}s"


@pytest.mark.performance
class TestConcurrentLoad:
    """Concurrent load performance tests."""
    
    def test_concurrent_api_load(self, test_client, sample_transaction_data):
        """Test API under concurrent load."""
        def worker_function(worker_id: int, num_requests: int) -> Dict[str, Any]:
            """Worker function for concurrent testing."""
            success_count = 0
            error_count = 0
            response_times = []
            
            for i in range(num_requests):
                start_time = time.time()
                try:
                    response = test_client.post("/api/v1/predict", json=sample_transaction_data)
                    if response.status_code == 200:
                        success_count += 1
                    else:
                        error_count += 1
                    
                    response_times.append(time.time() - start_time)
                except Exception:
                    error_count += 1
            
            return {
                "worker_id": worker_id,
                "success_count": success_count,
                "error_count": error_count,
                "avg_response_time": sum(response_times) / len(response_times) if response_times else 0,
                "max_response_time": max(response_times) if response_times else 0
            }
        
        # Run concurrent load test
        num_workers = 20
        requests_per_worker = 50
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(worker_function, worker_id, requests_per_worker)
                for worker_id in range(num_workers)
            ]
            
            results = [future.result() for future in as_completed(futures)]
        
        total_duration = time.time() - start_time
        
        # Aggregate results
        total_success = sum(r["success_count"] for r in results)
        total_errors = sum(r["error_count"] for r in results)
        total_requests = total_success + total_errors
        
        overall_rps = total_success / total_duration
        success_rate = total_success / total_requests if total_requests > 0 else 0
        avg_response_time = sum(r["avg_response_time"] for r in results) / len(results)
        max_response_time = max(r["max_response_time"] for r in results)
        
        # Assertions
        assert overall_rps >= 80, f"Expected >= 80 RPS under load, got {overall_rps:.1f}"
        assert success_rate >= 0.95, f"Expected >= 95% success rate under load, got {success_rate:.2%}"
        assert avg_response_time <= 0.2, f"Expected <= 200ms avg response time under load, got {avg_response_time:.3f}s"
        assert max_response_time <= 1.0, f"Expected <= 1s max response time under load, got {max_response_time:.3f}s"
    
    def test_sustained_load(self, test_client, sample_transaction_data):
        """Test API under sustained load."""
        duration_minutes = 2
        target_rps = 50
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        total_requests = 0
        successful_requests = 0
        response_times = []
        
        while time.time() < end_time:
            batch_start = time.time()
            
            # Send requests for this second
            with ThreadPoolExecutor(max_workers=target_rps) as executor:
                futures = [
                    executor.submit(
                        lambda: test_client.post("/api/v1/predict", json=sample_transaction_data)
                    )
                    for _ in range(target_rps)
                ]
                
                for future in as_completed(futures):
                    request_start = time.time()
                    try:
                        response = future.result(timeout=2.0)
                        if response.status_code == 200:
                            successful_requests += 1
                        response_times.append(time.time() - request_start)
                    except Exception:
                        pass
                    
                    total_requests += 1
            
            # Wait for next batch
            batch_duration = time.time() - batch_start
            if batch_duration < 1.0:
                time.sleep(1.0 - batch_duration)
        
        actual_duration = time.time() - start_time
        actual_rps = successful_requests / actual_duration
        success_rate = successful_requests / total_requests if total_requests > 0 else 0
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Assertions for sustained load
        assert actual_rps >= target_rps * 0.8, f"Expected >= {target_rps * 0.8} RPS sustained, got {actual_rps:.1f}"
        assert success_rate >= 0.90, f"Expected >= 90% success rate sustained, got {success_rate:.2%}"
        assert avg_response_time <= 0.3, f"Expected <= 300ms avg response time sustained, got {avg_response_time:.3f}s"


@pytest.mark.performance
@pytest.mark.benchmark
class TestBenchmarks:
    """Benchmark tests for performance regression detection."""
    
    def test_prediction_benchmark(self, benchmark, trained_xgboost_model, sample_features):
        """Benchmark single prediction performance."""
        result = benchmark(trained_xgboost_model.predict, sample_features)
        assert 0 <= result <= 1
    
    def test_batch_prediction_benchmark(self, benchmark, trained_xgboost_model, sample_features):
        """Benchmark batch prediction performance."""
        import numpy as np
        batch_features = np.tile(sample_features, (100, 1))
        
        result = benchmark(trained_xgboost_model.predict_batch, batch_features)
        assert len(result) == 100
    
    def test_feature_engineering_benchmark(self, benchmark, sample_transaction_data):
        """Benchmark feature engineering performance."""
        from features.feature_pipeline import FeaturePipeline
        
        pipeline = FeaturePipeline()
        result = benchmark(pipeline.process_transaction, sample_transaction_data)
        assert result is not None
    
    def test_api_endpoint_benchmark(self, benchmark, test_client, sample_transaction_data):
        """Benchmark API endpoint performance."""
        def make_api_call():
            response = test_client.post("/api/v1/predict", json=sample_transaction_data)
            assert response.status_code == 200
            return response.json()
        
        result = benchmark(make_api_call)
        assert "fraud_score" in result
        assert "risk_level" in result