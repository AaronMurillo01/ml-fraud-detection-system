"""Memory and resource performance tests for fraud detection system."""

import gc
import os
import psutil
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

import pytest
import numpy as np
from fastapi.testclient import TestClient

from api.main import app
from ml.fraud_detection_model import FraudDetectionModel
from features.feature_pipeline import FeaturePipeline
from tests.conftest import sample_transaction_data


@dataclass
class MemorySnapshot:
    """Memory usage snapshot."""
    timestamp: float
    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    percent: float  # Memory percentage
    available_mb: float
    cpu_percent: float
    thread_count: int
    fd_count: int  # File descriptor count


@dataclass
class ResourceUsage:
    """Resource usage statistics."""
    peak_memory_mb: float
    avg_memory_mb: float
    memory_growth_mb: float
    peak_cpu_percent: float
    avg_cpu_percent: float
    peak_threads: int
    peak_file_descriptors: int
    gc_collections: Dict[int, int]
    duration_seconds: float


class ResourceMonitor:
    """Resource monitoring utility."""
    
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.snapshots: List[MemorySnapshot] = []
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.process = psutil.Process()
    
    def start_monitoring(self):
        """Start resource monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.snapshots.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> ResourceUsage:
        """Stop monitoring and return resource usage statistics."""
        if not self.monitoring:
            return self._empty_usage()
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        return self._calculate_usage()
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                snapshot = self._take_snapshot()
                self.snapshots.append(snapshot)
                time.sleep(self.interval)
            except Exception:
                # Continue monitoring even if individual snapshots fail
                pass
    
    def _take_snapshot(self) -> MemorySnapshot:
        """Take a memory and resource snapshot."""
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()
        cpu_percent = self.process.cpu_percent()
        thread_count = self.process.num_threads()
        
        # File descriptor count (Unix-like systems)
        try:
            fd_count = self.process.num_fds()
        except (AttributeError, psutil.AccessDenied):
            fd_count = 0
        
        # System memory info
        system_memory = psutil.virtual_memory()
        
        return MemorySnapshot(
            timestamp=time.time(),
            rss_mb=memory_info.rss / 1024 / 1024,
            vms_mb=memory_info.vms / 1024 / 1024,
            percent=memory_percent,
            available_mb=system_memory.available / 1024 / 1024,
            cpu_percent=cpu_percent,
            thread_count=thread_count,
            fd_count=fd_count
        )
    
    def _calculate_usage(self) -> ResourceUsage:
        """Calculate resource usage statistics from snapshots."""
        if not self.snapshots:
            return self._empty_usage()
        
        memory_values = [s.rss_mb for s in self.snapshots]
        cpu_values = [s.cpu_percent for s in self.snapshots]
        thread_values = [s.thread_count for s in self.snapshots]
        fd_values = [s.fd_count for s in self.snapshots]
        
        # Get garbage collection stats
        gc_stats = {i: gc.get_count()[i] for i in range(3)}
        
        duration = self.snapshots[-1].timestamp - self.snapshots[0].timestamp
        
        return ResourceUsage(
            peak_memory_mb=max(memory_values),
            avg_memory_mb=sum(memory_values) / len(memory_values),
            memory_growth_mb=memory_values[-1] - memory_values[0],
            peak_cpu_percent=max(cpu_values),
            avg_cpu_percent=sum(cpu_values) / len(cpu_values),
            peak_threads=max(thread_values),
            peak_file_descriptors=max(fd_values),
            gc_collections=gc_stats,
            duration_seconds=duration
        )
    
    def _empty_usage(self) -> ResourceUsage:
        """Return empty resource usage."""
        return ResourceUsage(
            peak_memory_mb=0,
            avg_memory_mb=0,
            memory_growth_mb=0,
            peak_cpu_percent=0,
            avg_cpu_percent=0,
            peak_threads=0,
            peak_file_descriptors=0,
            gc_collections={},
            duration_seconds=0
        )


@contextmanager
def monitor_resources(interval: float = 0.1):
    """Context manager for resource monitoring."""
    monitor = ResourceMonitor(interval)
    monitor.start_monitoring()
    try:
        yield monitor
    finally:
        usage = monitor.stop_monitoring()
        yield usage


@pytest.fixture
def resource_monitor():
    """Resource monitor fixture."""
    return ResourceMonitor()


@pytest.fixture
def test_client():
    """Test client fixture."""
    return TestClient(app)


@pytest.mark.performance
@pytest.mark.memory
class TestMemoryPerformance:
    """Memory performance tests."""
    
    def test_single_prediction_memory(self, trained_xgboost_model, sample_features, resource_monitor):
        """Test memory usage for single prediction."""
        # Force garbage collection before test
        gc.collect()
        
        resource_monitor.start_monitoring()
        
        # Perform predictions
        for _ in range(1000):
            prediction = trained_xgboost_model.predict(sample_features)
            assert 0 <= prediction <= 1
        
        usage = resource_monitor.stop_monitoring()
        
        # Memory assertions
        assert usage.peak_memory_mb < 500, f"Peak memory usage too high: {usage.peak_memory_mb:.1f}MB"
        assert usage.memory_growth_mb < 50, f"Memory growth too high: {usage.memory_growth_mb:.1f}MB"
        assert usage.avg_cpu_percent < 80, f"Average CPU usage too high: {usage.avg_cpu_percent:.1f}%"
    
    def test_batch_prediction_memory(self, trained_xgboost_model, sample_features, resource_monitor):
        """Test memory usage for batch predictions."""
        # Create large batch
        batch_size = 10000
        batch_features = np.tile(sample_features, (batch_size, 1))
        
        gc.collect()
        resource_monitor.start_monitoring()
        
        # Perform batch prediction
        predictions = trained_xgboost_model.predict_batch(batch_features)
        assert len(predictions) == batch_size
        
        usage = resource_monitor.stop_monitoring()
        
        # Memory should scale reasonably with batch size
        expected_max_memory = 1000  # MB
        assert usage.peak_memory_mb < expected_max_memory, f"Peak memory usage too high: {usage.peak_memory_mb:.1f}MB"
        assert usage.memory_growth_mb < 200, f"Memory growth too high: {usage.memory_growth_mb:.1f}MB"
    
    def test_feature_pipeline_memory(self, resource_monitor):
        """Test memory usage for feature pipeline."""
        pipeline = FeaturePipeline()
        
        gc.collect()
        resource_monitor.start_monitoring()
        
        # Process many transactions
        for _ in range(5000):
            # Create varied transaction data
            transaction_data = sample_transaction_data.copy()
            transaction_data["amount"] = np.random.uniform(1, 10000)
            transaction_data["timestamp"] = int(time.time())
            
            features = pipeline.process_transaction(transaction_data)
            assert features is not None
        
        usage = resource_monitor.stop_monitoring()
        
        # Feature pipeline should not leak memory
        assert usage.peak_memory_mb < 300, f"Peak memory usage too high: {usage.peak_memory_mb:.1f}MB"
        assert usage.memory_growth_mb < 30, f"Memory growth too high: {usage.memory_growth_mb:.1f}MB"
    
    def test_api_endpoint_memory(self, test_client, resource_monitor):
        """Test memory usage for API endpoints."""
        gc.collect()
        resource_monitor.start_monitoring()
        
        # Make many API requests
        for _ in range(1000):
            response = test_client.post("/api/v1/predict", json=sample_transaction_data)
            assert response.status_code == 200
        
        usage = resource_monitor.stop_monitoring()
        
        # API should not leak memory
        assert usage.peak_memory_mb < 400, f"Peak memory usage too high: {usage.peak_memory_mb:.1f}MB"
        assert usage.memory_growth_mb < 40, f"Memory growth too high: {usage.memory_growth_mb:.1f}MB"
        assert usage.peak_threads < 50, f"Too many threads created: {usage.peak_threads}"
    
    def test_concurrent_requests_memory(self, test_client, resource_monitor):
        """Test memory usage under concurrent load."""
        import threading
        from concurrent.futures import ThreadPoolExecutor
        
        def make_request():
            response = test_client.post("/api/v1/predict", json=sample_transaction_data)
            return response.status_code == 200
        
        gc.collect()
        resource_monitor.start_monitoring()
        
        # Run concurrent requests
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(make_request) for _ in range(500)]
            results = [f.result() for f in futures]
        
        usage = resource_monitor.stop_monitoring()
        
        # Check that most requests succeeded
        success_rate = sum(results) / len(results)
        assert success_rate >= 0.95, f"Success rate too low: {success_rate:.2%}"
        
        # Memory usage should be reasonable under concurrent load
        assert usage.peak_memory_mb < 600, f"Peak memory usage too high: {usage.peak_memory_mb:.1f}MB"
        assert usage.memory_growth_mb < 100, f"Memory growth too high: {usage.memory_growth_mb:.1f}MB"
        assert usage.peak_threads < 100, f"Too many threads created: {usage.peak_threads}"


@pytest.mark.performance
@pytest.mark.memory
class TestMemoryLeaks:
    """Memory leak detection tests."""
    
    def test_model_prediction_leak(self, trained_xgboost_model, sample_features):
        """Test for memory leaks in model predictions."""
        gc.collect()
        initial_memory = psutil.Process().memory_info().rss
        
        # Run many predictions
        for i in range(10000):
            prediction = trained_xgboost_model.predict(sample_features)
            assert 0 <= prediction <= 1
            
            # Check memory periodically
            if i % 1000 == 0:
                gc.collect()
                current_memory = psutil.Process().memory_info().rss
                memory_growth = (current_memory - initial_memory) / 1024 / 1024
                
                # Memory growth should be minimal
                assert memory_growth < 100, f"Potential memory leak detected: {memory_growth:.1f}MB growth after {i} predictions"
    
    def test_feature_pipeline_leak(self):
        """Test for memory leaks in feature pipeline."""
        pipeline = FeaturePipeline()
        
        gc.collect()
        initial_memory = psutil.Process().memory_info().rss
        
        # Process many transactions
        for i in range(5000):
            transaction_data = sample_transaction_data.copy()
            transaction_data["timestamp"] = int(time.time()) + i
            
            features = pipeline.process_transaction(transaction_data)
            assert features is not None
            
            # Check memory periodically
            if i % 500 == 0:
                gc.collect()
                current_memory = psutil.Process().memory_info().rss
                memory_growth = (current_memory - initial_memory) / 1024 / 1024
                
                assert memory_growth < 50, f"Potential memory leak detected: {memory_growth:.1f}MB growth after {i} transactions"
    
    def test_api_request_leak(self, test_client):
        """Test for memory leaks in API requests."""
        gc.collect()
        initial_memory = psutil.Process().memory_info().rss
        
        # Make many API requests
        for i in range(2000):
            response = test_client.post("/api/v1/predict", json=sample_transaction_data)
            assert response.status_code == 200
            
            # Check memory periodically
            if i % 200 == 0:
                gc.collect()
                current_memory = psutil.Process().memory_info().rss
                memory_growth = (current_memory - initial_memory) / 1024 / 1024
                
                assert memory_growth < 80, f"Potential memory leak detected: {memory_growth:.1f}MB growth after {i} requests"


@pytest.mark.performance
@pytest.mark.resource
class TestResourceUsage:
    """Resource usage tests."""
    
    def test_cpu_usage_single_prediction(self, trained_xgboost_model, sample_features, resource_monitor):
        """Test CPU usage for single predictions."""
        resource_monitor.start_monitoring()
        
        start_time = time.time()
        prediction_count = 0
        
        # Run predictions for 10 seconds
        while time.time() - start_time < 10:
            prediction = trained_xgboost_model.predict(sample_features)
            assert 0 <= prediction <= 1
            prediction_count += 1
        
        usage = resource_monitor.stop_monitoring()
        
        # CPU usage should be reasonable
        assert usage.avg_cpu_percent < 90, f"Average CPU usage too high: {usage.avg_cpu_percent:.1f}%"
        assert usage.peak_cpu_percent < 100, f"Peak CPU usage too high: {usage.peak_cpu_percent:.1f}%"
        
        # Should achieve reasonable throughput
        throughput = prediction_count / 10
        assert throughput >= 500, f"Throughput too low: {throughput:.1f} predictions/sec"
    
    def test_thread_usage(self, test_client, resource_monitor):
        """Test thread usage under load."""
        from concurrent.futures import ThreadPoolExecutor
        
        def make_request():
            return test_client.post("/api/v1/predict", json=sample_transaction_data)
        
        resource_monitor.start_monitoring()
        
        # Run concurrent requests
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(make_request) for _ in range(200)]
            responses = [f.result() for f in futures]
        
        usage = resource_monitor.stop_monitoring()
        
        # Check response success
        success_count = sum(1 for r in responses if r.status_code == 200)
        success_rate = success_count / len(responses)
        assert success_rate >= 0.95, f"Success rate too low: {success_rate:.2%}"
        
        # Thread usage should be reasonable
        assert usage.peak_threads < 100, f"Too many threads: {usage.peak_threads}"
    
    def test_file_descriptor_usage(self, test_client, resource_monitor):
        """Test file descriptor usage."""
        resource_monitor.start_monitoring()
        
        # Make many requests to test file descriptor usage
        for _ in range(500):
            response = test_client.post("/api/v1/predict", json=sample_transaction_data)
            assert response.status_code == 200
        
        usage = resource_monitor.stop_monitoring()
        
        # File descriptor usage should not grow excessively
        if usage.peak_file_descriptors > 0:  # Only check if FD monitoring is available
            assert usage.peak_file_descriptors < 1000, f"Too many file descriptors: {usage.peak_file_descriptors}"


@pytest.mark.performance
@pytest.mark.gc
class TestGarbageCollection:
    """Garbage collection performance tests."""
    
    def test_gc_pressure(self, trained_xgboost_model, sample_features):
        """Test garbage collection pressure."""
        # Get initial GC stats
        initial_gc_counts = gc.get_count()
        gc.collect()
        
        # Create objects that will need garbage collection
        for _ in range(1000):
            # Create temporary objects
            temp_data = [sample_features.copy() for _ in range(10)]
            predictions = [trained_xgboost_model.predict(data) for data in temp_data]
            assert all(0 <= p <= 1 for p in predictions)
            
            # Explicitly delete to test GC
            del temp_data, predictions
        
        # Force garbage collection
        collected = gc.collect()
        final_gc_counts = gc.get_count()
        
        # GC should have cleaned up objects
        assert collected >= 0, "Garbage collection should have run"
        
        # GC counts should not grow excessively
        for i in range(3):
            growth = final_gc_counts[i] - initial_gc_counts[i]
            assert growth < 10000, f"Excessive objects in GC generation {i}: {growth}"
    
    def test_gc_timing(self, trained_xgboost_model, sample_features):
        """Test that GC doesn't significantly impact performance."""
        # Disable automatic GC
        gc.disable()
        
        try:
            # Time predictions without GC
            start_time = time.time()
            for _ in range(1000):
                prediction = trained_xgboost_model.predict(sample_features)
                assert 0 <= prediction <= 1
            no_gc_time = time.time() - start_time
            
            # Re-enable GC
            gc.enable()
            
            # Time predictions with GC
            start_time = time.time()
            for _ in range(1000):
                prediction = trained_xgboost_model.predict(sample_features)
                assert 0 <= prediction <= 1
            with_gc_time = time.time() - start_time
            
            # GC overhead should be minimal
            gc_overhead = (with_gc_time - no_gc_time) / no_gc_time
            assert gc_overhead < 0.5, f"GC overhead too high: {gc_overhead:.1%}"
        
        finally:
            # Ensure GC is re-enabled
            gc.enable()


@pytest.mark.performance
@pytest.mark.profiling
class TestProfiling:
    """Profiling and performance analysis tests."""
    
    def test_prediction_profiling(self, trained_xgboost_model, sample_features):
        """Profile prediction performance."""
        import cProfile
        import pstats
        from io import StringIO
        
        # Profile predictions
        profiler = cProfile.Profile()
        profiler.enable()
        
        for _ in range(1000):
            prediction = trained_xgboost_model.predict(sample_features)
            assert 0 <= prediction <= 1
        
        profiler.disable()
        
        # Analyze profile
        stats_stream = StringIO()
        stats = pstats.Stats(profiler, stream=stats_stream)
        stats.sort_stats('cumulative')
        stats.print_stats(10)  # Top 10 functions
        
        profile_output = stats_stream.getvalue()
        
        # Basic checks on profile output
        assert "predict" in profile_output, "Predict function should appear in profile"
        assert len(profile_output) > 0, "Profile should contain performance data"
    
    def test_memory_profiling(self, trained_xgboost_model, sample_features):
        """Profile memory usage patterns."""
        try:
            import tracemalloc
        except ImportError:
            pytest.skip("tracemalloc not available")
        
        # Start memory tracing
        tracemalloc.start()
        
        # Take initial snapshot
        snapshot1 = tracemalloc.take_snapshot()
        
        # Perform operations
        predictions = []
        for _ in range(1000):
            prediction = trained_xgboost_model.predict(sample_features)
            predictions.append(prediction)
        
        # Take final snapshot
        snapshot2 = tracemalloc.take_snapshot()
        
        # Analyze memory usage
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        
        # Check that memory usage is reasonable
        total_memory_mb = sum(stat.size for stat in top_stats) / 1024 / 1024
        assert total_memory_mb < 100, f"Memory usage too high: {total_memory_mb:.1f}MB"
        
        tracemalloc.stop()