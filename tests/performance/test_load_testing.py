"""Load testing for fraud detection system."""

import asyncio
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import pytest
import requests
from fastapi.testclient import TestClient

from api.main import app
from tests.conftest import sample_transaction_data


@dataclass
class LoadTestConfig:
    """Load test configuration."""
    duration_seconds: int = 60
    ramp_up_seconds: int = 10
    max_users: int = 100
    requests_per_user: int = 10
    think_time_min: float = 0.1
    think_time_max: float = 0.5
    timeout_seconds: float = 5.0


@dataclass
class LoadTestResult:
    """Load test results."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    duration_seconds: float
    requests_per_second: float
    success_rate: float
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    p50_response_time: float
    p95_response_time: float
    p99_response_time: float
    error_distribution: Dict[str, int]
    response_time_distribution: List[float]


class LoadTester:
    """Load testing utility class."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.results = []
        self.errors = []
    
    def generate_test_data(self) -> Dict[str, Any]:
        """Generate randomized test transaction data."""
        base_data = sample_transaction_data.copy()
        
        # Randomize some fields for more realistic load testing
        base_data["amount"] = round(random.uniform(1.0, 10000.0), 2)
        base_data["merchant_id"] = f"merchant_{random.randint(1, 1000)}"
        base_data["user_id"] = f"user_{random.randint(1, 10000)}"
        base_data["timestamp"] = int(time.time() - random.randint(0, 86400))  # Last 24 hours
        
        # Randomize location
        base_data["location"] = {
            "latitude": round(random.uniform(-90, 90), 6),
            "longitude": round(random.uniform(-180, 180), 6),
            "country": random.choice(["US", "CA", "GB", "DE", "FR", "JP", "AU"])
        }
        
        return base_data
    
    def make_request(self, endpoint: str, method: str = "POST", data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make a single HTTP request."""
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()
        
        try:
            if method == "POST":
                response = self.session.post(url, json=data, timeout=5.0)
            elif method == "GET":
                response = self.session.get(url, timeout=5.0)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response_time = time.time() - start_time
            
            return {
                "success": True,
                "status_code": response.status_code,
                "response_time": response_time,
                "response_size": len(response.content),
                "error": None
            }
        
        except Exception as e:
            response_time = time.time() - start_time
            return {
                "success": False,
                "status_code": 0,
                "response_time": response_time,
                "response_size": 0,
                "error": str(e)
            }
    
    def user_scenario(self, user_id: int, config: LoadTestConfig) -> List[Dict[str, Any]]:
        """Simulate a user scenario with multiple requests."""
        results = []
        
        for request_num in range(config.requests_per_user):
            # Generate test data
            test_data = self.generate_test_data()
            
            # Make prediction request
            result = self.make_request("/api/v1/predict", "POST", test_data)
            result["user_id"] = user_id
            result["request_num"] = request_num
            result["endpoint"] = "/api/v1/predict"
            results.append(result)
            
            # Think time between requests
            if request_num < config.requests_per_user - 1:
                think_time = random.uniform(config.think_time_min, config.think_time_max)
                time.sleep(think_time)
        
        return results
    
    def run_load_test(self, config: LoadTestConfig) -> LoadTestResult:
        """Run a complete load test."""
        print(f"Starting load test with {config.max_users} users for {config.duration_seconds}s")
        
        start_time = time.time()
        all_results = []
        
        # Calculate user ramp-up schedule
        users_per_second = config.max_users / config.ramp_up_seconds if config.ramp_up_seconds > 0 else config.max_users
        
        with ThreadPoolExecutor(max_workers=config.max_users) as executor:
            futures = []
            
            # Ramp up users gradually
            for user_id in range(config.max_users):
                # Calculate when this user should start
                start_delay = user_id / users_per_second if config.ramp_up_seconds > 0 else 0
                
                # Submit user scenario
                future = executor.submit(self._delayed_user_scenario, user_id, config, start_delay)
                futures.append(future)
            
            # Collect results as they complete
            for future in as_completed(futures, timeout=config.duration_seconds + 30):
                try:
                    user_results = future.result()
                    all_results.extend(user_results)
                except Exception as e:
                    print(f"User scenario failed: {e}")
        
        end_time = time.time()
        
        return self._analyze_results(all_results, end_time - start_time)
    
    def _delayed_user_scenario(self, user_id: int, config: LoadTestConfig, delay: float) -> List[Dict[str, Any]]:
        """Run user scenario with initial delay."""
        if delay > 0:
            time.sleep(delay)
        
        return self.user_scenario(user_id, config)
    
    def _analyze_results(self, results: List[Dict[str, Any]], duration: float) -> LoadTestResult:
        """Analyze load test results."""
        if not results:
            return LoadTestResult(
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                duration_seconds=duration,
                requests_per_second=0,
                success_rate=0,
                avg_response_time=0,
                min_response_time=0,
                max_response_time=0,
                p50_response_time=0,
                p95_response_time=0,
                p99_response_time=0,
                error_distribution={},
                response_time_distribution=[]
            )
        
        successful_results = [r for r in results if r["success"]]
        failed_results = [r for r in results if not r["success"]]
        
        response_times = [r["response_time"] for r in results]
        response_times.sort()
        
        # Calculate percentiles
        def percentile(data, p):
            if not data:
                return 0
            index = int(p / 100 * len(data))
            return data[min(index, len(data) - 1)]
        
        # Error distribution
        error_distribution = {}
        for result in failed_results:
            error = result.get("error", "Unknown")
            error_distribution[error] = error_distribution.get(error, 0) + 1
        
        return LoadTestResult(
            total_requests=len(results),
            successful_requests=len(successful_results),
            failed_requests=len(failed_results),
            duration_seconds=duration,
            requests_per_second=len(successful_results) / duration if duration > 0 else 0,
            success_rate=len(successful_results) / len(results) if results else 0,
            avg_response_time=sum(response_times) / len(response_times) if response_times else 0,
            min_response_time=min(response_times) if response_times else 0,
            max_response_time=max(response_times) if response_times else 0,
            p50_response_time=percentile(response_times, 50),
            p95_response_time=percentile(response_times, 95),
            p99_response_time=percentile(response_times, 99),
            error_distribution=error_distribution,
            response_time_distribution=response_times
        )


@pytest.fixture
def load_tester():
    """Load tester fixture."""
    return LoadTester()


@pytest.fixture
def light_load_config():
    """Light load test configuration."""
    return LoadTestConfig(
        duration_seconds=30,
        ramp_up_seconds=5,
        max_users=10,
        requests_per_user=5,
        think_time_min=0.1,
        think_time_max=0.3
    )


@pytest.fixture
def medium_load_config():
    """Medium load test configuration."""
    return LoadTestConfig(
        duration_seconds=60,
        ramp_up_seconds=10,
        max_users=50,
        requests_per_user=10,
        think_time_min=0.1,
        think_time_max=0.5
    )


@pytest.fixture
def heavy_load_config():
    """Heavy load test configuration."""
    return LoadTestConfig(
        duration_seconds=120,
        ramp_up_seconds=20,
        max_users=100,
        requests_per_user=20,
        think_time_min=0.05,
        think_time_max=0.2
    )


@pytest.mark.performance
@pytest.mark.load
class TestLoadTesting:
    """Load testing scenarios."""
    
    def test_light_load(self, load_tester, light_load_config):
        """Test system under light load."""
        result = load_tester.run_load_test(light_load_config)
        
        # Assertions for light load
        assert result.success_rate >= 0.98, f"Expected >= 98% success rate, got {result.success_rate:.2%}"
        assert result.requests_per_second >= 10, f"Expected >= 10 RPS, got {result.requests_per_second:.1f}"
        assert result.avg_response_time <= 0.1, f"Expected <= 100ms avg response time, got {result.avg_response_time:.3f}s"
        assert result.p95_response_time <= 0.2, f"Expected <= 200ms P95 response time, got {result.p95_response_time:.3f}s"
        assert result.p99_response_time <= 0.5, f"Expected <= 500ms P99 response time, got {result.p99_response_time:.3f}s"
    
    def test_medium_load(self, load_tester, medium_load_config):
        """Test system under medium load."""
        result = load_tester.run_load_test(medium_load_config)
        
        # Assertions for medium load
        assert result.success_rate >= 0.95, f"Expected >= 95% success rate, got {result.success_rate:.2%}"
        assert result.requests_per_second >= 30, f"Expected >= 30 RPS, got {result.requests_per_second:.1f}"
        assert result.avg_response_time <= 0.2, f"Expected <= 200ms avg response time, got {result.avg_response_time:.3f}s"
        assert result.p95_response_time <= 0.5, f"Expected <= 500ms P95 response time, got {result.p95_response_time:.3f}s"
        assert result.p99_response_time <= 1.0, f"Expected <= 1s P99 response time, got {result.p99_response_time:.3f}s"
    
    @pytest.mark.slow
    def test_heavy_load(self, load_tester, heavy_load_config):
        """Test system under heavy load."""
        result = load_tester.run_load_test(heavy_load_config)
        
        # Assertions for heavy load (more relaxed)
        assert result.success_rate >= 0.90, f"Expected >= 90% success rate, got {result.success_rate:.2%}"
        assert result.requests_per_second >= 50, f"Expected >= 50 RPS, got {result.requests_per_second:.1f}"
        assert result.avg_response_time <= 0.5, f"Expected <= 500ms avg response time, got {result.avg_response_time:.3f}s"
        assert result.p95_response_time <= 1.0, f"Expected <= 1s P95 response time, got {result.p95_response_time:.3f}s"
        assert result.p99_response_time <= 2.0, f"Expected <= 2s P99 response time, got {result.p99_response_time:.3f}s"
    
    def test_spike_load(self, load_tester):
        """Test system under sudden spike load."""
        spike_config = LoadTestConfig(
            duration_seconds=30,
            ramp_up_seconds=1,  # Very fast ramp-up
            max_users=50,
            requests_per_user=5,
            think_time_min=0.01,
            think_time_max=0.05
        )
        
        result = load_tester.run_load_test(spike_config)
        
        # Assertions for spike load (system should handle sudden load)
        assert result.success_rate >= 0.85, f"Expected >= 85% success rate under spike, got {result.success_rate:.2%}"
        assert result.requests_per_second >= 20, f"Expected >= 20 RPS under spike, got {result.requests_per_second:.1f}"
        assert result.avg_response_time <= 1.0, f"Expected <= 1s avg response time under spike, got {result.avg_response_time:.3f}s"
    
    def test_sustained_load(self, load_tester):
        """Test system under sustained load."""
        sustained_config = LoadTestConfig(
            duration_seconds=300,  # 5 minutes
            ramp_up_seconds=30,
            max_users=30,
            requests_per_user=50,
            think_time_min=0.2,
            think_time_max=0.8
        )
        
        result = load_tester.run_load_test(sustained_config)
        
        # Assertions for sustained load
        assert result.success_rate >= 0.95, f"Expected >= 95% success rate sustained, got {result.success_rate:.2%}"
        assert result.requests_per_second >= 20, f"Expected >= 20 RPS sustained, got {result.requests_per_second:.1f}"
        assert result.avg_response_time <= 0.3, f"Expected <= 300ms avg response time sustained, got {result.avg_response_time:.3f}s"
        assert result.p95_response_time <= 0.8, f"Expected <= 800ms P95 response time sustained, got {result.p95_response_time:.3f}s"


@pytest.mark.performance
@pytest.mark.stress
class TestStressTesting:
    """Stress testing scenarios."""
    
    def test_memory_stress(self, load_tester):
        """Test system under memory stress with large payloads."""
        # Create large transaction data
        large_data = sample_transaction_data.copy()
        large_data["metadata"] = {f"field_{i}": f"value_{i}" * 100 for i in range(100)}
        
        stress_config = LoadTestConfig(
            duration_seconds=60,
            ramp_up_seconds=10,
            max_users=20,
            requests_per_user=10,
            think_time_min=0.1,
            think_time_max=0.3
        )
        
        # Override the generate_test_data method for this test
        original_method = load_tester.generate_test_data
        load_tester.generate_test_data = lambda: large_data
        
        try:
            result = load_tester.run_load_test(stress_config)
            
            # System should still function under memory stress
            assert result.success_rate >= 0.80, f"Expected >= 80% success rate under memory stress, got {result.success_rate:.2%}"
            assert result.requests_per_second >= 5, f"Expected >= 5 RPS under memory stress, got {result.requests_per_second:.1f}"
            assert result.avg_response_time <= 2.0, f"Expected <= 2s avg response time under memory stress, got {result.avg_response_time:.3f}s"
        
        finally:
            # Restore original method
            load_tester.generate_test_data = original_method
    
    def test_connection_stress(self, load_tester):
        """Test system under connection stress."""
        connection_config = LoadTestConfig(
            duration_seconds=45,
            ramp_up_seconds=5,
            max_users=100,  # High number of concurrent connections
            requests_per_user=3,
            think_time_min=0.01,
            think_time_max=0.05
        )
        
        result = load_tester.run_load_test(connection_config)
        
        # System should handle many concurrent connections
        assert result.success_rate >= 0.75, f"Expected >= 75% success rate under connection stress, got {result.success_rate:.2%}"
        assert result.requests_per_second >= 30, f"Expected >= 30 RPS under connection stress, got {result.requests_per_second:.1f}"
        assert result.avg_response_time <= 1.5, f"Expected <= 1.5s avg response time under connection stress, got {result.avg_response_time:.3f}s"


@pytest.mark.performance
@pytest.mark.endurance
class TestEnduranceTesting:
    """Endurance testing scenarios."""
    
    @pytest.mark.slow
    def test_long_running_endurance(self, load_tester):
        """Test system endurance over extended period."""
        endurance_config = LoadTestConfig(
            duration_seconds=1800,  # 30 minutes
            ramp_up_seconds=60,
            max_users=25,
            requests_per_user=200,
            think_time_min=0.5,
            think_time_max=2.0
        )
        
        result = load_tester.run_load_test(endurance_config)
        
        # System should maintain performance over time
        assert result.success_rate >= 0.95, f"Expected >= 95% success rate over endurance, got {result.success_rate:.2%}"
        assert result.requests_per_second >= 15, f"Expected >= 15 RPS over endurance, got {result.requests_per_second:.1f}"
        assert result.avg_response_time <= 0.4, f"Expected <= 400ms avg response time over endurance, got {result.avg_response_time:.3f}s"
        assert result.p95_response_time <= 1.0, f"Expected <= 1s P95 response time over endurance, got {result.p95_response_time:.3f}s"
        
        # Check for performance degradation (response times shouldn't increase significantly)
        response_times = result.response_time_distribution
        if len(response_times) >= 100:
            first_quarter = response_times[:len(response_times)//4]
            last_quarter = response_times[-len(response_times)//4:]
            
            avg_first = sum(first_quarter) / len(first_quarter)
            avg_last = sum(last_quarter) / len(last_quarter)
            
            # Response time shouldn't degrade by more than 50%
            degradation_ratio = avg_last / avg_first if avg_first > 0 else 1
            assert degradation_ratio <= 1.5, f"Performance degraded by {degradation_ratio:.1f}x over time"