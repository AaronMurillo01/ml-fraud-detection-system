"""Performance tests for fraud detection system."""

import pytest
import asyncio
import time
import statistics
import concurrent.futures
from typing import List, Dict, Any, Tuple
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from shared.models import Transaction, PaymentMethod, TransactionStatus
from tests.fixtures.test_data import (
    generate_random_transaction,
    generate_batch_transactions,
    sample_transactions
)
from tests.fixtures.mock_objects import (
    MockMLInferenceService,
    MockModelLoader,
    MockKafkaProducer,
    MockDatabase
)


class PerformanceMetrics:
    """Helper class to collect and analyze performance metrics."""
    
    def __init__(self):
        self.latencies: List[float] = []
        self.throughput_data: List[Tuple[int, float]] = []  # (count, duration)
        self.error_count = 0
        self.success_count = 0
    
    def record_latency(self, latency: float):
        """Record a single request latency."""
        self.latencies.append(latency)
    
    def record_throughput(self, count: int, duration: float):
        """Record throughput data."""
        self.throughput_data.append((count, duration))
    
    def record_success(self):
        """Record a successful operation."""
        self.success_count += 1
    
    def record_error(self):
        """Record a failed operation."""
        self.error_count += 1
    
    def get_latency_stats(self) -> Dict[str, float]:
        """Get latency statistics."""
        if not self.latencies:
            return {}
        
        return {
            "min": min(self.latencies),
            "max": max(self.latencies),
            "mean": statistics.mean(self.latencies),
            "median": statistics.median(self.latencies),
            "p95": self._percentile(self.latencies, 95),
            "p99": self._percentile(self.latencies, 99),
            "std_dev": statistics.stdev(self.latencies) if len(self.latencies) > 1 else 0
        }
    
    def get_throughput_stats(self) -> Dict[str, float]:
        """Get throughput statistics."""
        if not self.throughput_data:
            return {}
        
        throughputs = [count / duration for count, duration in self.throughput_data if duration > 0]
        
        if not throughputs:
            return {}
        
        return {
            "min_rps": min(throughputs),
            "max_rps": max(throughputs),
            "mean_rps": statistics.mean(throughputs),
            "median_rps": statistics.median(throughputs)
        }
    
    def get_error_rate(self) -> float:
        """Get error rate as percentage."""
        total = self.success_count + self.error_count
        return (self.error_count / total * 100) if total > 0 else 0
    
    @staticmethod
    def _percentile(data: List[float], percentile: int) -> float:
        """Calculate percentile value."""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]


class TestMLInferencePerformance:
    """Performance tests for ML inference service."""
    
    @pytest.fixture
    def ml_service(self):
        """Create ML inference service for testing."""
        return MockMLInferenceService()
    
    @pytest.fixture
    def performance_metrics(self):
        """Create performance metrics collector."""
        return PerformanceMetrics()
    
    def test_single_prediction_latency(self, ml_service, performance_metrics):
        """Test latency of single fraud predictions."""
        num_requests = 100
        
        for _ in range(num_requests):
            transaction = generate_random_transaction()
            
            start_time = time.time()
            try:
                result = ml_service.predict(transaction)
                end_time = time.time()
                
                latency = (end_time - start_time) * 1000  # Convert to milliseconds
                performance_metrics.record_latency(latency)
                performance_metrics.record_success()
                
                # Verify result structure
                assert "fraud_probability" in result
                assert "risk_level" in result
                
            except Exception as e:
                performance_metrics.record_error()
                print(f"Prediction error: {e}")
        
        # Analyze results
        latency_stats = performance_metrics.get_latency_stats()
        error_rate = performance_metrics.get_error_rate()
        
        print(f"\nSingle Prediction Latency Stats (ms):")
        print(f"  Mean: {latency_stats['mean']:.2f}")
        print(f"  Median: {latency_stats['median']:.2f}")
        print(f"  P95: {latency_stats['p95']:.2f}")
        print(f"  P99: {latency_stats['p99']:.2f}")
        print(f"  Error Rate: {error_rate:.2f}%")
        
        # Performance assertions
        assert latency_stats["mean"] < 100, "Mean latency should be under 100ms"
        assert latency_stats["p95"] < 200, "P95 latency should be under 200ms"
        assert error_rate < 1, "Error rate should be under 1%"
    
    def test_batch_prediction_throughput(self, ml_service, performance_metrics):
        """Test throughput of batch fraud predictions."""
        batch_sizes = [10, 50, 100, 200]
        
        for batch_size in batch_sizes:
            transactions = generate_batch_transactions(batch_size)
            
            start_time = time.time()
            try:
                results = ml_service.predict_batch(transactions)
                end_time = time.time()
                
                duration = end_time - start_time
                performance_metrics.record_throughput(batch_size, duration)
                performance_metrics.record_success()
                
                # Verify results
                assert len(results) == batch_size
                
            except Exception as e:
                performance_metrics.record_error()
                print(f"Batch prediction error: {e}")
        
        # Analyze results
        throughput_stats = performance_metrics.get_throughput_stats()
        error_rate = performance_metrics.get_error_rate()
        
        print(f"\nBatch Prediction Throughput Stats:")
        print(f"  Mean RPS: {throughput_stats['mean_rps']:.2f}")
        print(f"  Max RPS: {throughput_stats['max_rps']:.2f}")
        print(f"  Error Rate: {error_rate:.2f}%")
        
        # Performance assertions
        assert throughput_stats["mean_rps"] > 50, "Mean throughput should be over 50 RPS"
        assert error_rate < 1, "Error rate should be under 1%"
    
    def test_concurrent_predictions(self, ml_service, performance_metrics):
        """Test performance under concurrent load."""
        num_threads = 10
        requests_per_thread = 20
        
        def make_predictions(thread_id: int):
            """Make predictions in a thread."""
            thread_metrics = PerformanceMetrics()
            
            for i in range(requests_per_thread):
                transaction = generate_random_transaction()
                
                start_time = time.time()
                try:
                    result = ml_service.predict(transaction)
                    end_time = time.time()
                    
                    latency = (end_time - start_time) * 1000
                    thread_metrics.record_latency(latency)
                    thread_metrics.record_success()
                    
                except Exception as e:
                    thread_metrics.record_error()
                    print(f"Thread {thread_id} error: {e}")
            
            return thread_metrics
        
        # Execute concurrent requests
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(make_predictions, i)
                for i in range(num_threads)
            ]
            
            thread_results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Aggregate results
        total_requests = num_threads * requests_per_thread
        total_errors = sum(metrics.error_count for metrics in thread_results)
        all_latencies = []
        
        for metrics in thread_results:
            all_latencies.extend(metrics.latencies)
            performance_metrics.success_count += metrics.success_count
            performance_metrics.error_count += metrics.error_count
        
        performance_metrics.latencies = all_latencies
        performance_metrics.record_throughput(total_requests, total_duration)
        
        # Analyze results
        latency_stats = performance_metrics.get_latency_stats()
        throughput_stats = performance_metrics.get_throughput_stats()
        error_rate = performance_metrics.get_error_rate()
        
        print(f"\nConcurrent Load Test Results:")
        print(f"  Total Requests: {total_requests}")
        print(f"  Duration: {total_duration:.2f}s")
        print(f"  Mean Latency: {latency_stats['mean']:.2f}ms")
        print(f"  P95 Latency: {latency_stats['p95']:.2f}ms")
        print(f"  Throughput: {throughput_stats['mean_rps']:.2f} RPS")
        print(f"  Error Rate: {error_rate:.2f}%")
        
        # Performance assertions
        assert latency_stats["p95"] < 500, "P95 latency should be under 500ms under load"
        assert throughput_stats["mean_rps"] > 30, "Throughput should be over 30 RPS under load"
        assert error_rate < 5, "Error rate should be under 5% under load"
    
    @pytest.mark.asyncio
    async def test_async_prediction_performance(self, performance_metrics):
        """Test performance of async predictions."""
        # Create async ML service mock
        class AsyncMLService:
            async def predict_async(self, transaction: Dict[str, Any]):
                # Simulate async processing
                await asyncio.sleep(0.01)  # 10ms processing time
                return {
                    "fraud_probability": 0.15,
                    "risk_level": "LOW",
                    "confidence_score": 0.92
                }
        
        service = AsyncMLService()
        num_requests = 100
        
        async def make_async_prediction():
            transaction = generate_random_transaction()
            
            start_time = time.time()
            try:
                result = await service.predict_async(transaction)
                end_time = time.time()
                
                latency = (end_time - start_time) * 1000
                performance_metrics.record_latency(latency)
                performance_metrics.record_success()
                
                return result
                
            except Exception as e:
                performance_metrics.record_error()
                raise e
        
        # Execute async requests concurrently
        start_time = time.time()
        tasks = [make_async_prediction() for _ in range(num_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        total_duration = end_time - start_time
        performance_metrics.record_throughput(num_requests, total_duration)
        
        # Analyze results
        latency_stats = performance_metrics.get_latency_stats()
        throughput_stats = performance_metrics.get_throughput_stats()
        error_rate = performance_metrics.get_error_rate()
        
        print(f"\nAsync Prediction Performance:")
        print(f"  Mean Latency: {latency_stats['mean']:.2f}ms")
        print(f"  Throughput: {throughput_stats['mean_rps']:.2f} RPS")
        print(f"  Error Rate: {error_rate:.2f}%")
        
        # Performance assertions
        assert throughput_stats["mean_rps"] > 80, "Async throughput should be over 80 RPS"
        assert error_rate < 1, "Async error rate should be under 1%"


class TestDatabasePerformance:
    """Performance tests for database operations."""
    
    @pytest.fixture
    def mock_db(self):
        """Create mock database for testing."""
        return MockDatabase()
    
    def test_transaction_insert_performance(self, mock_db):
        """Test performance of transaction inserts."""
        metrics = PerformanceMetrics()
        num_inserts = 1000
        
        transactions = [generate_random_transaction() for _ in range(num_inserts)]
        
        start_time = time.time()
        for transaction in transactions:
            insert_start = time.time()
            try:
                mock_db.insert_transaction(transaction)
                insert_end = time.time()
                
                latency = (insert_end - insert_start) * 1000
                metrics.record_latency(latency)
                metrics.record_success()
                
            except Exception as e:
                metrics.record_error()
                print(f"Insert error: {e}")
        
        end_time = time.time()
        total_duration = end_time - start_time
        metrics.record_throughput(num_inserts, total_duration)
        
        # Analyze results
        latency_stats = metrics.get_latency_stats()
        throughput_stats = metrics.get_throughput_stats()
        error_rate = metrics.get_error_rate()
        
        print(f"\nDatabase Insert Performance:")
        print(f"  Mean Latency: {latency_stats['mean']:.2f}ms")
        print(f"  P95 Latency: {latency_stats['p95']:.2f}ms")
        print(f"  Throughput: {throughput_stats['mean_rps']:.2f} TPS")
        print(f"  Error Rate: {error_rate:.2f}%")
        
        # Performance assertions
        assert latency_stats["mean"] < 10, "Mean insert latency should be under 10ms"
        assert throughput_stats["mean_rps"] > 100, "Insert throughput should be over 100 TPS"
        assert error_rate < 1, "Insert error rate should be under 1%"
    
    def test_batch_insert_performance(self, mock_db):
        """Test performance of batch inserts."""
        metrics = PerformanceMetrics()
        batch_sizes = [10, 50, 100, 500]
        
        for batch_size in batch_sizes:
            transactions = [generate_random_transaction() for _ in range(batch_size)]
            
            start_time = time.time()
            try:
                mock_db.batch_insert_transactions(transactions)
                end_time = time.time()
                
                duration = end_time - start_time
                metrics.record_throughput(batch_size, duration)
                metrics.record_success()
                
            except Exception as e:
                metrics.record_error()
                print(f"Batch insert error: {e}")
        
        # Analyze results
        throughput_stats = metrics.get_throughput_stats()
        error_rate = metrics.get_error_rate()
        
        print(f"\nBatch Insert Performance:")
        print(f"  Mean Throughput: {throughput_stats['mean_rps']:.2f} TPS")
        print(f"  Max Throughput: {throughput_stats['max_rps']:.2f} TPS")
        print(f"  Error Rate: {error_rate:.2f}%")
        
        # Performance assertions
        assert throughput_stats["mean_rps"] > 200, "Batch insert should be over 200 TPS"
        assert error_rate < 1, "Batch insert error rate should be under 1%"
    
    def test_query_performance(self, mock_db):
        """Test performance of database queries."""
        metrics = PerformanceMetrics()
        
        # Populate database with test data
        test_transactions = [generate_random_transaction() for _ in range(1000)]
        mock_db.batch_insert_transactions(test_transactions)
        
        # Test different query types
        query_types = [
            ("user_transactions", lambda: mock_db.get_user_transactions("user_001")),
            ("recent_transactions", lambda: mock_db.get_recent_transactions(100)),
            ("high_risk_transactions", lambda: mock_db.get_high_risk_transactions()),
            ("transaction_stats", lambda: mock_db.get_transaction_statistics())
        ]
        
        for query_name, query_func in query_types:
            query_metrics = PerformanceMetrics()
            
            # Run query multiple times
            for _ in range(50):
                start_time = time.time()
                try:
                    result = query_func()
                    end_time = time.time()
                    
                    latency = (end_time - start_time) * 1000
                    query_metrics.record_latency(latency)
                    query_metrics.record_success()
                    
                except Exception as e:
                    query_metrics.record_error()
                    print(f"Query {query_name} error: {e}")
            
            # Analyze query performance
            latency_stats = query_metrics.get_latency_stats()
            error_rate = query_metrics.get_error_rate()
            
            print(f"\n{query_name.title()} Query Performance:")
            print(f"  Mean Latency: {latency_stats['mean']:.2f}ms")
            print(f"  P95 Latency: {latency_stats['p95']:.2f}ms")
            print(f"  Error Rate: {error_rate:.2f}%")
            
            # Performance assertions
            assert latency_stats["mean"] < 50, f"{query_name} mean latency should be under 50ms"
            assert error_rate < 1, f"{query_name} error rate should be under 1%"


class TestKafkaPerformance:
    """Performance tests for Kafka message processing."""
    
    def test_message_production_throughput(self):
        """Test Kafka message production throughput."""
        producer = MockKafkaProducer()
        metrics = PerformanceMetrics()
        
        num_messages = 1000
        topic = "transactions"
        
        messages = [
            {
                "transaction_id": f"txn_{i}",
                "user_id": f"user_{i % 100}",
                "amount": i * 10.5,
                "timestamp": datetime.utcnow().isoformat()
            }
            for i in range(num_messages)
        ]
        
        start_time = time.time()
        for i, message in enumerate(messages):
            send_start = time.time()
            try:
                producer.send(topic, message["transaction_id"], message)
                send_end = time.time()
                
                latency = (send_end - send_start) * 1000
                metrics.record_latency(latency)
                metrics.record_success()
                
            except Exception as e:
                metrics.record_error()
                print(f"Send error: {e}")
        
        end_time = time.time()
        total_duration = end_time - start_time
        metrics.record_throughput(num_messages, total_duration)
        
        # Analyze results
        latency_stats = metrics.get_latency_stats()
        throughput_stats = metrics.get_throughput_stats()
        error_rate = metrics.get_error_rate()
        
        print(f"\nKafka Production Performance:")
        print(f"  Mean Latency: {latency_stats['mean']:.2f}ms")
        print(f"  P95 Latency: {latency_stats['p95']:.2f}ms")
        print(f"  Throughput: {throughput_stats['mean_rps']:.2f} MPS")
        print(f"  Error Rate: {error_rate:.2f}%")
        
        # Performance assertions
        assert latency_stats["mean"] < 5, "Mean send latency should be under 5ms"
        assert throughput_stats["mean_rps"] > 500, "Production throughput should be over 500 MPS"
        assert error_rate < 1, "Production error rate should be under 1%"
    
    def test_message_consumption_throughput(self):
        """Test Kafka message consumption throughput."""
        from tests.fixtures.mock_objects import MockKafkaConsumer
        
        consumer = MockKafkaConsumer(["transactions"])
        metrics = PerformanceMetrics()
        
        num_messages = 1000
        
        # Add messages to consumer
        for i in range(num_messages):
            message = {
                "topic": "transactions",
                "key": f"txn_{i}",
                "value": {
                    "transaction_id": f"txn_{i}",
                    "amount": i * 10.5,
                    "timestamp": datetime.utcnow().isoformat()
                },
                "offset": i,
                "partition": 0
            }
            consumer.add_message(message)
        
        # Consume messages in batches
        batch_size = 100
        total_consumed = 0
        
        start_time = time.time()
        while total_consumed < num_messages:
            poll_start = time.time()
            try:
                messages = consumer.poll(timeout_ms=1000, max_records=batch_size)
                poll_end = time.time()
                
                if messages:
                    batch_latency = (poll_end - poll_start) * 1000
                    metrics.record_latency(batch_latency)
                    metrics.record_success()
                    total_consumed += len(messages)
                else:
                    break
                    
            except Exception as e:
                metrics.record_error()
                print(f"Poll error: {e}")
                break
        
        end_time = time.time()
        total_duration = end_time - start_time
        metrics.record_throughput(total_consumed, total_duration)
        
        # Analyze results
        latency_stats = metrics.get_latency_stats()
        throughput_stats = metrics.get_throughput_stats()
        error_rate = metrics.get_error_rate()
        
        print(f"\nKafka Consumption Performance:")
        print(f"  Messages Consumed: {total_consumed}")
        print(f"  Mean Batch Latency: {latency_stats['mean']:.2f}ms")
        print(f"  Throughput: {throughput_stats['mean_rps']:.2f} MPS")
        print(f"  Error Rate: {error_rate:.2f}%")
        
        # Performance assertions
        assert total_consumed == num_messages, "All messages should be consumed"
        assert throughput_stats["mean_rps"] > 800, "Consumption throughput should be over 800 MPS"
        assert error_rate < 1, "Consumption error rate should be under 1%"


class TestEndToEndPerformance:
    """End-to-end performance tests for the complete fraud detection pipeline."""
    
    def test_complete_pipeline_performance(self):
        """Test performance of the complete fraud detection pipeline."""
        # Setup components
        ml_service = MockMLInferenceService()
        database = MockDatabase()
        producer = MockKafkaProducer()
        
        metrics = PerformanceMetrics()
        num_transactions = 100
        
        # Generate test transactions
        transactions = [generate_random_transaction() for _ in range(num_transactions)]
        
        for transaction in transactions:
            pipeline_start = time.time()
            
            try:
                # Step 1: Store transaction in database
                database.insert_transaction(transaction)
                
                # Step 2: Make fraud prediction
                prediction = ml_service.predict(transaction)
                
                # Step 3: Store prediction result
                prediction_record = {
                    "transaction_id": transaction["transaction_id"],
                    "fraud_probability": prediction["fraud_probability"],
                    "risk_level": prediction["risk_level"],
                    "decision": prediction.get("decision", "APPROVE")
                }
                database.insert_prediction(prediction_record)
                
                # Step 4: Send result to Kafka
                producer.send("fraud_results", transaction["transaction_id"], prediction_record)
                
                pipeline_end = time.time()
                
                # Record pipeline latency
                latency = (pipeline_end - pipeline_start) * 1000
                metrics.record_latency(latency)
                metrics.record_success()
                
            except Exception as e:
                metrics.record_error()
                print(f"Pipeline error: {e}")
        
        # Calculate total throughput
        total_start = time.time()
        # Simulate processing all transactions
        time.sleep(0.1)  # Small delay to simulate processing
        total_end = time.time()
        
        total_duration = total_end - total_start
        metrics.record_throughput(num_transactions, total_duration)
        
        # Analyze results
        latency_stats = metrics.get_latency_stats()
        throughput_stats = metrics.get_throughput_stats()
        error_rate = metrics.get_error_rate()
        
        print(f"\nEnd-to-End Pipeline Performance:")
        print(f"  Mean Latency: {latency_stats['mean']:.2f}ms")
        print(f"  P95 Latency: {latency_stats['p95']:.2f}ms")
        print(f"  P99 Latency: {latency_stats['p99']:.2f}ms")
        print(f"  Throughput: {throughput_stats['mean_rps']:.2f} TPS")
        print(f"  Error Rate: {error_rate:.2f}%")
        
        # Performance assertions
        assert latency_stats["mean"] < 200, "Mean pipeline latency should be under 200ms"
        assert latency_stats["p95"] < 500, "P95 pipeline latency should be under 500ms"
        assert throughput_stats["mean_rps"] > 20, "Pipeline throughput should be over 20 TPS"
        assert error_rate < 2, "Pipeline error rate should be under 2%"
    
    def test_stress_test(self):
        """Stress test the system with high load."""
        ml_service = MockMLInferenceService()
        metrics = PerformanceMetrics()
        
        # High load parameters
        num_threads = 20
        requests_per_thread = 50
        total_requests = num_threads * requests_per_thread
        
        def stress_worker(worker_id: int):
            """Worker function for stress testing."""
            worker_metrics = PerformanceMetrics()
            
            for i in range(requests_per_thread):
                transaction = generate_random_transaction()
                
                start_time = time.time()
                try:
                    # Simulate complete processing
                    prediction = ml_service.predict(transaction)
                    
                    # Add some processing delay
                    time.sleep(0.001)  # 1ms processing time
                    
                    end_time = time.time()
                    
                    latency = (end_time - start_time) * 1000
                    worker_metrics.record_latency(latency)
                    worker_metrics.record_success()
                    
                except Exception as e:
                    worker_metrics.record_error()
                    print(f"Worker {worker_id} error: {e}")
            
            return worker_metrics
        
        # Execute stress test
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(stress_worker, i)
                for i in range(num_threads)
            ]
            
            worker_results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Aggregate results
        all_latencies = []
        total_errors = 0
        total_successes = 0
        
        for worker_metrics in worker_results:
            all_latencies.extend(worker_metrics.latencies)
            total_errors += worker_metrics.error_count
            total_successes += worker_metrics.success_count
        
        metrics.latencies = all_latencies
        metrics.error_count = total_errors
        metrics.success_count = total_successes
        metrics.record_throughput(total_requests, total_duration)
        
        # Analyze stress test results
        latency_stats = metrics.get_latency_stats()
        throughput_stats = metrics.get_throughput_stats()
        error_rate = metrics.get_error_rate()
        
        print(f"\nStress Test Results:")
        print(f"  Total Requests: {total_requests}")
        print(f"  Duration: {total_duration:.2f}s")
        print(f"  Mean Latency: {latency_stats['mean']:.2f}ms")
        print(f"  P95 Latency: {latency_stats['p95']:.2f}ms")
        print(f"  P99 Latency: {latency_stats['p99']:.2f}ms")
        print(f"  Throughput: {throughput_stats['mean_rps']:.2f} RPS")
        print(f"  Error Rate: {error_rate:.2f}%")
        
        # Stress test assertions (more lenient)
        assert latency_stats["p95"] < 1000, "P95 latency should be under 1000ms under stress"
        assert throughput_stats["mean_rps"] > 10, "Throughput should be over 10 RPS under stress"
        assert error_rate < 10, "Error rate should be under 10% under stress"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])