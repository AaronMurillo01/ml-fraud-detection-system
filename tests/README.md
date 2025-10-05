# Fraud Detection System - Testing Suite

This directory contains a comprehensive testing suite for the fraud detection system, including unit tests, integration tests, performance tests, and test utilities.

## Table of Contents

- [Overview](#overview)
- [Test Structure](#test-structure)
- [Running Tests](#running-tests)
- [Test Categories](#test-categories)
- [Test Configuration](#test-configuration)
- [Writing Tests](#writing-tests)
- [CI/CD Integration](#cicd-integration)
- [Troubleshooting](#troubleshooting)

## Overview

The testing suite is designed to ensure the reliability, performance, and correctness of the fraud detection system. It covers:

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions and external dependencies
- **Performance Tests**: Benchmark system performance and identify bottlenecks
- **End-to-End Tests**: Test complete workflows from API to database

## Test Structure

```
tests/
├── README.md                     # This file
├── conftest.py                   # Global pytest configuration and fixtures
├── unit/                         # Unit tests
│   ├── __init__.py
│   ├── test_ml_inference.py      # ML service unit tests
│   ├── test_model_loader.py      # Model loader unit tests
│   └── test_service_models.py    # Service models unit tests
├── integration/                  # Integration tests
│   ├── __init__.py
│   ├── test_api_endpoints.py     # API endpoint tests
│   ├── test_database_integration.py  # Database integration tests
│   └── test_kafka_integration.py # Kafka integration tests
├── performance/                  # Performance tests
│   ├── __init__.py
│   └── test_performance.py       # Performance benchmarks
└── fixtures/                     # Test fixtures and utilities
    ├── __init__.py
    ├── mock_objects.py           # Mock classes and objects
    ├── test_data.py              # Sample test data
    └── database_fixtures.py      # Database test fixtures
```

## Running Tests

### Prerequisites

1. Install test dependencies:
   ```bash
   pip install -r requirements-test.txt
   ```

2. Set up test environment variables (optional):
   ```bash
   export TESTING=true
   export LOG_LEVEL=DEBUG
   export DATABASE_URL=sqlite:///:memory:
   ```

### Basic Test Execution

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=service --cov=database --cov=shared

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m performance   # Performance tests only

# Run tests in specific directories
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/

# Run specific test files
pytest tests/unit/test_ml_inference.py
pytest tests/integration/test_api_endpoints.py

# Run specific test functions
pytest tests/unit/test_ml_inference.py::test_predict_single_transaction
pytest -k "test_predict"  # Run tests matching pattern
```

### Advanced Test Options

```bash
# Verbose output
pytest -v

# Show local variables in tracebacks
pytest -l

# Stop on first failure
pytest -x

# Run tests in parallel (requires pytest-xdist)
pytest -n auto

# Generate HTML coverage report
pytest --cov=service --cov-report=html

# Run only failed tests from last run
pytest --lf

# Run tests that failed, then continue with rest
pytest --ff
```

## Test Categories

### Unit Tests (`tests/unit/`)

Test individual components in isolation using mocks and stubs.

**Markers**: `@pytest.mark.unit`

**Coverage**:
- ML inference service functionality
- Model loading and caching
- Data model validation
- Utility functions
- Error handling

**Example**:
```bash
pytest tests/unit/ -v
```

### Integration Tests (`tests/integration/`)

Test component interactions and external dependencies.

**Markers**: `@pytest.mark.integration`

**Coverage**:
- API endpoint functionality
- Database operations
- Kafka message processing
- Service-to-service communication

**Example**:
```bash
pytest tests/integration/ -v
```

### Performance Tests (`tests/performance/`)

Benchmark system performance and identify bottlenecks.

**Markers**: `@pytest.mark.performance`, `@pytest.mark.slow`

**Coverage**:
- Latency measurements
- Throughput benchmarks
- Concurrent load testing
- Memory usage analysis

**Example**:
```bash
# Run performance tests (may take longer)
pytest tests/performance/ -v

# Skip slow tests
pytest -m "not slow"
```

## Test Configuration

### pytest.ini

The `pytest.ini` file in the project root contains:
- Test discovery patterns
- Coverage configuration
- Custom markers
- Output formatting
- Warning filters

### conftest.py

Global pytest configuration includes:
- Shared fixtures
- Test environment setup
- Mock objects
- Database fixtures
- Event loop configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|----------|
| `TESTING` | Enable test mode | `true` |
| `LOG_LEVEL` | Logging level | `DEBUG` |
| `DATABASE_URL` | Test database URL | `sqlite:///:memory:` |
| `KAFKA_BOOTSTRAP_SERVERS` | Kafka servers | `localhost:9092` |
| `RUN_SLOW_TESTS` | Enable slow tests | `false` |
| `KAFKA_AVAILABLE` | Kafka availability | `false` |
| `DATABASE_AVAILABLE` | Database availability | `true` |

## Writing Tests

### Test Naming Conventions

- Test files: `test_*.py`
- Test functions: `test_*`
- Test classes: `Test*`
- Fixtures: descriptive names without `test_` prefix

### Test Structure

```python
# tests/unit/test_example.py
import pytest
from unittest.mock import Mock, patch

from service.example_module import ExampleClass
from shared.models import ExampleModel


class TestExampleClass:
    """Test suite for ExampleClass."""
    
    @pytest.fixture
    def example_instance(self):
        """Create ExampleClass instance for testing."""
        return ExampleClass(config={"setting": "value"})
    
    def test_example_method_success(self, example_instance):
        """Test successful execution of example method."""
        # Arrange
        input_data = {"key": "value"}
        expected_result = {"processed": True}
        
        # Act
        result = example_instance.example_method(input_data)
        
        # Assert
        assert result == expected_result
    
    def test_example_method_error_handling(self, example_instance):
        """Test error handling in example method."""
        # Arrange
        invalid_input = None
        
        # Act & Assert
        with pytest.raises(ValueError, match="Invalid input"):
            example_instance.example_method(invalid_input)
    
    @pytest.mark.asyncio
    async def test_async_example_method(self, example_instance):
        """Test async method execution."""
        # Arrange
        input_data = {"async": True}
        
        # Act
        result = await example_instance.async_example_method(input_data)
        
        # Assert
        assert result is not None
```

### Using Fixtures

```python
def test_with_mock_service(mock_ml_service, sample_transaction):
    """Test using mock service and sample data."""
    # Use fixtures directly in test parameters
    prediction = mock_ml_service.predict(sample_transaction)
    assert prediction.risk_score >= 0.0

def test_with_database(populated_db_session, sample_user):
    """Test with populated database session."""
    # Database operations
    user = populated_db_session.query(User).first()
    assert user is not None
```

### Parametrized Tests

```python
@pytest.mark.parametrize("risk_level,expected_action", [
    ("low", "approve"),
    ("medium", "review"),
    ("high", "decline"),
])
def test_risk_based_action(risk_level, expected_action):
    """Test risk-based decision making."""
    action = determine_action(risk_level)
    assert action == expected_action
```

### Async Tests

```python
@pytest.mark.asyncio
async def test_async_prediction(mock_ml_service):
    """Test async prediction functionality."""
    transaction = generate_random_transaction()
    prediction = await mock_ml_service.predict_async(transaction)
    assert prediction.confidence > 0.0
```

## CI/CD Integration

### GitHub Actions Example

```yaml
# .github/workflows/test.yml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-test.txt
    
    - name: Run unit tests
      run: |
        pytest tests/unit/ --cov=service --cov=database --cov=shared
    
    - name: Run integration tests
      run: |
        pytest tests/integration/
    
    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### Test Commands for CI

```bash
# Fast test suite (unit tests only)
pytest tests/unit/ --cov=service --cov=database --cov=shared --cov-fail-under=80

# Full test suite
pytest --cov=service --cov=database --cov=shared --cov-report=xml --cov-fail-under=75

# Performance regression tests
pytest tests/performance/ -m "not slow"
```

## Test Data Management

### Sample Data

The `tests/fixtures/test_data.py` module provides:
- Pre-defined sample transactions, users, merchants
- Random data generators
- Batch data creation utilities

### Mock Objects

The `tests/fixtures/mock_objects.py` module provides:
- Mock ML inference service
- Mock database connections
- Mock Kafka producers/consumers
- Mock external API clients

### Database Fixtures

The `tests/fixtures/database_fixtures.py` module provides:
- In-memory SQLite databases
- Pre-populated test data
- Async database support
- Transaction rollback utilities

## Performance Testing

### Benchmarking

```python
def test_prediction_latency(benchmark, mock_ml_service, sample_transaction):
    """Benchmark prediction latency."""
    result = benchmark(mock_ml_service.predict, sample_transaction)
    assert result.risk_score is not None
```

### Load Testing

```python
@pytest.mark.performance
def test_concurrent_predictions(mock_ml_service):
    """Test concurrent prediction handling."""
    import concurrent.futures
    
    transactions = generate_batch_transactions(100)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(mock_ml_service.predict, t) for t in transactions]
        results = [f.result() for f in futures]
    
    assert len(results) == 100
    assert all(r.risk_score is not None for r in results)
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure project root is in Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

2. **Database Connection Issues**
   ```bash
   # Use in-memory database for tests
   export DATABASE_URL="sqlite:///:memory:"
   ```

3. **Async Test Issues**
   ```bash
   # Install pytest-asyncio
   pip install pytest-asyncio
   ```

4. **Coverage Issues**
   ```bash
   # Generate detailed coverage report
   pytest --cov=service --cov-report=html --cov-report=term-missing
   ```

### Debug Mode

```bash
# Run tests with debugging
pytest --pdb  # Drop into debugger on failures
pytest --pdbcls=IPython.terminal.debugger:Pdb  # Use IPython debugger

# Capture output
pytest -s  # Don't capture stdout/stderr

# Verbose logging
pytest --log-cli-level=DEBUG
```

### Test Isolation

```bash
# Run tests in random order
pytest --random-order

# Run specific test in isolation
pytest tests/unit/test_ml_inference.py::test_predict_single_transaction -v
```

## Best Practices

1. **Test Independence**: Each test should be independent and not rely on other tests
2. **Clear Naming**: Use descriptive test names that explain what is being tested
3. **Arrange-Act-Assert**: Structure tests with clear setup, execution, and verification phases
4. **Mock External Dependencies**: Use mocks for external services, databases, and APIs
5. **Test Edge Cases**: Include tests for error conditions and boundary values
6. **Performance Awareness**: Monitor test execution time and optimize slow tests
7. **Documentation**: Document complex test scenarios and setup requirements
8. **Continuous Integration**: Run tests automatically on code changes
9. **Coverage Goals**: Aim for high test coverage but focus on critical paths
10. **Regular Maintenance**: Keep tests updated as code evolves

## Contributing

When adding new tests:

1. Follow the existing test structure and naming conventions
2. Add appropriate markers (`@pytest.mark.unit`, `@pytest.mark.integration`, etc.)
3. Include docstrings explaining what the test validates
4. Use existing fixtures when possible
5. Add new fixtures to `conftest.py` if they're reusable
6. Update this README if adding new test categories or significant changes

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [pytest-cov](https://pytest-cov.readthedocs.io/)
- [unittest.mock](https://docs.python.org/3/library/unittest.mock.html)
- [Pydantic Testing](https://pydantic-docs.helpmanual.io/usage/testing/)