# Contributing to FraudGuard AI

Thank you for your interest in contributing to FraudGuard AI. This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for all contributors.

### Our Standards

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/fraudguard-ai.git
   cd fraudguard-ai
   ```
3. Add upstream remote:
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/fraudguard-ai.git
   ```

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Docker and Docker Compose
- PostgreSQL 13+
- Redis 6+
- Git

### Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-test.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. Run database migrations:
   ```bash
   make db-migrate
   ```

5. Start the development server:
   ```bash
   make run-api
   ```

### Using Docker

```bash
docker-compose up -d
```

## How to Contribute

### Types of Contributions

We welcome various types of contributions:

- Bug fixes: Fix issues in the codebase
- New features: Add new functionality
- Documentation: Improve or add documentation
- Tests: Add or improve test coverage
- Performance improvements: Optimize existing code
- Code refactoring: Improve code quality

### Contribution Workflow

1. Create a new branch for your work:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. Make your changes following our coding standards

3. Write or update tests for your changes

4. Run tests to ensure everything works:
   ```bash
   make test
   ```

5. Commit your changes with clear, descriptive messages:
   ```bash
   git commit -m "Add feature: description of your changes"
   ```

6. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

7. Create a Pull Request on GitHub

## Coding Standards

### Python Style Guide

We follow PEP 8 and use the following tools:

- Black for code formatting (line length: 88)
- isort for import sorting
- flake8 for linting
- mypy for type checking

### Code Formatting

Before committing, format your code:

```bash
make format
```

Or manually:

```bash
black .
isort .
```

### Type Hints

Use type hints for all function signatures:

```python
def process_transaction(transaction: Transaction) -> PredictionResult:
    """Process a transaction and return fraud prediction."""
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def calculate_risk_score(features: Dict[str, float]) -> float:
    """Calculate risk score from transaction features.
    
    Args:
        features: Dictionary of feature names to values
        
    Returns:
        Risk score between 0 and 1
        
    Raises:
        ValueError: If features are invalid
    """
    pass
```

### Naming Conventions

- Classes: PascalCase (e.g., `FraudDetector`)
- Functions/Methods: snake_case (e.g., `predict_fraud`)
- Constants: UPPER_SNAKE_CASE (e.g., `MAX_RETRIES`)
- Private methods: prefix with underscore (e.g., `_internal_method`)

## Testing Guidelines

### Running Tests

```bash
# Run all tests
make test

# Run specific test categories
make test-unit
make test-integration
make test-performance

# Run with coverage
make coverage
```

### Writing Tests

- Write tests for all new features
- Maintain or improve code coverage (minimum 80%)
- Use descriptive test names
- Follow the Arrange-Act-Assert pattern

Example:

```python
def test_fraud_detection_high_risk_transaction():
    """Test that high-risk transactions are flagged as fraud."""
    # Arrange
    transaction = create_high_risk_transaction()
    detector = FraudDetector()
    
    # Act
    result = detector.predict(transaction)
    
    # Assert
    assert result.is_fraud is True
    assert result.risk_level == RiskLevel.HIGH
```

### Test Markers

Use pytest markers to categorize tests:

```python
@pytest.mark.unit
def test_feature_extraction():
    pass

@pytest.mark.integration
def test_database_connection():
    pass

@pytest.mark.slow
def test_model_training():
    pass
```

## Pull Request Process

### Before Submitting

1. Update documentation if needed
2. Add tests for new functionality
3. Run the full test suite: `make test`
4. Run linting: `make lint`
5. Update CHANGELOG.md if applicable

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe the tests you ran

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] No new warnings
```

### Review Process

1. At least one maintainer must review and approve
2. All CI checks must pass
3. Code coverage must not decrease
4. Documentation must be updated if needed

## Reporting Issues

### Bug Reports

When reporting bugs, include:

- Description: Clear description of the bug
- Steps to reproduce: Detailed steps to reproduce the issue
- Expected behavior: What you expected to happen
- Actual behavior: What actually happened
- Environment: OS, Python version, dependencies
- Logs: Relevant error messages or logs

### Feature Requests

When requesting features, include:

- Description: Clear description of the feature
- Use case: Why this feature would be useful
- Proposed solution: How you think it should work
- Alternatives: Other solutions you've considered

## Questions?

If you have questions about contributing, feel free to:

- Open an issue with the `question` label
- Contact the maintainers
- Check existing documentation

## License

By contributing to FraudGuard AI, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to FraudGuard AI!

