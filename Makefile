# Fraud Detection System - Development Makefile

.PHONY: help install test test-unit test-integration test-performance test-all test-smoke test-security
.PHONY: lint format coverage clean docker-test benchmark setup-dev setup-ci
.PHONY: run-api run-training run-streaming docs deploy

# Default target
help:
	@echo "Fraud Detection System - Available Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  install        Install dependencies"
	@echo "  setup-dev      Set up development environment"
	@echo "  setup-ci       Set up CI environment"
	@echo ""
	@echo "Testing:"
	@echo "  test           Run all tests"
	@echo "  test-unit      Run unit tests"
	@echo "  test-integration Run integration tests"
	@echo "  test-performance Run performance tests"
	@echo "  test-smoke     Run smoke tests"
	@echo "  test-security  Run security tests"
	@echo "  benchmark      Run benchmark tests"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint           Run linting checks"
	@echo "  format         Format code"
	@echo "  coverage       Generate coverage report"
	@echo ""
	@echo "Development:"
	@echo "  run-api        Start API server"
	@echo "  run-training   Run model training"
	@echo "  run-streaming  Start streaming processor"
	@echo "  docker-test    Test Docker build"
	@echo ""
	@echo "Utilities:"
	@echo "  clean          Clean build artifacts"
	@echo "  docs           Generate documentation"
	@echo "  deploy         Deploy to production"

# Installation and setup
install:
	@echo "📦 Installing dependencies..."
	poetry install

setup-dev: install
	@echo "🛠️  Setting up development environment..."
	poetry run pre-commit install
	@echo "✅ Development environment ready!"

setup-ci:
	@echo "🔧 Setting up CI environment..."
	pip install poetry
	poetry config virtualenvs.create false
	poetry install --no-dev

# Testing
test: test-lint test-unit test-integration test-performance
	@echo "✅ All tests completed!"

test-unit:
	@echo "🧪 Running unit tests..."
	poetry run python scripts/run_tests.py unit --verbose

test-integration:
	@echo "🔗 Running integration tests..."
	poetry run python scripts/run_tests.py integration --verbose

test-performance:
	@echo "⚡ Running performance tests..."
	poetry run python scripts/run_tests.py performance --verbose

test-all:
	@echo "🚀 Running all tests..."
	poetry run python scripts/run_tests.py all --verbose

test-smoke:
	@echo "💨 Running smoke tests..."
	poetry run python scripts/run_tests.py smoke --verbose

test-security:
	@echo "🔒 Running security tests..."
	poetry run python scripts/run_tests.py security --verbose

benchmark:
	@echo "📈 Running benchmarks..."
	poetry run python scripts/run_tests.py benchmark --verbose

# Code quality
lint:
	@echo "🧹 Running linting checks..."
	poetry run python scripts/run_tests.py lint

format:
	@echo "🎨 Formatting code..."
	poetry run black .
	poetry run isort .
	@echo "✅ Code formatted!"

coverage:
	@echo "📊 Generating coverage report..."
	poetry run python scripts/run_tests.py unit --coverage-report
	@echo "📈 Coverage report generated in htmlcov/"

# Development servers
run-api:
	@echo "🚀 Starting API server..."
	poetry run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

run-training:
	@echo "🤖 Starting model training..."
	poetry run python training/train_models.py

run-streaming:
	@echo "🌊 Starting streaming processor..."
	poetry run python streaming/stream_processor.py

# Docker
docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t fraud-detection:latest .

docker-test: docker-build
	@echo "🧪 Testing Docker image..."
	docker run --rm fraud-detection:latest python -c "import api.main; print('✅ Import successful')"
	docker run -d --name fraud-detection-test -p 8000:8000 fraud-detection:latest
	sleep 10
	curl -f http://localhost:8000/health || (docker stop fraud-detection-test && exit 1)
	docker stop fraud-detection-test
	@echo "✅ Docker test passed!"

docker-run: docker-build
	@echo "🚀 Running Docker container..."
	docker run -p 8000:8000 --env-file .env fraud-detection:latest

# Database
db-migrate:
	@echo "🗄️  Running database migrations..."
	poetry run alembic upgrade head

db-reset:
	@echo "🔄 Resetting database..."
	poetry run alembic downgrade base
	poetry run alembic upgrade head

db-seed:
	@echo "🌱 Seeding database..."
	poetry run python scripts/seed_database.py

# Monitoring
start-monitoring:
	@echo "📊 Starting monitoring stack..."
	docker-compose -f docker/monitoring/docker-compose.yml up -d

stop-monitoring:
	@echo "⏹️  Stopping monitoring stack..."
	docker-compose -f docker/monitoring/docker-compose.yml down

# Documentation
docs:
	@echo "📚 Generating documentation..."
	poetry run sphinx-build -b html docs/ docs/_build/html
	@echo "📖 Documentation available at docs/_build/html/index.html"

docs-serve:
	@echo "🌐 Serving documentation..."
	poetry run python -m http.server 8080 --directory docs/_build/html

# Deployment
deploy-staging:
	@echo "🚀 Deploying to staging..."
	# Add staging deployment commands here
	@echo "✅ Deployed to staging!"

deploy-production:
	@echo "🚀 Deploying to production..."
	# Add production deployment commands here
	@echo "✅ Deployed to production!"

# Utilities
clean:
	@echo "🧽 Cleaning build artifacts..."
	poetry run python scripts/run_tests.py all --clean
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ coverage.xml test-results*.xml
	rm -rf bandit-report.json safety-report.json benchmark-results.json
	rm -rf .coverage test.db test_models/
	rm -rf build/ dist/ *.egg-info/
	@echo "✅ Cleaned!"

clean-all: clean
	@echo "🧽 Deep cleaning..."
	poetry env remove --all 2>/dev/null || true
	docker system prune -f
	@echo "✅ Deep cleaned!"

# Security
security-scan:
	@echo "🔍 Running security scan..."
	poetry run bandit -r . -f json -o bandit-report.json
	poetry run safety check --json --output safety-report.json
	@echo "🔒 Security scan completed!"

# Performance profiling
profile-api:
	@echo "📊 Profiling API performance..."
	poetry run python -m cProfile -o api_profile.prof -m uvicorn api.main:app --host 0.0.0.0 --port 8000

profile-training:
	@echo "📊 Profiling training performance..."
	poetry run python -m cProfile -o training_profile.prof training/train_models.py

# Load testing
load-test:
	@echo "🔥 Running load tests..."
	poetry run locust -f tests/load/locustfile.py --host=http://localhost:8000

# Environment management
env-create:
	@echo "🌍 Creating environment file..."
	cp .env.example .env
	@echo "✏️  Please edit .env with your configuration"

env-validate:
	@echo "✅ Validating environment configuration..."
	poetry run python -c "from config.settings import Settings; Settings()"
	@echo "✅ Environment configuration is valid!"

# Git hooks
install-hooks:
	@echo "🪝 Installing git hooks..."
	poetry run pre-commit install
	poetry run pre-commit install --hook-type commit-msg
	@echo "✅ Git hooks installed!"

run-hooks:
	@echo "🪝 Running pre-commit hooks..."
	poetry run pre-commit run --all-files

# Quick development commands
dev: setup-dev
	@echo "🚀 Starting development environment..."
	make run-api

check: lint test-unit
	@echo "✅ Quick check completed!"

ci: setup-ci test-all security-scan
	@echo "✅ CI pipeline completed!"

# Help for specific commands
help-test:
	@echo "Testing Commands:"
	@echo "  test           - Run all tests (unit + integration + performance)"
	@echo "  test-unit      - Run unit tests only"
	@echo "  test-integration - Run integration tests (requires external services)"
	@echo "  test-performance - Run performance and benchmark tests"
	@echo "  test-smoke     - Run basic smoke tests"
	@echo "  test-security  - Run security scans and tests"
	@echo "  benchmark      - Run detailed benchmarks"

help-docker:
	@echo "Docker Commands:"
	@echo "  docker-build   - Build Docker image"
	@echo "  docker-test    - Test Docker image functionality"
	@echo "  docker-run     - Run application in Docker container"

help-dev:
	@echo "Development Commands:"
	@echo "  setup-dev      - Set up complete development environment"
	@echo "  run-api        - Start API server with hot reload"
	@echo "  run-training   - Run model training pipeline"
	@echo "  run-streaming  - Start real-time streaming processor"
	@echo "  format         - Format code with black and isort"
	@echo "  lint           - Run all linting checks"