# FraudGuard AI

A production-ready, real-time credit card fraud detection system powered by machine learning. Built with FastAPI, XGBoost, and modern MLOps practices.

---

## Features

### Core Capabilities

- Real-Time Detection: Analyze transactions with sub-second response times (under 100ms)
- Multiple ML Models: XGBoost, Random Forest, and Neural Networks with ensemble predictions
- Advanced Feature Engineering: 50+ engineered features including velocity checks, behavioral patterns, and risk scoring
- Scalable Architecture: Kubernetes-ready with horizontal pod autoscaling
- Streaming Support: Apache Kafka integration for real-time event processing
- Comprehensive Monitoring: Prometheus metrics and Grafana dashboards
- API-First Design: RESTful API with OpenAPI/Swagger documentation

### Production Features

- High Availability: Multi-replica deployment with health checks
- Caching Layer: Redis for feature caching and rate limiting
- Database: PostgreSQL with async support and connection pooling
- Security: JWT authentication, rate limiting, and API key management
- Observability: Structured logging, distributed tracing, and alerting
- CI/CD: Automated testing, building, and deployment pipelines

---

## Quick Start

### Prerequisites

- Python 3.10 or higher
- Docker and Docker Compose
- PostgreSQL 13+
- Redis 6+

### Installation

1. Clone the repository
   ```bash
   git clone https://github.com/AaronMurillo01/DNA_website.git
   cd "Desktop/Notes V/vs code/ML CRML"
   ```

2. Set up environment variables
   ```bash
   cp .env.example .env
   # Edit .env with your configuration - IMPORTANT: Set SECRET_KEY
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Run database migrations
   ```bash
   make db-migrate
   ```

5. Start the application
   ```bash
   python run_app.py
   ```

6. Access the dashboard
   - Web UI: http://localhost:8000
   - API Docs: http://localhost:8000/docs
   - Monitoring: http://localhost:8000/monitoring.html

### Using Docker

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

## Technology Stack

### Backend
- Python 3.10+: Core programming language
- FastAPI: Modern, high-performance web framework
- Uvicorn: ASGI server for production
- Pydantic: Data validation and settings management

### Machine Learning
- scikit-learn: ML algorithms and preprocessing
- XGBoost: Gradient boosting for fraud detection
- LightGBM: Fast gradient boosting framework
- SHAP: Model explainability and feature importance
- NumPy & Pandas: Data manipulation and analysis

### Data Storage
- PostgreSQL: Primary relational database
- Redis: Caching and session management
- SQLAlchemy: ORM with async support
- Alembic: Database migrations

### Streaming & Messaging
- Apache Kafka: Real-time event streaming
- kafka-python: Kafka client library

### Monitoring & Observability
- Prometheus: Metrics collection and alerting
- Grafana: Visualization and dashboards
- OpenTelemetry: Distributed tracing
- Structlog: Structured logging

### DevOps & Infrastructure
- Docker: Containerization
- Kubernetes: Container orchestration
- GitHub Actions: CI/CD pipelines
- Helm: Kubernetes package management

---

## Project Structure

```
fraudguard-ai/
├── api/                    # API endpoints and routing
├── service/               # Core business logic
├── features/              # Feature engineering
├── training/              # Model training pipelines
├── streaming/             # Kafka streaming
├── database/              # Database models and repositories
├── config/                # Configuration management
├── monitoring/            # Monitoring and health checks
├── tests/                 # Test suite
├── migrations/            # Database migrations
├── k8s/                   # Kubernetes manifests
├── infra/                 # Infrastructure as code
├── docs/                  # Documentation
├── static/                # Frontend assets
├── scripts/               # Utility scripts
├── Dockerfile             # Container definition
├── docker-compose.yml     # Local development setup
├── pyproject.toml         # Project metadata
├── requirements.txt       # Python dependencies
└── Makefile              # Development commands
```

---

## Configuration

### Environment Variables

Key environment variables (see `.env.example` for complete list):

```bash
# Application
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

# Security (REQUIRED)
SECRET_KEY=<generate-secure-key>

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/fraud_detection

# Redis
REDIS_URL=redis://localhost:6379/0

# Kafka (optional)
KAFKA_BOOTSTRAP_SERVERS=localhost:9092

# ML Models
ML_MODEL_PATH=./models
FRAUD_THRESHOLD=0.5
```

---

## Usage

### API Examples

#### Predict Fraud for Single Transaction

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction": {
      "transaction_id": "txn_123456",
      "user_id": "user_789",
      "amount": 150.75,
      "currency": "USD",
      "merchant_id": "merchant_abc",
      "timestamp": "2024-01-15T10:30:00Z"
    }
  }'
```

#### Batch Prediction

```bash
curl -X POST "http://localhost:8000/api/v1/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [...]
  }'
```

### Python Client Example

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/predict",
    json={
        "transaction": {
            "transaction_id": "txn_123",
            "user_id": "user_456",
            "amount": 99.99,
            "currency": "USD",
            "merchant_id": "merchant_789"
        }
    }
)

result = response.json()
print(f"Fraud Probability: {result['fraud_probability']}")
print(f"Risk Level: {result['risk_level']}")
```

---

## API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json

See docs/API_DOCUMENTATION.md for detailed API documentation.

---

## Testing

### Run All Tests

```bash
make test
```

### Run Specific Test Categories

```bash
# Unit tests only
make test-unit

# Integration tests
make test-integration

# Performance tests
make test-performance

# With coverage report
make coverage
```

### Code Quality

```bash
# Format code
make format

# Run linting
make lint

# Type checking
mypy .

# Security scan
make security-scan
```

---

## Deployment

### Docker Deployment

```bash
# Build image
docker build -t fraudguard-ai:latest .

# Run container
docker run -d \
  -p 8000:8000 \
  -e SECRET_KEY=your-secret-key \
  -e DATABASE_URL=postgresql://... \
  fraudguard-ai:latest
```

### Kubernetes Deployment

```bash
# Apply configurations
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap-secrets.yaml
kubectl apply -f k8s/deployment.yaml

# Check status
kubectl get pods -n fraud-detection
```

See docs/DEPLOYMENT.md for detailed deployment instructions.

---

## Contributing

We welcome contributions! Please see CONTRIBUTING.md for guidelines.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Acknowledgments

Built with modern technologies and best practices for production-ready machine learning systems.

---

## Support

For questions, issues, or feature requests:

- Open an issue on GitHub
- Check the documentation in docs/
- Review existing issues and discussions

