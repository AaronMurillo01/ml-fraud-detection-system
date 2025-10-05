# Multi-stage build for fraud detection system
FROM python:3.10-slim as base

# Set build arguments
ARG VERSION=dev
ARG BUILD_DATE
ARG VCS_REF

# Set labels for metadata
LABEL org.opencontainers.image.title="Fraud Detection System" \
      org.opencontainers.image.description="Real-time fraud detection ML system" \
      org.opencontainers.image.version="$VERSION" \
      org.opencontainers.image.created="$BUILD_DATE" \
      org.opencontainers.image.revision="$VCS_REF" \
      org.opencontainers.image.vendor="ML CRML" \
      org.opencontainers.image.source="https://github.com/ml-crml/fraud-detection"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# ============================================================================
# Dependencies stage
# ============================================================================
FROM base as dependencies

# Copy requirements files
COPY requirements.txt requirements-prod.txt ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -r requirements-prod.txt

# ============================================================================
# Development stage
# ============================================================================
FROM dependencies as development

# Copy development requirements
COPY requirements-dev.txt ./
RUN pip install -r requirements-dev.txt

# Copy source code
COPY . .

# Change ownership to appuser
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose ports
EXPOSE 8000 8001 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command for development
CMD ["python", "-m", "uvicorn", "service.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# ============================================================================
# Production stage
# ============================================================================
FROM dependencies as production

# Copy only necessary files for production
COPY service/ ./service/
COPY features/ ./features/
COPY streaming/ ./streaming/
COPY config/ ./config/
COPY migrations/ ./migrations/
COPY scripts/ ./scripts/
COPY alembic.ini ./
COPY pyproject.toml ./
COPY README.md ./

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/models

# Change ownership to appuser
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose ports
EXPOSE 8000 8001 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command for production
CMD ["python", "-m", "gunicorn", "api.main:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--max-requests", "1000", \
     "--max-requests-jitter", "100", \
     "--preload", \
     "--timeout", "30", \
     "--keep-alive", "5", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--log-level", "info"]

# ============================================================================
# ML Model Training stage
# ============================================================================
FROM dependencies as ml-training

# Install additional ML dependencies
RUN pip install \
    jupyter \
    notebook \
    mlflow \
    optuna \
    shap \
    lime

# Copy source code
COPY . .

# Create directories for ML artifacts
RUN mkdir -p /app/notebooks /app/experiments /app/models /app/data

# Change ownership to appuser
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose Jupyter port
EXPOSE 8888

# Default command for ML training
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# ============================================================================
# Streaming stage
# ============================================================================
FROM dependencies as streaming

# Copy streaming-specific files
COPY streaming/ ./streaming/
COPY service/models/ ./service/models/
COPY features/ ./features/
COPY config/ ./config/

# Change ownership to appuser
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Default command for streaming
CMD ["python", "-m", "streaming.kafka_consumer"]

# ============================================================================
# Testing stage
# ============================================================================
FROM development as testing

# Copy test files
COPY tests/ ./tests/
COPY pytest.ini ./
COPY .coveragerc ./

# Install test dependencies
COPY requirements-test.txt ./
RUN pip install -r requirements-test.txt

# Default command for testing
CMD ["pytest", "tests/", "-v", "--cov=service", "--cov=features", "--cov=streaming"]