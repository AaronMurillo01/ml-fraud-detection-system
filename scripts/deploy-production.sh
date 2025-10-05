#!/bin/bash

# Production Deployment Script for Fraud Detection System
# This script handles the complete production deployment process

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEPLOYMENT_ENV="${DEPLOYMENT_ENV:-production}"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
BACKUP_ENABLED="${BACKUP_ENABLED:-true}"
HEALTH_CHECK_TIMEOUT="${HEALTH_CHECK_TIMEOUT:-300}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Error handling
cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        log_error "Deployment failed with exit code $exit_code"
        log_info "Running cleanup procedures..."
        # Add cleanup logic here if needed
    fi
    exit $exit_code
}

trap cleanup EXIT

# Validation functions
validate_environment() {
    log_info "Validating deployment environment..."
    
    # Check required tools
    local required_tools=("docker" "docker-compose" "curl" "jq")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "Required tool '$tool' is not installed"
            exit 1
        fi
    done
    
    # Check environment variables
    local required_vars=("POSTGRES_PASSWORD" "GRAFANA_PASSWORD")
    for var in "${required_vars[@]}"; do
        if [ -z "${!var:-}" ]; then
            log_error "Required environment variable '$var' is not set"
            exit 1
        fi
    done
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    log_success "Environment validation completed"
}

# Pre-deployment checks
pre_deployment_checks() {
    log_info "Running pre-deployment checks..."
    
    # Check disk space (require at least 10GB free)
    local available_space=$(df "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
    local required_space=10485760  # 10GB in KB
    
    if [ "$available_space" -lt "$required_space" ]; then
        log_error "Insufficient disk space. Required: 10GB, Available: $(($available_space / 1024 / 1024))GB"
        exit 1
    fi
    
    # Check if ports are available
    local required_ports=(80 443 5432 6379 9092 9090 3000)
    for port in "${required_ports[@]}"; do
        if netstat -tuln | grep -q ":$port "; then
            log_warning "Port $port is already in use"
        fi
    done
    
    # Validate configuration files
    if [ ! -f "$PROJECT_ROOT/.env.production" ]; then
        log_error "Production environment file not found: .env.production"
        exit 1
    fi
    
    log_success "Pre-deployment checks completed"
}

# Database backup
backup_database() {
    if [ "$BACKUP_ENABLED" = "true" ]; then
        log_info "Creating database backup..."
        
        local backup_dir="$PROJECT_ROOT/backups"
        local backup_file="$backup_dir/fraud_detection_$(date +%Y%m%d_%H%M%S).sql"
        
        mkdir -p "$backup_dir"
        
        # Check if database is running
        if docker-compose -f "$PROJECT_ROOT/docker-compose.prod.yml" ps postgres | grep -q "Up"; then
            docker-compose -f "$PROJECT_ROOT/docker-compose.prod.yml" exec -T postgres \
                pg_dump -U fraud_user fraud_detection > "$backup_file"
            
            if [ -f "$backup_file" ] && [ -s "$backup_file" ]; then
                log_success "Database backup created: $backup_file"
            else
                log_error "Database backup failed"
                exit 1
            fi
        else
            log_info "Database not running, skipping backup"
        fi
    else
        log_info "Database backup disabled"
    fi
}

# Build and push images
build_images() {
    log_info "Building production images..."
    
    cd "$PROJECT_ROOT"
    
    # Set build arguments
    local build_args=(
        "--build-arg" "VERSION=${IMAGE_TAG}"
        "--build-arg" "BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
        "--build-arg" "VCS_REF=$(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
    )
    
    # Build main application image
    docker build "${build_args[@]}" \
        --target production \
        --tag "fraud-detection:${IMAGE_TAG}" \
        --tag "fraud-detection:latest" \
        .
    
    # Build streaming service image
    if [ -f "Dockerfile.streaming" ]; then
        docker build "${build_args[@]}" \
            --file Dockerfile.streaming \
            --target production \
            --tag "fraud-detection-streaming:${IMAGE_TAG}" \
            --tag "fraud-detection-streaming:latest" \
            .
    fi
    
    # Push to registry if configured
    if [ -n "$DOCKER_REGISTRY" ]; then
        log_info "Pushing images to registry: $DOCKER_REGISTRY"
        
        docker tag "fraud-detection:${IMAGE_TAG}" "$DOCKER_REGISTRY/fraud-detection:${IMAGE_TAG}"
        docker push "$DOCKER_REGISTRY/fraud-detection:${IMAGE_TAG}"
        
        if [ -f "Dockerfile.streaming" ]; then
            docker tag "fraud-detection-streaming:${IMAGE_TAG}" "$DOCKER_REGISTRY/fraud-detection-streaming:${IMAGE_TAG}"
            docker push "$DOCKER_REGISTRY/fraud-detection-streaming:${IMAGE_TAG}"
        fi
    fi
    
    log_success "Images built successfully"
}

# Deploy services
deploy_services() {
    log_info "Deploying production services..."
    
    cd "$PROJECT_ROOT"
    
    # Load environment variables
    if [ -f ".env.production" ]; then
        export $(grep -v '^#' .env.production | xargs)
    fi
    
    # Deploy infrastructure services first
    log_info "Starting infrastructure services..."
    docker-compose -f docker-compose.prod.yml up -d \
        postgres redis zookeeper kafka elasticsearch
    
    # Wait for infrastructure to be ready
    log_info "Waiting for infrastructure services to be ready..."
    sleep 30
    
    # Deploy application services
    log_info "Starting application services..."
    docker-compose -f docker-compose.prod.yml up -d \
        fraud-api streaming-processor
    
    # Deploy monitoring services
    log_info "Starting monitoring services..."
    docker-compose -f docker-compose.prod.yml up -d \
        prometheus grafana kibana nginx
    
    # Deploy exporters
    log_info "Starting monitoring exporters..."
    docker-compose -f docker-compose.prod.yml up -d \
        postgres-exporter redis-exporter node-exporter cadvisor
    
    log_success "Services deployed successfully"
}

# Health checks
health_checks() {
    log_info "Running health checks..."
    
    local services=(
        "http://localhost:8000/health:Fraud Detection API"
        "http://localhost:9090/-/healthy:Prometheus"
        "http://localhost:3000/api/health:Grafana"
    )
    
    local timeout=$HEALTH_CHECK_TIMEOUT
    local interval=10
    local elapsed=0
    
    for service_info in "${services[@]}"; do
        IFS=':' read -r url name <<< "$service_info"
        
        log_info "Checking health of $name..."
        
        while [ $elapsed -lt $timeout ]; do
            if curl -f -s "$url" > /dev/null 2>&1; then
                log_success "$name is healthy"
                break
            fi
            
            sleep $interval
            elapsed=$((elapsed + interval))
            
            if [ $elapsed -ge $timeout ]; then
                log_error "$name health check failed after ${timeout}s"
                return 1
            fi
        done
        
        elapsed=0
    done
    
    log_success "All health checks passed"
}

# Post-deployment tasks
post_deployment() {
    log_info "Running post-deployment tasks..."
    
    # Run database migrations
    log_info "Running database migrations..."
    docker-compose -f "$PROJECT_ROOT/docker-compose.prod.yml" exec -T fraud-api \
        python -m alembic upgrade head
    
    # Warm up ML models
    log_info "Warming up ML models..."
    docker-compose -f "$PROJECT_ROOT/docker-compose.prod.yml" exec -T fraud-api \
        python -c "from service.model_loader import ModelLoader; ModelLoader().preload_models()"
    
    # Create initial admin user if needed
    log_info "Setting up initial configuration..."
    docker-compose -f "$PROJECT_ROOT/docker-compose.prod.yml" exec -T fraud-api \
        python scripts/setup_initial_data.py
    
    log_success "Post-deployment tasks completed"
}

# Main deployment function
main() {
    log_info "Starting production deployment for Fraud Detection System"
    log_info "Environment: $DEPLOYMENT_ENV"
    log_info "Image Tag: $IMAGE_TAG"
    log_info "Registry: ${DOCKER_REGISTRY:-'local'}"
    
    validate_environment
    pre_deployment_checks
    backup_database
    build_images
    deploy_services
    health_checks
    post_deployment
    
    log_success "Production deployment completed successfully!"
    log_info "Services are available at:"
    log_info "  - Fraud Detection API: http://localhost:8000"
    log_info "  - Grafana Dashboard: http://localhost:3000"
    log_info "  - Prometheus: http://localhost:9090"
    log_info "  - Kibana: http://localhost:5601"
}

# Script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
