#!/bin/bash

# Kubernetes Production Deployment Script for Fraud Detection System
# This script handles the complete Kubernetes deployment process

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
NAMESPACE="${NAMESPACE:-fraud-detection}"
CLUSTER_NAME="${CLUSTER_NAME:-fraud-detection-cluster}"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
DOMAIN="${DOMAIN:-fraud-detection.company.com}"
DRY_RUN="${DRY_RUN:-false}"

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
        log_error "Kubernetes deployment failed with exit code $exit_code"
    fi
    exit $exit_code
}

trap cleanup EXIT

# Validation functions
validate_k8s_environment() {
    log_info "Validating Kubernetes environment..."
    
    # Check required tools
    local required_tools=("kubectl" "helm" "docker")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "Required tool '$tool' is not installed"
            exit 1
        fi
    done
    
    # Check kubectl connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check if namespace exists
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_info "Creating namespace: $NAMESPACE"
        kubectl create namespace "$NAMESPACE"
    fi
    
    # Validate cluster resources
    local nodes=$(kubectl get nodes --no-headers | wc -l)
    if [ "$nodes" -lt 3 ]; then
        log_warning "Cluster has only $nodes nodes. Recommended: 3+ nodes for production"
    fi
    
    log_success "Kubernetes environment validation completed"
}

# Build and push images to registry
build_and_push_images() {
    if [ -n "$DOCKER_REGISTRY" ]; then
        log_info "Building and pushing images to registry: $DOCKER_REGISTRY"
        
        cd "$PROJECT_ROOT"
        
        # Build and push main API image
        docker build \
            --build-arg VERSION="$IMAGE_TAG" \
            --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
            --build-arg VCS_REF="$(git rev-parse HEAD 2>/dev/null || echo 'unknown')" \
            --target production \
            --tag "$DOCKER_REGISTRY/fraud-detection:$IMAGE_TAG" \
            .
        
        docker push "$DOCKER_REGISTRY/fraud-detection:$IMAGE_TAG"
        
        # Build and push streaming service if exists
        if [ -f "Dockerfile.streaming" ]; then
            docker build \
                --build-arg VERSION="$IMAGE_TAG" \
                --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
                --build-arg VCS_REF="$(git rev-parse HEAD 2>/dev/null || echo 'unknown')" \
                --file Dockerfile.streaming \
                --target production \
                --tag "$DOCKER_REGISTRY/fraud-detection-streaming:$IMAGE_TAG" \
                .
            
            docker push "$DOCKER_REGISTRY/fraud-detection-streaming:$IMAGE_TAG"
        fi
        
        log_success "Images built and pushed successfully"
    else
        log_info "No registry specified, using local images"
    fi
}

# Deploy infrastructure components
deploy_infrastructure() {
    log_info "Deploying infrastructure components..."
    
    cd "$PROJECT_ROOT/k8s"
    
    # Deploy namespace and RBAC
    kubectl apply -f namespace.yaml
    
    # Deploy persistent volumes
    if [ -f "persistent-volumes.yaml" ]; then
        kubectl apply -f persistent-volumes.yaml
    fi
    
    # Deploy infrastructure services (PostgreSQL, Redis, Kafka)
    if [ -f "infrastructure.yaml" ]; then
        kubectl apply -f infrastructure.yaml
    fi
    
    # Wait for infrastructure to be ready
    log_info "Waiting for infrastructure services to be ready..."
    kubectl wait --for=condition=ready pod -l app=postgres -n "$NAMESPACE" --timeout=300s
    kubectl wait --for=condition=ready pod -l app=redis -n "$NAMESPACE" --timeout=300s
    kubectl wait --for=condition=ready pod -l app=kafka -n "$NAMESPACE" --timeout=300s
    
    log_success "Infrastructure components deployed successfully"
}

# Deploy monitoring stack
deploy_monitoring() {
    log_info "Deploying monitoring stack..."
    
    cd "$PROJECT_ROOT/k8s"
    
    # Deploy monitoring services
    if [ -f "monitoring.yaml" ]; then
        kubectl apply -f monitoring.yaml
    fi
    
    # Install Prometheus using Helm if not using custom manifests
    if ! kubectl get deployment prometheus -n "$NAMESPACE" &> /dev/null; then
        log_info "Installing Prometheus using Helm..."
        helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
        helm repo update
        
        helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
            --namespace "$NAMESPACE" \
            --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
            --set prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues=false \
            --set prometheus.prometheusSpec.retention=30d \
            --set grafana.adminPassword="$(kubectl get secret fraud-detection-secrets -n $NAMESPACE -o jsonpath='{.data.GRAFANA_PASSWORD}' | base64 -d)"
    fi
    
    log_success "Monitoring stack deployed successfully"
}

# Deploy application
deploy_application() {
    log_info "Deploying fraud detection application..."
    
    cd "$PROJECT_ROOT/k8s"
    
    # Update image references if using registry
    if [ -n "$DOCKER_REGISTRY" ]; then
        sed -i.bak "s|fraud-detection:latest|$DOCKER_REGISTRY/fraud-detection:$IMAGE_TAG|g" production-deployment.yaml
        sed -i.bak "s|fraud-detection-streaming:latest|$DOCKER_REGISTRY/fraud-detection-streaming:$IMAGE_TAG|g" production-deployment.yaml
    fi
    
    # Apply configuration and secrets
    if [ -f "configmap-secrets.yaml" ]; then
        kubectl apply -f configmap-secrets.yaml
    fi
    
    # Deploy the main application
    if [ "$DRY_RUN" = "true" ]; then
        kubectl apply --dry-run=client -f production-deployment.yaml
        log_info "Dry run completed successfully"
    else
        kubectl apply -f production-deployment.yaml
        
        # Wait for deployment to be ready
        log_info "Waiting for application deployment to be ready..."
        kubectl wait --for=condition=available deployment/fraud-detection-api -n "$NAMESPACE" --timeout=600s
        
        # Check if HPA is working
        kubectl get hpa fraud-detection-api-hpa -n "$NAMESPACE"
    fi
    
    # Restore original files if modified
    if [ -n "$DOCKER_REGISTRY" ] && [ -f "production-deployment.yaml.bak" ]; then
        mv production-deployment.yaml.bak production-deployment.yaml
    fi
    
    log_success "Application deployed successfully"
}

# Run database migrations
run_migrations() {
    log_info "Running database migrations..."
    
    # Get a pod name for the API deployment
    local pod_name=$(kubectl get pods -n "$NAMESPACE" -l app=fraud-detection-api -o jsonpath='{.items[0].metadata.name}')
    
    if [ -n "$pod_name" ]; then
        kubectl exec -n "$NAMESPACE" "$pod_name" -- python -m alembic upgrade head
        log_success "Database migrations completed"
    else
        log_error "No API pods found for running migrations"
        return 1
    fi
}

# Health checks
health_checks() {
    log_info "Running health checks..."
    
    # Check pod status
    kubectl get pods -n "$NAMESPACE" -l app=fraud-detection-api
    
    # Check service endpoints
    kubectl get endpoints -n "$NAMESPACE"
    
    # Test API health endpoint
    local service_ip=$(kubectl get service fraud-detection-api-service -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')
    
    if [ -n "$service_ip" ]; then
        # Port forward for testing
        kubectl port-forward -n "$NAMESPACE" service/fraud-detection-api-service 8080:8000 &
        local port_forward_pid=$!
        
        sleep 5
        
        if curl -f -s http://localhost:8080/health > /dev/null; then
            log_success "API health check passed"
        else
            log_error "API health check failed"
            kill $port_forward_pid 2>/dev/null || true
            return 1
        fi
        
        kill $port_forward_pid 2>/dev/null || true
    fi
    
    log_success "All health checks passed"
}

# Setup ingress and SSL
setup_ingress() {
    log_info "Setting up ingress and SSL..."
    
    # Install nginx-ingress if not present
    if ! kubectl get deployment nginx-ingress-controller -n ingress-nginx &> /dev/null; then
        log_info "Installing nginx-ingress controller..."
        kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.1/deploy/static/provider/cloud/deploy.yaml
        
        # Wait for ingress controller to be ready
        kubectl wait --namespace ingress-nginx \
            --for=condition=ready pod \
            --selector=app.kubernetes.io/component=controller \
            --timeout=300s
    fi
    
    # Install cert-manager if not present
    if ! kubectl get deployment cert-manager -n cert-manager &> /dev/null; then
        log_info "Installing cert-manager..."
        kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
        
        # Wait for cert-manager to be ready
        kubectl wait --namespace cert-manager \
            --for=condition=ready pod \
            --selector=app.kubernetes.io/instance=cert-manager \
            --timeout=300s
    fi
    
    log_success "Ingress and SSL setup completed"
}

# Display deployment information
display_info() {
    log_success "Kubernetes deployment completed successfully!"
    
    echo ""
    log_info "Deployment Information:"
    log_info "  Namespace: $NAMESPACE"
    log_info "  Cluster: $CLUSTER_NAME"
    log_info "  Domain: $DOMAIN"
    log_info "  Image Tag: $IMAGE_TAG"
    
    echo ""
    log_info "Service URLs:"
    log_info "  API: https://$DOMAIN"
    log_info "  Grafana: https://$DOMAIN/grafana"
    log_info "  Prometheus: https://$DOMAIN/prometheus"
    
    echo ""
    log_info "Useful Commands:"
    log_info "  View pods: kubectl get pods -n $NAMESPACE"
    log_info "  View logs: kubectl logs -f deployment/fraud-detection-api -n $NAMESPACE"
    log_info "  Scale deployment: kubectl scale deployment fraud-detection-api --replicas=5 -n $NAMESPACE"
    log_info "  Port forward API: kubectl port-forward service/fraud-detection-api-service 8080:8000 -n $NAMESPACE"
}

# Main deployment function
main() {
    log_info "Starting Kubernetes deployment for Fraud Detection System"
    log_info "Cluster: $CLUSTER_NAME"
    log_info "Namespace: $NAMESPACE"
    log_info "Domain: $DOMAIN"
    log_info "Image Tag: $IMAGE_TAG"
    log_info "Dry Run: $DRY_RUN"
    
    validate_k8s_environment
    build_and_push_images
    deploy_infrastructure
    deploy_monitoring
    setup_ingress
    deploy_application
    
    if [ "$DRY_RUN" != "true" ]; then
        run_migrations
        health_checks
    fi
    
    display_info
}

# Script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
