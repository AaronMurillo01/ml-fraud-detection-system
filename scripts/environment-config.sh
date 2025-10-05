#!/bin/bash

# Environment Configuration Script for Fraud Detection System
# This script manages environment-specific configurations and deployments

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$PROJECT_ROOT/logs/environment-config.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
}

info() { log "INFO" "$@"; }
warn() { log "WARN" "$@"; }
error() { log "ERROR" "$@"; }
success() { log "SUCCESS" "$@"; }

# Print colored output
print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Help function
show_help() {
    cat << EOF
Environment Configuration Script for Fraud Detection System

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    setup-env           Setup environment configuration
    switch-env          Switch between environments
    validate-env        Validate environment configuration
    deploy-env          Deploy to specific environment
    rollback-env        Rollback environment deployment
    scale-env           Scale environment resources
    monitor-env         Monitor environment health
    backup-env          Backup environment configuration
    restore-env         Restore environment from backup
    cleanup-env         Cleanup environment resources
    compare-envs        Compare configurations between environments
    sync-configs        Sync configurations across environments

Options:
    -e, --environment   Environment (dev|staging|prod)
    -c, --config        Configuration file path
    -f, --force         Force operation without confirmation
    -d, --dry-run       Show what would be done without executing
    -v, --verbose       Verbose output
    -w, --wait          Wait for deployment to complete
    -t, --timeout       Timeout in seconds (default: 600)
    -h, --help          Show this help message

Examples:
    $0 setup-env --environment dev
    $0 deploy-env --environment staging --wait
    $0 scale-env --environment prod --replicas 5
    $0 rollback-env --environment prod --force

EOF
}

# Default values
ENVIRONMENT="dev"
CONFIG_FILE=""
FORCE=false
DRY_RUN=false
VERBOSE=false
WAIT=false
TIMEOUT=600
REPLICAS=3

# Environment configurations
declare -A ENV_CONFIGS

# Development environment
ENV_CONFIGS[dev_namespace]="fraud-detection-dev"
ENV_CONFIGS[dev_replicas]="1"
ENV_CONFIGS[dev_cpu_request]="100m"
ENV_CONFIGS[dev_memory_request]="256Mi"
ENV_CONFIGS[dev_cpu_limit]="500m"
ENV_CONFIGS[dev_memory_limit]="512Mi"
ENV_CONFIGS[dev_storage_size]="10Gi"
ENV_CONFIGS[dev_ingress_host]="fraud-detection-dev.example.com"
ENV_CONFIGS[dev_log_level]="DEBUG"
ENV_CONFIGS[dev_monitoring]="false"

# Staging environment
ENV_CONFIGS[staging_namespace]="fraud-detection-staging"
ENV_CONFIGS[staging_replicas]="2"
ENV_CONFIGS[staging_cpu_request]="200m"
ENV_CONFIGS[staging_memory_request]="512Mi"
ENV_CONFIGS[staging_cpu_limit]="1000m"
ENV_CONFIGS[staging_memory_limit]="1Gi"
ENV_CONFIGS[staging_storage_size]="50Gi"
ENV_CONFIGS[staging_ingress_host]="fraud-detection-staging.example.com"
ENV_CONFIGS[staging_log_level]="INFO"
ENV_CONFIGS[staging_monitoring]="true"

# Production environment
ENV_CONFIGS[prod_namespace]="fraud-detection-prod"
ENV_CONFIGS[prod_replicas]="5"
ENV_CONFIGS[prod_cpu_request]="500m"
ENV_CONFIGS[prod_memory_request]="1Gi"
ENV_CONFIGS[prod_cpu_limit]="2000m"
ENV_CONFIGS[prod_memory_limit]="2Gi"
ENV_CONFIGS[prod_storage_size]="200Gi"
ENV_CONFIGS[prod_ingress_host]="fraud-detection.example.com"
ENV_CONFIGS[prod_log_level]="WARN"
ENV_CONFIGS[prod_monitoring]="true"

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -c|--config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            -f|--force)
                FORCE=true
                shift
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -w|--wait)
                WAIT=true
                shift
                ;;
            -t|--timeout)
                TIMEOUT="$2"
                shift 2
                ;;
            --replicas)
                REPLICAS="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            setup-env|switch-env|validate-env|deploy-env|rollback-env|scale-env|monitor-env|backup-env|restore-env|cleanup-env|compare-envs|sync-configs)
                COMMAND="$1"
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Validate environment
validate_environment() {
    case $ENVIRONMENT in
        dev|staging|prod)
            info "Using environment: $ENVIRONMENT"
            ;;
        *)
            error "Invalid environment: $ENVIRONMENT. Must be dev, staging, or prod."
            exit 1
            ;;
    esac
}

# Get environment configuration
get_env_config() {
    local key="${ENVIRONMENT}_$1"
    echo "${ENV_CONFIGS[$key]:-}"
}

# Check prerequisites
check_prerequisites() {
    local missing_tools=()
    
    # Check for required tools
    command -v kubectl >/dev/null 2>&1 || missing_tools+=("kubectl")
    command -v helm >/dev/null 2>&1 || missing_tools+=("helm")
    command -v docker >/dev/null 2>&1 || missing_tools+=("docker")
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        error "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi
    
    # Check kubectl connection
    if ! kubectl cluster-info >/dev/null 2>&1; then
        error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
}

# Setup environment
setup_env() {
    print_info "Setting up environment: $ENVIRONMENT"
    
    local namespace=$(get_env_config "namespace")
    
    # Create namespace if it doesn't exist
    if ! kubectl get namespace "$namespace" >/dev/null 2>&1; then
        info "Creating namespace: $namespace"
        if [[ $DRY_RUN == false ]]; then
            kubectl create namespace "$namespace"
            kubectl label namespace "$namespace" environment="$ENVIRONMENT"
        fi
    fi
    
    # Create environment-specific configuration
    local config_file="$PROJECT_ROOT/k8s/config-$ENVIRONMENT.yaml"
    
    cat > "$config_file" << EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: fraud-detection-config
  namespace: $namespace
  labels:
    app: fraud-detection
    environment: $ENVIRONMENT
data:
  # Application configuration
  ENVIRONMENT: "$ENVIRONMENT"
  LOG_LEVEL: "$(get_env_config "log_level")"
  DEBUG: "$([ "$ENVIRONMENT" = "dev" ] && echo "true" || echo "false")"
  
  # Resource configuration
  REPLICAS: "$(get_env_config "replicas")"
  CPU_REQUEST: "$(get_env_config "cpu_request")"
  MEMORY_REQUEST: "$(get_env_config "memory_request")"
  CPU_LIMIT: "$(get_env_config "cpu_limit")"
  MEMORY_LIMIT: "$(get_env_config "memory_limit")"
  
  # Storage configuration
  STORAGE_SIZE: "$(get_env_config "storage_size")"
  
  # Network configuration
  INGRESS_HOST: "$(get_env_config "ingress_host")"
  
  # Feature flags
  ENABLE_MONITORING: "$(get_env_config "monitoring")"
  ENABLE_TRACING: "$([ "$ENVIRONMENT" != "dev" ] && echo "true" || echo "false")"
  ENABLE_METRICS: "true"
  
  # Performance tuning
  MAX_WORKERS: "$([ "$ENVIRONMENT" = "prod" ] && echo "8" || echo "4")"
  BATCH_SIZE: "$([ "$ENVIRONMENT" = "prod" ] && echo "1000" || echo "100")"
  CACHE_TTL: "$([ "$ENVIRONMENT" = "prod" ] && echo "3600" || echo "300")"
  
  # Security settings
  CORS_ORIGINS: "$([ "$ENVIRONMENT" = "prod" ] && echo "https://fraud-detection.example.com" || echo "*")"
  RATE_LIMIT: "$([ "$ENVIRONMENT" = "prod" ] && echo "1000" || echo "100")"
EOF
    
    if [[ $DRY_RUN == false ]]; then
        kubectl apply -f "$config_file"
    fi
    
    success "Environment $ENVIRONMENT setup completed"
}

# Deploy environment
deploy_env() {
    print_info "Deploying to environment: $ENVIRONMENT"
    
    local namespace=$(get_env_config "namespace")
    
    # Ensure environment is setup
    setup_env
    
    # Deploy infrastructure components
    info "Deploying infrastructure components..."
    if [[ $DRY_RUN == false ]]; then
        kubectl apply -f "$PROJECT_ROOT/k8s/infrastructure.yaml" -n "$namespace"
    fi
    
    # Deploy monitoring if enabled
    if [[ $(get_env_config "monitoring") == "true" ]]; then
        info "Deploying monitoring components..."
        if [[ $DRY_RUN == false ]]; then
            kubectl apply -f "$PROJECT_ROOT/k8s/monitoring.yaml" -n "$namespace"
        fi
    fi
    
    # Deploy application
    info "Deploying application components..."
    if [[ $DRY_RUN == false ]]; then
        # Update deployment with environment-specific values
        local deployment_file="$PROJECT_ROOT/k8s/deployment-$ENVIRONMENT.yaml"
        
        # Generate environment-specific deployment
        sed "s/{{ENVIRONMENT}}/$ENVIRONMENT/g; \
             s/{{NAMESPACE}}/$namespace/g; \
             s/{{REPLICAS}}/$(get_env_config "replicas")/g; \
             s/{{CPU_REQUEST}}/$(get_env_config "cpu_request")/g; \
             s/{{MEMORY_REQUEST}}/$(get_env_config "memory_request")/g; \
             s/{{CPU_LIMIT}}/$(get_env_config "cpu_limit")/g; \
             s/{{MEMORY_LIMIT}}/$(get_env_config "memory_limit")/g; \
             s/{{INGRESS_HOST}}/$(get_env_config "ingress_host")/g" \
             "$PROJECT_ROOT/k8s/deployment.yaml" > "$deployment_file"
        
        kubectl apply -f "$deployment_file"
    fi
    
    # Wait for deployment if requested
    if [[ $WAIT == true ]]; then
        info "Waiting for deployment to complete..."
        if [[ $DRY_RUN == false ]]; then
            kubectl rollout status deployment/fraud-detection-api -n "$namespace" --timeout="${TIMEOUT}s"
            kubectl rollout status deployment/fraud-detection-streaming -n "$namespace" --timeout="${TIMEOUT}s"
        fi
    fi
    
    success "Deployment to $ENVIRONMENT completed successfully"
}

# Scale environment
scale_env() {
    print_info "Scaling environment: $ENVIRONMENT to $REPLICAS replicas"
    
    local namespace=$(get_env_config "namespace")
    
    if [[ $DRY_RUN == false ]]; then
        kubectl scale deployment fraud-detection-api -n "$namespace" --replicas="$REPLICAS"
        kubectl scale deployment fraud-detection-streaming -n "$namespace" --replicas="$REPLICAS"
        
        if [[ $WAIT == true ]]; then
            kubectl rollout status deployment/fraud-detection-api -n "$namespace" --timeout="${TIMEOUT}s"
            kubectl rollout status deployment/fraud-detection-streaming -n "$namespace" --timeout="${TIMEOUT}s"
        fi
    fi
    
    success "Environment $ENVIRONMENT scaled to $REPLICAS replicas"
}

# Monitor environment
monitor_env() {
    print_info "Monitoring environment: $ENVIRONMENT"
    
    local namespace=$(get_env_config "namespace")
    
    # Check pod status
    echo "Pod Status:"
    kubectl get pods -n "$namespace" -o wide
    
    # Check service status
    echo "\nService Status:"
    kubectl get services -n "$namespace"
    
    # Check ingress status
    echo "\nIngress Status:"
    kubectl get ingress -n "$namespace"
    
    # Check resource usage
    echo "\nResource Usage:"
    kubectl top pods -n "$namespace" 2>/dev/null || echo "Metrics server not available"
    
    # Check recent events
    echo "\nRecent Events:"
    kubectl get events -n "$namespace" --sort-by='.lastTimestamp' | tail -10
}

# Validate environment
validate_env() {
    print_info "Validating environment: $ENVIRONMENT"
    
    local namespace=$(get_env_config "namespace")
    local errors=0
    
    # Check namespace exists
    if kubectl get namespace "$namespace" >/dev/null 2>&1; then
        success "Namespace $namespace exists"
    else
        error "Namespace $namespace does not exist"
        ((errors++))
    fi
    
    # Check deployments
    local deployments=("fraud-detection-api" "fraud-detection-streaming")
    for deployment in "${deployments[@]}"; do
        if kubectl get deployment "$deployment" -n "$namespace" >/dev/null 2>&1; then
            local ready=$(kubectl get deployment "$deployment" -n "$namespace" -o jsonpath='{.status.readyReplicas}')
            local desired=$(kubectl get deployment "$deployment" -n "$namespace" -o jsonpath='{.spec.replicas}')
            
            if [[ "$ready" == "$desired" ]]; then
                success "Deployment $deployment is ready ($ready/$desired)"
            else
                error "Deployment $deployment is not ready ($ready/$desired)"
                ((errors++))
            fi
        else
            error "Deployment $deployment does not exist"
            ((errors++))
        fi
    done
    
    # Check services
    local services=("fraud-detection-api" "fraud-detection-streaming")
    for service in "${services[@]}"; do
        if kubectl get service "$service" -n "$namespace" >/dev/null 2>&1; then
            success "Service $service exists"
        else
            error "Service $service does not exist"
            ((errors++))
        fi
    done
    
    # Check secrets
    if kubectl get secret fraud-detection-secrets -n "$namespace" >/dev/null 2>&1; then
        success "Secrets are configured"
    else
        error "Secrets are not configured"
        ((errors++))
    fi
    
    if [[ $errors -eq 0 ]]; then
        success "Environment $ENVIRONMENT validation passed"
    else
        error "Environment $ENVIRONMENT validation failed with $errors errors"
        exit 1
    fi
}

# Rollback environment
rollback_env() {
    print_info "Rolling back environment: $ENVIRONMENT"
    
    local namespace=$(get_env_config "namespace")
    
    if [[ $FORCE == false ]]; then
        read -p "This will rollback the deployment. Continue? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Rollback cancelled"
            exit 0
        fi
    fi
    
    if [[ $DRY_RUN == false ]]; then
        kubectl rollout undo deployment/fraud-detection-api -n "$namespace"
        kubectl rollout undo deployment/fraud-detection-streaming -n "$namespace"
        
        if [[ $WAIT == true ]]; then
            kubectl rollout status deployment/fraud-detection-api -n "$namespace" --timeout="${TIMEOUT}s"
            kubectl rollout status deployment/fraud-detection-streaming -n "$namespace" --timeout="${TIMEOUT}s"
        fi
    fi
    
    success "Environment $ENVIRONMENT rolled back successfully"
}

# Cleanup environment
cleanup_env() {
    print_info "Cleaning up environment: $ENVIRONMENT"
    
    local namespace=$(get_env_config "namespace")
    
    if [[ $FORCE == false ]]; then
        read -p "This will delete all resources in $namespace. Continue? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Cleanup cancelled"
            exit 0
        fi
    fi
    
    if [[ $DRY_RUN == false ]]; then
        kubectl delete namespace "$namespace" --ignore-not-found=true
    fi
    
    success "Environment $ENVIRONMENT cleaned up successfully"
}

# Compare environments
compare_envs() {
    print_info "Comparing environment configurations"
    
    local envs=("dev" "staging" "prod")
    local configs=("replicas" "cpu_request" "memory_request" "cpu_limit" "memory_limit" "storage_size")
    
    printf "%-15s" "Config"
    for env in "${envs[@]}"; do
        printf "%-15s" "$env"
    done
    echo
    
    printf "%-15s" "---------------"
    for env in "${envs[@]}"; do
        printf "%-15s" "---------------"
    done
    echo
    
    for config in "${configs[@]}"; do
        printf "%-15s" "$config"
        for env in "${envs[@]}"; do
            local key="${env}_$config"
            printf "%-15s" "${ENV_CONFIGS[$key]:-N/A}"
        done
        echo
    done
}

# Main execution
main() {
    # Create log directory
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Parse arguments
    parse_args "$@"
    
    # Validate environment
    validate_environment
    
    # Check prerequisites
    check_prerequisites
    
    # Execute command
    case ${COMMAND:-} in
        setup-env)
            setup_env
            ;;
        switch-env)
            print_info "Switching to environment: $ENVIRONMENT"
            kubectl config set-context --current --namespace="$(get_env_config "namespace")"
            success "Switched to environment: $ENVIRONMENT"
            ;;
        validate-env)
            validate_env
            ;;
        deploy-env)
            deploy_env
            ;;
        rollback-env)
            rollback_env
            ;;
        scale-env)
            scale_env
            ;;
        monitor-env)
            monitor_env
            ;;
        backup-env)
            print_error "Backup functionality not implemented yet"
            exit 1
            ;;
        restore-env)
            print_error "Restore functionality not implemented yet"
            exit 1
            ;;
        cleanup-env)
            cleanup_env
            ;;
        compare-envs)
            compare_envs
            ;;
        sync-configs)
            print_error "Sync configs functionality not implemented yet"
            exit 1
            ;;
        "")
            print_error "No command specified"
            show_help
            exit 1
            ;;
        *)
            print_error "Unknown command: $COMMAND"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"