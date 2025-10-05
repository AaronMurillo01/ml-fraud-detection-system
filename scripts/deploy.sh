#!/bin/bash

# Fraud Detection System Deployment Script
# This script handles deployment to different environments

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="${PROJECT_ROOT}/logs/deployment.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

# Default values
ENVIRONMENT="staging"
IMAGE_TAG="latest"
SKIP_TESTS=false
SKIP_BACKUP=false
DRY_RUN=false
ROLLBACK=false
HEALTH_CHECK_TIMEOUT=300
DEPLOYMENT_TIMEOUT=600

# Usage function
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy the Fraud Detection System to specified environment.

Options:
    -e, --environment ENV     Target environment (staging|production) [default: staging]
    -t, --tag TAG            Docker image tag to deploy [default: latest]
    -s, --skip-tests         Skip running tests before deployment
    -b, --skip-backup        Skip database backup before deployment
    -d, --dry-run           Show what would be deployed without actually deploying
    -r, --rollback          Rollback to previous deployment
    --health-timeout SEC     Health check timeout in seconds [default: 300]
    --deploy-timeout SEC     Deployment timeout in seconds [default: 600]
    -h, --help              Show this help message

Examples:
    $0 -e staging -t v1.2.3
    $0 -e production -t v1.2.3 --skip-tests
    $0 --rollback -e production
    $0 --dry-run -e staging

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -s|--skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        -b|--skip-backup)
            SKIP_BACKUP=true
            shift
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -r|--rollback)
            ROLLBACK=true
            shift
            ;;
        --health-timeout)
            HEALTH_CHECK_TIMEOUT="$2"
            shift 2
            ;;
        --deploy-timeout)
            DEPLOYMENT_TIMEOUT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(staging|production)$ ]]; then
    log_error "Invalid environment: $ENVIRONMENT. Must be 'staging' or 'production'."
    exit 1
fi

# Set environment-specific variables
case $ENVIRONMENT in
    staging)
        NAMESPACE="fraud-detection-staging"
        DOMAIN="fraud-detection-staging.example.com"
        REPLICAS=2
        ;;
    production)
        NAMESPACE="fraud-detection-production"
        DOMAIN="fraud-detection.example.com"
        REPLICAS=4
        ;;
esac

# Create logs directory if it doesn't exist
mkdir -p "$(dirname "$LOG_FILE")"

# Start deployment
log_info "Starting deployment to $ENVIRONMENT environment"
log_info "Image tag: $IMAGE_TAG"
log_info "Namespace: $NAMESPACE"
log_info "Domain: $DOMAIN"
log_info "Replicas: $REPLICAS"

if [[ "$DRY_RUN" == "true" ]]; then
    log_warning "DRY RUN MODE - No actual changes will be made"
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check required tools
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    local missing_tools=()
    
    if ! command_exists kubectl; then
        missing_tools+=("kubectl")
    fi
    
    if ! command_exists docker; then
        missing_tools+=("docker")
    fi
    
    if ! command_exists helm; then
        missing_tools+=("helm")
    fi
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_error "Please install the missing tools and try again."
        exit 1
    fi
    
    log_success "All prerequisites satisfied"
}

# Function to run tests
run_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        log_warning "Skipping tests as requested"
        return 0
    fi
    
    log_info "Running tests..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would run: pytest tests/ -v --tb=short"
        return 0
    fi
    
    cd "$PROJECT_ROOT"
    
    # Run unit tests
    if ! pytest tests/unit/ -v --tb=short; then
        log_error "Unit tests failed"
        return 1
    fi
    
    # Run integration tests
    if ! pytest tests/integration/ -v --tb=short; then
        log_error "Integration tests failed"
        return 1
    fi
    
    log_success "All tests passed"
}

# Function to backup database
backup_database() {
    if [[ "$SKIP_BACKUP" == "true" ]]; then
        log_warning "Skipping database backup as requested"
        return 0
    fi
    
    log_info "Creating database backup..."
    
    local backup_name="fraud_detection_${ENVIRONMENT}_$(date +%Y%m%d_%H%M%S)"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would create backup: $backup_name"
        return 0
    fi
    
    # Create database backup using kubectl
    if ! kubectl exec -n "$NAMESPACE" deployment/postgres -- pg_dump -U postgres fraud_detection > "backups/${backup_name}.sql"; then
        log_error "Database backup failed"
        return 1
    fi
    
    log_success "Database backup created: ${backup_name}.sql"
}

# Function to deploy application
deploy_application() {
    log_info "Deploying application..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would deploy image: ghcr.io/ml-crml/fraud-detection:$IMAGE_TAG"
        log_info "[DRY RUN] Would update deployment in namespace: $NAMESPACE"
        return 0
    fi
    
    # Update deployment with new image
    if ! kubectl set image deployment/fraud-detection-api \
        fraud-detection-api="ghcr.io/ml-crml/fraud-detection:$IMAGE_TAG" \
        -n "$NAMESPACE"; then
        log_error "Failed to update deployment"
        return 1
    fi
    
    # Wait for rollout to complete
    log_info "Waiting for deployment to complete..."
    if ! kubectl rollout status deployment/fraud-detection-api -n "$NAMESPACE" --timeout="${DEPLOYMENT_TIMEOUT}s"; then
        log_error "Deployment rollout failed or timed out"
        return 1
    fi
    
    log_success "Application deployed successfully"
}

# Function to run health checks
run_health_checks() {
    log_info "Running health checks..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would check health endpoint: https://$DOMAIN/health"
        return 0
    fi
    
    local health_url="https://$DOMAIN/health"
    local start_time=$(date +%s)
    local timeout_time=$((start_time + HEALTH_CHECK_TIMEOUT))
    
    while [[ $(date +%s) -lt $timeout_time ]]; do
        if curl -f -s "$health_url" > /dev/null; then
            log_success "Health check passed"
            return 0
        fi
        
        log_info "Health check failed, retrying in 10 seconds..."
        sleep 10
    done
    
    log_error "Health check timed out after $HEALTH_CHECK_TIMEOUT seconds"
    return 1
}

# Function to rollback deployment
rollback_deployment() {
    log_info "Rolling back deployment..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would rollback deployment in namespace: $NAMESPACE"
        return 0
    fi
    
    # Rollback to previous revision
    if ! kubectl rollout undo deployment/fraud-detection-api -n "$NAMESPACE"; then
        log_error "Rollback failed"
        return 1
    fi
    
    # Wait for rollback to complete
    log_info "Waiting for rollback to complete..."
    if ! kubectl rollout status deployment/fraud-detection-api -n "$NAMESPACE" --timeout="${DEPLOYMENT_TIMEOUT}s"; then
        log_error "Rollback rollout failed or timed out"
        return 1
    fi
    
    log_success "Rollback completed successfully"
}

# Function to send notifications
send_notifications() {
    local status="$1"
    local message="$2"
    
    log_info "Sending deployment notifications..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would send notification: $status - $message"
        return 0
    fi
    
    # Send Slack notification if webhook URL is configured
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        local color="good"
        if [[ "$status" == "failed" ]]; then
            color="danger"
        fi
        
        curl -X POST -H 'Content-type: application/json' \
            --data "{
                \"attachments\": [{
                    \"color\": \"$color\",
                    \"title\": \"Deployment $status\",
                    \"text\": \"$message\",
                    \"fields\": [
                        {\"title\": \"Environment\", \"value\": \"$ENVIRONMENT\", \"short\": true},
                        {\"title\": \"Image Tag\", \"value\": \"$IMAGE_TAG\", \"short\": true}
                    ]
                }]
            }" \
            "$SLACK_WEBHOOK_URL" || log_warning "Failed to send Slack notification"
    fi
    
    log_success "Notifications sent"
}

# Function to cleanup old resources
cleanup_old_resources() {
    log_info "Cleaning up old resources..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would cleanup old ReplicaSets and images"
        return 0
    fi
    
    # Keep only the last 3 ReplicaSets
    kubectl delete replicaset -n "$NAMESPACE" \
        --field-selector=status.replicas=0 \
        --sort-by=.metadata.creationTimestamp \
        -o name | head -n -3 | xargs -r kubectl delete -n "$NAMESPACE" || true
    
    log_success "Cleanup completed"
}

# Main deployment function
main() {
    local exit_code=0
    
    # Trap to ensure cleanup on exit
    trap 'log_info "Deployment script finished with exit code: $exit_code"' EXIT
    
    # Handle rollback
    if [[ "$ROLLBACK" == "true" ]]; then
        check_prerequisites
        if rollback_deployment && run_health_checks; then
            send_notifications "success" "Rollback to $ENVIRONMENT completed successfully"
            log_success "Rollback completed successfully"
        else
            send_notifications "failed" "Rollback to $ENVIRONMENT failed"
            log_error "Rollback failed"
            exit_code=1
        fi
        exit $exit_code
    fi
    
    # Normal deployment flow
    if check_prerequisites && \
       run_tests && \
       backup_database && \
       deploy_application && \
       run_health_checks; then
        
        cleanup_old_resources
        send_notifications "success" "Deployment to $ENVIRONMENT completed successfully"
        log_success "Deployment completed successfully"
    else
        log_error "Deployment failed"
        
        # Attempt automatic rollback on production failures
        if [[ "$ENVIRONMENT" == "production" ]]; then
            log_warning "Attempting automatic rollback..."
            if rollback_deployment && run_health_checks; then
                send_notifications "warning" "Deployment to $ENVIRONMENT failed, automatic rollback successful"
                log_warning "Automatic rollback completed"
            else
                send_notifications "failed" "Deployment to $ENVIRONMENT failed, automatic rollback also failed"
                log_error "Automatic rollback failed"
            fi
        else
            send_notifications "failed" "Deployment to $ENVIRONMENT failed"
        fi
        
        exit_code=1
    fi
    
    exit $exit_code
}

# Run main function
main "$@"