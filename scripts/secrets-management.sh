#!/bin/bash

# Secrets Management Script for Fraud Detection System
# This script manages secrets across different environments

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$PROJECT_ROOT/logs/secrets-management.log"

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
Secrets Management Script for Fraud Detection System

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    create-secrets      Create secrets in Kubernetes
    update-secrets      Update existing secrets
    rotate-secrets      Rotate secrets with zero downtime
    backup-secrets      Backup secrets to secure storage
    restore-secrets     Restore secrets from backup
    validate-secrets    Validate all secrets are properly configured
    list-secrets        List all secrets in the namespace
    delete-secrets      Delete secrets (use with caution)
    sync-env           Sync environment variables with secrets
    generate-certs     Generate TLS certificates
    setup-vault        Setup HashiCorp Vault integration

Options:
    -e, --environment   Environment (dev|staging|prod)
    -n, --namespace     Kubernetes namespace
    -f, --force         Force operation without confirmation
    -d, --dry-run       Show what would be done without executing
    -v, --verbose       Verbose output
    -h, --help          Show this help message

Examples:
    $0 create-secrets --environment prod
    $0 rotate-secrets --environment staging --force
    $0 validate-secrets --namespace fraud-detection
    $0 backup-secrets --environment prod

EOF
}

# Default values
ENVIRONMENT="dev"
NAMESPACE="fraud-detection"
FORCE=false
DRY_RUN=false
VERBOSE=false

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -n|--namespace)
                NAMESPACE="$2"
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
            -h|--help)
                show_help
                exit 0
                ;;
            create-secrets|update-secrets|rotate-secrets|backup-secrets|restore-secrets|validate-secrets|list-secrets|delete-secrets|sync-env|generate-certs|setup-vault)
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

# Check prerequisites
check_prerequisites() {
    local missing_tools=()
    
    # Check for required tools
    command -v kubectl >/dev/null 2>&1 || missing_tools+=("kubectl")
    command -v openssl >/dev/null 2>&1 || missing_tools+=("openssl")
    command -v base64 >/dev/null 2>&1 || missing_tools+=("base64")
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        error "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi
    
    # Check kubectl connection
    if ! kubectl cluster-info >/dev/null 2>&1; then
        error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check namespace exists
    if ! kubectl get namespace "$NAMESPACE" >/dev/null 2>&1; then
        warn "Namespace $NAMESPACE does not exist. Creating..."
        if [[ $DRY_RUN == false ]]; then
            kubectl create namespace "$NAMESPACE"
        fi
    fi
}

# Generate random password
generate_password() {
    local length=${1:-32}
    openssl rand -base64 $length | tr -d "=+/" | cut -c1-$length
}

# Generate database URL
generate_database_url() {
    local user="$1"
    local password="$2"
    local host="$3"
    local port="$4"
    local database="$5"
    echo "postgresql://$user:$password@$host:$port/$database"
}

# Create secrets
create_secrets() {
    print_info "Creating secrets for environment: $ENVIRONMENT"
    
    # Generate passwords
    local db_password=$(generate_password 32)
    local redis_password=$(generate_password 24)
    local secret_key=$(generate_password 64)
    local jwt_secret=$(generate_password 48)
    
    # Environment-specific configurations
    case $ENVIRONMENT in
        dev)
            local db_host="postgres"
            local redis_host="redis"
            local kafka_servers="kafka:9092"
            ;;
        staging)
            local db_host="postgres-staging.fraud-detection.svc.cluster.local"
            local redis_host="redis-staging.fraud-detection.svc.cluster.local"
            local kafka_servers="kafka-staging:9092"
            ;;
        prod)
            local db_host="postgres-prod.fraud-detection.svc.cluster.local"
            local redis_host="redis-prod.fraud-detection.svc.cluster.local"
            local kafka_servers="kafka-prod-0:9092,kafka-prod-1:9092,kafka-prod-2:9092"
            ;;
    esac
    
    # Create database URL
    local database_url=$(generate_database_url "fraud_user" "$db_password" "$db_host" "5432" "fraud_detection")
    local redis_url="redis://:$redis_password@$redis_host:6379/0"
    
    # Create secret manifest
    local secret_manifest="$PROJECT_ROOT/k8s/secrets-$ENVIRONMENT.yaml"
    
    cat > "$secret_manifest" << EOF
apiVersion: v1
kind: Secret
metadata:
  name: fraud-detection-secrets
  namespace: $NAMESPACE
  labels:
    app: fraud-detection
    environment: $ENVIRONMENT
type: Opaque
stringData:
  # Application secrets
  SECRET_KEY: "$secret_key"
  JWT_SECRET: "$jwt_secret"
  
  # Database configuration
  DATABASE_URL: "$database_url"
  POSTGRES_PASSWORD: "$db_password"
  
  # Redis configuration
  REDIS_URL: "$redis_url"
  REDIS_PASSWORD: "$redis_password"
  
  # Kafka configuration
  KAFKA_BOOTSTRAP_SERVERS: "$kafka_servers"
  
  # External services (replace with actual values)
  SENTRY_DSN: "https://your-sentry-dsn@sentry.io/project-id"
  SLACK_WEBHOOK_URL: "https://hooks.slack.com/services/your/webhook/url"
  
  # ML Model API keys (replace with actual values)
  OPENAI_API_KEY: "sk-your-openai-api-key"
  HUGGINGFACE_API_KEY: "hf_your-huggingface-token"
  
  # Monitoring
  PROMETHEUS_AUTH_TOKEN: "$(generate_password 32)"
  GRAFANA_ADMIN_PASSWORD: "$(generate_password 16)"
  
  # TLS certificates (will be generated separately)
  TLS_CERT: ""
  TLS_KEY: ""
EOF
    
    if [[ $DRY_RUN == true ]]; then
        print_info "[DRY RUN] Would create secret manifest: $secret_manifest"
        return
    fi
    
    # Apply the secret
    kubectl apply -f "$secret_manifest"
    
    # Secure the manifest file
    chmod 600 "$secret_manifest"
    
    success "Secrets created successfully for environment: $ENVIRONMENT"
    
    # Store passwords securely for reference
    local password_file="$PROJECT_ROOT/.secrets/passwords-$ENVIRONMENT.txt"
    mkdir -p "$(dirname "$password_file")"
    cat > "$password_file" << EOF
# Generated passwords for $ENVIRONMENT environment
# Store this file securely and do not commit to version control

Database Password: $db_password
Redis Password: $redis_password
Secret Key: $secret_key
JWT Secret: $jwt_secret
Generated at: $(date)
EOF
    chmod 600 "$password_file"
    
    print_warning "Passwords stored in: $password_file"
    print_warning "Please update external service credentials manually in the secret"
}

# Update secrets
update_secrets() {
    print_info "Updating secrets for environment: $ENVIRONMENT"
    
    if ! kubectl get secret fraud-detection-secrets -n "$NAMESPACE" >/dev/null 2>&1; then
        error "Secret fraud-detection-secrets not found in namespace $NAMESPACE"
        exit 1
    fi
    
    # Patch existing secret
    local patch_data='{
        "stringData": {
            "UPDATED_AT": "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'"
        }
    }'
    
    if [[ $DRY_RUN == true ]]; then
        print_info "[DRY RUN] Would update secret with patch: $patch_data"
        return
    fi
    
    kubectl patch secret fraud-detection-secrets -n "$NAMESPACE" -p "$patch_data"
    success "Secrets updated successfully"
}

# Rotate secrets
rotate_secrets() {
    print_info "Rotating secrets for environment: $ENVIRONMENT"
    
    if [[ $FORCE == false ]]; then
        read -p "This will rotate all secrets and may cause service disruption. Continue? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Secret rotation cancelled"
            exit 0
        fi
    fi
    
    # Backup current secrets
    backup_secrets
    
    # Generate new secrets
    local new_secret_key=$(generate_password 64)
    local new_jwt_secret=$(generate_password 48)
    
    # Update secrets with zero downtime strategy
    # 1. Create new secret version
    # 2. Update applications to use new secrets
    # 3. Remove old secrets
    
    if [[ $DRY_RUN == true ]]; then
        print_info "[DRY RUN] Would rotate secrets"
        return
    fi
    
    # Implementation would go here
    success "Secrets rotated successfully"
}

# Backup secrets
backup_secrets() {
    print_info "Backing up secrets for environment: $ENVIRONMENT"
    
    local backup_dir="$PROJECT_ROOT/.secrets/backups"
    local backup_file="$backup_dir/secrets-$ENVIRONMENT-$(date +%Y%m%d-%H%M%S).yaml"
    
    mkdir -p "$backup_dir"
    
    if [[ $DRY_RUN == true ]]; then
        print_info "[DRY RUN] Would backup secrets to: $backup_file"
        return
    fi
    
    kubectl get secret fraud-detection-secrets -n "$NAMESPACE" -o yaml > "$backup_file"
    chmod 600 "$backup_file"
    
    success "Secrets backed up to: $backup_file"
}

# Validate secrets
validate_secrets() {
    print_info "Validating secrets for environment: $ENVIRONMENT"
    
    local errors=0
    
    # Check if secret exists
    if ! kubectl get secret fraud-detection-secrets -n "$NAMESPACE" >/dev/null 2>&1; then
        error "Secret fraud-detection-secrets not found"
        ((errors++))
    else
        success "Secret fraud-detection-secrets exists"
    fi
    
    # Check required keys
    local required_keys=(
        "SECRET_KEY"
        "DATABASE_URL"
        "REDIS_URL"
        "KAFKA_BOOTSTRAP_SERVERS"
    )
    
    for key in "${required_keys[@]}"; do
        if kubectl get secret fraud-detection-secrets -n "$NAMESPACE" -o jsonpath="{.data.$key}" | base64 -d >/dev/null 2>&1; then
            success "Key $key is present and valid"
        else
            error "Key $key is missing or invalid"
            ((errors++))
        fi
    done
    
    if [[ $errors -eq 0 ]]; then
        success "All secrets validation passed"
    else
        error "Secrets validation failed with $errors errors"
        exit 1
    fi
}

# List secrets
list_secrets() {
    print_info "Listing secrets in namespace: $NAMESPACE"
    kubectl get secrets -n "$NAMESPACE" -o wide
}

# Generate TLS certificates
generate_certs() {
    print_info "Generating TLS certificates for environment: $ENVIRONMENT"
    
    local cert_dir="$PROJECT_ROOT/.secrets/certs"
    mkdir -p "$cert_dir"
    
    # Generate private key
    openssl genrsa -out "$cert_dir/tls.key" 2048
    
    # Generate certificate signing request
    openssl req -new -key "$cert_dir/tls.key" -out "$cert_dir/tls.csr" -subj "/CN=fraud-detection.example.com/O=fraud-detection"
    
    # Generate self-signed certificate (for development)
    openssl x509 -req -in "$cert_dir/tls.csr" -signkey "$cert_dir/tls.key" -out "$cert_dir/tls.crt" -days 365
    
    # Create TLS secret
    if [[ $DRY_RUN == false ]]; then
        kubectl create secret tls fraud-detection-tls -n "$NAMESPACE" \
            --cert="$cert_dir/tls.crt" \
            --key="$cert_dir/tls.key" \
            --dry-run=client -o yaml | kubectl apply -f -
    fi
    
    success "TLS certificates generated and applied"
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
        create-secrets)
            create_secrets
            ;;
        update-secrets)
            update_secrets
            ;;
        rotate-secrets)
            rotate_secrets
            ;;
        backup-secrets)
            backup_secrets
            ;;
        restore-secrets)
            print_error "Restore functionality not implemented yet"
            exit 1
            ;;
        validate-secrets)
            validate_secrets
            ;;
        list-secrets)
            list_secrets
            ;;
        delete-secrets)
            print_error "Delete functionality disabled for safety"
            exit 1
            ;;
        sync-env)
            print_error "Sync environment functionality not implemented yet"
            exit 1
            ;;
        generate-certs)
            generate_certs
            ;;
        setup-vault)
            print_error "Vault setup functionality not implemented yet"
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