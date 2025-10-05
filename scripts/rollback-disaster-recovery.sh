#!/bin/bash

# Rollback and Disaster Recovery Script for Fraud Detection System
# This script handles rollbacks, disaster recovery, and backup operations

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$PROJECT_ROOT/logs/rollback-recovery.log"
BACKUP_DIR="$PROJECT_ROOT/backups"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Recovery configuration
MAX_ROLLBACK_VERSIONS=10
BACKUP_RETENTION_DAYS=30
RECOVERY_TIMEOUT=600

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
Rollback and Disaster Recovery Script for Fraud Detection System

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    rollback            Rollback to previous version
    list-versions       List available rollback versions
    create-backup       Create full system backup
    restore-backup      Restore from backup
    disaster-recovery   Execute disaster recovery plan
    validate-backup     Validate backup integrity
    cleanup-backups     Clean up old backups
    emergency-stop      Emergency stop all services
    emergency-start     Emergency start all services
    health-restore      Restore system to healthy state
    data-recovery       Recover lost or corrupted data
    config-rollback     Rollback configuration changes
    db-rollback         Rollback database changes
    canary-rollback     Rollback canary deployment

Options:
    -e, --environment   Environment (dev|staging|prod)
    -v, --version       Specific version to rollback to
    -b, --backup-id     Backup ID to restore from
    -f, --force         Force operation without confirmation
    -d, --dry-run       Show what would be done without executing
    -t, --timeout       Timeout in seconds (default: 600)
    -r, --reason        Reason for rollback/recovery
    --skip-validation   Skip backup validation
    --preserve-data     Preserve user data during rollback
    -h, --help          Show this help message

Examples:
    $0 rollback --environment prod --version v1.2.3
    $0 create-backup --environment prod
    $0 disaster-recovery --environment prod --reason "database-failure"
    $0 restore-backup --backup-id backup-20231201-120000

EOF
}

# Default values
ENVIRONMENT="dev"
VERSION=""
BACKUP_ID=""
FORCE=false
DRY_RUN=false
TIMEOUT=600
REASON=""
SKIP_VALIDATION=false
PRESERVE_DATA=true

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -v|--version)
                VERSION="$2"
                shift 2
                ;;
            -b|--backup-id)
                BACKUP_ID="$2"
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
            -t|--timeout)
                TIMEOUT="$2"
                shift 2
                ;;
            -r|--reason)
                REASON="$2"
                shift 2
                ;;
            --skip-validation)
                SKIP_VALIDATION=true
                shift
                ;;
            --preserve-data)
                PRESERVE_DATA=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            rollback|list-versions|create-backup|restore-backup|disaster-recovery|validate-backup|cleanup-backups|emergency-stop|emergency-start|health-restore|data-recovery|config-rollback|db-rollback|canary-rollback)
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
            NAMESPACE="fraud-detection-$ENVIRONMENT"
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
    command -v helm >/dev/null 2>&1 || missing_tools+=("helm")
    command -v pg_dump >/dev/null 2>&1 || missing_tools+=("pg_dump")
    command -v tar >/dev/null 2>&1 || missing_tools+=("tar")
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        error "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi
    
    # Check kubectl connection
    if ! kubectl cluster-info >/dev/null 2>&1; then
        error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Create backup directory
    mkdir -p "$BACKUP_DIR"
}

# Get current deployment version
get_current_version() {
    local deployment="$1"
    kubectl get deployment "$deployment" -n "$NAMESPACE" -o jsonpath='{.metadata.labels.version}' 2>/dev/null || echo "unknown"
}

# List available versions for rollback
list_versions() {
    print_info "Listing available rollback versions for environment: $ENVIRONMENT"
    
    # Get deployment history
    local deployments=("fraud-detection-api" "fraud-detection-streaming")
    
    for deployment in "${deployments[@]}"; do
        echo "\nDeployment: $deployment"
        echo "Current version: $(get_current_version "$deployment")"
        echo "Rollout history:"
        kubectl rollout history deployment "$deployment" -n "$NAMESPACE" 2>/dev/null || echo "  No history available"
    done
    
    # List available backup versions
    echo "\nAvailable backups:"
    if [[ -d "$BACKUP_DIR" ]]; then
        ls -la "$BACKUP_DIR" | grep "backup-" | awk '{print $9, $5, $6, $7, $8}' | column -t
    else
        echo "  No backups available"
    fi
}

# Create system backup
create_backup() {
    print_info "Creating full system backup for environment: $ENVIRONMENT"
    
    local backup_timestamp=$(date +%Y%m%d-%H%M%S)
    local backup_name="backup-$ENVIRONMENT-$backup_timestamp"
    local backup_path="$BACKUP_DIR/$backup_name"
    
    mkdir -p "$backup_path"
    
    # Backup Kubernetes resources
    info "Backing up Kubernetes resources..."
    if [[ $DRY_RUN == false ]]; then
        kubectl get all -n "$NAMESPACE" -o yaml > "$backup_path/k8s-resources.yaml"
        kubectl get configmaps -n "$NAMESPACE" -o yaml > "$backup_path/configmaps.yaml"
        kubectl get secrets -n "$NAMESPACE" -o yaml > "$backup_path/secrets.yaml"
        kubectl get pvc -n "$NAMESPACE" -o yaml > "$backup_path/persistent-volumes.yaml"
    fi
    
    # Backup database
    info "Backing up database..."
    if [[ $DRY_RUN == false ]]; then
        local db_pod=$(kubectl get pods -n "$NAMESPACE" -l app=postgres -o jsonpath='{.items[0].metadata.name}')
        if [[ -n "$db_pod" ]]; then
            kubectl exec -n "$NAMESPACE" "$db_pod" -- pg_dump -U fraud_user fraud_detection > "$backup_path/database.sql"
        else
            warn "Database pod not found, skipping database backup"
        fi
    fi
    
    # Backup Redis data
    info "Backing up Redis data..."
    if [[ $DRY_RUN == false ]]; then
        local redis_pod=$(kubectl get pods -n "$NAMESPACE" -l app=redis -o jsonpath='{.items[0].metadata.name}')
        if [[ -n "$redis_pod" ]]; then
            kubectl exec -n "$NAMESPACE" "$redis_pod" -- redis-cli BGSAVE
            sleep 5  # Wait for background save to complete
            kubectl cp -n "$NAMESPACE" "$redis_pod:/data/dump.rdb" "$backup_path/redis-dump.rdb"
        else
            warn "Redis pod not found, skipping Redis backup"
        fi
    fi
    
    # Backup application configuration
    info "Backing up application configuration..."
    if [[ $DRY_RUN == false ]]; then
        cp -r "$PROJECT_ROOT/k8s" "$backup_path/"
        cp -r "$PROJECT_ROOT/scripts" "$backup_path/"
        cp "$PROJECT_ROOT/.env.example" "$backup_path/" 2>/dev/null || true
    fi
    
    # Create backup metadata
    cat > "$backup_path/metadata.json" << EOF
{
  "backup_id": "$backup_name",
  "environment": "$ENVIRONMENT",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "reason": "${REASON:-manual-backup}",
  "versions": {
    "api": "$(get_current_version "fraud-detection-api")",
    "streaming": "$(get_current_version "fraud-detection-streaming")"
  },
  "components": {
    "kubernetes": true,
    "database": $([ -f "$backup_path/database.sql" ] && echo "true" || echo "false"),
    "redis": $([ -f "$backup_path/redis-dump.rdb" ] && echo "true" || echo "false"),
    "configuration": true
  }
}
EOF
    
    # Compress backup
    if [[ $DRY_RUN == false ]]; then
        tar -czf "$backup_path.tar.gz" -C "$BACKUP_DIR" "$backup_name"
        rm -rf "$backup_path"
    fi
    
    success "Backup created: $backup_name.tar.gz"
    echo "Backup ID: $backup_name"
}

# Validate backup integrity
validate_backup() {
    local backup_file="$1"
    
    print_info "Validating backup: $backup_file"
    
    if [[ ! -f "$backup_file" ]]; then
        error "Backup file not found: $backup_file"
        return 1
    fi
    
    # Check if backup is a valid tar.gz file
    if ! tar -tzf "$backup_file" >/dev/null 2>&1; then
        error "Backup file is corrupted or not a valid tar.gz file"
        return 1
    fi
    
    # Extract and validate metadata
    local temp_dir=$(mktemp -d)
    tar -xzf "$backup_file" -C "$temp_dir"
    
    local backup_name=$(basename "$backup_file" .tar.gz)
    local metadata_file="$temp_dir/$backup_name/metadata.json"
    
    if [[ ! -f "$metadata_file" ]]; then
        error "Backup metadata not found"
        rm -rf "$temp_dir"
        return 1
    fi
    
    # Validate required files
    local required_files=("k8s-resources.yaml" "configmaps.yaml" "secrets.yaml")
    for file in "${required_files[@]}"; do
        if [[ ! -f "$temp_dir/$backup_name/$file" ]]; then
            error "Required backup file missing: $file"
            rm -rf "$temp_dir"
            return 1
        fi
    done
    
    rm -rf "$temp_dir"
    success "Backup validation passed"
    return 0
}

# Rollback deployment
rollback() {
    print_info "Rolling back environment: $ENVIRONMENT"
    
    if [[ -z "$VERSION" ]]; then
        # Rollback to previous version
        info "Rolling back to previous version..."
        
        local deployments=("fraud-detection-api" "fraud-detection-streaming")
        
        for deployment in "${deployments[@]}"; do
            info "Rolling back deployment: $deployment"
            
            if [[ $DRY_RUN == false ]]; then
                kubectl rollout undo deployment "$deployment" -n "$NAMESPACE"
                
                # Wait for rollback to complete
                kubectl rollout status deployment "$deployment" -n "$NAMESPACE" --timeout="${TIMEOUT}s"
            fi
        done
    else
        # Rollback to specific version
        info "Rolling back to version: $VERSION"
        
        # This would require version-specific deployment manifests
        # Implementation depends on your versioning strategy
        warn "Specific version rollback not fully implemented"
    fi
    
    # Verify rollback
    if [[ $DRY_RUN == false ]]; then
        sleep 10
        
        # Run health checks
        if "$SCRIPT_DIR/monitoring-health-check.sh" health-check --environment "$ENVIRONMENT" --quiet; then
            success "Rollback completed successfully and system is healthy"
        else
            error "Rollback completed but system health checks failed"
            return 1
        fi
    fi
    
    success "Rollback completed for environment: $ENVIRONMENT"
}

# Restore from backup
restore_backup() {
    print_info "Restoring from backup: $BACKUP_ID"
    
    local backup_file="$BACKUP_DIR/$BACKUP_ID.tar.gz"
    
    if [[ ! -f "$backup_file" ]]; then
        error "Backup file not found: $backup_file"
        exit 1
    fi
    
    # Validate backup if not skipped
    if [[ $SKIP_VALIDATION == false ]]; then
        if ! validate_backup "$backup_file"; then
            error "Backup validation failed"
            exit 1
        fi
    fi
    
    if [[ $FORCE == false ]]; then
        read -p "This will restore the system from backup and may cause data loss. Continue? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Restore cancelled"
            exit 0
        fi
    fi
    
    # Extract backup
    local temp_dir=$(mktemp -d)
    tar -xzf "$backup_file" -C "$temp_dir"
    local backup_name=$(basename "$backup_file" .tar.gz)
    local backup_path="$temp_dir/$backup_name"
    
    if [[ $DRY_RUN == false ]]; then
        # Stop current services
        info "Stopping current services..."
        kubectl scale deployment --all --replicas=0 -n "$NAMESPACE" 2>/dev/null || true
        
        # Restore Kubernetes resources
        info "Restoring Kubernetes resources..."
        kubectl apply -f "$backup_path/k8s-resources.yaml" -n "$NAMESPACE"
        kubectl apply -f "$backup_path/configmaps.yaml" -n "$NAMESPACE"
        kubectl apply -f "$backup_path/secrets.yaml" -n "$NAMESPACE"
        
        # Restore database
        if [[ -f "$backup_path/database.sql" ]]; then
            info "Restoring database..."
            local db_pod=$(kubectl get pods -n "$NAMESPACE" -l app=postgres -o jsonpath='{.items[0].metadata.name}')
            if [[ -n "$db_pod" ]]; then
                kubectl exec -i -n "$NAMESPACE" "$db_pod" -- psql -U fraud_user -d fraud_detection < "$backup_path/database.sql"
            fi
        fi
        
        # Restore Redis data
        if [[ -f "$backup_path/redis-dump.rdb" ]]; then
            info "Restoring Redis data..."
            local redis_pod=$(kubectl get pods -n "$NAMESPACE" -l app=redis -o jsonpath='{.items[0].metadata.name}')
            if [[ -n "$redis_pod" ]]; then
                kubectl cp "$backup_path/redis-dump.rdb" -n "$NAMESPACE" "$redis_pod:/data/dump.rdb"
                kubectl exec -n "$NAMESPACE" "$redis_pod" -- redis-cli DEBUG RESTART
            fi
        fi
        
        # Wait for services to be ready
        info "Waiting for services to be ready..."
        kubectl wait --for=condition=available --timeout="${TIMEOUT}s" deployment --all -n "$NAMESPACE"
    fi
    
    # Cleanup
    rm -rf "$temp_dir"
    
    success "Restore completed from backup: $BACKUP_ID"
}

# Emergency stop all services
emergency_stop() {
    print_info "Emergency stop for environment: $ENVIRONMENT"
    
    if [[ $FORCE == false ]]; then
        read -p "This will stop all services immediately. Continue? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Emergency stop cancelled"
            exit 0
        fi
    fi
    
    if [[ $DRY_RUN == false ]]; then
        # Scale down all deployments
        kubectl scale deployment --all --replicas=0 -n "$NAMESPACE"
        
        # Delete all pods forcefully if needed
        kubectl delete pods --all -n "$NAMESPACE" --force --grace-period=0 2>/dev/null || true
    fi
    
    success "Emergency stop completed for environment: $ENVIRONMENT"
}

# Emergency start all services
emergency_start() {
    print_info "Emergency start for environment: $ENVIRONMENT"
    
    if [[ $DRY_RUN == false ]]; then
        # Scale up deployments to their original replicas
        local deployments=("fraud-detection-api:3" "fraud-detection-streaming:2")
        
        for deployment_config in "${deployments[@]}"; do
            local deployment=$(echo "$deployment_config" | cut -d: -f1)
            local replicas=$(echo "$deployment_config" | cut -d: -f2)
            
            kubectl scale deployment "$deployment" --replicas="$replicas" -n "$NAMESPACE"
        done
        
        # Wait for deployments to be ready
        kubectl wait --for=condition=available --timeout="${TIMEOUT}s" deployment --all -n "$NAMESPACE"
    fi
    
    success "Emergency start completed for environment: $ENVIRONMENT"
}

# Disaster recovery
disaster_recovery() {
    print_info "Executing disaster recovery for environment: $ENVIRONMENT"
    print_info "Reason: ${REASON:-unknown}"
    
    # Create emergency backup first
    info "Creating emergency backup before recovery..."
    REASON="disaster-recovery-backup" create_backup
    
    # Determine recovery strategy based on reason
    case "$REASON" in
        "database-failure")
            info "Executing database failure recovery..."
            # Database-specific recovery steps
            ;;
        "network-failure")
            info "Executing network failure recovery..."
            # Network-specific recovery steps
            ;;
        "storage-failure")
            info "Executing storage failure recovery..."
            # Storage-specific recovery steps
            ;;
        *)
            info "Executing general disaster recovery..."
            # General recovery steps
            emergency_stop
            sleep 10
            emergency_start
            ;;
    esac
    
    # Verify recovery
    if "$SCRIPT_DIR/monitoring-health-check.sh" health-check --environment "$ENVIRONMENT"; then
        success "Disaster recovery completed successfully"
    else
        error "Disaster recovery completed but system is not healthy"
        return 1
    fi
}

# Cleanup old backups
cleanup_backups() {
    print_info "Cleaning up old backups (retention: $BACKUP_RETENTION_DAYS days)"
    
    if [[ -d "$BACKUP_DIR" ]]; then
        local deleted_count=0
        
        while IFS= read -r -d '' backup_file; do
            local file_age_days=$(( ($(date +%s) - $(stat -c %Y "$backup_file")) / 86400 ))
            
            if [[ $file_age_days -gt $BACKUP_RETENTION_DAYS ]]; then
                info "Deleting old backup: $(basename "$backup_file") (age: ${file_age_days} days)"
                
                if [[ $DRY_RUN == false ]]; then
                    rm -f "$backup_file"
                fi
                
                ((deleted_count++))
            fi
        done < <(find "$BACKUP_DIR" -name "backup-*.tar.gz" -print0)
        
        success "Cleanup completed. Deleted $deleted_count old backups."
    else
        info "No backup directory found"
    fi
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
        rollback)
            rollback
            ;;
        list-versions)
            list_versions
            ;;
        create-backup)
            create_backup
            ;;
        restore-backup)
            if [[ -z "$BACKUP_ID" ]]; then
                error "Backup ID is required for restore operation"
                exit 1
            fi
            restore_backup
            ;;
        disaster-recovery)
            disaster_recovery
            ;;
        validate-backup)
            if [[ -z "$BACKUP_ID" ]]; then
                error "Backup ID is required for validation"
                exit 1
            fi
            validate_backup "$BACKUP_DIR/$BACKUP_ID.tar.gz"
            ;;
        cleanup-backups)
            cleanup_backups
            ;;
        emergency-stop)
            emergency_stop
            ;;
        emergency-start)
            emergency_start
            ;;
        health-restore)
            print_error "Health restore functionality not implemented yet"
            exit 1
            ;;
        data-recovery)
            print_error "Data recovery functionality not implemented yet"
            exit 1
            ;;
        config-rollback)
            print_error "Config rollback functionality not implemented yet"
            exit 1
            ;;
        db-rollback)
            print_error "Database rollback functionality not implemented yet"
            exit 1
            ;;
        canary-rollback)
            print_error "Canary rollback functionality not implemented yet"
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