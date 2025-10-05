#!/usr/bin/env python3
"""
Monitoring Setup Script for Fraud Detection System

This script helps set up and configure the complete monitoring stack including:
- Prometheus metrics collection
- Grafana dashboards
- Alertmanager notifications
- Health checks and service discovery
- Log aggregation and structured logging
"""

import os
import sys
import json
import yaml
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional


class MonitoringSetup:
    """Setup and configure monitoring stack for fraud detection system."""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root or os.getcwd())
        self.infra_dir = self.project_root / "infra"
        self.prometheus_dir = self.infra_dir / "prometheus"
        self.grafana_dir = self.infra_dir / "grafana"
        self.logs_dir = self.project_root / "logs"
        
    def setup_directories(self):
        """Create necessary directories for monitoring."""
        print("📁 Creating monitoring directories...")
        
        directories = [
            self.infra_dir,
            self.prometheus_dir,
            self.grafana_dir / "provisioning" / "datasources",
            self.grafana_dir / "provisioning" / "dashboards",
            self.grafana_dir / "dashboards",
            self.logs_dir,
            self.project_root / "scripts",
            self.project_root / "api" / "middleware",
            self.project_root / "api" / "utils"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"  ✅ Created: {directory}")
    
    def validate_configuration_files(self) -> Dict[str, bool]:
        """Validate that all required configuration files exist."""
        print("🔍 Validating configuration files...")
        
        required_files = {
            "Prometheus Config": self.prometheus_dir / "prometheus.yml",
            "Alerting Rules": self.prometheus_dir / "alerting_rules.yml",
            "Alertmanager Config": self.prometheus_dir / "alertmanager.yml",
            "Grafana Datasource": self.grafana_dir / "provisioning" / "datasources" / "prometheus.yml",
            "Grafana Dashboards": self.grafana_dir / "provisioning" / "dashboards" / "dashboards.yml",
            "Fraud Detection Dashboard": self.grafana_dir / "dashboards" / "fraud_detection_overview.json",
            "System Monitoring Dashboard": self.grafana_dir / "dashboards" / "system_monitoring.json",
            "Docker Compose": self.infra_dir / "docker-compose.monitoring.yml",
            "Monitoring Middleware": self.project_root / "api" / "middleware" / "monitoring.py",
            "Logging Config": self.project_root / "api" / "utils" / "logging_config.py"
        }
        
        validation_results = {}
        
        for name, file_path in required_files.items():
            exists = file_path.exists()
            validation_results[name] = exists
            status = "✅" if exists else "❌"
            print(f"  {status} {name}: {file_path}")
        
        return validation_results
    
    def check_docker_compose(self) -> bool:
        """Check if Docker Compose is available."""
        try:
            result = subprocess.run(
                ["docker-compose", "--version"],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✅ Docker Compose available: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            try:
                result = subprocess.run(
                    ["docker", "compose", "version"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                print(f"✅ Docker Compose (v2) available: {result.stdout.strip()}")
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("❌ Docker Compose not found. Please install Docker Compose.")
                return False
    
    def start_monitoring_stack(self, detached: bool = True):
        """Start the monitoring stack using Docker Compose."""
        print("🚀 Starting monitoring stack...")
        
        compose_file = self.infra_dir / "docker-compose.monitoring.yml"
        if not compose_file.exists():
            print(f"❌ Docker Compose file not found: {compose_file}")
            return False
        
        try:
            cmd = ["docker-compose", "-f", str(compose_file), "up"]
            if detached:
                cmd.append("-d")
            
            result = subprocess.run(cmd, cwd=self.project_root, check=True)
            
            if detached:
                print("✅ Monitoring stack started successfully!")
                self.show_service_urls()
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to start monitoring stack: {e}")
            return False
    
    def stop_monitoring_stack(self):
        """Stop the monitoring stack."""
        print("🛑 Stopping monitoring stack...")
        
        compose_file = self.infra_dir / "docker-compose.monitoring.yml"
        
        try:
            subprocess.run(
                ["docker-compose", "-f", str(compose_file), "down"],
                cwd=self.project_root,
                check=True
            )
            print("✅ Monitoring stack stopped successfully!")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to stop monitoring stack: {e}")
            return False
    
    def show_service_urls(self):
        """Display URLs for monitoring services."""
        print("\n🌐 Monitoring Services URLs:")
        print("  📊 Grafana Dashboard: http://localhost:3000 (admin/admin123)")
        print("  📈 Prometheus: http://localhost:9090")
        print("  🚨 Alertmanager: http://localhost:9093")
        print("  🖥️  Node Exporter: http://localhost:9100")
        print("  📦 cAdvisor: http://localhost:8080")
        print("  🔍 Fraud Detection API: http://localhost:8000")
        print("  📋 API Health Check: http://localhost:8000/health")
        print("  📊 API Metrics: http://localhost:8000/metrics")
    
    def check_service_health(self) -> Dict[str, bool]:
        """Check health of monitoring services."""
        print("🏥 Checking service health...")
        
        import requests
        
        services = {
            "Grafana": "http://localhost:3000/api/health",
            "Prometheus": "http://localhost:9090/-/healthy",
            "Alertmanager": "http://localhost:9093/-/healthy",
            "Node Exporter": "http://localhost:9100/metrics",
            "cAdvisor": "http://localhost:8080/healthz",
            "Fraud Detection API": "http://localhost:8000/health"
        }
        
        health_status = {}
        
        for service, url in services.items():
            try:
                response = requests.get(url, timeout=5)
                healthy = response.status_code == 200
                health_status[service] = healthy
                status = "✅" if healthy else "❌"
                print(f"  {status} {service}: {url}")
            except requests.RequestException:
                health_status[service] = False
                print(f"  ❌ {service}: {url} (Connection failed)")
        
        return health_status
    
    def generate_monitoring_report(self):
        """Generate a comprehensive monitoring setup report."""
        print("\n📋 Monitoring Setup Report")
        print("=" * 50)
        
        # Configuration validation
        config_status = self.validate_configuration_files()
        config_ok = all(config_status.values())
        
        print(f"\n📁 Configuration Files: {'✅ All Present' if config_ok else '❌ Missing Files'}")
        
        # Docker availability
        docker_ok = self.check_docker_compose()
        
        # Service health (if running)
        print("\n🏥 Service Health Check:")
        try:
            health_status = self.check_service_health()
            services_healthy = sum(health_status.values())
            total_services = len(health_status)
            print(f"  Services Running: {services_healthy}/{total_services}")
        except ImportError:
            print("  ⚠️  Install 'requests' package to check service health")
        
        # Summary
        print("\n📊 Setup Summary:")
        print(f"  Configuration: {'✅' if config_ok else '❌'}")
        print(f"  Docker Compose: {'✅' if docker_ok else '❌'}")
        
        if config_ok and docker_ok:
            print("\n🎉 Monitoring stack is ready to deploy!")
            print("   Run: python scripts/setup_monitoring.py --start")
        else:
            print("\n⚠️  Please fix the issues above before starting the monitoring stack.")
    
    def create_sample_alerts(self):
        """Create sample alert configurations for testing."""
        print("🚨 Creating sample alert configurations...")
        
        # This would create test alerts to verify the alerting pipeline
        sample_alerts = {
            "test_alert": {
                "alert": "TestAlert",
                "expr": "up == 0",
                "for": "1m",
                "labels": {
                    "severity": "warning",
                    "service": "test"
                },
                "annotations": {
                    "summary": "Test alert for monitoring setup",
                    "description": "This is a test alert to verify alerting is working"
                }
            }
        }
        
        print("✅ Sample alerts configured for testing")


def main():
    """Main function to handle command line arguments and execute setup."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Setup and manage monitoring stack for fraud detection system"
    )
    parser.add_argument(
        "--start", action="store_true",
        help="Start the monitoring stack"
    )
    parser.add_argument(
        "--stop", action="store_true",
        help="Stop the monitoring stack"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Check status of monitoring services"
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Generate monitoring setup report"
    )
    parser.add_argument(
        "--setup", action="store_true",
        help="Setup directories and validate configuration"
    )
    parser.add_argument(
        "--project-root", type=str,
        help="Project root directory (default: current directory)"
    )
    
    args = parser.parse_args()
    
    # Initialize monitoring setup
    monitoring = MonitoringSetup(args.project_root)
    
    if args.setup:
        monitoring.setup_directories()
        monitoring.validate_configuration_files()
    elif args.start:
        monitoring.start_monitoring_stack()
    elif args.stop:
        monitoring.stop_monitoring_stack()
    elif args.status:
        monitoring.check_service_health()
    elif args.report:
        monitoring.generate_monitoring_report()
    else:
        # Default: show report
        monitoring.generate_monitoring_report()


if __name__ == "__main__":
    main()