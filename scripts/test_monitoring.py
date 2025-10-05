#!/usr/bin/env python3
"""
Monitoring Stack Testing Script

This script validates the monitoring infrastructure setup including:
- Prometheus metrics collection
- Grafana dashboard accessibility
- Alertmanager configuration
- Health check endpoints
- Custom metrics validation
- Alert rule testing
"""

import asyncio
import json
import logging
import requests
import time
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MonitoringTester:
    """Comprehensive monitoring stack testing."""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.services = {
            'prometheus': 'http://localhost:9090',
            'grafana': 'http://localhost:3000',
            'alertmanager': 'http://localhost:9093',
            'fraud_api': 'http://localhost:8000'
        }
        self.test_results = {}
        
    def run_all_tests(self) -> Dict[str, bool]:
        """Run comprehensive monitoring tests."""
        logger.info("Starting comprehensive monitoring tests...")
        
        tests = [
            ('Service Health', self.test_service_health),
            ('Prometheus Config', self.test_prometheus_config),
            ('Metrics Collection', self.test_metrics_collection),
            ('Grafana Dashboards', self.test_grafana_dashboards),
            ('Alert Rules', self.test_alert_rules),
            ('Alertmanager Config', self.test_alertmanager_config),
            ('API Health Endpoints', self.test_api_health_endpoints),
            ('Custom Metrics', self.test_custom_metrics),
            ('Performance Monitoring', self.test_performance_monitoring),
            ('Log Integration', self.test_log_integration)
        ]
        
        for test_name, test_func in tests:
            try:
                logger.info(f"Running test: {test_name}")
                result = test_func()
                self.test_results[test_name] = result
                status = "✅ PASS" if result else "❌ FAIL"
                logger.info(f"{test_name}: {status}")
            except Exception as e:
                logger.error(f"{test_name}: ❌ ERROR - {str(e)}")
                self.test_results[test_name] = False
        
        self.generate_test_report()
        return self.test_results
    
    def test_service_health(self) -> bool:
        """Test if all monitoring services are healthy."""
        all_healthy = True
        
        health_endpoints = {
            'prometheus': f"{self.services['prometheus']}/-/healthy",
            'grafana': f"{self.services['grafana']}/api/health",
            'alertmanager': f"{self.services['alertmanager']}/-/healthy",
            'fraud_api': f"{self.services['fraud_api']}/health"
        }
        
        for service, endpoint in health_endpoints.items():
            try:
                response = requests.get(endpoint, timeout=10)
                if response.status_code == 200:
                    logger.info(f"✅ {service} is healthy")
                else:
                    logger.error(f"❌ {service} health check failed: {response.status_code}")
                    all_healthy = False
            except requests.RequestException as e:
                logger.error(f"❌ {service} is unreachable: {str(e)}")
                all_healthy = False
        
        return all_healthy
    
    def test_prometheus_config(self) -> bool:
        """Test Prometheus configuration and targets."""
        try:
            # Check Prometheus config
            config_response = requests.get(f"{self.services['prometheus']}/api/v1/status/config")
            if config_response.status_code != 200:
                logger.error("Failed to retrieve Prometheus config")
                return False
            
            # Check targets
            targets_response = requests.get(f"{self.services['prometheus']}/api/v1/targets")
            if targets_response.status_code != 200:
                logger.error("Failed to retrieve Prometheus targets")
                return False
            
            targets_data = targets_response.json()
            active_targets = targets_data.get('data', {}).get('activeTargets', [])
            
            expected_jobs = ['prometheus', 'fraud-detection-api', 'node-exporter', 'cadvisor']
            found_jobs = set(target['labels']['job'] for target in active_targets)
            
            missing_jobs = set(expected_jobs) - found_jobs
            if missing_jobs:
                logger.error(f"Missing Prometheus jobs: {missing_jobs}")
                return False
            
            # Check for unhealthy targets
            unhealthy_targets = [t for t in active_targets if t['health'] != 'up']
            if unhealthy_targets:
                logger.warning(f"Unhealthy targets found: {len(unhealthy_targets)}")
                for target in unhealthy_targets:
                    logger.warning(f"  - {target['labels']['job']}: {target['lastError']}")
            
            logger.info(f"✅ Prometheus config valid, {len(active_targets)} targets configured")
            return True
            
        except Exception as e:
            logger.error(f"Prometheus config test failed: {str(e)}")
            return False
    
    def test_metrics_collection(self) -> bool:
        """Test if metrics are being collected properly."""
        try:
            # Test basic Prometheus metrics
            metrics_response = requests.get(f"{self.services['prometheus']}/api/v1/label/__name__/values")
            if metrics_response.status_code != 200:
                logger.error("Failed to retrieve metrics list")
                return False
            
            metrics_data = metrics_response.json()
            available_metrics = metrics_data.get('data', [])
            
            # Check for essential metrics
            essential_metrics = [
                'up',
                'prometheus_config_last_reload_successful',
                'http_requests_total',
                'process_cpu_seconds_total'
            ]
            
            missing_metrics = [m for m in essential_metrics if m not in available_metrics]
            if missing_metrics:
                logger.error(f"Missing essential metrics: {missing_metrics}")
                return False
            
            # Test API-specific metrics
            api_metrics_response = requests.get(f"{self.services['fraud_api']}/metrics")
            if api_metrics_response.status_code == 200:
                api_metrics_text = api_metrics_response.text
                expected_api_metrics = [
                    'http_requests_total',
                    'http_request_duration_seconds',
                    'fraud_predictions_total'
                ]
                
                for metric in expected_api_metrics:
                    if metric not in api_metrics_text:
                        logger.warning(f"API metric '{metric}' not found")
            
            logger.info(f"✅ Metrics collection working, {len(available_metrics)} metrics available")
            return True
            
        except Exception as e:
            logger.error(f"Metrics collection test failed: {str(e)}")
            return False
    
    def test_grafana_dashboards(self) -> bool:
        """Test Grafana dashboard accessibility and data sources."""
        try:
            # Test Grafana API access
            auth = ('admin', 'admin123')
            
            # Check data sources
            datasources_response = requests.get(
                f"{self.services['grafana']}/api/datasources",
                auth=auth
            )
            
            if datasources_response.status_code != 200:
                logger.error("Failed to access Grafana datasources")
                return False
            
            datasources = datasources_response.json()
            prometheus_ds = next((ds for ds in datasources if ds['type'] == 'prometheus'), None)
            
            if not prometheus_ds:
                logger.error("Prometheus datasource not configured in Grafana")
                return False
            
            # Test datasource connectivity
            ds_test_response = requests.get(
                f"{self.services['grafana']}/api/datasources/{prometheus_ds['id']}/health",
                auth=auth
            )
            
            if ds_test_response.status_code != 200:
                logger.error("Prometheus datasource health check failed")
                return False
            
            # Check dashboards
            dashboards_response = requests.get(
                f"{self.services['grafana']}/api/search?type=dash-db",
                auth=auth
            )
            
            if dashboards_response.status_code == 200:
                dashboards = dashboards_response.json()
                logger.info(f"✅ Grafana accessible, {len(dashboards)} dashboards found")
            else:
                logger.warning("Could not retrieve dashboard list")
            
            return True
            
        except Exception as e:
            logger.error(f"Grafana dashboard test failed: {str(e)}")
            return False
    
    def test_alert_rules(self) -> bool:
        """Test Prometheus alert rules configuration."""
        try:
            # Check alert rules
            rules_response = requests.get(f"{self.services['prometheus']}/api/v1/rules")
            if rules_response.status_code != 200:
                logger.error("Failed to retrieve alert rules")
                return False
            
            rules_data = rules_response.json()
            rule_groups = rules_data.get('data', {}).get('groups', [])
            
            if not rule_groups:
                logger.error("No alert rule groups found")
                return False
            
            total_rules = sum(len(group.get('rules', [])) for group in rule_groups)
            
            # Check for specific rule groups
            expected_groups = ['api_alerts', 'ml_alerts', 'system_alerts', 'business_alerts']
            found_groups = [group['name'] for group in rule_groups]
            
            missing_groups = set(expected_groups) - set(found_groups)
            if missing_groups:
                logger.warning(f"Missing alert rule groups: {missing_groups}")
            
            # Check active alerts
            alerts_response = requests.get(f"{self.services['prometheus']}/api/v1/alerts")
            if alerts_response.status_code == 200:
                alerts_data = alerts_response.json()
                active_alerts = alerts_data.get('data', {}).get('alerts', [])
                firing_alerts = [a for a in active_alerts if a['state'] == 'firing']
                
                if firing_alerts:
                    logger.warning(f"⚠️ {len(firing_alerts)} alerts currently firing")
                    for alert in firing_alerts[:3]:  # Show first 3
                        logger.warning(f"  - {alert['labels']['alertname']}: {alert['annotations'].get('summary', 'No summary')}")
            
            logger.info(f"✅ Alert rules configured, {total_rules} rules in {len(rule_groups)} groups")
            return True
            
        except Exception as e:
            logger.error(f"Alert rules test failed: {str(e)}")
            return False
    
    def test_alertmanager_config(self) -> bool:
        """Test Alertmanager configuration and status."""
        try:
            # Check Alertmanager status
            status_response = requests.get(f"{self.services['alertmanager']}/api/v1/status")
            if status_response.status_code != 200:
                logger.error("Failed to retrieve Alertmanager status")
                return False
            
            # Check configuration
            config_response = requests.get(f"{self.services['alertmanager']}/api/v1/status")
            if config_response.status_code == 200:
                logger.info("✅ Alertmanager configuration accessible")
            
            # Check receivers
            receivers_response = requests.get(f"{self.services['alertmanager']}/api/v1/receivers")
            if receivers_response.status_code == 200:
                receivers_data = receivers_response.json()
                receivers = receivers_data.get('data', [])
                logger.info(f"✅ Alertmanager has {len(receivers)} receivers configured")
            
            return True
            
        except Exception as e:
            logger.error(f"Alertmanager config test failed: {str(e)}")
            return False
    
    def test_api_health_endpoints(self) -> bool:
        """Test API health check endpoints."""
        try:
            health_endpoints = {
                '/health': 'Basic health check',
                '/health/detailed': 'Detailed health check',
                '/health/ready': 'Readiness probe',
                '/health/live': 'Liveness probe'
            }
            
            all_healthy = True
            
            for endpoint, description in health_endpoints.items():
                try:
                    response = requests.get(f"{self.services['fraud_api']}{endpoint}", timeout=5)
                    if response.status_code == 200:
                        data = response.json()
                        status = data.get('status', 'unknown')
                        logger.info(f"✅ {description}: {status}")
                    else:
                        logger.error(f"❌ {description} failed: {response.status_code}")
                        all_healthy = False
                except requests.RequestException as e:
                    logger.error(f"❌ {description} error: {str(e)}")
                    all_healthy = False
            
            return all_healthy
            
        except Exception as e:
            logger.error(f"API health endpoints test failed: {str(e)}")
            return False
    
    def test_custom_metrics(self) -> bool:
        """Test custom application metrics."""
        try:
            # Generate some test requests to create metrics
            test_endpoints = [
                '/health',
                '/metrics',
                '/docs'
            ]
            
            logger.info("Generating test requests to create metrics...")
            for endpoint in test_endpoints:
                for _ in range(3):
                    try:
                        requests.get(f"{self.services['fraud_api']}{endpoint}", timeout=5)
                    except:
                        pass
            
            time.sleep(2)  # Wait for metrics to be scraped
            
            # Check if custom metrics are available
            metrics_response = requests.get(f"{self.services['fraud_api']}/metrics")
            if metrics_response.status_code != 200:
                logger.error("Failed to retrieve API metrics")
                return False
            
            metrics_text = metrics_response.text
            
            # Check for custom metrics
            custom_metrics = [
                'http_requests_total',
                'http_request_duration_seconds',
                'active_connections_total'
            ]
            
            found_metrics = []
            for metric in custom_metrics:
                if metric in metrics_text:
                    found_metrics.append(metric)
                else:
                    logger.warning(f"Custom metric '{metric}' not found")
            
            logger.info(f"✅ Custom metrics test completed, {len(found_metrics)}/{len(custom_metrics)} metrics found")
            return len(found_metrics) > 0
            
        except Exception as e:
            logger.error(f"Custom metrics test failed: {str(e)}")
            return False
    
    def test_performance_monitoring(self) -> bool:
        """Test performance monitoring capabilities."""
        try:
            # Test performance metrics endpoint
            perf_response = requests.get(f"{self.services['fraud_api']}/health/detailed")
            if perf_response.status_code != 200:
                logger.error("Failed to retrieve performance metrics")
                return False
            
            perf_data = perf_response.json()
            
            # Check for performance metrics
            expected_metrics = ['system_metrics', 'performance_metrics']
            found_metrics = []
            
            for metric in expected_metrics:
                if metric in perf_data:
                    found_metrics.append(metric)
                    logger.info(f"✅ Performance metric '{metric}' available")
                else:
                    logger.warning(f"Performance metric '{metric}' not found")
            
            return len(found_metrics) > 0
            
        except Exception as e:
            logger.error(f"Performance monitoring test failed: {str(e)}")
            return False
    
    def test_log_integration(self) -> bool:
        """Test structured logging integration."""
        try:
            # Check if logging configuration exists
            logging_config_path = self.base_dir / 'api' / 'utils' / 'logging_config.py'
            if not logging_config_path.exists():
                logger.error("Logging configuration file not found")
                return False
            
            # Test log format by making API requests
            logger.info("Testing log integration with API requests...")
            
            # Make test requests to generate logs
            test_requests = [
                '/health',
                '/docs',
                '/openapi.json'
            ]
            
            for endpoint in test_requests:
                try:
                    requests.get(f"{self.services['fraud_api']}{endpoint}", timeout=5)
                except:
                    pass
            
            logger.info("✅ Log integration test completed")
            return True
            
        except Exception as e:
            logger.error(f"Log integration test failed: {str(e)}")
            return False
    
    def generate_test_report(self):
        """Generate comprehensive test report."""
        report_path = self.base_dir / 'monitoring_test_report.json'
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'test_results': self.test_results,
            'summary': {
                'total_tests': len(self.test_results),
                'passed': sum(1 for result in self.test_results.values() if result),
                'failed': sum(1 for result in self.test_results.values() if not result),
                'success_rate': f"{(sum(1 for result in self.test_results.values() if result) / len(self.test_results) * 100):.1f}%"
            },
            'recommendations': self._generate_recommendations()
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\n📊 Test Report Generated: {report_path}")
        logger.info(f"📈 Success Rate: {report['summary']['success_rate']}")
        logger.info(f"✅ Passed: {report['summary']['passed']}")
        logger.info(f"❌ Failed: {report['summary']['failed']}")
        
        if report['recommendations']:
            logger.info("\n💡 Recommendations:")
            for rec in report['recommendations']:
                logger.info(f"  - {rec}")
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        if not self.test_results.get('Service Health', True):
            recommendations.append("Check Docker services and restart monitoring stack")
        
        if not self.test_results.get('Prometheus Config', True):
            recommendations.append("Review Prometheus configuration and target endpoints")
        
        if not self.test_results.get('Metrics Collection', True):
            recommendations.append("Verify metrics endpoints and middleware configuration")
        
        if not self.test_results.get('Grafana Dashboards', True):
            recommendations.append("Check Grafana datasource configuration and credentials")
        
        if not self.test_results.get('Alert Rules', True):
            recommendations.append("Review alert rules configuration and syntax")
        
        if not self.test_results.get('Custom Metrics', True):
            recommendations.append("Implement custom metrics in application middleware")
        
        return recommendations

def main():
    """Main function to run monitoring tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test monitoring infrastructure')
    parser.add_argument('--quick', action='store_true', help='Run quick health checks only')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    tester = MonitoringTester()
    
    if args.quick:
        logger.info("Running quick health checks...")
        result = tester.test_service_health()
        if result:
            logger.info("✅ Quick health check passed")
        else:
            logger.error("❌ Quick health check failed")
    else:
        logger.info("Running comprehensive monitoring tests...")
        results = tester.run_all_tests()
        
        success_rate = sum(1 for r in results.values() if r) / len(results) * 100
        
        if success_rate >= 80:
            logger.info(f"🎉 Monitoring stack is healthy! ({success_rate:.1f}% success rate)")
        elif success_rate >= 60:
            logger.warning(f"⚠️ Monitoring stack has issues ({success_rate:.1f}% success rate)")
        else:
            logger.error(f"❌ Monitoring stack needs attention ({success_rate:.1f}% success rate)")

if __name__ == '__main__':
    main()