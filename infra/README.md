# Fraud Detection System - Monitoring Infrastructure

This directory contains the complete monitoring and observability stack for the fraud detection system, including Prometheus metrics collection, Grafana dashboards, Alertmanager notifications, and structured logging configuration.

## 📋 Overview

The monitoring stack provides:

- **Metrics Collection**: Prometheus for collecting and storing time-series metrics
- **Visualization**: Grafana dashboards for system and business metrics
- **Alerting**: Alertmanager for intelligent alert routing and notifications
- **Health Monitoring**: Comprehensive health checks and service discovery
- **Log Aggregation**: Structured logging with multiple formatters
- **Performance Monitoring**: Real-time performance metrics and SLA tracking

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Fraud API     │───▶│   Prometheus    │───▶│    Grafana      │
│   (FastAPI)     │    │   (Metrics)     │    │  (Dashboards)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Structured     │    │  Alertmanager   │    │   Monitoring    │
│   Logging       │    │  (Notifications)│    │   Middleware    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📁 Directory Structure

```
infra/
├── prometheus/
│   ├── prometheus.yml          # Prometheus configuration
│   ├── alerting_rules.yml      # Alert rules and thresholds
│   └── alertmanager.yml        # Alert routing and notifications
├── grafana/
│   ├── provisioning/
│   │   ├── datasources/
│   │   │   └── prometheus.yml  # Grafana datasource config
│   │   └── dashboards/
│   │       └── dashboards.yml  # Dashboard provisioning
│   └── dashboards/
│       ├── fraud_detection_overview.json
│       └── system_monitoring.json
├── docker-compose.monitoring.yml  # Complete monitoring stack
└── README.md                      # This file
```

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose installed
- Python 3.8+ for the setup script
- At least 4GB RAM available for the monitoring stack

### 1. Setup Monitoring Infrastructure

```bash
# Run the monitoring setup script
python scripts/setup_monitoring.py --setup

# Generate setup report
python scripts/setup_monitoring.py --report
```

### 2. Start the Monitoring Stack

```bash
# Start all monitoring services
python scripts/setup_monitoring.py --start

# Or manually with Docker Compose
docker-compose -f infra/docker-compose.monitoring.yml up -d
```

### 3. Access Monitoring Services

- **Grafana Dashboard**: http://localhost:3000 (admin/admin123)
- **Prometheus**: http://localhost:9090
- **Alertmanager**: http://localhost:9093
- **Fraud Detection API**: http://localhost:8000
- **API Health Check**: http://localhost:8000/health
- **API Metrics**: http://localhost:8000/metrics

## 📊 Dashboards

### Fraud Detection Overview
- API request rate and response times
- Fraud detection accuracy and model performance
- Business metrics (transaction volume, fraud rate)
- Error rates and system health

### System Infrastructure Monitoring
- CPU, Memory, and Disk usage
- Network I/O and system load
- Service availability and uptime
- Container metrics and resource utilization

## 🚨 Alerting

### Alert Categories

1. **API Alerts**
   - High error rate (>5%)
   - Slow response time (>2s)
   - Service unavailability

2. **ML Model Alerts**
   - Model inference failures
   - Prediction accuracy degradation
   - Feature pipeline issues

3. **System Alerts**
   - High CPU/Memory usage (>85%)
   - Disk space low (<15%)
   - Database connection issues

4. **Business Alerts**
   - Unusual fraud detection patterns
   - Transaction volume anomalies
   - Model drift detection

### Alert Routing

- **Critical**: PagerDuty + Slack + Email
- **Warning**: Slack + Email
- **Info**: Email only

## 📈 Metrics Collection

### Application Metrics

```python
# Custom metrics in your code
from api.middleware.monitoring import (
    request_count,
    request_duration,
    fraud_predictions_total,
    model_inference_duration
)

# Track business metrics
fraud_predictions_total.labels(result="fraud").inc()
model_inference_duration.observe(inference_time)
```

### Available Metrics

- `http_requests_total`: Total HTTP requests
- `http_request_duration_seconds`: Request duration
- `fraud_predictions_total`: Fraud prediction counts
- `model_inference_duration_seconds`: ML model inference time
- `database_query_duration_seconds`: Database query performance
- `active_connections`: Current active connections
- `feature_processing_duration_seconds`: Feature pipeline performance

## 🔧 Configuration

### Prometheus Configuration

Edit `prometheus/prometheus.yml` to:
- Add new scrape targets
- Modify scrape intervals
- Configure service discovery

### Grafana Configuration

Dashboards are automatically provisioned from `grafana/dashboards/`.
To add new dashboards:
1. Export dashboard JSON from Grafana UI
2. Save to `grafana/dashboards/`
3. Restart Grafana service

### Alert Configuration

Edit `prometheus/alerting_rules.yml` to:
- Add new alert rules
- Modify thresholds
- Update alert labels and annotations

Edit `prometheus/alertmanager.yml` to:
- Configure notification channels
- Set up alert routing rules
- Define inhibition rules

## 🔍 Troubleshooting

### Common Issues

1. **Services not starting**
   ```bash
   # Check Docker logs
   docker-compose -f infra/docker-compose.monitoring.yml logs
   
   # Check service health
   python scripts/setup_monitoring.py --status
   ```

2. **Metrics not appearing**
   - Verify Prometheus targets: http://localhost:9090/targets
   - Check API metrics endpoint: http://localhost:8000/metrics
   - Ensure middleware is properly configured

3. **Alerts not firing**
   - Check alert rules: http://localhost:9090/alerts
   - Verify Alertmanager config: http://localhost:9093
   - Test notification channels

4. **Grafana dashboards empty**
   - Verify Prometheus datasource connection
   - Check dashboard queries and time ranges
   - Ensure metrics are being collected

### Health Checks

```bash
# Check all service health
python scripts/setup_monitoring.py --status

# Individual service checks
curl http://localhost:9090/-/healthy  # Prometheus
curl http://localhost:3000/api/health # Grafana
curl http://localhost:9093/-/healthy  # Alertmanager
curl http://localhost:8000/health     # Fraud Detection API
```

## 🔒 Security Considerations

1. **Change Default Passwords**
   ```bash
   # Update Grafana admin password
   docker exec -it grafana grafana-cli admin reset-admin-password newpassword
   ```

2. **Network Security**
   - Use Docker networks for service isolation
   - Configure firewall rules for external access
   - Enable HTTPS in production

3. **Data Retention**
   - Configure Prometheus retention policies
   - Set up log rotation for application logs
   - Implement backup strategies for metrics data

## 📚 Advanced Configuration

### Custom Metrics

```python
# Add custom business metrics
from prometheus_client import Counter, Histogram, Gauge

# Define custom metrics
custom_counter = Counter('custom_events_total', 'Custom events', ['event_type'])
custom_histogram = Histogram('custom_duration_seconds', 'Custom duration')
custom_gauge = Gauge('custom_value', 'Custom gauge value')

# Use in your code
custom_counter.labels(event_type='fraud_detected').inc()
with custom_histogram.time():
    # Your code here
    pass
custom_gauge.set(42)
```

### Alert Webhook Integration

```yaml
# Add to alertmanager.yml
receivers:
- name: 'webhook'
  webhook_configs:
  - url: 'http://your-webhook-endpoint/alerts'
    send_resolved: true
```

### Log Aggregation

```python
# Use structured logging
from api.utils.logging_config import get_logger, log_business_event

logger = get_logger(__name__)

# Log business events
log_business_event(
    event_type="fraud_detected",
    transaction_id="txn_123",
    risk_score=0.95,
    model_version="v1.2.3"
)
```

## 🤝 Contributing

When adding new monitoring features:
1. Update relevant configuration files
2. Add corresponding dashboard panels
3. Create appropriate alert rules
4. Update this documentation
5. Test the complete monitoring pipeline

## 📞 Support

For monitoring-related issues:
1. Check the troubleshooting section above
2. Review service logs and health checks
3. Consult Prometheus and Grafana documentation
4. Create an issue with detailed error information

---

**Note**: This monitoring setup is designed for development and testing. For production deployment, consider additional security hardening, high availability setup, and proper backup strategies.