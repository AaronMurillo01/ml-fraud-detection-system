"""
Monitoring and Observability Endpoints

This module provides all the monitoring endpoints for the FraudGuard AI system.
Think of this as the "health dashboard" for our application - it tells us if everything
is running smoothly or if something needs attention.

What's in here:
- Health checks (is the system alive and working?)
- Metrics (how well is the system performing?)
- Alerts (what problems have we detected?)
- System stats (overall system information)
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import PlainTextResponse, JSONResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from config import settings
from monitoring import (
    run_health_checks,
    SystemHealth,
    HealthStatus,
    get_health_checker
)
from monitoring.tracing import get_tracing_manager, get_current_correlation_id
from monitoring.alerting import get_alert_manager, Alert, AlertSeverity
from config.monitoring import registry, metrics_collector
from api.dependencies import require_admin_role

# Set up logging so we can track what's happening
logger = logging.getLogger(__name__)

# Create the router - this groups all our monitoring endpoints together
router = APIRouter(prefix="/monitoring", tags=["Monitoring"])


@router.get("/health", response_model=Dict[str, Any])
async def health_check():
    """
    Comprehensive Health Check

    This is like asking "How are you feeling?" to the entire system.
    It checks all the important parts (database, cache, models, etc.) and
    tells us if everything is working properly.

    Returns:
        - status: "healthy", "degraded", or "unhealthy"
        - uptime: how long the system has been running
        - checks: detailed status of each component
    """
    try:
        # Run all the health checks (database, redis, models, etc.)
        system_health = await run_health_checks()

        # If something looks wrong, create an alert so someone can investigate
        alert_manager = get_alert_manager()
        await alert_manager.check_system_health_alerts(system_health)

        # Package up the results in a nice format for the API response
        response = {
            "status": system_health.status.value,
            "timestamp": system_health.timestamp.isoformat(),
            "uptime_seconds": system_health.uptime_seconds,
            "version": system_health.version,
            "environment": system_health.environment,
            "checks": [
                {
                    "name": check.name,
                    "status": check.status.value,
                    "message": check.message,
                    "duration_ms": check.duration_ms,
                    "timestamp": check.timestamp.isoformat(),
                    "details": check.details,
                    "error": check.error
                }
                for check in system_health.checks
            ]
        }

        # Choose the right HTTP status code based on health
        # 200 = everything is fine (or mostly fine)
        # 503 = something is seriously wrong
        if system_health.status == HealthStatus.HEALTHY:
            status_code = 200
        elif system_health.status == HealthStatus.DEGRADED:
            status_code = 200  # Still working, just not perfectly
        else:
            status_code = 503  # System is down or broken

        return JSONResponse(content=response, status_code=status_code)

    except Exception as e:
        # If the health check itself fails, that's really bad
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            },
            status_code=503
        )


@router.get("/health/live")
async def liveness_probe():
    """
    Liveness Probe (Kubernetes)

    This is the simplest check - it just confirms the application is running.
    Think of it as checking if someone is breathing. If this fails, Kubernetes
    will restart the container.
    """
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}


@router.get("/health/ready")
async def readiness_probe():
    """
    Readiness Probe (Kubernetes)

    This checks if the application is ready to handle traffic. It's like asking
    "Are you ready to work?" We check critical components like the database
    and cache. If they're not working, we tell Kubernetes not to send us traffic yet.
    """
    try:
        # Get the health checker that knows how to test each component
        health_checker = get_health_checker()

        # Only check the absolutely critical stuff (database and cache)
        # We don't want to wait for everything - just the essentials
        critical_checks = ["database", "redis"]
        tasks = []

        for check_name in critical_checks:
            if check_name in health_checker.checks:
                tasks.append(health_checker.checks[check_name]())

        # If there are no critical checks configured, we're ready by default
        if not tasks:
            return {"status": "ready", "timestamp": datetime.utcnow().isoformat()}

        # Run all the critical checks at the same time (faster!)
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # If ANY critical component is broken, we're not ready
        for result in results:
            if isinstance(result, Exception):
                # The check itself crashed - definitely not ready
                return JSONResponse(
                    content={
                        "status": "not_ready",
                        "error": str(result),
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    status_code=503
                )
            elif result.status == HealthStatus.UNHEALTHY:
                # The component is unhealthy - not ready
                return JSONResponse(
                    content={
                        "status": "not_ready",
                        "message": result.message,
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    status_code=503
                )

        # All critical components are healthy - we're ready!
        return {"status": "ready", "timestamp": datetime.utcnow().isoformat()}

    except Exception as e:
        # Something went wrong with the readiness check itself
        logger.error(f"Readiness probe failed: {e}")
        return JSONResponse(
            content={
                "status": "not_ready",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            },
            status_code=503
        )


@router.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    try:
        return PlainTextResponse(
            generate_latest(registry),
            media_type=CONTENT_TYPE_LATEST
        )
    except Exception as e:
        logger.error(f"Failed to generate metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate metrics")


@router.get("/alerts", dependencies=[Depends(require_admin_role)])
async def get_alerts(
    active_only: bool = Query(True, description="Return only active alerts"),
    severity: Optional[AlertSeverity] = Query(None, description="Filter by severity"),
    limit: int = Query(100, description="Maximum number of alerts to return")
):
    """Get system alerts."""
    try:
        alert_manager = get_alert_manager()
        
        if active_only:
            alerts = alert_manager.get_active_alerts()
        else:
            alerts = alert_manager.get_alert_history(limit=limit)
        
        # Filter by severity if specified
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
        
        # Convert to response format
        response = [
            {
                "id": alert.id,
                "type": alert.type.value,
                "severity": alert.severity.value,
                "title": alert.title,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "resolved": alert.resolved,
                "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None,
                "correlation_id": alert.correlation_id,
                "metadata": alert.metadata
            }
            for alert in alerts
        ]
        
        return {
            "alerts": response,
            "total": len(response),
            "active_count": len(alert_manager.get_active_alerts())
        }
        
    except Exception as e:
        logger.error(f"Failed to get alerts: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve alerts")


@router.post("/alerts/{alert_id}/resolve", dependencies=[Depends(require_admin_role)])
async def resolve_alert(alert_id: str):
    """Resolve an active alert."""
    try:
        alert_manager = get_alert_manager()
        alert_manager.resolve_alert(alert_id)
        
        return {
            "message": f"Alert {alert_id} resolved",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to resolve alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to resolve alert")


@router.get("/trace", dependencies=[Depends(require_admin_role)])
async def get_trace_info():
    """Get current trace and correlation information."""
    try:
        tracing_manager = get_tracing_manager()
        correlation_context = tracing_manager.get_correlation_context()
        
        return {
            "correlation_id": get_current_correlation_id(),
            "context": correlation_context,
            "tracing_enabled": settings.enable_tracing,
            "jaeger_endpoint": getattr(settings, 'jaeger_endpoint', None),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get trace info: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve trace information")


@router.get("/stats", dependencies=[Depends(require_admin_role)])
async def get_system_stats():
    """Get comprehensive system statistics."""
    try:
        # Get health check results
        system_health = await run_health_checks()
        
        # Get alert statistics
        alert_manager = get_alert_manager()
        active_alerts = alert_manager.get_active_alerts()
        alert_history = alert_manager.get_alert_history(limit=1000)
        
        # Calculate alert statistics
        alert_stats = {
            "active_count": len(active_alerts),
            "total_count": len(alert_history),
            "by_severity": {},
            "by_type": {},
            "recent_count": 0
        }
        
        # Count alerts by severity and type
        for alert in alert_history:
            severity = alert.severity.value
            alert_type = alert.type.value
            
            alert_stats["by_severity"][severity] = alert_stats["by_severity"].get(severity, 0) + 1
            alert_stats["by_type"][alert_type] = alert_stats["by_type"].get(alert_type, 0) + 1
            
            # Count recent alerts (last 24 hours)
            if alert.timestamp > datetime.utcnow() - timedelta(hours=24):
                alert_stats["recent_count"] += 1
        
        # Get model loader statistics (if available)
        model_stats = {}
        try:
            from service.model_loader import get_model_loader
            model_loader = get_model_loader()
            if model_loader:
                model_stats = model_loader.get_statistics()
        except Exception:
            pass
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": system_health.uptime_seconds,
            "environment": settings.environment.value,
            "version": settings.app_version,
            "health": {
                "status": system_health.status.value,
                "checks_total": len(system_health.checks),
                "checks_healthy": len([c for c in system_health.checks if c.status == HealthStatus.HEALTHY]),
                "checks_degraded": len([c for c in system_health.checks if c.status == HealthStatus.DEGRADED]),
                "checks_unhealthy": len([c for c in system_health.checks if c.status == HealthStatus.UNHEALTHY])
            },
            "alerts": alert_stats,
            "models": model_stats,
            "configuration": {
                "debug": settings.debug,
                "metrics_enabled": settings.enable_metrics,
                "tracing_enabled": settings.enable_tracing,
                "rate_limiting_enabled": settings.enable_rate_limiting,
                "authentication_required": settings.require_authentication
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get system stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system statistics")


@router.get("/dashboard")
async def monitoring_dashboard():
    """Simple monitoring dashboard (HTML)."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fraud Detection API - Monitoring Dashboard</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; }
            .card { background: white; padding: 20px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .status-healthy { color: #28a745; }
            .status-degraded { color: #ffc107; }
            .status-unhealthy { color: #dc3545; }
            .metric { display: inline-block; margin: 10px 20px 10px 0; }
            .metric-value { font-size: 24px; font-weight: bold; }
            .metric-label { font-size: 14px; color: #666; }
            .refresh-btn { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
            .refresh-btn:hover { background: #0056b3; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Fraud Detection API - Monitoring Dashboard</h1>
            
            <div class="card">
                <h2>System Status</h2>
                <div id="system-status">Loading...</div>
                <button class="refresh-btn" onclick="refreshData()">Refresh</button>
            </div>
            
            <div class="card">
                <h2>Health Checks</h2>
                <div id="health-checks">Loading...</div>
            </div>
            
            <div class="card">
                <h2>Active Alerts</h2>
                <div id="active-alerts">Loading...</div>
            </div>
            
            <div class="card">
                <h2>System Metrics</h2>
                <div id="system-metrics">Loading...</div>
            </div>
        </div>
        
        <script>
            async function fetchData(url) {
                try {
                    const response = await fetch(url);
                    return await response.json();
                } catch (error) {
                    console.error('Error fetching data:', error);
                    return null;
                }
            }
            
            async function refreshData() {
                // Fetch health data
                const healthData = await fetchData('/monitoring/health');
                if (healthData) {
                    updateSystemStatus(healthData);
                    updateHealthChecks(healthData.checks);
                }
                
                // Fetch alerts
                const alertsData = await fetchData('/monitoring/alerts');
                if (alertsData) {
                    updateActiveAlerts(alertsData.alerts);
                }
                
                // Fetch stats
                const statsData = await fetchData('/monitoring/stats');
                if (statsData) {
                    updateSystemMetrics(statsData);
                }
            }
            
            function updateSystemStatus(data) {
                const statusClass = `status-${data.status}`;
                document.getElementById('system-status').innerHTML = `
                    <div class="metric">
                        <div class="metric-value ${statusClass}">${data.status.toUpperCase()}</div>
                        <div class="metric-label">Overall Status</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${Math.round(data.uptime_seconds / 3600)}h</div>
                        <div class="metric-label">Uptime</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${data.environment}</div>
                        <div class="metric-label">Environment</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${data.version}</div>
                        <div class="metric-label">Version</div>
                    </div>
                `;
            }
            
            function updateHealthChecks(checks) {
                const checksHtml = checks.map(check => `
                    <div style="margin: 10px 0; padding: 10px; border-left: 4px solid ${getStatusColor(check.status)};">
                        <strong>${check.name}</strong>: ${check.message}
                        <br><small>Duration: ${check.duration_ms.toFixed(1)}ms</small>
                    </div>
                `).join('');
                document.getElementById('health-checks').innerHTML = checksHtml;
            }
            
            function updateActiveAlerts(alerts) {
                if (alerts.length === 0) {
                    document.getElementById('active-alerts').innerHTML = '<p>No active alerts</p>';
                    return;
                }
                
                const alertsHtml = alerts.map(alert => `
                    <div style="margin: 10px 0; padding: 10px; border-left: 4px solid ${getSeverityColor(alert.severity)};">
                        <strong>${alert.title}</strong> (${alert.severity})
                        <br>${alert.message}
                        <br><small>${new Date(alert.timestamp).toLocaleString()}</small>
                    </div>
                `).join('');
                document.getElementById('active-alerts').innerHTML = alertsHtml;
            }
            
            function updateSystemMetrics(data) {
                document.getElementById('system-metrics').innerHTML = `
                    <div class="metric">
                        <div class="metric-value">${data.health.checks_healthy}</div>
                        <div class="metric-label">Healthy Checks</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${data.alerts.active_count}</div>
                        <div class="metric-label">Active Alerts</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${data.alerts.recent_count}</div>
                        <div class="metric-label">Recent Alerts (24h)</div>
                    </div>
                `;
            }
            
            function getStatusColor(status) {
                switch(status) {
                    case 'healthy': return '#28a745';
                    case 'degraded': return '#ffc107';
                    case 'unhealthy': return '#dc3545';
                    default: return '#6c757d';
                }
            }
            
            function getSeverityColor(severity) {
                switch(severity) {
                    case 'low': return '#28a745';
                    case 'medium': return '#ffc107';
                    case 'high': return '#fd7e14';
                    case 'critical': return '#dc3545';
                    default: return '#6c757d';
                }
            }
            
            // Initial load and auto-refresh
            refreshData();
            setInterval(refreshData, 30000); // Refresh every 30 seconds
        </script>
    </body>
    </html>
    """
    
    return PlainTextResponse(html_content, media_type="text/html")
