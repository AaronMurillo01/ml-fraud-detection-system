"""Alerting system for fraud detection API."""

import asyncio
import logging
import smtplib
import json
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import aiohttp
from jinja2 import Template

from config import settings
from .health_checks import HealthStatus, SystemHealth

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(str, Enum):
    """Types of alerts."""
    HEALTH_CHECK = "health_check"
    PERFORMANCE = "performance"
    SECURITY = "security"
    ERROR_RATE = "error_rate"
    RESOURCE_USAGE = "resource_usage"
    MODEL_PERFORMANCE = "model_performance"
    CUSTOM = "custom"


@dataclass
class Alert:
    """Alert information."""
    id: str
    type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    correlation_id: Optional[str] = None


@dataclass
class AlertRule:
    """Alert rule configuration."""
    name: str
    type: AlertType
    severity: AlertSeverity
    condition: Callable[[Dict[str, Any]], bool]
    message_template: str
    cooldown_minutes: int = 15
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class AlertManager:
    """Manager for alerts and notifications."""
    
    def __init__(self):
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.last_alert_times: Dict[str, datetime] = {}
        self.notification_channels: List[Callable[[Alert], None]] = []
        
        # Register default alert rules
        self.register_default_rules()
        
        # Setup notification channels
        self.setup_notification_channels()
    
    def register_default_rules(self):
        """Register default alert rules."""
        
        # Health check alerts
        self.register_rule(AlertRule(
            name="system_unhealthy",
            type=AlertType.HEALTH_CHECK,
            severity=AlertSeverity.CRITICAL,
            condition=lambda data: data.get("health_status") == HealthStatus.UNHEALTHY,
            message_template="System health is UNHEALTHY: {{ details }}",
            cooldown_minutes=5
        ))
        
        self.register_rule(AlertRule(
            name="system_degraded",
            type=AlertType.HEALTH_CHECK,
            severity=AlertSeverity.HIGH,
            condition=lambda data: data.get("health_status") == HealthStatus.DEGRADED,
            message_template="System health is DEGRADED: {{ details }}",
            cooldown_minutes=10
        ))
        
        # Performance alerts
        self.register_rule(AlertRule(
            name="high_response_time",
            type=AlertType.PERFORMANCE,
            severity=AlertSeverity.MEDIUM,
            condition=lambda data: data.get("avg_response_time", 0) > 2.0,
            message_template="High average response time: {{ avg_response_time }}s",
            cooldown_minutes=15
        ))
        
        self.register_rule(AlertRule(
            name="high_error_rate",
            type=AlertType.ERROR_RATE,
            severity=AlertSeverity.HIGH,
            condition=lambda data: data.get("error_rate", 0) > 0.05,  # 5% error rate
            message_template="High error rate detected: {{ error_rate }}%",
            cooldown_minutes=10
        ))
        
        # Resource usage alerts
        self.register_rule(AlertRule(
            name="high_cpu_usage",
            type=AlertType.RESOURCE_USAGE,
            severity=AlertSeverity.HIGH,
            condition=lambda data: data.get("cpu_percent", 0) > 90,
            message_template="High CPU usage: {{ cpu_percent }}%",
            cooldown_minutes=15
        ))
        
        self.register_rule(AlertRule(
            name="high_memory_usage",
            type=AlertType.RESOURCE_USAGE,
            severity=AlertSeverity.HIGH,
            condition=lambda data: data.get("memory_percent", 0) > 90,
            message_template="High memory usage: {{ memory_percent }}%",
            cooldown_minutes=15
        ))
        
        self.register_rule(AlertRule(
            name="low_disk_space",
            type=AlertType.RESOURCE_USAGE,
            severity=AlertSeverity.CRITICAL,
            condition=lambda data: data.get("disk_usage_percent", 0) > 95,
            message_template="Critical disk space: {{ disk_usage_percent }}% used",
            cooldown_minutes=30
        ))
        
        # Model performance alerts
        self.register_rule(AlertRule(
            name="model_accuracy_drop",
            type=AlertType.MODEL_PERFORMANCE,
            severity=AlertSeverity.HIGH,
            condition=lambda data: data.get("model_accuracy", 1.0) < 0.8,
            message_template="Model accuracy dropped: {{ model_accuracy }}",
            cooldown_minutes=60
        ))
        
        # Security alerts
        self.register_rule(AlertRule(
            name="authentication_failures",
            type=AlertType.SECURITY,
            severity=AlertSeverity.MEDIUM,
            condition=lambda data: data.get("auth_failures_per_minute", 0) > 10,
            message_template="High authentication failure rate: {{ auth_failures_per_minute }}/min",
            cooldown_minutes=10
        ))
    
    def setup_notification_channels(self):
        """Setup notification channels based on configuration."""
        
        # Email notifications
        if hasattr(settings, 'alert_email_recipients') and settings.alert_email_recipients:
            self.notification_channels.append(self.send_email_notification)
        
        # Webhook notifications
        if hasattr(settings, 'alert_webhook_url') and settings.alert_webhook_url:
            self.notification_channels.append(self.send_webhook_notification)
        
        # Slack notifications (if configured)
        if hasattr(settings, 'slack_webhook_url') and getattr(settings, 'slack_webhook_url', None):
            self.notification_channels.append(self.send_slack_notification)
    
    def register_rule(self, rule: AlertRule):
        """Register an alert rule.
        
        Args:
            rule: Alert rule to register
        """
        self.rules[rule.name] = rule
        logger.info(f"Registered alert rule: {rule.name}")
    
    def remove_rule(self, rule_name: str):
        """Remove an alert rule.
        
        Args:
            rule_name: Name of rule to remove
        """
        if rule_name in self.rules:
            del self.rules[rule_name]
            logger.info(f"Removed alert rule: {rule_name}")
    
    async def evaluate_rules(self, data: Dict[str, Any]):
        """Evaluate all alert rules against provided data.
        
        Args:
            data: Data to evaluate rules against
        """
        for rule_name, rule in self.rules.items():
            if not rule.enabled:
                continue
            
            try:
                # Check cooldown period
                if self._is_in_cooldown(rule_name, rule.cooldown_minutes):
                    continue
                
                # Evaluate condition
                if rule.condition(data):
                    await self._trigger_alert(rule, data)
                    
            except Exception as e:
                logger.error(f"Error evaluating alert rule {rule_name}: {e}")
    
    def _is_in_cooldown(self, rule_name: str, cooldown_minutes: int) -> bool:
        """Check if rule is in cooldown period.
        
        Args:
            rule_name: Name of the rule
            cooldown_minutes: Cooldown period in minutes
            
        Returns:
            True if in cooldown, False otherwise
        """
        if rule_name not in self.last_alert_times:
            return False
        
        last_alert_time = self.last_alert_times[rule_name]
        cooldown_period = timedelta(minutes=cooldown_minutes)
        
        return datetime.utcnow() - last_alert_time < cooldown_period
    
    async def _trigger_alert(self, rule: AlertRule, data: Dict[str, Any]):
        """Trigger an alert.
        
        Args:
            rule: Alert rule that was triggered
            data: Data that triggered the alert
        """
        # Generate alert ID
        alert_id = f"{rule.name}_{int(datetime.utcnow().timestamp())}"
        
        # Render message template
        template = Template(rule.message_template)
        message = template.render(**data)
        
        # Create alert
        alert = Alert(
            id=alert_id,
            type=rule.type,
            severity=rule.severity,
            title=f"Alert: {rule.name}",
            message=message,
            metadata=data.copy()
        )
        
        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        self.last_alert_times[rule.name] = datetime.utcnow()
        
        # Send notifications
        await self._send_notifications(alert)
        
        logger.warning(f"Alert triggered: {rule.name} - {message}")
    
    async def _send_notifications(self, alert: Alert):
        """Send alert notifications through all configured channels.
        
        Args:
            alert: Alert to send notifications for
        """
        for channel in self.notification_channels:
            try:
                if asyncio.iscoroutinefunction(channel):
                    await channel(alert)
                else:
                    channel(alert)
            except Exception as e:
                logger.error(f"Failed to send notification through channel: {e}")
    
    async def send_email_notification(self, alert: Alert):
        """Send email notification.
        
        Args:
            alert: Alert to send email for
        """
        try:
            if not hasattr(settings, 'alert_email_recipients'):
                return
            
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = getattr(settings, 'smtp_from_email', 'alerts@fraud-detection.com')
            msg['To'] = ', '.join(settings.alert_email_recipients)
            msg['Subject'] = f"[{alert.severity.upper()}] {alert.title}"
            
            # Email body
            body = f"""
Alert Details:
- Type: {alert.type.value}
- Severity: {alert.severity.value}
- Time: {alert.timestamp.isoformat()}
- Message: {alert.message}

Metadata:
{json.dumps(alert.metadata, indent=2)}

Environment: {settings.environment.value}
Service: Fraud Detection API
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email (would need SMTP configuration)
            logger.info(f"Email notification prepared for alert: {alert.id}")
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
    
    async def send_webhook_notification(self, alert: Alert):
        """Send webhook notification.
        
        Args:
            alert: Alert to send webhook for
        """
        try:
            if not hasattr(settings, 'alert_webhook_url'):
                return
            
            payload = {
                "alert_id": alert.id,
                "type": alert.type.value,
                "severity": alert.severity.value,
                "title": alert.title,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "environment": settings.environment.value,
                "service": "fraud-detection-api",
                "metadata": alert.metadata
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    settings.alert_webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        logger.info(f"Webhook notification sent for alert: {alert.id}")
                    else:
                        logger.error(f"Webhook notification failed: {response.status}")
                        
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
    
    async def send_slack_notification(self, alert: Alert):
        """Send Slack notification.
        
        Args:
            alert: Alert to send Slack message for
        """
        try:
            slack_webhook_url = getattr(settings, 'slack_webhook_url', None)
            if not slack_webhook_url:
                return
            
            # Determine color based on severity
            color_map = {
                AlertSeverity.LOW: "#36a64f",      # Green
                AlertSeverity.MEDIUM: "#ff9500",   # Orange
                AlertSeverity.HIGH: "#ff0000",     # Red
                AlertSeverity.CRITICAL: "#8b0000"  # Dark Red
            }
            
            payload = {
                "attachments": [{
                    "color": color_map.get(alert.severity, "#ff0000"),
                    "title": alert.title,
                    "text": alert.message,
                    "fields": [
                        {
                            "title": "Severity",
                            "value": alert.severity.value.upper(),
                            "short": True
                        },
                        {
                            "title": "Type",
                            "value": alert.type.value,
                            "short": True
                        },
                        {
                            "title": "Environment",
                            "value": settings.environment.value,
                            "short": True
                        },
                        {
                            "title": "Time",
                            "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
                            "short": True
                        }
                    ],
                    "footer": "Fraud Detection API",
                    "ts": int(alert.timestamp.timestamp())
                }]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    slack_webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        logger.info(f"Slack notification sent for alert: {alert.id}")
                    else:
                        logger.error(f"Slack notification failed: {response.status}")
                        
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
    
    def resolve_alert(self, alert_id: str):
        """Resolve an active alert.
        
        Args:
            alert_id: ID of alert to resolve
        """
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.utcnow()
            del self.active_alerts[alert_id]
            logger.info(f"Alert resolved: {alert_id}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get list of active alerts.
        
        Returns:
            List of active alerts
        """
        return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history.
        
        Args:
            limit: Maximum number of alerts to return
            
        Returns:
            List of historical alerts
        """
        return self.alert_history[-limit:]
    
    async def check_system_health_alerts(self, system_health: SystemHealth):
        """Check for system health-related alerts.
        
        Args:
            system_health: System health status
        """
        data = {
            "health_status": system_health.status,
            "details": f"{len([c for c in system_health.checks if c.status == HealthStatus.UNHEALTHY])} unhealthy checks"
        }
        
        await self.evaluate_rules(data)


# Global alert manager instance
_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Get global alert manager instance.
    
    Returns:
        Alert manager instance
    """
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager


async def trigger_custom_alert(
    name: str,
    severity: AlertSeverity,
    message: str,
    alert_type: AlertType = AlertType.CUSTOM,
    metadata: Optional[Dict[str, Any]] = None
):
    """Trigger a custom alert.
    
    Args:
        name: Alert name
        severity: Alert severity
        message: Alert message
        alert_type: Type of alert
        metadata: Additional metadata
    """
    alert_manager = get_alert_manager()
    
    # Create temporary rule for custom alert
    rule = AlertRule(
        name=name,
        type=alert_type,
        severity=severity,
        condition=lambda data: True,  # Always trigger
        message_template=message,
        cooldown_minutes=0  # No cooldown for custom alerts
    )
    
    await alert_manager._trigger_alert(rule, metadata or {})
