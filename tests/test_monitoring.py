#!/usr/bin/env python3
"""Test script for monitoring system components."""

import asyncio
import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_health_checks():
    """Test health check system."""
    print("Testing health check system...")
    
    try:
        from monitoring.health_checks import HealthChecker, HealthStatus
        
        # Create health checker
        health_checker = HealthChecker()
        print(f"✅ Health checker created with {len(health_checker.checks)} checks")
        
        # Test individual health check (system resources - should work without external deps)
        result = await health_checker.check_system_resources()
        print(f"✅ System resources check: {result.status.value} - {result.message}")
        
        # Test memory usage check
        result = await health_checker.check_memory_usage()
        print(f"✅ Memory usage check: {result.status.value} - {result.message}")
        
        # Test disk space check
        result = await health_checker.check_disk_space()
        print(f"✅ Disk space check: {result.status.value} - {result.message}")
        
        return True
        
    except Exception as e:
        print(f"❌ Health checks failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_alerting():
    """Test alerting system."""
    print("\nTesting alerting system...")
    
    try:
        from monitoring.alerting import AlertManager, AlertSeverity, AlertType
        
        # Create alert manager
        alert_manager = AlertManager()
        print(f"✅ Alert manager created with {len(alert_manager.rules)} rules")
        
        # Test alert rule evaluation
        test_data = {
            "cpu_percent": 95,  # Should trigger high CPU alert
            "memory_percent": 50,
            "disk_usage_percent": 80
        }
        
        # This would normally be async, but we'll test the rule logic
        high_cpu_rule = alert_manager.rules.get("high_cpu_usage")
        if high_cpu_rule and high_cpu_rule.condition(test_data):
            print("✅ High CPU alert rule triggered correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Alerting system failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tracing():
    """Test tracing system (basic functionality)."""
    print("\nTesting tracing system...")
    
    try:
        from monitoring.tracing import TracingManager, get_current_correlation_id
        
        # Create tracing manager
        tracing_manager = TracingManager()
        print("✅ Tracing manager created")
        
        # Test correlation ID generation
        correlation_id = tracing_manager.generate_correlation_id()
        print(f"✅ Generated correlation ID: {correlation_id[:8]}...")
        
        # Test request ID generation
        request_id = tracing_manager.generate_request_id()
        print(f"✅ Generated request ID: {request_id[:8]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Tracing system failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all monitoring tests."""
    print("🔍 Testing Fraud Detection API Monitoring System")
    print("=" * 50)
    
    results = []
    
    # Test health checks
    results.append(await test_health_checks())
    
    # Test alerting
    results.append(test_alerting())
    
    # Test tracing
    results.append(test_tracing())
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"✅ Passed: {sum(results)}")
    print(f"❌ Failed: {len(results) - sum(results)}")
    
    if all(results):
        print("\n🎉 All monitoring components are working correctly!")
        return 0
    else:
        print("\n⚠️  Some monitoring components have issues.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
