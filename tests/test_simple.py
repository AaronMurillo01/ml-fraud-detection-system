#!/usr/bin/env python3
"""Simple test for monitoring imports."""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing basic imports...")

try:
    import psutil
    print("[OK] psutil available")
except ImportError:
    print("[ERROR] psutil not available - installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
    import psutil
    print("[OK] psutil installed and imported")

try:
    from config import settings
    print("[OK] Config imported")
except Exception as e:
    print(f"[ERROR] Config error: {e}")

try:
    from monitoring.health_checks import HealthChecker
    health_checker = HealthChecker()
    print(f"[OK] Health checker created with {len(health_checker.checks)} checks")
except Exception as e:
    print(f"[ERROR] Health checker error: {e}")
    import traceback
    traceback.print_exc()

print("Test completed!")
