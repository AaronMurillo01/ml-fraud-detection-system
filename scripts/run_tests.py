#!/usr/bin/env python3
"""Test runner script for fraud detection system."""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


class TestRunner:
    """Test runner for fraud detection system."""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.test_dir = self.project_root / "tests"
    
    def run_unit_tests(self, coverage: bool = True, verbose: bool = False) -> int:
        """Run unit tests."""
        print("🧪 Running unit tests...")
        
        cmd = ["python", "-m", "pytest", "tests/unit/"]
        
        if coverage:
            cmd.extend([
                "--cov=.",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov",
                "--cov-report=xml:coverage.xml"
            ])
        
        if verbose:
            cmd.append("-v")
        
        cmd.extend([
            "--junitxml=test-results-unit.xml",
            "-m", "unit"
        ])
        
        return self._run_command(cmd)
    
    def run_integration_tests(self, verbose: bool = False) -> int:
        """Run integration tests."""
        print("🔗 Running integration tests...")
        
        # Check if external services are available
        if not self._check_external_services():
            print("⚠️  External services not available, skipping integration tests")
            return 0
        
        cmd = [
            "python", "-m", "pytest", "tests/integration/",
            "--junitxml=test-results-integration.xml",
            "-m", "integration"
        ]
        
        if verbose:
            cmd.append("-v")
        
        return self._run_command(cmd)
    
    def run_performance_tests(self, verbose: bool = False) -> int:
        """Run performance tests."""
        print("⚡ Running performance tests...")
        
        cmd = [
            "python", "-m", "pytest", "tests/performance/",
            "--junitxml=test-results-performance.xml",
            "-m", "performance"
        ]
        
        if verbose:
            cmd.append("-v")
        
        return self._run_command(cmd)
    
    def run_all_tests(self, coverage: bool = True, verbose: bool = False) -> int:
        """Run all test suites."""
        print("🚀 Running all tests...")
        
        # Run unit tests first
        result = self.run_unit_tests(coverage=coverage, verbose=verbose)
        if result != 0:
            print("❌ Unit tests failed")
            return result
        
        # Run integration tests
        result = self.run_integration_tests(verbose=verbose)
        if result != 0:
            print("❌ Integration tests failed")
            return result
        
        # Run performance tests
        result = self.run_performance_tests(verbose=verbose)
        if result != 0:
            print("❌ Performance tests failed")
            return result
        
        print("✅ All tests passed!")
        return 0
    
    def run_smoke_tests(self, verbose: bool = False) -> int:
        """Run smoke tests for basic functionality."""
        print("💨 Running smoke tests...")
        
        cmd = [
            "python", "-m", "pytest",
            "--junitxml=test-results-smoke.xml",
            "-m", "smoke"
        ]
        
        if verbose:
            cmd.append("-v")
        
        return self._run_command(cmd)
    
    def run_security_tests(self, verbose: bool = False) -> int:
        """Run security tests."""
        print("🔒 Running security tests...")
        
        # Run bandit security scan
        print("Running bandit security scan...")
        bandit_result = self._run_command([
            "python", "-m", "bandit", "-r", ".", "-f", "json", "-o", "bandit-report.json"
        ])
        
        # Run safety check
        print("Running safety dependency check...")
        safety_result = self._run_command([
            "python", "-m", "safety", "check", "--json", "--output", "safety-report.json"
        ])
        
        # Run security-specific tests
        cmd = [
            "python", "-m", "pytest",
            "--junitxml=test-results-security.xml",
            "-m", "security"
        ]
        
        if verbose:
            cmd.append("-v")
        
        pytest_result = self._run_command(cmd)
        
        # Return non-zero if any security check failed
        return max(bandit_result, safety_result, pytest_result)
    
    def run_linting(self) -> int:
        """Run code linting and formatting checks."""
        print("🧹 Running linting checks...")
        
        results = []
        
        # Run flake8
        print("Running flake8...")
        results.append(self._run_command(["python", "-m", "flake8", "."]))
        
        # Run black check
        print("Running black check...")
        results.append(self._run_command(["python", "-m", "black", "--check", "."]))
        
        # Run isort check
        print("Running isort check...")
        results.append(self._run_command(["python", "-m", "isort", "--check-only", "."]))
        
        # Run mypy
        print("Running mypy...")
        results.append(self._run_command(["python", "-m", "mypy", "."]))
        
        return max(results) if results else 0
    
    def generate_coverage_report(self) -> int:
        """Generate coverage report."""
        print("📊 Generating coverage report...")
        
        # Generate HTML report
        html_result = self._run_command([
            "python", "-m", "coverage", "html", "--directory=htmlcov"
        ])
        
        # Generate XML report
        xml_result = self._run_command([
            "python", "-m", "coverage", "xml", "-o", "coverage.xml"
        ])
        
        # Print coverage report
        report_result = self._run_command([
            "python", "-m", "coverage", "report", "--show-missing"
        ])
        
        return max(html_result, xml_result, report_result)
    
    def run_benchmarks(self, verbose: bool = False) -> int:
        """Run benchmark tests."""
        print("📈 Running benchmarks...")
        
        cmd = [
            "python", "-m", "pytest", "tests/performance/",
            "--benchmark-only",
            "--benchmark-json=benchmark-results.json",
            "-m", "benchmark"
        ]
        
        if verbose:
            cmd.append("-v")
        
        return self._run_command(cmd)
    
    def clean_test_artifacts(self):
        """Clean up test artifacts."""
        print("🧽 Cleaning test artifacts...")
        
        artifacts = [
            "test-results*.xml",
            "coverage.xml",
            "htmlcov/",
            "bandit-report.json",
            "safety-report.json",
            "benchmark-results.json",
            ".coverage",
            "__pycache__/",
            "*.pyc",
            ".pytest_cache/",
            "test.db",
            "test_models/"
        ]
        
        for pattern in artifacts:
            self._run_command(["rm", "-rf", pattern], ignore_errors=True)
    
    def _run_command(self, cmd: List[str], ignore_errors: bool = False) -> int:
        """Run shell command."""
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=False,
                text=True
            )
            return result.returncode
        except subprocess.CalledProcessError as e:
            if not ignore_errors:
                print(f"Command failed: {' '.join(cmd)}")
                print(f"Error: {e}")
            return e.returncode
        except FileNotFoundError:
            if not ignore_errors:
                print(f"Command not found: {cmd[0]}")
            return 1
    
    def _check_external_services(self) -> bool:
        """Check if external services are available."""
        services = {
            "Redis": ("redis-cli", "ping"),
            "PostgreSQL": ("pg_isready", "-h", "localhost"),
        }
        
        available = True
        for service, cmd in services.items():
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode != 0:
                    print(f"⚠️  {service} not available")
                    available = False
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
                print(f"⚠️  {service} not available")
                available = False
        
        return available


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run fraud detection system tests")
    parser.add_argument(
        "test_type",
        choices=["unit", "integration", "performance", "all", "smoke", "security", "lint", "benchmark"],
        help="Type of tests to run"
    )
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Disable coverage reporting"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean test artifacts before running"
    )
    parser.add_argument(
        "--coverage-report",
        action="store_true",
        help="Generate coverage report after tests"
    )
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    if args.clean:
        runner.clean_test_artifacts()
    
    # Run tests based on type
    if args.test_type == "unit":
        result = runner.run_unit_tests(coverage=not args.no_coverage, verbose=args.verbose)
    elif args.test_type == "integration":
        result = runner.run_integration_tests(verbose=args.verbose)
    elif args.test_type == "performance":
        result = runner.run_performance_tests(verbose=args.verbose)
    elif args.test_type == "all":
        result = runner.run_all_tests(coverage=not args.no_coverage, verbose=args.verbose)
    elif args.test_type == "smoke":
        result = runner.run_smoke_tests(verbose=args.verbose)
    elif args.test_type == "security":
        result = runner.run_security_tests(verbose=args.verbose)
    elif args.test_type == "lint":
        result = runner.run_linting()
    elif args.test_type == "benchmark":
        result = runner.run_benchmarks(verbose=args.verbose)
    else:
        print(f"Unknown test type: {args.test_type}")
        return 1
    
    # Generate coverage report if requested
    if args.coverage_report and not args.no_coverage:
        coverage_result = runner.generate_coverage_report()
        result = max(result, coverage_result)
    
    return result


if __name__ == "__main__":
    sys.exit(main())