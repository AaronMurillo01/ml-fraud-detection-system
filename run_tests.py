#!/usr/bin/env python3
"""Test runner script for fraud detection system.

This script provides a convenient way to run different types of tests
with various configurations and reporting options.
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
from typing import List, Optional


class TestRunner:
    """Test runner for fraud detection system."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.test_dir = self.project_root / "tests"
        
    def run_command(self, cmd: List[str], capture_output: bool = False) -> subprocess.CompletedProcess:
        """Run a command and return the result."""
        print(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=capture_output,
                text=True,
                check=False
            )
            return result
        except FileNotFoundError as e:
            print(f"Error: Command not found - {e}")
            sys.exit(1)
    
    def run_unit_tests(self, verbose: bool = False, coverage: bool = True) -> int:
        """Run unit tests."""
        print("\n=== Running Unit Tests ===")
        
        cmd = ["python", "-m", "pytest", "-m", "unit"]
        
        if verbose:
            cmd.append("-v")
        
        if coverage:
            cmd.extend(["--cov=service", "--cov=shared", "--cov-report=term-missing"])
        
        cmd.append("tests/unit/")
        
        result = self.run_command(cmd)
        return result.returncode
    
    def run_integration_tests(self, verbose: bool = False) -> int:
        """Run integration tests."""
        print("\n=== Running Integration Tests ===")
        
        cmd = ["python", "-m", "pytest", "-m", "integration"]
        
        if verbose:
            cmd.append("-v")
        
        cmd.append("tests/integration/")
        
        result = self.run_command(cmd)
        return result.returncode
    
    def run_performance_tests(self, verbose: bool = False) -> int:
        """Run performance tests."""
        print("\n=== Running Performance Tests ===")
        
        cmd = ["python", "-m", "pytest", "-m", "performance"]
        
        if verbose:
            cmd.append("-v")
        
        cmd.append("tests/performance/")
        
        result = self.run_command(cmd)
        return result.returncode
    
    def run_smoke_tests(self, verbose: bool = False) -> int:
        """Run smoke tests for basic functionality."""
        print("\n=== Running Smoke Tests ===")
        
        cmd = ["python", "-m", "pytest", "-m", "smoke"]
        
        if verbose:
            cmd.append("-v")
        
        result = self.run_command(cmd)
        return result.returncode
    
    def run_all_tests(self, verbose: bool = False, coverage: bool = True, 
                     skip_slow: bool = False, parallel: bool = False) -> int:
        """Run all tests."""
        print("\n=== Running All Tests ===")
        
        cmd = ["python", "-m", "pytest"]
        
        if verbose:
            cmd.append("-v")
        
        if coverage:
            cmd.extend([
                "--cov=service",
                "--cov=shared", 
                "--cov=streaming",
                "--cov=features",
                "--cov=training",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov"
            ])
        
        if skip_slow:
            cmd.extend(["-m", "not slow"])
        
        if parallel:
            cmd.extend(["-n", "auto"])
        
        cmd.append("tests/")
        
        result = self.run_command(cmd)
        return result.returncode
    
    def run_specific_test(self, test_path: str, verbose: bool = False) -> int:
        """Run a specific test file or test function."""
        print(f"\n=== Running Specific Test: {test_path} ===")
        
        cmd = ["python", "-m", "pytest"]
        
        if verbose:
            cmd.append("-v")
        
        cmd.append(test_path)
        
        result = self.run_command(cmd)
        return result.returncode
    
    def run_with_markers(self, markers: List[str], verbose: bool = False) -> int:
        """Run tests with specific markers."""
        marker_expr = " and ".join(markers)
        print(f"\n=== Running Tests with Markers: {marker_expr} ===")
        
        cmd = ["python", "-m", "pytest", "-m", marker_expr]
        
        if verbose:
            cmd.append("-v")
        
        result = self.run_command(cmd)
        return result.returncode
    
    def generate_coverage_report(self) -> int:
        """Generate detailed coverage report."""
        print("\n=== Generating Coverage Report ===")
        
        # Run tests with coverage
        cmd = [
            "python", "-m", "pytest",
            "--cov=service",
            "--cov=shared",
            "--cov=streaming", 
            "--cov=features",
            "--cov=training",
            "--cov-report=html:htmlcov",
            "--cov-report=xml:coverage.xml",
            "--cov-report=term-missing",
            "tests/"
        ]
        
        result = self.run_command(cmd)
        
        if result.returncode == 0:
            print("\nCoverage reports generated:")
            print("  - HTML: htmlcov/index.html")
            print("  - XML: coverage.xml")
        
        return result.returncode
    
    def run_linting(self) -> int:
        """Run code linting and formatting checks."""
        print("\n=== Running Code Quality Checks ===")
        
        exit_code = 0
        
        # Run black formatting check
        print("\nChecking code formatting with black...")
        result = self.run_command(["python", "-m", "black", "--check", "--diff", "."])
        if result.returncode != 0:
            print("❌ Code formatting issues found. Run 'black .' to fix.")
            exit_code = 1
        else:
            print("✅ Code formatting is correct.")
        
        # Run isort import sorting check
        print("\nChecking import sorting with isort...")
        result = self.run_command(["python", "-m", "isort", "--check-only", "--diff", "."])
        if result.returncode != 0:
            print("❌ Import sorting issues found. Run 'isort .' to fix.")
            exit_code = 1
        else:
            print("✅ Import sorting is correct.")
        
        # Run flake8 linting
        print("\nRunning flake8 linting...")
        result = self.run_command(["python", "-m", "flake8", "."])
        if result.returncode != 0:
            print("❌ Linting issues found.")
            exit_code = 1
        else:
            print("✅ No linting issues found.")
        
        # Run mypy type checking
        print("\nRunning mypy type checking...")
        result = self.run_command(["python", "-m", "mypy", "service", "shared", "streaming"])
        if result.returncode != 0:
            print("❌ Type checking issues found.")
            exit_code = 1
        else:
            print("✅ No type checking issues found.")
        
        return exit_code
    
    def run_security_checks(self) -> int:
        """Run security vulnerability checks."""
        print("\n=== Running Security Checks ===")
        
        exit_code = 0
        
        # Run bandit security linting
        print("\nRunning bandit security checks...")
        result = self.run_command(["python", "-m", "bandit", "-r", "service", "shared", "streaming"])
        if result.returncode != 0:
            print("❌ Security issues found.")
            exit_code = 1
        else:
            print("✅ No security issues found.")
        
        # Run safety dependency vulnerability check
        print("\nRunning safety dependency checks...")
        result = self.run_command(["python", "-m", "safety", "check"])
        if result.returncode != 0:
            print("❌ Vulnerable dependencies found.")
            exit_code = 1
        else:
            print("✅ No vulnerable dependencies found.")
        
        return exit_code
    
    def setup_test_environment(self) -> int:
        """Setup test environment and dependencies."""
        print("\n=== Setting Up Test Environment ===")
        
        # Install test dependencies
        print("Installing test dependencies...")
        result = self.run_command(["pip", "install", "-e", ".[test]"])
        
        if result.returncode != 0:
            print("❌ Failed to install test dependencies.")
            return 1
        
        print("✅ Test environment setup complete.")
        return 0
    
    def clean_test_artifacts(self) -> int:
        """Clean test artifacts and cache files."""
        print("\n=== Cleaning Test Artifacts ===")
        
        artifacts = [
            "htmlcov",
            "coverage.xml", 
            ".coverage",
            ".pytest_cache",
            "__pycache__",
            "*.pyc",
            "*.pyo",
            "*.pyd",
            ".mypy_cache"
        ]
        
        for artifact in artifacts:
            if os.path.exists(artifact):
                if os.path.isdir(artifact):
                    result = self.run_command(["rm", "-rf", artifact])
                else:
                    result = self.run_command(["rm", "-f", artifact])
        
        print("✅ Test artifacts cleaned.")
        return 0


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(
        description="Test runner for fraud detection system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py --unit                    # Run unit tests
  python run_tests.py --integration             # Run integration tests
  python run_tests.py --performance             # Run performance tests
  python run_tests.py --all                     # Run all tests
  python run_tests.py --all --parallel          # Run all tests in parallel
  python run_tests.py --markers api ml          # Run tests with api AND ml markers
  python run_tests.py --test tests/unit/test_models.py  # Run specific test
  python run_tests.py --coverage                # Generate coverage report
  python run_tests.py --lint                    # Run code quality checks
  python run_tests.py --security                # Run security checks
  python run_tests.py --setup                   # Setup test environment
  python run_tests.py --clean                   # Clean test artifacts
        """
    )
    
    # Test type options
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument("--unit", action="store_true", help="Run unit tests")
    test_group.add_argument("--integration", action="store_true", help="Run integration tests")
    test_group.add_argument("--performance", action="store_true", help="Run performance tests")
    test_group.add_argument("--smoke", action="store_true", help="Run smoke tests")
    test_group.add_argument("--all", action="store_true", help="Run all tests")
    test_group.add_argument("--test", type=str, help="Run specific test file or function")
    test_group.add_argument("--markers", nargs="+", help="Run tests with specific markers")
    
    # Utility options
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--lint", action="store_true", help="Run code quality checks")
    parser.add_argument("--security", action="store_true", help="Run security checks")
    parser.add_argument("--setup", action="store_true", help="Setup test environment")
    parser.add_argument("--clean", action="store_true", help="Clean test artifacts")
    
    # Test execution options
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--no-coverage", action="store_true", help="Disable coverage reporting")
    parser.add_argument("--skip-slow", action="store_true", help="Skip slow tests")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    
    args = parser.parse_args()
    
    runner = TestRunner()
    exit_code = 0
    
    try:
        if args.setup:
            exit_code = runner.setup_test_environment()
        elif args.clean:
            exit_code = runner.clean_test_artifacts()
        elif args.lint:
            exit_code = runner.run_linting()
        elif args.security:
            exit_code = runner.run_security_checks()
        elif args.coverage:
            exit_code = runner.generate_coverage_report()
        elif args.unit:
            exit_code = runner.run_unit_tests(
                verbose=args.verbose,
                coverage=not args.no_coverage
            )
        elif args.integration:
            exit_code = runner.run_integration_tests(verbose=args.verbose)
        elif args.performance:
            exit_code = runner.run_performance_tests(verbose=args.verbose)
        elif args.smoke:
            exit_code = runner.run_smoke_tests(verbose=args.verbose)
        elif args.all:
            exit_code = runner.run_all_tests(
                verbose=args.verbose,
                coverage=not args.no_coverage,
                skip_slow=args.skip_slow,
                parallel=args.parallel
            )
        elif args.test:
            exit_code = runner.run_specific_test(
                test_path=args.test,
                verbose=args.verbose
            )
        elif args.markers:
            exit_code = runner.run_with_markers(
                markers=args.markers,
                verbose=args.verbose
            )
        else:
            # Default: run smoke tests
            print("No test type specified. Running smoke tests...")
            exit_code = runner.run_smoke_tests(verbose=args.verbose)
    
    except KeyboardInterrupt:
        print("\n\nTest execution interrupted by user.")
        exit_code = 130
    except Exception as e:
        print(f"\n\nError running tests: {e}")
        exit_code = 1
    
    if exit_code == 0:
        print("\n✅ All tests completed successfully!")
    else:
        print(f"\n❌ Tests failed with exit code {exit_code}")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()