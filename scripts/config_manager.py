#!/usr/bin/env python3
"""Configuration management CLI tool."""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import (
    Environment,
    ConfigFactory,
    get_settings_for_environment,
    validate_environment_file,
    get_config_summary
)
from config.validation import ConfigValidator, generate_config_report
from config.secrets import SecretsManager, generate_secret_key, generate_master_key


def validate_config_command(args):
    """Validate configuration for specified environment."""
    print(f"Validating {args.environment} configuration...")
    print("=" * 50)
    
    try:
        # Get configuration for environment
        env = Environment(args.environment)
        config = get_settings_for_environment(env)
        
        # Generate comprehensive report
        report = generate_config_report(config)
        
        # Print validation results
        print(f"Environment: {report['environment']}")
        print(f"Configuration Valid: {'✓' if report['validation']['config_valid'] else '✗'}")
        print(f"Environment Variables Valid: {'✓' if report['validation']['env_valid'] else '✗'}")
        
        if report['validation']['config_errors']:
            print("\nConfiguration Errors:")
            for error in report['validation']['config_errors']:
                print(f"  ✗ {error}")
        
        if report['validation']['env_errors']:
            print("\nEnvironment Variable Errors:")
            for error in report['validation']['env_errors']:
                print(f"  ✗ {error}")
        
        if report['validation']['env_warnings']:
            print("\nWarnings:")
            for warning in report['validation']['env_warnings']:
                print(f"  ⚠ {warning}")
        
        # Print security summary
        print(f"\nSecurity Settings:")
        security = report['security']
        print(f"  Debug Enabled: {'✗' if security['debug_enabled'] else '✓'}")
        print(f"  Docs Enabled: {'⚠' if security['docs_enabled'] else '✓'}")
        print(f"  Authentication Required: {'✓' if security['authentication_required'] else '✗'}")
        print(f"  Rate Limiting Enabled: {'✓' if security['rate_limiting_enabled'] else '✗'}")
        print(f"  SSL Enabled: {'✓' if security['ssl_enabled'] else '⚠'}")
        
        # Print performance summary
        print(f"\nPerformance Settings:")
        perf = report['performance']
        print(f"  API Workers: {perf['api_workers']}")
        print(f"  Database Pool Size: {perf['database_pool_size']}")
        print(f"  Redis Max Connections: {perf['redis_max_connections']}")
        print(f"  Model Cache Size: {perf['model_cache_size']}")
        
        return report['validation']['config_valid'] and report['validation']['env_valid']
        
    except Exception as e:
        print(f"Validation failed: {e}")
        return False


def show_config_command(args):
    """Show configuration for specified environment."""
    try:
        env = Environment(args.environment)
        config = get_settings_for_environment(env)
        summary = get_config_summary(config)
        
        print(f"Configuration for {args.environment}:")
        print("=" * 50)
        print(json.dumps(summary, indent=2))
        
    except Exception as e:
        print(f"Failed to show configuration: {e}")


def generate_secrets_command(args):
    """Generate secure secrets for production."""
    print("Generating secure secrets...")
    print("=" * 30)
    
    secrets = {
        "SECRET_KEY": generate_secret_key(64),
        "JWT_SIGNING_KEY": generate_secret_key(64),
        "API_KEY_ENCRYPTION_KEY": generate_secret_key(32),
        "MASTER_KEY": generate_master_key()
    }
    
    if args.output:
        # Write to file
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            for key, value in secrets.items():
                f.write(f"{key}={value}\n")
        print(f"Secrets written to {output_path}")
        print("⚠ Keep this file secure and do not commit to version control!")
    else:
        # Print to stdout
        for key, value in secrets.items():
            print(f"{key}={value}")
        print("\n⚠ Copy these secrets to your production environment file")


def check_env_file_command(args):
    """Check environment file exists and is readable."""
    env = Environment(args.environment)
    result = validate_environment_file(env)
    
    print(f"Environment file check for {args.environment}:")
    print("=" * 40)
    print(f"Environment: {result['environment']}")
    print(f"Expected file: {result['env_file']}")
    print(f"File exists: {'✓' if result['exists'] else '✗'}")
    print(f"File readable: {'✓' if result['readable'] else '✗'}")
    
    if result['errors']:
        print("\nErrors:")
        for error in result['errors']:
            print(f"  ✗ {error}")
    
    return result['exists'] and result['readable']


def create_env_template_command(args):
    """Create environment template file."""
    env = Environment(args.environment)
    
    # Template content based on environment
    if env == Environment.DEVELOPMENT:
        template_file = ".env.development.template"
    elif env == Environment.STAGING:
        template_file = ".env.staging.template"
    elif env == Environment.PRODUCTION:
        template_file = ".env.production.template"
    else:
        template_file = ".env.testing.template"
    
    source_path = project_root / template_file
    target_path = project_root / f".env.{args.environment}"
    
    if source_path.exists():
        if target_path.exists() and not args.force:
            print(f"Environment file {target_path} already exists. Use --force to overwrite.")
            return False
        
        # Copy template to target
        with open(source_path, 'r') as src, open(target_path, 'w') as dst:
            content = src.read()
            dst.write(content)
        
        print(f"Created environment file: {target_path}")
        print(f"Please edit {target_path} and update the values for your environment.")
        return True
    else:
        print(f"Template file {template_file} not found.")
        return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Configuration management tool")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate configuration')
    validate_parser.add_argument('environment', choices=['development', 'staging', 'production', 'testing'],
                               help='Environment to validate')
    
    # Show command
    show_parser = subparsers.add_parser('show', help='Show configuration')
    show_parser.add_argument('environment', choices=['development', 'staging', 'production', 'testing'],
                           help='Environment to show')
    
    # Generate secrets command
    secrets_parser = subparsers.add_parser('generate-secrets', help='Generate secure secrets')
    secrets_parser.add_argument('--output', '-o', help='Output file for secrets')
    
    # Check env file command
    check_parser = subparsers.add_parser('check-env', help='Check environment file')
    check_parser.add_argument('environment', choices=['development', 'staging', 'production', 'testing'],
                            help='Environment to check')
    
    # Create env template command
    template_parser = subparsers.add_parser('create-env', help='Create environment file from template')
    template_parser.add_argument('environment', choices=['development', 'staging', 'production', 'testing'],
                               help='Environment to create')
    template_parser.add_argument('--force', action='store_true', help='Overwrite existing file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'validate':
            success = validate_config_command(args)
            return 0 if success else 1
        elif args.command == 'show':
            show_config_command(args)
            return 0
        elif args.command == 'generate-secrets':
            generate_secrets_command(args)
            return 0
        elif args.command == 'check-env':
            success = check_env_file_command(args)
            return 0 if success else 1
        elif args.command == 'create-env':
            success = create_env_template_command(args)
            return 0 if success else 1
        else:
            print(f"Unknown command: {args.command}")
            return 1
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
