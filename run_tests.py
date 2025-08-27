#!/usr/bin/env python3
"""
Test runner for RPCA Multiverse implementation.

Usage:
    python run_tests.py                    # Run all fast tests
    python run_tests.py --all              # Run all tests including slow ones
    python run_tests.py --unit             # Run only unit tests
    python run_tests.py --integration      # Run only integration tests
    python run_tests.py --benchmark        # Run performance benchmarks
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list, description: str) -> bool:
    """Run command and return success status."""
    print(f"\n{description}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED (exit code: {e.returncode})")
        return False
    except FileNotFoundError:
        print(f"❌ {description} - COMMAND NOT FOUND")
        return False


def main():
    parser = argparse.ArgumentParser(description='RPCA Multiverse Test Runner')
    
    # Test selection options
    parser.add_argument('--all', action='store_true',
                       help='Run all tests including slow integration tests')
    parser.add_argument('--unit', action='store_true', 
                       help='Run only unit tests')
    parser.add_argument('--integration', action='store_true',
                       help='Run only integration tests')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmarks')
    
    # Output options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose test output')
    parser.add_argument('--coverage', action='store_true',
                       help='Run with coverage reporting')
    parser.add_argument('--html-report', action='store_true',
                       help='Generate HTML coverage report')
    
    # Test filtering
    parser.add_argument('--pattern', '-k', type=str,
                       help='Run tests matching pattern')
    parser.add_argument('--file', type=str,
                       help='Run specific test file')
    
    args = parser.parse_args()
    
    # Check if pytest is available
    try:
        subprocess.run(['pytest', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ pytest is not installed. Please install it with: pip install pytest")
        sys.exit(1)
        
    success_count = 0
    total_tests = 0
    
    # Build pytest command
    cmd = ['python', '-m', 'pytest']
    
    if args.verbose:
        cmd.append('-v')
        
    if args.coverage:
        cmd.extend(['--cov=src', '--cov-report=term-missing'])
        if args.html_report:
            cmd.append('--cov-report=html')
            
    # Test selection
    if args.unit:
        cmd.extend(['-m', 'unit'])
        description = "Unit Tests"
    elif args.integration:
        cmd.extend(['-m', 'integration']) 
        description = "Integration Tests"
    elif args.all:
        description = "All Tests (including slow)"
    else:
        cmd.extend(['-m', 'not slow'])
        description = "Fast Tests"
        
    # Pattern filtering
    if args.pattern:
        cmd.extend(['-k', args.pattern])
        
    # Specific file
    if args.file:
        test_file = Path(args.file)
        if not test_file.exists():
            test_file = Path('tests') / args.file
        if not test_file.exists():
            print(f"❌ Test file not found: {args.file}")
            sys.exit(1)
        cmd.append(str(test_file))
    else:
        cmd.append('tests/')
        
    # Special handling for benchmarks
    if args.benchmark:
        benchmark_cmd = ['python', 'tests/test_rpca_integration.py', '--benchmark']
        success = run_command(benchmark_cmd, "Performance Benchmarks")
        if success:
            success_count += 1
        total_tests += 1
    else:
        # Run main test suite
        success = run_command(cmd, description)
        if success:
            success_count += 1
        total_tests += 1
        
    # Additional checks
    if not args.benchmark:
        # Run configuration validation
        config_cmd = ['python', 'validate_rpca_config.py', 
                     '--config', 'config/env/racing.yaml']
        success = run_command(config_cmd, "Configuration Validation")
        if success:
            success_count += 1
        total_tests += 1
        
        # Check code style (if available)
        try:
            subprocess.run(['flake8', '--version'], capture_output=True, check=True)
            style_cmd = ['flake8', 'src/', '--max-line-length=100', '--ignore=E203,W503']
            success = run_command(style_cmd, "Code Style Check")
            if success:
                success_count += 1
            total_tests += 1
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("ℹ️  flake8 not available - skipping code style check")
            
    # Final summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Passed: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("🎉 All tests passed!")
        exit_code = 0
    else:
        print("💥 Some tests failed!")
        exit_code = 1
        
    if args.coverage and args.html_report:
        print(f"\n📊 Coverage report generated in htmlcov/index.html")
        
    sys.exit(exit_code)


if __name__ == "__main__":
    main()