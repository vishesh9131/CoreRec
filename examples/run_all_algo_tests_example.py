#!/usr/bin/env python3
"""
CoreRec Algorithm Test Runner - Enhanced Script

This script runs all algorithm tests and provides detailed reporting.
It's an enhanced wrapper around corerec.run_algo_tests with additional features.

Usage:
    python examples/run_all_algo_tests_example.py
    python examples/run_all_algo_tests_example.py --json report.json
    python examples/run_all_algo_tests_example.py --pattern "*import*" --quiet
    python examples/run_all_algo_tests_example.py --verbose

Options:
    --json <file>      Save detailed JSON report to file
    --pattern <glob>   Test file pattern (default: "*_test*.py")
    --quiet            Minimal output (only summary)
    --verbose          Show detailed test output
    --paths <dirs>     Additional directories to search (comma-separated)
    --exit-zero        Always exit with code 0 (useful for CI/CD)
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from corerec.run_algo_tests import main as run_main
except ImportError as e:
    print(f"❌ Failed to import test runner: {e}")
    print("Make sure corerec is properly installed.")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error importing test runner: {e}")
    sys.exit(1)


def parse_args():
    """Parse command-line arguments with enhanced options."""
    parser = argparse.ArgumentParser(
        description="CoreRec Algorithm Test Runner - Enhanced",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        default=None,
        help="Save detailed JSON report to file (e.g., --json report.json)"
    )
    parser.add_argument(
        "--pattern",
        default="*_test*.py",
        help="Test file pattern for discovery (default: '*_test*.py')"
    )
    # Note: Filtering is not yet implemented in the underlying runner
    # parser.add_argument(
    #     "--filter",
    #     dest="filter_str",
    #     default=None,
    #     help="Filter tests by name/substring (case-insensitive)"
    # )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output (only final summary)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed unittest output for individual tests"
    )
    parser.add_argument(
        "--paths",
        default=None,
        help="Additional test directories (comma-separated, relative to project root)"
    )
    parser.add_argument(
        "--exit-zero",
        action="store_true",
        help="Always exit with code 0 (useful for CI/CD)"
    )
    return parser.parse_args()


def print_summary(total_time=0):
    """Print a nice summary message."""
    print("\n" + "=" * 70)
    print("📊 TEST RUN COMPLETED")
    print("=" * 70)
    if total_time > 0:
        print(f"⏱️  Total Time: {total_time:.2f}s")
    print("=" * 70)


def main():
    """Main entry point with enhanced features."""
    args = parse_args()
    
    # Build command line arguments for the underlying test runner
    cmd_args = [sys.argv[0]]
    
    # Default to tests directory
    test_paths = [str(ROOT / "tests")]
    
    # Add additional paths if specified
    if args.paths:
        additional = [p.strip() for p in args.paths.split(",")]
        test_paths.extend(additional)
    
    cmd_args.extend(["--paths"] + test_paths)
    cmd_args.extend(["--pattern", args.pattern])
    
    if args.json_output:
        cmd_args.extend(["--json", args.json_output])
    
    if args.verbose:
        cmd_args.append("--verbose")
    
    # Save original argv and set new one
    original_argv = sys.argv[:]
    sys.argv = cmd_args
    
    start_time = datetime.now()
    
    try:
        # Run the test runner
        # Note: run_main() will call sys.exit(), so we catch it
        run_main()
    except SystemExit as e:
        exit_code = e.code if isinstance(e.code, int) else 0
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        if not args.quiet and total_time > 0:
            print_summary(total_time)
        
        # Exit with appropriate code unless --exit-zero is set
        if args.exit_zero:
            sys.exit(0)
        else:
            sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test run interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error running tests: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
