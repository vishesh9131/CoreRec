#!/usr/bin/env python3
"""
Run All NN Base Tests
"""

import os
import sys
import subprocess

# Test scripts to run
TEST_SCRIPTS = [
    "test_ncf.py",
    "test_deepfm.py",
    "test_dcn.py",
    "test_din.py",
    "test_caser.py",
    "test_gru_cf.py",
    "test_nextitnet.py",
    "test_afm.py",
    "test_autoint.py",
]


def run_all_tests():
    """Run all test scripts"""
    print("=" * 70)
    print(" Running All NN Base Tests")
    print("=" * 70)

    test_dir = os.path.dirname(os.path.abspath(__file__))
    results = {}

    for test_script in TEST_SCRIPTS:
        test_path = os.path.join(test_dir, test_script)

        if not os.path.exists(test_path):
            print(f"\n⚠️  Skipping {test_script} (not found)")
            results[test_script] = "SKIPPED"
            continue

        print(f"\n{'=' * 70}")
        print(f"Running {test_script}...")
        print(f"{'=' * 70}")

        try:
            result = subprocess.run(
                [sys.executable, test_path],
                capture_output=False,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode == 0:
                results[test_script] = "PASSED"
                print(f"✓ {test_script} PASSED")
            else:
                results[test_script] = "FAILED"
                print(f"✗ {test_script} FAILED")
        except subprocess.TimeoutExpired:
            results[test_script] = "TIMEOUT"
            print(f"⏱️  {test_script} TIMEOUT")
        except Exception as e:
            results[test_script] = "ERROR"
            print(f"❌ {test_script} ERROR: {e}")

    # Print summary
    print("\n" + "=" * 70)
    print(" Test Summary")
    print("=" * 70)

    passed = sum(1 for v in results.values() if v == "PASSED")
    failed = sum(1 for v in results.values() if v == "FAILED")
    skipped = sum(1 for v in results.values() if v == "SKIPPED")
    errors = sum(1 for v in results.values() if v in ["ERROR", "TIMEOUT"])

    for test_script, result in results.items():
        status_icon = {
            "PASSED": "✓",
            "FAILED": "✗",
            "SKIPPED": "⚠️",
            "ERROR": "❌",
            "TIMEOUT": "⏱️",
        }.get(result, "?")

        print(f"{status_icon} {test_script:<30} {result}")

    print(f"\nTotal: {len(results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Errors/Timeouts: {errors}")

    return passed == len(results) - skipped


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
