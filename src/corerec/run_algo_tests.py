#!/usr/bin/env python3
"""
CoreRec Algorithm Test Runner

Discovers and runs algorithm test files across the codebase and
produces a pass/fail report per algorithm.

- Test naming convention: algoname_test.py or *_test*.py
- Default search roots:
  - corerec/engines/
  - corerec/cf_engine/
  - corerec/uf_engine/
  - tests/

Usage:
  python -m corerec.run_algo_tests
  python corerec/run_algo_tests.py --pattern "*_test*.py" --json report.json --verbose

Outputs:
- Console report with per-algorithm status
- Optional JSON file with detailed results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import unittest
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SEARCH_DIRS = [
    PROJECT_ROOT / "tests",
]

DEFAULT_PATTERN = "*_test*.py"


@dataclass
class TestCaseResult:
    test_id: str
    outcome: str  # passed, failed, error, skipped
    message: str = ""


@dataclass
class AlgorithmTestReport:
    algorithm: str
    module: str
    file: str
    total: int
    passed: int
    failed: int
    errors: int
    skipped: int
    duration_s: float
    cases: List[TestCaseResult]


class AggregatingTestResult(unittest.TestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.case_results: List[TestCaseResult] = []
        self._start_times: Dict[str, float] = {}

    def startTest(self, test):
        super().startTest(test)
        self._start_times[str(test)] = time.time()

    def addSuccess(self, test):
        super().addSuccess(test)
        self.case_results.append(TestCaseResult(test_id=str(test), outcome="passed"))

    def addFailure(self, test, err):
        super().addFailure(test, err)
        msg = self._exc_info_to_string(err, test)
        self.case_results.append(TestCaseResult(test_id=str(test), outcome="failed", message=msg))

    def addError(self, test, err):
        super().addError(test, err)
        msg = self._exc_info_to_string(err, test)
        self.case_results.append(TestCaseResult(test_id=str(test), outcome="error", message=msg))

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.case_results.append(
            TestCaseResult(test_id=str(test), outcome="skipped", message=str(reason))
        )


def discover_tests(search_dirs: List[Path], pattern: str) -> List[unittest.TestSuite]:
    loader = unittest.TestLoader()
    suites: List[unittest.TestSuite] = []
    for root in search_dirs:
        if not root.exists():
            continue
        try:
            suite = loader.discover(
                start_dir=str(root), pattern=pattern, top_level_dir=str(PROJECT_ROOT)
            )
            suites.append(suite)
        except Exception:
            # Continue even if a directory has issues
            traceback.print_exc()
            continue
    return suites


def flatten_suite(suite: unittest.TestSuite) -> List[unittest.TestSuite]:
    """Flatten nested suites into a list of leaf suites."""
    items = []
    for s in suite:
        if isinstance(s, unittest.TestSuite):
            items.extend(flatten_suite(s))
        else:
            items.append(s)
    return items


def extract_algo_name_from_file(filepath: Path) -> str:
    """Derive algorithm name from test filename using the convention algoname_test*.py."""
    name = filepath.stem
    # Remove typical suffixes
    for suffix in ["_test", "_tests", "-test", "-tests"]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    # If pattern like something_test_extra
    if "_test" in name:
        return name.split("_test")[0]
    return name


def _is_loader_failed_test(test: unittest.case.TestCase) -> bool:
    cls_name = test.__class__.__name__
    mod_name = getattr(test, "__module__", "")
    return cls_name == "_FailedTest" or mod_name.startswith("unittest.loader")


def run_tests_and_aggregate(
    search_dirs: List[Path], pattern: str, verbose: bool = False
) -> Tuple[List[AlgorithmTestReport], int]:
    all_suites = discover_tests(search_dirs, pattern)
    # Map: (module_file) -> suite
    reports: List[AlgorithmTestReport] = []
    total_failures = 0

    for suite in all_suites:
        # Each suite may contain tests from multiple files; iterate by module
        for subsuite in suite:
            # Attempt to infer file/module for the subsuite
            # Flatten tests to determine file origin
            flattened = flatten_suite(subsuite)
            if not flattened:
                continue

            # Skip loader failures that come from discovery hiccups
            if all(_is_loader_failed_test(t) for t in flattened if hasattr(t, "__class__")):
                continue

            # Identify file/module from first non-loader test
            test0 = next((t for t in flattened if not _is_loader_failed_test(t)), flattened[0])
            module_name = getattr(test0, "__module__", "unknown")
            module_obj = sys.modules.get(module_name)
            file_path: Optional[Path] = None
            if module_obj is not None and hasattr(module_obj, "__file__"):
                file_path = Path(module_obj.__file__) if module_obj.__file__ else None

            # Fallback: derive from id string
            if file_path is None:
                test_id = getattr(test0, "id", lambda: "")()
                # id looks like: package.module.ClassName.test_method
                mod_part = ".".join(test_id.split(".")[:-2]) if "." in test_id else module_name
                try:
                    module_obj = __import__(mod_part, fromlist=["*"])
                    file_path = Path(module_obj.__file__)
                except Exception:
                    file_path = Path("unknown")

            algo_name = (
                extract_algo_name_from_file(file_path)
                if file_path and file_path.name
                else module_name
            )

            # Run this subsuite
            result = AggregatingTestResult()
            runner = unittest.TextTestRunner(
                stream=sys.stdout if verbose else open(os.devnull, "w"),
                verbosity=2 if verbose else 1,
                resultclass=lambda *a, **k: result,
            )
            start = time.time()
            runner.run(subsuite)
            duration = time.time() - start

            # Aggregate counts using case-level outcomes (handles subTest correctly)
            if result.case_results:
                total = len(result.case_results)
                failed = sum(1 for c in result.case_results if c.outcome == "failed")
                errors = sum(1 for c in result.case_results if c.outcome == "error")
                skipped = sum(1 for c in result.case_results if c.outcome == "skipped")
                passed = sum(1 for c in result.case_results if c.outcome == "passed")
            else:
                total = result.testsRun
                failed = len(result.failures)
                errors = len(result.errors)
                skipped = len(result.skipped)
                passed = total - failed - errors - skipped
            total_failures += failed + errors

            reports.append(
                AlgorithmTestReport(
                    algorithm=algo_name,
                    module=module_name,
                    file=str(file_path) if file_path else "unknown",
                    total=total,
                    passed=passed,
                    failed=failed,
                    errors=errors,
                    skipped=skipped,
                    duration_s=round(duration, 3),
                    cases=result.case_results,
                )
            )
    return reports, total_failures


def print_report(reports: List[AlgorithmTestReport]):
    if not reports:
        print("\nNo tests found. Adjust --pattern or --paths.")
        return

    print("\n=== CoreRec Algorithm Test Report ===")
    # Group by algorithm
    by_algo: Dict[str, List[AlgorithmTestReport]] = {}
    for r in reports:
        by_algo.setdefault(r.algorithm, []).append(r)

    # Print per algorithm summary
    lines = []
    header = f"{'ALGORITHM':30} {'TOTAL':>5} {'PASS':>5} {'FAIL':>5} {'ERR':>4} {'SKIP':>5} {'TIME(s)':>7} STATUS"
    print(header)
    print("-" * len(header))

    overall_total = overall_pass = overall_fail = overall_err = overall_skip = 0

    for algo, algo_reports in sorted(by_algo.items()):
        total = sum(r.total for r in algo_reports)
        passed = sum(r.passed for r in algo_reports)
        failed = sum(r.failed for r in algo_reports)
        errors = sum(r.errors for r in algo_reports)
        skipped = sum(r.skipped for r in algo_reports)
        duration = sum(r.duration_s for r in algo_reports)
        status = "OK" if (failed == 0 and errors == 0 and total > 0) else "BROKEN"

        overall_total += total
        overall_pass += passed
        overall_fail += failed
        overall_err += errors
        overall_skip += skipped

        print(
            f"{algo:30} {total:5d} {passed:5d} {failed:5d} {errors:4d} {skipped:5d} {duration:7.2f} {status}"
        )

    print("-" * len(header))
    print(
        f"{'TOTAL':30} {overall_total:5d} {overall_pass:5d} {overall_fail:5d} {overall_err:4d} {overall_skip:5d}"
    )


def save_json_report(reports: List[AlgorithmTestReport], output_path: Path):
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "project_root": str(PROJECT_ROOT),
        "reports": [
            {
                **asdict(r),
                "cases": [asdict(c) for c in r.cases],
            }
            for r in reports
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"\nJSON report saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CoreRec Algorithm Test Runner")
    parser.add_argument(
        "--paths",
        nargs="*",
        default=[str(p) for p in DEFAULT_SEARCH_DIRS],
        help="Directories to search for tests (default: tests)",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help=f"Test file pattern for discovery (default: {DEFAULT_PATTERN})",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        default=None,
        help="Optional path to save JSON report",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show unittest output for individual tests",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Build search dirs
    search_dirs: List[Path] = []
    for p in args.paths:
        path = Path(p)
        if not path.is_absolute():
            path = PROJECT_ROOT / p
        if path.exists():
            search_dirs.append(path)

    print("Searching for tests in:")
    for d in search_dirs:
        print(f" - {d}")
    print(f"Pattern: {args.pattern}")

    reports, total_failures = run_tests_and_aggregate(
        search_dirs, args.pattern, verbose=args.verbose
    )
    print_report(reports)

    if args.json_output:
        save_json_report(reports, Path(args.json_output))

    # Exit code: non-zero if any failures/errors
    sys.exit(1 if total_failures > 0 else 0)


if __name__ == "__main__":
    main()
