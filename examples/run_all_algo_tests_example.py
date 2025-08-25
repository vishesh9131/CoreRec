#!/usr/bin/env python3
# triggers the unified algo test runner and prints a brief summary

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from corerec.run_algo_tests import main as run_main
except Exception as e:
    print("Failed to import test runner:", e)
    sys.exit(1)

if __name__ == "__main__":
    # Limit search to tests/ to avoid heavy imports; use default pattern
    # Equivalent to: python -m corerec.run_algo_tests --dirs tests --verbose
    sys.argv = [sys.argv[0], "--dirs", str(ROOT / "tests"), "--verbose"]
    run_main() 