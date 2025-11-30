#!/bin/bash
# Format code using Black
# Usage: ./scripts/format_code.sh [--check]

set -e

cd "$(dirname "$0")/.."

echo "Running Black formatter..."

if [ "$1" == "--check" ]; then
    python -m black --check corerec/ tests/ examples/
    echo "✓ Black check passed"
else
    python -m black corerec/ tests/ examples/
    echo "✓ Code formatted with Black"
fi

