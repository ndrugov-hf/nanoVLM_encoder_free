#!/usr/bin/env bash
# Run the typing checks: ruff (syntax/style) + pyright (type correctness).
# Usage:
#   ./check.sh                         # whole repo
#   ./check.sh models/vision_embedder.py   # one file or dir
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUFF="$ROOT/.venv/bin/ruff"
PYRIGHT="$ROOT/.venv/bin/pyright"
TARGET="${1:-.}"

echo "=== ruff ($TARGET) ==="
"$RUFF" check "$TARGET"
ruff_status=$?

echo
echo "=== pyright ($TARGET) ==="
"$PYRIGHT" "$TARGET"
pyright_status=$?

# Non-zero exit if either tool failed, so it works in CI / pre-commit too.
exit $(( ruff_status || pyright_status ))
