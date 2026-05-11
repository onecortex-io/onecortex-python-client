#!/usr/bin/env bash
set -euo pipefail

STAGED_SRC=$(git diff --cached --name-only --diff-filter=ACM | grep -E '^src/' || true)
if [ -n "$STAGED_SRC" ]; then
  echo ""
  echo "SDK PARITY REMINDER: staged changes detected in src/."
  echo "Ensure the counterpart SDK is updated before pushing."
  echo "  Python  <->  TypeScript"
  echo ""
fi
exit 0
