#!/usr/bin/env bash
# Runs the same gates as .github/workflows/checks.yml and scripts/check.ps1.
set -uo pipefail

run_check() {
    local name=$1
    shift
    printf '== %s\n' "$name"
    if ! "$@"; then
        printf '== %s FAILED (exit %d)\n' "$name" "$?" >&2
        exit 1
    fi
}

run_check compileall  python -m compileall urban_canopy -q
run_check ruff        python -m ruff check urban_canopy
run_check black       python -m black --check urban_canopy
run_check pyright     python -m pyright
run_check pytest      python -m pytest --cov=urban_canopy --cov-report=term-missing --cov-report=json:coverage.json --cov-fail-under=80
run_check coverage    python scripts/check_coverage.py coverage.json --fail-under 60
run_check diagnostics python -m urban_canopy.diagnostics

printf 'All checks passed\n'
