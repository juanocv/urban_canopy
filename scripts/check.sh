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
run_check pytest      python -m pytest
run_check diagnostics python -m urban_canopy.diagnostics

printf 'All checks passed\n'
