#!/usr/bin/env bash
# Create a venv and install the package with the requested optional layers.
#   ./scripts/setup-dev.sh [--api] [--ml]
set -euo pipefail

extras="dev"
for arg in "$@"; do
    case "$arg" in
        --api) extras="$extras,api" ;;
        --ml)  extras="$extras,ml" ;;
        *) echo "Unknown flag: $arg" >&2; exit 2 ;;
    esac
done

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[$extras]"
echo "Done. Activate with: source .venv/bin/activate"
