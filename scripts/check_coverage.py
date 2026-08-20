#!/usr/bin/env python
"""Fail when any measured production module falls below its coverage floor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path, nargs="?", default=Path("coverage.json"))
    parser.add_argument("--fail-under", type=float, default=60.0)
    args = parser.parse_args()

    payload = json.loads(args.report.read_text(encoding="utf-8"))
    measured: list[tuple[str, float]] = []
    for raw_name, data in payload.get("files", {}).items():
        name = raw_name.replace("\\", "/")
        if not name.startswith("urban_canopy/") or "/tests/" in name:
            continue
        measured.append((name, float(data["summary"]["percent_covered"])))

    if not measured:
        parser.error(f"{args.report} contains no measured urban_canopy modules")

    failing = sorted((name, pct) for name, pct in measured if pct < args.fail_under)
    for name, pct in sorted(measured, key=lambda item: item[1])[:10]:
        print(f"{pct:6.2f}%  {name}")
    if failing:
        print(f"\nModules below {args.fail_under:.2f}%:")
        for name, pct in failing:
            print(f"  {pct:6.2f}%  {name}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
