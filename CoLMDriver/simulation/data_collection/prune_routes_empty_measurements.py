#!/usr/bin/env python3
"""
Under dataset/weather-<id>/data/<routes_*>/, if any ego_vehicle_*/measurements/
is missing or has no files, remove the whole routes_* folder.

Default: only weather-0. Use --all-weathers for every weather-*/data/.

Dry-run unless --apply.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def _route_dirs_data(root: Path, *, all_weathers: bool) -> list[Path]:
    out: list[Path] = []
    if all_weathers:
        for w in sorted(root.glob("weather-*")):
            data = w / "data"
            if not data.is_dir():
                continue
            for p in sorted(data.iterdir()):
                if p.is_dir():
                    out.append(p)
    else:
        data = root / "weather-0" / "data"
        if data.is_dir():
            for p in sorted(data.iterdir()):
                if p.is_dir():
                    out.append(p)
    return out


def _has_empty_measurements(route: Path) -> tuple[bool, str]:
    """True if any ego_vehicle_* has no measurements dir or empty measurements."""
    has_ego = False
    for ego in sorted(route.iterdir()):
        if not ego.is_dir() or not ego.name.startswith("ego_vehicle"):
            continue
        has_ego = True
        mea = ego / "measurements"
        if not mea.is_dir():
            return True, f"{ego.name}: no measurements/"
        try:
            if not any(mea.iterdir()):
                return True, f"{ego.name}: measurements/ empty"
        except OSError as e:
            return True, f"{ego.name}: measurements/ unreadable ({e})"
    if not has_ego:
        return False, "no ego_vehicle_* (skipped)"
    return False, ""


def main() -> int:
    p = argparse.ArgumentParser(description="Remove routes_* with empty ego measurements/")
    p.add_argument("--root", default="dataset", help="Dataset root (contains weather-0/...)")
    p.add_argument(
        "--all-weathers",
        action="store_true",
        help="Scan weather-*/data/ instead of only weather-0/data/",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete directories (default is dry-run)",
    )
    args = p.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 1

    routes = _route_dirs_data(root, all_weathers=args.all_weathers)
    if not routes:
        scope = "weather-*/data" if args.all_weathers else "weather-0/data"
        print(f"No subfolders under {root}/{scope}")
        return 2

    bad: list[tuple[Path, str]] = []
    for route in routes:
        empty, reason = _has_empty_measurements(route)
        if empty:
            bad.append((route, reason))

    if not bad:
        print("Nothing to remove (all ego_vehicle_*/measurements/ non-empty).")
        return 0

    for route, reason in bad:
        rel = route.relative_to(root)
        if args.apply:
            shutil.rmtree(route)
            print(f"deleted {rel}  ({reason})")
        else:
            print(f"would delete {rel}  ({reason})")

    if not args.apply:
        print(
            f"\nDry-run: {len(bad)} route folder(s). Re-run with --apply to delete.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
