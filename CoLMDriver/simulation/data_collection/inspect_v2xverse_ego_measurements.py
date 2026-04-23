#!/usr/bin/env python3
"""
Under --root, only under .../weather-*/data/** (not loose ego dirs under dataset root):
  - Default: print each ego_vehicle_* and len(os.listdir(...)).
  - --prune-routes: parent folder named routes_* is removed if ANY ego_vehicle_* there
    has listdir length < --expect (default 15). Without --apply, only print paths.

Usage:
  python simulation/data_collection/inspect_v2xverse_ego_measurements.py --root dataset
  python simulation/data_collection/inspect_v2xverse_ego_measurements.py --root dataset --prune-routes
  python simulation/data_collection/inspect_v2xverse_ego_measurements.py --root dataset --prune-routes --apply
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def _is_under_weather_data(ego: Path) -> bool:
    """True if path is .../weather-<id>/data/.../ego_vehicle_*"""
    parts = ego.parts
    for i, part in enumerate(parts):
        if part.startswith("weather-") and i + 1 < len(parts) and parts[i + 1] == "data":
            return True
    return False


def _iter_ego_vehicle_dirs(root: Path):
    for ego in sorted(root.rglob("ego_vehicle_*")):
        if not ego.is_dir() or not ego.name.startswith("ego_vehicle"):
            continue
        if not _is_under_weather_data(ego):
            continue
        yield ego


def _route_dir_for_ego(ego: Path) -> Path | None:
    """Only treat parents named routes_* as deletable route roots."""
    p = ego.parent
    if p.name.startswith("routes_"):
        return p
    return None


def main() -> int:
    p = argparse.ArgumentParser(
        description="List ego_vehicle listdir lengths or prune bad routes (only .../weather-*/data/**)."
    )
    p.add_argument("--root", default="dataset", help="Dataset root. Default: dataset")
    p.add_argument(
        "--expect",
        type=int,
        default=15,
        help="Minimum len(os.listdir(ego)) for each ego_vehicle_* when using --prune-routes; below this removes the route. Default: 15",
    )
    p.add_argument(
        "--prune-routes",
        action="store_true",
        help="Remove routes_* dirs where any ego_vehicle_* has count < --expect",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="With --prune-routes, actually delete directories (otherwise dry-run)",
    )
    args = p.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 1

    if args.prune_routes:
        bad_routes: dict[Path, list[tuple[str, int]]] = {}
        for ego in _iter_ego_vehicle_dirs(root):
            route = _route_dir_for_ego(ego)
            if route is None:
                continue
            n = len(os.listdir(ego))
            if n < args.expect:
                bad_routes.setdefault(route, []).append((str(ego.relative_to(root)), n))

        if not bad_routes:
            print("No routes_* need pruning under .../weather-*/data/ (all ego_vehicle_* have count >= --expect).")
            return 0

        for route in sorted(bad_routes):
            rel = route.relative_to(root)
            reasons = "; ".join(f"{e}={c}" for e, c in bad_routes[route])
            if args.apply:
                shutil.rmtree(route)
                print(f"deleted {rel}  ({reasons})")
            else:
                print(f"would delete {rel}  ({reasons})")

        if not args.apply:
            print(f"\nDry-run only. Re-run with --apply to delete {len(bad_routes)} route folder(s).", file=sys.stderr)
        return 0

    found = False
    for ego in _iter_ego_vehicle_dirs(root):
        found = True
        n = len(os.listdir(ego))
        print(f"{ego.relative_to(root)}: {n}")

    if not found:
        print(f"No ego_vehicle_* under {root} in .../weather-*/data/**", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
