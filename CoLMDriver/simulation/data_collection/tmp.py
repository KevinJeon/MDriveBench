#!/usr/bin/env python3
"""Peek into ego_vehicle_*/measurements/ dirs under --root (names + optional JSON keys)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="dataset", help="Dataset root")
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of measurements dirs to print (0 = no limit)",
    )
    p.add_argument(
        "--sample-keys",
        action="store_true",
        help="For each dir, print top-level keys of the first *.json (lexicographic)",
    )
    p.add_argument(
        "--files",
        type=int,
        default=30,
        help="Max file names listed per dir (default 30)",
    )
    args = p.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 1

    dirs: list[Path] = []
    for d in sorted(root.rglob("measurements")):
        if not d.is_dir():
            continue
        if not d.parent.name.startswith("ego_vehicle"):
            continue
        dirs.append(d)

    if not dirs:
        print(f"No ego_vehicle_*/measurements under {root}")
        return 2

    n = 0
    for mea in dirs:
        if args.limit and n >= args.limit:
            print(f"... stopped after {args.limit} dirs (--limit)")
            break
        n += 1
        rel = mea.relative_to(root)
        names = sorted(p.name for p in mea.iterdir())
        print(f"{rel}  (n={len(names)})")
        for name in names[: args.files]:
            print(f"  {name}")
        if len(names) > args.files:
            print(f"  ... +{len(names) - args.files} more")

        if args.sample_keys:
            jsons = sorted(mea.glob("*.json"))
            if not jsons:
                print("  (no .json)")
                continue
            try:
                with open(jsons[0], encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    ks = sorted(data.keys())
                    print(f"  sample {jsons[0].name} keys ({len(ks)}): {', '.join(ks)}")
                else:
                    print(f"  sample {jsons[0].name}: not a dict")
            except (OSError, json.JSONDecodeError) as e:
                print(f"  sample read error: {e}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
