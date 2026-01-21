#!/usr/bin/env python3
"""
Export PKL geometry to JSON for manual XODR alignment in the browser.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import align_network_pkl_to_xodr as align  # noqa: E402


def _collect_pkl_paths(entries: List[str]) -> List[Path]:
    pkl_paths: List[Path] = []
    for entry in entries:
        path = Path(entry)
        if path.is_dir():
            pkl_paths.extend(sorted(path.rglob("*.pkl")))
        else:
            pkl_paths.append(path)
    return pkl_paths


def _sample_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    return align._sample_points(points, max_points, seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkls", nargs="+", required=True, help="PKL files or directories.")
    parser.add_argument("--out_dir", default="alignment_manual_inputs", help="Output directory for JSON files.")
    parser.add_argument("--key", default=None, help="Optional key path to select a subtree in the PKL.")
    parser.add_argument("--max_depth", type=int, default=6, help="Traversal depth when extracting geometry.")
    parser.add_argument("--max_points", type=int, default=200000, help="Max points to keep (0 = no limit).")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for downsampling.")
    parser.add_argument("--strict_pickle", action="store_true", help="Fail on unknown pickle classes.")
    parser.add_argument(
        "--include_polylines",
        action="store_true",
        help="Include polylines in JSON (larger output).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pkl_paths = _collect_pkl_paths(args.pkls)
    if not pkl_paths:
        raise SystemExit("No PKL files found.")

    for pkl_path in pkl_paths:
        if not pkl_path.exists():
            print(f"[WARN] Missing PKL: {pkl_path}")
            continue

        geom = align._load_pkl_geometry(
            pkl_path,
            key=args.key,
            max_depth=args.max_depth,
            strict_pickle=args.strict_pickle,
        )
        points = geom["points"]
        total_points = int(points.shape[0])
        if args.max_points > 0:
            points = _sample_points(points, args.max_points, args.seed)

        payload = {
            "type": "pkl-geometry",
            "source_pkl": str(pkl_path),
            "point_count": int(points.shape[0]),
            "point_count_total": total_points,
            "points": points.tolist(),
        }
        if args.include_polylines:
            payload["polylines"] = [poly.tolist() for poly in geom["polylines"]]

        out_path = out_dir / f"{pkl_path.stem}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"[OK] Wrote {out_path} ({payload['point_count']} points)")


if __name__ == "__main__":
    main()
