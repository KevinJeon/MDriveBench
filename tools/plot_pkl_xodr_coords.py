#!/usr/bin/env python3
"""
Plot coordinates from V2XPNP map PKLs and an OpenDRIVE XODR on the same map.
"""

from __future__ import annotations

import argparse
import math
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Iterable, List, Tuple

import numpy as np


class _MapStubBase:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["state"] = state


class _MapStub(_MapStubBase):
    pass


class _LaneStub(_MapStubBase):
    pass


class _MapPointStub(_MapStubBase):
    pass


class _MapUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "opencood.data_utils.datasets.map.map_types":
            if name == "Map":
                return _MapStub
            if name == "Lane":
                return _LaneStub
            if name == "MapPoint":
                return _MapPointStub
        return super().find_class(module, name)


def _iter_map_points(items: Iterable) -> Iterable[Tuple[float, float]]:
    for item in items:
        if hasattr(item, "x") and hasattr(item, "y"):
            try:
                yield float(item.x), float(item.y)
            except Exception:
                continue


def _load_map_pkl_coords(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        obj = _MapUnpickler(f).load()

    coords: List[Tuple[float, float]] = []
    map_features = getattr(obj, "map_features", None)
    if not isinstance(map_features, list):
        return np.zeros((0, 2), dtype=np.float64)

    for feature in map_features:
        polyline = getattr(feature, "polyline", None)
        if isinstance(polyline, list):
            coords.extend(_iter_map_points(polyline))
        boundary = getattr(feature, "boundary", None)
        if isinstance(boundary, list):
            coords.extend(_iter_map_points(boundary))

    if not coords:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(coords, dtype=np.float64)


def _integrate_geometry(
    x0: float,
    y0: float,
    hdg: float,
    length: float,
    curv_fn: Callable[[float], float],
    step: float,
) -> List[Tuple[float, float]]:
    if length <= 0.0:
        return []
    step = max(step, 0.1)
    n = max(1, int(math.ceil(length / step)))
    ds = length / n
    x = x0
    y = y0
    theta = hdg
    points = [(x, y)]
    for i in range(n):
        s_mid = (i + 0.5) * ds
        kappa = curv_fn(s_mid)
        theta_mid = theta + 0.5 * kappa * ds
        x += ds * math.cos(theta_mid)
        y += ds * math.sin(theta_mid)
        theta += kappa * ds
        points.append((x, y))
    return points


def _sample_geometry(geom: ET.Element, step: float) -> List[Tuple[float, float]]:
    x0 = float(geom.attrib.get("x", 0.0))
    y0 = float(geom.attrib.get("y", 0.0))
    hdg = float(geom.attrib.get("hdg", 0.0))
    length = float(geom.attrib.get("length", 0.0))

    child = next(iter(geom), None)
    if child is None:
        return [(x0, y0)]

    if child.tag == "line":
        curv_fn = lambda s: 0.0
    elif child.tag == "arc":
        curvature = float(child.attrib.get("curvature", 0.0))
        curv_fn = lambda s, k=curvature: k
    elif child.tag == "spiral":
        curv_start = float(child.attrib.get("curvStart", 0.0))
        curv_end = float(child.attrib.get("curvEnd", 0.0))

        def curv_fn(s: float, cs=curv_start, ce=curv_end, total=length) -> float:
            if total <= 0.0:
                return cs
            return cs + (ce - cs) * (s / total)
    else:
        curv_fn = lambda s: 0.0

    return _integrate_geometry(x0, y0, hdg, length, curv_fn, step)


def _load_xodr_coords(path: Path, step: float) -> np.ndarray:
    root = ET.parse(path).getroot()
    points: List[Tuple[float, float]] = []
    for geom in root.findall(".//planView/geometry"):
        points.extend(_sample_geometry(geom, step))
    if not points:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(points, dtype=np.float64)


def _sample_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pkl",
        required=True,
        help="Path to a PKL file or a directory containing PKL files.",
    )
    parser.add_argument("--xodr", required=True, help="Path to OpenDRIVE XODR file.")
    parser.add_argument(
        "--xodr-step",
        type=float,
        default=2.0,
        help="Sampling step size (meters) along XODR geometry (default: 2.0).",
    )
    parser.add_argument(
        "--max-pkl-points",
        type=int,
        default=0,
        help="Randomly sample this many PKL points (0 = no limit).",
    )
    parser.add_argument(
        "--max-xodr-points",
        type=int,
        default=0,
        help="Randomly sample this many XODR points (0 = no limit).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Seed for random downsampling.",
    )
    parser.add_argument(
        "--output",
        default="pkl_xodr_map.png",
        help="Output image path.",
    )
    parser.add_argument("--show", action="store_true", help="Show interactive plot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pkl_path = Path(args.pkl)
    xodr_path = Path(args.xodr)
    if not pkl_path.exists():
        raise FileNotFoundError(pkl_path)
    if not xodr_path.exists():
        raise FileNotFoundError(xodr_path)

    pkl_files: List[Path] = []
    if pkl_path.is_dir():
        pkl_files = sorted(pkl_path.rglob("*.pkl"))
    else:
        pkl_files = [pkl_path]

    all_pkl_points: List[np.ndarray] = []
    for pkl_file in pkl_files:
        try:
            pts = _load_map_pkl_coords(pkl_file)
        except Exception:
            continue
        if pts.size:
            all_pkl_points.append(pts)

    if all_pkl_points:
        pkl_points = np.vstack(all_pkl_points)
    else:
        pkl_points = np.zeros((0, 2), dtype=np.float64)

    xodr_points = _load_xodr_coords(xodr_path, args.xodr_step)

    pkl_points = _sample_points(pkl_points, args.max_pkl_points, args.seed)
    xodr_points = _sample_points(xodr_points, args.max_xodr_points, args.seed)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10))
    if xodr_points.size:
        plt.scatter(
            xodr_points[:, 0],
            xodr_points[:, 1],
            s=1,
            c="#1f77b4",
            alpha=0.6,
            label="XODR",
        )
    if pkl_points.size:
        plt.scatter(
            pkl_points[:, 0],
            pkl_points[:, 1],
            s=2,
            c="#d62728",
            alpha=0.6,
            label="PKL trajectories",
        )
    plt.axis("equal")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("PKL trajectories vs XODR geometry")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
