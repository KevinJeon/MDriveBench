#!/usr/bin/env python3
"""
Align PKL road network geometry to an OpenDRIVE XODR map using a 2D rigid/similarity transform.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import time
import xml.etree.ElementTree as ET
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree


AXIS_VARIANTS = {
    "xy": lambda p: p,
    "x_neg_y": lambda p: np.stack([p[:, 0], -p[:, 1]], axis=1),
    "neg_x_y": lambda p: np.stack([-p[:, 0], p[:, 1]], axis=1),
    "neg_x_neg_y": lambda p: np.stack([-p[:, 0], -p[:, 1]], axis=1),
    "y_x": lambda p: np.stack([p[:, 1], p[:, 0]], axis=1),
    "y_neg_x": lambda p: np.stack([p[:, 1], -p[:, 0]], axis=1),
    "neg_y_x": lambda p: np.stack([-p[:, 1], p[:, 0]], axis=1),
    "neg_y_neg_x": lambda p: np.stack([-p[:, 1], -p[:, 0]], axis=1),
}

class _StubBase:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["state"] = state


class _StubMap(_StubBase):
    pass


class _StubLane(_StubBase):
    pass


class _StubMapPoint(_StubBase):
    pass


class _GenericStub(_StubBase):
    pass


class _SafeUnpickler(pickle.Unpickler):
    def __init__(self, *args, strict: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._strict = strict

    def find_class(self, module, name):
        if module == "opencood.data_utils.datasets.map.map_types":
            if name == "Map":
                return _StubMap
            if name == "Lane":
                return _StubLane
            if name == "MapPoint":
                return _StubMapPoint
        if self._strict:
            return super().find_class(module, name)
        return _GenericStub


@dataclass(frozen=True)
class _PolySegment:
    s: float
    a: float
    b: float
    c: float
    d: float


@dataclass
class _Lane:
    lane_id: int
    lane_type: str
    width_segments: List[_PolySegment]
    width_starts: List[float]


@dataclass
class _LaneSection:
    s: float
    left: List[_Lane]
    right: List[_Lane]


def _sample_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


def _log(msg: str) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    print(f"[{timestamp}] {msg}", flush=True)


def _pca_stats(points: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
    if points.shape[0] == 0:
        return np.zeros(2), 0.0, 1.0, 1.0
    centroid = points.mean(axis=0)
    centered = points - centroid
    if centered.shape[0] < 2:
        return centroid, 0.0, 1.0, 1.0
    cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    principal = eigvecs[:, 0]
    theta = math.atan2(principal[1], principal[0])
    rms = math.sqrt(float(max(eigvals.sum(), 1e-12)))
    ratio = float(eigvals[0] / max(eigvals[1], 1e-12)) if eigvals.size > 1 else 1.0
    return centroid, theta, rms, ratio


def _auto_search_params(pkl_points: np.ndarray, map_points: np.ndarray) -> Tuple[float, float, float, float]:
    map_centroid, map_theta, map_rms, map_ratio = _pca_stats(map_points)
    pkl_centroid, pkl_theta, pkl_rms, pkl_ratio = _pca_stats(pkl_points)
    map_span = map_points.max(axis=0) - map_points.min(axis=0)
    pkl_span = pkl_points.max(axis=0) - pkl_points.min(axis=0)
    map_diag = float(np.linalg.norm(map_span))
    pkl_diag = float(np.linalg.norm(pkl_span))
    centroid_dist = float(np.linalg.norm(map_centroid - pkl_centroid))
    theta_deg = 180.0 if max(map_ratio, pkl_ratio) < 1.5 else 90.0
    theta_step_deg = max(15.0, theta_deg / 6.0)
    trans_m = max(centroid_dist * 1.2, 0.6 * max(map_diag, pkl_diag), 50.0)
    trans_step_m = max(10.0, trans_m / 10.0)
    return theta_deg, theta_step_deg, trans_m, trans_step_m


def _parse_xodr(
    xodr_path: Path,
    step_m: float,
    approx_spiral: bool,
    lane_mode: str,
    lane_types: Optional[set],
) -> Tuple[np.ndarray, int]:
    root = ET.parse(xodr_path).getroot()
    points: List[np.ndarray] = []
    spiral_skipped = 0

    for road in root.findall(".//road"):
        ref_samples, skipped = _sample_reference_line(road, step_m, approx_spiral)
        spiral_skipped += skipped
        if ref_samples["xy"].size:
            points.append(ref_samples["xy"])

        if lane_mode != "reference":
            lane_points = _sample_lane_points(road, ref_samples, lane_mode, lane_types)
            if lane_points.size:
                points.append(lane_points)

    if not points:
        return np.zeros((0, 2), dtype=np.float64), spiral_skipped
    return np.vstack(points), spiral_skipped


def _sample_reference_line(
    road_elem: ET.Element,
    step: float,
    approx_spiral: bool,
) -> Tuple[Dict[str, np.ndarray], int]:
    s_list: List[np.ndarray] = []
    x_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    h_list: List[np.ndarray] = []
    spiral_skipped = 0

    for geom in road_elem.findall("./planView/geometry"):
        s0 = float(geom.attrib.get("s", 0.0))
        x0 = float(geom.attrib.get("x", 0.0))
        y0 = float(geom.attrib.get("y", 0.0))
        hdg = float(geom.attrib.get("hdg", 0.0))
        length = float(geom.attrib.get("length", 0.0))
        if length <= 0.0:
            continue
        child = next(iter(geom), None)
        if child is None:
            continue

        if child.tag == "line":
            s, x, y, h = _sample_line_with_heading(x0, y0, hdg, length, step, s0)
        elif child.tag == "arc":
            curvature = float(child.attrib.get("curvature", 0.0))
            s, x, y, h = _sample_arc_with_heading(x0, y0, hdg, length, curvature, step, s0)
        elif child.tag == "spiral":
            if approx_spiral:
                curv_start = float(child.attrib.get("curvStart", 0.0))
                curv_end = float(child.attrib.get("curvEnd", 0.0))
                s, x, y, h = _sample_spiral_with_heading(
                    x0, y0, hdg, length, curv_start, curv_end, step, s0
                )
            else:
                spiral_skipped += 1
                continue
        else:
            continue

        s_list.append(s)
        x_list.append(x)
        y_list.append(y)
        h_list.append(h)

    if not s_list:
        empty = np.zeros((0,), dtype=np.float64)
        return {"s": empty, "x": empty, "y": empty, "hdg": empty, "xy": np.zeros((0, 2))}, spiral_skipped

    s_vals = np.concatenate(s_list)
    x_vals = np.concatenate(x_list)
    y_vals = np.concatenate(y_list)
    h_vals = np.concatenate(h_list)
    xy = np.stack([x_vals, y_vals], axis=1)
    return {"s": s_vals, "x": x_vals, "y": y_vals, "hdg": h_vals, "xy": xy}, spiral_skipped


def _sample_line_with_heading(
    x0: float, y0: float, hdg: float, length: float, step: float, s0: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    step = max(step, 0.1)
    n = max(1, int(math.ceil(length / step)))
    s_local = np.linspace(0.0, length, n + 1)
    cos_h = math.cos(hdg)
    sin_h = math.sin(hdg)
    x = x0 + s_local * cos_h
    y = y0 + s_local * sin_h
    s = s0 + s_local
    h = np.full_like(s, hdg)
    return s, x, y, h


def _sample_arc_with_heading(
    x0: float,
    y0: float,
    hdg: float,
    length: float,
    curvature: float,
    step: float,
    s0: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if abs(curvature) < 1e-6:
        return _sample_line_with_heading(x0, y0, hdg, length, step, s0)
    step = max(step, 0.1)
    n = max(1, int(math.ceil(length / step)))
    s_local = np.linspace(0.0, length, n + 1)
    theta = hdg + curvature * s_local
    x = x0 + (np.sin(theta) - math.sin(hdg)) / curvature
    y = y0 - (np.cos(theta) - math.cos(hdg)) / curvature
    s = s0 + s_local
    h = theta
    return s, x, y, h


def _sample_spiral_with_heading(
    x0: float,
    y0: float,
    hdg: float,
    length: float,
    curv_start: float,
    curv_end: float,
    step: float,
    s0: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    step = max(step, 0.1)
    n = max(1, int(math.ceil(length / step)))
    ds = length / n
    s_vals = np.zeros(n + 1, dtype=np.float64)
    x_vals = np.zeros(n + 1, dtype=np.float64)
    y_vals = np.zeros(n + 1, dtype=np.float64)
    h_vals = np.zeros(n + 1, dtype=np.float64)
    x = x0
    y = y0
    theta = hdg
    s_vals[0] = s0
    x_vals[0] = x0
    y_vals[0] = y0
    h_vals[0] = hdg
    for i in range(n):
        s_mid = (i + 0.5) * ds
        curv = curv_start + (curv_end - curv_start) * (s_mid / length)
        theta_mid = theta + 0.5 * curv * ds
        x += ds * math.cos(theta_mid)
        y += ds * math.sin(theta_mid)
        theta += curv * ds
        s_vals[i + 1] = s0 + (i + 1) * ds
        x_vals[i + 1] = x
        y_vals[i + 1] = y
        h_vals[i + 1] = theta
    return s_vals, x_vals, y_vals, h_vals


def _sample_lane_points(
    road_elem: ET.Element,
    ref_samples: Dict[str, np.ndarray],
    lane_mode: str,
    lane_types: Optional[set],
) -> np.ndarray:
    lanes_elem = road_elem.find("lanes")
    if lanes_elem is None or ref_samples["s"].size == 0:
        return np.zeros((0, 2), dtype=np.float64)

    lane_offsets, offset_starts = _parse_lane_offsets(lanes_elem)
    lane_sections = _parse_lane_sections(lanes_elem)
    if not lane_sections:
        return np.zeros((0, 2), dtype=np.float64)
    section_starts = [section.s for section in lane_sections]

    points: List[Tuple[float, float]] = []
    s_vals = ref_samples["s"]
    x_vals = ref_samples["x"]
    y_vals = ref_samples["y"]
    h_vals = ref_samples["hdg"]

    for i in range(s_vals.size):
        s = float(s_vals[i])
        idx = bisect_right(section_starts, s) - 1
        if idx < 0:
            idx = 0
        section = lane_sections[idx]
        ds = s - section.s
        lane_offset = _eval_poly_segments(lane_offsets, offset_starts, s)

        if lane_mode in ("center", "both"):
            offsets = _lane_center_offsets(section, ds, lane_offset, lane_types)
            for off in offsets:
                points.append(_offset_point(x_vals[i], y_vals[i], h_vals[i], off))

        if lane_mode in ("boundary", "both"):
            offsets = _lane_boundary_offsets(section, ds, lane_offset, lane_types)
            for off in offsets:
                points.append(_offset_point(x_vals[i], y_vals[i], h_vals[i], off))

    if not points:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(points, dtype=np.float64)


def _lane_center_offsets(
    section: _LaneSection, ds: float, lane_offset: float, lane_types: Optional[set]
) -> List[float]:
    offsets: List[float] = []
    cum = 0.0
    for lane in section.left:
        width = _eval_lane_width(lane, ds)
        if width <= 0.0:
            continue
        if lane_types is None or lane.lane_type in lane_types:
            offsets.append(lane_offset + cum + width / 2.0)
        cum += width
    cum = 0.0
    for lane in section.right:
        width = _eval_lane_width(lane, ds)
        if width <= 0.0:
            continue
        if lane_types is None or lane.lane_type in lane_types:
            offsets.append(lane_offset - cum - width / 2.0)
        cum += width
    return offsets


def _lane_boundary_offsets(
    section: _LaneSection, ds: float, lane_offset: float, lane_types: Optional[set]
) -> List[float]:
    offsets: List[float] = []
    cum = 0.0
    for lane in section.left:
        width = _eval_lane_width(lane, ds)
        if width <= 0.0:
            continue
        cum += width
        if lane_types is None or lane.lane_type in lane_types:
            offsets.append(lane_offset + cum)
    cum = 0.0
    for lane in section.right:
        width = _eval_lane_width(lane, ds)
        if width <= 0.0:
            continue
        cum += width
        if lane_types is None or lane.lane_type in lane_types:
            offsets.append(lane_offset - cum)
    return offsets


def _offset_point(x: float, y: float, hdg: float, offset: float) -> Tuple[float, float]:
    nx = -math.sin(hdg)
    ny = math.cos(hdg)
    return x + offset * nx, y + offset * ny


def _parse_lane_offsets(lanes_elem: ET.Element) -> Tuple[List[_PolySegment], List[float]]:
    segments: List[_PolySegment] = []
    for elem in lanes_elem.findall("laneOffset"):
        segments.append(
            _PolySegment(
                s=float(elem.attrib.get("s", 0.0)),
                a=float(elem.attrib.get("a", 0.0)),
                b=float(elem.attrib.get("b", 0.0)),
                c=float(elem.attrib.get("c", 0.0)),
                d=float(elem.attrib.get("d", 0.0)),
            )
        )
    segments.sort(key=lambda seg: seg.s)
    starts = [seg.s for seg in segments]
    return segments, starts


def _parse_lane_sections(lanes_elem: ET.Element) -> List[_LaneSection]:
    sections: List[_LaneSection] = []
    for section in lanes_elem.findall("laneSection"):
        s = float(section.attrib.get("s", 0.0))
        left_lanes = _parse_lanes(section.find("left"), side="left")
        right_lanes = _parse_lanes(section.find("right"), side="right")
        sections.append(_LaneSection(s=s, left=left_lanes, right=right_lanes))
    sections.sort(key=lambda sec: sec.s)
    return sections


def _parse_lanes(container: Optional[ET.Element], side: str) -> List[_Lane]:
    lanes: List[_Lane] = []
    if container is None:
        return lanes
    for lane_elem in container.findall("lane"):
        lane_id = int(lane_elem.attrib.get("id", "0"))
        if lane_id == 0:
            continue
        lane_type = lane_elem.attrib.get("type", "unknown").lower()
        width_segments = _parse_lane_widths(lane_elem)
        width_starts = [seg.s for seg in width_segments]
        lanes.append(
            _Lane(
                lane_id=lane_id,
                lane_type=lane_type,
                width_segments=width_segments,
                width_starts=width_starts,
            )
        )
    if side == "left":
        lanes.sort(key=lambda lane: lane.lane_id)
    else:
        lanes.sort(key=lambda lane: lane.lane_id, reverse=True)
    return lanes


def _parse_lane_widths(lane_elem: ET.Element) -> List[_PolySegment]:
    widths: List[_PolySegment] = []
    for width in lane_elem.findall("width"):
        widths.append(
            _PolySegment(
                s=float(width.attrib.get("sOffset", 0.0)),
                a=float(width.attrib.get("a", 0.0)),
                b=float(width.attrib.get("b", 0.0)),
                c=float(width.attrib.get("c", 0.0)),
                d=float(width.attrib.get("d", 0.0)),
            )
        )
    widths.sort(key=lambda seg: seg.s)
    return widths


def _eval_poly_segments(segments: List[_PolySegment], starts: List[float], s: float) -> float:
    if not segments:
        return 0.0
    idx = bisect_right(starts, s) - 1
    if idx < 0:
        idx = 0
    seg = segments[idx]
    ds = s - seg.s
    return seg.a + seg.b * ds + seg.c * ds ** 2 + seg.d * ds ** 3


def _eval_lane_width(lane: _Lane, ds: float) -> float:
    if not lane.width_segments:
        return 0.0
    idx = bisect_right(lane.width_starts, ds) - 1
    if idx < 0:
        idx = 0
    seg = lane.width_segments[idx]
    ds_local = ds - seg.s
    width = seg.a + seg.b * ds_local + seg.c * ds_local ** 2 + seg.d * ds_local ** 3
    return max(0.0, float(width))


def _parse_lane_types(value: str) -> Optional[set]:
    cleaned = value.strip().lower()
    if not cleaned or cleaned in {"all", "*"}:
        return None
    return {item.strip().lower() for item in cleaned.split(",") if item.strip()}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.floating, np.integer))


def _as_point(obj: Any) -> Optional[Tuple[float, float]]:
    if hasattr(obj, "x") and hasattr(obj, "y"):
        try:
            return float(obj.x), float(obj.y)
        except Exception:
            return None
    if isinstance(obj, dict) and "x" in obj and "y" in obj:
        if _is_number(obj["x"]) and _is_number(obj["y"]):
            return float(obj["x"]), float(obj["y"])
    if isinstance(obj, (list, tuple)) and len(obj) >= 2 and all(_is_number(v) for v in obj[:2]):
        return float(obj[0]), float(obj[1])
    if isinstance(obj, np.ndarray) and obj.ndim == 1 and obj.shape[0] >= 2:
        return float(obj[0]), float(obj[1])
    return None


def _convert_polyline(points: List[Tuple[float, float]]) -> Optional[np.ndarray]:
    if len(points) < 2:
        return None
    return np.asarray(points, dtype=np.float64)


def _extract_geometry(
    obj: Any,
    path_tokens: List[str],
    polylines: List[np.ndarray],
    points: List[Tuple[float, float]],
    visited: set,
    max_depth: int,
    depth: int,
) -> None:
    if depth > max_depth:
        return
    oid = id(obj)
    if oid in visited:
        return
    visited.add(oid)

    if isinstance(obj, np.ndarray):
        if obj.ndim == 2 and obj.shape[1] >= 2:
            poly = obj[:, :2].astype(np.float64, copy=False)
            if poly.shape[0] >= 2:
                polylines.append(poly)
            elif poly.shape[0] == 1:
                points.append((float(poly[0, 0]), float(poly[0, 1])))
        return

    if hasattr(obj, "x") and hasattr(obj, "y"):
        pt = _as_point(obj)
        if pt is not None:
            points.append(pt)
            return

    if isinstance(obj, dict):
        if "x" in obj and "y" in obj and _is_number(obj.get("x")) and _is_number(obj.get("y")):
            points.append((float(obj["x"]), float(obj["y"])))
            return
        if "x" in obj and "y" in obj and not _is_number(obj.get("x")):
            xs = np.asarray(obj.get("x"))
            ys = np.asarray(obj.get("y"))
            if xs.shape == ys.shape and xs.ndim == 1 and xs.size >= 2:
                poly = np.stack([xs, ys], axis=1).astype(np.float64, copy=False)
                polylines.append(poly)
                return
        for key, value in obj.items():
            _extract_geometry(value, path_tokens + [str(key)], polylines, points, visited, max_depth, depth + 1)
        return

    if isinstance(obj, (list, tuple)):
        if not obj:
            return
        point_candidates = []
        for item in obj:
            pt = _as_point(item)
            if pt is not None:
                point_candidates.append(pt)
        if len(point_candidates) >= 2 and len(point_candidates) >= int(0.7 * len(obj)):
            poly = _convert_polyline(point_candidates)
            if poly is not None:
                polylines.append(poly)
                return
        for idx, item in enumerate(obj):
            _extract_geometry(item, path_tokens + [str(idx)], polylines, points, visited, max_depth, depth + 1)
        return

    if hasattr(obj, "__dict__"):
        for key, value in obj.__dict__.items():
            _extract_geometry(value, path_tokens + [str(key)], polylines, points, visited, max_depth, depth + 1)


def _load_pkl_geometry(path: Path, key: Optional[str], max_depth: int, strict_pickle: bool) -> Dict[str, Any]:
    with path.open("rb") as f:
        obj = _SafeUnpickler(f, strict=strict_pickle).load()

    root = obj
    if key:
        root = _select_subtree(obj, key)

    polylines: List[np.ndarray] = []
    points: List[Tuple[float, float]] = []
    _extract_geometry(root, [], polylines, points, set(), max_depth=max_depth, depth=0)

    if points:
        polylines.append(np.asarray(points, dtype=np.float64))

    if polylines:
        merged = np.vstack(polylines)
    else:
        merged = np.zeros((0, 2), dtype=np.float64)

    return {
        "object": obj,
        "root": root,
        "polylines": polylines,
        "points": merged,
    }


def _select_subtree(obj: Any, key_path: str) -> Any:
    tokens = [t for t in key_path.replace("\\", "/").split("/") if t]
    cur = obj
    for tok in tokens:
        if isinstance(cur, dict) and tok in cur:
            cur = cur[tok]
            continue
        if hasattr(cur, tok):
            cur = getattr(cur, tok)
            continue
        if isinstance(cur, (list, tuple)):
            try:
                idx = int(tok)
            except ValueError:
                raise KeyError(f"List index '{tok}' not valid for path '{key_path}'")
            cur = cur[idx]
            continue
        raise KeyError(f"Key '{tok}' not found for path '{key_path}'")
    return cur


def _apply_transform(points: np.ndarray, scale: float, theta: float, tx: float, ty: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return scale * (points @ rot.T) + np.array([tx, ty], dtype=np.float64)


def _score_alignment(
    tree: cKDTree,
    points: np.ndarray,
    scale: float,
    theta: float,
    tx: float,
    ty: float,
    trim_fraction: float,
    inlier_thresh: float,
    min_inlier_ratio: float,
    inlier_penalty: float,
) -> Tuple[float, float, float]:
    transformed = _apply_transform(points, scale, theta, tx, ty)
    dists, _ = tree.query(transformed, k=1)
    if dists.size == 0:
        return float("inf"), 0.0, float("inf")
    dists = np.asarray(dists)
    n_keep = max(1, int((1.0 - trim_fraction) * dists.size))
    trimmed = np.partition(dists, n_keep - 1)[:n_keep]
    trimmed_mean = float(trimmed.mean())
    inlier_ratio = float((dists < inlier_thresh).mean())
    penalty = max(0.0, min_inlier_ratio - inlier_ratio) * inlier_penalty
    return trimmed_mean + penalty, inlier_ratio, trimmed_mean


def _grid_search_centered(
    tree: cKDTree,
    points: np.ndarray,
    scales: Sequence[float],
    theta_center_deg: float,
    theta_range_deg: float,
    theta_step_deg: float,
    tx_center: float,
    ty_center: float,
    trans_range_m: float,
    trans_step_m: float,
    trim_fraction: float,
    inlier_thresh: float,
    min_inlier_ratio: float,
    inlier_penalty: float,
    progress_label: str,
    log_interval_s: float,
) -> Tuple[float, float, float, float, float]:
    best = (float("inf"), 1.0, 0.0, 0.0, 0.0)  # score, scale, theta, tx, ty
    theta_vals = _frange(theta_center_deg - theta_range_deg, theta_center_deg + theta_range_deg, theta_step_deg)
    trans_x_vals = _frange(tx_center - trans_range_m, tx_center + trans_range_m, trans_step_m)
    trans_y_vals = _frange(ty_center - trans_range_m, ty_center + trans_range_m, trans_step_m)
    total = max(1, len(scales) * len(theta_vals) * len(trans_x_vals) * len(trans_y_vals))
    start = time.monotonic()
    last_log = start
    count = 0
    if log_interval_s > 0.0:
        _log(f"{progress_label} grid search evals={total}")

    for scale in scales:
        for theta_deg in theta_vals:
            theta = math.radians(theta_deg)
            for tx in trans_x_vals:
                for ty in trans_y_vals:
                    count += 1
                    score, inlier_ratio, _ = _score_alignment(
                        tree,
                        points,
                        scale,
                        theta,
                        tx,
                        ty,
                        trim_fraction,
                        inlier_thresh,
                        min_inlier_ratio,
                        inlier_penalty,
                    )
                    if score < best[0]:
                        best = (score, scale, theta, tx, ty)
                    if log_interval_s > 0.0 and count % 1000 == 0:
                        now = time.monotonic()
                        if now - last_log >= log_interval_s:
                            rate = count / max(now - start, 1e-6)
                            remaining = (total - count) / max(rate, 1e-6)
                            _log(
                                f"{progress_label} progress {count}/{total} "
                                f"({100.0 * count / total:.1f}%) "
                                f"eta={remaining:.1f}s"
                            )
                            last_log = now
    if log_interval_s > 0.0:
        elapsed = time.monotonic() - start
        _log(f"{progress_label} grid search done in {elapsed:.1f}s")
    return best


def _frange(start: float, stop: float, step: float) -> List[float]:
    if step <= 0:
        return [start]
    vals = []
    v = start
    while v <= stop + 1e-9:
        vals.append(v)
        v += step
    return vals


def _umeyama_2d(src: np.ndarray, dst: np.ndarray, allow_scale: bool) -> Tuple[float, np.ndarray, np.ndarray]:
    if src.shape[0] < 2:
        return 1.0, np.eye(2), np.zeros(2)
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    cov = dst_c.T @ src_c / src.shape[0]
    u, s, vt = np.linalg.svd(cov)
    d = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[-1, -1] = -1
    r = u @ d @ vt
    if allow_scale:
        var_src = (src_c ** 2).sum() / src.shape[0]
        scale = float(np.trace(np.diag(s) @ d) / max(var_src, 1e-12))
    else:
        scale = 1.0
    t = mu_dst - scale * (r @ mu_src)
    return scale, r, t


def _compose_transform(
    s1: float,
    theta1: float,
    t1: np.ndarray,
    s2: float,
    theta2: float,
    t2: np.ndarray,
) -> Tuple[float, float, np.ndarray]:
    r1 = _rot_from_theta(theta1)
    r2 = _rot_from_theta(theta2)
    r = r2 @ r1
    s = s2 * s1
    t = s2 * (r2 @ t1) + t2
    theta = math.atan2(r[1, 0], r[0, 0])
    return s, theta, t


def _rot_from_theta(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _icp_refine(
    tree: cKDTree,
    points: np.ndarray,
    init_scale: float,
    init_theta: float,
    init_tx: float,
    init_ty: float,
    allow_scale: bool,
    max_iters: int,
    icp_thresh: float,
    trim_fraction: float,
    tol: float,
) -> Tuple[float, float, float, float, float]:
    scale = init_scale
    theta = init_theta
    t = np.array([init_tx, init_ty], dtype=np.float64)
    prev_score = float("inf")

    for _ in range(max_iters):
        transformed = _apply_transform(points, scale, theta, t[0], t[1])
        dists, idx = tree.query(transformed, k=1)
        dists = np.asarray(dists)
        idx = np.asarray(idx)
        inliers = dists < icp_thresh
        if inliers.sum() < 5:
            break

        src = transformed[inliers]
        dst = tree.data[idx[inliers]]

        if trim_fraction > 0.0:
            n_keep = max(1, int((1.0 - trim_fraction) * src.shape[0]))
            keep_idx = np.argpartition(dists[inliers], n_keep - 1)[:n_keep]
            src = src[keep_idx]
            dst = dst[keep_idx]

        s2, r2, t2 = _umeyama_2d(src, dst, allow_scale)
        theta2 = math.atan2(r2[1, 0], r2[0, 0])
        scale, theta, t = _compose_transform(scale, theta, t, s2, theta2, t2)

        score, _, trimmed_mean = _score_alignment(
            tree,
            points,
            scale,
            theta,
            t[0],
            t[1],
            trim_fraction=trim_fraction,
            inlier_thresh=icp_thresh,
            min_inlier_ratio=0.0,
            inlier_penalty=0.0,
        )
        if prev_score - score < tol:
            prev_score = score
            break
        prev_score = score

    final_score, inlier_ratio, trimmed_mean = _score_alignment(
        tree,
        points,
        scale,
        theta,
        t[0],
        t[1],
        trim_fraction=trim_fraction,
        inlier_thresh=icp_thresh,
        min_inlier_ratio=0.0,
        inlier_penalty=0.0,
    )
    return scale, theta, t[0], t[1], final_score


def _residual_stats(tree: cKDTree, points: np.ndarray, scale: float, theta: float, tx: float, ty: float) -> Dict[str, float]:
    transformed = _apply_transform(points, scale, theta, tx, ty)
    dists, _ = tree.query(transformed, k=1)
    dists = np.asarray(dists)
    if dists.size == 0:
        return {
            "median": float("inf"),
            "mean_trimmed": float("inf"),
            "p90": float("inf"),
            "inlier_ratio": 0.0,
        }
    d_sorted = np.sort(dists)
    n_keep = max(1, int(0.7 * d_sorted.size))
    trimmed = d_sorted[:n_keep]
    return {
        "median": float(np.median(dists)),
        "mean_trimmed": float(trimmed.mean()),
        "p90": float(np.percentile(dists, 90)),
        "inlier_ratio": float((dists < 2.5).mean()),
    }


@dataclass
class AlignmentResult:
    scale: float
    theta: float
    tx: float
    ty: float
    axis_variant: str
    score: float
    residuals: Dict[str, float]


def _align_point_cloud(
    pkl_points: np.ndarray,
    map_points: np.ndarray,
    score_points: np.ndarray,
    allow_scale: bool,
    scales: Sequence[float],
    theta_range_deg: float,
    theta_step_deg: float,
    trans_range_m: float,
    trans_step_m: float,
    refine_factor: float,
    trim_fraction: float,
    inlier_thresh: float,
    min_inlier_ratio: float,
    inlier_penalty: float,
    variant_topk: int,
    full_search: bool,
    log_interval_s: float,
    icp_iters: int,
    icp_thresh: float,
    icp_trim_fraction: float,
    icp_tol: float,
) -> AlignmentResult:
    tree = cKDTree(map_points)
    best_result: Optional[AlignmentResult] = None
    map_centroid, map_theta, map_rms, map_ratio = _pca_stats(map_points)

    variant_candidates: List[Tuple[str, float, float, float, float]] = []
    for variant_name, variant_fn in AXIS_VARIANTS.items():
        variant_score_points = variant_fn(score_points)
        if full_search:
            variant_candidates.append((variant_name, 1.0, 0.0, 0.0, 0.0))
            continue

        centroid, theta, rms, ratio = _pca_stats(variant_score_points)
        base_scale = map_rms / max(rms, 1e-12) if allow_scale else 1.0
        theta_delta = map_theta - theta
        candidates = [theta_delta, theta_delta + math.pi]
        if ratio < 1.2 or map_ratio < 1.2:
            candidates.extend([theta_delta + math.pi / 2.0, theta_delta - math.pi / 2.0])

        candidate_score = float("inf")
        candidate_theta = theta_delta
        candidate_tx = 0.0
        candidate_ty = 0.0
        for cand_theta in candidates:
            rot = _rot_from_theta(cand_theta)
            t = map_centroid - base_scale * (rot @ centroid)
            score, _, _ = _score_alignment(
                tree,
                variant_score_points,
                base_scale,
                cand_theta,
                float(t[0]),
                float(t[1]),
                trim_fraction,
                inlier_thresh,
                min_inlier_ratio,
                inlier_penalty,
            )
            if score < candidate_score:
                candidate_score = score
                candidate_theta = cand_theta
                candidate_tx = float(t[0])
                candidate_ty = float(t[1])

        variant_candidates.append((variant_name, base_scale, candidate_theta, candidate_tx, candidate_ty))

    if full_search:
        ranked_variants = list(AXIS_VARIANTS.keys())
    else:
        scored_variants: List[Tuple[str, float]] = []
        for variant_name, scale, theta, tx, ty in variant_candidates:
            variant_score_points = AXIS_VARIANTS[variant_name](score_points)
            score, _, _ = _score_alignment(
                tree,
                variant_score_points,
                scale,
                theta,
                tx,
                ty,
                trim_fraction,
                inlier_thresh,
                min_inlier_ratio,
                inlier_penalty,
            )
            scored_variants.append((variant_name, score))
        scored_variants.sort(key=lambda x: x[1])
        ranked_variants = [name for name, _ in scored_variants[: max(1, variant_topk)]]

    for variant_name in ranked_variants:
        variant_points = AXIS_VARIANTS[variant_name](pkl_points)
        variant_score_points = AXIS_VARIANTS[variant_name](score_points)
        _log(f"Variant {variant_name}: starting coarse search")
        if full_search:
            init_scale, init_theta, init_tx, init_ty = 1.0, 0.0, 0.0, 0.0
        else:
            init_scale, init_theta, init_tx, init_ty = next(
                (s, t, x, y)
                for name, s, t, x, y in variant_candidates
                if name == variant_name
            )

        best = _grid_search_centered(
            tree,
            variant_score_points,
            scales if full_search else _local_scales(init_scale, allow_scale),
            math.degrees(init_theta),
            theta_range_deg,
            theta_step_deg,
            init_tx,
            init_ty,
            trans_range_m,
            trans_step_m,
            trim_fraction,
            inlier_thresh,
            min_inlier_ratio,
            inlier_penalty,
            progress_label=f"{variant_name} coarse",
            log_interval_s=log_interval_s,
        )
        _, scale, theta, tx, ty = best

        refine_theta_step = theta_step_deg / max(refine_factor, 1.0)
        refine_trans_step = trans_step_m / max(refine_factor, 1.0)
        refine_theta_range = max(theta_step_deg * 2.0, refine_theta_step)
        refine_trans_range = max(trans_step_m * 2.0, refine_trans_step)
        _log(f"Variant {variant_name}: starting refine search")
        best_refined = _grid_search_centered(
            tree,
            variant_score_points,
            [scale] if not allow_scale else _local_scales(scale, allow_scale),
            math.degrees(theta),
            refine_theta_range,
            refine_theta_step,
            tx,
            ty,
            refine_trans_range,
            refine_trans_step,
            trim_fraction,
            inlier_thresh,
            min_inlier_ratio,
            inlier_penalty,
            progress_label=f"{variant_name} refine",
            log_interval_s=log_interval_s,
        )

        _, scale, theta, tx, ty = best_refined

        _log(f"Variant {variant_name}: ICP refine")
        scale, theta, tx, ty, score = _icp_refine(
            tree,
            variant_points,
            scale,
            theta,
            tx,
            ty,
            allow_scale=allow_scale,
            max_iters=icp_iters,
            icp_thresh=icp_thresh,
            trim_fraction=icp_trim_fraction,
            tol=icp_tol,
        )

        residuals = _residual_stats(tree, variant_points, scale, theta, tx, ty)
        _log(
            f"Variant {variant_name}: score={score:.3f} "
            f"median={residuals['median']:.2f} p90={residuals['p90']:.2f}"
        )
        result = AlignmentResult(
            scale=scale,
            theta=theta,
            tx=tx,
            ty=ty,
            axis_variant=variant_name,
            score=score,
            residuals=residuals,
        )
        if best_result is None or result.score < best_result.score:
            best_result = result

    if best_result is None:
        raise RuntimeError("Alignment failed: no candidates evaluated.")
    return best_result


def _apply_variant_and_transform(
    polylines: List[np.ndarray],
    variant_name: str,
    scale: float,
    theta: float,
    tx: float,
    ty: float,
) -> List[np.ndarray]:
    variant_fn = AXIS_VARIANTS[variant_name]
    aligned = []
    for poly in polylines:
        pts = variant_fn(poly)
        aligned.append(_apply_transform(pts, scale, theta, tx, ty))
    return aligned


def _is_simple_structure(obj: Any, depth: int = 0, max_depth: int = 6) -> bool:
    if depth > max_depth:
        return False
    if obj is None or isinstance(obj, (int, float, str, bool, np.ndarray)):
        return True
    if isinstance(obj, (list, tuple)):
        return all(_is_simple_structure(v, depth + 1, max_depth) for v in obj)
    if isinstance(obj, dict):
        return all(_is_simple_structure(v, depth + 1, max_depth) for v in obj.values())
    return False


def _write_outputs(
    out_dir: Path,
    pkl_path: Path,
    original_obj: Any,
    polylines: List[np.ndarray],
    aligned_polylines: List[np.ndarray],
    result: AlignmentResult,
    residuals: Dict[str, float],
    plot: bool,
    map_points: np.ndarray,
    raw_points: np.ndarray,
    seed: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    transform = {
        "scale": result.scale,
        "theta_rad": result.theta,
        "tx": result.tx,
        "ty": result.ty,
        "axis_variant": result.axis_variant,
        "residuals": residuals,
        "score": result.score,
        "source_pkl": str(pkl_path),
    }
    with (out_dir / "transform.json").open("w", encoding="utf-8") as f:
        json.dump(transform, f, indent=2, sort_keys=True)

    aligned_points = np.vstack(aligned_polylines) if aligned_polylines else np.zeros((0, 2))
    if isinstance(original_obj, list) and all(isinstance(p, np.ndarray) for p in polylines):
        output_obj: Any = aligned_polylines
    elif isinstance(original_obj, dict) and _is_simple_structure(original_obj):
        output_obj = dict(original_obj)
        output_obj["aligned_polylines"] = aligned_polylines
        output_obj["aligned_points"] = aligned_points
        output_obj["alignment_meta"] = transform
    else:
        output_obj = {
            "aligned_polylines": aligned_polylines,
            "aligned_points": aligned_points,
            "alignment_meta": transform,
        }
    with (out_dir / "aligned_network.pkl").open("wb") as f:
        pickle.dump(output_obj, f)

    if plot:
        import matplotlib.pyplot as plt

        plot_map = _sample_points(map_points, 200000, seed)
        plot_raw = _sample_points(raw_points, 100000, seed)
        plot_aligned = _sample_points(aligned_points, 100000, seed)

        plt.figure(figsize=(10, 10))
        if plot_map.size:
            plt.scatter(plot_map[:, 0], plot_map[:, 1], s=1, c="#1f77b4", alpha=0.35, label="XODR")
        if plot_raw.size:
            plt.scatter(plot_raw[:, 0], plot_raw[:, 1], s=2, c="#d62728", alpha=0.4, label="PKL (raw)")
        if plot_aligned.size:
            plt.scatter(plot_aligned[:, 0], plot_aligned[:, 1], s=2, c="#2ca02c", alpha=0.6, label="PKL (aligned)")
        plt.axis("equal")
        plt.legend(loc="best")
        plt.title(pkl_path.name)
        plt.tight_layout()
        plt.savefig(out_dir / "debug_overlay.png", dpi=200)


def _run_self_test(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(13)
    base = np.stack([np.linspace(0, 200, 200), 20 * np.sin(np.linspace(0, 6, 200))], axis=1)
    offset = np.array([[50, 80], [60, 100], [80, 130]], dtype=np.float64)
    polylines = [base + off for off in offset]
    map_points = np.vstack(polylines)

    true_scale = 1.02 if args.allow_scale else 1.0
    true_theta = math.radians(17.0)
    true_t = np.array([120.0, -45.0])
    r_inv = _rot_from_theta(-true_theta)
    pkl_points = (map_points - true_t) @ r_inv.T / true_scale

    pkl_points = _sample_points(pkl_points, args.max_points, args.seed)
    map_points = _sample_points(map_points, args.max_xodr_points, args.seed)
    score_points = _sample_points(pkl_points, args.score_points, args.seed)

    theta_deg = args.theta_deg if args.theta_deg is not None else 90.0
    theta_step_deg = args.theta_step_deg if args.theta_step_deg is not None else 15.0
    trans_m = args.trans_m if args.trans_m is not None else 100.0
    trans_step_m = args.trans_step_m if args.trans_step_m is not None else 20.0

    result = _align_point_cloud(
        pkl_points,
        map_points,
        score_points,
        allow_scale=args.allow_scale,
        scales=_build_scales(args.allow_scale),
        theta_range_deg=theta_deg,
        theta_step_deg=theta_step_deg,
        trans_range_m=trans_m,
        trans_step_m=trans_step_m,
        refine_factor=args.refine_factor,
        trim_fraction=args.trim_fraction,
        inlier_thresh=args.inlier_thresh,
        min_inlier_ratio=args.min_inlier_ratio,
        inlier_penalty=args.inlier_penalty,
        variant_topk=args.variant_topk,
        full_search=args.full_search,
        log_interval_s=args.log_interval_s,
        icp_iters=args.icp_iters,
        icp_thresh=args.icp_thresh,
        icp_trim_fraction=args.icp_trim_fraction,
        icp_tol=args.icp_tol,
    )
    print("Self-test result:")
    print(f"  scale={result.scale:.4f} theta_deg={math.degrees(result.theta):.2f} tx={result.tx:.2f} ty={result.ty:.2f}")
    print(f"  residuals={result.residuals}")


def _build_scales(allow_scale: bool) -> List[float]:
    if not allow_scale:
        return [1.0]
    return [0.98, 0.99, 1.0, 1.01, 1.02]


def _local_scales(center: float, allow_scale: bool) -> List[float]:
    if not allow_scale:
        return [1.0]
    scales = [center * f for f in (0.98, 0.99, 1.0, 1.01, 1.02)]
    return [s for s in scales if s > 0.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xodr", required=False, help="Path to OpenDRIVE XODR file.")
    parser.add_argument("--pkls", nargs="+", required=False, help="PKL files or directories to scan.")
    parser.add_argument("--out_dir", default="alignment_outputs", help="Output directory.")
    parser.add_argument("--xodr_step", type=float, default=1.0, help="Sampling step for XODR (meters).")
    parser.add_argument("--approx_spiral", action="store_true", help="Approximate spiral geometry instead of skipping.")
    parser.add_argument(
        "--lane_mode",
        choices=["reference", "center", "boundary", "both"],
        default="center",
        help="XODR lane sampling mode (reference adds no lane offsets).",
    )
    parser.add_argument(
        "--lane_types",
        default="all",
        help="Comma-separated lane types to include (e.g., driving,shoulder) or 'all'.",
    )
    parser.add_argument("--max_points", type=int, default=20000, help="Max PKL points for alignment.")
    parser.add_argument("--score_points", type=int, default=5000, help="Max points used for scoring.")
    parser.add_argument("--max_xodr_points", type=int, default=100000, help="Max XODR points for alignment.")
    parser.add_argument("--key", default="", help="Optional subtree key path for PKL geometry extraction.")
    parser.add_argument("--max_depth", type=int, default=8, help="Max recursion depth for PKL extraction.")
    parser.add_argument("--allow_scale", action="store_true", help="Allow similarity scaling.")
    parser.add_argument("--theta_deg", type=float, default=None, help="Theta search range in degrees (auto if omitted).")
    parser.add_argument("--theta_step_deg", type=float, default=None, help="Theta grid step in degrees (auto if omitted).")
    parser.add_argument("--trans_m", type=float, default=None, help="Translation search range in meters (auto if omitted).")
    parser.add_argument("--trans_step_m", type=float, default=None, help="Translation grid step in meters (auto if omitted).")
    parser.add_argument("--refine_factor", type=float, default=3.0, help="Refinement factor for grid steps.")
    parser.add_argument("--trim_fraction", type=float, default=0.3, help="Trim fraction for coarse scoring.")
    parser.add_argument("--inlier_thresh", type=float, default=2.5, help="Inlier threshold for scoring.")
    parser.add_argument("--min_inlier_ratio", type=float, default=0.3, help="Minimum inlier ratio before penalty.")
    parser.add_argument("--inlier_penalty", type=float, default=10.0, help="Penalty weight for low inliers.")
    parser.add_argument("--variant_topk", type=int, default=4, help="Number of axis variants to refine.")
    parser.add_argument("--full_search", action="store_true", help="Run full grid search for all variants.")
    parser.add_argument("--auto_search", dest="auto_search", action="store_true", default=True, help="Auto-tune search ranges.")
    parser.add_argument("--no_auto_search", dest="auto_search", action="store_false", help="Disable auto search tuning.")
    parser.add_argument("--auto_retry", dest="auto_retry", action="store_true", default=True, help="Retry with broader search if fit is poor.")
    parser.add_argument("--no_auto_retry", dest="auto_retry", action="store_false", help="Disable auto retry.")
    parser.add_argument("--retry_median_thresh", type=float, default=10.0, help="Median error threshold for retry.")
    parser.add_argument("--retry_inlier_thresh", type=float, default=0.2, help="Inlier ratio threshold for retry.")
    parser.add_argument("--icp_iters", type=int, default=10, help="ICP iterations.")
    parser.add_argument("--icp_thresh", type=float, default=3.0, help="ICP inlier threshold.")
    parser.add_argument("--icp_trim_fraction", type=float, default=0.2, help="ICP trim fraction.")
    parser.add_argument("--icp_tol", type=float, default=1e-3, help="ICP convergence tolerance.")
    parser.add_argument("--plot", action="store_true", help="Write debug overlay PNG.")
    parser.add_argument("--strict_pickle", action="store_true", help="Require pickle classes to resolve normally.")
    parser.add_argument("--self_test", action="store_true", help="Run synthetic self-test and exit.")
    parser.add_argument("--log_interval_s", type=float, default=10.0, help="Progress log interval in seconds.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for subsampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        _run_self_test(args)
        return

    if not args.xodr or not args.pkls:
        raise SystemExit("--xodr and --pkls are required unless --self_test is set.")

    xodr_path = Path(args.xodr)
    if not xodr_path.exists():
        raise FileNotFoundError(xodr_path)

    pkl_paths: List[Path] = []
    for entry in args.pkls:
        p = Path(entry)
        if p.is_dir():
            pkl_paths.extend(sorted(p.rglob("*.pkl")))
        else:
            pkl_paths.append(p)

    if not pkl_paths:
        raise SystemExit("No PKL files found.")

    lane_types = _parse_lane_types(args.lane_types)
    if args.lane_mode != "reference":
        lane_label = "all" if lane_types is None else ",".join(sorted(lane_types))
        _log(f"XODR lanes: mode={args.lane_mode}, types={lane_label}")
    map_points, spiral_skipped = _parse_xodr(
        xodr_path,
        args.xodr_step,
        args.approx_spiral,
        args.lane_mode,
        lane_types,
    )
    if spiral_skipped:
        print(f"[WARN] Skipped {spiral_skipped} spiral segments in XODR.")
    if map_points.size == 0:
        raise SystemExit("No map points parsed from XODR.")

    map_points = _sample_points(map_points, args.max_xodr_points, args.seed)
    _log(f"XODR points: {map_points.shape[0]}")

    scales = _build_scales(args.allow_scale)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for pkl_path in pkl_paths:
        if not pkl_path.exists():
            print(f"[WARN] Missing PKL: {pkl_path}")
            continue

        _log(f"Processing {pkl_path}")
        geom = _load_pkl_geometry(
            pkl_path,
            key=args.key or None,
            max_depth=args.max_depth,
            strict_pickle=args.strict_pickle,
        )
        pkl_points = geom["points"]
        if pkl_points.size == 0:
            _log(f"[WARN] No geometry extracted from {pkl_path}")
            continue

        pkl_sample = _sample_points(pkl_points, args.max_points, args.seed)
        _log(f"PKL points: {pkl_points.shape[0]} (sampled {pkl_sample.shape[0]})")
        score_points = (
            _sample_points(pkl_sample, args.score_points, args.seed)
            if args.score_points > 0
            else pkl_sample
        )
        _log(f"Score points: {score_points.shape[0]}")

        if args.auto_search:
            auto_theta_deg, auto_theta_step_deg, auto_trans_m, auto_trans_step_m = _auto_search_params(
                pkl_sample, map_points
            )
        else:
            auto_theta_deg, auto_theta_step_deg, auto_trans_m, auto_trans_step_m = 30.0, 10.0, 50.0, 10.0

        theta_deg = args.theta_deg if args.theta_deg is not None else auto_theta_deg
        theta_step_deg = args.theta_step_deg if args.theta_step_deg is not None else auto_theta_step_deg
        trans_m = args.trans_m if args.trans_m is not None else auto_trans_m
        trans_step_m = args.trans_step_m if args.trans_step_m is not None else auto_trans_step_m

        _log(
            f"Search params: theta_deg={theta_deg:.1f}, theta_step_deg={theta_step_deg:.1f}, "
            f"trans_m={trans_m:.1f}, trans_step_m={trans_step_m:.1f}"
        )
        result = _align_point_cloud(
            pkl_sample,
            map_points,
            score_points,
            allow_scale=args.allow_scale,
            scales=scales,
            theta_range_deg=theta_deg,
            theta_step_deg=theta_step_deg,
            trans_range_m=trans_m,
            trans_step_m=trans_step_m,
            refine_factor=args.refine_factor,
            trim_fraction=args.trim_fraction,
            inlier_thresh=args.inlier_thresh,
            min_inlier_ratio=args.min_inlier_ratio,
            inlier_penalty=args.inlier_penalty,
            variant_topk=args.variant_topk,
            full_search=args.full_search,
            log_interval_s=args.log_interval_s,
            icp_iters=args.icp_iters,
            icp_thresh=args.icp_thresh,
            icp_trim_fraction=args.icp_trim_fraction,
            icp_tol=args.icp_tol,
        )

        residuals = _residual_stats(
            cKDTree(map_points), pkl_sample, result.scale, result.theta, result.tx, result.ty
        )

        if args.auto_retry:
            if residuals["median"] > args.retry_median_thresh or residuals["inlier_ratio"] < args.retry_inlier_thresh:
                _log("Alignment looks poor; retrying with broader search.")
                retry_theta_deg = max(theta_deg, 180.0)
                retry_trans_m = max(trans_m, auto_trans_m * 1.5)
                retry_theta_step = max(theta_step_deg, retry_theta_deg / 8.0)
                retry_trans_step = max(trans_step_m, retry_trans_m / 12.0)
                _log(
                    f"Retry params: theta_deg={retry_theta_deg:.1f}, theta_step_deg={retry_theta_step:.1f}, "
                    f"trans_m={retry_trans_m:.1f}, trans_step_m={retry_trans_step:.1f}"
                )
                retry_result = _align_point_cloud(
                    pkl_sample,
                    map_points,
                    score_points,
                    allow_scale=args.allow_scale,
                    scales=scales,
                    theta_range_deg=retry_theta_deg,
                    theta_step_deg=retry_theta_step,
                    trans_range_m=retry_trans_m,
                    trans_step_m=retry_trans_step,
                    refine_factor=args.refine_factor,
                    trim_fraction=args.trim_fraction,
                    inlier_thresh=args.inlier_thresh,
                    min_inlier_ratio=args.min_inlier_ratio,
                    inlier_penalty=args.inlier_penalty,
                    variant_topk=max(args.variant_topk, 6),
                    full_search=args.full_search,
                    log_interval_s=args.log_interval_s,
                    icp_iters=args.icp_iters,
                    icp_thresh=args.icp_thresh,
                    icp_trim_fraction=args.icp_trim_fraction,
                    icp_tol=args.icp_tol,
                )
                retry_residuals = _residual_stats(
                    cKDTree(map_points),
                    pkl_sample,
                    retry_result.scale,
                    retry_result.theta,
                    retry_result.tx,
                    retry_result.ty,
                )
                if retry_residuals["median"] < residuals["median"]:
                    result = retry_result
                    residuals = retry_residuals

        aligned_polylines = _apply_variant_and_transform(
            geom["polylines"], result.axis_variant, result.scale, result.theta, result.tx, result.ty
        )

        out_dir = out_root / pkl_path.stem
        _write_outputs(
            out_dir,
            pkl_path,
            geom["object"],
            geom["polylines"],
            aligned_polylines,
            result,
            residuals,
            plot=args.plot,
            map_points=map_points,
            raw_points=pkl_sample,
            seed=args.seed,
        )
        _log(
            f"Done {pkl_path.name}: scale={result.scale:.4f}, "
            f"theta_deg={math.degrees(result.theta):.2f}, tx={result.tx:.2f}, ty={result.ty:.2f}"
        )


if __name__ == "__main__":
    main()
