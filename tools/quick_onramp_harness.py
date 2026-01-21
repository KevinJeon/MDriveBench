#!/usr/bin/env python3
"""
Quick harness to test on-ramp crop selection and ramp-path availability without full scenario generation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scenario_generator"))

import generate_legal_paths as glp
from pipeline.step_01_crop.candidates import build_candidate_crops_for_town
from pipeline.step_01_crop.csp import solve_assignment
from pipeline.step_01_crop.features import compute_crop_features, detect_junction_centers
from pipeline.step_01_crop.models import CropKey, GeometrySpec, Scenario
from pipeline.step_01_crop.scoring import crop_satisfies_spec
from pipeline.step_01_crop.viz import save_viz
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def _parse_radii(text: str) -> List[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def _parse_crop(text: str) -> CropKey:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError("crop must be xmin,xmax,ymin,ymax")
    xmin, xmax, ymin, ymax = [float(p) for p in parts]
    return CropKey(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)


def _build_geometry_spec() -> GeometrySpec:
    return GeometrySpec(
        topology="highway",
        degree=0,
        required_maneuvers={"straight": 1, "left": 0, "right": 0},
        needs_oncoming=False,
        needs_merge_onto_same_road=True,
        needs_on_ramp=True,
        needs_multi_lane=True,
        min_lane_count=3,
        min_entry_runup_m=28.0,
        min_exit_runout_m=18.0,
        preferred_entry_cardinals=[],
        avoid_extra_intersections=True,
        confidence=1.0,
        notes="quick_onramp_harness",
    )


def _candidate_entry_road_id(cand: Dict[str, object]) -> Optional[int]:
    sig = (cand or {}).get("signature", {})
    ent = (sig or {}).get("entry", {})
    rid = (ent or {}).get("road_id", None)
    try:
        return int(rid) if rid is not None else None
    except Exception:
        return None


def _candidate_exit_road_id(cand: Dict[str, object]) -> Optional[int]:
    sig = (cand or {}).get("signature", {})
    ex = (sig or {}).get("exit", {})
    rid = (ex or {}).get("road_id", None)
    try:
        return int(rid) if rid is not None else None
    except Exception:
        return None


def _candidate_entry_cardinal(cand: Dict[str, object]) -> str:
    sig = (cand or {}).get("signature", {})
    ent = (sig or {}).get("entry", {})
    return str((ent or {}).get("cardinal4", "")).strip().upper()


def _candidate_exit_cardinal(cand: Dict[str, object]) -> str:
    sig = (cand or {}).get("signature", {})
    ex = (sig or {}).get("exit", {})
    return str((ex or {}).get("cardinal4", "")).strip().upper()


def _build_road_corridors(candidates: List[Dict[str, object]]) -> Dict[int, set]:
    corridor_pairs = []
    for c in candidates:
        sig = (c or {}).get("signature", {})
        if str((sig or {}).get("entry_to_exit_turn", "")).strip().lower() == "straight":
            ent_rid = _candidate_entry_road_id(c)
            ex_rid = _candidate_exit_road_id(c)
            if ent_rid is not None and ex_rid is not None:
                corridor_pairs.append((ent_rid, ex_rid))

    parent: Dict[int, int] = {}

    def find(x: int) -> int:
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: int, b: int) -> None:
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pa] = pb

    for a, b in corridor_pairs:
        union(a, b)

    for c in candidates:
        ent_rid = _candidate_entry_road_id(c)
        ex_rid = _candidate_exit_road_id(c)
        if ent_rid is not None:
            find(ent_rid)
        if ex_rid is not None:
            find(ex_rid)

    corridors: Dict[int, set] = {}
    for rid in parent:
        root = find(rid)
        corridors.setdefault(root, set()).add(rid)

    road_to_corridor: Dict[int, set] = {}
    for rid in parent:
        root = find(rid)
        road_to_corridor[rid] = corridors[root]
    return road_to_corridor


def _corridor_key(road_corridors: Dict[int, set], road_id: Optional[int]) -> Optional[frozenset]:
    if road_id is None:
        return None
    return frozenset(road_corridors.get(road_id, {road_id}))


def _pick_crop(
    town: str,
    nodes_path: Path,
    radii: List[float],
    min_path_len: float,
    max_paths: int,
    max_depth: int,
    scenario_text: str,
) -> Tuple[CropKey, Optional[object]]:
    crops = build_candidate_crops_for_town(
        town_name=town,
        town_json_path=str(nodes_path),
        radii=radii,
        min_path_len=min_path_len,
        max_paths=max_paths,
        max_depth=max_depth,
    )
    if not crops:
        raise SystemExit("No candidate crops built.")

    spec = _build_geometry_spec()
    if not any(crop_satisfies_spec(spec, c) for c in crops):
        raise SystemExit("No crops satisfy the highway on-ramp geometry spec.")

    scenario = Scenario(sid="quick_onramp", text=scenario_text)
    res = solve_assignment(
        scenarios=[scenario],
        specs={"quick_onramp": spec},
        crops=crops,
        domain_k=50,
        capacity_per_crop=10,
        reuse_weight=4000.0,
        junction_penalty=25000.0,
        log_every=0,
    )

    assignment = res.detailed.get("assignments", {}).get("quick_onramp")
    if not assignment:
        raise SystemExit("No crop assigned for quick_onramp scenario.")

    crop_vals = assignment["crop"]
    crop_key = CropKey(xmin=crop_vals[0], xmax=crop_vals[1], ymin=crop_vals[2], ymax=crop_vals[3])
    crop_lookup = {(c.town, c.crop.to_str()): c for c in crops}
    crop_feat = crop_lookup.get((town, crop_key.to_str()))
    return crop_key, crop_feat


def _use_explicit_crop(
    town: str,
    nodes_path: Path,
    crop_key: CropKey,
    min_path_len: float,
    max_paths: int,
    max_depth: int,
) -> Optional[object]:
    data = glp.load_nodes(str(nodes_path))
    segs = glp.build_segments(data)
    adj_full = glp.build_connectivity(segs)
    jcenters = detect_junction_centers(segs, adj_full)
    center_xy = ((crop_key.xmin + crop_key.xmax) / 2.0, (crop_key.ymin + crop_key.ymax) / 2.0)
    return compute_crop_features(
        town_name=town,
        segments_full=segs,
        junction_centers=jcenters,
        center_xy=center_xy,
        crop=crop_key,
        min_path_len=min_path_len,
        max_paths=max_paths,
        max_depth=max_depth,
    )


def _build_candidates(legal_paths):
    candidates = []
    for i, p in enumerate(legal_paths):
        sig = glp.build_path_signature(p)
        name = glp.make_path_name(i, sig)
        candidates.append({"name": name, "signature": sig})
    return candidates


def _analyze_ramp_candidates(
    candidates: List[Dict[str, object]],
    lane_counts_by_road: Optional[Dict[int, int]] = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    road_corridors = _build_road_corridors(candidates)
    lane_count_by_corridor: Dict[frozenset, int] = {}
    if lane_counts_by_road:
        for rid_raw, count_raw in lane_counts_by_road.items():
            try:
                rid_i = int(rid_raw)
                count_i = int(count_raw)
            except Exception:
                continue
            ck = _corridor_key(road_corridors, rid_i)
            if ck is None:
                continue
            lane_count_by_corridor[ck] = max(lane_count_by_corridor.get(ck, 0), count_i)
    else:
        lane_ids_by_road: Dict[int, set] = {}
        for c in candidates:
            sig = (c or {}).get("signature", {})
            roads = sig.get("roads", [])
            lanes = sig.get("lanes", [])
            if isinstance(roads, list) and isinstance(lanes, list):
                for rid, lid in zip(roads, lanes):
                    try:
                        rid_i = int(rid)
                        lid_i = int(lid)
                    except Exception:
                        continue
                    lane_ids_by_road.setdefault(rid_i, set()).add(lid_i)
            try:
                ent = int(sig["entry"]["road_id"])
                ent_l = int(sig["entry"]["lane_id"])
                lane_ids_by_road.setdefault(ent, set()).add(ent_l)
            except Exception:
                pass
            try:
                ex = int(sig["exit"]["road_id"])
                ex_l = int(sig["exit"]["lane_id"])
                lane_ids_by_road.setdefault(ex, set()).add(ex_l)
            except Exception:
                pass

        for rid, lanes in lane_ids_by_road.items():
            ck = _corridor_key(road_corridors, rid)
            if ck is None:
                continue
            lane_count_by_corridor[ck] = max(lane_count_by_corridor.get(ck, 0), len(lanes))
    exit_corridor_counts: Dict[frozenset, int] = {}
    for c in candidates:
        exrid = _candidate_exit_road_id(c)
        ck = _corridor_key(road_corridors, exrid)
        if ck is None:
            continue
        exit_corridor_counts[ck] = exit_corridor_counts.get(ck, 0) + 1

    main_exit_corridor = None
    if exit_corridor_counts:
        main_exit_corridor = max(exit_corridor_counts.items(), key=lambda kv: kv[1])[0]

    ramp = []
    mainline = []
    main_lanes = lane_count_by_corridor.get(main_exit_corridor, 0) if main_exit_corridor else 0
    for c in candidates:
        exrid = _candidate_exit_road_id(c)
        entrid = _candidate_entry_road_id(c)
        exit_ck = _corridor_key(road_corridors, exrid)
        entry_ck = _corridor_key(road_corridors, entrid)
        if not main_exit_corridor or not exit_ck or not entry_ck:
            continue
        if exit_ck != main_exit_corridor:
            continue
        if entry_ck == main_exit_corridor:
            mainline.append(c)
            continue
        entry_lanes = lane_count_by_corridor.get(entry_ck, 0)
        if not (main_lanes >= 3 and entry_lanes > 0 and entry_lanes < main_lanes and entry_lanes <= 2):
            continue
        turn = str((c.get("signature") or {}).get("entry_to_exit_turn", "")).strip().lower()
        if turn == "uturn":
            continue
        cand = dict(c)
        cand["_ramp_entry_lanes"] = entry_lanes
        ramp.append(cand)

    def by_len(x):
        return float((x.get("signature") or {}).get("length_m", 0.0))

    if ramp:
        single_lane = [c for c in ramp if c.get("_ramp_entry_lanes") == 1]
        if single_lane:
            ramp = single_lane
    ramp.sort(key=by_len, reverse=True)
    mainline.sort(key=by_len, reverse=True)
    return ramp, mainline


def _print_candidate_summary(ramp: List[Dict[str, object]], mainline: List[Dict[str, object]], top_k: int) -> None:
    print(f"[INFO] ramp candidates: {len(ramp)}")
    print(f"[INFO] mainline candidates: {len(mainline)}")
    for i, c in enumerate(ramp[:top_k], start=1):
        sig = c.get("signature", {})
        ent = sig.get("entry", {})
        ex = sig.get("exit", {})
        entry_lanes = c.get("_ramp_entry_lanes")
        print(
            f"  ramp #{i}: {c.get('name')} "
            f"entry road={ent.get('road_id')} exit road={ex.get('road_id')} "
            f"len={sig.get('length_m'):.1f}m entry_lanes={entry_lanes}"
        )


def _load_nodes_xy(nodes_path: Path, crop: Optional[CropKey], step: int) -> Tuple[List[float], List[float], List[float]]:
    with open(nodes_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    payload = data.get("payload", {})
    xs = payload.get("x", [])
    ys = payload.get("y", [])
    yaws = payload.get("yaw", [])
    n = min(len(xs), len(ys))
    step = max(1, int(step))
    out_x = []
    out_y = []
    out_yaw = []
    if crop:
        pad = 6.0
        xmin = crop.xmin - pad
        xmax = crop.xmax + pad
        ymin = crop.ymin - pad
        ymax = crop.ymax + pad
        for i in range(0, n, step):
            x = float(xs[i])
            y = float(ys[i])
            if xmin <= x <= xmax and ymin <= y <= ymax:
                out_x.append(x)
                out_y.append(y)
                if i < len(yaws):
                    out_yaw.append(float(yaws[i]))
    else:
        for i in range(0, n, step):
            out_x.append(float(xs[i]))
            out_y.append(float(ys[i]))
            if i < len(yaws):
                out_yaw.append(float(yaws[i]))
    return out_x, out_y, out_yaw


def _draw_segment_arrow(
    ax,
    pts: List[Tuple[float, float]],
    color: str,
    alpha: float,
    width: float,
    head_width: float,
    head_length: float,
) -> None:
    if len(pts) < 2:
        return
    mid = max(1, len(pts) // 2)
    x0, y0 = pts[mid - 1]
    x1, y1 = pts[mid]
    dx = x1 - x0
    dy = y1 - y0
    seg_len = math.hypot(dx, dy)
    if seg_len < 1e-3:
        return
    arrow_len = min(4.0, max(1.6, seg_len * 0.35))
    scale = arrow_len / seg_len
    ax.arrow(
        x0,
        y0,
        dx * scale,
        dy * scale,
        width=width,
        head_width=head_width,
        head_length=head_length,
        length_includes_head=True,
        color=color,
        alpha=alpha,
        linewidth=0,
        zorder=2,
    )


def _plot_paths_together(
    all_segments,
    picked: List[Dict[str, object]],
    crop: Dict[str, float],
    out_path: Path,
    show: bool,
    nodes_xy: Optional[Tuple[List[float], List[float], List[float]]] = None,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib not available; install it or disable --viz")

    seg_by_id = {int(s.seg_id): s for s in all_segments}
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal", adjustable="box")

    if crop and all(k in crop for k in ("xmin", "xmax", "ymin", "ymax")):
        xmin, xmax, ymin, ymax = crop["xmin"], crop["xmax"], crop["ymin"], crop["ymax"]
        ax.set_xlim(xmin - 5, xmax + 5)
        ax.set_ylim(ymin - 5, ymax + 5)
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, linestyle="--", linewidth=2)
        ax.add_patch(rect)

    if nodes_xy:
        xs, ys, yaws = nodes_xy
        ax.scatter(xs, ys, s=2, alpha=0.2, color="black")
        if yaws:
            node_arrow_len = 1.8
            dxs = [math.cos(math.radians(y)) * node_arrow_len for y in yaws]
            dys = [math.sin(math.radians(y)) * node_arrow_len for y in yaws]
            ax.quiver(
                xs,
                ys,
                dxs,
                dys,
                angles="xy",
                scale_units="xy",
                scale=1.0,
                width=0.002,
                color="black",
                alpha=0.25,
                zorder=1,
            )

    for seg in all_segments:
        pts = seg.points
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, linewidth=0.8, alpha=0.2, color="gray")
        _draw_segment_arrow(
            ax,
            pts,
            color="gray",
            alpha=0.35,
            width=0.002,
            head_width=0.9,
            head_length=1.2,
        )

    ax.grid(True, alpha=0.3)
    ax.set_title(f"Picked Paths (n={len(picked)})")

    cmap = plt.cm.get_cmap("tab20")
    for i, entry in enumerate(picked):
        sig = entry.get("signature", {})
        seg_ids = sig.get("segment_ids", [])
        color = cmap(i % 20)
        for sid in seg_ids:
            seg = seg_by_id.get(int(sid))
            if not seg:
                continue
            pts = seg.points
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, linewidth=2.5, alpha=0.85, color=color)
            _draw_segment_arrow(
                ax,
                pts,
                color=color,
                alpha=0.8,
                width=0.003,
                head_width=1.2,
                head_length=1.6,
            )

        ent = sig.get("entry", {}).get("point", None)
        ex = sig.get("exit", {}).get("point", None)
        if ent:
            ax.plot(ent["x"], ent["y"], marker="o", markersize=8, color=color)
        if ex:
            ax.plot(ex["x"], ex["y"], marker="s", markersize=8, color=color)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Quick on-ramp harness (no full scenario generation).")
    ap.add_argument("--town", default="Town06")
    ap.add_argument("--out-dir", default="quick_onramp_harness")
    ap.add_argument("--radii", default="45,55,65", help="Crop radii list, e.g. 45,55,65")
    ap.add_argument("--min-path-len", type=float, default=22.0)
    ap.add_argument("--max-paths", type=int, default=80)
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--crop", default="", help="Optional crop: xmin,xmax,ymin,ymax")
    ap.add_argument("--top-k", type=int, default=5, help="Show top K ramp candidates")
    ap.add_argument("--viz", action="store_true", help="Write crop + picked path visualizations")
    ap.add_argument("--viz-show", action="store_true", help="Show plot windows (requires GUI)")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--viz-nodes", dest="viz_nodes", action="store_true", help="Show node points in viz")
    group.add_argument("--no-viz-nodes", dest="viz_nodes", action="store_false", help="Hide node points in viz")
    ap.set_defaults(viz_nodes=True)
    ap.add_argument("--viz-node-step", type=int, default=8, help="Subsample nodes by this step (default: 8)")
    args = ap.parse_args()

    nodes_path = REPO_ROOT / "scenario_generator" / "town_nodes" / f"{args.town}.json"
    if not nodes_path.exists():
        raise SystemExit(f"Nodes file not found: {nodes_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenario_text = (
        "Vehicle 1 travels along the main highway lane and continues straight. "
        "Vehicle 2 enters from the on-ramp and merges into Vehicle 1's lane. "
        "Vehicle 3 travels along the main highway lane to the right of Vehicle 1."
    )

    crop_feat = None
    if args.crop:
        crop_key = _parse_crop(args.crop)
        crop_feat = _use_explicit_crop(
            town=args.town,
            nodes_path=nodes_path,
            crop_key=crop_key,
            min_path_len=args.min_path_len,
            max_paths=args.max_paths,
            max_depth=args.max_depth,
        )
        if crop_feat is None:
            raise SystemExit("No features available for the provided crop.")
    else:
        radii = _parse_radii(args.radii)
        crop_key, crop_feat = _pick_crop(
            town=args.town,
            nodes_path=nodes_path,
            radii=radii,
            min_path_len=args.min_path_len,
            max_paths=args.max_paths,
            max_depth=args.max_depth,
            scenario_text=scenario_text,
        )

    if crop_feat:
        print(
            f"[INFO] crop={crop_key.to_str()} lanes={crop_feat.lane_count_est} "
            f"on_ramp={crop_feat.has_on_ramp} merge={crop_feat.has_merge_onto_same_road} "
            f"highway={crop_feat.is_highway} ramp_main_lanes={crop_feat.ramp_main_lanes} "
            f"ramp_entry_min={crop_feat.ramp_entry_min_lanes}"
        )
    else:
        print(f"[INFO] crop={crop_key.to_str()}")

    cb = glp.CropBox(crop_key.xmin, crop_key.xmax, crop_key.ymin, crop_key.ymax)
    data = glp.load_nodes(str(nodes_path))
    segments = glp.build_segments(data)
    cropped_segments = glp.crop_segments(segments, cb)
    if not cropped_segments:
        raise SystemExit("No segments found in crop region.")
    lane_ids_by_road: Dict[int, set] = {}
    for s in cropped_segments:
        try:
            rid = int(s.road_id)
            lid = int(s.lane_id)
        except Exception:
            continue
        lane_ids_by_road.setdefault(rid, set()).add(lid)
    lane_counts_by_road = {rid: len(lanes) for rid, lanes in lane_ids_by_road.items()}

    adj = glp.build_connectivity(cropped_segments, connect_radius_m=6.0, connect_yaw_tol_deg=60.0)
    legal_paths = glp.generate_legal_paths(
        cropped_segments,
        adj,
        cb,
        min_path_length=args.min_path_len,
        max_paths=args.max_paths,
        max_depth=args.max_depth,
        allow_within_region_fallback=False,
    )
    if not legal_paths:
        raise SystemExit("No legal paths found for this crop.")

    candidates = _build_candidates(legal_paths)
    agg_path = out_dir / "legal_paths_detailed.json"
    glp.save_aggregated_signatures_json(
        str(agg_path),
        crop=cb,
        nodes_path=str(nodes_path),
        params={
            "max_yaw_diff_deg": 60.0,
            "connect_radius_m": 6.0,
            "min_path_length_m": args.min_path_len,
            "max_paths": args.max_paths,
            "max_depth": args.max_depth,
            "turn_frame": "WORLD_FRAME",
        },
        paths_named=candidates,
        lane_counts_by_road=lane_counts_by_road,
    )

    ramp, mainline = _analyze_ramp_candidates(candidates, lane_counts_by_road=lane_counts_by_road)
    _print_candidate_summary(ramp, mainline, args.top_k)

    picked = []
    if mainline:
        picked.append({"vehicle": "Vehicle 1", "name": mainline[0]["name"], "signature": mainline[0]["signature"]})
    if ramp:
        picked.append({"vehicle": "Vehicle 2", "name": ramp[0]["name"], "signature": ramp[0]["signature"]})
    if len(mainline) > 1:
        picked.append({"vehicle": "Vehicle 3", "name": mainline[1]["name"], "signature": mainline[1]["signature"]})

    picked_path = out_dir / "picked_paths_detailed.json"
    with open(picked_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "nodes": str(nodes_path),
                "crop_region": {"xmin": cb.xmin, "xmax": cb.xmax, "ymin": cb.ymin, "ymax": cb.ymax},
                "picked": picked,
            },
            f,
            indent=2,
        )
    print(f"[INFO] Wrote picked paths: {picked_path}")

    if args.viz:
        nodes_xy = None
        if args.viz_nodes:
            nodes_xy = _load_nodes_xy(nodes_path, crop_key, args.viz_node_step)
        if crop_feat:
            crop_viz_path = out_dir / "crop_viz.png"
            save_viz(
                out_png=str(crop_viz_path),
                scenario_id="quick_onramp",
                scenario_text=scenario_text,
                crop=crop_key,
                crop_feat=crop_feat,
                invert_x=False,
                dpi=150,
            )
            print(f"[INFO] Wrote crop visualization: {crop_viz_path}")

        map_viz = out_dir / "map_viz.png"
        _plot_paths_together(
            all_segments=cropped_segments,
            picked=[],
            crop={"xmin": cb.xmin, "xmax": cb.xmax, "ymin": cb.ymin, "ymax": cb.ymax},
            out_path=str(map_viz),
            show=args.viz_show,
            nodes_xy=nodes_xy,
        )

        paths_viz = out_dir / "picked_paths_viz.png"
        _plot_paths_together(
            all_segments=cropped_segments,
            picked=picked,
            crop={"xmin": cb.xmin, "xmax": cb.xmax, "ymin": cb.ymin, "ymax": cb.ymax},
            out_path=str(paths_viz),
            show=args.viz_show,
            nodes_xy=nodes_xy,
        )


if __name__ == "__main__":
    main()
