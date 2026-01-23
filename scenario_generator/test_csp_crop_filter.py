#!/usr/bin/env python3
"""
test_csp_crop_filter.py

Isolated CSP / path picking script that imports functions from the pipeline,
runs the CSP with crop filtering, and generates visualizations.

Usage:
    python test_csp_crop_filter.py --legal-paths /path/to/legal_paths_detailed.json --schema /path/to/schema_normalized.json
"""

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import numpy as np

# Add scenario_generator to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.step_03_path_picker.candidates import (
    _candidate_entry_cardinal,
    _candidate_entry_in_crop,
    _candidate_entry_point,
    _candidate_entry_road_id,
)
from pipeline.step_03_path_picker.constraints import (
    _infer_road_role_sets,
    _normalize_constraints_obj,
)
from pipeline.step_03_path_picker.csp import _solve_paths_csp
from pipeline.step_03_path_picker.viz import _load_nodes, _build_segments_minimal


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def draw_crop_region(ax, crop_region: Dict, color='green', linestyle='-', linewidth=2, label=None, fill=True, alpha=0.1):
    """Draw crop region rectangle on axes."""
    if not crop_region:
        return
    xmin = crop_region.get("xmin", 0)
    xmax = crop_region.get("xmax", 0)
    ymin = crop_region.get("ymin", 0)
    ymax = crop_region.get("ymax", 0)
    rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                     fill=fill, facecolor=color if fill else 'none',
                     edgecolor=color, linewidth=linewidth, linestyle=linestyle,
                     alpha=alpha if fill else 1.0, label=label, zorder=1)
    ax.add_patch(rect)


def draw_road_network(ax, all_segments: List[Dict], alpha=0.2, color='gray', linewidth=0.5):
    """Draw all road segments as background."""
    for seg in all_segments:
        pts = seg.get("points", [])
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color=color, alpha=alpha, linewidth=linewidth, zorder=1)


def draw_path(ax, candidate: Dict, all_segments: List[Dict], color='blue', linewidth=2.5, alpha=0.8, label=None):
    """Draw a full path with its segments."""
    sig = candidate.get("signature", {})
    seg_ids = sig.get("segment_ids", [])
    seg_by_id = {s["seg_id"]: s for s in all_segments}
    
    first = True
    for sid in seg_ids:
        seg = seg_by_id.get(sid)
        if not seg:
            continue
        pts = seg.get("points", [])
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=alpha, zorder=5, 
                label=label if first else None)
        first = False
    
    # Entry marker
    ent = sig.get("entry", {}).get("point")
    if ent:
        ax.plot(ent["x"], ent["y"], marker='o', markersize=12, 
               color=color, markeredgecolor='black', markeredgewidth=1.5, zorder=10)
    
    # Exit marker
    ext = sig.get("exit", {}).get("point")
    if ext:
        ax.plot(ext["x"], ext["y"], marker='s', markersize=10,
               color=color, markeredgecolor='black', markeredgewidth=1, zorder=10)


def run_test(legal_paths_file: str, schema_file: str, output_dir: str):
    """Run the CSP with crop filtering and visualize results."""
    print(f"{'='*80}")
    print("LOADING DATA")
    print(f"{'='*80}")
    
    agg = load_json(legal_paths_file)
    schema = load_json(schema_file)
    
    candidates = agg.get("candidates", [])
    crop_region = agg.get("crop_region", {})
    nodes_path = agg.get("nodes")
    
    print(f"Loaded {len(candidates)} candidates")
    print(f"Crop region: {crop_region}")
    
    # Load nodes for visualization
    all_segments = []
    if nodes_path and os.path.exists(nodes_path):
        nodes = _load_nodes(nodes_path)
        all_segments = _build_segments_minimal(nodes)
        print(f"Loaded {len(all_segments)} road segments")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # =========================================================================
    # STAGE 1: Show ALL candidates (before crop filter)
    # =========================================================================
    print(f"\n{'='*80}")
    print("STAGE 1: ALL CANDIDATES (BEFORE CROP FILTER)")
    print(f"{'='*80}")
    
    print(f"\n{'Name':<60} {'Y':>10} {'Cardinal':>10} {'InCrop':>10}")
    print("-" * 95)
    for c in candidates:
        name = c.get("name", "")[:55]
        pt = _candidate_entry_point(c)
        y = pt[1] if pt else 0
        cardinal = _candidate_entry_cardinal(c)
        in_crop = _candidate_entry_in_crop(c, crop_region, margin=0)
        print(f"{name:<60} {y:>10.2f} {cardinal:>10} {'YES' if in_crop else 'NO':>10}")
    
    # Role inference BEFORE crop filter
    role_sets_before = _infer_road_role_sets(candidates)
    print(f"\nRole sets BEFORE crop filter:")
    print(f"  Entry set: {role_sets_before.get('entry_set', set())}")
    print(f"  Main entry: {role_sets_before.get('main_entry', set())}")
    print(f"  Side entry: {role_sets_before.get('side_entry', set())}")
    
    # =========================================================================
    # STAGE 2: Apply crop filter (what CSP does now)
    # =========================================================================
    print(f"\n{'='*80}")
    print("STAGE 2: CANDIDATES AFTER CROP FILTER (margin=0)")
    print(f"{'='*80}")
    
    filtered_candidates = [c for c in candidates if _candidate_entry_in_crop(c, crop_region, margin=0)]
    print(f"\nFiltered from {len(candidates)} to {len(filtered_candidates)} candidates")
    
    print(f"\n{'Name':<60} {'Y':>10} {'Cardinal':>10}")
    print("-" * 85)
    for c in filtered_candidates:
        name = c.get("name", "")[:55]
        pt = _candidate_entry_point(c)
        y = pt[1] if pt else 0
        cardinal = _candidate_entry_cardinal(c)
        print(f"{name:<60} {y:>10.2f} {cardinal:>10}")
    
    # Role inference AFTER crop filter
    role_sets_after = _infer_road_role_sets(filtered_candidates)
    print(f"\nRole sets AFTER crop filter:")
    print(f"  Entry set: {role_sets_after.get('entry_set', set())}")
    print(f"  Main entry: {role_sets_after.get('main_entry', set())}")
    print(f"  Side entry: {role_sets_after.get('side_entry', set())}")
    
    # =========================================================================
    # STAGE 3: Run the actual CSP
    # =========================================================================
    print(f"\n{'='*80}")
    print("STAGE 3: RUNNING CSP WITH CROP FILTER")
    print(f"{'='*80}")
    
    # Build constraints_obj from schema
    vehicles = schema.get("ego_vehicles", [])
    constraints = schema.get("vehicle_constraints", [])
    description = schema.get("description", "")
    
    constraints_obj = {
        "vehicles": [
            {
                "vehicle": v.get("vehicle_id"),
                "maneuver": v.get("maneuver"),
                "entry_road": v.get("entry_road"),
                "exit_road": v.get("exit_road"),
            }
            for v in vehicles
        ],
        "constraints": constraints,
    }
    
    print(f"\nVehicles from schema:")
    for v in constraints_obj["vehicles"]:
        print(f"  {v['vehicle']}: maneuver={v['maneuver']}, entry={v['entry_road']}, exit={v['exit_road']}")
    
    try:
        result = _solve_paths_csp(
            constraints_obj=constraints_obj,
            candidates=candidates,  # Pass ALL candidates - CSP will filter internally
            description=description,
            require_straight=False,
            require_on_ramp=False,
            crop_region=crop_region,  # This triggers the crop filtering
        )
        
        print(f"\nCSP RESULT:")
        for r in result:
            print(f"  {r['vehicle']}: {r['path_name']} (confidence={r.get('confidence', 'N/A')})")
            
    except Exception as e:
        print(f"\nCSP FAILED: {e}")
        result = []
    
    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    print(f"\n{'='*80}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*80}")
    
    cmap = plt.colormaps['tab10']
    
    # VIZ 1: Before vs After crop filter
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Left: Before
    ax = axes[0]
    ax.set_title(f"BEFORE Crop Filter\n({len(candidates)} candidates)\nMain: {role_sets_before.get('main_entry', set())}, Side: {role_sets_before.get('side_entry', set())}", 
                 fontsize=11, fontweight='bold')
    if all_segments:
        draw_road_network(ax, all_segments, alpha=0.2)
    draw_crop_region(ax, crop_region, fill=True, alpha=0.15, color='lightgreen')
    
    for i, c in enumerate(candidates):
        in_crop = _candidate_entry_in_crop(c, crop_region, margin=0)
        color = cmap(i % 10) if in_crop else 'red'
        alpha = 0.9 if in_crop else 0.4
        draw_path(ax, c, all_segments, color=color, linewidth=2.5 if in_crop else 1.5, alpha=alpha)
        # Label
        pt = _candidate_entry_point(c)
        if pt:
            name = c.get("name", "").split("__")[0]
            ax.annotate(name, (pt[0], pt[1]), fontsize=7, ha='left', xytext=(5, 5), textcoords='offset points')
    
    if crop_region:
        ax.set_xlim(crop_region["xmin"] - 80, crop_region["xmax"] + 80)
        ax.set_ylim(crop_region["ymin"] - 80, crop_region["ymax"] + 80)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    
    # Right: After
    ax = axes[1]
    ax.set_title(f"AFTER Crop Filter\n({len(filtered_candidates)} candidates)\nMain: {role_sets_after.get('main_entry', set())}, Side: {role_sets_after.get('side_entry', set())}", 
                 fontsize=11, fontweight='bold')
    if all_segments:
        draw_road_network(ax, all_segments, alpha=0.2)
    draw_crop_region(ax, crop_region, fill=True, alpha=0.15, color='lightgreen')
    
    for i, c in enumerate(filtered_candidates):
        cardinal = _candidate_entry_cardinal(c)
        is_side = cardinal in role_sets_after.get('side_entry', set())
        color = 'blue' if is_side else cmap(i % 10)
        draw_path(ax, c, all_segments, color=color, linewidth=3 if is_side else 2, alpha=0.9)
        # Label
        pt = _candidate_entry_point(c)
        if pt:
            name = c.get("name", "").split("__")[0]
            ax.annotate(name, (pt[0], pt[1]), fontsize=8, fontweight='bold' if is_side else 'normal',
                       ha='left', xytext=(5, 5), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow' if is_side else 'white', alpha=0.8))
    
    if crop_region:
        ax.set_xlim(crop_region["xmin"] - 40, crop_region["xmax"] + 40)
        ax.set_ylim(crop_region["ymin"] - 40, crop_region["ymax"] + 40)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, "csp_before_after_crop.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()
    
    # VIZ 2: CSP Result
    if result:
        fig, ax = plt.subplots(figsize=(14, 14))
        ax.set_title(f"CSP RESULT: Picked Paths", fontsize=14, fontweight='bold')
        
        if all_segments:
            draw_road_network(ax, all_segments, alpha=0.2)
        draw_crop_region(ax, crop_region, fill=True, alpha=0.15, color='lightgreen')
        
        # Find picked candidates
        picked_names = {r['path_name'] for r in result}
        cand_by_name = {c.get("name"): c for c in candidates}
        
        for i, r in enumerate(result):
            name = r['path_name']
            vehicle = r['vehicle']
            c = cand_by_name.get(name)
            if c:
                color = cmap(i % 10)
                draw_path(ax, c, all_segments, color=color, linewidth=4, alpha=0.95)
                pt = _candidate_entry_point(c)
                if pt:
                    ax.annotate(f"{vehicle}\n{name.split('__')[0]}", (pt[0], pt[1]), 
                               fontsize=10, fontweight='bold', ha='left', 
                               xytext=(10, 10), textcoords='offset points',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8, edgecolor='black'),
                               color='white')
        
        if crop_region:
            ax.set_xlim(crop_region["xmin"] - 40, crop_region["xmax"] + 40)
            ax.set_ylim(crop_region["ymin"] - 40, crop_region["ymax"] + 40)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        
        plt.tight_layout()
        out_path = os.path.join(output_dir, "csp_result.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {out_path}")
        plt.close()
    
    print(f"\n{'='*80}")
    print("DONE")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Test CSP with crop filtering")
    parser.add_argument("--legal-paths", required=True, help="Path to legal_paths_detailed.json")
    parser.add_argument("--schema", required=True, help="Path to schema_normalized.json")
    parser.add_argument("--output-dir", default="./csp_test_output", help="Output directory")
    args = parser.parse_args()
    
    run_test(args.legal_paths, args.schema, args.output_dir)


if __name__ == "__main__":
    main()
