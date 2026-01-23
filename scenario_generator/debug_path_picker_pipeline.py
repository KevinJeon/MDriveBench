#!/usr/bin/env python3
"""
debug_path_picker_pipeline.py

Comprehensive diagnostic script to trace path selection through every stage of the pipeline.
Generates detailed visualizations showing nodes, full paths, and filtering at each step.

Usage:
    python debug_path_picker_pipeline.py --legal-paths /path/to/legal_paths_detailed.json --schema /path/to/schema_normalized.json
"""

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
from matplotlib.collections import LineCollection
import numpy as np

# Add scenario_generator to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.step_03_path_picker.candidates import (
    _build_road_corridors,
    _candidate_entry_cardinal,
    _candidate_entry_in_crop,
    _candidate_entry_lane_id,
    _candidate_entry_point,
    _candidate_entry_road_id,
    _candidate_exit_cardinal,
    _candidate_exit_road_id,
    _candidate_length,
)
from pipeline.step_03_path_picker.constraints import (
    _candidate_matches_unary,
    _infer_road_role_sets,
    _normalize_constraints_obj,
)
from pipeline.step_03_path_picker.viz import _load_nodes, _build_segments_minimal


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def get_path_segments(candidate: Dict, all_segments: List[Dict]) -> List[Dict]:
    """Get the segment data for a candidate path."""
    sig = candidate.get("signature", {})
    seg_ids = sig.get("segment_ids", [])
    seg_by_id = {s["seg_id"]: s for s in all_segments}
    return [seg_by_id[sid] for sid in seg_ids if sid in seg_by_id]


def draw_crop_region(ax, crop_region: Dict, margin: float = 0, color='green', linestyle='--', linewidth=2, label=None, fill=False, alpha=0.1):
    """Draw crop region rectangle on axes."""
    if not crop_region or not all(k in crop_region for k in ("xmin", "xmax", "ymin", "ymax")):
        return
    xmin = crop_region.get("xmin", 0) - margin
    xmax = crop_region.get("xmax", 0) + margin
    ymin = crop_region.get("ymin", 0) - margin
    ymax = crop_region.get("ymax", 0) + margin
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


def draw_path(ax, candidate: Dict, all_segments: List[Dict], color='blue', linewidth=2.5, alpha=0.8, 
              draw_entry=True, draw_exit=True, entry_marker='o', exit_marker='s', markersize=10, label=None):
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
        
        # Draw direction arrow at midpoint
        mid = len(pts) // 2
        if mid > 0:
            x0, y0 = pts[mid - 1]
            x1, y1 = pts[mid]
            dx, dy = x1 - x0, y1 - y0
            seg_len = math.hypot(dx, dy)
            if seg_len > 0.5:
                ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                           arrowprops=dict(arrowstyle='->', color=color, lw=1.5, alpha=alpha),
                           zorder=6)
    
    # Entry and exit markers
    if draw_entry:
        ent = sig.get("entry", {}).get("point")
        if ent:
            ax.plot(ent["x"], ent["y"], marker=entry_marker, markersize=markersize, 
                   color=color, markeredgecolor='black', markeredgewidth=1, zorder=10)
    
    if draw_exit:
        ext = sig.get("exit", {}).get("point")
        if ext:
            ax.plot(ext["x"], ext["y"], marker=exit_marker, markersize=markersize,
                   color=color, markeredgecolor='black', markeredgewidth=1, zorder=10)


def draw_entry_point_with_label(ax, candidate: Dict, color='blue', show_details=True, fontsize=7):
    """Draw entry point with detailed label."""
    sig = candidate.get("signature", {})
    name = candidate.get("name", "")
    ent = sig.get("entry", {})
    pt = ent.get("point")
    if not pt:
        return
    
    x, y = pt["x"], pt["y"]
    cardinal = ent.get("cardinal4", "?")
    road_id = ent.get("road_id", "?")
    
    ax.plot(x, y, 'o', color=color, markersize=12, markeredgecolor='black', markeredgewidth=1.5, zorder=10)
    
    if show_details:
        short_name = name.split("__")[0] if "__" in name else name[:12]
        maneuver = "?"
        if "__man=" in name:
            maneuver = name.split("__man=")[1].split("__")[0]
        label = f"{short_name}\n{cardinal} rd={road_id}\ny={y:.1f}\nman={maneuver}"
        ax.annotate(label, (x, y), fontsize=fontsize, ha='left', va='bottom',
                   xytext=(10, 5), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.9),
                   zorder=15)


def analyze_side_road_candidates(candidates: List[Dict], role_sets: Dict) -> Tuple[Set[str], Set[str]]:
    """Find which candidates are classified as side-road entries."""
    side_entry = role_sets.get("side_entry", set())
    main_entry = role_sets.get("main_entry", set())
    
    side_candidates = set()
    main_candidates = set()
    
    for c in candidates:
        name = c.get("name", "")
        cardinal = _candidate_entry_cardinal(c)
        if cardinal in side_entry:
            side_candidates.add(name)
        if cardinal in main_entry:
            main_candidates.add(name)
    
    return side_candidates, main_candidates


def run_diagnostic(legal_paths_file: str, schema_file: str, output_dir: str, 
                   picked_paths_file: Optional[str] = None, refined_paths_file: Optional[str] = None):
    """Run full diagnostic pipeline."""
    print(f"[DEBUG] Loading legal paths from: {legal_paths_file}")
    agg = load_json(legal_paths_file)
    
    print(f"[DEBUG] Loading schema from: {schema_file}")
    schema = load_json(schema_file)
    
    candidates = agg.get("candidates", [])
    crop_region = agg.get("crop_region", {})
    nodes_path = agg.get("nodes")
    
    # Load nodes and build segments for visualization
    all_segments = []
    if nodes_path and os.path.exists(nodes_path):
        print(f"[DEBUG] Loading nodes from: {nodes_path}")
        nodes = _load_nodes(nodes_path)
        all_segments = _build_segments_minimal(nodes)
        print(f"[DEBUG] Built {len(all_segments)} road segments for visualization")
    
    # Load picked paths if available
    picked_data = None
    if picked_paths_file and os.path.exists(picked_paths_file):
        print(f"[DEBUG] Loading picked paths from: {picked_paths_file}")
        picked_data = load_json(picked_paths_file)
    
    # Load refined paths if available
    refined_data = None
    if refined_paths_file and os.path.exists(refined_paths_file):
        print(f"[DEBUG] Loading refined paths from: {refined_paths_file}")
        refined_data = load_json(refined_paths_file)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # =========================================================================
    # STAGE 0: Console output
    # =========================================================================
    print(f"\n{'='*80}")
    print("STAGE 0: RAW DATA OVERVIEW")
    print(f"{'='*80}")
    print(f"Total candidates: {len(candidates)}")
    print(f"Crop region: {crop_region}")
    
    print("\nAll candidate entry points:")
    print(f"{'Name':<70} {'X':>10} {'Y':>10} {'Cardinal':>10} {'Road':>8} {'InCrop':>8}")
    print("-" * 120)
    for c in candidates:
        name = c.get("name", "")[:65]
        pt = _candidate_entry_point(c)
        if pt:
            x, y = pt
            cardinal = _candidate_entry_cardinal(c)
            road_id = _candidate_entry_road_id(c)
            in_crop = _candidate_entry_in_crop(c, crop_region, margin=10.0)
            print(f"{name:<70} {x:>10.2f} {y:>10.2f} {cardinal:>10} {road_id:>8} {'YES' if in_crop else 'NO':>8}")
    
    # Stage 1: Infer road role sets
    print(f"\n{'='*80}")
    print("STAGE 1: ROAD ROLE INFERENCE")
    print(f"{'='*80}")
    
    role_sets = _infer_road_role_sets(candidates)
    print(f"Entry set (all directions): {role_sets.get('entry_set', set())}")
    print(f"Main entry cardinals: {role_sets.get('main_entry', set())}")
    print(f"Side entry cardinals: {role_sets.get('side_entry', set())}")
    
    side_candidates, main_candidates = analyze_side_road_candidates(candidates, role_sets)
    print(f"\nSIDE-road candidates: {len(side_candidates)}")
    for name in sorted(side_candidates):
        c = next((c for c in candidates if c.get("name") == name), None)
        if c:
            pt = _candidate_entry_point(c)
            in_crop = _candidate_entry_in_crop(c, crop_region, margin=10.0)
            y_str = f"{pt[1]:.1f}" if pt else "?"
            print(f"  - {name[:60]}... y={y_str} in_crop={in_crop}")
    
    # =========================================================================
    # VISUALIZATION 1: Road Network Overview
    # =========================================================================
    print(f"\n{'='*80}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*80}")
    
    cmap = plt.cm.get_cmap('tab20')
    
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.set_title("VIZ 1: Road Network & All Candidate Paths", fontsize=14, fontweight='bold')
    
    # Draw road network
    if all_segments:
        draw_road_network(ax, all_segments, alpha=0.3, color='lightgray', linewidth=1)
    
    # Draw crop region
    draw_crop_region(ax, crop_region, margin=0, color='green', linestyle='-', linewidth=3, 
                     label='Crop Region', fill=True, alpha=0.1)
    draw_crop_region(ax, crop_region, margin=10, color='green', linestyle=':', linewidth=1,
                     label='Crop +10m margin')
    
    # Draw all candidate paths
    for i, c in enumerate(candidates):
        color = cmap(i % 20)
        is_side = c.get("name") in side_candidates
        draw_path(ax, c, all_segments, color=color, linewidth=3 if is_side else 1.5, 
                  alpha=0.9 if is_side else 0.5, label=c.get("name", "")[:30])
    
    # Set axis limits
    if crop_region:
        margin_view = 60
        ax.set_xlim(crop_region["xmin"] - margin_view, crop_region["xmax"] + margin_view)
        ax.set_ylim(crop_region["ymin"] - margin_view, crop_region["ymax"] + margin_view)
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=6, ncol=2)
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    
    out_path = os.path.join(output_dir, "viz1_road_network_all_paths.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()
    
    # =========================================================================
    # VISUALIZATION 2: Side Road vs Main Road Classification
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Left: Side road paths
    ax = axes[0]
    ax.set_title("VIZ 2a: SIDE-Road Candidate Paths", fontsize=12, fontweight='bold')
    if all_segments:
        draw_road_network(ax, all_segments, alpha=0.2)
    draw_crop_region(ax, crop_region, fill=True, alpha=0.15, color='green')
    draw_crop_region(ax, crop_region, margin=10, linestyle=':', color='green')
    
    for c in candidates:
        if c.get("name") in side_candidates:
            in_crop = _candidate_entry_in_crop(c, crop_region, margin=10.0)
            color = 'blue' if in_crop else 'red'
            draw_path(ax, c, all_segments, color=color, linewidth=3, alpha=0.9)
            draw_entry_point_with_label(ax, c, color=color)
    
    if crop_region:
        ax.set_xlim(crop_region["xmin"] - 60, crop_region["xmax"] + 60)
        ax.set_ylim(crop_region["ymin"] - 60, crop_region["ymax"] + 60)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    
    # Right: Main road paths
    ax = axes[1]
    ax.set_title("VIZ 2b: MAIN-Road Candidate Paths", fontsize=12, fontweight='bold')
    if all_segments:
        draw_road_network(ax, all_segments, alpha=0.2)
    draw_crop_region(ax, crop_region, fill=True, alpha=0.15, color='green')
    
    for i, c in enumerate(candidates):
        if c.get("name") in main_candidates:
            in_crop = _candidate_entry_in_crop(c, crop_region, margin=10.0)
            color = cmap(i % 20)
            draw_path(ax, c, all_segments, color=color, linewidth=2, alpha=0.8)
    
    if crop_region:
        ax.set_xlim(crop_region["xmin"] - 60, crop_region["xmax"] + 60)
        ax.set_ylim(crop_region["ymin"] - 60, crop_region["ymax"] + 60)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, "viz2_side_vs_main_roads.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()
    
    # =========================================================================
    # VISUALIZATION 3: Entry Points with Y-Coordinate Analysis
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Left: Map view
    ax = axes[0]
    ax.set_title("VIZ 3a: Entry Points - Map View\n(Blue=In Crop, Red=Out of Crop)", fontsize=12, fontweight='bold')
    if all_segments:
        draw_road_network(ax, all_segments, alpha=0.2)
    draw_crop_region(ax, crop_region, fill=True, alpha=0.15, color='lightgreen')
    draw_crop_region(ax, crop_region, margin=10, linestyle=':', color='green', linewidth=2)
    
    # Draw horizontal lines for crop Y bounds
    if crop_region:
        ymin, ymax = crop_region["ymin"], crop_region["ymax"]
        ax.axhline(y=ymin, color='green', linestyle='-', linewidth=2, alpha=0.7, label=f'Crop ymin={ymin:.1f}')
        ax.axhline(y=ymax, color='green', linestyle='-', linewidth=2, alpha=0.7, label=f'Crop ymax={ymax:.1f}')
        ax.axhline(y=ymin-10, color='green', linestyle=':', linewidth=1, alpha=0.5)
        ax.axhline(y=ymax+10, color='green', linestyle=':', linewidth=1, alpha=0.5)
    
    # Plot all entry points
    for c in candidates:
        pt = _candidate_entry_point(c)
        if not pt:
            continue
        x, y = pt
        in_crop = _candidate_entry_in_crop(c, crop_region, margin=10.0)
        is_side = c.get("name") in side_candidates
        
        if is_side:
            color = 'blue' if in_crop else 'red'
            marker = 'o'
            size = 200
            draw_entry_point_with_label(ax, c, color=color)
        else:
            color = 'gray'
            marker = 's'
            size = 80
            ax.scatter(x, y, c=color, s=size, marker=marker, alpha=0.6, zorder=5)
    
    if crop_region:
        ax.set_xlim(crop_region["xmin"] - 80, crop_region["xmax"] + 80)
        ax.set_ylim(crop_region["ymin"] - 80, crop_region["ymax"] + 80)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    
    # Right: Y-coordinate bar chart
    ax = axes[1]
    ax.set_title("VIZ 3b: Entry Point Y-Coordinates\n(Side-Road Candidates Only)", fontsize=12, fontweight='bold')
    
    side_cands = [c for c in candidates if c.get("name") in side_candidates]
    if side_cands:
        names = []
        y_vals = []
        colors = []
        for c in side_cands:
            pt = _candidate_entry_point(c)
            if pt:
                names.append(c.get("name", "")[:25])
                y_vals.append(pt[1])
                in_crop = _candidate_entry_in_crop(c, crop_region, margin=10.0)
                colors.append('blue' if in_crop else 'red')
        
        y_pos = range(len(names))
        ax.barh(y_pos, y_vals, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel('Y Coordinate (meters)')
        
        # Draw crop region lines
        if crop_region:
            ymin, ymax = crop_region["ymin"], crop_region["ymax"]
            ax.axvline(x=ymin, color='green', linestyle='-', linewidth=2, label=f'Crop ymin={ymin:.1f}')
            ax.axvline(x=ymax, color='green', linestyle='-', linewidth=2, label=f'Crop ymax={ymax:.1f}')
            ax.axvline(x=ymin-10, color='green', linestyle=':', linewidth=1, alpha=0.5)
            ax.axvline(x=ymax+10, color='green', linestyle=':', linewidth=1, alpha=0.5)
            ax.legend()
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, "viz3_entry_points_analysis.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()
    
    # =========================================================================
    # VISUALIZATION 4: Per-Vehicle Domain Analysis
    # =========================================================================
    vehicles = schema.get("ego_vehicles", [])
    constraints = schema.get("vehicle_constraints", [])
    
    # Convert schema to constraints_obj format
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
    
    norm = _normalize_constraints_obj(constraints_obj)
    
    # Find side-road vehicles
    side_vehicles = [v for v in norm.get("vehicles", []) 
                     if str(v.get("entry_road", "")).lower() == "side"]
    
    if side_vehicles:
        n_side = len(side_vehicles)
        fig, axes = plt.subplots(1, n_side, figsize=(10*n_side, 10))
        if n_side == 1:
            axes = [axes]
        
        for i, v_obj in enumerate(side_vehicles):
            ax = axes[i]
            v = v_obj.get("vehicle")
            man = str(v_obj.get("maneuver", "unknown")).strip().lower()
            er = str(v_obj.get("entry_road", "unknown")).strip().lower()
            xr = str(v_obj.get("exit_road", "unknown")).strip().lower()
            
            ax.set_title(f"VIZ 4: {v}\nmaneuver={man}, entry_road={er}, exit_road={xr}", 
                        fontsize=12, fontweight='bold')
            
            if all_segments:
                draw_road_network(ax, all_segments, alpha=0.2)
            draw_crop_region(ax, crop_region, fill=True, alpha=0.15, color='lightgreen')
            draw_crop_region(ax, crop_region, margin=10, linestyle=':', color='green')
            
            # Find matching candidates
            dom = [c for c in candidates if _candidate_matches_unary(c, man, er, xr, role_sets)]
            
            print(f"\n{v} domain: {len(dom)} candidates")
            for c in dom:
                in_crop = _candidate_entry_in_crop(c, crop_region, margin=10.0)
                color = 'blue' if in_crop else 'red'
                draw_path(ax, c, all_segments, color=color, linewidth=3, alpha=0.8)
                draw_entry_point_with_label(ax, c, color=color)
                pt = _candidate_entry_point(c)
                y_str = f"{pt[1]:.1f}" if pt else "?"
                print(f"  {'✓' if in_crop else '✗'} {c.get('name')[:50]}... y={y_str}")
            
            if crop_region:
                ax.set_xlim(crop_region["xmin"] - 60, crop_region["xmax"] + 60)
                ax.set_ylim(crop_region["ymin"] - 60, crop_region["ymax"] + 60)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('X (meters)')
            ax.set_ylabel('Y (meters)')
        
        plt.tight_layout()
        out_path = os.path.join(output_dir, "viz4_side_vehicle_domains.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {out_path}")
        plt.close()
    
    # =========================================================================
    # VISUALIZATION 5: Picked Paths (if available)
    # =========================================================================
    if picked_data:
        picked = picked_data.get("picked", [])
        
        fig, ax = plt.subplots(figsize=(16, 16))
        ax.set_title(f"VIZ 5: PICKED Paths (n={len(picked)})", fontsize=14, fontweight='bold')
        
        if all_segments:
            draw_road_network(ax, all_segments, alpha=0.2)
        draw_crop_region(ax, crop_region, fill=True, alpha=0.15, color='lightgreen')
        
        for i, entry in enumerate(picked):
            color = cmap(i % 20)
            vehicle = entry.get("vehicle", f"V{i}")
            sig = entry.get("signature", {})
            
            # Create a pseudo-candidate for drawing
            pseudo_cand = {"name": entry.get("name", ""), "signature": sig}
            draw_path(ax, pseudo_cand, all_segments, color=color, linewidth=4, alpha=0.9)
            
            # Label entry point with vehicle name
            ent = sig.get("entry", {}).get("point")
            if ent:
                ax.annotate(vehicle, (ent["x"], ent["y"]), fontsize=10, fontweight='bold',
                           ha='left', va='bottom', xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8, edgecolor='black'),
                           color='white', zorder=20)
        
        if crop_region:
            ax.set_xlim(crop_region["xmin"] - 40, crop_region["xmax"] + 40)
            ax.set_ylim(crop_region["ymin"] - 40, crop_region["ymax"] + 40)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        
        out_path = os.path.join(output_dir, "viz5_picked_paths.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {out_path}")
        plt.close()
    
    # =========================================================================
    # VISUALIZATION 6: Refined Paths (if available)
    # =========================================================================
    if refined_data:
        refined = refined_data.get("refined", [])
        
        fig, ax = plt.subplots(figsize=(16, 16))
        ax.set_title(f"VIZ 6: REFINED Paths (n={len(refined)})", fontsize=14, fontweight='bold')
        
        if all_segments:
            draw_road_network(ax, all_segments, alpha=0.2)
        draw_crop_region(ax, crop_region, fill=True, alpha=0.15, color='lightgreen')
        
        for i, entry in enumerate(refined):
            color = cmap(i % 20)
            vehicle = entry.get("vehicle", f"V{i}")
            
            # Draw refined waypoints
            waypoints = entry.get("waypoints", [])
            if waypoints:
                xs = [w.get("x", 0) for w in waypoints]
                ys = [w.get("y", 0) for w in waypoints]
                ax.plot(xs, ys, color=color, linewidth=4, alpha=0.9, marker='o', markersize=3)
                
                # Label start point
                if xs and ys:
                    ax.annotate(vehicle, (xs[0], ys[0]), fontsize=10, fontweight='bold',
                               ha='left', va='bottom', xytext=(10, 10), textcoords='offset points',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8, edgecolor='black'),
                               color='white', zorder=20)
        
        if crop_region:
            ax.set_xlim(crop_region["xmin"] - 40, crop_region["xmax"] + 40)
            ax.set_ylim(crop_region["ymin"] - 40, crop_region["ymax"] + 40)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        
        out_path = os.path.join(output_dir, "viz6_refined_paths.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {out_path}")
        plt.close()
    
    # =========================================================================
    # VISUALIZATION 7: Combined Multi-Stage View
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    
    titles = [
        "Stage 1: All Candidates",
        "Stage 2: Side-Road Classification",
        "Stage 3: Crop Filtering",
        "Stage 4: Unary Constraints",
        "Stage 5: Picked Paths",
        "Stage 6: Refined Paths"
    ]
    
    for i, (ax, title) in enumerate(zip(axes.flat, titles)):
        ax.set_title(title, fontsize=11, fontweight='bold')
        if all_segments:
            draw_road_network(ax, all_segments, alpha=0.15)
        draw_crop_region(ax, crop_region, fill=True, alpha=0.1, color='lightgreen')
        
        if i == 0:  # All candidates
            for j, c in enumerate(candidates):
                color = cmap(j % 20)
                draw_path(ax, c, all_segments, color=color, linewidth=1.5, alpha=0.6, draw_entry=True, draw_exit=False)
        
        elif i == 1:  # Side-road classification
            for c in candidates:
                is_side = c.get("name") in side_candidates
                color = 'blue' if is_side else 'gray'
                lw = 3 if is_side else 1
                draw_path(ax, c, all_segments, color=color, linewidth=lw, alpha=0.7 if is_side else 0.3)
        
        elif i == 2:  # Crop filtering
            for c in candidates:
                if c.get("name") in side_candidates:
                    in_crop = _candidate_entry_in_crop(c, crop_region, margin=10.0)
                    color = 'blue' if in_crop else 'red'
                    draw_path(ax, c, all_segments, color=color, linewidth=3, alpha=0.8)
        
        elif i == 3:  # Unary constraints (show side vehicle domains)
            if side_vehicles:
                v_obj = side_vehicles[0]  # Just show first side vehicle
                man = str(v_obj.get("maneuver", "unknown")).strip().lower()
                er = str(v_obj.get("entry_road", "unknown")).strip().lower()
                xr = str(v_obj.get("exit_road", "unknown")).strip().lower()
                dom = [c for c in candidates if _candidate_matches_unary(c, man, er, xr, role_sets)]
                for c in dom:
                    in_crop = _candidate_entry_in_crop(c, crop_region, margin=10.0)
                    color = 'blue' if in_crop else 'red'
                    draw_path(ax, c, all_segments, color=color, linewidth=3, alpha=0.8)
        
        elif i == 4:  # Picked paths
            if picked_data:
                for j, entry in enumerate(picked_data.get("picked", [])):
                    color = cmap(j % 20)
                    pseudo_cand = {"name": entry.get("name", ""), "signature": entry.get("signature", {})}
                    draw_path(ax, pseudo_cand, all_segments, color=color, linewidth=3, alpha=0.9)
        
        elif i == 5:  # Refined paths
            if refined_data:
                for j, entry in enumerate(refined_data.get("refined", [])):
                    color = cmap(j % 20)
                    waypoints = entry.get("waypoints", [])
                    if waypoints:
                        xs = [w.get("x", 0) for w in waypoints]
                        ys = [w.get("y", 0) for w in waypoints]
                        ax.plot(xs, ys, color=color, linewidth=3, alpha=0.9)
        
        if crop_region:
            ax.set_xlim(crop_region["xmin"] - 40, crop_region["xmax"] + 40)
            ax.set_ylim(crop_region["ymin"] - 40, crop_region["ymax"] + 40)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, "viz7_pipeline_stages_combined.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()
    
    print(f"\n{'='*80}")
    print("DIAGNOSTIC COMPLETE")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Debug path picker pipeline with visualizations")
    parser.add_argument("--legal-paths", required=True, help="Path to legal_paths_detailed.json")
    parser.add_argument("--schema", required=True, help="Path to schema_normalized.json")
    parser.add_argument("--picked-paths", help="Path to picked_paths_detailed.json (optional)")
    parser.add_argument("--refined-paths", help="Path to picked_paths_refined.json (optional)")
    parser.add_argument("--output-dir", default="./debug_output", help="Output directory for visualizations")
    args = parser.parse_args()
    
    # Auto-detect picked and refined paths if not specified
    base_dir = os.path.dirname(args.legal_paths)
    picked = args.picked_paths or os.path.join(base_dir, "picked_paths_detailed.json")
    refined = args.refined_paths or os.path.join(base_dir, "picked_paths_refined.json")
    
    run_diagnostic(
        args.legal_paths, 
        args.schema, 
        args.output_dir,
        picked_paths_file=picked if os.path.exists(picked) else None,
        refined_paths_file=refined if os.path.exists(refined) else None,
    )


if __name__ == "__main__":
    main()
