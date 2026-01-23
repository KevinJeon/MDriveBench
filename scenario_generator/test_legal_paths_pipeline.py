#!/usr/bin/env python3
"""
test_legal_paths_pipeline.py

Isolated test script that:
1. Generates legal paths from nodes + crop region (step_02)
2. Runs CSP path picking (step_03)
3. Visualizes results at each stage

Usage:
    python test_legal_paths_pipeline.py --nodes town_nodes/Town02.json --crop 143.5 233.5 145.1 235.1 --output-dir ./test_output
    
    # Or with a specific T-junction crop:
    python test_legal_paths_pipeline.py --nodes town_nodes/Town02.json --t-junction --output-dir ./test_output
"""

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

# Add scenario_generator to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Step 02 imports - Legal path generation
from pipeline.step_02_legal_paths.connectivity import build_connectivity, generate_legal_paths
from pipeline.step_02_legal_paths.models import CropBox
from pipeline.step_02_legal_paths.segments import build_segments, crop_segments, crop_segments_t_junction, load_nodes
from pipeline.step_02_legal_paths.signatures import build_path_signature, build_segments_detailed_for_path, make_path_name

# Step 01 imports - Crop picker
from pipeline.step_01_crop.candidates import build_candidate_crops_for_town
from pipeline.step_01_crop.models import CropKey

# Step 03 imports - Path picking
from pipeline.step_03_path_picker.candidates import (
    _candidate_entry_cardinal,
    _candidate_entry_in_crop,
    _candidate_entry_point,
)
from pipeline.step_03_path_picker.constraints import _infer_road_role_sets
from pipeline.step_03_path_picker.csp import _solve_paths_csp


def draw_crop_region(ax, crop: CropBox, color='green', linestyle='-', linewidth=2, label=None, fill=True, alpha=0.1):
    """Draw crop region rectangle."""
    rect = Rectangle((crop.xmin, crop.ymin), crop.xmax - crop.xmin, crop.ymax - crop.ymin,
                     fill=fill, facecolor=color if fill else 'none',
                     edgecolor=color, linewidth=linewidth, linestyle=linestyle,
                     alpha=alpha if fill else 1.0, label=label, zorder=1)
    ax.add_patch(rect)


def draw_segments(ax, segments, alpha=0.3, color='gray', linewidth=1):
    """Draw road segments (LaneSegment objects)."""
    for seg in segments:
        pts = seg.points  # LaneSegment has .points attribute
        if len(pts) < 2:
            continue
        xs = pts[:, 0]
        ys = pts[:, 1]
        ax.plot(xs, ys, color=color, alpha=alpha, linewidth=linewidth, zorder=1)


def draw_path_from_segments(ax, path_segments, color='blue', linewidth=2.5, alpha=0.8, label=None):
    """Draw a path from its segment list (LaneSegment objects)."""
    first = True
    for seg in path_segments:
        pts = seg.points
        if len(pts) < 2:
            continue
        xs = pts[:, 0]
        ys = pts[:, 1]
        ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=alpha, zorder=5,
                label=label if first else None)
        first = False
        
        # Draw entry point marker for first segment
        if first:
            ax.plot(xs[0], ys[0], 'o', color=color, markersize=10, markeredgecolor='black', zorder=10)


def run_pipeline(nodes_path: str, crop: CropBox, output_dir: str, 
                 max_paths: int = 50, min_path_length: float = 20.0,
                 schema: Optional[Dict] = None, is_t_junction: bool = False,
                 junction_center: Optional[Tuple[float, float]] = None):
    """Run the legal paths generation and CSP pipeline."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"{'='*80}")
    print("STEP 1: LOAD NODES AND BUILD SEGMENTS")
    print(f"{'='*80}")
    
    print(f"Loading nodes from: {nodes_path}")
    data = load_nodes(nodes_path)
    
    print("Building segments...")
    segments = build_segments(data)
    print(f"Total segments: {len(segments)}")
    
    print(f"\nCrop region: x=[{crop.xmin:.1f}, {crop.xmax:.1f}] y=[{crop.ymin:.1f}, {crop.ymax:.1f}]")
    
    # Use T-junction specific cropping if applicable
    if is_t_junction:
        print("Using T-junction crop mode (partial segments must be straight)")
        print(f"Junction center: {junction_center}")
        cropped_segments = crop_segments_t_junction(segments, crop, junction_center=junction_center)
    else:
        cropped_segments = crop_segments(segments, crop)
    print(f"Cropped segments: {len(cropped_segments)}")
    
    # Print segment table with indices for reference
    print(f"\n{'Idx':<5} {'Seg ID':<30} {'R/S/L':<12} {'Straight?':<10} {'Pts':<6} {'Start(x,y)':<20} {'End(x,y)':<20}")
    print("-" * 110)
    for idx, seg in enumerate(cropped_segments):
        start = f"({seg.points[0,0]:.1f}, {seg.points[0,1]:.1f})" if len(seg.points) > 0 else "N/A"
        end = f"({seg.points[-1,0]:.1f}, {seg.points[-1,1]:.1f})" if len(seg.points) > 0 else "N/A"
        straight = "Yes" if seg.is_straight() else "No"
        rsl = f"R{seg.road_id}/S{seg.section_id}/L{seg.lane_id}"
        print(f"{idx:<5} {seg.seg_id:<30} {rsl:<12} {straight:<10} {len(seg.points):<6} {start:<20} {end:<20}")
    
    # =========================================================================
    # STEP 2: BUILD CONNECTIVITY AND GENERATE PATHS
    # =========================================================================
    print(f"\n{'='*80}")
    print("STEP 2: GENERATE LEGAL PATHS")
    print(f"{'='*80}")
    
    print("Building connectivity graph...")
    graph = build_connectivity(cropped_segments, connect_radius_m=6.0, connect_yaw_tol_deg=60.0)
    
    print(f"Generating legal paths (max={max_paths}, min_length={min_path_length}m)...")
    paths = generate_legal_paths(
        segments=cropped_segments,
        adj=graph,
        crop=crop,
        max_paths=max_paths,
        max_depth=5,
        min_path_length=min_path_length,
    )
    print(f"Generated {len(paths)} legal paths")
    
    # Build signatures for each path
    # paths is a list of LegalPath objects
    print("\nBuilding path signatures...")
    candidates = []
    crop_dict = {"xmin": crop.xmin, "xmax": crop.xmax, "ymin": crop.ymin, "ymax": crop.ymax}
    
    for i, path in enumerate(paths):
        sig = build_path_signature(path)  # path is a LegalPath object
        name = make_path_name(i + 1, sig)
        segments_detailed = build_segments_detailed_for_path(path)
        
        candidate = {
            "name": name,
            "signature": sig,
        }
        candidates.append(candidate)
    
    # For T-junctions, filter out paths from other junctions (entry too far from crop)
    if is_t_junction:
        # Use a margin for entry points - paths should start near the crop
        # The crop center is the T-junction; paths from other junctions will have
        # entry points far outside the crop
        entry_margin = 20.0  # Allow entries within 20m of crop boundary
        
        filtered_candidates = []
        for c in candidates:
            pt = _candidate_entry_point(c)
            if pt is None:
                continue
            # Check if entry is within expanded crop bounds
            in_range = (crop.xmin - entry_margin <= pt[0] <= crop.xmax + entry_margin and
                       crop.ymin - entry_margin <= pt[1] <= crop.ymax + entry_margin)
            if in_range:
                filtered_candidates.append(c)
        
        print(f"\nT-junction filter: kept {len(filtered_candidates)}/{len(candidates)} paths (entry within {entry_margin}m of crop)")
        candidates = filtered_candidates
    
    # Print candidates
    print(f"\n{'Name':<60} {'Y':>10} {'Cardinal':>10}")
    print("-" * 85)
    
    for c in candidates:
        name = c.get("name", "")[:55]
        pt = _candidate_entry_point(c)
        y = pt[1] if pt else 0
        cardinal = _candidate_entry_cardinal(c)
        in_crop = _candidate_entry_in_crop(c, crop_dict, margin=10)
        marker = "✓" if in_crop else "✗"
        print(f"{marker} {name:<58} {y:>10.1f} {cardinal:>10}")
    
    # Save legal paths JSON
    legal_paths_data = {
        "nodes": nodes_path,
        "crop_region": crop_dict,
        "candidates": candidates,
    }
    
    legal_paths_file = os.path.join(output_dir, "legal_paths_detailed.json")
    with open(legal_paths_file, "w") as f:
        json.dump(legal_paths_data, f, indent=2)
    print(f"\nSaved: {legal_paths_file}")
    
    # =========================================================================
    # STEP 3: ROLE INFERENCE
    # =========================================================================
    print(f"\n{'='*80}")
    print("STEP 3: ROAD ROLE INFERENCE")
    print(f"{'='*80}")
    
    role_sets = _infer_road_role_sets(candidates)
    print(f"Entry set: {role_sets.get('entry_set', set())}")
    print(f"Main entry: {role_sets.get('main_entry', set())}")
    print(f"Side entry: {role_sets.get('side_entry', set())}")
    
    # Classify candidates
    side_candidates = []
    main_candidates = []
    for c in candidates:
        cardinal = _candidate_entry_cardinal(c)
        if cardinal in role_sets.get('side_entry', set()):
            side_candidates.append(c)
        elif cardinal in role_sets.get('main_entry', set()):
            main_candidates.append(c)
    
    print(f"\nSide-road candidates: {len(side_candidates)}")
    for c in side_candidates:
        pt = _candidate_entry_point(c)
        print(f"  - {c['name'][:50]}... y={pt[1]:.1f}")
    
    print(f"\nMain-road candidates: {len(main_candidates)}")
    
    # =========================================================================
    # STEP 4: RUN CSP (if schema provided)
    # =========================================================================
    csp_result = None
    if schema:
        print(f"\n{'='*80}")
        print("STEP 4: RUN CSP PATH PICKER")
        print(f"{'='*80}")
        
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
        
        print("Vehicles:")
        for v in constraints_obj["vehicles"]:
            print(f"  {v['vehicle']}: maneuver={v['maneuver']}, entry={v['entry_road']}")
        
        try:
            csp_result = _solve_paths_csp(
                constraints_obj=constraints_obj,
                candidates=candidates,
                description=description,
                crop_region=crop_dict,
            )
            
            print("\nCSP Result:")
            for r in csp_result:
                print(f"  {r['vehicle']}: {r['path_name']}")
        except Exception as e:
            print(f"\nCSP Failed: {e}")
    
    # =========================================================================
    # VISUALIZATIONS
    # =========================================================================
    print(f"\n{'='*80}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*80}")
    
    cmap = plt.colormaps['tab10']
    
    # Create segment lookup for drawing paths (LaneSegment objects)
    seg_by_id = {s.seg_id: s for s in cropped_segments}
    
    # VIZ 1: All segments and crop region
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_title(f"Road Segments in Crop Region\n({len(cropped_segments)} segments)", fontsize=14, fontweight='bold')
    
    draw_segments(ax, cropped_segments, alpha=0.5, color='blue', linewidth=1.5)
    draw_crop_region(ax, crop, fill=True, alpha=0.1, color='green', label='Crop Region')
    
    # Mark segment endpoints AND add numeric labels at midpoints
    for idx, seg in enumerate(cropped_segments):
        pts = seg.points
        if len(pts) > 0:
            ax.plot(pts[0, 0], pts[0, 1], 'go', markersize=3, alpha=0.5)
            ax.plot(pts[-1, 0], pts[-1, 1], 'ro', markersize=3, alpha=0.5)
            
            # Add numeric label at midpoint of segment
            mid_idx = len(pts) // 2
            mid_x, mid_y = pts[mid_idx, 0], pts[mid_idx, 1]
            ax.annotate(
                f"{idx}",
                (mid_x, mid_y),
                fontsize=8,
                fontweight='bold',
                ha='center',
                va='center',
                bbox=dict(boxstyle='circle,pad=0.2', facecolor='yellow', edgecolor='black', alpha=0.8)
            )
    
    ax.set_xlim(crop.xmin - 20, crop.xmax + 20)
    ax.set_ylim(crop.ymin - 20, crop.ymax + 20)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    
    out_path = os.path.join(output_dir, "viz1_segments.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()
    
    # VIZ 2: All legal paths
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_title(f"All Legal Paths\n({len(candidates)} paths)", fontsize=14, fontweight='bold')
    
    draw_segments(ax, cropped_segments, alpha=0.15, color='gray', linewidth=0.5)
    draw_crop_region(ax, crop, fill=True, alpha=0.1, color='lightgreen')
    
    for i, c in enumerate(candidates):
        color = cmap(i % 10)
        sig = c.get("signature", {})
        seg_ids = sig.get("segment_ids", [])
        
        # Draw path segments
        for sid in seg_ids:
            seg = seg_by_id.get(sid)
            if seg:
                pts = seg.points
                if len(pts) >= 2:
                    ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=2, alpha=0.7)
        
        # Mark entry point
        pt = _candidate_entry_point(c)
        if pt:
            ax.plot(pt[0], pt[1], 'o', color=color, markersize=8, markeredgecolor='black', zorder=10)
            ax.annotate(c['name'].split('__')[0], (pt[0], pt[1]), fontsize=6, 
                       xytext=(3, 3), textcoords='offset points')
    
    ax.set_xlim(crop.xmin - 20, crop.xmax + 20)
    ax.set_ylim(crop.ymin - 20, crop.ymax + 20)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    
    out_path = os.path.join(output_dir, "viz2_all_paths.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()
    
    # VIZ 3: Side vs Main roads
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Side roads
    ax = axes[0]
    ax.set_title(f"SIDE-Road Paths ({len(side_candidates)})\nCardinals: {role_sets.get('side_entry', set())}", 
                 fontsize=12, fontweight='bold')
    draw_segments(ax, cropped_segments, alpha=0.15, color='gray')
    draw_crop_region(ax, crop, fill=True, alpha=0.1, color='lightgreen')
    
    for i, c in enumerate(side_candidates):
        color = 'blue'
        sig = c.get("signature", {})
        seg_ids = sig.get("segment_ids", [])
        for sid in seg_ids:
            seg = seg_by_id.get(sid)
            if seg:
                pts = seg.points
                if len(pts) >= 2:
                    ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=3, alpha=0.8)
        
        pt = _candidate_entry_point(c)
        if pt:
            in_crop = _candidate_entry_in_crop(c, crop_dict, margin=10)
            ax.plot(pt[0], pt[1], 'o', color='blue' if in_crop else 'red', 
                   markersize=12, markeredgecolor='black', zorder=10)
            ax.annotate(f"{c['name'].split('__')[0]}\ny={pt[1]:.1f}", (pt[0], pt[1]), 
                       fontsize=8, fontweight='bold',
                       xytext=(10, 5), textcoords='offset points',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    ax.set_xlim(crop.xmin - 30, crop.xmax + 30)
    ax.set_ylim(crop.ymin - 30, crop.ymax + 30)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    
    # Main roads
    ax = axes[1]
    ax.set_title(f"MAIN-Road Paths ({len(main_candidates)})\nCardinals: {role_sets.get('main_entry', set())}", 
                 fontsize=12, fontweight='bold')
    draw_segments(ax, cropped_segments, alpha=0.15, color='gray')
    draw_crop_region(ax, crop, fill=True, alpha=0.1, color='lightgreen')
    
    for i, c in enumerate(main_candidates[:20]):  # Limit to 20 for readability
        color = cmap(i % 10)
        sig = c.get("signature", {})
        seg_ids = sig.get("segment_ids", [])
        for sid in seg_ids:
            seg = seg_by_id.get(sid)
            if seg:
                pts = seg.points
                if len(pts) >= 2:
                    ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=2, alpha=0.7)
        
        pt = _candidate_entry_point(c)
        if pt:
            ax.plot(pt[0], pt[1], 'o', color=color, markersize=8, markeredgecolor='black', zorder=10)
    
    ax.set_xlim(crop.xmin - 30, crop.xmax + 30)
    ax.set_ylim(crop.ymin - 30, crop.ymax + 30)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, "viz3_side_vs_main.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()
    
    # VIZ 4: Entry points by Y-coordinate
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_title("Entry Point Y-Coordinates", fontsize=14, fontweight='bold')
    
    names = []
    y_vals = []
    colors = []
    for c in candidates:
        pt = _candidate_entry_point(c)
        if pt:
            names.append(c['name'].split('__')[0])
            y_vals.append(pt[1])
            cardinal = _candidate_entry_cardinal(c)
            if cardinal in role_sets.get('side_entry', set()):
                colors.append('blue')
            else:
                colors.append('gray')
    
    y_pos = range(len(names))
    ax.barh(y_pos, y_vals, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel('Y Coordinate (meters)')
    
    # Draw crop region Y bounds
    ax.axvline(x=crop.ymin, color='green', linestyle='-', linewidth=2, label=f'Crop ymin={crop.ymin:.1f}')
    ax.axvline(x=crop.ymax, color='green', linestyle='-', linewidth=2, label=f'Crop ymax={crop.ymax:.1f}')
    ax.axvline(x=crop.ymin - 10, color='green', linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(x=crop.ymax + 10, color='green', linestyle=':', linewidth=1, alpha=0.5, label='±10m margin')
    ax.legend()
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, "viz4_entry_y_coords.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()
    
    # VIZ 5: CSP Result (if available)
    if csp_result:
        fig, ax = plt.subplots(figsize=(14, 14))
        ax.set_title("CSP Result: Picked Paths", fontsize=14, fontweight='bold')
        
        draw_segments(ax, cropped_segments, alpha=0.15, color='gray')
        draw_crop_region(ax, crop, fill=True, alpha=0.1, color='lightgreen')
        
        cand_by_name = {c['name']: c for c in candidates}
        
        for i, r in enumerate(csp_result):
            name = r['path_name']
            vehicle = r['vehicle']
            c = cand_by_name.get(name)
            if c:
                color = cmap(i % 10)
                sig = c.get("signature", {})
                seg_ids = sig.get("segment_ids", [])
                for sid in seg_ids:
                    seg = seg_by_id.get(sid)
                    if seg:
                        pts = seg.get("points", [])
                        if len(pts) >= 2:
                            xs = [p[0] for p in pts]
                            ys = [p[1] for p in pts]
                            ax.plot(xs, ys, color=color, linewidth=4, alpha=0.9)
                
                pt = _candidate_entry_point(c)
                if pt:
                    ax.annotate(f"{vehicle}\n{name.split('__')[0]}", (pt[0], pt[1]),
                               fontsize=10, fontweight='bold',
                               xytext=(10, 10), textcoords='offset points',
                               bbox=dict(boxstyle='round', facecolor=color, alpha=0.8, edgecolor='black'),
                               color='white')
        
        ax.set_xlim(crop.xmin - 20, crop.xmax + 20)
        ax.set_ylim(crop.ymin - 20, crop.ymax + 20)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        
        out_path = os.path.join(output_dir, "viz5_csp_result.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {out_path}")
        plt.close()
    
    print(f"\n{'='*80}")
    print("DONE")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")
    print(f"Legal paths JSON: {legal_paths_file}")
    
    return candidates, role_sets, csp_result


def main():
    parser = argparse.ArgumentParser(description="Test legal paths generation pipeline")
    parser.add_argument("--nodes", type=str, default="town_nodes/Town02.json", 
                       help="Path to town nodes JSON file")
    parser.add_argument("--town", type=str, default="Town02",
                       help="Town name (used for crop picker)")
    parser.add_argument("--crop", type=float, nargs=4, metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                       help="Crop region: xmin xmax ymin ymax")
    parser.add_argument("--t-junction", action="store_true",
                       help="Use crop picker to find a T-junction crop region")
    parser.add_argument("--output-dir", type=str, default="./legal_paths_test",
                       help="Output directory for results")
    parser.add_argument("--max-paths", type=int, default=50,
                       help="Maximum number of paths to generate")
    parser.add_argument("--min-length", type=float, default=20.0,
                       help="Minimum path length in meters")
    parser.add_argument("--schema", type=str, default=None,
                       help="Optional schema JSON file for CSP testing")
    parser.add_argument("--radius", type=float, default=45.0,
                       help="Crop radius for T-junction picker (default 45m)")
    
    args = parser.parse_args()
    
    # Resolve nodes path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.nodes):
        nodes_path = os.path.join(script_dir, args.nodes)
    else:
        nodes_path = args.nodes
    
    # Determine crop region
    junction_center = None
    if args.t_junction:
        # Use the actual crop picker from step_01_crop
        print(f"Running crop picker to find T-junction in {args.town}...")
        crops = build_candidate_crops_for_town(
            town_name=args.town,
            town_json_path=nodes_path,
            radii=[args.radius],
            min_path_len=args.min_length,
            max_paths=args.max_paths,
            max_depth=8,
        )
        
        # Find T-junction crops
        t_junction_crops = [c for c in crops if c.is_t_junction]
        print(f"Found {len(t_junction_crops)} T-junction crops out of {len(crops)} total crops")
        
        if not t_junction_crops:
            print("ERROR: No T-junction crops found! Listing all crops:")
            for c in crops[:10]:
                print(f"  - center=({c.center_xy[0]:.1f}, {c.center_xy[1]:.1f}), "
                      f"t_junction={c.is_t_junction}, four_way={c.is_four_way}")
            sys.exit(1)
        
        # Pick the first T-junction
        chosen = t_junction_crops[0]
        junction_center = chosen.center_xy
        print(f"Using T-junction crop: center=({chosen.center_xy[0]:.1f}, {chosen.center_xy[1]:.1f})")
        print(f"  Crop bounds: x=[{chosen.crop.xmin:.1f}, {chosen.crop.xmax:.1f}] "
              f"y=[{chosen.crop.ymin:.1f}, {chosen.crop.ymax:.1f}]")
        
        # Convert CropKey to CropBox
        crop = CropBox(xmin=chosen.crop.xmin, xmax=chosen.crop.xmax,
                      ymin=chosen.crop.ymin, ymax=chosen.crop.ymax)
    elif args.crop:
        crop = CropBox(xmin=args.crop[0], xmax=args.crop[1], ymin=args.crop[2], ymax=args.crop[3])
        print(f"Using specified crop: x=[{crop.xmin:.1f}, {crop.xmax:.1f}] y=[{crop.ymin:.1f}, {crop.ymax:.1f}]")
    else:
        print("ERROR: Must specify either --t-junction or --crop")
        sys.exit(1)
    
    # Load schema if provided
    schema = None
    if args.schema and os.path.exists(args.schema):
        with open(args.schema) as f:
            schema = json.load(f)
    
    run_pipeline(
        nodes_path=nodes_path,
        crop=crop,
        output_dir=args.output_dir,
        max_paths=args.max_paths,
        min_path_length=args.min_length,
        schema=schema,
        is_t_junction=args.t_junction,
        junction_center=junction_center,
    )


if __name__ == "__main__":
    main()
