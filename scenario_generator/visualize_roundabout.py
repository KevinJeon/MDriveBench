#!/usr/bin/env python3
"""Visualize roundabout segments and legal paths for Town03."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.step_02_legal_paths.segments import load_nodes, build_segments, crop_segments
from pipeline.step_02_legal_paths.connectivity import build_connectivity, identify_boundary_segments, generate_legal_paths
from pipeline.step_02_legal_paths.models import CropBox
import matplotlib.pyplot as plt
from collections import Counter

# Town03 roundabout bounds (hardcoded)
ROUNDABOUT_CROP = CropBox(xmin=-60.0, xmax=55.0, ymin=-60.0, ymax=40.0)

def main():
    town_json = os.path.join(os.path.dirname(__file__), "town_nodes", "Town03.json")
    
    print("=" * 60)
    print("ROUNDABOUT VISUALIZATION - Town03")
    print("=" * 60)
    print(f"Crop bounds: x=[{ROUNDABOUT_CROP.xmin}, {ROUNDABOUT_CROP.xmax}], y=[{ROUNDABOUT_CROP.ymin}, {ROUNDABOUT_CROP.ymax}]")
    
    # Load and build segments
    print("\n--- Loading nodes ---")
    data = load_nodes(town_json)
    print(f"Loaded data keys: {list(data.keys())}")
    
    print("\n--- Building segments ---")
    all_segments = build_segments(data)
    print(f"Built {len(all_segments)} total segments")
    
    print("\n--- Cropping segments to roundabout ---")
    cropped = crop_segments(all_segments, ROUNDABOUT_CROP)
    print(f"Cropped to {len(cropped)} segments in roundabout region")
    
    # Build connectivity
    print("\n--- Building connectivity ---")
    adj = build_connectivity(cropped)
    total_edges = sum(len(neighbors) for neighbors in adj)
    print(f"Connectivity graph: {total_edges} edges")
    
    # Identify boundary segments
    print("\n--- Identifying boundary segments ---")
    entry_segs, exit_segs = identify_boundary_segments(cropped, ROUNDABOUT_CROP, adj=adj)
    print(f"Entry segments: {len(entry_segs)}")
    print(f"Exit segments: {len(exit_segs)}")
    
    # Generate legal paths
    print("\n--- Generating legal paths ---")
    paths = generate_legal_paths(cropped, entry_segs, exit_segs, adj)
    print(f"Generated {len(paths)} legal paths")
    
    # Analyze paths
    print("\n--- Path Analysis ---")
    entry_dirs = sorted(set(p.entry_dir for p in paths))
    exit_dirs = sorted(set(p.exit_dir for p in paths))
    turns = sorted(set(p.turn for p in paths))
    
    print(f"Entry directions: {entry_dirs}")
    print(f"Exit directions: {exit_dirs}")
    print(f"Turn types: {turns}")
    
    print("\nPaths by turn type:")
    turn_counts = Counter(p.turn for p in paths)
    for turn, count in sorted(turn_counts.items()):
        print(f"  {turn}: {count}")
    
    print("\nPaths by entry->exit:")
    route_counts = Counter((p.entry_dir, p.exit_dir) for p in paths)
    for (entry, exit_d), count in sorted(route_counts.items()):
        print(f"  {entry} -> {exit_d}: {count}")
    
    # Sample paths
    print("\n--- Sample Paths (first 10) ---")
    for i, p in enumerate(paths[:10]):
        print(f"  {i+1}. {p.entry_dir} -> {p.exit_dir} ({p.turn}), {len(p.waypoints)} waypoints")
    
    # Visualize
    print("\n--- Generating visualization ---")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: All segments
    ax1 = axes[0]
    ax1.set_title(f"All Cropped Segments ({len(cropped)})")
    for seg in cropped:
        xs = [pt[0] for pt in seg.points]
        ys = [pt[1] for pt in seg.points]
        ax1.plot(xs, ys, 'b-', alpha=0.5, linewidth=1)
    # Mark entry/exit
    for seg in entry_segs:
        ax1.plot(seg.points[0][0], seg.points[0][1], 'go', markersize=8)
    for seg in exit_segs:
        ax1.plot(seg.points[-1][0], seg.points[-1][1], 'r^', markersize=8)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_aspect('equal')
    ax1.legend(['Segments', 'Entry', 'Exit'], loc='upper right')
    
    # Plot 2: Paths colored by turn type
    ax2 = axes[1]
    ax2.set_title(f"Paths by Turn Type ({len(paths)})")
    colors = {'straight': 'blue', 'left': 'green', 'right': 'red', 'uturn': 'purple'}
    for p in paths:
        xs = [pt[0] for pt in p.waypoints]
        ys = [pt[1] for pt in p.waypoints]
        ax2.plot(xs, ys, color=colors.get(p.turn, 'gray'), alpha=0.3, linewidth=0.5)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_aspect('equal')
    
    # Plot 3: Paths colored by entry direction
    ax3 = axes[2]
    ax3.set_title(f"Paths by Entry Direction")
    dir_colors = {'N': 'blue', 'S': 'red', 'E': 'green', 'W': 'orange'}
    for p in paths:
        xs = [pt[0] for pt in p.waypoints]
        ys = [pt[1] for pt in p.waypoints]
        ax3.plot(xs, ys, color=dir_colors.get(p.entry_dir, 'gray'), alpha=0.3, linewidth=0.5)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_aspect('equal')
    
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "roundabout_visualization.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close()
    
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

if __name__ == "__main__":
    main()
