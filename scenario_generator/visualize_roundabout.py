#!/usr/bin/env python3
"""Visualize roundabout segments and legal paths for Town03."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.step_02_legal_paths.segments import load_nodes, build_segments, crop_segments
from pipeline.step_02_legal_paths.connectivity import build_connectivity, identify_boundary_segments, generate_legal_paths
from pipeline.step_02_legal_paths.signatures import build_path_signature
from pipeline.step_02_legal_paths.models import CropBox
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
import numpy as np

# Town03 roundabout bounds (hardcoded)
ROUNDABOUT_CROP = CropBox(xmin=-60.0, xmax=55.0, ymin=-35.0, ymax=40.0)

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
    # Use min_points=2 to include short connector segments in roundabout
    all_segments = build_segments(data, min_points=2)
    print(f"Built {len(all_segments)} total segments")
    
    # First, show ALL segments to understand where the roundabout is
    print("\n--- Finding roundabout area ---")
    # Look for circular segments (segments that curve back)
    all_xs = []
    all_ys = []
    for seg in all_segments:
        all_xs.extend([pt[0] for pt in seg.points])
        all_ys.extend([pt[1] for pt in seg.points])
    print(f"Full map extent: x=[{min(all_xs):.1f}, {max(all_xs):.1f}], y=[{min(all_ys):.1f}, {max(all_ys):.1f}]")
    
    print("\n--- Cropping segments to roundabout ---")
    cropped = crop_segments(all_segments, ROUNDABOUT_CROP)
    print(f"Cropped to {len(cropped)} segments in roundabout region")
    
    # Build connectivity
    print("\n--- Building connectivity ---")
    adj = build_connectivity(cropped)
    total_edges = sum(len(neighbors) for neighbors in adj)
    print(f"Connectivity graph: {total_edges} edges")
    
    # Debug: Find segments near y=10 that might be missing
    print("\n--- Checking for gaps in connectivity near y=10 ---")
    # Check which segments around y=10, x around 0-20 have connectivity issues
    gap_region_segs = []
    for i, seg in enumerate(cropped):
        cx = (seg.bbox()[0] + seg.bbox()[1]) / 2
        cy = (seg.bbox()[2] + seg.bbox()[3]) / 2
        # Look for segments in the gap region (adjust based on visualization)
        if -5 < cx < 50 and 0 < cy < 15:
            gap_region_segs.append((i, seg))
    print(f"Segments in gap region (x: -5 to 50, y: 0 to 15): {len(gap_region_segs)}")
    for i, seg in gap_region_segs:
        bbox = seg.bbox()
        incoming = sum(1 for j, neighbors in enumerate(adj) if i in neighbors)
        outgoing = len(adj[i])
        print(f"  [{i}] Seg {seg.seg_id}: road={seg.road_id}, lane={seg.lane_id}, "
              f"x=[{bbox[0]:.1f},{bbox[1]:.1f}], y=[{bbox[2]:.1f},{bbox[3]:.1f}], "
              f"in={incoming}, out={outgoing}")
    
    # Identify boundary segments
    print("\n--- Identifying boundary segments ---")
    entry_segs, exit_segs = identify_boundary_segments(cropped, ROUNDABOUT_CROP, adj=adj)
    print(f"Entry segments: {len(entry_segs)}")
    print(f"Exit segments: {len(exit_segs)}")
    
    # Print details of entry/exit segments
    print("\nEntry segment details:")
    for idx in entry_segs:
        seg = cropped[idx]
        start = seg.points[0]
        end = seg.points[-1]
        heading = np.degrees(np.arctan2(end[1] - start[1], end[0] - start[0]))
        print(f"  [{idx}] road={seg.road_id}, lane={seg.lane_id}, "
              f"start=({start[0]:.1f}, {start[1]:.1f}), end=({end[0]:.1f}, {end[1]:.1f}), "
              f"heading={heading:.0f}°")
    
    print("\nExit segment details:")
    for idx in exit_segs:
        seg = cropped[idx]
        start = seg.points[0]
        end = seg.points[-1]
        heading = np.degrees(np.arctan2(end[1] - start[1], end[0] - start[0]))
        print(f"  [{idx}] road={seg.road_id}, lane={seg.lane_id}, "
              f"start=({start[0]:.1f}, {start[1]:.1f}), end=({end[0]:.1f}, {end[1]:.1f}), "
              f"heading={heading:.0f}°")
    
    # Generate legal paths
    print("\n--- Generating legal paths ---")
    paths = generate_legal_paths(cropped, adj, ROUNDABOUT_CROP, roundabout_mode=True)
    print(f"Generated {len(paths)} legal paths")
    
    # Analyze paths
    print("\n--- Path Analysis ---")
    
    # Build signatures for each path
    sigs = [build_path_signature(p) for p in paths]
    
    # Track which entry segments are actually being used
    print("\nPaths by entry segment:")
    entry_seg_counts = Counter()
    for p, s in zip(paths, sigs):
        first_seg = p.segments[0]
        entry_seg_counts[first_seg.seg_id] += 1
    for seg_id, count in sorted(entry_seg_counts.items()):
        # Find this segment's details
        for idx in entry_segs:
            seg = cropped[idx]
            if seg.seg_id == seg_id:
                start = seg.points[0]
                print(f"  Seg {seg_id} (road={seg.road_id}): {count} paths, "
                      f"start=({start[0]:.1f}, {start[1]:.1f})")
                break
    
    entry_dirs = sorted(set(s["entry"]["cardinal4"] for s in sigs))
    exit_dirs = sorted(set(s["exit"]["cardinal4"] for s in sigs))
    turns = sorted(set(s["entry_to_exit_turn"] for s in sigs))
    
    print(f"Entry directions: {entry_dirs}")
    print(f"Exit directions: {exit_dirs}")
    print(f"Turn types: {turns}")
    
    print("\nPaths by turn type:")
    turn_counts = Counter(s["entry_to_exit_turn"] for s in sigs)
    for turn, count in sorted(turn_counts.items()):
        print(f"  {turn}: {count}")
    
    print("\nPaths by entry->exit:")
    route_counts = Counter((s["entry"]["cardinal4"], s["exit"]["cardinal4"]) for s in sigs)
    for (entry, exit_d), count in sorted(route_counts.items()):
        print(f"  {entry} -> {exit_d}: {count}")
    
    # Sample paths
    print("\n--- Sample Paths (first 10) ---")
    for i, (p, s) in enumerate(zip(paths[:10], sigs[:10])):
        print(f"  {i+1}. {s['entry']['cardinal4']} -> {s['exit']['cardinal4']} ({s['entry_to_exit_turn']}), {len(p.segments)} segments")
        # Show segment details for debugging jitter
        for j, seg in enumerate(p.segments):
            print(f"      [{j}] road={seg.road_id}, lane={seg.lane_id}, len={seg.length():.1f}m")
    
    # Visualize
    print("\n--- Generating visualization ---")
    
    # Helper to get all points for a path
    def path_points(p):
        pts = []
        for seg in p.segments:
            pts.extend(seg.points.tolist())
        return pts
    
    # Generate distinct colors for each path
    import colorsys
    n_paths = len(paths)
    path_colors = []
    for i in range(n_paths):
        hue = i / n_paths
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        path_colors.append(rgb)
    
    # Single large figure showing all paths with unique colors
    fig, ax = plt.subplots(1, 1, figsize=(16, 14))
    ax.set_title(f"All {len(paths)} Legal Paths - Town03 Roundabout")
    
    # Draw each path with its own color
    for i, (p, s) in enumerate(zip(paths, sigs)):
        pts = path_points(p)
        xs = [pt[0] for pt in pts]
        ys = [pt[1] for pt in pts]
        label = f"{s['entry']['cardinal4']}→{s['exit']['cardinal4']} ({s['entry_to_exit_turn']})"
        ax.plot(xs, ys, color=path_colors[i], alpha=0.7, linewidth=1.5, label=label if i < 20 else None)
        # Mark start with a dot
        ax.plot(xs[0], ys[0], 'o', color=path_colors[i], markersize=6)
        # Mark end with an arrow-like triangle
        ax.plot(xs[-1], ys[-1], '^', color=path_colors[i], markersize=6)
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Add legend for first 20 paths (otherwise too cluttered)
    if len(paths) <= 20:
        ax.legend(loc='upper left', fontsize=8)
    
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "roundabout_visualization.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close()
    
    # Also create a grid view showing paths individually
    print("--- Generating individual path grid ---")
    n_show = len(paths)  # Show ALL paths
    cols = 10
    rows = (n_show + cols - 1) // cols
    fig2, axes2 = plt.subplots(rows, cols, figsize=(20, 4 * rows))
    axes2 = axes2.flatten() if n_show > 1 else [axes2]
    
    for i in range(n_show):
        ax = axes2[i]
        p, s = paths[i], sigs[i]
        pts = path_points(p)
        xs = [pt[0] for pt in pts]
        ys = [pt[1] for pt in pts]
        
        # Draw all segments in gray for context
        for seg in cropped:
            sx = [pt[0] for pt in seg.points]
            sy = [pt[1] for pt in seg.points]
            ax.plot(sx, sy, 'lightgray', linewidth=0.5)
        
        # Draw this path in color
        ax.plot(xs, ys, color=path_colors[i], linewidth=2)
        ax.plot(xs[0], ys[0], 'go', markersize=8)  # Start = green
        ax.plot(xs[-1], ys[-1], 'r^', markersize=8)  # End = red
        
        ax.set_title(f"Path {i+1}: {s['entry']['cardinal4']}→{s['exit']['cardinal4']} ({s['entry_to_exit_turn']})", fontsize=9)
        ax.set_aspect('equal')
        ax.set_xlim(ROUNDABOUT_CROP.xmin - 5, ROUNDABOUT_CROP.xmax + 5)
        ax.set_ylim(ROUNDABOUT_CROP.ymin - 5, ROUNDABOUT_CROP.ymax + 5)
    
    # Hide empty subplots
    for i in range(n_show, len(axes2)):
        axes2[i].axis('off')
    
    plt.tight_layout()
    out_path2 = os.path.join(os.path.dirname(__file__), "roundabout_paths_grid.png")
    plt.savefig(out_path2, dpi=120)
    print(f"Saved: {out_path2}")
    plt.close()
    
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

if __name__ == "__main__":
    main()
