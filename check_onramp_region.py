#!/usr/bin/env python3
"""Check the on-ramp region specified by the user (x: 60-200, y: 190-300)"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

REPO_ROOT = Path("/data2/marco/CoLMDriver")
sys.path.insert(0, str(REPO_ROOT))

from scenario_generator.common import CarlaMap

# Load Town06
carla_map = CarlaMap("Town06")
segs = carla_map.get_segments()

# Region boundaries
xmin, xmax = 60, 200
ymin, ymax = 190, 300

# Find segments in region
segs_in_region = []
for seg in segs:
    start_x, start_y = seg["start"]["x"], seg["start"]["y"]
    end_x, end_y = seg["end"]["x"], seg["end"]["y"]
    
    # Check if segment is in region
    seg_xmin, seg_xmax = min(start_x, end_x), max(start_x, end_x)
    seg_ymin, seg_ymax = min(start_y, end_y), max(start_y, end_y)
    
    if (seg_xmin <= xmax and seg_xmax >= xmin and 
        seg_ymin <= ymax and seg_ymax >= ymin):
        segs_in_region.append(seg)

print(f"Found {len(segs_in_region)} segments in on-ramp region")
print("\nSegment details:")
for i, seg in enumerate(segs_in_region):
    road_id = seg["road_id"]
    section_id = seg["section_id"]
    lane_id = seg["lane_id"]
    start = seg["start"]
    end = seg["end"]
    print(f"  {i}: road={road_id}, section={section_id}, lane={lane_id}")
    print(f"     start=({start['x']:.1f}, {start['y']:.1f}) end=({end['x']:.1f}, {end['y']:.1f})")

# Analyze road connectivity
print("\nRoad connectivity in region:")
roads_in_region = set(seg["road_id"] for seg in segs_in_region)
for road_id in sorted(roads_in_region):
    segs_for_road = [s for s in segs_in_region if s["road_id"] == road_id]
    print(f"  Road {road_id}: {len(segs_for_road)} segments")

# Visualize
fig, ax = plt.subplots(figsize=(12, 10))

# Draw all segments with different colors per road
colors = {}
color_list = plt.cm.tab20(range(len(roads_in_region)))
for i, road_id in enumerate(sorted(roads_in_region)):
    colors[road_id] = color_list[i]

for seg in segs_in_region:
    road_id = seg["road_id"]
    start = seg["start"]
    end = seg["end"]
    
    color = colors[road_id]
    ax.plot([start["x"], end["x"]], [start["y"], end["y"]], 
            color=color, linewidth=2, label=f"Road {road_id}" if seg == segs_in_region[0] else "")
    
    # Draw small circles at endpoints to show direction
    ax.plot(start["x"], start["y"], "o", color=color, markersize=4)

# Draw region boundary
rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, 
                          linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
ax.add_patch(rect)

ax.set_xlim(xmin-10, xmax+10)
ax.set_ylim(ymin-10, ymax+10)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title('Town06 On-Ramp Region (x: 60-200, y: 190-300)')

# Create legend with unique roads
handles = []
labels = []
for road_id in sorted(roads_in_region):
    handles.append(plt.Line2D([0], [0], color=colors[road_id], lw=2))
    labels.append(f"Road {road_id}")
ax.legend(handles, labels, loc='upper left')

output_file = Path("onramp_region_visualization.png")
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\nVisualization saved to {output_file}")
plt.close()

# Check legal paths through this region
print("\nGenerating legal paths through region...")
from scenario_generator.pipeline.step_01_crop.models import CropKey
import scenario_generator.gamelogic_tools.gamelogic_provider as glp

crop = CropKey(xmin, xmax, ymin, ymax)
cb = glp.CropBox(xmin, xmax, ymin, ymax)
segs_crop = glp.crop_segments(segs, cb)

print(f"Segments in crop: {len(segs_crop)}")

adj_crop = glp.build_connectivity(segs_crop)
paths = glp.generate_legal_paths(
    segs_crop, adj_crop, cb,
    min_path_length=20.0,
    max_paths=50,
    max_depth=5,
    allow_within_region_fallback=False
)

print(f"Legal paths found: {len(paths)}")

# Analyze paths for merge geometry
print("\nPath analysis:")
by_exit = {}
for p in paths:
    sig = glp.build_path_signature(p)
    exit_key = (int(sig["exit"]["road_id"]), int(sig["exit"]["section_id"]))
    if exit_key not in by_exit:
        by_exit[exit_key] = set()
    by_exit[exit_key].add(sig["entry_to_exit_turn"])

print(f"Unique exit points: {len(by_exit)}")
for exit_key, turns in by_exit.items():
    print(f"  Exit (road={exit_key[0]}, section={exit_key[1]}): maneuvers={turns}")
    # Check for merge pattern
    if "straight" in turns and ("left" in turns or "right" in turns):
        print(f"    → HAS MERGE PATTERN (straight + turning)")

