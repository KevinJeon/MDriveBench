#!/usr/bin/env python3
"""Debug Town02 T-junction path generation."""

import sys
sys.path.insert(0, 'scenario_generator')

import carla
import numpy as np
from collections import defaultdict

# Import pipeline modules
from pipeline.step_02_legal_paths.segments import extract_lane_segments
from pipeline.step_02_legal_paths.connectivity import build_connectivity, generate_legal_paths
from pipeline.step_02_legal_paths.signatures import classify_turn_world, wrap180


def analyze_junction(world, junction_id, radius=45.0):
    """Analyze a specific junction and its generated paths."""
    carla_map = world.get_map()
    
    # Get junction
    all_junctions = carla_map.get_topology()
    junction_wp = None
    for wp_pair in all_junctions:
        wp = wp_pair[0]
        if abs(wp.transform.location.x - junction_id[0]) < 5 and \
           abs(wp.transform.location.y - junction_id[1]) < 5:
            junction_wp = wp
            break
    
    if not junction_wp:
        print(f"Could not find junction at {junction_id}")
        return
    
    print(f"Analyzing junction at ({junction_wp.transform.location.x:.1f}, {junction_wp.transform.location.y:.1f})")
    print(f"Junction ID: {junction_wp.get_junction().id if junction_wp.is_junction else 'N/A'}")
    print()
    
    # Extract segments in the crop region
    bbox = (
        junction_id[0] - radius, junction_id[0] + radius,
        junction_id[1] - radius, junction_id[1] + radius
    )
    segments = extract_lane_segments(carla_map, bbox, min_length=5.0)
    print(f"Extracted {len(segments)} lane segments")
    
    # Build connectivity
    connectivity = build_connectivity(segments, max_gap=3.0)
    print(f"Built connectivity with {len(connectivity)} connections")
    print()
    
    # Generate legal paths
    paths = generate_legal_paths(
        segments,
        connectivity,
        min_path_len=20.0,
        max_paths=100,
        max_depth=100
    )
    
    print(f"Generated {len(paths)} legal paths")
    print()
    
    # Analyze paths
    maneuver_counts = defaultdict(int)
    road_pairs = defaultdict(list)
    
    for i, path in enumerate(paths):
        # Get start and end segments
        start_seg = path.segments[0]
        end_seg = path.segments[-1]
        
        # Compute heading change
        start_heading = start_seg.heading_at_end()
        end_heading = end_seg.heading_at_start()
        
        # Get maneuver classification
        maneuver = classify_turn_world(start_heading, end_heading)
        maneuver_counts[maneuver] += 1
        
        # Track road pairs
        road_pair = (start_seg.road_id, end_seg.road_id)
        road_pairs[road_pair].append(maneuver)
        
        if i < 10:  # Print first 10 paths
            heading_change = wrap180(end_heading - start_heading)
            print(f"Path {i}:")
            print(f"  Road {start_seg.road_id} lane {start_seg.lane_id} → Road {end_seg.road_id} lane {end_seg.lane_id}")
            print(f"  Start heading: {start_heading:.1f}° → End heading: {end_heading:.1f}°")
            print(f"  Heading change: {heading_change:.1f}°")
            print(f"  Maneuver: {maneuver}")
            print(f"  Length: {path.length():.1f}m")
            print()
    
    print(f"Maneuver counts: {dict(maneuver_counts)}")
    print()
    
    print("Road pair analysis:")
    for road_pair, maneuvers in sorted(road_pairs.items()):
        maneuver_summary = {}
        for m in maneuvers:
            maneuver_summary[m] = maneuver_summary.get(m, 0) + 1
        print(f"  Road {road_pair[0]} → Road {road_pair[1]}: {maneuver_summary}")
    

if __name__ == "__main__":
    # Connect to CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    
    # Load Town02
    world = client.load_world('Town02')
    
    # Analyze the T-junction at (43.6, 301.8)
    analyze_junction(world, (43.6, 301.8), radius=45.0)
