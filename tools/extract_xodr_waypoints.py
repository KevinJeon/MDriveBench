#!/usr/bin/env python3
"""
Extract waypoints from an OpenDRIVE (XODR) file for use in route XML files.

This utility helps you identify valid coordinates and roads in your custom map
that you can use for creating route definitions.

Usage:
    python extract_xodr_waypoints.py --xodr ucla_v2.xodr --list-roads
    python extract_xodr_waypoints.py --xodr ucla_v2.xodr --road 0 --count 10
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Optional
import json


def parse_xodr(xodr_path: Path) -> ET.Element:
    """Parse an OpenDRIVE XML file."""
    tree = ET.parse(xodr_path)
    return tree.getroot()


def get_road_ids(root: ET.Element) -> List[str]:
    """Get all road IDs from the XODR file."""
    roads = root.findall("road")
    return [road.get("id") for road in roads if road.get("id")]


def get_road_info(root: ET.Element, road_id: str) -> Optional[dict]:
    """Get information about a specific road."""
    for road in root.findall("road"):
        if road.get("id") == road_id:
            return {
                "id": road.get("id"),
                "name": road.get("name", ""),
                "length": float(road.get("length", 0)),
                "junction": road.get("junction", "-1"),
            }
    return None


def extract_geometry_points(root: ET.Element, road_id: str, count: int = 10) -> List[Tuple[float, float, float]]:
    """
    Extract geometry points from a road's planView section.
    
    Returns list of (x, y, heading) tuples
    """
    for road in root.findall("road"):
        if road.get("id") != road_id:
            continue
        
        planview = road.find("planView")
        if planview is None:
            return []
        
        geometries = planview.findall("geometry")
        points = []
        
        for geom in geometries:
            s = float(geom.get("s", 0))
            x = float(geom.get("x", 0))
            y = float(geom.get("y", 0))
            hdg = float(geom.get("hdg", 0))
            
            # Get the length of this geometry
            line = geom.find("line")
            arc = geom.find("arc")
            spiral = geom.find("spiral")
            
            length = 0
            if line is not None:
                # Line geometry - use parametric approach
                length = 50.0  # Default segment length
            elif arc is not None:
                curvature = float(arc.get("curvature", 0))
                if curvature != 0:
                    length = 1.0 / abs(curvature) if abs(curvature) > 1e-6 else 50.0
                else:
                    length = 50.0
            
            points.append((x, y, hdg))
        
        return points[:count]
    
    return []


def list_roads(root: ET.Element) -> None:
    """List all roads in the XODR file with basic info."""
    roads = root.findall("road")
    print(f"\nFound {len(roads)} roads in the XODR file:\n")
    print(f"{'Road ID':<10} {'Name':<30} {'Length (m)':<15} {'Junction':<10}")
    print("-" * 65)
    
    for road in roads:
        rid = road.get("id", "?")
        name = road.get("name", "")[:28]
        length = float(road.get("length", 0))
        junction = road.get("junction", "-1")
        print(f"{rid:<10} {name:<30} {length:<15.2f} {junction:<10}")
    
    print()


def extract_waypoints(root: ET.Element, road_id: str, count: int = 10) -> None:
    """Extract and display waypoints from a road."""
    
    road_info = get_road_info(root, road_id)
    if not road_info:
        print(f"[ERROR] Road {road_id} not found!")
        return
    
    print(f"\n[INFO] Road: {road_info['name']} (ID: {road_id})")
    print(f"[INFO] Length: {road_info['length']:.2f}m")
    print(f"[INFO] Junction: {road_info['junction']}")
    
    points = extract_geometry_points(root, road_id, count)
    
    if not points:
        print("[WARN] No geometry points found for this road")
        return
    
    print(f"\n[INFO] Extracted {len(points)} waypoint(s):\n")
    print(f"{'Index':<6} {'X':<15} {'Y':<15} {'Heading (rad)':<15}")
    print("-" * 52)
    
    for i, (x, y, hdg) in enumerate(points):
        print(f"{i:<6} {x:<15.6f} {y:<15.6f} {hdg:<15.6f}")
    
    print(f"\n[INFO] Example route XML:")
    print("""
<route id="custom_route_001" town="UCLA" role="ego">""")
    
    for x, y, hdg in points:
        print(f'  <waypoint x="{x:.6f}" y="{y:.6f}" z="0.0" yaw="{hdg:.6f}" pitch="0.0" roll="0.0"/>')
    
    print("""</route>
""")
    
    # Also save as JSON for easy copying
    json_output = {
        "road_id": road_id,
        "road_name": road_info['name'],
        "waypoints": [
            {"x": x, "y": y, "yaw": hdg, "z": 0.0}
            for x, y, hdg in points
        ]
    }
    
    print("[INFO] Also available as JSON:")
    print(json.dumps(json_output, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Extract waypoints from OpenDRIVE XODR files",
        epilog="""
Examples:
  # List all roads
  python extract_xodr_waypoints.py --xodr ucla_v2.xodr --list-roads
  
  # Extract waypoints from road 0
  python extract_xodr_waypoints.py --xodr ucla_v2.xodr --road 0 --count 10
  
  # Extract points from multiple roads
  python extract_xodr_waypoints.py --xodr ucla_v2.xodr --road 0 1 2 --count 5
        """
    )
    
    parser.add_argument(
        "--xodr",
        type=Path,
        required=True,
        help="Path to OpenDRIVE XODR file"
    )
    parser.add_argument(
        "--list-roads",
        action="store_true",
        help="List all roads in the XODR file"
    )
    parser.add_argument(
        "--road",
        nargs="+",
        help="Road ID(s) to extract waypoints from"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of waypoints to extract from each road (default: 10)"
    )
    
    args = parser.parse_args()
    
    if not args.xodr.exists():
        print(f"[ERROR] File not found: {args.xodr}")
        return
    
    print(f"[INFO] Loading XODR file: {args.xodr}")
    root = parse_xodr(args.xodr)
    print("[OK] Loaded successfully")
    
    if args.list_roads:
        list_roads(root)
    elif args.road:
        for road_id in args.road:
            extract_waypoints(root, road_id, args.count)
    else:
        print("[INFO] Use --list-roads to see available roads")
        print("[INFO] Use --road <id> to extract waypoints")
        parser.print_help()


if __name__ == "__main__":
    main()
