#!/usr/bin/env python3
"""
Setup script to prepare UCLA custom map for CARLA 9.10.1.

This script helps integrate a custom OpenDRIVE map (ucla_v2.xodr) into CARLA
and prepares it for use with run_custom_eval.py.

Usage:
    python setup_ucla_map.py --xodr /path/to/ucla_v2.xodr --output-dir CustomRoutes/ucla
"""

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Optional


def setup_custom_map(
    xodr_path: Path,
    output_dir: Path,
    carla_root: Optional[Path] = None,
    town_name: str = "UCLA",
) -> None:
    """
    Set up a custom XODR map for CARLA.
    
    Args:
        xodr_path: Path to the OpenDRIVE (.xodr) file
        output_dir: Output directory for scenario files
        carla_root: Path to CARLA installation (auto-detected if not provided)
        town_name: Name for the custom map/town
    """
    xodr_path = Path(xodr_path).resolve()
    output_dir = Path(output_dir).resolve()
    
    if not xodr_path.exists():
        raise FileNotFoundError(f"XODR file not found: {xodr_path}")
    
    print(f"[INFO] Setting up custom map from {xodr_path}")
    print(f"[INFO] Town name: {town_name}")
    print(f"[INFO] Output directory: {output_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy XODR file to output directory
    xodr_dest = output_dir / xodr_path.name
    shutil.copy(xodr_path, xodr_dest)
    print(f"[OK] Copied XODR file to {xodr_dest}")
    
    # Create a metadata file for the custom map
    metadata = {
        "name": town_name,
        "xodr_file": str(xodr_path.name),
        "description": f"Custom map: {town_name}",
        "carla_version": "9.10.1",
    }
    
    import json
    metadata_file = output_dir / "map_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[OK] Created metadata file: {metadata_file}")
    
    # Create sample route XML files
    create_sample_routes(output_dir, town_name)
    
    print(f"\n[SUCCESS] Custom map setup complete!")
    print(f"[INFO] To use this map with run_custom_eval.py:")
    print(f"  1. Copy {xodr_dest} to your CARLA server's content directory")
    print(f"  2. Run: python tools/run_custom_eval.py --routes-dir {output_dir} --planner tcp --custom-map {town_name}")


def create_sample_routes(output_dir: Path, town_name: str) -> None:
    """Create sample route XML files for the custom map."""
    
    # Create actors subdirectory
    actors_dir = output_dir / "actors"
    actors_dir.mkdir(exist_ok=True)
    
    # Create a sample ego route XML
    # These coordinates are placeholders - user should modify them based on their map
    sample_route_xml = f"""<?xml version="1.0" encoding="utf-8"?>
<routes>
  <route id="sample_001" town="{town_name}" role="ego">
    <waypoint x="0.0" y="0.0" z="0.0" yaw="0.0" pitch="0.0" roll="0.0"/>
    <waypoint x="50.0" y="0.0" z="0.0" yaw="0.0" pitch="0.0" roll="0.0"/>
    <waypoint x="100.0" y="0.0" z="0.0" yaw="0.0" pitch="0.0" roll="0.0"/>
  </route>
</routes>
"""
    
    route_file = output_dir / f"{town_name.lower()}_custom_ego_vehicle_0.xml"
    with open(route_file, "w") as f:
        f.write(sample_route_xml)
    print(f"[OK] Created sample route: {route_file}")
    
    # Create manifest
    manifest = {
        "ego": [
            {
                "file": f"{town_name.lower()}_custom_ego_vehicle_0.xml",
                "kind": "ego",
                "name": "Vehicle 1",
                "route_id": "sample_001",
                "town": town_name,
                "speed": 8.0
            }
        ],
        "npc": [],
        "pedestrian": [],
        "bicycle": [],
        "static": []
    }
    
    import json
    manifest_file = output_dir / "actors_manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[OK] Created actors manifest: {manifest_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Setup custom OpenDRIVE map for CARLA",
        epilog="Example: python setup_ucla_map.py --xodr ucla_v2.xodr --output-dir simulation/leaderboard/data/CustomRoutes/ucla"
    )
    parser.add_argument(
        "--xodr",
        type=Path,
        required=True,
        help="Path to the OpenDRIVE (.xodr) file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("simulation/leaderboard/data/CustomRoutes/ucla"),
        help="Output directory for scenario files (default: simulation/leaderboard/data/CustomRoutes/ucla)"
    )
    parser.add_argument(
        "--town-name",
        default="UCLA",
        help="Name for the custom town/map (default: UCLA)"
    )
    parser.add_argument(
        "--carla-root",
        type=Path,
        help="Path to CARLA installation (auto-detected if not provided)"
    )
    
    args = parser.parse_args()
    
    try:
        setup_custom_map(
            args.xodr,
            args.output_dir,
            carla_root=args.carla_root,
            town_name=args.town_name,
        )
    except Exception as e:
        print(f"[ERROR] {e}")
        raise


if __name__ == "__main__":
    main()
