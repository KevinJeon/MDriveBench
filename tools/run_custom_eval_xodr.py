#!/usr/bin/env python3
"""
Wrapper for run_custom_eval.py that adds support for custom OpenDRIVE (XODR) maps in CARLA 9.10.1

This script:
1. Loads a custom XODR map into CARLA
2. Runs the leaderboard evaluation

Usage:
    python tools/run_custom_eval_xodr.py \
        --xodr /data2/marco/CoLMDriver/ucla_v2.xodr \
        --routes-dir simulation/leaderboard/data/CustomRoutes/ucla \
        --town-name ucla \
        --planner tcp \
        --port 2000
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

try:
    import carla
except ImportError:
    print("[ERROR] CARLA Python API not found")
    sys.exit(1)


def load_xodr_map(carla_host: str, carla_port: int, xodr_path: Path, town_name: str = "Custom") -> bool:
    """
    Load an OpenDRIVE (XODR) map into a running CARLA server using generate_opendrive_world.
    """
    if not xodr_path.exists():
        print(f"[ERROR] XODR file not found: {xodr_path}")
        return False
    
    print(f"[INFO] Connecting to CARLA at {carla_host}:{carla_port}...")
    try:
        client = carla.Client(carla_host, carla_port)
        client.set_timeout(30.0)
    except Exception as e:
        print(f"[ERROR] Failed to connect to CARLA: {e}")
        return False
    
    print(f"[INFO] Reading OpenDRIVE file: {xodr_path}")
    try:
        with open(xodr_path, 'r') as f:
            xodr_content = f.read()
        print(f"[INFO] XODR file size: {len(xodr_content)} bytes")
    except Exception as e:
        print(f"[ERROR] Failed to read XODR file: {e}")
        return False
    
    print(f"[INFO] Loading OpenDRIVE world...")
    try:
        world = client.generate_opendrive_world(xodr_content)
        print(f"[OK] Successfully loaded OpenDRIVE map")
        
        # Verify the map
        current_map = world.get_map()
        print(f"[INFO] Current map: {current_map.name}")
        
        return True
    except Exception as e:
        print(f"[ERROR] Failed to load XODR map: {e}")
        print(f"[ERROR] Make sure CARLA 9.10.1 is built with OpenDRIVE support")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run custom eval with OpenDRIVE (XODR) map support",
    )
    
    parser.add_argument(
        "--xodr",
        type=Path,
        required=True,
        help="Path to the XODR file",
    )
    parser.add_argument(
        "--routes-dir",
        type=Path,
        required=True,
        help="Directory containing route XML files",
    )
    parser.add_argument(
        "--town-name",
        default="Custom",
        help="Town name for the XODR map (default: Custom)",
    )
    parser.add_argument(
        "--planner",
        choices=["tcp", "colmdriver", "vad", "uniad", "codriving", "colmdriver_rulebase", "lmdrive"],
        default="tcp",
        help="Planner to use",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=2000,
        help="CARLA server port",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="CARLA server host",
    )
    parser.add_argument(
        "--skip-xodr-load",
        action="store_true",
        help="Skip loading XODR (assume already loaded in CARLA)",
    )
    
    # Pass-through arguments
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--scenario-name", type=str)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--agent", type=str)
    parser.add_argument("--agent-config", type=str)
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.xodr.exists():
        print(f"[ERROR] XODR file not found: {args.xodr}")
        sys.exit(1)
    
    if not args.routes_dir.exists():
        print(f"[ERROR] Routes directory not found: {args.routes_dir}")
        sys.exit(1)
    
    # Load XODR map into CARLA
    if not args.skip_xodr_load:
        print(f"\n{'='*70}")
        print("STEP 1: Loading OpenDRIVE (XODR) map into CARLA")
        print(f"{'='*70}\n")
        
        if not load_xodr_map(args.host, args.port, args.xodr, args.town_name):
            print(f"\n[ERROR] Failed to load XODR map. Aborting.")
            sys.exit(1)
        
        print(f"\n[INFO] Waiting for CARLA to finish loading...")
        time.sleep(5)
    
    # Now update route XML files to use the correct town name
    print(f"\n{'='*70}")
    print("STEP 2: Preparing routes for the loaded map")
    print(f"{'='*70}\n")
    
    # Update all route files to reference the loaded map
    for xml_file in args.routes_dir.glob("*.xml"):
        try:
            with open(xml_file, 'r') as f:
                content = f.read()
            
            # Replace town attribute with the loaded map name
            # This is a simple replacement - assumes format: town="..."
            original = content
            
            # Handle both town="..." and town='...'
            import re
            content = re.sub(r'town="[^"]*"', f'town="{args.town_name}"', content)
            content = re.sub(r"town='[^']*'", f"town='{args.town_name}'", content)
            
            if content != original:
                with open(xml_file, 'w') as f:
                    f.write(content)
                print(f"[OK] Updated {xml_file.name} to use town='{args.town_name}'")
        except Exception as e:
            print(f"[WARN] Could not update {xml_file.name}: {e}")
    
    # Run the evaluation
    print(f"\n{'='*70}")
    print("STEP 3: Running leaderboard evaluation")
    print(f"{'='*70}\n")
    
    cmd = [
        sys.executable,
        "tools/run_custom_eval.py",
        "--routes-dir",
        str(args.routes_dir),
        "--planner",
        args.planner,
        "--port",
        str(args.port),
    ]
    
    if args.overwrite:
        cmd.append("--overwrite")
    if args.scenario_name:
        cmd.extend(["--scenario-name", args.scenario_name])
    if args.repetitions:
        cmd.extend(["--repetitions", str(args.repetitions)])
    if args.agent:
        cmd.extend(["--agent", args.agent])
    if args.agent_config:
        cmd.extend(["--agent-config", args.agent_config])
    
    print(f"[INFO] Running: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd="/data2/marco/CoLMDriver")
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
