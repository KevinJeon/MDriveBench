#!/usr/bin/env python3
"""
Test script to verify XODR map loading with CARLA 9.10.1
"""

import sys
import time
from pathlib import Path

try:
    import carla
    print("[OK] CARLA Python API imported successfully")
except ImportError as e:
    print(f"[ERROR] Failed to import CARLA: {e}")
    sys.exit(1)

CARLA_HOST = "127.0.0.1"
CARLA_PORT = 2000
XODR_PATH = Path("/data2/marco/CoLMDriver/ucla_v2.xodr")

def test_xodr_loading():
    """Test if XODR map can be loaded successfully"""
    
    print(f"\n{'='*70}")
    print("TEST: XODR Map Loading with CARLA 9.10.1")
    print(f"{'='*70}\n")
    
    # Step 1: Check XODR file
    print(f"[1/4] Checking XODR file...")
    if not XODR_PATH.exists():
        print(f"[ERROR] XODR file not found: {XODR_PATH}")
        return False
    print(f"[OK] XODR file found: {XODR_PATH.stat().st_size} bytes")
    
    # Step 2: Connect to CARLA
    print(f"\n[2/4] Connecting to CARLA at {CARLA_HOST}:{CARLA_PORT}...")
    try:
        client = carla.Client(CARLA_HOST, CARLA_PORT)
        client.set_timeout(30.0)
        print(f"[OK] Connected to CARLA")
    except Exception as e:
        print(f"[ERROR] Failed to connect: {e}")
        print(f"[HINT] Make sure CARLA is running: ./CarlaUE4.sh -port={CARLA_PORT}")
        return False
    
    # Step 3: Load XODR
    print(f"\n[3/4] Loading XODR map...")
    try:
        with open(XODR_PATH, 'r') as f:
            xodr_content = f.read()
        print(f"[OK] XODR file read ({len(xodr_content)} bytes)")
        
        world = client.generate_opendrive_world(xodr_content)
        print(f"[OK] XODR world generated")
    except AttributeError:
        print(f"[ERROR] generate_opendrive_world() not available in this CARLA version")
        print(f"[HINT] CARLA 9.10.1+ with OpenDRIVE support required")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to load XODR: {e}")
        return False
    
    # Step 4: Verify map loaded
    print(f"\n[4/4] Verifying map...")
    try:
        time.sleep(2)  # Wait for world to stabilize
        current_map = world.get_map()
        print(f"[OK] Map loaded: {current_map.name}")
        
        # Get map topology for basic validation
        topology = current_map.get_topology()
        print(f"[OK] Map has {len(topology)} lane pairs in topology")
        
        # Check spawn points
        spawn_points = current_map.get_spawn_points()
        print(f"[OK] Map has {len(spawn_points)} spawn points available")
        
        return True
    except Exception as e:
        print(f"[ERROR] Failed to verify map: {e}")
        return False

if __name__ == "__main__":
    success = test_xodr_loading()
    
    print(f"\n{'='*70}")
    if success:
        print("RESULT: XODR loading test PASSED ✓")
        print("\nYou can now run:")
        print("  python tools/run_custom_eval_xodr.py \\")
        print("    --xodr ucla_v2.xodr \\")
        print("    --routes-dir simulation/leaderboard/data/CustomRoutes/ucla \\")
        print("    --planner tcp \\")
        print("    --port 2000")
    else:
        print("RESULT: XODR loading test FAILED ✗")
        print("\nPlease fix the issues above and try again.")
    print(f"{'='*70}\n")
    
    sys.exit(0 if success else 1)
