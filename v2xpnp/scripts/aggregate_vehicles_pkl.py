#!/usr/bin/env python3
"""Aggregate vehicle trajectories from V2XPnP trajectory database PKL into CSV and visualizations.

Given a trajectory database PKL file, this script extracts vehicle positions at each timestep
and produces a CSV where:
  - Each row is a vehicle id.
  - The second column is the obj_type (if available).
  - Each subsequent column corresponds to a timestep.
    The cell value is a JSON object containing location [x, y, z].
Missing detections are left blank.

If --viz is provided, a PNG is written per timestep showing top-down boxes colored by obj_type.
If --viz-over-time is provided, frames are stitched into a single GIF.

Usage:
    python tools/aggregate_vehicles_pkl.py \
        --pkl-file /data2/marco/CoLMDriver/v2xpnp/V2XPnP_Sample_Data/trajectory_database_sample.pkl \
        --output vehicles_from_pkl.csv \
        --viz-over-time \
        --map-pkl /data2/marco/CoLMDriver/v2xpnp/map/v2x_intersection_vector_map.pkl
"""

import argparse
import csv
import json
import math
import os
import pickle
import yaml
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib import patches, transforms

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover
    imageio = None

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required: pip install pyyaml") from exc

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise SystemExit("NumPy is required: pip install numpy") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate vehicles from trajectory database PKL")
    parser.add_argument("--pkl-file", required=True, help="Path to trajectory_database.pkl")
    parser.add_argument(
        "--yaml-dir",
        default=None,
        help="Directory containing YAML folders to extract obj_type (e.g., .../2023-03-17-16-12-12_3_0)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Defaults to same directory as pkl with _vehicles.csv suffix",
    )
    parser.add_argument(
        "--scenario",
        type=int,
        default=None,
        help="Scenario/timestep index to extract (if pkl has multiple scenarios). If omitted, processes all.",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="If set, generate per-timestep visualizations",
    )
    parser.add_argument(
        "--viz-dir",
        default=None,
        help="Directory to save visualization PNGs (defaults to <pkl-dir>/viz_pkl)",
    )
    parser.add_argument(
        "--viz-over-time",
        action="store_true",
        help="If set, also stitch per-timestep frames into a single GIF",
    )
    parser.add_argument(
        "--viz-video",
        default=None,
        help="Path for the GIF/animation (defaults to <pkl-dir>/trajectories.gif)",
    )
    parser.add_argument(
        "--plot-ego",
        action="store_true",
        default=True,
        help="Plot ego pose for each timestep (drawn as a triangle)",
    )
    parser.add_argument(
        "--no-plot-ego",
        action="store_false",
        dest="plot_ego",
        help="Disable ego pose plotting",
    )
    parser.add_argument(
        "--map-pkl",
        default=None,
        help="Optional PKL path with map polylines to overlay on visualization",
    )
    return parser.parse_args()


def yaw_from_pose(pose: Any) -> float:
    """Extract yaw (degrees) from a 6-DOF pose array."""
    if isinstance(pose, Sequence) and len(pose) >= 5:
        return float(pose[4])
    return 0.0


def ensure_color(obj_type: str, palette: Dict[str, str], color_cycle: List[str]) -> str:
    if obj_type in palette:
        return palette[obj_type]
    color = color_cycle[len(palette) % len(color_cycle)]
    palette[obj_type] = color
    return color


def plot_timestep(
    timestep: int,
    vehicles: Dict[int, Dict[str, Any]],
    out_path: str,
    palette: Dict[str, str],
    color_cycle: List[str],
    ego_pose: Any = None,
    axes_limits: Tuple[float, float, float, float] = None,
    map_lines: List[List[Tuple[float, float]]] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(f"Timestep {timestep}")
    ax.set_aspect("equal", adjustable="box")

    xs: List[float] = []
    ys: List[float] = []
    legend_handles: Dict[str, Any] = {}

    for vid, data in vehicles.items():
        if timestep not in data.get("positions", {}):
            continue
        pos = data["positions"][timestep]
        x, y = pos[0], pos[1]
        obj_type = data.get("obj_type", "") or "Unknown"

        # Use fixed extent since we don't have it in pkl
        width = 4.0
        height = 2.0
        color = ensure_color(obj_type, palette, color_cycle)

        rect = patches.Rectangle(
            (x - width / 2, y - height / 2),
            width,
            height,
            linewidth=1.0,
            edgecolor=color,
            facecolor=color,
            alpha=0.4,
        )
        ax.add_patch(rect)
        ax.text(x, y, f"{vid}", ha="center", va="center", fontsize=8, color="black")

        xs.append(x)
        ys.append(y)

        if obj_type not in legend_handles:
            legend_handles[obj_type] = patches.Patch(color=color, label=obj_type)

    if ego_pose:
        ex, ey = float(ego_pose[0]), float(ego_pose[1])
        yaw_deg = yaw_from_pose(ego_pose) if len(ego_pose) >= 5 else 0.0
        ego_length = 4.5
        color = ensure_color("ego", palette, color_cycle)
        triangle = patches.RegularPolygon(
            (ex, ey),
            numVertices=3,
            radius=ego_length / 2,
            orientation=math.radians(yaw_deg),
            color=color,
            alpha=0.6,
        )
        ax.add_patch(triangle)
        ax.text(ex, ey, "ego", ha="center", va="center", fontsize=7, color="black")
        xs.append(ex)
        ys.append(ey)
        if "ego" not in legend_handles:
            legend_handles["ego"] = patches.Patch(color=color, label="ego")

    if map_lines:
        for line in map_lines:
            if len(line) < 2:
                continue
            lx = [p[0] for p in line]
            ly = [p[1] for p in line]
            ax.plot(lx, ly, color="gray", linewidth=1.0, alpha=0.5)

    if axes_limits:
        minx, maxx, miny, maxy = axes_limits
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
    elif xs and ys:
        pad = 10.0
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)

    ax.grid(True, linestyle="--", alpha=0.4)
    if legend_handles:
        ax.legend(handles=list(legend_handles.values()), loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_map_pkl(pkl_path: str, warnings: List[str]) -> List[List[Tuple[float, float]]]:
    """Load map polylines from pkl file."""
    map_lines: List[List[Tuple[float, float]]] = []
    map_obj = None

    class _StubClass:
        """Stub for missing classes during unpickling."""
        def __init__(self, *args, **kwargs):
            self.__dict__.update(kwargs)
            for i, arg in enumerate(args):
                setattr(self, f"_arg{i}", arg)

    class _SafeUnpickler(pickle.Unpickler):
        def find_class(self, module, name):  # pragma: no cover
            try:
                return super().find_class(module, name)
            except Exception:
                return _StubClass

    def _try_load(loader):
        with open(pkl_path, "rb") as pf:
            return loader(pf)

    try:
        map_obj = _try_load(pickle.load)
    except Exception:
        try:
            map_obj = _try_load(lambda f: pickle.load(f, encoding="latin1"))
        except Exception:
            try:
                map_obj = _SafeUnpickler(open(pkl_path, "rb")).load()
            except Exception as exc:  # pragma: no cover
                warnings.append(f"Failed to load map PKL {pkl_path}: {exc}")
                return map_lines

    if map_obj is not None:
        def _extract_lines(obj, depth=0):
            if obj is None or depth > 10:
                return
            
            if hasattr(obj, "__dict__") and not isinstance(obj, (dict, list, tuple)):
                _extract_lines(obj.__dict__, depth + 1)
            
            if isinstance(obj, dict):
                if "x" in obj and "y" in obj:
                    try:
                        x, y = float(obj["x"]), float(obj["y"])
                        return [(x, y)]
                    except Exception:
                        pass
                for key in ["centerline", "boundary", "points", "nodes", "polyline", "coordinates", "line"]:
                    if key in obj:
                        result = _extract_lines(obj[key], depth + 1)
                        if result:
                            return result
                for v in obj.values():
                    _extract_lines(v, depth + 1)
                    
            elif isinstance(obj, (list, tuple)):
                if not obj:
                    return
                
                try:
                    if all(hasattr(item, "__len__") and len(item) >= 2 for item in obj[:3] if item is not None):
                        pts = [(float(p[0]), float(p[1])) for p in obj if p is not None and len(p) >= 2]
                        if len(pts) >= 2:
                            map_lines.append(pts)
                            return pts
                except Exception:
                    pass
                
                try:
                    pts = []
                    for item in obj:
                        if hasattr(item, "x") and hasattr(item, "y"):
                            pts.append((float(item.x), float(item.y)))
                        elif hasattr(item, "__dict__") and "x" in item.__dict__ and "y" in item.__dict__:
                            pts.append((float(item.__dict__["x"]), float(item.__dict__["y"])))
                        elif isinstance(item, dict) and "x" in item and "y" in item:
                            pts.append((float(item["x"]), float(item["y"])))
                    if len(pts) >= 2:
                        map_lines.append(pts)
                        return pts
                except Exception:
                    pass
                
                for v in obj:
                    _extract_lines(v, depth + 1)
                    
            else:
                try:
                    if hasattr(obj, "shape") and len(getattr(obj, "shape", [])) == 2:
                        arr = np.asarray(obj)
                        if arr.shape[1] >= 2 and arr.shape[0] >= 2:
                            pts = [(float(p[0]), float(p[1])) for p in arr]
                            map_lines.append(pts)
                            return pts
                except Exception:
                    pass
                
                if hasattr(obj, "x") and hasattr(obj, "y"):
                    try:
                        return [(float(obj.x), float(obj.y))]
                    except Exception:
                        pass

        _extract_lines(map_obj)
        if not map_lines:
            warnings.append(f"Map PKL {pkl_path} loaded but no polylines recognized; skipping map overlay")
    
    return map_lines


def load_obj_types_from_yaml(yaml_dir: str, warnings: List[str]) -> Dict[int, str]:
    """
    Load obj_type for each vehicle ID from YAML files in yaml_dir.
    Returns dict: {vehicle_id: obj_type}
    """
    obj_types: Dict[int, str] = {}
    
    if not os.path.isdir(yaml_dir):
        warnings.append(f"YAML directory not found: {yaml_dir}")
        return obj_types
    
    print(f"Loading obj_type from YAML files in {yaml_dir}...")
    
    # Look for YAML files in subfolders like -1, -2, 1, 2
    for folder in os.listdir(yaml_dir):
        folder_path = os.path.join(yaml_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        
        # Look for YAML files like 000000.yaml, 000001.yaml, etc.
        yaml_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.yaml')])
        
        for yaml_file in yaml_files:
            yaml_path = os.path.join(folder_path, yaml_file)
            try:
                with open(yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                
                if 'vehicles' in data:
                    for vid_str, vdata in data['vehicles'].items():
                        vid = int(vid_str)
                        if vid not in obj_types and 'obj_type' in vdata:
                            obj_types[vid] = vdata['obj_type']
            except Exception as e:
                warnings.append(f"Failed to load {yaml_path}: {e}")
    
    print(f"Loaded obj_type for {len(obj_types)} vehicles from YAML")
    return obj_types


def main() -> None:
    args = parse_args()
    pkl_path = os.path.abspath(args.pkl_file)
    pkl_dir = os.path.dirname(pkl_path)
    pkl_base = os.path.splitext(os.path.basename(pkl_path))[0]
    
    output_csv = args.output or os.path.join(pkl_dir, f"{pkl_base}_vehicles.csv")
    viz_dir = args.viz_dir or os.path.join(pkl_dir, "viz_pkl")
    viz_video = args.viz_video or os.path.join(pkl_dir, f"{pkl_base}_trajectories.gif")
    
    warnings: List[str] = []
    map_lines: List[List[Tuple[float, float]]] = []
    
    if args.map_pkl:
        map_lines = load_map_pkl(args.map_pkl, warnings)
    
    # Load obj_types from YAML if directory provided
    obj_type_map: Dict[int, str] = {}
    if args.yaml_dir:
        obj_type_map = load_obj_types_from_yaml(args.yaml_dir, warnings)
    
    # Load trajectory database
    print(f"Loading {pkl_path}...")
    with open(pkl_path, "rb") as f:
        traj_db = pickle.load(f)
    
    # Determine which scenarios to process
    if isinstance(traj_db, dict):
        scenario_indices = [args.scenario] if args.scenario is not None else sorted(traj_db.keys())
    else:
        raise SystemExit(f"Unexpected pkl structure: {type(traj_db)}")
    
    print(f"Found scenarios: {sorted(traj_db.keys())}")
    print(f"Processing scenarios: {scenario_indices}")
    
    # Extract vehicle data: vehicles[vid] = {"obj_type": ..., "positions": {timestep: [x, y, z]}}
    vehicles: Dict[int, Dict[str, Any]] = {}
    all_timesteps = set()
    
    for scenario_idx in scenario_indices:
        if scenario_idx not in traj_db:
            warnings.append(f"Scenario {scenario_idx} not found in pkl")
            continue
        
        scenario_data = traj_db[scenario_idx]
        for vid, vdata in scenario_data.items():
            trajectory = vdata.get("trajectory")
            trajectory_mask = vdata.get("trajectory_mask")
            
            if trajectory is None:
                continue
            
            if not isinstance(trajectory, np.ndarray):
                trajectory = np.array(trajectory)
            
            if vid not in vehicles:
                vehicles[vid] = {
                    "vehicle_id": vid,
                    "obj_type": obj_type_map.get(vid, "Unknown"),
                    "positions": {}
                }
            
            # trajectory shape: (T, 4) where columns are [timestep, x, y, z]
            for i, row in enumerate(trajectory):
                if trajectory_mask is not None and i < len(trajectory_mask) and trajectory_mask[i] == 0:
                    continue
                
                ts = int(row[0])
                x, y, z = float(row[1]), float(row[2]), float(row[3])
                vehicles[vid]["positions"][ts] = [x, y, z]
                all_timesteps.add(ts)
    
    timesteps = sorted(all_timesteps)
    print(f"Extracted {len(vehicles)} vehicles across {len(timesteps)} timesteps")
    
    # Write CSV
    header = ["vehicle_id", "obj_type"] + [f"t_{ts}" for ts in timesteps]
    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for vid in sorted(vehicles.keys()):
            row = {"vehicle_id": vid, "obj_type": vehicles[vid].get("obj_type", "")}
            for ts in timesteps:
                if ts in vehicles[vid]["positions"]:
                    pos = vehicles[vid]["positions"][ts]
                    row[f"t_{ts}"] = json.dumps({"location": pos}, separators=(",", ":"))
                else:
                    row[f"t_{ts}"] = ""
            writer.writerow(row)
    
    print(f"CSV written: {output_csv}")
    
    # Compute axes limits (vehicles only, not map)
    all_points: List[Tuple[float, float]] = []
    for vdata in vehicles.values():
        for pos in vdata["positions"].values():
            all_points.append((pos[0], pos[1]))
    
    axes_limits: Tuple[float, float, float, float] = None
    if all_points:
        xs, ys = zip(*all_points)
        pad = 10.0
        axes_limits = (min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad)
    
    # Generate visualizations
    generated_frames: List[str] = []
    
    if args.viz or args.viz_over_time:
        os.makedirs(viz_dir, exist_ok=True)
        palette: Dict[str, str] = {}
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3", "C4"])
        print(f"Generating {len(timesteps)} frames...")
        for idx, ts in enumerate(timesteps):
            if idx % 10 == 0:
                print(f"  Frame {idx+1}/{len(timesteps)}...")
            out_path = os.path.join(viz_dir, f"t_{ts:06d}.png")
            plot_timestep(
                ts,
                vehicles,
                out_path,
                palette,
                color_cycle,
                ego_pose=None,  # No ego in pkl currently
                axes_limits=axes_limits,
                map_lines=map_lines,
            )
            generated_frames.append(out_path)
        if args.viz:
            print(f"Frames written: {viz_dir}")
    
    if args.viz_over_time:
        if imageio is None:
            print("imageio not available; skipping GIF creation. Install imageio to enable this feature.")
        elif not generated_frames:
            print("No frames available to stitch; skipping GIF creation.")
        else:
            frames = [imageio.imread(p) for p in generated_frames]
            imageio.mimsave(viz_video, frames, duration=0.5)
            print(f"GIF written: {viz_video}")
    
    if warnings:
        print("Completed with warnings:")
        for w in warnings:
            print(f"  - {w}")


if __name__ == "__main__":  # pragma: no cover
    main()
