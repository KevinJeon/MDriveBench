#!/usr/bin/env python3
"""
CARLA "is anything loading?" mega-debug script (FIXED for CARLA 0.9.10 + Python shutdown)

Key fixes vs your version:
- NO numpy / heavy work inside sensor callbacks (avoids fatal "thread state must be current")
- Callback copies raw bytes into a queue; main thread does stats + saving (robust)
- Explicit sensor stop + destroy + drain queue before exit
- Restore world settings on the SAME world object (no client.get_world() in finally)
- Extra "heartbeat" logging (tells you exactly where things stall)
"""

import os
import time
import json
import math
import traceback
import queue
import argparse
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import carla

# =========================
# CONFIG
# =========================
HOST = "127.0.0.1"
PORT = 2000
TIMEOUT_S = 120.0

MAP_PATH = "/Game/UCLA/Maps/UCLA/UCLA"

OUT_ROOT = "out"
IMAGE_W = 1280
IMAGE_H = 720
FOV = 90

SYNC_MODE = True
FIXED_DT = 0.05  # 20Hz
WARMUP_TICKS = 20
OVERHEAD_ONLY = False
OVERHEAD_ZOOM_FRACTION = 0.8         # 0.8 = 20% closer than full road-bounds fit

CAPTURE_TICKS_PER_VIEW = 20
CAPTURE_EVERY_N_TICKS = 5  # save images every N ticks per view

# Vehicle driving
SPAWN_VEHICLE = True
VEHICLE_FILTER = "vehicle.tesla.model3"
ENABLE_AUTOPILOT = False
DRIVE_TICKS = 200
SPAWN_NPCS = False
NPC_COUNT = 10

# Camera views to try (relative to a base location)
TRY_SPAWNPOINT_VIEW = True
TRY_HIGH_ORTHO_VIEW = True
TRY_ORBIT_VIEWS = True
ORBIT_RADIUS_M = 80.0
ORBIT_HEIGHT_M = 35.0
ORBIT_COUNT = 6

# Weather presets to try
WEATHERS = [
    ("ClearNoon", carla.WeatherParameters.ClearNoon),
]

# Optional camera tweaks (applied only if the attribute exists in your CARLA build)
CAMERA_ATTR_PRESETS = [
  ("manual_strong", {
      "enable_postprocess_effects":"true",
      "gamma":"2.4",
      "exposure_mode":"manual",
      "exposure_compensation":"5.0",
      "shutter_speed":"250.0",
      "iso":"1600.0",
  }),
]

# Sensors to attempt (keep RGB only by default)
SENSOR_SPECS = [
    ("rgb", "sensor.camera.rgb"),
    # ("depth", "sensor.camera.depth"),
    # ("semseg", "sensor.camera.semantic_segmentation"),
    # ("instseg", "sensor.camera.instance_segmentation"),
]


# =========================
# UTILS
# =========================

def safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def print_kv(title: str, d: Dict[str, Any], indent: int = 2) -> None:
    print(title)
    pad = " " * indent
    for k, v in d.items():
        print(f"{pad}{k}: {v}")

def transform_to_dict(t: carla.Transform) -> Dict[str, Any]:
    return {
        "location": {"x": float(t.location.x), "y": float(t.location.y), "z": float(t.location.z)},
        "rotation": {"pitch": float(t.rotation.pitch), "yaw": float(t.rotation.yaw), "roll": float(t.rotation.roll)},
    }

def apply_camera_attrs(bp: carla.ActorBlueprint, attrs: Dict[str, str]) -> Dict[str, str]:
    """Apply only attributes that exist; return which ones were set."""
    applied = {}
    for k, v in attrs.items():
        try:
            if bp.has_attribute(k):
                bp.set_attribute(k, v)
                applied[k] = v
        except Exception:
            pass
    return applied

def image_stats_bgra_bytes(raw: bytes, w: int, h: int) -> Dict[str, Any]:
    """
    Compute stats on MAIN thread only.
    raw is BGRA uint8.
    """
    arr = np.frombuffer(raw, dtype=np.uint8)
    if arr.size != w * h * 4:
        return {"error": f"unexpected raw_data size {arr.size} != {w*h*4}"}
    img = arr.reshape((h, w, 4))
    rgb = img[:, :, :3]  # BGR actually, but stats are identical
    mn = int(rgb.min())
    mx = int(rgb.max())
    mean = float(rgb.mean())
    near_black = np.all(rgb <= 2, axis=2)
    pct_near_black = float(near_black.mean() * 100.0)
    return {"min": mn, "max": mx, "mean": round(mean, 3), "pct_near_black": round(pct_near_black, 3)}

def seg_stats_bytes(raw: bytes, w: int, h: int) -> Dict[str, Any]:
    arr = np.frombuffer(raw, dtype=np.uint8)
    if arr.size != w * h * 4:
        return {"error": f"unexpected raw_data size {arr.size} != {w*h*4}"}
    img = arr.reshape((h, w, 4))
    rgb = img[:, :, :3]
    sub = rgb[::4, ::4, :].reshape((-1, 3))
    uniq = np.unique(sub, axis=0)
    return {
        "min": int(rgb.min()),
        "max": int(rgb.max()),
        "mean": round(float(rgb.mean()), 3),
        "unique_colors_subsample": int(uniq.shape[0]),
    }

def look_at(from_loc: carla.Location, to_loc: carla.Location) -> carla.Rotation:
    dx = to_loc.x - from_loc.x
    dy = to_loc.y - from_loc.y
    dz = to_loc.z - from_loc.z
    yaw = math.degrees(math.atan2(dy, dx))
    dist_xy = math.sqrt(dx*dx + dy*dy)
    pitch = math.degrees(math.atan2(dz, dist_xy))
    return carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0)

def stop_and_destroy(actor: Optional[carla.Actor]) -> None:
    if actor is None:
        return
    try:
        actor.stop()
    except Exception:
        pass
    try:
        actor.destroy()
    except Exception:
        pass

def heartbeat(client: carla.Client, tag: str) -> None:
    try:
        w = client.get_world()
        snap = w.get_snapshot()
        print(f"[HB] {tag}: ok frame={snap.frame}")
    except Exception as e:
        print(f"[HB] {tag}: FAILED: {e}")
        raise


# =========================
# QUEUE-BASED SENSOR CAPTURE (SAFE)
# =========================

@dataclass
class SensorQueueItem:
    frame: int
    timestamp: float
    width: int
    height: int
    raw: bytes  # copied bytes

class SensorCollector:
    """
    Callback thread: copies bytes + metadata into a bounded queue.
    Main thread: drains queue, computes stats, saves images.
    """
    def __init__(self, spec_name: str, out_dir: str, save_every_n_ticks: int, filename_prefix: Optional[str] = None):
        self.spec_name = spec_name
        self.out_dir = out_dir
        self.save_every_n_ticks = max(1, int(save_every_n_ticks))
        safe_makedirs(out_dir)
        self.filename_prefix = filename_prefix or spec_name

        self.q: "queue.Queue[SensorQueueItem]" = queue.Queue(maxsize=128)
        self.frames_seen = 0
        self.frames_enqueued = 0
        self.frames_dropped = 0
        self.frames_saved = 0

        self.stats_log_path = os.path.join(out_dir, f"{spec_name}_stats.jsonl")
        self._alive = True

    def stop(self):
        self._alive = False

    def cb(self, image: carla.Image):
        # Sensor thread: DO NOT call numpy here.
        if not self._alive:
            return
        self.frames_seen += 1
        try:
            item = SensorQueueItem(
                frame=int(image.frame),
                timestamp=float(image.timestamp),
                width=int(image.width),
                height=int(image.height),
                raw=bytes(image.raw_data),  # copy bytes NOW
            )
            try:
                self.q.put_nowait(item)
                self.frames_enqueued += 1
            except queue.Full:
                self.frames_dropped += 1
        except Exception:
            # Never let callback throw.
            self.frames_dropped += 1

    def process_one(self, timeout_s: float = 0.0) -> bool:
        try:
            item = self.q.get(timeout=timeout_s)
        except queue.Empty:
            return False

        # MAIN thread: stats + saving
        if self.spec_name in ("semseg", "instseg"):
            stats = seg_stats_bytes(item.raw, item.width, item.height)
        else:
            stats = image_stats_bgra_bytes(item.raw, item.width, item.height)

        rec = {
            "frame": item.frame,
            "timestamp": item.timestamp,
            "width": item.width,
            "height": item.height,
            "stats": stats,
        }
        with open(self.stats_log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

        # Save based on FRAME number (stable) instead of "seen count"
        if (item.frame % self.save_every_n_ticks) == 0:
            try:
                # Use PIL saving from raw bytes (more deterministic than save_to_disk in callback)
                from PIL import Image
                arr = np.frombuffer(item.raw, dtype=np.uint8).reshape((item.height, item.width, 4))
                # CARLA raw is BGRA; convert to RGB for viewing
                rgb = arr[:, :, :3][:, :, ::-1]  # BGR -> RGB
                out_path = os.path.join(self.out_dir, f"{self.filename_prefix}_{item.frame:06d}.png")
                Image.fromarray(rgb).save(out_path)
                self.frames_saved += 1
            except Exception as e:
                with open(self.stats_log_path, "a") as f:
                    f.write(json.dumps({"frame": item.frame, "save_error": str(e)}) + "\n")

        return True

    def drain(self, max_items: int = 100000):
        n = 0
        while n < max_items and self.process_one(timeout_s=0.0):
            n += 1
        return n


# =========================
# WORLD / MAP INSPECTION
# =========================

def dump_world_info(world: carla.World, out_dir: str) -> None:
    safe_makedirs(out_dir)

    settings = world.get_settings()
    weather = world.get_weather()

    def sget(name, default=None):
        return getattr(settings, name, default)

    info = {
        "settings": {
            "synchronous_mode": bool(sget("synchronous_mode", False)),
            "fixed_delta_seconds": sget("fixed_delta_seconds", None),
            "no_rendering_mode": bool(sget("no_rendering_mode", False)),
            "max_substep_delta_time": sget("max_substep_delta_time", None),
            "max_substeps": sget("max_substeps", None),
            "deterministic_ragdolls": sget("deterministic_ragdolls", None),
            "substepping": sget("substepping", None),
        },
        "settings_available_fields": sorted([a for a in dir(settings) if not a.startswith("_")]),
        "weather": {k: getattr(weather, k, None) for k in [
            "cloudiness", "precipitation", "precipitation_deposits", "wind_intensity",
            "sun_azimuth_angle", "sun_altitude_angle", "fog_density", "fog_distance",
            "fog_falloff", "wetness", "scattering_intensity", "mie_scattering_scale",
            "rayleigh_scattering_scale", "dust_storm"
        ]},
        "snapshot": {
            "frame": int(world.get_snapshot().frame),
            "timestamp_elapsed_seconds": float(world.get_snapshot().timestamp.elapsed_seconds),
            "timestamp_delta_seconds": float(world.get_snapshot().timestamp.delta_seconds),
            "platform_timestamp": float(world.get_snapshot().timestamp.platform_timestamp),
        }
    }

    with open(os.path.join(out_dir, "world_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print_kv("[WORLD] settings", info["settings"])
    print(f"[WORLD] settings fields available: {len(info['settings_available_fields'])}")
    print_kv("[WORLD] weather", {
        "sun_altitude_angle": info["weather"].get("sun_altitude_angle"),
        "sun_azimuth_angle": info["weather"].get("sun_azimuth_angle"),
        "cloudiness": info["weather"].get("cloudiness"),
        "precipitation": info["weather"].get("precipitation"),
        "fog_density": info["weather"].get("fog_density"),
        "wetness": info["weather"].get("wetness"),
    })

def dump_map_info(world: carla.World, out_dir: str) -> Dict[str, Any]:
    safe_makedirs(out_dir)
    m = world.get_map()
    sp = m.get_spawn_points()
    topo = m.get_topology()

    sample = {}
    if sp:
        w0 = m.get_waypoint(sp[0].location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if w0:
            sample["waypoint0"] = {
                "road_id": int(w0.road_id),
                "lane_id": int(w0.lane_id),
                "s": float(w0.s),
                "lane_width": float(w0.lane_width),
                "transform": transform_to_dict(w0.transform),
            }

    info = {
        "map_name": m.name,
        "spawn_points_count": len(sp),
        "topology_edges_count": len(topo),
        "sample": sample,
    }
    with open(os.path.join(out_dir, "map_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print_kv("[MAP] info", info)
    return info

def estimate_map_bounds(world: carla.World, m: carla.Map) -> Tuple[carla.Location, carla.Location]:
    """
    Returns (min_loc, max_loc) using road network extremes first (topology waypoints),
    then spawn points, then level bounding boxes as a fallback. This keeps the overhead
    framing tied to the actual drivable network.
    """
    xs, ys, zs = [], [], []

    # Road network nodes (priority)
    try:
        topo = m.get_topology()
        for a, b in topo:
            for wp in (a, b):
                loc = wp.transform.location
                xs.append(loc.x); ys.append(loc.y); zs.append(loc.z)
    except Exception:
        topo = []

    # Spawn points (if topology missing)
    if not xs:
        for t in m.get_spawn_points():
            xs.append(t.location.x); ys.append(t.location.y); zs.append(t.location.z)

    # Level bounding boxes (last resort)
    if not xs:
        try:
            bbs = world.get_level_bbs(carla.CityObjectLabel.Any)
            for bb in bbs:
                loc = bb.location
                ext = bb.extent
                xs.extend([loc.x - ext.x, loc.x + ext.x])
                ys.extend([loc.y - ext.y, loc.y + ext.y])
                zs.extend([loc.z - ext.z, loc.z + ext.z])
        except Exception:
            pass

    # Topology endpoints (roads)
    # Already included above; no-op retained for clarity

    if not xs:
        # absolute fallback
        return carla.Location(-50, -50, 0), carla.Location(50, 50, 0)

    min_loc = carla.Location(min(xs), min(ys), min(zs))
    max_loc = carla.Location(max(xs), max(ys), max(zs))
    return min_loc, max_loc

def overhead_full_map_transform(world: carla.World, m: carla.Map, fov_deg: float, margin: float = 1.25, zoom_fraction: float = 1.0) -> carla.Transform:
    """
    Places a top-down camera over the map center at an altitude that (roughly) fits the full map in view.
    margin > 1 makes it a bit higher so nothing clips.
    """
    min_loc, max_loc = estimate_map_bounds(world, m)
    cx = 0.5 * (min_loc.x + max_loc.x)
    cy = 0.5 * (min_loc.y + max_loc.y)

    # Need to fit the larger span into the camera frustum
    span_x = (max_loc.x - min_loc.x)
    span_y = (max_loc.y - min_loc.y)
    span = max(span_x, span_y) * margin * max(zoom_fraction, 1e-3)

    # Perspective geometry: at height H, half-width visible = H * tan(fov/2)
    fov = math.radians(fov_deg)
    half = 0.5 * span
    H = half / math.tan(fov / 2.0)

    # Ensure we're not *too* low if spans are tiny
    H = max(H, 150.0)

    loc = carla.Location(x=cx, y=cy, z=max_loc.z + H + 20.0)
    rot = carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0)
    return carla.Transform(loc, rot)


def dump_actor_inventory(world: carla.World, out_dir: str) -> None:
    safe_makedirs(out_dir)
    actors = world.get_actors()
    counts = {}
    examples = {}
    for a in actors:
        tid = a.type_id
        key = tid.split(".")[0] if "." in tid else tid
        counts[key] = counts.get(key, 0) + 1
        if key not in examples:
            examples[key] = tid

    inv = {
        "total_actors": len(actors),
        "counts_by_prefix": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "example_type_ids": examples,
    }
    with open(os.path.join(out_dir, "actor_inventory.json"), "w") as f:
        json.dump(inv, f, indent=2)

    print(f"[ACTORS] total: {inv['total_actors']}")
    for k, v in list(inv["counts_by_prefix"].items())[:10]:
        print(f"  {k}: {v} (e.g. {inv['example_type_ids'].get(k)})")

def dump_level_bbs(world: carla.World, out_dir: str) -> None:
    safe_makedirs(out_dir)
    labels = [
        ("Any", carla.CityObjectLabel.Any),
        ("Buildings", carla.CityObjectLabel.Buildings),
        ("Roads", carla.CityObjectLabel.Roads),
        ("Fences", carla.CityObjectLabel.Fences),
        ("Vegetation", carla.CityObjectLabel.Vegetation),
        ("TrafficSigns", carla.CityObjectLabel.TrafficSigns),
        ("TrafficLight", carla.CityObjectLabel.TrafficLight),
    ]
    rec = {}
    for name, label in labels:
        try:
            bbs = world.get_level_bbs(label)
            rec[name] = len(bbs)
        except Exception as e:
            rec[name] = f"error: {e}"

    with open(os.path.join(out_dir, "level_bbs_counts.json"), "w") as f:
        json.dump(rec, f, indent=2)
    print_kv("[LEVEL_BBS] counts", rec)


# =========================
# VEHICLE / DRIVING
# =========================

def spawn_vehicle(world: carla.World, m: carla.Map, out_dir: str) -> Optional[carla.Vehicle]:
    safe_makedirs(out_dir)
    bp_lib = world.get_blueprint_library()
    bps = bp_lib.filter(VEHICLE_FILTER)
    if not bps:
        print(f"[WARN] no vehicle blueprints match {VEHICLE_FILTER}, trying vehicle.*")
        bps = bp_lib.filter("vehicle.*")
        if not bps:
            print("[ERROR] no vehicle blueprints available.")
            return None
    bp = bps[0]

    sp = m.get_spawn_points()
    if not sp:
        print("[WARN] no spawn points to spawn vehicle.")
        return None

    v = None
    for i in range(min(10, len(sp))):
        try:
            v = world.spawn_actor(bp, sp[i])
            print(f"[VEH] spawned {v.type_id} at spawn point {i}")
            with open(os.path.join(out_dir, "vehicle_spawn.json"), "w") as f:
                json.dump({"type_id": v.type_id, "spawn_point_index": i, "transform": transform_to_dict(sp[i])}, f, indent=2)
            break
        except Exception as e:
            print(f"[WARN] failed to spawn vehicle at sp[{i}]: {e}")
    return v


# =========================
# MAIN EXPERIMENT LOOP
# =========================

def tick_n(world: carla.World, n: int, collectors: Optional[List[SensorCollector]] = None) -> None:
    for _ in range(n):
        world.tick()
        if collectors:
            # process a few items each tick to keep queues small
            for c in collectors:
                c.process_one(timeout_s=0.0)

def make_view_transforms(world: carla.World, base_t: carla.Transform, m: carla.Map, overhead_only: bool = False) -> List[Tuple[str, carla.Transform]]:
    views = []

    if overhead_only:
        z = OVERHEAD_ZOOM_FRACTION
        return [("overhead_full_map_zoomed", overhead_full_map_transform(world, m, FOV, zoom_fraction=z))]

    if TRY_SPAWNPOINT_VIEW:
        views.append(("spawnpoint", base_t))

    if TRY_HIGH_ORTHO_VIEW:
        loc = carla.Location(base_t.location.x, base_t.location.y, base_t.location.z + 120.0)
        rot = carla.Rotation(pitch=-90.0, yaw=base_t.rotation.yaw, roll=0.0)
        views.append(("overhead_down", carla.Transform(loc, rot)))

        loc2 = carla.Location(base_t.location.x, base_t.location.y, base_t.location.z + 80.0)
        rot2 = carla.Rotation(pitch=-60.0, yaw=base_t.rotation.yaw, roll=0.0)
        views.append(("overhead_tilt", carla.Transform(loc2, rot2)))

    if TRY_ORBIT_VIEWS:
        center = base_t.location
        for k in range(ORBIT_COUNT):
            ang = 2.0 * math.pi * (k / ORBIT_COUNT)
            x = center.x + ORBIT_RADIUS_M * math.cos(ang)
            y = center.y + ORBIT_RADIUS_M * math.sin(ang)
            z = center.z + ORBIT_HEIGHT_M
            from_loc = carla.Location(x=x, y=y, z=z)
            rot = look_at(from_loc, carla.Location(center.x, center.y, center.z + 5.0))
            views.append((f"orbit_{k}", carla.Transform(from_loc, rot)))

    views.append(("overhead_full_map", overhead_full_map_transform(world, m, FOV)))

    views.append(("origin_high", carla.Transform(carla.Location(0, 0, 200), carla.Rotation(pitch=-90, yaw=0, roll=0))))
    return views

def spawn_sensor(world: carla.World,
                 parent: Optional[carla.Actor],
                 spec_name: str,
                 blueprint_id: str,
                 transform: carla.Transform,
                 out_dir: str,
                 cam_attrs: Dict[str, str],
                 filename_prefix: Optional[str] = None) -> Tuple[Optional[carla.Actor], Optional[SensorCollector]]:
    bp_lib = world.get_blueprint_library()
    try:
        bp = bp_lib.find(blueprint_id)
    except Exception:
        print(f"[WARN] blueprint not found: {blueprint_id}")
        return None, None

    if bp.has_attribute("image_size_x"):
        bp.set_attribute("image_size_x", str(IMAGE_W))
    if bp.has_attribute("image_size_y"):
        bp.set_attribute("image_size_y", str(IMAGE_H))
    if bp.has_attribute("fov"):
        bp.set_attribute("fov", str(FOV))

    applied = apply_camera_attrs(bp, cam_attrs)

    try:
        if parent is None:
            sensor = world.spawn_actor(bp, transform)
        else:
            sensor = world.spawn_actor(bp, transform, attach_to=parent)
    except Exception as e:
        print(f"[WARN] failed to spawn {spec_name} ({blueprint_id}): {e}")
        return None, None

    safe_makedirs(out_dir)
    with open(os.path.join(out_dir, f"{spec_name}_meta.json"), "w") as f:
        json.dump({
            "spec_name": spec_name,
            "blueprint_id": blueprint_id,
            "transform": transform_to_dict(transform),
            "applied_camera_attrs": applied,
        }, f, indent=2)

    collector = SensorCollector(
        spec_name=spec_name,
        out_dir=out_dir,
        save_every_n_ticks=CAPTURE_EVERY_N_TICKS,
        filename_prefix=filename_prefix,
    )
    sensor.listen(collector.cb)
    return sensor, collector

def main(overhead_only: bool = OVERHEAD_ONLY):
    client = carla.Client(HOST, PORT)
    client.set_timeout(TIMEOUT_S)

    stamp = now_stamp()
    map_slug = MAP_PATH.split("/")[-1]
    out_dir = os.path.join(OUT_ROOT, f"carla_debug_{map_slug}_{stamp}")
    safe_makedirs(out_dir)
    print(f"[OUT] writing to: {out_dir}")

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({
            "HOST": HOST,
            "PORT": PORT,
            "MAP_PATH": MAP_PATH,
            "IMAGE_W": IMAGE_W,
            "IMAGE_H": IMAGE_H,
            "FOV": FOV,
            "SYNC_MODE": SYNC_MODE,
            "FIXED_DT": FIXED_DT,
            "WARMUP_TICKS": WARMUP_TICKS,
            "CAPTURE_TICKS_PER_VIEW": CAPTURE_TICKS_PER_VIEW,
            "CAPTURE_EVERY_N_TICKS": CAPTURE_EVERY_N_TICKS,
            "OVERHEAD_ONLY": overhead_only,
            "OVERHEAD_ZOOM_FRACTION": OVERHEAD_ZOOM_FRACTION,
            "WEATHERS": [w[0] for w in WEATHERS],
            "CAMERA_ATTR_PRESETS": [p[0] for p in CAMERA_ATTR_PRESETS],
            "SENSORS": [s[0] for s in SENSOR_SPECS],
            "SPAWN_VEHICLE": SPAWN_VEHICLE,
            "ENABLE_AUTOPILOT": ENABLE_AUTOPILOT,
            "DRIVE_TICKS": DRIVE_TICKS,
        }, f, indent=2)

    print(f"[LOAD] loading world: {MAP_PATH}")
    world = client.load_world(MAP_PATH)
    m = world.get_map()
    print(f"[LOAD] map now: {m.name}")

    original_settings = world.get_settings()
    spawned_actors: List[carla.Actor] = []

    try:
        # Apply sync
        settings = world.get_settings()
        settings.synchronous_mode = bool(SYNC_MODE)
        settings.fixed_delta_seconds = FIXED_DT if SYNC_MODE else None
        world.apply_settings(settings)
        heartbeat(client, "after apply_settings")

        # Inspections
        dump_world_info(world, os.path.join(out_dir, "00_world"))
        dump_map_info(world, os.path.join(out_dir, "00_world"))
        dump_actor_inventory(world, os.path.join(out_dir, "00_world"))
        dump_level_bbs(world, os.path.join(out_dir, "00_world"))

        # Base transform
        spawn_points = m.get_spawn_points()
        if spawn_points:
            base_t = spawn_points[0]
            print(f"[BASE] using spawn point 0: {base_t}")
        else:
            base_t = carla.Transform(carla.Location(0, 0, 200), carla.Rotation(pitch=-90))
            print("[BASE] no spawn points; using fallback base transform:", base_t)

        try:
            world.get_spectator().set_transform(base_t)
        except Exception:
            pass

        print(f"[WARMUP] ticking {WARMUP_TICKS} frames")
        tick_n(world, WARMUP_TICKS)

        # Spawn vehicle
        vehicle = None
        if SPAWN_VEHICLE:
            vehicle = spawn_vehicle(world, m, os.path.join(out_dir, "01_vehicle"))
            if vehicle is not None:
                spawned_actors.append(vehicle)
                tick_n(world, 5)
                if not ENABLE_AUTOPILOT:
                    # give it a tiny forward nudge so it’s not totally static
                    try:
                        vehicle.apply_control(carla.VehicleControl(throttle=0.35, steer=0.0))
                    except Exception:
                        pass

        # Capture matrix
        for weather_name, weather_param in WEATHERS:
            print(f"\n[WEATHER] set: {weather_name}")
            world.set_weather(weather_param)
            tick_n(world, 10)

            for preset_name, cam_attrs in CAMERA_ATTR_PRESETS:
                print(f"\n[CAM_PRESET] {preset_name} attrs={cam_attrs}")
                preset_dir = os.path.join(out_dir, "02_captures", weather_name, preset_name)
                safe_makedirs(preset_dir)

                views = make_view_transforms(world, base_t, m, overhead_only=overhead_only)
                for view_name, view_t in views:
                    view_dir = os.path.join(preset_dir, view_name)
                    safe_makedirs(view_dir)

                    try:
                        world.get_spectator().set_transform(view_t)
                    except Exception:
                        pass

                    print(f"[VIEW] {view_name} at {view_t}")

                    # World sensors (unattached)
                    world_sensors: List[carla.Actor] = []
                    world_collectors: List[SensorCollector] = []
                    for spec_name, blueprint_id in SENSOR_SPECS:
                        s_actor, collector = spawn_sensor(
                            world=world,
                            parent=None,
                            spec_name=spec_name,
                            blueprint_id=blueprint_id,
                            transform=view_t,
                            out_dir=os.path.join(view_dir, "world_sensors"),
                            cam_attrs=cam_attrs,
                            filename_prefix=f"{spec_name}_{view_name}",
                        )
                        if s_actor is not None:
                            world_sensors.append(s_actor)
                            spawned_actors.append(s_actor)
                        if collector is not None:
                            world_collectors.append(collector)

                    # Ego sensors (attached)
                    ego_sensors: List[carla.Actor] = []
                    ego_collectors: List[SensorCollector] = []
                    if vehicle is not None:
                        ego_cam_t = carla.Transform(carla.Location(x=1.5, z=1.6), carla.Rotation(pitch=0, yaw=0, roll=0))
                        for spec_name, blueprint_id in SENSOR_SPECS:
                            s_actor, collector = spawn_sensor(
                                world=world,
                                parent=vehicle,
                                spec_name=spec_name,
                                blueprint_id=blueprint_id,
                                transform=ego_cam_t,
                                out_dir=os.path.join(view_dir, "ego_sensors"),
                                cam_attrs=cam_attrs,
                                filename_prefix=f"{spec_name}_{view_name}",
                            )
                            if s_actor is not None:
                                ego_sensors.append(s_actor)
                                spawned_actors.append(s_actor)
                            if collector is not None:
                                ego_collectors.append(collector)

                    collectors = world_collectors + ego_collectors

                    print(f"[CAPTURE] ticking {CAPTURE_TICKS_PER_VIEW} frames (save every {CAPTURE_EVERY_N_TICKS} frames)")
                    tick_n(world, CAPTURE_TICKS_PER_VIEW, collectors=collectors)

                    # Stop collectors first (prevents callback from queueing while destroying)
                    for c in collectors:
                        c.stop()

                    # Destroy sensors
                    for a in world_sensors + ego_sensors:
                        stop_and_destroy(a)

                    # Drain remaining queued items (main thread) so stats/images flush
                    for c in collectors:
                        c.drain()

        # Drive phase with ego RGB (queue based)
        if vehicle is not None and DRIVE_TICKS > 0:
            drive_dir = os.path.join(out_dir, "03_drive")
            safe_makedirs(drive_dir)

            ego_cam_t = carla.Transform(carla.Location(x=1.5, z=1.6), carla.Rotation(pitch=0, yaw=0, roll=0))
            ego_cam, collector = spawn_sensor(
                world=world,
                parent=vehicle,
                spec_name="ego_rgb",
                blueprint_id="sensor.camera.rgb",
                transform=ego_cam_t,
                out_dir=drive_dir,
                cam_attrs={"enable_postprocess_effects": "true", "gamma": "2.2", "exposure_compensation": "8.0"},
                filename_prefix="ego_rgb_drive",
            )
            if ego_cam is not None:
                spawned_actors.append(ego_cam)

            with open(os.path.join(drive_dir, "drive_meta.json"), "w") as f:
                json.dump({
                    "vehicle_type_id": vehicle.type_id,
                    "ego_camera_transform": transform_to_dict(ego_cam_t),
                    "ticks": DRIVE_TICKS,
                    "save_every_frame_mod": CAPTURE_EVERY_N_TICKS,
                }, f, indent=2)

            print(f"\n[DRIVE] ticking {DRIVE_TICKS} frames with ego cam attached")
            # keep pushing forward a bit
            try:
                vehicle.apply_control(carla.VehicleControl(throttle=0.4, steer=0.0))
            except Exception:
                pass

            tick_n(world, DRIVE_TICKS, collectors=[collector] if collector else None)

            if collector:
                collector.stop()
            if ego_cam:
                stop_and_destroy(ego_cam)
            if collector:
                collector.drain()

            if collector:
                print(f"[DRIVE] ego frames: seen={collector.frames_seen} enq={collector.frames_enqueued} "
                      f"dropped={collector.frames_dropped} saved={collector.frames_saved}")

        print(f"\n[DONE] outputs written to: {out_dir}")
        print("Key files to check:")
        print(f"  {os.path.join(out_dir, '00_world', 'world_info.json')}")
        print(f"  {os.path.join(out_dir, '00_world', 'level_bbs_counts.json')}")
        print(f"  {os.path.join(out_dir, '02_captures')}  (images + stats)")
        print(f"  {os.path.join(out_dir, '03_drive')}     (ego drive images + stats)")

    except Exception:
        print("\n[ERROR] exception occurred:")
        traceback.print_exc()

    finally:
        # Stop collectors before Python finalizes (best effort)
        # Destroy actors we spawned (best effort)
        for a in reversed(spawned_actors):
            stop_and_destroy(a)

        # Restore original settings on the SAME world object
        try:
            world.apply_settings(original_settings)
        except Exception as e:
            print("[CLEANUP] failed to restore settings:", e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CARLA capture/debug script")
    parser.add_argument("--overhead-only", action="store_true", help="Only capture overhead_full_map view")
    args = parser.parse_args()

    main(overhead_only=args.overhead_only)
