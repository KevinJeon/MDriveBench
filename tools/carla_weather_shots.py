#!/usr/bin/env python3
"""
Capture weather screenshots and animated GIFs from CARLA.

Usage:
  python tools/carla_weather_shots.py --host 127.0.0.1 --port 2000 --output ./weather_shots

Requires a running CARLA server. Uses synchronous mode, spawns a single vehicle
with a front-facing RGB camera, cycles weather presets, and saves:
  - PNGs in <output>/pngs/<preset>_001.png, <preset>_002.png, ...
  - GIFs in <output>/gifs/<preset>.gif (created from PNGs)
"""

import argparse
import glob
import os
import sys
import time
from pathlib import Path
from queue import Queue, Empty
import inspect
import numpy as np


# =============================================================================
# CUSTOM WEATHER PRESETS
# =============================================================================

NIGHT_PRESETS = [
    ("Night1_MoonlessClear", dict(
        cloudiness=5.0,
        precipitation=0.0,
        precipitation_deposits=0.0,
        wind_intensity=5.0,
        sun_azimuth_angle=0.0,
        sun_altitude_angle=-90.0,
        fog_density=0.0,
        fog_distance=0.0,
        wetness=0.0,
        fog_falloff=0.2,
    )),
    ("Night2_Overcast", dict(
        cloudiness=95.0,
        precipitation=0.0,
        precipitation_deposits=0.0,
        wind_intensity=15.0,
        sun_azimuth_angle=90.0,
        sun_altitude_angle=-90.0,
        fog_density=8.0,
        fog_distance=30.0,
        wetness=5.0,
        fog_falloff=0.3,
    )),
    ("Night3_BlueHour", dict(
        cloudiness=40.0,
        precipitation=0.0,
        precipitation_deposits=0.0,
        wind_intensity=10.0,
        sun_azimuth_angle=180.0,
        sun_altitude_angle=-15.0,
        fog_density=2.0,
        fog_distance=80.0,
        wetness=0.0,
        fog_falloff=0.2,
    )),
]

FOG_PRESETS = [
    ("Fog1_DaytimeWhiteout", dict(
        cloudiness=80.0,
        precipitation=0.0,
        precipitation_deposits=0.0,
        wind_intensity=5.0,
        sun_azimuth_angle=0.0,
        sun_altitude_angle=35.0,
        fog_density=95.0,
        fog_distance=5.0,
        wetness=10.0,
        fog_falloff=0.15,
    )),
    ("Fog2_GroundFog", dict(
        cloudiness=60.0,
        precipitation=0.0,
        precipitation_deposits=0.0,
        wind_intensity=0.0,
        sun_azimuth_angle=90.0,
        sun_altitude_angle=20.0,
        fog_density=85.0,
        fog_distance=10.0,
        wetness=20.0,
        fog_falloff=0.8,
    )),
    ("Fog3_NightFog", dict(
        cloudiness=70.0,
        precipitation=0.0,
        precipitation_deposits=0.0,
        wind_intensity=10.0,
        sun_azimuth_angle=180.0,
        sun_altitude_angle=-90.0,
        fog_density=90.0,
        fog_distance=8.0,
        wetness=15.0,
        fog_falloff=0.25,
    )),
]

RAIN_PRESETS = [
    ("Rain1_Storm", dict(
        cloudiness=100.0,
        precipitation=100.0,
        precipitation_deposits=90.0,
        wind_intensity=60.0,
        sun_azimuth_angle=0.0,
        sun_altitude_angle=45.0,
        fog_density=10.0,
        fog_distance=15.0,
        wetness=90.0,
        fog_falloff=0.2,
    )),
    ("Rain2_TropicalDownpour", dict(
        cloudiness=100.0,
        precipitation=100.0,
        precipitation_deposits=100.0,
        wind_intensity=25.0,
        sun_azimuth_angle=90.0,
        sun_altitude_angle=25.0,
        fog_density=25.0,
        fog_distance=8.0,
        wetness=100.0,
        fog_falloff=0.25,
    )),
    ("Rain3_WindySquall", dict(
        cloudiness=95.0,
        precipitation=90.0,
        precipitation_deposits=70.0,
        wind_intensity=90.0,
        sun_azimuth_angle=180.0,
        sun_altitude_angle=35.0,
        fog_density=15.0,
        fog_distance=12.0,
        wetness=85.0,
        fog_falloff=0.2,
    )),
]

# All 9 custom presets combined
CUSTOM_PRESETS = NIGHT_PRESETS + FOG_PRESETS + RAIN_PRESETS


def _add_carla_to_path(carla_root: Path) -> None:
    """Append CARLA Python API egg to sys.path."""
    egg_pattern = str(carla_root / "PythonAPI" / "carla" / "dist" / "carla-*py3*.egg")
    eggs = glob.glob(egg_pattern)
    if not eggs:
        raise SystemExit(f"Could not find CARLA egg at {egg_pattern}")
    sys.path.append(eggs[0])


def _bgra_to_rgb(raw_data, height, width):
    """Convert CARLA's BGRA raw_data to RGB numpy array."""
    arr = np.frombuffer(raw_data, dtype=np.uint8)
    arr = arr.reshape((height, width, 4))
    # CARLA uses BGRA format - convert to RGB by reversing BGR channels
    rgb = arr[:, :, :3][:, :, ::-1].copy()
    return rgb


def _save_frames_as_gif(frames, gif_path, fps):
    """Save list of RGB numpy arrays as an animated GIF using PIL."""
    from PIL import Image
    
    if not frames:
        print(f"[WARN] No frames to save for {gif_path}")
        return False
    
    # Convert numpy arrays to PIL Images
    pil_frames = [Image.fromarray(f) for f in frames]
    
    # Save as GIF with proper duration
    duration_ms = int(1000 / fps)
    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,  # 0 = infinite loop
        optimize=False,
    )
    print(f"  Saved GIF: {gif_path} ({len(frames)} frames, {fps} fps, {duration_ms}ms/frame)")
    return True


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1", help="CARLA host (default: 127.0.0.1)")
    ap.add_argument("--port", type=int, default=2000, help="CARLA port (default: 2000)")
    ap.add_argument("--timeout", type=float, default=10.0, help="Client timeout seconds")
    ap.add_argument(
        "--carla-root",
        default="/data2/marco/CoLMDriver/carla",
        help="Path to CARLA root containing PythonAPI (default: /data2/marco/CoLMDriver/carla)",
    )
    ap.add_argument(
        "--output",
        default="weather_shots",
        help="Directory to save screenshots (created if missing)",
    )
    ap.add_argument(
        "--settle-ticks",
        type=int,
        default=60,
        help="World ticks after weather change before capturing (default: 60). Higher = more time for weather to render.",
    )
    ap.add_argument(
        "--frames-per-preset",
        type=int,
        default=40,
        help="Number of frames to capture per preset for GIF (default: 40).",
    )
    ap.add_argument(
        "--ticks-between-frames",
        type=int,
        default=2,
        help="World ticks between each captured frame (default: 2). Creates motion in the scene.",
    )
    ap.add_argument(
        "--gif-fps",
        type=int,
        default=15,
        help="GIF frame rate (default: 15 fps).",
    )
    ap.add_argument(
        "--list-presets",
        action="store_true",
        help="List available presets and exit.",
    )
    ap.add_argument(
        "--presets",
        nargs="*",
        default=None,
        help="Only capture these specific presets (space-separated). If not provided, captures all.",
    )
    ap.add_argument(
        "--custom",
        action="store_true",
        help="Use custom weather presets (9 total: 3 night, 3 fog, 3 rain) instead of CARLA built-in presets.",
    )
    return ap.parse_args()


def _discover_presets(carla):
    """Return list of (name, WeatherParameters) discovered from the CARLA module."""
    presets = []
    for name, val in inspect.getmembers(carla.WeatherParameters):
        if isinstance(val, carla.WeatherParameters):
            presets.append((name, val))
    # Deduplicate by name and sort for stability
    seen = set()
    unique = []
    for name, val in sorted(presets, key=lambda x: x[0].lower()):
        if name in seen:
            continue
        seen.add(name)
        unique.append((name, val))
    return unique


def main() -> None:
    args = parse_args()
    _add_carla_to_path(Path(args.carla_root))

    import carla  # noqa: WPS433

    print(f"[INFO] Connecting to CARLA at {args.host}:{args.port}...")
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)

    world = client.get_world()
    original_settings = world.get_settings()
    print(f"[INFO] Connected to map: {world.get_map().name}")

    # Choose between custom presets or built-in CARLA presets
    if args.custom:
        print("[INFO] Using custom weather presets (9 total: 3 night, 3 fog, 3 rain)")
        presets = [(name, carla.WeatherParameters(**params)) for name, params in CUSTOM_PRESETS]
    else:
        presets = _discover_presets(carla)
        if not presets:
            raise SystemExit("No WeatherParameters presets discovered from CARLA API.")
    
    # Filter presets if specific ones requested
    if args.presets:
        preset_names = set(args.presets)
        presets = [(name, val) for name, val in presets if name in preset_names]
        if not presets:
            raise SystemExit(f"None of the requested presets found: {args.presets}")
    
    if args.list_presets:
        print("Available presets:")
        for name, _ in presets:
            print(f"  - {name}")
        return

    # Create output directories: pngs/ and gifs/
    output_dir = Path(args.output)
    png_dir = output_dir / "pngs"
    gif_dir = output_dir / "gifs"
    png_dir.mkdir(parents=True, exist_ok=True)
    gif_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directories:")
    print(f"       PNGs -> {png_dir}")
    print(f"       GIFs -> {gif_dir}")

    vehicle = None
    camera = None
    actors = []
    try:
        # Enable synchronous mode
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS simulation
        world.apply_settings(settings)
        print("[INFO] Enabled synchronous mode")

        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter("vehicle.*model3*")
        vehicle_bp = vehicle_bp[0] if vehicle_bp else blueprint_library.filter("vehicle.*")[0]

        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points available")
        vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
        actors.append(vehicle)
        print(f"[INFO] Spawned vehicle at {spawn_points[0].location}")

        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "1280")
        camera_bp.set_attribute("image_size_y", "720")
        camera_bp.set_attribute("fov", "90")
        cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, cam_transform, attach_to=vehicle)
        actors.append(camera)
        print("[INFO] Attached RGB camera")

        q: "Queue[carla.Image]" = Queue()
        camera.listen(q.put)

        # Initial warm-up ticks
        print("[INFO] Warming up simulation...")
        for _ in range(30):
            world.tick()
        
        # Flush queue
        while not q.empty():
            try:
                q.get_nowait()
            except Empty:
                break

        print(f"[INFO] Capturing {len(presets)} weather presets...")
        print(f"[INFO] Settings: settle_ticks={args.settle_ticks}, frames={args.frames_per_preset}, ticks_between={args.ticks_between_frames}")

        for preset_idx, (name, weather) in enumerate(presets):
            print(f"\n[{preset_idx+1}/{len(presets)}] Capturing weather: {name}")
            
            # Apply weather
            world.set_weather(weather)
            
            # Flush old frames from queue
            while not q.empty():
                try:
                    q.get_nowait()
                except Empty:
                    break
            
            # Let CARLA settle with new weather
            print(f"  Settling for {args.settle_ticks} ticks...")
            for _ in range(max(1, args.settle_ticks)):
                world.tick()
                # Drain queue during settling to prevent buildup
                while not q.empty():
                    try:
                        q.get_nowait()
                    except Empty:
                        break
            
            # Small additional wait for rendering to stabilize
            time.sleep(0.3)
            
            # Flush again after sleep
            while not q.empty():
                try:
                    q.get_nowait()
                except Empty:
                    break

            # Capture frames
            frames = []
            print(f"  Capturing {args.frames_per_preset} frames...")
            for frame_idx in range(max(1, args.frames_per_preset)):
                # Tick simulation (multiple ticks between frames for motion)
                for _ in range(args.ticks_between_frames):
                    world.tick()
                
                # Get the latest frame
                try:
                    img = q.get(timeout=5.0)
                except Empty:
                    print(f"  [WARN] No image received for frame {frame_idx}, stopping")
                    break
                
                # Convert BGRA to RGB (fix color issue)
                rgb = _bgra_to_rgb(img.raw_data, img.height, img.width)
                frames.append(rgb)
                
                # Save individual PNG to pngs folder: <preset>_001.png, <preset>_002.png, ...
                png_path = png_dir / f"{name}_{frame_idx+1:03d}.png"
                try:
                    from PIL import Image
                    Image.fromarray(rgb).save(png_path)
                except Exception as e:
                    print(f"  [WARN] Failed to save PNG {png_path}: {e}")

            if not frames:
                print(f"  [WARN] No frames captured for {name}")
                continue
            
            print(f"  Captured {len(frames)} frames, saved to {png_dir}/{name}_*.png")

            # Create GIF from frames in gifs folder
            gif_path = gif_dir / f"{name}.gif"
            try:
                _save_frames_as_gif(frames, gif_path, args.gif_fps)
            except Exception as exc:
                print(f"  [ERROR] Failed to create GIF: {exc}")
            
            # Also save first frame as representative PNG in gifs folder
            representative_png = gif_dir / f"{name}.png"
            try:
                from PIL import Image
                Image.fromarray(frames[0]).save(representative_png)
            except Exception:
                pass

        print(f"\n[INFO] Done! Output saved to {output_dir}")
        print(f"       PNGs: {png_dir}")
        print(f"       GIFs: {gif_dir}")

    finally:
        print("\n[INFO] Cleaning up...")
        if camera:
            camera.stop()
        for actor in actors:
            try:
                actor.destroy()
            except Exception:
                pass
        world.apply_settings(original_settings)
        print("[INFO] Restored original world settings")


if __name__ == "__main__":
    main()
