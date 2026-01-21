#!/usr/bin/env python3
import argparse
import os
import queue
import time
import math

import carla


def get_map_center(world: carla.World):
    """
    Heuristic: use spawn points' mean as center. Works for many maps, including OpenDRIVE standalone.
    """
    m = world.get_map()
    sps = m.get_spawn_points()
    if not sps:
        # fallback: origin
        return carla.Location(x=0.0, y=0.0, z=0.0)
    xs = [sp.location.x for sp in sps]
    ys = [sp.location.y for sp in sps]
    zs = [sp.location.z for sp in sps]
    return carla.Location(x=sum(xs) / len(xs), y=sum(ys) / len(ys), z=sum(zs) / len(zs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--xodr", required=True, help="Path to .xodr file")
    ap.add_argument("--out", default="bev.png", help="Output screenshot path (.png)")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--alt", type=float, default=120.0, help="Camera altitude (meters)")
    ap.add_argument("--step", type=float, default=0.05, help="Fixed delta seconds for sync mode")
    args = ap.parse_args()

    if not os.path.exists(args.xodr):
        raise FileNotFoundError(args.xodr)

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)

    # --- Load OpenDRIVE world from XODR content (string) ---
    with open(args.xodr, "r", encoding="utf-8") as f:
        xodr_str = f.read()

    # Optional mesh generation params (available in docs for OpenDRIVE mode)
    # If this class isn't present in your build, you can delete the params argument
    try:
        params = carla.OpendriveGenerationParameters(
            vertex_distance=2.0,
            max_road_length=50.0,
            wall_height=0.0,
            additional_width=0.6,
            smooth_junctions=True,
            enable_mesh_visibility=True
        )
        world = client.generate_opendrive_world(xodr_str, params)
    except Exception:
        # fallback: call without params (still supported: opendrive as string)
        world = client.generate_opendrive_world(xodr_str)

    # --- Enable synchronous mode for deterministic frame capture ---
    settings = world.get_settings()
    original_settings = carla.WorldSettings(
        no_rendering_mode=settings.no_rendering_mode,
        synchronous_mode=settings.synchronous_mode,
        fixed_delta_seconds=settings.fixed_delta_seconds,
        max_substep_delta_time=settings.max_substep_delta_time,
        max_substeps=settings.max_substeps
    )
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = args.step
    settings.no_rendering_mode = False
    world.apply_settings(settings)

    actors = []

    try:
        # Choose a center point and place spectator overhead looking straight down
        center = get_map_center(world)
        spectator = world.get_spectator()
        spectator_tf = carla.Transform(
            carla.Location(x=center.x, y=center.y, z=center.z + args.alt),
            carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0)
        )
        spectator.set_transform(spectator_tf)

        # Spawn an RGB camera attached to spectator for BEV screenshot
        bp_lib = world.get_blueprint_library()
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(args.width))
        cam_bp.set_attribute("image_size_y", str(args.height))
        cam_bp.set_attribute("fov", str(args.fov))

        cam_tf = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0),
                                 carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0))
        camera = world.spawn_actor(cam_bp, cam_tf, attach_to=spectator)
        actors.append(camera)

        q = queue.Queue()

        def _on_image(image):
            q.put(image)

        camera.listen(_on_image)

        # Tick a few frames so the camera produces a valid image
        for _ in range(5):
            world.tick()

        image = q.get(timeout=5.0)
        # Save single frame
        out_path = os.path.abspath(args.out)
        image.save_to_disk(out_path)
        print(f"Saved BEV screenshot to: {out_path}")

    finally:
        # Cleanup
        for a in actors:
            a.destroy()
        world.apply_settings(original_settings)


if __name__ == "__main__":
    main()
