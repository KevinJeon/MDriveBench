# Route Alignment Studio

`Route Alignment Studio` is a local, browser-based editor for fast multi-actor alignment of CARLA route XML files.

## What It Does

- Loads many scenarios/routes from one ZIP bundle (or directory tree).
- Lets you step scenario-by-scenario and edit paths one by one.
- Supports multi-actor selection and direct drag alignment.
- Supports waypoint-level editing with multi-select, drag, smoothing, and resampling.
- Supports snap-to-road with adjustable lateral offset and optional yaw alignment.
- Provides undo/redo and export back to ZIP.

## Launch

```bash
python tools/route_alignment_studio.py \
  --bundle /path/to/alignment_bundle.zip \
  --map-pkl /path/to/carla_map_cache.pkl \
  --open-browser
```

You can also load from a directory instead of ZIP:

```bash
python tools/route_alignment_studio.py \
  --routes-root /path/to/routes_root \
  --map-lines-json /path/to/map_lines.json \
  --open-browser
```

## Workflow

1. Generate/edit routes with your pipeline.
2. Build an alignment bundle ZIP from batch replay (enabled by default in `run_train1_logreplay_batch.py`).
3. Copy bundle to your local machine (if server is headless).
4. Run `route_alignment_studio.py` locally.
5. Edit alignment in the browser:
   - Actor mode: select multiple actors and drag/translate/rotate.
   - Waypoint mode: multi-select waypoints, drag, smooth, resample.
   - Road snap: snap full actors or selected waypoints to nearest road geometry.
6. Export ZIP and use it downstream.

## Notes

- For best road snapping results, provide `--map-pkl` or `--map-lines-json`.
- Exports preserve XML structure/paths and include non-XML bundle files in scope.
- Keyboard shortcuts are listed in the left panel of the app.
