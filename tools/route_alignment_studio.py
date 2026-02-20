#!/usr/bin/env python3
"""
Route Alignment Studio (local web app backend).

Purpose:
- Load many route XML files from a ZIP bundle or directory tree.
- Provide a browser UI for high-speed multi-actor alignment edits.
- Export edited routes back to ZIP while preserving bundle structure.

Usage examples:
  python tools/route_alignment_studio.py \
    --bundle /path/to/alignment_bundle.zip \
    --map-pkl /path/to/carla_map_cache.pkl \
    --open-browser

  python tools/route_alignment_studio.py \
    --routes-root /path/to/scenarios_root \
    --map-lines-json /path/to/map_lines.json
"""

from __future__ import annotations

import argparse
import io
import json
import math
import mimetypes
import os
import pickle
import re
import urllib.parse
import webbrowser
import zipfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

KNOWN_WAYPOINT_KEYS = {"x", "y", "z", "yaw", "pitch", "roll", "time"}
MACOS_IGNORED_PREFIXES = ("__MACOSX/",)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _format_float(value: Any) -> str:
    return f"{_safe_float(value):.6f}"


def _normalize_relpath(path: str) -> str:
    rel = str(PurePosixPath(path))
    rel = rel.lstrip("/")
    return rel


def _is_ignored_member(name: str) -> bool:
    rel = _normalize_relpath(name)
    if not rel:
        return True
    if any(rel.startswith(prefix) for prefix in MACOS_IGNORED_PREFIXES):
        return True
    base = PurePosixPath(rel).name
    if base.startswith("._"):
        return True
    return False


def _extract_lines_generic(obj: Any, out: List[List[Tuple[float, float]]], depth: int = 0) -> None:
    if obj is None or depth > 12:
        return
    if isinstance(obj, dict):
        # common direct polyline containers
        for key in ("lines", "centerline", "boundary", "polyline", "coordinates", "line", "points", "nodes"):
            if key in obj:
                _extract_lines_generic(obj[key], out, depth + 1)
        if "x" in obj and "y" in obj:
            out.append([(_safe_float(obj.get("x")), _safe_float(obj.get("y")))])
        for value in obj.values():
            _extract_lines_generic(value, out, depth + 1)
        return
    if isinstance(obj, (list, tuple)):
        # looks like polyline [ [x,y], [x,y], ... ]
        if obj:
            try:
                pts = [(_safe_float(p[0]), _safe_float(p[1])) for p in obj if isinstance(p, (list, tuple)) and len(p) >= 2]
                if len(pts) >= 2:
                    out.append(pts)
                    return
            except Exception:
                pass
        for value in obj:
            _extract_lines_generic(value, out, depth + 1)
        return
    if hasattr(obj, "x") and hasattr(obj, "y"):
        out.append([(_safe_float(getattr(obj, "x")), _safe_float(getattr(obj, "y")))])
        return
    if hasattr(obj, "__dict__"):
        _extract_lines_generic(obj.__dict__, out, depth + 1)


def _line_from_sequence(seq: Sequence[Any]) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    for item in seq:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            pts.append((_safe_float(item[0]), _safe_float(item[1])))
        elif isinstance(item, dict) and "x" in item and "y" in item:
            pts.append((_safe_float(item.get("x")), _safe_float(item.get("y"))))
    return pts


def _load_lines_from_json(path: Path) -> List[List[Tuple[float, float]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    lines: List[List[Tuple[float, float]]] = []

    if isinstance(data, dict) and "lines" in data:
        src = data.get("lines")
        if isinstance(src, list):
            for item in src:
                if isinstance(item, (list, tuple)):
                    line = _line_from_sequence(item)
                    if len(line) >= 2:
                        lines.append(line)
            return lines

    if isinstance(data, list):
        for item in data:
            if isinstance(item, (list, tuple)):
                line = _line_from_sequence(item)
                if len(line) >= 2:
                    lines.append(line)
            elif isinstance(item, dict):
                # Try common wrappers per feature/geometry entries
                for key in ("coordinates", "line", "polyline", "points", "nodes"):
                    if key in item and isinstance(item[key], (list, tuple)):
                        line = _line_from_sequence(item[key])
                        if len(line) >= 2:
                            lines.append(line)
        if lines:
            return lines

    _extract_lines_generic(data, lines)
    return [line for line in lines if len(line) >= 2]


def _load_lines_from_pkl(path: Path) -> List[List[Tuple[float, float]]]:
    class _Stub:
        def __init__(self, *args: Any, **kwargs: Any):
            self.__dict__.update(kwargs)
            for idx, value in enumerate(args):
                setattr(self, f"_arg{idx}", value)

    class _SafeUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str):  # pragma: no cover
            try:
                return super().find_class(module, name)
            except Exception:
                return _Stub

    try:
        with path.open("rb") as handle:
            data = pickle.load(handle)
    except Exception:
        with path.open("rb") as handle:
            data = _SafeUnpickler(handle).load()

    lines: List[List[Tuple[float, float]]] = []
    if isinstance(data, dict) and "lines" in data and isinstance(data["lines"], list):
        for item in data["lines"]:
            if isinstance(item, (list, tuple)):
                line = _line_from_sequence(item)
                if len(line) >= 2:
                    lines.append(line)
        if lines:
            return lines

    _extract_lines_generic(data, lines)
    return [line for line in lines if len(line) >= 2]


def _downsample_lines(
    lines: List[List[Tuple[float, float]]],
    *,
    max_lines: int,
    max_points_per_line: int,
) -> List[List[Tuple[float, float]]]:
    if not lines:
        return []

    out = lines
    if max_lines > 0 and len(out) > max_lines:
        stride = int(math.ceil(len(out) / float(max_lines)))
        out = out[:: max(1, stride)]

    final: List[List[Tuple[float, float]]] = []
    for line in out:
        if len(line) <= 2:
            final.append(line)
            continue
        if max_points_per_line > 0 and len(line) > max_points_per_line:
            stride = int(math.ceil(len(line) / float(max_points_per_line)))
            sampled = line[:: max(1, stride)]
            if sampled[-1] != line[-1]:
                sampled.append(line[-1])
            final.append(sampled)
        else:
            final.append(line)
    return final


def load_map_lines(
    map_lines_json: Optional[Path],
    map_pkl: Optional[Path],
    *,
    max_lines: int,
    max_points_per_line: int,
) -> List[List[Tuple[float, float]]]:
    lines: List[List[Tuple[float, float]]] = []
    if map_lines_json is not None:
        lines.extend(_load_lines_from_json(map_lines_json.expanduser().resolve()))
    if map_pkl is not None:
        lines.extend(_load_lines_from_pkl(map_pkl.expanduser().resolve()))

    # deduplicate exact lines
    dedup: List[List[Tuple[float, float]]] = []
    seen: set[Tuple[Tuple[float, float], ...]] = set()
    for line in lines:
        rounded = tuple((round(x, 4), round(y, 4)) for (x, y) in line)
        if rounded in seen:
            continue
        seen.add(rounded)
        dedup.append([(float(x), float(y)) for (x, y) in line])

    return _downsample_lines(
        dedup,
        max_lines=max_lines,
        max_points_per_line=max_points_per_line,
    )


def _infer_scenario_name(relpath: str) -> str:
    rel = _normalize_relpath(relpath)
    parts = [p for p in PurePosixPath(rel).parts if p not in (".", "")]
    if not parts:
        return "default"

    if "carla_log_export" in parts:
        idx = parts.index("carla_log_export")
        if idx > 0:
            return parts[idx - 1]

    # Common exported route layout: <scenario>/.../*.xml
    if len(parts) >= 2 and parts[0] not in ("actors", "routes"):
        return parts[0]

    return "default"


def _parse_route_xml(xml_bytes: bytes) -> Dict[str, Any]:
    root = ET.fromstring(xml_bytes)
    route_node = root.find("route")
    if route_node is None:
        route_node = root.find(".//route")
    if route_node is None:
        raise ValueError("Missing <route> element")

    route_attrs = dict(route_node.attrib)
    role = str(route_attrs.get("role", "npc"))
    town = str(route_attrs.get("town", ""))
    route_id = str(route_attrs.get("id", "0"))

    waypoints: List[Dict[str, Any]] = []
    for idx, wp_node in enumerate(route_node.findall("waypoint")):
        attrs = dict(wp_node.attrib)
        waypoint: Dict[str, Any] = {
            "index": idx,
            "x": _safe_float(attrs.get("x", 0.0)),
            "y": _safe_float(attrs.get("y", 0.0)),
            "z": _safe_float(attrs.get("z", 0.0)),
            "yaw": _safe_float(attrs.get("yaw", 0.0)),
            "pitch": _safe_float(attrs.get("pitch", 0.0)),
            "roll": _safe_float(attrs.get("roll", 0.0)),
            "time": attrs.get("time"),
            "extra_attrs": {k: v for k, v in attrs.items() if k not in KNOWN_WAYPOINT_KEYS},
        }
        waypoints.append(waypoint)

    return {
        "route_attrs": route_attrs,
        "role": role,
        "town": town,
        "route_id": route_id,
        "waypoints": waypoints,
    }


def _indent_xml(elem: ET.Element, level: int = 0) -> None:
    indent = "  "
    pad = "\n" + level * indent
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = pad + indent
        for child in elem:
            _indent_xml(child, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = pad
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = pad


def _serialize_route_xml(route: Dict[str, Any]) -> bytes:
    route_attrs = dict(route.get("route_attrs") or {})

    # Keep attrs coherent with edited route metadata.
    route_attrs["id"] = str(route.get("route_id", route_attrs.get("id", "0")))
    route_attrs["town"] = str(route.get("town", route_attrs.get("town", "")))
    route_attrs["role"] = str(route.get("role", route_attrs.get("role", "npc")))

    root = ET.Element("routes")
    route_node = ET.SubElement(root, "route", route_attrs)

    for wp in route.get("waypoints") or []:
        attrs: Dict[str, str] = {
            "x": _format_float(wp.get("x", 0.0)),
            "y": _format_float(wp.get("y", 0.0)),
            "z": _format_float(wp.get("z", 0.0)),
            "yaw": _format_float(wp.get("yaw", 0.0)),
            "pitch": _format_float(wp.get("pitch", 0.0)),
            "roll": _format_float(wp.get("roll", 0.0)),
        }
        time_value = wp.get("time")
        if time_value is not None and str(time_value) != "":
            try:
                attrs["time"] = _format_float(time_value)
            except Exception:
                attrs["time"] = str(time_value)

        extra_attrs = wp.get("extra_attrs") or {}
        if isinstance(extra_attrs, dict):
            for key, value in extra_attrs.items():
                if key in attrs:
                    continue
                attrs[str(key)] = str(value)

        ET.SubElement(route_node, "waypoint", attrs)

    _indent_xml(root)
    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return xml_bytes


def _load_project_from_zip(bundle_path: Path) -> Tuple[Dict[str, Any], Dict[str, bytes]]:
    scenarios: Dict[str, List[Dict[str, Any]]] = {}
    extra_files: Dict[str, bytes] = {}

    bundle_path = bundle_path.expanduser().resolve()
    with zipfile.ZipFile(bundle_path, "r") as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            rel = _normalize_relpath(member.filename)
            if _is_ignored_member(rel):
                continue
            data = archive.read(member.filename)

            if rel.lower().endswith(".xml"):
                try:
                    parsed = _parse_route_xml(data)
                except Exception:
                    extra_files[rel] = data
                    continue
                scenario_name = _infer_scenario_name(rel)
                route = {
                    "uid": f"{scenario_name}:{rel}",
                    "name": PurePosixPath(rel).stem,
                    "relpath": rel,
                    "role": parsed["role"],
                    "town": parsed["town"],
                    "route_id": parsed["route_id"],
                    "route_attrs": parsed["route_attrs"],
                    "waypoints": parsed["waypoints"],
                }
                scenarios.setdefault(scenario_name, []).append(route)
            else:
                extra_files[rel] = data

    scenario_items: List[Dict[str, Any]] = []
    for scenario_name in sorted(scenarios.keys()):
        routes = sorted(scenarios[scenario_name], key=lambda r: str(r.get("relpath", "")))
        scenario_items.append(
            {
                "name": scenario_name,
                "routes": routes,
            }
        )

    project = {
        "project_name": bundle_path.stem,
        "source_type": "zip",
        "source_path": str(bundle_path),
        "scenarios": scenario_items,
    }
    return project, extra_files


def _load_project_from_routes_root(routes_root: Path) -> Tuple[Dict[str, Any], Dict[str, bytes]]:
    scenarios: Dict[str, List[Dict[str, Any]]] = {}
    extra_files: Dict[str, bytes] = {}

    routes_root = routes_root.expanduser().resolve()
    if not routes_root.exists():
        raise FileNotFoundError(routes_root)

    for file_path in sorted(p for p in routes_root.rglob("*") if p.is_file()):
        rel = _normalize_relpath(file_path.relative_to(routes_root).as_posix())
        if _is_ignored_member(rel):
            continue
        data = file_path.read_bytes()

        if file_path.suffix.lower() == ".xml":
            try:
                parsed = _parse_route_xml(data)
            except Exception:
                extra_files[rel] = data
                continue
            scenario_name = _infer_scenario_name(rel)
            route = {
                "uid": f"{scenario_name}:{rel}",
                "name": PurePosixPath(rel).stem,
                "relpath": rel,
                "role": parsed["role"],
                "town": parsed["town"],
                "route_id": parsed["route_id"],
                "route_attrs": parsed["route_attrs"],
                "waypoints": parsed["waypoints"],
            }
            scenarios.setdefault(scenario_name, []).append(route)
        else:
            extra_files[rel] = data

    scenario_items: List[Dict[str, Any]] = []
    for scenario_name in sorted(scenarios.keys()):
        routes = sorted(scenarios[scenario_name], key=lambda r: str(r.get("relpath", "")))
        scenario_items.append({"name": scenario_name, "routes": routes})

    project = {
        "project_name": routes_root.name,
        "source_type": "directory",
        "source_path": str(routes_root),
        "scenarios": scenario_items,
    }
    return project, extra_files


def _project_stats(project: Dict[str, Any]) -> Dict[str, int]:
    scenario_count = 0
    route_count = 0
    waypoint_count = 0
    for scenario in project.get("scenarios", []):
        scenario_count += 1
        for route in scenario.get("routes", []):
            route_count += 1
            waypoint_count += len(route.get("waypoints", []))
    return {
        "scenario_count": scenario_count,
        "route_count": route_count,
        "waypoint_count": waypoint_count,
    }


def _should_include_file_for_scope(relpath: str, selected_scenarios: set[str], include_all: bool) -> bool:
    if include_all:
        return True
    if not selected_scenarios:
        return False
    scenario = _infer_scenario_name(relpath)
    if scenario in selected_scenarios:
        return True
    if len(selected_scenarios) == 1 and scenario == "default":
        # root-level metadata for single-scenario exports
        return True
    return False


def _create_export_zip(
    project: Dict[str, Any],
    extra_files: Dict[str, bytes],
    *,
    scope: str,
    scenario_names: Sequence[str],
) -> bytes:
    include_all = str(scope).lower() == "all"
    selected_scenarios = set(str(name) for name in scenario_names)

    if include_all:
        selected_scenarios = {str(s.get("name")) for s in project.get("scenarios", [])}

    xml_payload_by_relpath: Dict[str, bytes] = {}
    for scenario in project.get("scenarios", []):
        scenario_name = str(scenario.get("name"))
        if not include_all and scenario_name not in selected_scenarios:
            continue
        for route in scenario.get("routes", []):
            relpath = _normalize_relpath(str(route.get("relpath", "")))
            if not relpath:
                continue
            xml_payload_by_relpath[relpath] = _serialize_route_xml(route)

    if not xml_payload_by_relpath:
        raise ValueError("No XML routes selected for export.")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        # Write edited XML files first.
        for relpath in sorted(xml_payload_by_relpath.keys()):
            archive.writestr(relpath, xml_payload_by_relpath[relpath])

        # Keep non-XML files from original source where in selected scope.
        for relpath, payload in sorted(extra_files.items()):
            rel = _normalize_relpath(relpath)
            if rel in xml_payload_by_relpath:
                continue
            if not _should_include_file_for_scope(rel, selected_scenarios, include_all):
                continue
            archive.writestr(rel, payload)

    return buffer.getvalue()


class StudioHTTPHandler(BaseHTTPRequestHandler):
    assets_dir: Path
    project_payload: Dict[str, Any]
    extra_files: Dict[str, bytes]

    server_version = "RouteAlignmentStudio/1.0"

    def _send_json(self, payload: Dict[str, Any], status: int = HTTPStatus.OK) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_bytes(self, data: bytes, *, content_type: str, status: int = HTTPStatus.OK, filename: Optional[str] = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(data)

    def _send_text(self, text: str, *, status: int = HTTPStatus.OK) -> None:
        data = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_static(self, relpath: str) -> None:
        requested = Path(relpath)
        path = (self.assets_dir / requested).resolve()
        if self.assets_dir not in path.parents and path != self.assets_dir:
            self._send_text("Not Found", status=HTTPStatus.NOT_FOUND)
            return
        if not path.exists() or not path.is_file():
            self._send_text("Not Found", status=HTTPStatus.NOT_FOUND)
            return

        ctype, _ = mimetypes.guess_type(str(path))
        if not ctype:
            ctype = "application/octet-stream"
        self._send_bytes(path.read_bytes(), content_type=ctype)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self._serve_static("index.html")
            return

        if path == "/api/health":
            self._send_json({"ok": True})
            return

        if path == "/api/project":
            self._send_json({"project": self.project_payload})
            return

        if path.startswith("/"):
            self._serve_static(path.lstrip("/"))
            return

        self._send_text("Not Found", status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path != "/api/export":
            self._send_text("Not Found", status=HTTPStatus.NOT_FOUND)
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except Exception:
            content_length = 0
        raw = self.rfile.read(max(0, content_length))
        try:
            payload = json.loads(raw.decode("utf-8") if raw else "{}")
        except Exception:
            self._send_json({"error": "Invalid JSON body."}, status=HTTPStatus.BAD_REQUEST)
            return

        project = payload.get("project")
        if not isinstance(project, dict):
            self._send_json({"error": "Missing project payload."}, status=HTTPStatus.BAD_REQUEST)
            return

        scope = str(payload.get("scope", "all"))
        scenario_names_raw = payload.get("scenario_names") or []
        if not isinstance(scenario_names_raw, list):
            scenario_names_raw = []
        scenario_names = [str(name) for name in scenario_names_raw]

        try:
            zip_bytes = _create_export_zip(
                project,
                self.extra_files,
                scope=scope,
                scenario_names=scenario_names,
            )
        except Exception as exc:  # pylint: disable=broad-except
            self._send_json({"error": f"Export failed: {exc}"}, status=HTTPStatus.BAD_REQUEST)
            return

        filename = str(payload.get("filename") or "aligned_routes.zip")
        if not filename.lower().endswith(".zip"):
            filename = f"{filename}.zip"
        self._send_bytes(zip_bytes, content_type="application/zip", filename=filename)

    def log_message(self, fmt: str, *args: Any) -> None:
        # Keep console concise and focused.
        print(f"[Studio] {self.address_string()} - {fmt % args}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Route Alignment Studio local server")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bundle", type=Path, help="ZIP bundle containing many scenario XML routes.")
    source.add_argument("--routes-root", type=Path, help="Directory containing scenario XML routes.")

    parser.add_argument(
        "--map-lines-json",
        type=Path,
        default=None,
        help="Optional JSON file with map polylines for snap-to-road tools.",
    )
    parser.add_argument(
        "--map-pkl",
        type=Path,
        default=None,
        help="Optional PKL map cache with polylines for snap-to-road tools.",
    )
    parser.add_argument(
        "--max-map-lines",
        type=int,
        default=25000,
        help="Limit number of map polylines sent to browser (default: 25000).",
    )
    parser.add_argument(
        "--max-points-per-line",
        type=int,
        default=200,
        help="Limit number of points per map polyline sent to browser (default: 200).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765)")
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open the studio URL automatically in your default browser.",
    )
    return parser.parse_args()


def _load_project_and_extras(args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, bytes]]:
    if args.bundle is not None:
        project, extra_files = _load_project_from_zip(args.bundle)
    else:
        assert args.routes_root is not None
        project, extra_files = _load_project_from_routes_root(args.routes_root)

    map_lines = load_map_lines(
        args.map_lines_json,
        args.map_pkl,
        max_lines=max(0, int(args.max_map_lines)),
        max_points_per_line=max(0, int(args.max_points_per_line)),
    )

    # Add map data + stats to payload
    project["map_lines"] = [[[float(x), float(y)] for (x, y) in line] for line in map_lines]
    project["stats"] = _project_stats(project)
    return project, extra_files


def main() -> None:
    args = parse_args()

    assets_dir = (Path(__file__).resolve().parent / "route_alignment_studio").resolve()
    if not (assets_dir / "index.html").exists():
        raise FileNotFoundError(f"Missing frontend asset: {assets_dir / 'index.html'}")

    project, extra_files = _load_project_and_extras(args)
    stats = project.get("stats", {})
    print(
        "[INFO] Loaded project: "
        f"scenarios={stats.get('scenario_count', 0)}, "
        f"routes={stats.get('route_count', 0)}, "
        f"waypoints={stats.get('waypoint_count', 0)}, "
        f"map_lines={len(project.get('map_lines', []))}"
    )

    handler_cls = type(
        "BoundStudioHTTPHandler",
        (StudioHTTPHandler,),
        {
            "assets_dir": assets_dir,
            "project_payload": project,
            "extra_files": extra_files,
        },
    )

    server = ThreadingHTTPServer((args.host, int(args.port)), handler_cls)
    url = f"http://{args.host}:{int(args.port)}/"
    print(f"[INFO] Route Alignment Studio running at {url}")
    print("[INFO] Press Ctrl+C to stop.")

    if args.open_browser:
        try:
            webbrowser.open(url)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[WARN] Could not open browser automatically: {exc}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down Route Alignment Studio.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
