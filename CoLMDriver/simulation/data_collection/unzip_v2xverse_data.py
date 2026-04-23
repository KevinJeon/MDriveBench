#!/usr/bin/env python3
"""
Unzip route archives: create a folder named like the zip (without .zip) and
extract all contents into that folder.

  dataset/weather-0/data/foo.zip  ->  dataset/weather-0/data/foo/...
  dataset/bar.zip                 ->  dataset/bar/...   (zip directly under --root)

Typical V2Xverse layout:
  dataset/weather-0/data/routes_town01_0_w0_....zip

Usage (from repo root, e.g. CoLMDriver as cwd):
  python simulation/data_collection/unzip_v2xverse_data.py
  python simulation/data_collection/unzip_v2xverse_data.py --root dataset --dry-run
  python simulation/data_collection/unzip_v2xverse_data.py --root /app/dataset --delete-zip

For measurement key stats only, use inspect_v2xverse_ego_measurements.py
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path


def _is_under_weather_data(zip_path: Path) -> bool:
    """True if path is .../weather-<id>/data/.../<file>.zip"""
    parts = zip_path.parts
    for i, part in enumerate(parts):
        if part.startswith("weather-") and i + 1 < len(parts) and parts[i + 1] == "data":
            return True
    return False


def _eligible_zip(zip_path: Path, root: Path) -> bool:
    """Zip under .../weather-*/data/** or directly under root (e.g. dataset/*.zip)."""
    try:
        if zip_path.parent.resolve() == root.resolve():
            return True
    except OSError:
        pass
    return _is_under_weather_data(zip_path)


def _dest_dir_for_zip(zip_path: Path) -> Path:
    """Directory named after the zip stem, next to the zip file."""
    return zip_path.parent / zip_path.stem


def unzip_one(
    zip_path: Path,
    dest: Path,
    *,
    skip_if_exists: bool,
    dry_run: bool,
    delete_zip: bool,
) -> str:
    """Returns 'ok' | 'skip' | 'dry' | 'err'. `dest` is .../<zip_stem>/"""
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            namelist = zf.namelist()
            if not namelist:
                print(f"[skip empty] {zip_path}", file=sys.stderr)
                return "skip"

            if skip_if_exists and dest.exists() and dest.is_dir():
                try:
                    if any(dest.iterdir()):
                        print(f"[skip exists] {zip_path} -> {dest}")
                        return "skip"
                except OSError:
                    pass

            if dry_run:
                print(f"[dry-run] would extract {zip_path} -> {dest}")
                return "dry"

            dest.mkdir(parents=True, exist_ok=True)
            zf.extractall(dest)

        if delete_zip:
            zip_path.unlink()
            print(f"[done + rm] {zip_path} -> {dest}")
        else:
            print(f"[done] {zip_path} -> {dest}")
        return "ok"
    except zipfile.BadZipFile as e:
        print(f"[bad zip] {zip_path}: {e}", file=sys.stderr)
        return "err"
    except OSError as e:
        print(f"[io error] {zip_path}: {e}", file=sys.stderr)
        return "err"


def main() -> int:
    p = argparse.ArgumentParser(
        description="Unzip zips into a sibling folder named <zip_basename_without_ext>/"
    )
    p.add_argument(
        "--root",
        default="dataset",
        help="Dataset root (contains weather-0, ... or zips). Default: dataset",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be extracted",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Extract even if the target folder already exists and is non-empty",
    )
    p.add_argument(
        "--delete-zip",
        action="store_true",
        help="Remove zip after successful extract (use when disk is tight)",
    )
    args = p.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 1

    all_zips = sorted(root.rglob("*.zip"))
    eligible = [z for z in all_zips if _eligible_zip(z, root)]

    if not eligible:
        print(f"No eligible .zip files under {root}")
        return 0

    stats = {"ok": 0, "skip": 0, "dry": 0, "err": 0}
    for zip_path in eligible:
        dest_dir = _dest_dir_for_zip(zip_path)
        r = unzip_one(
            zip_path,
            dest_dir,
            skip_if_exists=not args.force,
            dry_run=args.dry_run,
            delete_zip=args.delete_zip,
        )
        stats[r] = stats.get(r, 0) + 1

    skipped = len(all_zips) - len(eligible)
    if skipped:
        print(f"(skipped {skipped} zip(s) outside .../weather-*/data/ or {root}/)")
    print(
        "Unzip summary:",
        f"ok={stats.get('ok', 0)}",
        f"skip={stats.get('skip', 0)}",
        f"dry={stats.get('dry', 0)}",
        f"err={stats.get('err', 0)}",
    )

    return 0 if stats.get("err", 0) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
