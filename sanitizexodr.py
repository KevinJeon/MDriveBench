#!/usr/bin/env python3
import argparse
import xml.etree.ElementTree as ET

def strip_children(parent, tag):
    removed = 0
    for child in list(parent):
        if child.tag == tag:
            parent.remove(child)
            removed += 1
    return removed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_xodr", required=True)
    ap.add_argument("--out_xodr", required=True)
    ap.add_argument("--drop_lateral_profile", action="store_true", default=True,
                    help="Remove <lateralProfile> blocks (default: true)")
    ap.add_argument("--drop_elevation_profile", action="store_true", default=True,
                    help="Remove <elevationProfile> blocks (default: true)")
    ap.add_argument("--drop_lane_offset", action="store_true",
                    help="Remove <laneOffset> entries (try if still failing)")
    ap.add_argument("--drop_objects", action="store_true",
                    help="Remove <objects> blocks from roads (parking spaces, etc.)")
    ap.add_argument("--drop_signals", action="store_true",
                    help="Remove <signals> blocks from roads")
    ap.add_argument("--drop_road_mark", action="store_true",
                    help="Remove <roadMark> entries inside lanes")
    ap.add_argument("--drop_lane_height", action="store_true", default=True,
                    help="Remove <height> entries inside lanes (default: true)")
    ap.add_argument("--drop_lane_speed", action="store_true", default=True,
                    help="Remove <speed> entries inside lanes (default: true)")
    ap.add_argument("--drop_user_data", action="store_true", default=True,
                    help="Remove <userData> blocks everywhere (default: true)")
    ap.add_argument("--drop_geo_reference", action="store_true",
                    help="Remove <geoReference> from header")
    ap.add_argument("--min_geometry_length", type=float, default=0.01,
                    help="Remove geometry segments shorter than this (meters, default: 0.01)")
    args = ap.parse_args()

    tree = ET.parse(args.in_xodr)
    root = tree.getroot()

    removed_lp = removed_ep = removed_lo = 0
    removed_objects = removed_signals = removed_roadmark = 0
    removed_userdata = removed_georef = 0
    removed_height = removed_speed = 0
    removed_tiny_geoms = 0

    header = root.find("header")
    if header is not None:
        if args.drop_user_data:
            removed_userdata += strip_children(header, "userData")
        if args.drop_geo_reference:
            removed_georef += strip_children(header, "geoReference")

    for road in root.findall("road"):
        if args.drop_lateral_profile:
            removed_lp += strip_children(road, "lateralProfile")
        if args.drop_elevation_profile:
            removed_ep += strip_children(road, "elevationProfile")
        if args.drop_objects:
            removed_objects += strip_children(road, "objects")
        if args.drop_signals:
            removed_signals += strip_children(road, "signals")
        
        # Remove tiny geometry segments that CARLA can't parse
        planView = road.find("planView")
        if planView is not None and args.min_geometry_length > 0:
            for geom in list(planView.findall("geometry")):
                length = float(geom.get("length", 0))
                if length < args.min_geometry_length:
                    planView.remove(geom)
                    removed_tiny_geoms += 1

        lanes = road.find("lanes")
        if lanes is not None:
            if args.drop_lane_offset:
                # remove all <laneOffset .../>
                for child in list(lanes):
                    if child.tag == "laneOffset":
                        lanes.remove(child)
                        removed_lo += 1
            
            if args.drop_road_mark:
                for lane in lanes.findall(".//lane"):
                    for child in list(lane):
                        if child.tag == "roadMark":
                            lane.remove(child)
                            removed_roadmark += 1
            
            if args.drop_user_data:
                # Remove userData from all lanes
                for lane in lanes.findall(".//lane"):
                    removed_userdata += strip_children(lane, "userData")
            
            if args.drop_lane_height:
                # Remove height from all lanes
                for lane in lanes.findall(".//lane"):
                    removed_height += strip_children(lane, "height")
            
            if args.drop_lane_speed:
                # Remove speed from all lanes
                for lane in lanes.findall(".//lane"):
                    removed_speed += strip_children(lane, "speed")

    tree.write(args.out_xodr, encoding="UTF-8", xml_declaration=True)
    print("Wrote:", args.out_xodr)
    print("Removed lateralProfile:", removed_lp)
    print("Removed elevationProfile:", removed_ep)
    print("Removed laneOffset:", removed_lo)
    print("Removed objects:", removed_objects)
    print("Removed signals:", removed_signals)
    print("Removed roadMark:", removed_roadmark)
    print("Removed userData:", removed_userdata)
    print("Removed geoReference:", removed_georef)
    print("Removed lane height:", removed_height)
    print("Removed lane speed:", removed_speed)
    print("Removed tiny geometries:", removed_tiny_geoms)

if __name__ == "__main__":
    main()
