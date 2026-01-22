from typing import List, Optional

import numpy as np

import generate_legal_paths as glp

from .features import compute_crop_features, detect_junction_centers
from .models import CropFeatures, CropKey


def build_candidate_crops_for_town(
    town_name: str,
    town_json_path: str,
    radii: List[float],
    min_path_len: float,
    max_paths: int,
    max_depth: int,
) -> List[CropFeatures]:
    data = glp.load_nodes(town_json_path)
    segments_full = glp.build_segments(data, min_points=6)
    adj_full = glp.build_connectivity(segments_full)
    jcenters = detect_junction_centers(segments_full, adj_full)

    feats: List[CropFeatures] = []
    for jc in jcenters:
        cx, cy = float(jc[0]), float(jc[1])
        for r in radii:
            ck = CropKey(cx - r, cx + r, cy - r, cy + r)
            f = compute_crop_features(
                town_name=town_name,
                segments_full=segments_full,
                junction_centers=jcenters,
                center_xy=(cx, cy),
                crop=ck,
                min_path_len=min_path_len,
                max_paths=max_paths,
                max_depth=max_depth,
            )
            if f is not None:
                feats.append(f)

    uniq = {}
    for f in feats:
        k = f.crop.to_str()
        if k not in uniq:
            uniq[k] = f
        else:
            a = (f.junction_count, f.area)
            b = (uniq[k].junction_count, uniq[k].area)
            if a < b:
                uniq[k] = f

    out = list(uniq.values())
    out.sort(key=lambda x: (x.junction_count, x.area))
    return out


def build_corridor_candidate_crops_for_town(
    town_name: str,
    town_json_path: str,
    min_corridor_length: float = 80.0,
    crop_width: float = 20.0,
    min_path_len: float = 15.0,
    max_paths: int = 100,
    max_depth: int = 5,
) -> List[CropFeatures]:
    """
    Build corridor-specific crop candidates by sampling from long straight road segments.
    
    Unlike build_candidate_crops_for_town() which centers crops on junction centers,
    this function identifies long straight road segments and creates crops centered
    on their midpoints, specifically for TWO_LANE_CORRIDOR topology.
    
    Args:
        town_name: Name of the CARLA town
        town_json_path: Path to the town's node JSON file
        min_corridor_length: Minimum length of road segment to consider as corridor (meters)
        crop_width: Width of crop perpendicular to road direction (meters)
        min_path_len: Minimum path length for feature computation
        max_paths: Maximum paths to generate
        max_depth: Maximum DFS depth for path generation
        
    Returns:
        List of CropFeatures for valid corridor regions (no junctions inside, 2 directions only)
    """
    data = glp.load_nodes(town_json_path)
    segments_full = glp.build_segments(data, min_points=6)
    adj_full = glp.build_connectivity(segments_full)
    jcenters = detect_junction_centers(segments_full, adj_full)
    
    # Group segments by road_id to find long corridors
    segments_by_road: dict = {}
    for seg in segments_full:
        rid = seg.road_id
        if rid not in segments_by_road:
            segments_by_road[rid] = []
        segments_by_road[rid].append(seg)
    
    # Find long road segments that could be corridors
    corridor_candidates = []
    
    for rid, segs in segments_by_road.items():
        # Compute total length of this road
        total_length = sum(seg.length() for seg in segs)
        
        if total_length < min_corridor_length:
            continue
            
        # Find the extent of this road
        all_points = []
        for seg in segs:
            all_points.extend(seg.points)
        
        if not all_points:
            continue
            
        points_arr = np.array(all_points)
        x_min, y_min = points_arr.min(axis=0)
        x_max, y_max = points_arr.max(axis=0)
        
        # Determine road orientation (horizontal vs vertical)
        x_span = x_max - x_min
        y_span = y_max - y_min
        is_horizontal = x_span > y_span
        
        # Find midpoint of the road
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        
        # Check distance from midpoint to nearest junction
        min_dist_to_junction = float('inf')
        for jc in jcenters:
            dist = np.sqrt((jc[0] - cx)**2 + (jc[1] - cy)**2)
            min_dist_to_junction = min(min_dist_to_junction, dist)
        
        # Only consider if midpoint is far enough from junctions
        if min_dist_to_junction < 30.0:
            continue
            
        corridor_candidates.append({
            'road_id': rid,
            'center': (cx, cy),
            'is_horizontal': is_horizontal,
            'x_range': (x_min, x_max),
            'y_range': (y_min, y_max),
            'length': total_length,
            'dist_to_junction': min_dist_to_junction,
        })
    
    # Generate corridor crops
    feats: List[CropFeatures] = []
    
    for cand in corridor_candidates:
        cx, cy = cand['center']
        is_horizontal = cand['is_horizontal']
        x_min_road, x_max_road = cand['x_range']
        y_min_road, y_max_road = cand['y_range']
        
        # Create elongated crop along the road direction
        if is_horizontal:
            # Horizontal road: wide X, narrow Y
            # Find safe X range that avoids junctions
            safe_x_min = x_min_road
            safe_x_max = x_max_road
            for jc in jcenters:
                if y_min_road - 20 <= jc[1] <= y_max_road + 20:
                    # Junction is near this road's Y range
                    if jc[0] < cx:
                        safe_x_min = max(safe_x_min, jc[0] + 15)
                    else:
                        safe_x_max = min(safe_x_max, jc[0] - 15)
            
            # Create crop in the safe zone
            crop_x_min = safe_x_min + 5
            crop_x_max = safe_x_max - 5
            crop_y_min = cy - crop_width
            crop_y_max = cy + crop_width
        else:
            # Vertical road: narrow X, wide Y
            safe_y_min = y_min_road
            safe_y_max = y_max_road
            for jc in jcenters:
                if x_min_road - 20 <= jc[0] <= x_max_road + 20:
                    if jc[1] < cy:
                        safe_y_min = max(safe_y_min, jc[1] + 15)
                    else:
                        safe_y_max = min(safe_y_max, jc[1] - 15)
            
            crop_x_min = cx - crop_width
            crop_x_max = cx + crop_width
            crop_y_min = safe_y_min + 5
            crop_y_max = safe_y_max - 5
        
        # Validate crop dimensions
        if crop_x_max - crop_x_min < 40 or crop_y_max - crop_y_min < 10:
            continue
            
        crop = CropKey(crop_x_min, crop_x_max, crop_y_min, crop_y_max)
        crop_center = ((crop_x_min + crop_x_max) / 2, (crop_y_min + crop_y_max) / 2)
        
        # Check that no junctions are inside this crop
        jcts_inside = [
            jc for jc in jcenters
            if crop_x_min <= jc[0] <= crop_x_max and crop_y_min <= jc[1] <= crop_y_max
        ]
        if jcts_inside:
            continue
        
        # Compute features with corridor_mode=True
        f = compute_crop_features(
            town_name=town_name,
            segments_full=segments_full,
            junction_centers=jcenters,
            center_xy=crop_center,
            crop=crop,
            min_path_len=min_path_len,
            max_paths=max_paths,
            max_depth=max_depth,
            corridor_mode=True,
        )
        
        if f is None:
            continue
            
        # Validate it's a true corridor: only 2 directions, no T-junction/4-way
        if f.is_t_junction or f.is_four_way:
            continue
        if len(f.dirs) > 2:
            continue
        if f.junction_count > 0:
            continue
            
        feats.append(f)
        print(f"[INFO] Found valid corridor crop: road {cand['road_id']}, "
              f"center=({crop_center[0]:.1f}, {crop_center[1]:.1f}), "
              f"dirs={f.dirs}, jct_count={f.junction_count}")
    
    # Deduplicate and sort
    uniq = {}
    for f in feats:
        k = f.crop.to_str()
        if k not in uniq:
            uniq[k] = f
    
    out = list(uniq.values())
    out.sort(key=lambda x: -x.n_paths)  # Prefer crops with more paths
    return out
