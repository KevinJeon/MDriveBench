from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .geometry import ang_diff_deg
from .models import CropBox, LaneSegment, LegalPath


def build_connectivity(segments: List[LaneSegment],
                       connect_radius_m: float = 6.0,
                       connect_yaw_tol_deg: float = 60.0) -> List[List[int]]:
    """
    Build segment-to-segment connectivity graph.
    Two segments are connected if:
    1. End of segment A is close to start of segment B (within connect_radius_m)
    2. Heading at end of A aligns with heading at start of B (within connect_yaw_tol_deg)
    """
    n = len(segments)
    adj: List[List[int]] = [[] for _ in range(n)]
    if n == 0:
        return adj

    starts = np.vstack([seg.points[0] for seg in segments])
    tree = cKDTree(starts)

    for i, seg_a in enumerate(segments):
        end_pt = seg_a.points[-1]
        end_heading = seg_a.heading_at_end()

        candidates = tree.query_ball_point(end_pt, r=connect_radius_m)
        for j in candidates:
            if i == j:
                continue
            seg_b = segments[j]
            start_heading = seg_b.heading_at_start()
            if ang_diff_deg(end_heading, start_heading) <= connect_yaw_tol_deg:
                adj[i].append(j)

    return adj


def identify_boundary_segments(segments: List[LaneSegment],
                               crop: CropBox,
                               corridor_mode: bool = False,
                               boundary_margin: float = 3.0,
                               adj: Optional[List[List[int]]] = None) -> Tuple[List[int], List[int]]:
    """
    Identify segments that cross or are near the crop boundary.

    Entry segments: start outside (or near boundary), enter inside
    Exit segments: start inside, exit outside (or near boundary)
    
    Additionally, "terminal" segments (one end has no connectivity) are also
    treated as entry/exit. This handles cases where roads end within the crop
    region (e.g., dead ends or roads that were clipped).
    
    Args:
        segments: List of lane segments
        crop: Crop bounding box
        corridor_mode: If True, allow pass-through segments (start outside, end outside,
                       middle inside) to be classified as BOTH entry AND exit.
                       This is needed for corridor topologies where a road runs
                       entirely through the crop region.
        boundary_margin: Distance from boundary to consider as "at the boundary".
                        This handles segments that start/end slightly inside the crop
                        but should still be treated as entry/exit segments.
        adj: Adjacency list for segment connectivity. If provided, segments with
             terminal endpoints (no incoming/outgoing connections) are also
             classified as entry/exit segments.
    """
    entry_segments = []
    exit_segments = []
    
    def is_near_boundary(pt: np.ndarray) -> bool:
        """Check if a point is near (within margin of) any crop boundary."""
        x, y = float(pt[0]), float(pt[1])
        near_xmin = abs(x - crop.xmin) < boundary_margin
        near_xmax = abs(x - crop.xmax) < boundary_margin
        near_ymin = abs(y - crop.ymin) < boundary_margin
        near_ymax = abs(y - crop.ymax) < boundary_margin
        return near_xmin or near_xmax or near_ymin or near_ymax
    
    def is_deeply_inside(pt: np.ndarray) -> bool:
        """Check if point is well inside the crop (not near any boundary)."""
        return crop.contains(pt) and not is_near_boundary(pt)
    
    # Build reverse adjacency to detect incoming connections
    incoming = [[] for _ in segments] if adj else None
    if adj:
        for i, neighbors in enumerate(adj):
            for j in neighbors:
                incoming[j].append(i)
    
    def has_no_incoming(seg_idx: int) -> bool:
        """Check if segment has no incoming connections (terminal start)."""
        if incoming is None:
            return False
        return len(incoming[seg_idx]) == 0
    
    def has_no_outgoing(seg_idx: int) -> bool:
        """Check if segment has no outgoing connections (terminal end)."""
        if adj is None:
            return False
        return len(adj[seg_idx]) == 0

    for i, seg in enumerate(segments):
        start_pt = seg.points[0]
        end_pt = seg.points[-1]
        
        start_inside = crop.contains(start_pt)
        end_inside = crop.contains(end_pt)
        start_at_boundary = is_near_boundary(start_pt)
        end_at_boundary = is_near_boundary(end_pt)

        # Entry segment: starts outside/at-boundary, ends inside
        if not start_inside and end_inside:
            entry_segments.append(i)
        elif start_inside and start_at_boundary and is_deeply_inside(end_pt):
            # Start is technically inside but very near boundary, end is deep inside
            entry_segments.append(i)
        elif not start_inside and not end_inside:
            for pt in seg.points[1:-1]:
                if crop.contains(pt):
                    entry_segments.append(i)
                    break
        elif start_inside and end_inside and has_no_incoming(i):
            # Fully inside but no incoming connections - terminal start = entry
            entry_segments.append(i)

        # Exit segment: starts inside, ends outside/at-boundary
        if start_inside and not end_inside:
            exit_segments.append(i)
        elif is_deeply_inside(start_pt) and end_inside and end_at_boundary:
            # Start is deep inside, end is technically inside but very near boundary
            exit_segments.append(i)
        elif not start_inside and not end_inside:
            # Pass-through segment: starts outside, ends outside, but passes through crop
            if corridor_mode:
                # In corridor mode, pass-through segments can serve as BOTH entry and exit
                # This allows straight roads that run entirely through the crop to generate paths
                for pt in seg.points[1:-1]:
                    if crop.contains(pt):
                        if i not in exit_segments:
                            exit_segments.append(i)
                        break
            else:
                # Original behavior: only add to exit if not already an entry
                if i not in entry_segments:
                    for pt in seg.points[1:-1]:
                        if crop.contains(pt):
                            exit_segments.append(i)
                            break
        elif start_inside and end_inside and has_no_outgoing(i):
            # Fully inside but no outgoing connections - terminal end = exit
            if i not in exit_segments:
                exit_segments.append(i)

    return entry_segments, exit_segments


def generate_legal_paths(segments: List[LaneSegment],
                         adj: List[List[int]],
                         crop: CropBox,
                         min_path_length: float = 20.0,
                         max_paths: int = 100,
                         max_depth: int = 10,
                         allow_within_region_fallback: bool = True,
                         corridor_mode: bool = False) -> List[LegalPath]:
    """
    Generate legal paths that go from outside the crop area to outside.
    
    Args:
        segments: List of lane segments
        adj: Adjacency list for segment connectivity
        crop: Crop bounding box
        min_path_length: Minimum path length in meters
        max_paths: Maximum number of paths to generate
        max_depth: Maximum DFS depth
        allow_within_region_fallback: If True, fall back to any paths within region
        corridor_mode: If True, use corridor-specific boundary segment detection
                       that allows pass-through segments as both entry and exit.
                       Also allows single-segment paths for corridors.
    """
    legal_paths: List[LegalPath] = []

    entry_segments, exit_segments = identify_boundary_segments(segments, crop, corridor_mode=corridor_mode, adj=adj)
    print(f"[INFO] Found {len(entry_segments)} entry segments and {len(exit_segments)} exit segments")

    if len(entry_segments) == 0 or len(exit_segments) == 0:
        print("[WARNING] No entry or exit segments found. Paths must cross crop boundary.")
        if not allow_within_region_fallback:
            print("[INFO] Returning 0 legal paths for this crop (requires boundary-crossing paths).")
            return []
        print("[INFO] Falling back to any paths within the region...")
        entry_segments = list(range(len(segments)))
        exit_segments = list(range(len(segments)))

    exit_set = set(exit_segments)
    
    # In corridor mode, allow single-segment paths (a road that passes entirely through)
    min_path_segments = 1 if corridor_mode else 2

    def dfs(current_idx: int, path: List[int], total_length: float, depth: int):
        if len(legal_paths) >= max_paths:
            return

        if current_idx in exit_set and len(path) >= min_path_segments:
            if total_length >= min_path_length:
                path_segments = [segments[i] for i in path]
                legal_paths.append(LegalPath(path_segments, total_length))
                # In corridor mode with single-segment path, don't continue exploring
                if corridor_mode and len(path) == 1:
                    return

        if depth >= max_depth:
            return

        for next_idx in adj[current_idx]:
            if next_idx in path:
                continue
            next_seg = segments[next_idx]
            new_length = total_length + next_seg.length()
            dfs(next_idx, path + [next_idx], new_length, depth + 1)

    for entry_idx in entry_segments:
        if len(legal_paths) >= max_paths:
            break
        dfs(entry_idx, [entry_idx], segments[entry_idx].length(), 1)

    return legal_paths
