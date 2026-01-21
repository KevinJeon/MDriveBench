"""
Schema-based scenario generation.

This module produces structured ScenarioSpec JSON instead of free-form text.
"""

import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure scenario_generator/ is on sys.path for pipeline imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.step_01_crop.llm_utils import _extract_first_json_object

from .capabilities import (
    CATEGORY_DEFINITIONS,
    ActorKind,
    ConstraintType,
    EgoManeuver,
    GroupPattern,
    LateralPosition,
    MotionType,
    SpeedHint,
    TimingPhase,
    TopologyType,
    get_available_categories,
)
from .constraints import (
    EgoVehicleSpec,
    InterVehicleConstraint,
    NonEgoActorSpec,
    ScenarioSpec,
    validate_spec,
    spec_from_dict,
)
from .schema_utils import description_from_spec


_CONSTRAINT_VALUES = [c.value for c in ConstraintType]
_MANEUVER_VALUES = [m.value for m in EgoManeuver]
_TOPOLOGY_VALUES = [t.value for t in TopologyType if t != TopologyType.UNKNOWN]
_ACTOR_KIND_VALUES = [a.value for a in ActorKind]
_GROUP_PATTERN_VALUES = [g.value for g in GroupPattern]
_LATERAL_VALUES = [l.value for l in LateralPosition]
_TIMING_VALUES = [t.value for t in TimingPhase]
_MOTION_VALUES = [m.value for m in MotionType]
_SPEED_VALUES = [s.value for s in SpeedHint]


def _parse_variation_axes(cat_info: Any) -> Dict[str, List[str]]:
    axes: Dict[str, List[str]] = {}
    for axis in cat_info.variation_axes:
        if ": " in axis:
            axis_name, options_str = axis.split(": ", 1)
            options = [opt.strip() for opt in options_str.split(" vs ") if opt.strip()]
            if options:
                axes[axis_name] = options
    return axes


def select_variation_values(
    cat_info: Any,
    used_combinations: Optional[Set[str]] = None,
) -> Dict[str, str]:
    axes = _parse_variation_axes(cat_info)
    used_combinations = used_combinations or set()
    if not axes:
        return {}

    for _ in range(50):
        selections = {name: random.choice(options) for name, options in axes.items()}
        combo_key = "|".join(f"{k}={v}" for k, v in sorted(selections.items()))
        if combo_key not in used_combinations:
            return selections

    return {name: random.choice(options) for name, options in axes.items()}


def _enum_list(values: List[str]) -> str:
    return " | ".join(values)


def build_schema_system_prompt() -> str:
    return (
        "You generate structured JSON scenario specs for a driving pipeline.\n"
        "Output ONLY a JSON object that matches the requested schema.\n"
        "Do not include markdown, comments, or extra text."
    )


def build_schema_generation_prompt(
    category: str,
    difficulty: int,
    cat_info: Any,
    forced_variations: Optional[Dict[str, str]] = None,
    previous_validation_feedback: Optional[Dict[str, Any]] = None,
) -> str:
    required_flags = []
    if cat_info.needs_oncoming:
        required_flags.append("needs_oncoming=true")
    if cat_info.needs_multi_lane:
        required_flags.append("needs_multi_lane=true")
    if cat_info.needs_on_ramp:
        required_flags.append("needs_on_ramp=true")
    if cat_info.needs_merge:
        required_flags.append("needs_merge=true")
    required_flags_str = ", ".join(required_flags) if required_flags else "none"

    # Vehicle count scaling: match validator logic (no hidden hard-coding elsewhere)
    if cat_info.name == "Highway On-Ramp Merge":
        # Exact counts: d1=2, d2=3, d3=4, d4=5, d5=6
        if difficulty == 1:
            expected = 2
        elif difficulty == 2:
            expected = 3
        else:
            expected = min(6, difficulty + 1)
        min_vehicles = max_vehicles = expected
    elif cat_info.name == "Pedestrian Crosswalk":
        # D1-D3 follow vehicle scaling; D4-D5 focus on pedestrians (2-3 vehicles max)
        if difficulty == 1:
            min_vehicles = max_vehicles = 2
        elif difficulty == 2:
            min_vehicles = max_vehicles = 3
        elif difficulty == 3:
            min_vehicles = max_vehicles = 4
        else:
            min_vehicles = 2
            max_vehicles = 3
    elif cat_info.name == "Intersection Deadlock Resolution":
        # Each difficulty adds 1 vehicle: d1=2, d2=3, d3=4, d4=5, d5=6
        min_vehicles = difficulty + 1
        max_vehicles = min_vehicles
    elif cat_info.name == "Weaving Section":
        # Strict vehicle count for Weaving Section to enforce difficulty scaling
        if difficulty == 1:
            min_vehicles = 2
            max_vehicles = 2
        elif difficulty == 2:
            min_vehicles = 3
            max_vehicles = 3
        else:
            min_vehicles = 3
            max_vehicles = min(5, difficulty + 2)
    else:
        min_vehicles = 2
        if difficulty >= 3:
            min_vehicles = 3
        max_vehicles = min(6, difficulty + 2)
    
    # Categories with merge/on-ramp requirements need at least 1 constraint at all difficulties
    # Categories with multi-lane need at least 1 constraint at all difficulties
    # Other categories require constraints starting at difficulty 2
    # Intersection Deadlock Resolution: each vehicle beyond first needs 1 constraint
    if cat_info.name == "Intersection Deadlock Resolution":
        min_constraints = difficulty  # d1=1, d2=2, d3=3, d4=4, d5=5 (matches vehicle count - 1)
    elif cat_info.name == "Weaving Section":
        # Weaving Section constraint scaling
        if difficulty <= 1:
            min_constraints = 1
        elif difficulty == 2:
            min_constraints = 2  # d2 needs 2+ constraints for trio interactions
        else:
            min_constraints = difficulty  # d3=3, d4=4, etc
    else:
        min_constraints = 1 if (difficulty >= 2 or cat_info.needs_merge or cat_info.needs_on_ramp or cat_info.needs_multi_lane) else 0

    variation_lines = []
    if forced_variations:
        for k, v in forced_variations.items():
            variation_lines.append(f"- {k}: {v}")
    variation_block = "\n".join(variation_lines) if variation_lines else "- (no forced variation)"

    schema = (
        "{\n"
        f"  \"category\": \"{category}\",\n"
        f"  \"difficulty\": {difficulty},\n"
        f"  \"topology\": \"{_enum_list(_TOPOLOGY_VALUES)}\",\n"
        "  \"needs_oncoming\": true|false,\n"
        "  \"needs_multi_lane\": true|false,\n"
        "  \"needs_on_ramp\": true|false,\n"
        "  \"needs_merge\": true|false,\n"
        "  \"ego_vehicles\": [\n"
        "    {\n"
        "      \"vehicle_id\": \"Vehicle 1\",\n"
        f"      \"maneuver\": \"{_enum_list(_MANEUVER_VALUES)}\",\n"
        "      \"lane_change_phase\": \"before_intersection\" | \"after_intersection\" | \"unknown\",\n"
        "      \"entry_road\": \"main\" | \"side\" | \"unknown\",\n"
        "      \"exit_road\": \"main\" | \"side\" | \"unknown\"\n"
        "    }\n"
        "  ],\n"
        "  \"vehicle_constraints\": [\n"
        "    {\n"
        f"      \"type\": \"{_enum_list(_CONSTRAINT_VALUES)}\",\n"
        "      \"a\": \"Vehicle X\",\n"
        "      \"b\": \"Vehicle Y\"\n"
        "    }\n"
        "  ],\n"
        "  \"actors\": [\n"
        "    {\n"
        "      \"actor_id\": \"traffic cones\",\n"
        f"      \"kind\": \"{_enum_list(_ACTOR_KIND_VALUES)}\",\n"
        "      \"quantity\": 1,\n"
        f"      \"group_pattern\": \"{_enum_list(_GROUP_PATTERN_VALUES)}\",\n"
        f"      \"start_lateral\": \"{_enum_list(_LATERAL_VALUES)}\" | null,\n"
        f"      \"end_lateral\": \"{_enum_list(_LATERAL_VALUES)}\" | null,\n"
        "      \"affects_vehicle\": \"Vehicle 1\" | null,\n"
        f"      \"timing_phase\": \"{_enum_list(_TIMING_VALUES)}\",\n"
        f"      \"lateral_position\": \"{_enum_list(_LATERAL_VALUES)}\",\n"
        f"      \"motion\": \"{_enum_list(_MOTION_VALUES)}\",\n"
        f"      \"speed\": \"{_enum_list(_SPEED_VALUES)}\",\n"
        "      \"crossing_direction\": \"left\" | \"right\" | null,\n"
        "      \"direction_relative_to\": \"same\" | \"opposite\" | null\n"
        "    }\n"
        "  ]\n"
        "}\n"
    )

    # Build actor guidance based on category
    obstacle_focused_categories = {
        "Blocked Lane (Obstacle)",
        "Construction Zone", 
        "Parked Vehicle",
        "Lane Drop / Alternating Merge",
    }
    is_obstacle_focused = cat_info.name in obstacle_focused_categories
    
    if cat_info.name == "Lane Drop / Alternating Merge":
        actor_guidance = (
            "REQUIRED: Include 3-6 static_prop obstacles (traffic cones) that mark the lane drop/taper visually.\n"
            "  • timing_phase=after_merge (cones mark the merge point and immediate downstream area)\n"
            "  • lateral_position=half_left OR half_right (choose ONE side only for d1-d3; centered in dropping lane)\n"
            "  • group_pattern=along_lane (cones form a visual line along the closing lane)\n"
            "  • quantity=3-6 (enough cones to clearly indicate the lane closure)\n"
            "  • The cones are DECORATIVE markers that indicate the lane closure, not physical blockers.\n"
            "  • DIFFICULTY SCALING:\n"
            "    - d1-d3: ONE side only (left_edge OR right_edge), 3-5 cones\n"
            "    - d4-d5: Can use both sides if desired, 5-6 cones per side\n"
            "  • DO NOT place cones on BOTH sides at d1-d3 - creates excessive bottleneck.\n"
            "  • Example d2: quantity=4, kind=static_prop, lateral_position=left_edge, group_pattern=along_lane\n"
            "  • Vehicles can navigate around cones if needed - they're visual guidance, not barriers."
        )
    elif cat_info.name == "Occluded Hazard":
        actor_guidance = (
            "REQUIRED: Include 2 actors to create an occluded hazard scenario:\n"
            "  1. OCCLUDER (blocks visibility, NOT the lane):\n"
            "     • kind=parked_vehicle (prefer large truck/van/bus) or large static_prop (barrier/container)\n"
            "     • motion=static\n"
            "     • timing_phase=on_approach or in_intersection\n"
            "     • lateral_position=half_right or right_edge (OFF the ego's driving path)\n"
            "     • affects_vehicle=Vehicle 1 (or omit to let it occlude multiple vehicles)\n"
            "     • Use wording like 'parked box truck' or 'large barrier' so the occluder is big enough\n"
            "     • Example: parked truck at right_edge that blocks view of crosswalk\n"
            "  2. HIDDEN HAZARD (revealed late, starts moving when ego is close):\n"
            "     • kind=walker or cyclist (NOT a vehicle)\n"
            "     • motion=cross_perpendicular (crosses from behind occluder)\n"
            "     • timing_phase=on_approach (triggers when ego approaches)\n"
            "     • crossing_direction=left or right (emerges from behind occluder)\n"
            "     • lateral_position=right_edge or half_right so it emerges from behind the occluder\n"
            "     • affects_vehicle=same as occluder (or omit to target multiple vehicles)\n"
            "     • The walker/cyclist will be triggered to START MOTION when ego gets within ~8m\n"
            "     • This creates the 'late reveal' effect - hazard appears suddenly from behind occluder\n"
            "  \n"
            "  CRITICAL: The occluder must be positioned to HIDE the moving actor, not block the lane.\n"
            "  The moving actor emerges from behind the occluder when ego gets close.\n"
            "  Make the occlusion apply to as many approaching vehicles as possible (not just Vehicle 1)."
        )
    elif cat_info.name == "Weaving Section":
        # Weaving Section: NO props at any difficulty level - pure vehicle-to-vehicle weaving
        actor_guidance = (
            f"Do NOT include any non-ego actors (props, pedestrians, parked vehicles, etc) at difficulty {difficulty}.\n"
            "Weaving Section is PURE VEHICLE-TO-VEHICLE lane changing.\n"
            "All conflict comes from vehicles changing into each other's lanes.\n"
            "No obstacles, no props, no pedestrians - just cars weaving between lanes."
        )
    elif cat_info.name == "Pedestrian Crosswalk":
        # Pedestrian Crosswalk specific guidance
        # D1-D3: Focus on pedestrian interaction with multiple vehicles
        # D4-D5: Focus on pedestrian count scaling (2/3 pedestrians) with occlusions
        if difficulty <= 3:
            actor_guidance = (
                f"REQUIRED for Pedestrian Crosswalk d{difficulty}:\n"
                "  PEDESTRIAN REQUIREMENTS:\n"
                "  • Include 1 walker actor (single pedestrian for d1-d3)\n"
                "  • kind=walker (NO npc_vehicle, NO cyclist, NO moving obstacles)\n"
                "  • motion=cross_perpendicular (pedestrian crosses the street)\n"
                "  • timing_phase=on_approach (pedestrian triggers when vehicle(s) approach)\n"
                "  • start_lateral=right_edge OR left_edge (MUST start from side of road, never center)\n"
                "  • end_lateral=opposite side (left_edge if starting from right_edge, vice versa)\n"
                "  • crossing_direction=left OR right (direction pedestrian crosses)\n"
                "  • quantity=1\n"
                "  • TRIGGER DISTANCE: Typically 8m, but can use 10-12m for variety\n"
                "  • Pedestrian should cross in the path of MULTIPLE vehicles when possible\n"
                "  \n"
                "  NO OCCLUSION at d1-d3 (pedestrians are visible to approaching vehicles)\n"
                "  \n"
                "  FORBIDDEN:\n"
                "  • NO npc_vehicle actors (no moving NPC cars)\n"
                "  • NO cyclist actors\n"
                "  • NO static obstacles blocking vehicle paths\n"
                "  • NO pedestrians starting from center\n"
            )
        else:  # d4-d5
            pedestrian_count = difficulty - 2  # d4=2, d5=3 pedestrians
            actor_guidance = (
                f"REQUIRED for Pedestrian Crosswalk d{difficulty}:\n"
                f"  PEDESTRIAN REQUIREMENTS (primary focus for d4-d5):\n"
                f"  • MUST have exactly {pedestrian_count} walker actors (d4=2, d5=3 pedestrians)\n"
                "  • kind=walker (NO npc_vehicle, NO cyclist)\n"
                "  • motion=cross_perpendicular (pedestrians cross the street)\n"
                "  • timing_phase=on_approach (pedestrian triggers when vehicle(s) approach)\n"
                "  • start_lateral=right_edge OR left_edge (MUST start from side of road, never center)\n"
                "  • end_lateral=opposite side (crossing to the other edge)\n"
                "  • crossing_direction=left OR right\n"
                "  • quantity=1 per pedestrian actor\n"
                "  • MULTI-VEHICLE INTERACTION: Pedestrians should cross in the path of MULTIPLE vehicles\n"
                "  • TRIGGER DISTANCE: Vary between 8-12m for harder scenarios\n"
                "  \n"
                "  OCCLUSION REQUIRED for d4-d5 (pedestrians spawn BEHIND occlusions):\n"
                "  • kind=parked_vehicle (prefer LARGE vehicles: box truck, van, bus, delivery truck)\n"
                "  • motion=static (occluder is stationary)\n"
                "  • timing_phase=on_approach (present when vehicles approach)\n"
                "  • lateral_position=right_edge OR offroad_right (positioned to block sightline to pedestrian)\n"
                "  • The occluder MUST be positioned so pedestrian is HIDDEN from oncoming vehicles\n"
                "  • Pedestrian emerges FROM BEHIND the parked vehicle when triggered\n"
                "  • The parked vehicle must NOT be in the driving path of ego vehicles - it's roadside/offroad\n"
                "  • Use actor_id like 'parked_delivery_truck' or 'parked_box_van' to ensure proper size\n"
                "  \n"
                "  VEHICLE COUNT: 2-3 ego vehicles (enough for pedestrians to cross multiple paths)\n"
                "  \n"
                "  FORBIDDEN:\n"
                "  • NO npc_vehicle actors (no moving NPC cars)\n"
                "  • NO cyclist actors\n"
                "  • NO static obstacles blocking vehicle driving paths (occlusions block VIEW only, not lane)\n"
                "  • NO pedestrians starting from center\n"
            )
    elif cat_info.name == "Multi Left Turn Conflict":
        # Multi Left Turn Conflict specific guidance
        actor_guidance = (
            f"VEHICLE REQUIREMENTS for Multi Left Turn Conflict (difficulty {difficulty}):\n"
            "  • MUST have at least 2 ego vehicles with maneuver='left'\n"
            "  • Left-turning vehicles MUST be from DIFFERENT approaches (main road and side road)\n"
            "  • Use perpendicular_left_of OR perpendicular_right_of constraints (NOT opposite_approach_of)\n"
            "  • The left turn paths MUST intersect in the intersection\n"
            "  • Example: Vehicle 1 turns left from main road (east), Vehicle 2 turns left from side road (north)\n"
            "  • WRONG: Two vehicles from opposite directions both turning left (that's opposing left turns)\n"
            "  • RIGHT: Two vehicles from perpendicular directions both turning left (paths cross)\n"
            "  \n"
            "  DIFFICULTY SCALING:\n"
            "  • d1-d2: 2 left-turning vehicles from perpendicular approaches\n"
            "  • d3: 2-3 left-turning vehicles, can add follow_route_of chains\n"
            "  • d4-d5: 3+ left-turning vehicles or 2 turners with deeper queues\n"
            "  \n"
            "  OPTIONAL ACTORS:\n"
            "  • Can include walker crossing turn destination (timing_phase=after_turn or in_intersection)\n"
            "  • Can include straight-through vehicles on other approaches for added complexity\n"
            "  • NO moving NPC vehicles in turn paths\n"
            "  • NO obstacles blocking the intersection\n"
        )
    elif is_obstacle_focused and difficulty >= 1:
        actor_guidance = (
            f"REQUIRED: Include at least 1 non-ego actor (parked_vehicle, traffic_cone, etc) at difficulty {difficulty}. "
            f"This category centers on obstacles blocking ego vehicles."
        )
    elif cat_info.name == "Highway On-Ramp Merge" and difficulty < cat_info.non_ego_actors_min_difficulty:
        actor_guidance = (
            "Do NOT include non-ego actors at this difficulty (keep the merge interaction focused on vehicles)."
        )
    elif cat_info.name == "Multi Left Turn Conflict":
        # Multi Left Turn Conflict - specific guidance for intersecting left turns
        actor_guidance = (
            f"Optional: Include 0-2 non-ego actors at difficulty {difficulty} if desired:\n"
            "  • kind=walker (pedestrian crossing turn destination)\n"
            "  • kind=parked_vehicle (occlusion at corner)\n"
            "  • Actors should complicate turn path visibility or execution\n"
        )
    elif cat_info.name == "Major/Minor Unsignalized Entry":
        # Major/Minor Unsignalized Entry - T-junction yield scenario
        actor_guidance = (
            f"REQUIRED for Major/Minor Unsignalized Entry d{difficulty}:\n"
            "  VEHICLE ROAD ASSIGNMENT (critical for realistic T-junction scenario):\n"
            "  • AT LEAST ONE vehicle MUST have entry_road='main' (main road traffic)\n"
            "  • AT LEAST ONE vehicle MUST have entry_road='side' (side street vehicle)\n"
            "  • This creates the classic yield scenario: side street vehicle yields to main road traffic\n"
            "  • Main road vehicles typically go straight (entry_road='main', exit_road='main')\n"
            "  • Side street vehicles turn onto main road (entry_road='side', exit_road='main')\n"
            "  • DO NOT use all unknown entry_roads - this misses the main vs side interaction\n"
            "  \n"
            "  CONSTRAINTS:\n"
            "  • Use perpendicular constraints (perpendicular_left_of, perpendicular_right_of)\n"
            "  • Side street vehicle should be perpendicular to main road vehicle\n"
            "  • Can add follow_route_of for main road queues at higher difficulties\n"
            "  \n"
            "  OPTIONAL ACTORS:\n"
            "  • Parked vehicles near intersection for occlusion (timing_phase=on_approach)\n"
            "  • Traffic cones or barriers (kind=static_prop)\n"
            "  • Pedestrians crossing after turn completion (timing_phase=after_exit)\n"
        )
    elif cat_info.uses_non_ego_actors and difficulty >= cat_info.non_ego_actors_min_difficulty:
        actor_guidance = (
            "Include 1-2 non-ego actors (optional) to increase scenario complexity if desired."
        )
    else:
        actor_guidance = "Actors may be empty if not needed."
    difficulty_lines = _build_difficulty_requirement_lines(difficulty, cat_info)
    difficulty_block = "\n".join(f"- {line}" for line in difficulty_lines) if difficulty_lines else "- (none)"

    # Build constraint semantics guidance
    constraint_semantics = """
CONSTRAINT TYPE DEFINITIONS:

PASSIVE CONSTRAINTS (No spatial conflict):
  same_approach_as: Both vehicles approach from same direction and area.
    • Creates only directional alignment, NO conflict.
    • Acceptable as secondary constraint only.
    
  follow_route_of: Vehicle B follows along the predetermined route of Vehicle A.
    • Creates only temporal dependency (B reacts to A), NO spatial conflict.
    • Acceptable as secondary constraint only.

ACTIVE CONSTRAINTS (Creates spatial conflict):
  opposite_approach_of: Vehicles approach each other head-on from opposite directions.
    • GEOMETRY: Paths cross at angle > 120° (ideally 150-180°)
    • CONFLICT: High severity - must decide right-of-way or collision risk
    • INTERACTION: Vehicles MUST reach conflict zone within 5 seconds of each other
    
  perpendicular_left_of: Vehicle approaches/enters from the left side.
    • GEOMETRY: Paths intersect within 5 meters at angle 70-110°
    • CONFLICT: Medium-high severity - merge or intersection priority
    • INTERACTION: Paths must cross at similar times (within 5s arrival)
    
  perpendicular_right_of: Vehicle approaches/enters from the right side.
    • GEOMETRY: Paths intersect within 5 meters at angle 70-110°
    • CONFLICT: Medium-high severity - symmetric to left approach
    • INTERACTION: Paths must cross at similar times (within 5s arrival)
    
  left_lane_of: Vehicle travels in left lane relative to another vehicle.
    • GEOMETRY: Paths parallel (angle difference < 30°), 3-10m apart, same direction
    • CONFLICT: Medium severity - lane change, overtaking decisions
    • INTERACTION: Vehicles must be spatially co-located (within 20m) for meaningful interaction
    • WARNING: Two vehicles in adjacent lanes just driving straight = NO INTERACTION
    
  right_lane_of: Vehicle travels in right lane relative to another vehicle.
    • GEOMETRY: Paths parallel (angle difference < 30°), 3-10m apart, same direction
    • CONFLICT: Medium severity - lane change, merging decisions
    • INTERACTION: Vehicles must be spatially co-located (within 20m) for meaningful interaction
    • WARNING: Two vehicles in adjacent lanes just driving straight = NO INTERACTION
    
  same_lane_as: Vehicle travels in the EXACT SAME physical lane as another vehicle.
    • GEOMETRY: Same lane_id, same approach direction, vehicles queued in same lane
    • CONFLICT: High severity - following, rear-end risk, queuing
    • INTERACTION: Creates natural following/queuing behavior
    • USE FOR: Highway queues, platoons, following chains where vehicles MUST share the same lane
    • DIFFERENCE FROM follow_route_of: follow_route_of allows parallel lanes going same direction;
      same_lane_as enforces exact lane match (critical for multi-lane highways)
    
  merges_into_lane_of: Vehicle merges from adjacent lane or ramp into another's lane.
    • GEOMETRY: Paths converge, one joins the trajectory of the other
    • CONFLICT: Medium severity - merge acceptance/rejection, speed coordination
    • INTERACTION: Merger must reach merge point while target vehicle is nearby (< 30m)

KEY RULE: Passive constraints (same_approach_as, follow_route_of) CANNOT be the only constraint at difficulty >= 2.
Each scenario needs at least one ACTIVE constraint (opposite/perpendicular/lane/merge) to create real conflict.

TEMPORAL-SPATIAL INTERACTION REQUIREMENT:
  All multi-vehicle scenarios MUST ensure vehicles can actually interact:
  • Opposite/perpendicular: Vehicles reach conflict zone within 5 seconds
  • Adjacent lanes (left/right_lane_of): Vehicles must overlap spatially by at least 20m
  • Merge scenarios: Merging vehicle reaches merge point while target is within 30m
  
  BAD EXAMPLE (NO INTERACTION):
    Vehicle 1 and Vehicle 2 both go straight in adjacent lanes from same spawn area.
    → They just drive parallel forever with no interaction potential.
  
  GOOD EXAMPLE (HAS INTERACTION):
    Vehicle 1 approaches from north, Vehicle 2 from east (perpendicular).
    → They reach intersection at similar times, creating conflict.
"""

    difficulty_requirements = f"""
DIFFICULTY REQUIREMENTS:

Difficulty 1 (Simplest):
  • 1 vehicle alone (no constraint needed), OR
  • 2 vehicles with 0-1 constraint
  • Constraints can be passive (follow_route_of, same_approach_as)

Difficulty 2:
  • MUST have: 2 vehicles
  • MUST have: 1-2 constraints
  • MUST include: At least 1 ACTIVE constraint (perpendicular, opposite, or lane relationship)
  • Can include: follow_route_of as secondary passive constraint
  • MUST ensure: Vehicles reach conflict point at similar times (within 5s)
  • Examples: V2 perpendicular_right_of V1, V2 left_lane_of V1 (with spatial overlap)

Difficulty 3:
  • MUST have: 3 vehicles
  • MUST have: 2-3 constraints total
  • MUST include: At least 2 ACTIVE constraints (different types)
  • Each added vehicle (V2, V3) should introduce a DIFFERENT conflict type
  • DO NOT: Add Vehicle 3 with only passive constraints (e.g., just follow_route_of)
  • MUST ensure: All vehicles have temporal-spatial interaction potential
  • Correct: V2 perpendicular_right_of V1, V3 opposite_approach_of V1 (both reach intersection)
  • Wrong: V2 perpendicular_right_of V1, V3 follow_route_of V1 (V3 adds no conflict)
  • Wrong: V1, V2, V3 all straight in adjacent lanes from same spawn (no interaction)

Difficulty 4-5:
  • MUST have: 4+ vehicles, 4+ constraints, 3+ ACTIVE constraints
  • MUST create: Multiple simultaneous interaction opportunities
  • AVOID: Adding vehicles that never interact with others
"""

    reasoning_template = f"""
REASONING FORMAT:
Before generating the scenario JSON, explain your reasoning:

{{
  "reasoning": {{
    "difficulty_requirement": "Difficulty {difficulty} requires...",
    "strategy": "I will create a scenario with...",
    "vehicle_1_description": "Main vehicle traveling...",
    "vehicle_2_plan": {{
      "description": "Role and behavior",
      "constraint": "Vehicle 2 [type] Vehicle 1",
      "conflict_type": "Type of conflict this creates"
    }},
    "constraint_summary": "X active constraints, Y passive. Total: Z"
  }},
  "scenario_spec": {{
    ... (your actual scenario JSON here)
  }}
}}
"""

    anti_patterns = """
ANTI-PATTERNS TO AVOID:

❌ Passive-only at D2+:
   "Vehicle 2 follow_route_of Vehicle 1" - No real conflict

❌ D3 with V3 adding nothing:
   "V2 perpendicular_right_of V1, V3 follow_route_of V1" - V3 doesn't help

❌ Adjacent lanes with no interaction:
   "V1 straight main lane, V2 straight left lane, both spawn at same location"
   → They just drive parallel forever, never interact spatially

❌ Impossible combinations:
   "V2 perpendicular_right_of V1 AND opposite_approach_of V1" - Can't do both

❌ Meaningless lane constraints:
   "V1 and V2 in adjacent lanes, no lane changes, no merge, no spatial overlap"
   → Constraint exists but creates zero coordination challenge

✓ CORRECT patterns:
   D2: One active constraint WITH temporal-spatial interaction
   D3: Two different active constraints from different vehicles, all interact
   
✓ GOOD adjacent lane scenario:
   "V1 traveling straight, V2 in left lane approaching from behind, lane changes 
   into V1's lane after 30m" → Creates merge/overtaking interaction

✓ GOOD perpendicular scenario:
   "V1 approaches from east, V2 from north, paths cross at intersection"
   → Both reach intersection within 3-4 seconds, creating conflict
"""

    # Format validation feedback if provided
    validation_feedback_section = ""
    if previous_validation_feedback:
        score = previous_validation_feedback.get("score", 0.0)
        issues = previous_validation_feedback.get("issues", [])
        
        # Extract banned constraints from INEFFECTIVE analysis
        banned_constraints = []
        for issue in issues:
            suggestion = issue.get("suggestion", "")
            # Parse constraint analysis to find ineffective constraints
            if "CONSTRAINT ANALYSIS:" in suggestion:
                # Extract constraint patterns like "follow_route_of(Vehicle 4 -> Vehicle 1)"
                import re
                ineffective_matches = re.findall(
                    r'(\w+)\(Vehicle\s*(\d+)\s*->\s*Vehicle\s*(\d+)\)[^[]*\[INEFFECTIVE\]',
                    suggestion
                )
                for constraint_type, va, vb in ineffective_matches:
                    banned_constraints.append(f"{constraint_type}(Vehicle {va} -> Vehicle {vb})")
            
            # Check for duplicate detection - ban the same pattern
            if "DUPLICATE_OUTPUT" in issue.get("category", "") or "OUTPUT DUPLICATE" in issue.get("message", ""):
                # The whole approach is wrong - need fundamentally different constraints
                banned_constraints.append("follow_route_of (any pair) - does not create unique paths")
        
        if issues:
            error_list = "\n".join(
                f"  - [{issue.get('severity', 'error').upper()}] {issue.get('message', 'Unknown issue')}\n"
                f"    Expected: {issue.get('expected', 'N/A')}\n"
                f"    Actual: {issue.get('actual', 'N/A')}\n"
                f"    Suggestion: {issue.get('suggestion', 'N/A')}"
                for issue in issues[:10]  # Limit to top 10 issues
            )
            
            # Build banned constraints section if any were identified
            banned_section = ""
            if banned_constraints:
                banned_list = "\n".join(f"  ❌ {bc}" for bc in banned_constraints)
                banned_section = (
                    "\n🚫 BANNED CONSTRAINTS (DO NOT USE THESE):\n"
                    f"{banned_list}\n"
                    "These constraints FAILED to create interactions. You MUST use DIFFERENT constraint types.\n"
                    "ALTERNATIVES that DO work:\n"
                    "  ✓ same_lane_as - queues vehicles in the SAME physical lane (for highway following)\n"
                    "  ✓ merges_into_lane_of - creates merge conflict\n"
                    "  ✓ perpendicular_left_of / perpendicular_right_of - cross-traffic conflict\n"
                    "  ✓ opposite_approach_of - head-on approach\n"
                    "  ⚠️ left_lane_of/right_lane_of - CAUTION: on highways, may place in non-interacting lanes\n\n"
                )
            
            validation_feedback_section = (
                "\n\n⚠️  PREVIOUS ATTEMPT FAILED ⚠️\n"
                f"Validation Score: {score:.2f} (needs >= 0.6)\n"
                f"Issues found:\n{error_list}\n\n"
                f"{banned_section}"
                "FIX THESE ISSUES: Generate a COMPLETELY DIFFERENT scenario.\n"
                "Do NOT repeat ANY of the same constraint patterns.\n"
                "The previous constraints DID NOT WORK - you must try something fundamentally different.\n\n"
            )
    
    return (
        f"You are generating a JSON scenario spec that the pipeline can execute. \n"
        f"Category: {category}\n"
        f"Difficulty: {difficulty}\n"
        f"Required topology: {cat_info.required_topology.value}\n"
        f"Required flags: {required_flags_str}\n"
        f"Vehicle count: use {min_vehicles}-{max_vehicles} ego vehicles.\n"
        f"Constraints: include at least {min_constraints} inter-vehicle constraints.\n"
        f"{validation_feedback_section}"
        f"{constraint_semantics}"
        f"{difficulty_requirements}"
        f"{reasoning_template}"
        f"{anti_patterns}"
        f"Conflict requirements:\n{difficulty_block}\n"
        f"Variation targets:\n{variation_block}\n"
        f"Actor guidance: {actor_guidance}\n"
        "\n"
        "CRITICAL RULES:\n"
        "- Vehicle IDs must be 'Vehicle 1', 'Vehicle 2', ... with no gaps.\n"
        "- All constraints must reference existing vehicles.\n"
        "- If topology is corridor, prefer straight/lane_change maneuvers.\n"
        "- If needs_oncoming=true, include an opposite_approach_of constraint.\n"
        "- If needs_multi_lane=true, include at least one left/right lane or merge constraint.\n"
        "- TURN MANEUVERS: If maneuver='left' or 'right', entry_road and exit_road MUST be different.\n"
        "  * Correct: maneuver='left', entry_road='main', exit_road='side' (turns from main to side road)\n"
        "  * Correct: maneuver='right', entry_road='side', exit_road='main' (turns from side to main road)\n"
        "  * WRONG: maneuver='left', entry_road='main', exit_road='main' (creates impossible U-turn)\n"
        "  * For straight/lane_change maneuvers, entry_road can equal exit_road.\n"
        + ("- MULTI LEFT TURN CONFLICT: At least 2 vehicles MUST have maneuver='left'. Use perpendicular constraints (NOT opposite_approach_of). Vehicles approach from different directions (e.g., main+side roads) and their left turn paths MUST intersect.\n" if cat_info.name == 'Multi Left Turn Conflict' else "")
        + ("- MAJOR/MINOR UNSIGNALIZED ENTRY: MANDATORY road assignment for T-junction realism:\n" if cat_info.name == 'Major/Minor Unsignalized Entry' else "")
        + ("  * AT LEAST ONE vehicle MUST have entry_road='main' (main road traffic going straight)\n" if cat_info.name == 'Major/Minor Unsignalized Entry' else "")
        + ("  * AT LEAST ONE vehicle MUST have entry_road='side' (side street vehicle turning onto main)\n" if cat_info.name == 'Major/Minor Unsignalized Entry' else "")
        + ("  * DO NOT use entry_road='unknown' for all vehicles - this breaks the major/minor interaction\n" if cat_info.name == 'Major/Minor Unsignalized Entry' else "")
        + ("  * Example: Vehicle 1: entry_road='main', exit_road='main', maneuver='straight'\n" if cat_info.name == 'Major/Minor Unsignalized Entry' else "")
        + ("  * Example: Vehicle 2: entry_road='side', exit_road='main', maneuver='right'\n" if cat_info.name == 'Major/Minor Unsignalized Entry' else "")
        + (f"- INTERSECTION DEADLOCK: Use EXACTLY {min_vehicles} vehicles and {min_constraints} constraints.\n" if cat_info.name == 'Intersection Deadlock Resolution' else "")
        + ("  Each vehicle beyond Vehicle 1 MUST have at least one perpendicular/opposite constraint to another vehicle.\n" if cat_info.name == 'Intersection Deadlock Resolution' else "")
        + "- If needs_on_ramp=true:\n"
        "  * At least ONE vehicle MUST have entry_road='main' (mainline highway vehicle)\n"
        "  * At least ONE vehicle MUST have entry_road='side' (on-ramp vehicle)\n"
        + (f"  * CRITICAL FOR DIFFICULTY {difficulty}: You MUST include AT LEAST TWO (2) vehicles with entry_road='side' (multiple on-ramp vehicles). This is REQUIRED for d3+ complexity. Do NOT generate only one side vehicle.\n" if difficulty >= 3 and cat_info.needs_on_ramp else "")
        + "  * On-ramp vehicle MUST have exit_road='main' (merging onto highway)\n"
        + "  * MUST include merges_into_lane_of constraint from side vehicle to main vehicle\n"
        + (f"  * DIFFICULTY {difficulty} MAINLINE VEHICLE QUEUING:\n" if difficulty >= 3 and cat_info.needs_on_ramp else "")
        + (f"    - For mainline vehicles (entry_road='main'), use same_lane_as to queue them in the SAME lane as Vehicle 1\n" if difficulty >= 3 and cat_info.needs_on_ramp else "")
        + (f"    - DO NOT use follow_route_of for mainline vehicles - it places them in PARALLEL non-interacting lanes!\n" if difficulty >= 3 and cat_info.needs_on_ramp else "")
        + (f"    - Example: Vehicle 4 (entry_road='main') same_lane_as Vehicle 1 → queues behind V1 in merge target lane\n" if difficulty >= 3 and cat_info.needs_on_ramp else "")
        + (f"  * DIFFICULTY {difficulty} VEHICLE INTERACTION REQUIREMENTS:\n" if difficulty >= 4 and cat_info.needs_on_ramp else "")
        + (f"    - Vehicles 4+ MUST interact with the ON-RAMP MERGE scenario (not just be random highway traffic)\n" if difficulty >= 4 and cat_info.needs_on_ramp else "")
        + (f"    - Options: (a) Add more on-ramp vehicles (entry_road='side'), (b) Highway queues using same_lane_as (NOT follow_route_of), (c) Vehicles in merge target lane that interact with merging traffic\n" if difficulty >= 4 and cat_info.needs_on_ramp else "")
        + (f"    - WRONG: Using follow_route_of for mainline highway vehicles - this places them in PARALLEL lanes that don't interact!\n" if difficulty >= 4 and cat_info.needs_on_ramp else "")
        + (f"    - WRONG: Using left_lane_of to put vehicles in lanes that don't converge with the merge zone\n" if difficulty >= 4 and cat_info.needs_on_ramp else "")
        + (f"    - CORRECT: Vehicle 4 entry_road='side' merges_into_lane_of Vehicle 1 (adds second merge conflict)\n" if difficulty >= 4 and cat_info.needs_on_ramp else "")
        + (f"    - CORRECT: Vehicle 5 same_lane_as Vehicle 1 (queues behind V1 in the merge target lane)\n" if difficulty >= 4 and cat_info.needs_on_ramp else "")
        + "  * Example: Vehicle 1: entry_road='main', Vehicle 2: entry_road='side', exit_road='main', maneuver='lane_change'\n"
        + "  * Constraint: Vehicle 2 merges_into_lane_of Vehicle 1\n"
        + (f"  * MANDATORY D{difficulty} example: V1 entry_road='main', V2 entry_road='side', V3 entry_road='side' (2 merging), V4 entry_road='main' - BOTH V2 and V3 MUST have entry_road='side'\n" if difficulty >= 3 and cat_info.needs_on_ramp else "")
        + "  * Description: 'Vehicle 2 enters from on-ramp and merges into Vehicle 1's lane' NOT 'changes lanes'\n"
        "- For on-ramp scenarios, set entry_road='main' for mainline vehicles and entry_road='side' for on-ramp vehicles.\n"
        "  On-ramp vehicles should have exit_road='main'.\n"
        "- Every ego vehicle beyond Vehicle 1 MUST have an explicit relation:\n"
        "  same_approach_as OR opposite_approach_of OR perpendicular_left_of OR perpendicular_right_of OR\n"
        "  left_lane_of OR right_lane_of OR merges_into_lane_of OR follow_route_of.\n"
        "- When a vehicle is on the SAME approach as another, include a lane relation:\n"
        "  left_lane_of OR right_lane_of OR merges_into_lane_of OR same_lane_as.\n"
        "- IMPORTANT: On multi-lane highways, use same_lane_as (NOT follow_route_of) to queue vehicles in the same lane.\n"
        "  follow_route_of only matches route direction and may place vehicles in PARALLEL non-interacting lanes.\n"
        "- Use null for optional fields you cannot specify.\n"
        "\n"
        f"For Difficulty {difficulty}, remember to include:\n"
        f"{'- At least 1 ACTIVE constraint (not just passive)' if difficulty >= 2 else ''}\n"
        f"{'- At least 2 ACTIVE constraints from different vehicles' if difficulty >= 3 else ''}\n"
        "\n"
        "Return JSON with reasoning and scenario_spec:\n"
        "{\n"
        "  \"reasoning\": {\n"
        "    \"difficulty_requirement\": \"...\",\n"
        "    \"strategy\": \"...\",\n"
        "    \"vehicle_2_plan\": {...},\n"
        "    \"constraint_summary\": \"X active, Y passive, Total: Z\"\n"
        "  },\n"
        "  \"scenario_spec\": {\n"
        f"{schema}"
        "  }\n"
        "}\n"
    )


def build_schema_repair_prompt(
    bad_payload: str,
    errors: List[str],
    category: str,
    difficulty: int,
    cat_info: Any,
) -> str:
    valid_constraint_types = _enum_list(_CONSTRAINT_VALUES)
    error_lines = "\n".join(f"- {e}" for e in errors) if errors else "- (unknown errors)"
    difficulty_lines = _build_difficulty_requirement_lines(difficulty, cat_info)
    difficulty_block = "\n".join(f"- {line}" for line in difficulty_lines) if difficulty_lines else "- (none)"
    
    # Add critical reminders based on category requirements
    critical_reminders = []
    if cat_info.needs_on_ramp:
        critical_reminders.append("- MUST have entry_road='main' for at least one vehicle (mainline)")
        if difficulty >= 3:
            critical_reminders.append(f"- CRITICAL D{difficulty}: MUST have entry_road='side' for AT LEAST TWO vehicles (multiple on-ramp mergers required)")
        else:
            critical_reminders.append("- MUST have entry_road='side' for at least one vehicle (on-ramp)")
        critical_reminders.append("- MUST have merges_into_lane_of from side vehicle to main vehicle")
        critical_reminders.append("- To queue mainline vehicles behind each other, use same_lane_as (keeps them in the SAME physical lane)")
    if difficulty >= 2 and cat_info.required_topology != TopologyType.CORRIDOR:
        critical_reminders.append("- MUST have at least 2 inter-vehicle constraints (e.g., merges_into_lane_of + same_approach_as)")
    if cat_info.needs_multi_lane:
        critical_reminders.append("- MUST include a lane relation (left_lane_of/right_lane_of/merges_into_lane_of)")
    
    reminders_block = "\n".join(critical_reminders) if critical_reminders else ""
    
    return (
        "Fix the JSON scenario spec to satisfy the errors below. Edit the prior JSON directly; change only what is needed.\n"
        f"Category: {category}\n"
        f"Difficulty: {difficulty}\n"
        f"Required topology: {cat_info.required_topology.value}\n"
        f"Conflict requirements:\n{difficulty_block}\n"
        "\n"
        "CRITICAL REMINDERS:\n"
        f"{reminders_block}\n"
        "\n"
        "ERRORS TO FIX:\n"
        f"{error_lines}\n"
        "\n"
        "CONSTRAINT RULES:\n"
        f"- Allowed constraint types: {valid_constraint_types}\n"
        "- DO NOT invent new constraint types or keys. Only use the allowed values above.\n"
        "- At difficulty >=2 you MUST include at least one ACTIVE constraint (perpendicular_left_of/right_of/opposite_approach_of/left_lane_of/right_lane_of/merges_into_lane_of/same_lane_as).\n"
        "\n"
        "YOUR PREVIOUS ATTEMPT (failed validation) — edit THIS JSON:\n"
        f"{bad_payload}\n"
        "\n"
        "Return ONLY a corrected JSON object with:\n"
        "1. All required fields properly set (especially entry_road for on-ramp scenarios)\n"
        "2. Enough constraints to meet difficulty requirements\n"
        "3. Valid vehicle IDs and constraint references\n"
        "4. Reuse vehicle IDs unless an error explicitly requires adding/removing vehicles.\n"
    )


def _build_difficulty_requirement_lines(difficulty: int, cat_info: Any) -> List[str]:
    lines: List[str] = []
    if cat_info.required_topology in {TopologyType.INTERSECTION, TopologyType.T_JUNCTION}:
        lines.append(
            "At least one crossing relation: opposite_approach_of or perpendicular_left_of/perpendicular_right_of."
        )
    if cat_info.needs_oncoming:
        lines.append("Must include opposite_approach_of (oncoming conflict).")
    if cat_info.needs_multi_lane:
        lines.append("Must include a lane relation: left_lane_of/right_lane_of/merges_into_lane_of.")
    if cat_info.needs_merge or cat_info.needs_on_ramp:
        lines.append("Must include merges_into_lane_of (merge conflict).")
    if cat_info.name == "Lane Drop / Alternating Merge":
        lines.append(
            "Include 3-6 static_prop obstacles (cones) positioned to MARK the lane taper visually: "
            "timing_phase=after_merge, lateral_position=half_left or half_right (centered in dropping lane), group_pattern=along_lane. "
            "For d1-d3: use ONE side only. For d4+: can use both sides or parked_vehicle. "
            "These obstacles are decorative markers indicating which lane is closing, not physical barriers."
        )
    if cat_info.name == "Pedestrian Crosswalk":
        lines.append("Pedestrians MUST start from side of road (right_edge or left_edge), never center.")
        lines.append("NO moving NPC vehicles, cyclists, or obstacles blocking vehicle paths allowed.")
        if difficulty <= 3:
            lines.append(f"Include 1 walker actor crossing the street (no occlusion at d{difficulty}).")
        else:  # d4-d5
            pedestrian_count = difficulty - 2  # d4=2, d5=3
            lines.append(f"MUST have exactly {pedestrian_count} walker actors (d4=2, d5=3 pedestrians).")
            lines.append("Include 2-3 ego vehicles (enough for pedestrians to cross multiple paths).")
            lines.append(
                "Pedestrians spawn BEHIND parked_vehicle occlusions (large trucks, vans, buses positioned roadside/offroad)."
            )
            lines.append("Occlusions block VIEW only - NOT in driving path of ego vehicles.")
        lines.append("Trigger distance can vary between 8-12m (not always 8m).")
    if cat_info.name == "Multi Left Turn Conflict":
        lines.append("At least 2 vehicles MUST have maneuver='left' (all turning left from different approaches).")
        lines.append("Use perpendicular_left_of or perpendicular_right_of constraints (NOT opposite_approach_of).")
        lines.append("Vehicles approach from different directions (main road + side road) and turn left with intersecting paths.")
        lines.append("DO NOT create opposing left turns - vehicles should be perpendicular to each other, not opposite.")
        lines.append("Left turn paths MUST cross within the intersection (e.g., Vehicle 1 from north turning left to west, Vehicle 2 from east turning left to north).")
    if difficulty >= 2:
        lines.append("At least 2 inter-vehicle constraints total.")
    if difficulty >= 3:
        lines.append("At least 3 ego vehicles.")
    if difficulty >= 4:
        lines.append("At least 3 constraints total.")
        if cat_info.required_topology in {TopologyType.INTERSECTION, TopologyType.T_JUNCTION}:
            lines.append("At least 2 crossing relations.")
        if cat_info.needs_multi_lane:
            lines.append("At least 2 lane/merge relations.")
    if difficulty >= 5 and cat_info.uses_non_ego_actors and difficulty >= cat_info.non_ego_actors_min_difficulty:
        # Skip the general requirement if this is Pedestrian Crosswalk (already has specific rules)
        if cat_info.name != "Pedestrian Crosswalk":
            lines.append("Include at least 1 non-ego actor.")
    lines.append("Do NOT place all vehicles in the same lane with identical maneuvers.")
    return lines


def _actor_conflict_info(spec: ScenarioSpec) -> Tuple[bool, Set[str], bool]:
    has_conflict = False
    actor_targets: Set[str] = set()
    affects_all = False

    for actor in spec.actors:
        if actor.affects_vehicle:
            actor_targets.add(actor.affects_vehicle)

        if actor.motion == MotionType.CROSS_PERPENDICULAR and actor.kind in {ActorKind.WALKER, ActorKind.CYCLIST}:
            has_conflict = True
            affects_all = True
            continue

        if actor.motion == MotionType.STATIC and actor.kind in {ActorKind.PARKED_VEHICLE, ActorKind.STATIC_PROP}:
            if actor.affects_vehicle or actor.lateral_position != LateralPosition.CENTER:
                has_conflict = True

    return has_conflict, actor_targets, affects_all


def _conflict_findings(spec: ScenarioSpec, cat_info: Any) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    vehicles = [v.vehicle_id for v in spec.ego_vehicles]
    num_vehicles = len(vehicles)
    difficulty = spec.difficulty

    actor_conflict, actor_targets, actor_affects_all = _actor_conflict_info(spec)

    # Phase 1: Define passive and active constraint types
    passive_constraint_types = {
        ConstraintType.FOLLOW_ROUTE_OF,
        ConstraintType.SAME_APPROACH_AS,
    }
    active_constraint_types = {
        ConstraintType.OPPOSITE_APPROACH_OF,
        ConstraintType.PERPENDICULAR_LEFT_OF,
        ConstraintType.PERPENDICULAR_RIGHT_OF,
        ConstraintType.LEFT_LANE_OF,
        ConstraintType.RIGHT_LANE_OF,
        ConstraintType.MERGES_INTO_LANE_OF,
    }

    crossing_types = {
        ConstraintType.OPPOSITE_APPROACH_OF,
        ConstraintType.PERPENDICULAR_LEFT_OF,
        ConstraintType.PERPENDICULAR_RIGHT_OF,
    }
    lane_types = {
        ConstraintType.LEFT_LANE_OF,
        ConstraintType.RIGHT_LANE_OF,
        ConstraintType.MERGES_INTO_LANE_OF,
    }
    merge_types = {ConstraintType.MERGES_INTO_LANE_OF}

    # Phase 1: Count active vs passive constraints
    active_constraints = [c for c in spec.vehicle_constraints if c.constraint_type in active_constraint_types]
    passive_constraints = [c for c in spec.vehicle_constraints if c.constraint_type in passive_constraint_types]
    num_active = len(active_constraints)
    num_passive = len(passive_constraints)

    crossing_count = sum(1 for c in spec.vehicle_constraints if c.constraint_type in crossing_types)
    lane_count = sum(1 for c in spec.vehicle_constraints if c.constraint_type in lane_types)
    merge_count = sum(1 for c in spec.vehicle_constraints if c.constraint_type in merge_types)
    constraint_count = len(spec.vehicle_constraints)
    has_vehicle_conflict = num_active > 0
    has_any_conflict = has_vehicle_conflict or actor_conflict

    if num_vehicles >= 2 and not has_any_conflict:
        errors.append(
            "No conflict source found: add an active inter-vehicle constraint or an actor that creates interaction."
        )

    # Phase 1: Validate passive-only scenarios at D2+ (always error to force active conflict)
    if difficulty >= 2 and num_active == 0 and constraint_count > 0:
        msg = (
            f"Difficulty {difficulty} requires at least 1 ACTIVE constraint. "
            f"Found {num_passive} passive constraint(s) only (follow_route_of, same_approach_as). "
            "Add perpendicular, opposite, or lane relationships."
        )
        errors.append(msg)
    
    # Phase 1: Validate D3+ has multiple active constraints
    if difficulty >= 3 and num_active < 2:
        msg = (
            f"Difficulty {difficulty} requires at least 2 ACTIVE constraints. "
            f"Found {num_active} active constraint(s). Add different conflict types for each vehicle."
        )
        if actor_conflict:
            warnings.append(msg)
        else:
            errors.append(msg)

    if cat_info.required_topology in {TopologyType.INTERSECTION, TopologyType.T_JUNCTION}:
        if crossing_count == 0:
            errors.append("Needs at least one crossing relation (opposite/perpendicular).")

    if cat_info.needs_oncoming:
        if not any(c.constraint_type == ConstraintType.OPPOSITE_APPROACH_OF for c in spec.vehicle_constraints):
            errors.append("Category requires oncoming traffic: include opposite_approach_of.")

    if cat_info.needs_multi_lane and lane_count == 0:
        errors.append("Category requires multi-lane conflict: include left/right lane or merge relation.")

    if (cat_info.needs_merge or cat_info.needs_on_ramp) and merge_count == 0:
        errors.append("Category requires merge conflict: include merges_into_lane_of.")

    if cat_info.needs_on_ramp:
        entry_by_vehicle = {v.vehicle_id: v.entry_road for v in spec.ego_vehicles}
        side_vehicles = [v for v, entry in entry_by_vehicle.items() if entry == "side"]
        main_vehicles = [v for v, entry in entry_by_vehicle.items() if entry == "main"]
        if not side_vehicles:
            errors.append("On-ramp scenarios require at least one vehicle with entry_road='side' (on-ramp).")
        if not main_vehicles:
            errors.append("On-ramp scenarios require at least one vehicle with entry_road='main' (highway).")
        if merge_count > 0:
            valid_merge = False
            for c in spec.vehicle_constraints:
                if c.constraint_type != ConstraintType.MERGES_INTO_LANE_OF:
                    continue
                a_entry = entry_by_vehicle.get(c.vehicle_a, "unknown")
                b_entry = entry_by_vehicle.get(c.vehicle_b, "unknown")
                if a_entry == "side" and b_entry == "main":
                    valid_merge = True
                    break
            if not valid_merge:
                errors.append(
                    "On-ramp merge must be merges_into_lane_of from a side-road vehicle into a main-road vehicle."
                )
        for c in spec.vehicle_constraints:
            if c.constraint_type != ConstraintType.SAME_APPROACH_AS:
                continue
            a_entry = entry_by_vehicle.get(c.vehicle_a, "unknown")
            b_entry = entry_by_vehicle.get(c.vehicle_b, "unknown")
            if {a_entry, b_entry} == {"side", "main"}:
                errors.append(
                    "On-ramp scenarios should not use same_approach_as between side-road and main-road vehicles."
                )
                break

    # Phase 0: Category-specific validation
    
    # Highway On-Ramp Merge: enforce exact vehicle counts
    if cat_info.name == "Highway On-Ramp Merge":
        if difficulty == 1 and num_vehicles != 2:
            errors.append(
                f"Highway On-Ramp d1 must have exactly 2 vehicles (got {num_vehicles}): 1 main road + 1 merging."
            )
        elif difficulty == 2 and num_vehicles != 3:
            errors.append(
                f"Highway On-Ramp d2 must have exactly 3 vehicles (got {num_vehicles}): 1 main + 1 merging + 1 following."
            )
        elif difficulty >= 3:
            # Enforce linear scaling: d3=4, d4=5, d5=6 vehicles
            expected_vehicles = difficulty + 1
            side_road_vehicles = sum(1 for v in spec.ego_vehicles if v.entry_road == "side")
            main_road_vehicles = sum(1 for v in spec.ego_vehicles if v.entry_road == "main")
            
            if num_vehicles != expected_vehicles:
                errors.append(
                    f"Highway On-Ramp d{difficulty} must have exactly {expected_vehicles} vehicles (got {num_vehicles}). "
                    f"Each difficulty adds 1 vehicle: d3=4, d4=5, d5=6."
                )
            
            if side_road_vehicles < 2:
                errors.append(
                    f"Highway On-Ramp d{difficulty} must have at least 2 vehicles from on-ramp (got {side_road_vehicles}). "
                    f"Higher difficulties require multiple merging vehicles for complexity."
                )
            
            if main_road_vehicles < 1:
                errors.append(
                    f"Highway On-Ramp d{difficulty} must have at least 1 vehicle on main highway (got {main_road_vehicles})."
                )
    
    # Intersection Deadlock Resolution: enforce linear scaling
    if cat_info.name == "Intersection Deadlock Resolution":
        expected_vehicles = difficulty + 1  # d1=2, d2=3, d3=4, d4=5, d5=6
        expected_constraints = difficulty  # d1=1, d2=2, d3=3, d4=4, d5=5
        if num_vehicles != expected_vehicles:
            errors.append(
                f"Intersection Deadlock Resolution d{difficulty} must have exactly {expected_vehicles} vehicles "
                f"(got {num_vehicles}). Each difficulty adds 1 vehicle."
            )
        if constraint_count != expected_constraints:
            errors.append(
                f"Intersection Deadlock Resolution d{difficulty} must have exactly {expected_constraints} constraints "
                f"(got {constraint_count}). Each vehicle beyond first needs 1 direct interaction."
            )
    
    # Validate turn maneuvers have different entry/exit roads
    for vehicle in spec.ego_vehicles:
        if vehicle.maneuver in {EgoManeuver.LEFT, EgoManeuver.RIGHT}:
            if vehicle.entry_road == vehicle.exit_road and vehicle.entry_road not in {"unknown", None}:
                errors.append(
                    f"{vehicle.vehicle_id} has maneuver='{vehicle.maneuver.value}' but entry_road=exit_road='{vehicle.entry_road}'. "
                    f"Turn maneuvers MUST have different entry and exit roads (e.g., entry='main', exit='side')."
                )

    # Phase 1: Constraint requirements for inter-vehicle conflicts
    # Corridor topologies (e.g., "Blocked Lane (Obstacle)") can have single constraints since actors (not vehicles) block
    # Intersection/junction topologies need crossing relations
    is_corridor = cat_info.required_topology == TopologyType.CORRIDOR
    is_highway_onramp = cat_info.needs_on_ramp
    
    custom_scaling_cats = {
        "Pedestrian Crosswalk",
        "Highway On-Ramp Merge",
        "Intersection Deadlock Resolution",
        "Weaving Section",
    }

    # Special case: Highway on-ramp at difficulty 2 only needs 1 constraint (the merge itself is complex)
    if cat_info.name != "Pedestrian Crosswalk":
        if not is_corridor and difficulty >= 2 and constraint_count < 2:
            if not (is_highway_onramp and difficulty == 2 and constraint_count >= 1):
                msg = "Difficulty >=2 requires at least 2 inter-vehicle constraints."
                if actor_conflict:
                    warnings.append(msg)
                else:
                    errors.append(msg)
        elif is_corridor and difficulty >= 2 and constraint_count < 1:
            # Corridor scenarios need at least 1 constraint to position ego vehicles relative to each other
            msg = "Difficulty >=2 requires at least 1 inter-vehicle constraint."
            if actor_conflict:
                warnings.append(msg)
            else:
                errors.append(msg)
    
    if difficulty >= 3 and num_vehicles < 3 and cat_info.name not in custom_scaling_cats:
        errors.append("Difficulty >=3 requires at least 3 ego vehicles.")
    if difficulty >= 4 and constraint_count < 3 and cat_info.name not in custom_scaling_cats:
        errors.append("Difficulty >=4 requires at least 3 constraints.")
    if difficulty >= 4 and cat_info.required_topology in {TopologyType.INTERSECTION, TopologyType.T_JUNCTION}:
        if crossing_count < 2 and cat_info.name not in custom_scaling_cats:
            errors.append("Difficulty >=4 requires at least 2 crossing relations.")
    if difficulty >= 4 and cat_info.needs_multi_lane and lane_count < 2:
        errors.append("Difficulty >=4 requires at least 2 lane/merge relations.")

    if cat_info.name == "Highway On-Ramp Merge":
        if difficulty < cat_info.non_ego_actors_min_difficulty and spec.actors:
            errors.append(
                "Highway On-Ramp Merge should not include non-ego actors at this difficulty."
            )
        if difficulty <= 2 and num_vehicles > 3:
            errors.append(
                "Highway On-Ramp Merge at difficulty <=2 should use at most 3 ego vehicles."
            )
    
    # Weaving Section: enforce difficulty-appropriate complexity
    if cat_info.name == "Weaving Section":
        if difficulty == 1 and num_vehicles != 2:
            errors.append(
                "Weaving Section D1 requires exactly 2 ego vehicles (simple pair weaving)."
            )
        elif difficulty == 2 and num_vehicles != 3:
            errors.append(
                "Weaving Section D2 requires exactly 3 ego vehicles (trio with 2+ lane changes)."
            )
        elif difficulty == 2 and constraint_count < 2:
            errors.append(
                "Weaving Section D2 requires at least 2 constraints (multiple lane change interactions)."
            )
        elif difficulty >= 3 and num_vehicles < 3:
            errors.append(
                f"Weaving Section D{difficulty} requires at least 3 ego vehicles."
            )
        elif difficulty >= 3 and constraint_count < 2:
            errors.append(
                f"Weaving Section D{difficulty} requires at least 2 constraints."
            )
        elif difficulty >= 4 and constraint_count < 3:
            errors.append(
                f"Weaving Section D{difficulty} requires at least 3 constraints (dense constraint web)."
            )
        # Weaving Section should NEVER have obstacles/props at any difficulty
        if spec.actors:
            errors.append(
                f"Weaving Section D{difficulty} should NOT include any obstacles/props. "
                "This is pure vehicle-to-vehicle lane weaving."
            )
    
    # Multi Left Turn Conflict: enforce 2+ left turns from perpendicular approaches
    if cat_info.name == "Multi Left Turn Conflict":
        left_turners = [v for v in spec.ego_vehicles if v.maneuver == EgoManeuver.LEFT]
        num_left_turners = len(left_turners)
        
        if num_left_turners < 2:
            errors.append(
                f"Multi Left Turn Conflict requires at least 2 vehicles with maneuver='left' "
                f"(got {num_left_turners}). This is a Multi Left Turn Conflict scenario."
            )
        
        # Check for perpendicular constraints between left turners
        perpendicular_constraints = [
            c for c in spec.vehicle_constraints 
            if c.constraint_type in {ConstraintType.PERPENDICULAR_LEFT_OF, ConstraintType.PERPENDICULAR_RIGHT_OF}
        ]
        
        # Check if there's an opposite_approach_of (which is wrong for this category)
        opposite_constraints = [
            c for c in spec.vehicle_constraints 
            if c.constraint_type == ConstraintType.OPPOSITE_APPROACH_OF
        ]
        
        if opposite_constraints:
            errors.append(
                "Multi Left Turn Conflict should NOT use opposite_approach_of. "
                "This is Multi Left Turn Conflict with perpendicular approaches (main road + side road), "
                "NOT opposing left turns. Use perpendicular_left_of or perpendicular_right_of instead."
            )
        
        if not perpendicular_constraints:
            errors.append(
                "Multi Left Turn Conflict requires perpendicular_left_of or perpendicular_right_of constraints "
                "between left-turning vehicles (they approach from different directions: main road and side road)."
            )
        
        # Verify at least 2 left-turning vehicles have perpendicular relationship
        left_turner_ids = {v.vehicle_id for v in left_turners}
        has_perpendicular_left_turners = False
        for c in perpendicular_constraints:
            if c.vehicle_a in left_turner_ids and c.vehicle_b in left_turner_ids:
                has_perpendicular_left_turners = True
                break
        
        if num_left_turners >= 2 and not has_perpendicular_left_turners:
            errors.append(
                "At least 2 left-turning vehicles must have a perpendicular relationship "
                "(perpendicular_left_of or perpendicular_right_of) to ensure their turn paths intersect."
            )
    
    # Check for required non-ego actors only for obstacle-focused categories at D1+
    # Categories like "Blocked Lane (Obstacle)", "Construction Zone", "Parked Vehicle" are centered on obstacles
    obstacle_focused_categories = {
        "Blocked Lane (Obstacle)",
        "Construction Zone",
        "Parked Vehicle",
        "Lane Drop / Alternating Merge",
    }
    is_obstacle_focused = cat_info.name in obstacle_focused_categories
    
    if is_obstacle_focused and difficulty >= 1 and not spec.actors:
        errors.append(
            f"Category '{cat_info.name}' requires obstacles/props (parked_vehicle, traffic_cone, etc) at D{difficulty}. "
            "Add 'actors' field with at least 1 non-ego actor."
        )
    
    # Pedestrian Crosswalk: enforce pedestrian/occlusion requirements per difficulty
    if cat_info.name == "Pedestrian Crosswalk":
        # Count walkers
        walker_count = sum(1 for a in spec.actors if a.kind == ActorKind.WALKER)
        # Count parked vehicles (occlusions)
        parked_count = sum(1 for a in spec.actors if a.kind == ActorKind.PARKED_VEHICLE and a.motion == MotionType.STATIC)
        
        if difficulty <= 3:
            # D1-D3: Single pedestrian, no occlusions
            if walker_count != 1:
                errors.append(
                    f"Pedestrian Crosswalk d{difficulty} must have exactly 1 walker actor "
                    f"(got {walker_count}). D1-D3 use single pedestrian."
                )
            if parked_count > 0:
                warnings.append(
                    f"Pedestrian Crosswalk d{difficulty} should NOT have occlusions (parked vehicles). "
                    "Occlusions are for D4-D5 only."
                )
        else:
            # D4-D5: Focus on pedestrian count (2/3 pedestrians) with occlusions
            expected_pedestrians = difficulty - 2  # d4=2, d5=3
            if walker_count != expected_pedestrians:
                errors.append(
                    f"Pedestrian Crosswalk d{difficulty} must have exactly {expected_pedestrians} walker actors "
                    f"(got {walker_count}). D4=2, D5=3 pedestrians."
                )
            if parked_count < 1:
                errors.append(
                    f"Pedestrian Crosswalk d{difficulty} REQUIRES at least 1 parked_vehicle occlusion. "
                    "Pedestrians must spawn BEHIND parked vehicles (trucks, vans, buses) positioned roadside."
                )
            if num_vehicles < 2 or num_vehicles > 3:
                warnings.append(
                    f"Pedestrian Crosswalk d{difficulty} should have 2-3 ego vehicles "
                    f"(got {num_vehicles}). D4-D5 focus on pedestrian count, not vehicle count."
                )
    
    # Occluded Hazard: requires occluder + hidden moving actor
    if cat_info.name == "Occluded Hazard":
        if not spec.actors or len(spec.actors) < 2:
            errors.append(
                "Occluded Hazard requires 2 actors: (1) stationary occluder (parked_vehicle/static_prop) "
                "and (2) moving hazard (walker/cyclist) that emerges from behind."
            )
        else:
            if any(actor.kind == ActorKind.NPC_VEHICLE for actor in spec.actors):
                errors.append(
                    "Occluded Hazard should not use npc_vehicle actors; the moving hazard must be a walker or cyclist."
                )
            # Check for occluder (static)
            has_occluder = any(
                actor.motion == MotionType.STATIC and 
                actor.kind in {ActorKind.PARKED_VEHICLE, ActorKind.STATIC_PROP}
                for actor in spec.actors
            )
            # Check for moving hazard
            has_moving = any(
                actor.motion == MotionType.CROSS_PERPENDICULAR and
                actor.kind in {ActorKind.WALKER, ActorKind.CYCLIST}
                for actor in spec.actors
            )
            if not has_occluder:
                errors.append(
                    "Occluded Hazard requires a stationary occluder (parked_vehicle or static_prop with motion=static)."
                )
            if not has_moving:
                errors.append(
                    "Occluded Hazard requires a moving hazard (walker or cyclist with motion=cross_perpendicular)."
                )
            for actor in spec.actors:
                if actor.kind == ActorKind.WALKER:
                    aid = (actor.actor_id or "").lower()
                    if not any(tok in aid for tok in ("pedestrian", "walker", "person")):
                        errors.append(
                            "Occluded Hazard walker should be labeled as a pedestrian/walker in actor_id."
                        )
                if actor.kind == ActorKind.CYCLIST:
                    aid = (actor.actor_id or "").lower()
                    if not any(tok in aid for tok in ("cyclist", "bike", "bicycle")):
                        errors.append(
                            "Occluded Hazard cyclist should be labeled as a cyclist/bike in actor_id."
                        )
    
    if cat_info.name == "Lane Drop / Alternating Merge" and spec.actors:
        allowed_kinds = {ActorKind.STATIC_PROP, ActorKind.PARKED_VEHICLE}
        has_allowed = any(actor.kind in allowed_kinds for actor in spec.actors)
        if not has_allowed:
            errors.append(
                "Lane Drop / Alternating Merge requires a static obstacle (static_prop or parked_vehicle)."
            )
        valid_phases = {TimingPhase.AFTER_MERGE, TimingPhase.AFTER_EXIT}
        has_after_merge = any(actor.timing_phase in valid_phases for actor in spec.actors)
        if not has_after_merge:
            errors.append(
                "Lane Drop / Alternating Merge requires obstacle timing_phase=after_merge or after_exit."
            )
        # Accept either edge positions or half positions (half positions are better - more centered)
        acceptable_positions = {
            LateralPosition.RIGHT_EDGE, LateralPosition.LEFT_EDGE,
            LateralPosition.HALF_RIGHT, LateralPosition.HALF_LEFT
        }
        has_acceptable = any(actor.lateral_position in acceptable_positions for actor in spec.actors)
        if not has_acceptable:
            errors.append(
                "Lane Drop / Alternating Merge requires obstacles at half_left, half_right, left_edge, or right_edge."
            )
        # Check that static_prop obstacles have sufficient quantity to create taper
        for actor in spec.actors:
            if actor.kind == ActorKind.STATIC_PROP and actor.quantity < 3:
                errors.append(
                    f"Lane Drop obstacles should have quantity >= 3 to create visible taper (got {actor.quantity})."
                )
        # Prevent both-sided obstacles at lower difficulties
        if difficulty <= 3:
            left_count = sum(1 for a in spec.actors if a.lateral_position == LateralPosition.LEFT_EDGE)
            right_count = sum(1 for a in spec.actors if a.lateral_position == LateralPosition.RIGHT_EDGE)
            if left_count > 0 and right_count > 0:
                errors.append(
                    f"Lane Drop d{difficulty} should use obstacles on ONE side only (left_edge OR right_edge), "
                    "not both. Obstacles on both sides create excessive bottleneck at lower difficulties."
                )
        if any(actor.quantity > 1 for actor in spec.actors):
            has_along_lane = any(
                actor.group_pattern == GroupPattern.ALONG_LANE for actor in spec.actors if actor.quantity > 1
            )
            if not has_along_lane:
                errors.append(
                    "Lane Drop / Alternating Merge with multiple obstacles requires group_pattern=along_lane."
                )
    
    # General actor requirement for D5 (other categories can have optional actors)
    if difficulty >= 5 and cat_info.uses_non_ego_actors and difficulty >= cat_info.non_ego_actors_min_difficulty:
        if not spec.actors:
            errors.append("Difficulty >=5 requires at least 1 non-ego actor.")

    if num_vehicles >= 2:
        maneuvers = {v.maneuver for v in spec.ego_vehicles}
        same_only = all(
            c.constraint_type in {ConstraintType.SAME_APPROACH_AS, ConstraintType.FOLLOW_ROUTE_OF}
            for c in spec.vehicle_constraints
        ) if spec.vehicle_constraints else False
        if len(maneuvers) == 1 and same_only and crossing_count == 0 and lane_count == 0:
            msg = "All vehicles share the same approach/lane/maneuver; add conflict relations."
            warnings.append(msg)

    if num_vehicles >= 2:
        covered: Set[str] = set()
        for c in spec.vehicle_constraints:
            covered.add(c.vehicle_a)
            covered.add(c.vehicle_b)
        if actor_affects_all:
            covered.update(vehicles)
        else:
            covered.update(actor_targets)
        uncovered = [v for v in vehicles if v not in covered]
        if uncovered:
            warnings.append(
                f"{', '.join(uncovered)} not referenced by any constraint or actor; "
                "ensure each vehicle has an interaction source."
            )

    # Direction/lane clarity requirements
    direction_types = {
        ConstraintType.SAME_APPROACH_AS,
        ConstraintType.OPPOSITE_APPROACH_OF,
        ConstraintType.PERPENDICULAR_LEFT_OF,
        ConstraintType.PERPENDICULAR_RIGHT_OF,
        ConstraintType.LEFT_LANE_OF,
        ConstraintType.RIGHT_LANE_OF,
        ConstraintType.MERGES_INTO_LANE_OF,
        ConstraintType.FOLLOW_ROUTE_OF,
    }
    lane_detail_types = {
        ConstraintType.LEFT_LANE_OF,
        ConstraintType.RIGHT_LANE_OF,
        ConstraintType.MERGES_INTO_LANE_OF,
        ConstraintType.FOLLOW_ROUTE_OF,
    }

    for vehicle_id in vehicles[1:]:
        has_direction = any(
            c.vehicle_a == vehicle_id and c.constraint_type in direction_types
            for c in spec.vehicle_constraints
        )
        if not has_direction:
            msg = f"{vehicle_id} missing an approach relation (same/opposite/perpendicular/lane/merge)."
            if actor_conflict:
                warnings.append(msg)
            else:
                errors.append(msg)

        has_same_approach = any(
            c.vehicle_a == vehicle_id and c.constraint_type == ConstraintType.SAME_APPROACH_AS
            for c in spec.vehicle_constraints
        )
        if has_same_approach:
            has_lane_detail = any(
                c.vehicle_a == vehicle_id and c.constraint_type in lane_detail_types
                for c in spec.vehicle_constraints
            )
            if not has_lane_detail:
                errors.append(f"{vehicle_id} on same approach requires a lane relation (left/right/merge/follow).")

    return errors, warnings


def _ensure_direction_and_lane_constraints(spec: ScenarioSpec) -> ScenarioSpec:
    if len(spec.ego_vehicles) < 2:
        return spec

    actor_conflict, _, _ = _actor_conflict_info(spec)
    vehicle_b = spec.ego_vehicles[0].vehicle_id
    vehicle_a = spec.ego_vehicles[1].vehicle_id

    existing_types = {c.constraint_type for c in spec.vehicle_constraints}
    existing_keys = {(c.constraint_type, c.vehicle_a, c.vehicle_b) for c in spec.vehicle_constraints}
    crossing_types = {
        ConstraintType.OPPOSITE_APPROACH_OF,
        ConstraintType.PERPENDICULAR_LEFT_OF,
        ConstraintType.PERPENDICULAR_RIGHT_OF,
    }
    lane_types = {
        ConstraintType.LEFT_LANE_OF,
        ConstraintType.RIGHT_LANE_OF,
        ConstraintType.MERGES_INTO_LANE_OF,
    }

    def add_constraint(constraint_type: ConstraintType, a: str, b: str) -> None:
        key = (constraint_type, a, b)
        if key in existing_keys:
            return
        spec.vehicle_constraints.append(
            InterVehicleConstraint(
                constraint_type=constraint_type,
                vehicle_a=a,
                vehicle_b=b,
            )
        )
        existing_keys.add(key)
        existing_types.add(constraint_type)

    if spec.needs_oncoming:
        add_constraint(ConstraintType.OPPOSITE_APPROACH_OF, vehicle_a, vehicle_b)

    if spec.needs_merge or spec.needs_on_ramp:
        add_constraint(ConstraintType.MERGES_INTO_LANE_OF, vehicle_a, vehicle_b)

    if spec.needs_multi_lane and not any(c.constraint_type in lane_types for c in spec.vehicle_constraints):
        add_constraint(ConstraintType.LEFT_LANE_OF, vehicle_a, vehicle_b)

    if (
        spec.topology in {TopologyType.INTERSECTION, TopologyType.T_JUNCTION}
        and not any(c.constraint_type in crossing_types for c in spec.vehicle_constraints)
        and not spec.needs_oncoming
    ):
        add_constraint(ConstraintType.PERPENDICULAR_RIGHT_OF, vehicle_a, vehicle_b)

    if actor_conflict or spec.topology in {TopologyType.CORRIDOR, TopologyType.HIGHWAY}:
        direction_types = {
            ConstraintType.SAME_APPROACH_AS,
            ConstraintType.OPPOSITE_APPROACH_OF,
            ConstraintType.PERPENDICULAR_LEFT_OF,
            ConstraintType.PERPENDICULAR_RIGHT_OF,
            ConstraintType.LEFT_LANE_OF,
            ConstraintType.RIGHT_LANE_OF,
            ConstraintType.MERGES_INTO_LANE_OF,
            ConstraintType.FOLLOW_ROUTE_OF,
        }
        for vehicle in spec.ego_vehicles[1:]:
            has_direction = any(
                c.vehicle_a == vehicle.vehicle_id and c.constraint_type in direction_types
                for c in spec.vehicle_constraints
            )
            if not has_direction:
                add_constraint(ConstraintType.FOLLOW_ROUTE_OF, vehicle.vehicle_id, vehicle_b)

    return spec


@dataclass
class SchemaGenerationConfig:
    model_id: str = "Qwen/Qwen2.5-32B-Instruct-AWQ"
    max_new_tokens: int = 1024
    temperature: float = 0.6
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    do_sample: bool = True
    max_retries: int = 3
    similarity_threshold: float = 0.7
    allow_template_fallback: bool = True


class TemplateSchemaGenerator:
    """Deterministic schema generator for quick testing."""

    def generate_spec(self, category: str, difficulty: int) -> ScenarioSpec:
        cat_info = CATEGORY_DEFINITIONS[category]
        topology = cat_info.required_topology

        # Scale vehicle count with difficulty
        # For on-ramp: d1=2, d2=3, d3=4, d4=4, d5=5
        # For Lane Drop: d1=2, d2=3, d3=4, d4=5, d5=6 (needs more vehicles to show complexity)
        # For other categories: d1=2, d2=2, d3=3, d4=4, d5=4
        if cat_info.name == "Lane Drop / Alternating Merge":
            # Progressive scaling for merge scenarios - each difficulty adds a vehicle
            ego_count = min(difficulty + 1, 6)  # d1=2, d2=3, d3=4, d4=5, d5=6
        elif cat_info.name == "Weaving Section":
            if difficulty == 1:
                ego_count = 2
            elif difficulty == 2:
                ego_count = 3
            elif difficulty == 3:
                ego_count = 4
            elif difficulty == 4:
                ego_count = 4
            else:
                ego_count = 5
        elif cat_info.name == "Pedestrian Crosswalk":
            # Match validation: d1=2, d2=3, d3=4, d4=2, d5=3
            if difficulty == 1:
                ego_count = 2
            elif difficulty == 2:
                ego_count = 3
            elif difficulty == 3:
                ego_count = 4
            elif difficulty == 4:
                ego_count = 2
            else:
                ego_count = 3
        elif cat_info.needs_on_ramp:
            # On-ramp scenarios: d1=2, d2=3, d3=4, d4=5, d5=6
            ego_count = min(difficulty + 1, 6)  # d1=2, d2=3, d3=4, d4=5, d5=6
        else:
            # Standard scaling
            if difficulty == 1:
                ego_count = 2
            elif difficulty <= 2:
                ego_count = 2
            elif difficulty == 3:
                ego_count = 3
            else:
                ego_count = 4

        vehicles: List[EgoVehicleSpec] = []
        constraints: List[InterVehicleConstraint] = []

        # Special handling for Weaving Section - all vehicles do lane changes
        if cat_info.name == "Weaving Section":
            base_maneuver = EgoManeuver.LANE_CHANGE
        else:
            base_maneuver = EgoManeuver.STRAIGHT
            if "Left Turn" in category:
                base_maneuver = EgoManeuver.LEFT
            elif "Lane Change" in category:
                base_maneuver = EgoManeuver.LANE_CHANGE

        vehicles.append(
            EgoVehicleSpec(
                vehicle_id="Vehicle 1",
                maneuver=base_maneuver,
                lane_change_phase="unknown" if cat_info.name == "Weaving Section" else ("after_intersection" if base_maneuver == EgoManeuver.LANE_CHANGE else "unknown"),
                entry_road="unknown" if cat_info.name == "Weaving Section" else ("side" if "Side Street" in category else ("main" if cat_info.needs_on_ramp else "unknown")),
                exit_road="unknown" if cat_info.name == "Weaving Section" else ("main" if ("Side Street" in category or cat_info.needs_on_ramp) else "unknown"),
            )
        )

        if ego_count >= 2:
            if cat_info.name == "Weaving Section":
                second_maneuver = EgoManeuver.LANE_CHANGE
                second_lc_phase = "unknown"
                second_entry = "unknown"
                second_exit = "unknown"
            else:
                second_maneuver = EgoManeuver.STRAIGHT if not cat_info.needs_on_ramp else EgoManeuver.LANE_CHANGE
                second_lc_phase = "after_intersection" if cat_info.needs_on_ramp else "unknown"
                second_entry = "main" if "Side Street" in category else ("side" if cat_info.needs_on_ramp else "unknown")
                second_exit = "main" if cat_info.needs_on_ramp else "unknown"
            
            vehicles.append(
                EgoVehicleSpec(
                    vehicle_id="Vehicle 2",
                    maneuver=second_maneuver,
                    lane_change_phase=second_lc_phase,
                    entry_road=second_entry,
                    exit_road=second_exit,
                )
            )

        if cat_info.needs_oncoming and ego_count >= 2:
            constraints.append(
                InterVehicleConstraint(
                    constraint_type=ConstraintType.OPPOSITE_APPROACH_OF,
                    vehicle_a="Vehicle 2",
                    vehicle_b="Vehicle 1",
                )
            )
        elif topology in {TopologyType.INTERSECTION, TopologyType.T_JUNCTION} and ego_count >= 2:
            constraints.append(
                InterVehicleConstraint(
                    constraint_type=ConstraintType.PERPENDICULAR_RIGHT_OF,
                    vehicle_a="Vehicle 2",
                    vehicle_b="Vehicle 1",
                )
            )
        elif topology in {TopologyType.CORRIDOR, TopologyType.HIGHWAY} and ego_count >= 2:
            # For merge/on-ramp categories, Vehicle 2 should merge into Vehicle 1
            if cat_info.needs_merge or cat_info.needs_on_ramp:
                constraints.append(
                    InterVehicleConstraint(
                        constraint_type=ConstraintType.MERGES_INTO_LANE_OF,
                        vehicle_a="Vehicle 2",
                        vehicle_b="Vehicle 1",
                    )
                )
                # For difficulty 2 non-onramp highway merges, add second constraint
                if difficulty == 2 and ego_count == 2 and not cat_info.needs_on_ramp:
                    # Only for non-onramp merges (lane change merges)
                    constraints.append(
                        InterVehicleConstraint(
                            constraint_type=ConstraintType.SAME_APPROACH_AS,
                            vehicle_a="Vehicle 2",
                            vehicle_b="Vehicle 1",
                        )
                    )
            else:
                constraints.append(
                    InterVehicleConstraint(
                        constraint_type=ConstraintType.LEFT_LANE_OF,
                        vehicle_a="Vehicle 2",
                        vehicle_b="Vehicle 1",
                    )
                )

        if ego_count >= 3:
            vehicles.append(
                EgoVehicleSpec(
                    vehicle_id="Vehicle 3",
                    maneuver=EgoManeuver.LANE_CHANGE if cat_info.name == "Weaving Section" else EgoManeuver.STRAIGHT,
                    lane_change_phase="unknown",
                    entry_road="unknown",
                    exit_road="unknown",
                )
            )
            if topology in {TopologyType.INTERSECTION, TopologyType.T_JUNCTION}:
                if cat_info.needs_oncoming:
                    constraints.append(
                        InterVehicleConstraint(
                            constraint_type=ConstraintType.PERPENDICULAR_RIGHT_OF,
                            vehicle_a="Vehicle 3",
                            vehicle_b="Vehicle 1",
                        )
                    )
                else:
                    constraints.append(
                        InterVehicleConstraint(
                            constraint_type=ConstraintType.OPPOSITE_APPROACH_OF,
                            vehicle_a="Vehicle 3",
                            vehicle_b="Vehicle 1",
                        )
                    )
            elif topology in {TopologyType.CORRIDOR, TopologyType.HIGHWAY} and (cat_info.needs_merge or cat_info.needs_on_ramp):
                if cat_info.needs_on_ramp:
                    # For d3+, Vehicle 3 should be a second on-ramp vehicle
                    vehicles[-1].entry_road = "side"
                    vehicles[-1].exit_road = "main"
                    vehicles[-1].maneuver = EgoManeuver.LANE_CHANGE
                    # Use MERGES_INTO_LANE_OF for the second on-ramp vehicle
                    constraints.append(
                        InterVehicleConstraint(
                            constraint_type=ConstraintType.MERGES_INTO_LANE_OF,
                            vehicle_a="Vehicle 3",
                            vehicle_b="Vehicle 1",
                        )
                    )
                else:
                    vehicles[-1].maneuver = EgoManeuver.LANE_CHANGE
                    vehicles[-1].lane_change_phase = "unknown"
                    constraints.append(
                        InterVehicleConstraint(
                            constraint_type=ConstraintType.MERGES_INTO_LANE_OF,
                            vehicle_a="Vehicle 3",
                            vehicle_b="Vehicle 1",
                        )
                    )
            else:
                constraints.append(
                    InterVehicleConstraint(
                        constraint_type=ConstraintType.FOLLOW_ROUTE_OF,
                        vehicle_a="Vehicle 3",
                        vehicle_b="Vehicle 2",
                    )
                )

        if ego_count >= 4:
            vehicles.append(
                EgoVehicleSpec(
                    vehicle_id="Vehicle 4",
                    maneuver=EgoManeuver.LANE_CHANGE if (cat_info.name == "Weaving Section" or topology in {TopologyType.CORRIDOR, TopologyType.HIGHWAY}) else EgoManeuver.STRAIGHT,
                    lane_change_phase="after_intersection" if topology not in {TopologyType.CORRIDOR, TopologyType.HIGHWAY} else "unknown",
                    entry_road="unknown",
                    exit_road="unknown",
                )
            )
            if topology in {TopologyType.CORRIDOR, TopologyType.HIGHWAY} and (cat_info.needs_merge or cat_info.needs_on_ramp):
                if cat_info.needs_on_ramp:
                    vehicles[-1].maneuver = EgoManeuver.STRAIGHT
                    vehicles[-1].entry_road = "main"
                    vehicles[-1].exit_road = "main"
                    # Use same_lane_as to queue mainline vehicles in the same lane
                    constraints.append(
                        InterVehicleConstraint(
                            constraint_type=ConstraintType.SAME_LANE_AS,
                            vehicle_a="Vehicle 4",
                            vehicle_b="Vehicle 1",
                        )
                    )
                else:
                    constraints.append(
                        InterVehicleConstraint(
                            constraint_type=ConstraintType.MERGES_INTO_LANE_OF,
                            vehicle_a="Vehicle 4",
                            vehicle_b="Vehicle 1",
                        )
                    )
            elif topology in {TopologyType.CORRIDOR, TopologyType.HIGHWAY} and cat_info.needs_multi_lane:
                constraints.append(
                    InterVehicleConstraint(
                        constraint_type=ConstraintType.RIGHT_LANE_OF,
                        vehicle_a="Vehicle 4",
                        vehicle_b="Vehicle 1",
                    )
                )
            else:
                constraints.append(
                    InterVehicleConstraint(
                        constraint_type=ConstraintType.FOLLOW_ROUTE_OF,
                        vehicle_a="Vehicle 4",
                        vehicle_b="Vehicle 2",
                    )
                )

        if ego_count >= 5:
            vehicles.append(
                EgoVehicleSpec(
                    vehicle_id="Vehicle 5",
                    maneuver=EgoManeuver.LANE_CHANGE if (cat_info.name == "Weaving Section" or topology in {TopologyType.CORRIDOR, TopologyType.HIGHWAY}) else EgoManeuver.STRAIGHT,
                    lane_change_phase="unknown",
                    entry_road="unknown",
                    exit_road="unknown",
                )
            )

        if cat_info.name == "Weaving Section":
            # Weaving Section: all vehicles get LANE_CHANGE maneuver
            constraints = []
            # Build constraints to create weaving pattern
            if ego_count >= 2:
                # Vehicle 2 starts left of Vehicle 1, will weave
                constraints.append(
                    InterVehicleConstraint(
                        constraint_type=ConstraintType.LEFT_LANE_OF,
                        vehicle_a="Vehicle 2",
                        vehicle_b="Vehicle 1",
                    )
                )
            if ego_count >= 3:
                # Vehicle 3 starts right of Vehicle 1, creating 3-lane spread
                constraints.append(
                    InterVehicleConstraint(
                        constraint_type=ConstraintType.RIGHT_LANE_OF,
                        vehicle_a="Vehicle 3",
                        vehicle_b="Vehicle 1",
                    )
                )
                # Add lane change interaction for d2+
                if difficulty >= 2:
                    constraints.append(
                        InterVehicleConstraint(
                            constraint_type=ConstraintType.MERGES_INTO_LANE_OF,
                            vehicle_a="Vehicle 2",
                            vehicle_b="Vehicle 3",
                        )
                    )
            if ego_count >= 4:
                # Vehicle 4 positioned relative to others
                constraints.append(
                    InterVehicleConstraint(
                        constraint_type=ConstraintType.LEFT_LANE_OF,
                        vehicle_a="Vehicle 4",
                        vehicle_b="Vehicle 2",
                    )
                )
                # Add more merging for d3+
                if difficulty >= 3:
                    constraints.append(
                        InterVehicleConstraint(
                            constraint_type=ConstraintType.MERGES_INTO_LANE_OF,
                            vehicle_a="Vehicle 3",
                            vehicle_b="Vehicle 1",
                        )
                    )
            if ego_count >= 5:
                # Vehicle 5 creates another lane position
                constraints.append(
                    InterVehicleConstraint(
                        constraint_type=ConstraintType.RIGHT_LANE_OF,
                        vehicle_a="Vehicle 5",
                        vehicle_b="Vehicle 3",
                    )
                )
                # Add more merging for d4+
                if difficulty >= 4:
                    constraints.append(
                        InterVehicleConstraint(
                            constraint_type=ConstraintType.MERGES_INTO_LANE_OF,
                            vehicle_a="Vehicle 4",
                            vehicle_b="Vehicle 1",
                        )
                    )
                # Even more merging for d5
                if difficulty >= 5:
                    constraints.append(
                        InterVehicleConstraint(
                            constraint_type=ConstraintType.MERGES_INTO_LANE_OF,
                            vehicle_a="Vehicle 1",
                            vehicle_b="Vehicle 2",
                        )
                    )

        if cat_info.name == "Pedestrian Crosswalk" and ego_count >= 2:
            constraints = [
                c for c in constraints
                if c.constraint_type not in {ConstraintType.LEFT_LANE_OF, ConstraintType.RIGHT_LANE_OF}
            ]
            if not any(c.constraint_type == ConstraintType.FOLLOW_ROUTE_OF for c in constraints):
                constraints.append(
                    InterVehicleConstraint(
                        constraint_type=ConstraintType.FOLLOW_ROUTE_OF,
                        vehicle_a="Vehicle 2",
                        vehicle_b="Vehicle 1",
                    )
                )

        actors: List[NonEgoActorSpec] = []
        if cat_info.uses_non_ego_actors and difficulty >= cat_info.non_ego_actors_min_difficulty:
            # Special handling for Occluded Hazard - requires 2 actors
            if cat_info.name == "Pedestrian Crosswalk":
                # Pedestrian Crosswalk actors follow validation rules
                walker_count = 1 if difficulty <= 3 else max(2, min(3, difficulty - 2))
                # Alternate start sides for variety
                start_positions = [LateralPosition.RIGHT_EDGE, LateralPosition.LEFT_EDGE]
                for i in range(walker_count):
                    actors.append(
                        NonEgoActorSpec(
                            actor_id=f"pedestrian_{i+1}",
                            kind=ActorKind.WALKER,
                            quantity=1,
                            affects_vehicle="Vehicle 1",
                            timing_phase=TimingPhase.ON_APPROACH,
                            lateral_position=start_positions[i % len(start_positions)],
                            group_pattern=GroupPattern.UNKNOWN,
                            start_lateral=start_positions[i % len(start_positions)],
                            end_lateral=LateralPosition.LEFT_EDGE if start_positions[i % len(start_positions)] == LateralPosition.RIGHT_EDGE else LateralPosition.RIGHT_EDGE,
                            motion=MotionType.CROSS_PERPENDICULAR,
                            speed=SpeedHint.UNKNOWN,
                            crossing_direction="left" if i % 2 == 0 else "right",
                        )
                    )
                if difficulty >= 4:
                    # Add required occluder parked vehicle (roadside, not blocking lane)
                    actors.append(
                        NonEgoActorSpec(
                            actor_id="parked_box_truck",
                            kind=ActorKind.PARKED_VEHICLE,
                            quantity=1,
                            affects_vehicle="Vehicle 1",
                            timing_phase=TimingPhase.ON_APPROACH,
                            lateral_position=LateralPosition.OFFROAD_RIGHT,
                            group_pattern=GroupPattern.UNKNOWN,
                            start_lateral=LateralPosition.OFFROAD_RIGHT,
                            end_lateral=LateralPosition.OFFROAD_RIGHT,
                            motion=MotionType.STATIC,
                            speed=SpeedHint.UNKNOWN,
                            crossing_direction=None,
                        )
                    )
            elif cat_info.name == "Occluded Hazard":
                # Actor 1: Stationary occluder (parked vehicle)
                actors.append(
                    NonEgoActorSpec(
                        actor_id="parked box truck",
                        kind=ActorKind.PARKED_VEHICLE,
                        quantity=1,
                        affects_vehicle="Vehicle 1",
                        timing_phase=TimingPhase.ON_APPROACH,
                        lateral_position=LateralPosition.RIGHT_EDGE,
                        group_pattern=GroupPattern.UNKNOWN,
                        start_lateral=None,
                        end_lateral=None,
                        motion=MotionType.STATIC,
                        speed=SpeedHint.UNKNOWN,
                        crossing_direction=None,
                    )
                )
                # Actor 2: Moving hazard (walker/cyclist crossing)
                actors.append(
                    NonEgoActorSpec(
                        actor_id="pedestrian" if difficulty <= 2 else "cyclist",
                        kind=ActorKind.WALKER if difficulty <= 2 else ActorKind.CYCLIST,
                        quantity=1,
                        affects_vehicle="Vehicle 1",
                        timing_phase=TimingPhase.ON_APPROACH,
                        lateral_position=LateralPosition.HALF_RIGHT,
                        group_pattern=GroupPattern.UNKNOWN,
                        start_lateral=None,
                        end_lateral=None,
                        motion=MotionType.CROSS_PERPENDICULAR,
                        speed=SpeedHint.UNKNOWN,
                        crossing_direction="left",
                    )
                )
            else:
                # Standard single-actor logic for other categories
                actor_kind = ActorKind.STATIC_PROP
                actor_id = "traffic cones"
                motion = MotionType.STATIC
                group_pattern = GroupPattern.DIAGONAL if cat_info.needs_merge else GroupPattern.UNKNOWN
                timing_phase = TimingPhase.AFTER_EXIT if topology not in {TopologyType.CORRIDOR, TopologyType.HIGHWAY} else TimingPhase.ON_APPROACH
                lateral_position = LateralPosition.RIGHT_EDGE
                quantity = 3
                if "Pedestrian" in category:
                    actor_kind = ActorKind.WALKER
                    actor_id = "pedestrian"
                    motion = MotionType.CROSS_PERPENDICULAR
                    group_pattern = GroupPattern.UNKNOWN
                    quantity = 1
                elif "Blocked Lane" in category or "Construction" in category:
                    actor_kind = ActorKind.PARKED_VEHICLE
                    actor_id = "parked vehicle"
                    group_pattern = GroupPattern.UNKNOWN
                    quantity = 1
                    lateral_position = LateralPosition.RIGHT_EDGE

                if cat_info.name == "Lane Drop / Alternating Merge":
                    group_pattern = GroupPattern.ALONG_LANE
                    timing_phase = TimingPhase.AFTER_MERGE
                    # Use half_left/half_right to center obstacles in dropping lane (not at edge)
                    # This prevents creeping into adjacent lane
                    lateral_position = LateralPosition.HALF_LEFT if difficulty % 2 == 0 else LateralPosition.HALF_RIGHT
                    start_lateral = lateral_position
                    end_lateral = lateral_position
                    quantity = min(3 + difficulty, 6)  # 4-6 obstacles depending on difficulty
                    
                    # Vary obstacle type: use parked_vehicle for higher difficulties
                    if difficulty >= 4:
                        actor_kind = ActorKind.PARKED_VEHICLE
                        actor_id = "parked vehicle"
                        quantity = 1  # Single parked vehicle creates strong blockage
                    else:
                        actor_kind = ActorKind.STATIC_PROP
                        actor_id = "traffic cones"
                else:
                    start_lateral = LateralPosition.RIGHT_EDGE if group_pattern == GroupPattern.DIAGONAL else None
                    end_lateral = LateralPosition.LEFT_EDGE if group_pattern == GroupPattern.DIAGONAL else None

                actors.append(
                    NonEgoActorSpec(
                        actor_id=actor_id,
                        kind=actor_kind,
                        quantity=quantity,
                        affects_vehicle="Vehicle 1",
                        timing_phase=timing_phase,
                        lateral_position=lateral_position,
                        group_pattern=group_pattern,
                        start_lateral=start_lateral,
                        end_lateral=end_lateral,
                        motion=motion,
                        speed=SpeedHint.UNKNOWN,
                        crossing_direction="left" if motion == MotionType.CROSS_PERPENDICULAR else None,
                    )
                )

        spec = ScenarioSpec(
            category=category,
            difficulty=difficulty,
            topology=topology,
            needs_oncoming=cat_info.needs_oncoming,
            needs_multi_lane=cat_info.needs_multi_lane,
            needs_on_ramp=cat_info.needs_on_ramp,
            needs_merge=cat_info.needs_merge,
            ego_vehicles=vehicles,
            vehicle_constraints=constraints,
            actors=actors,
        )
        spec = _ensure_direction_and_lane_constraints(spec)
        spec.description = description_from_spec(spec)
        return spec


class SchemaScenarioGenerator:
    """
    LLM-backed schema generator with validation and optional template fallback.
    """

    def __init__(
        self,
        config: Optional[SchemaGenerationConfig] = None,
        model=None,
        tokenizer=None,
        template_only: bool = False,
    ):
        self.config = config or SchemaGenerationConfig()
        self.template_only = template_only
        self._model = model
        self._tokenizer = tokenizer
        self._template = TemplateSchemaGenerator()
        self.available_categories = get_available_categories()
        self.used_combinations: Dict[str, Set[str]] = {c: set() for c in self.available_categories}
        self.generated_signatures: Dict[str, Set[str]] = {c: set() for c in self.available_categories}

    def _load_model(self):
        if self._model is not None and self._tokenizer is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_id, use_fast=True)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self._model.eval()

    def _generate_text(self, prompt: str, temperature_override: Optional[float] = None) -> str:
        self._load_model()
        import torch

        messages = [
            {"role": "system", "content": build_schema_system_prompt()},
            {"role": "user", "content": prompt},
        ]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self._tokenizer(text, return_tensors="pt")
        if hasattr(self._model, "device"):
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        do_sample = bool(self.config.do_sample)
        effective_temp = temperature_override if temperature_override is not None else self.config.temperature
        gen_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": effective_temp if do_sample else 0.0,
            "top_p": self.config.top_p if do_sample else 1.0,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.eos_token_id,
            "repetition_penalty": self.config.repetition_penalty,
        }

        with torch.no_grad():
            outputs = self._model.generate(**inputs, **gen_kwargs)

        generated_tokens = outputs[0, inputs["input_ids"].shape[-1]:]
        response = self._tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return response

    def _normalize_schema_obj(
        self,
        obj: Dict[str, Any],
        category: str,
        difficulty: int,
        cat_info: Any,
    ) -> Dict[str, Any]:
        # Phase 1: Extract reasoning if present (new format: {"reasoning": {...}, "scenario_spec": {...}})
        reasoning = None
        if "reasoning" in obj and "scenario_spec" in obj:
            reasoning = obj.get("reasoning", {})
            obj = obj.get("scenario_spec", {})
        # If repair format uses {"repair_of": {...}, "scenario_spec": {...}}, keep scenario_spec
        if "repair_of" in obj and "scenario_spec" in obj:
            obj = obj.get("scenario_spec", {})
        
        out = dict(obj) if isinstance(obj, dict) else {}
        
        # Phase 1: Store reasoning in metadata for validation/debugging
        if reasoning:
            out["_reasoning"] = reasoning
        
        out["category"] = category
        out["difficulty"] = difficulty
        out["topology"] = cat_info.required_topology.value
        out["needs_oncoming"] = bool(out.get("needs_oncoming", cat_info.needs_oncoming)) or cat_info.needs_oncoming
        out["needs_multi_lane"] = bool(out.get("needs_multi_lane", cat_info.needs_multi_lane)) or cat_info.needs_multi_lane
        out["needs_on_ramp"] = bool(out.get("needs_on_ramp", cat_info.needs_on_ramp)) or cat_info.needs_on_ramp
        out["needs_merge"] = bool(out.get("needs_merge", cat_info.needs_merge)) or cat_info.needs_merge

        vehicles = out.get("ego_vehicles")
        if not isinstance(vehicles, list) or not vehicles:
            vehicles = [{"vehicle_id": "Vehicle 1", "maneuver": "straight", "lane_change_phase": "unknown",
                         "entry_road": "unknown", "exit_road": "unknown"}]
        for idx, v in enumerate(vehicles, start=1):
            if not isinstance(v, dict):
                vehicles[idx - 1] = {
                    "vehicle_id": f"Vehicle {idx}",
                    "maneuver": "straight",
                    "lane_change_phase": "unknown",
                    "entry_road": "unknown",
                    "exit_road": "unknown",
                }
                continue
            v.setdefault("vehicle_id", f"Vehicle {idx}")
            v.setdefault("maneuver", "straight")
            v.setdefault("lane_change_phase", "unknown")
            v.setdefault("entry_road", "unknown")
            v.setdefault("exit_road", "unknown")
        vehicle_ids = {v.get("vehicle_id") for v in vehicles if isinstance(v, dict)}
        if "Vehicle 1" not in vehicle_ids:
            vehicles.insert(0, {
                "vehicle_id": "Vehicle 1",
                "maneuver": "straight",
                "lane_change_phase": "unknown",
                "entry_road": "unknown",
                "exit_road": "unknown",
            })
        out["ego_vehicles"] = vehicles

        constraints = out.get("vehicle_constraints")
        if not isinstance(constraints, list):
            constraints = []
        out["vehicle_constraints"] = constraints

        actors = out.get("actors")
        if not isinstance(actors, list):
            actors = []
        out["actors"] = actors

        return out

    def _signature(self, spec: ScenarioSpec) -> str:
        vehicle_sig = ",".join(f"{v.vehicle_id}:{v.maneuver.value}" for v in spec.ego_vehicles)
        constraint_sig = ",".join(f"{c.constraint_type.value}:{c.vehicle_a}->{c.vehicle_b}" for c in spec.vehicle_constraints)
        actor_sig = ",".join(f"{a.kind.value}:{a.quantity}" for a in spec.actors)
        return "|".join([vehicle_sig, constraint_sig, actor_sig])

    def generate_spec(
        self,
        category: str,
        difficulty: int,
        stats: Optional[Dict[str, Any]] = None,
        exclude_signatures: Optional[Set[str]] = None,
        previous_validation_feedback: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[ScenarioSpec], List[str], List[str]]:
        """Generate a scenario spec.
        
        Args:
            category: Scenario category
            difficulty: Difficulty level (1-3)
            stats: Optional dict to populate with generation statistics
            exclude_signatures: Optional set of spec signatures to avoid (for retry attempts)
            previous_validation_feedback: Optional dict with validation results from failed attempt:
                {"score": float, "issues": [{"severity": str, "message": str, ...}]}
        
        Returns:
            Tuple of (spec, errors, warnings). spec is None on failure.
        """
        if category not in self.available_categories:
            return None, [f"Category '{category}' is not supported"], []

        cat_info = CATEGORY_DEFINITIONS[category]
        if self.template_only:
            return self._template.generate_spec(category, difficulty), [], []

        exclude_signatures = exclude_signatures or set()
        used = self.used_combinations.get(category, set())
        forced_variations = select_variation_values(cat_info, used)
        combo_key = "|".join(f"{k}={v}" for k, v in sorted(forced_variations.items()))

        # Derive banned constraints from previous validation feedback (INEFFECTIVE hints)
        banned_constraints: Set[str] = set()
        if previous_validation_feedback:
            for issue in previous_validation_feedback.get("issues", []):
                text = " ".join([
                    issue.get("message", "") or "",
                    issue.get("suggestion", "") or "",
                ])
                import re
                for m in re.findall(r'(\w+)\(Vehicle\s*(\d+)\s*->\s*Vehicle\s*(\d+)\)', text):
                    banned_constraints.add(f"{m[0].lower()}(Vehicle {m[1]} -> Vehicle {m[2]})")

        prompt = build_schema_generation_prompt(
            category, difficulty, cat_info, forced_variations, 
            previous_validation_feedback=previous_validation_feedback
        )
        last_errors: List[str] = []
        last_warnings: List[str] = []
        last_payload = ""

        attempt_count = 0
        repair_attempts = 0
        # Use higher temperature for retries to encourage exploration
        is_outer_retry = previous_validation_feedback is not None
        retry_temp_boost = 0.2 if is_outer_retry else 0.0  # Boost temp by 0.2 on outer retries
        for attempt in range(self.config.max_retries):
            attempt_count += 1
            if attempt > 0:
                repair_attempts += 1
            # Further boost temperature on inner repair attempts
            inner_temp_boost = 0.1 * attempt  # +0.1 per repair attempt
            # For repair attempts, keep temperature modest to reduce hallucinated constraint types
            base_temp = self.config.temperature + retry_temp_boost + inner_temp_boost
            effective_temp = min(0.8 if attempt > 0 else 1.0, base_temp)
            response = self._generate_text(prompt, temperature_override=effective_temp)
            last_payload = response
            obj = _extract_first_json_object(response)
            if obj is None:
                last_errors = ["No valid JSON object found"]
                last_warnings = []
            else:
                normalized = self._normalize_schema_obj(obj, category, difficulty, cat_info)
                try:
                    spec = spec_from_dict(normalized)
                except Exception as exc:
                    last_errors = [f"Schema parse error: {exc}"]
                    last_warnings = []
                    # If parse failed due to invalid constraint types, tighten the prompt for next attempt
                    invalid_constraints = []
                    msg = str(exc)
                    if "ConstraintType" in msg:
                        invalid_constraints.append(msg)
                    if invalid_constraints and attempt < self.config.max_retries - 1:
                        prompt = build_schema_repair_prompt(last_payload, invalid_constraints, category, difficulty, cat_info)
                else:
                    spec = _ensure_direction_and_lane_constraints(spec)
                    valid, errors = validate_spec(spec)
                    conflict_errors, conflict_warnings = _conflict_findings(spec, cat_info)
                    if conflict_errors:
                        errors = errors + conflict_errors
                        valid = False
                    last_warnings = conflict_warnings
                    # Reject specs that reuse banned constraints from previous feedback
                    if valid:
                        constraint_signatures = {
                            f"{c.constraint_type.value}(Vehicle {c.vehicle_a.split()[-1]} -> Vehicle {c.vehicle_b.split()[-1]})"
                            for c in spec.vehicle_constraints
                        }
                        if banned_constraints and any(sig.lower() in banned_constraints for sig in constraint_signatures):
                            valid = False
                            errors = [f"Spec reuses banned constraints: {sorted(banned_constraints)}"]
                    if valid:
                        spec.description = description_from_spec(spec)
                        sig = self._signature(spec)
                        if sig in self.generated_signatures[category] or sig in exclude_signatures:
                            if sig in exclude_signatures:
                                last_errors = ["Generated spec matches a previously failed scenario - trying alternative"]
                            else:
                                last_errors = ["Generated spec too similar to existing scenario"]
                            last_warnings = []
                        else:
                            self.used_combinations[category].add(combo_key)
                            self.generated_signatures[category].add(sig)
                            if stats is not None:
                                stats["schema_generation_attempts"] = attempt_count
                                stats["schema_generation_repair_attempts"] = repair_attempts
                                stats["schema_template_fallback"] = 0
                            return spec, [], conflict_warnings
                    else:
                        last_errors = errors

            if attempt < self.config.max_retries - 1:
                prompt = build_schema_repair_prompt(last_payload, last_errors, category, difficulty, cat_info)

        # Disable template fallback when we have validation feedback (force new LLM sample)
        allow_fallback = self.config.allow_template_fallback and not previous_validation_feedback
        if allow_fallback:
            template_spec = self._template.generate_spec(category, difficulty)
            template_errors, _template_warnings = _conflict_findings(template_spec, cat_info)
            if template_errors:
                if stats is not None:
                    stats["schema_generation_attempts"] = attempt_count
                    stats["schema_generation_repair_attempts"] = repair_attempts
                    stats["schema_template_fallback"] = 0
                return None, template_errors, _template_warnings
            if stats is not None:
                stats["schema_generation_attempts"] = attempt_count
                stats["schema_generation_repair_attempts"] = repair_attempts
                stats["schema_template_fallback"] = 1
            return template_spec, last_errors, last_warnings
        if stats is not None:
            stats["schema_generation_attempts"] = attempt_count
            stats["schema_generation_repair_attempts"] = repair_attempts
            stats["schema_template_fallback"] = 0
        return None, last_errors, last_warnings
