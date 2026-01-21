#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-32B-Instruct-AWQ}"
OUT_DIR="${OUT_DIR:-log_behavior_smoke}"
ROUTES_DIR="${ROUTES_DIR:-routes_behavior_smoke}"
SCENARIOS_FILE="${SCENARIOS_FILE:-$OUT_DIR/behavior_smoke_scenarios.txt}"
SCENARIO_INDEX="${SCENARIO_INDEX:-}"
CARLA_ASSETS="${CARLA_ASSETS:-$ROOT_DIR/scenario_generator/carla_assets.json}"
TOWN_NODES_DIR="${TOWN_NODES_DIR:-}"

if [ -z "$TOWN_NODES_DIR" ]; then
  TOWN_NODES_DIR="$(find "$ROOT_DIR" -maxdepth 4 -type d -name 'town_nodes*' -print -quit || true)"
fi
if [ -z "$TOWN_NODES_DIR" ] || [ ! -d "$TOWN_NODES_DIR" ]; then
  echo "[ERROR] town_nodes directory not found. Set TOWN_NODES_DIR explicitly."
  exit 1
fi
if [ ! -f "$CARLA_ASSETS" ]; then
  echo "[ERROR] carla_assets.json not found. Set CARLA_ASSETS explicitly."
  exit 1
fi

mkdir -p "$OUT_DIR"

cat > "$SCENARIOS_FILE" <<'EOF'
Vehicle 1 goes straight through an intersection. After the intersection on Vehicle 1's exit road, a pedestrian waits on the right sidewalk. When Vehicle 1 gets within 8 meters, the pedestrian starts to cross from right to left.
Vehicle 1 goes straight through an intersection. On Vehicle 1's exit road, an NPC vehicle travels ahead in the same lane. When Vehicle 1 gets within 10 meters, the NPC vehicle brakes hard to a stop.
Vehicle 1 goes straight through an intersection. An NPC vehicle travels ahead of Vehicle 1 in the lane to the right of Vehicle 1. When Vehicle 1 gets within 25 meters, the NPC vehicle changes lanes to the left.
Vehicle 1 goes straight through an intersection onto a two-lane exit road. An NPC vehicle travels ahead in the lane to the left of Vehicle 1. When Vehicle 1 gets within 12 meters, the NPC vehicle changes lanes into Vehicle 1's lane.
Vehicle 1 goes straight through an intersection. On the exit road, an oncoming NPC vehicle travels in the opposite direction in its lane. Vehicle 1 continues straight.
Vehicle 1 goes straight through an intersection. On the exit road, a cyclist waits at the left edge. When Vehicle 1 gets within 9 meters, the cyclist starts moving along the lane in the same direction as Vehicle 1.
EOF
if [ -n "$SCENARIO_INDEX" ]; then
  if ! [[ "$SCENARIO_INDEX" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] SCENARIO_INDEX must be a 1-based integer."
    exit 1
  fi
  if ! selected_scenario=$(awk -v idx="$SCENARIO_INDEX" 'NR==idx {print; found=1} END {exit (found?0:2)}' "$SCENARIOS_FILE"); then
    echo "[ERROR] SCENARIO_INDEX=$SCENARIO_INDEX out of range."
    exit 1
  fi
  echo "[INFO] Using scenario #$SCENARIO_INDEX: $selected_scenario"
  INDICES_ARGS=(--indices "$SCENARIO_INDEX")
else
  INDICES_ARGS=()
fi

python "$ROOT_DIR/scenario_generator/run_scenario_pipeline.py" \
  --model "$MODEL_ID" \
  --scenarios-file "$SCENARIOS_FILE" \
  --out-dir "$OUT_DIR" \
  --routes-out-dir "$ROUTES_DIR" \
  --viz-objects \
  --carla-assets "$CARLA_ASSETS" \
  --town-nodes-dir "$TOWN_NODES_DIR" \
  --carla-host 127.0.0.1 \
  --carla-port 2000 \
  "${INDICES_ARGS[@]}"

echo "[OK] Behavior smoke run complete."
echo "Output log: $OUT_DIR"
echo "Routes:     $ROUTES_DIR"
