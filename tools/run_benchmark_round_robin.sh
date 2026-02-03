#!/usr/bin/env bash
# Round-robin evaluator: iterates scenarios so that runs are interleaved by category.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_DIR="${ROOT_DIR}/benchmarkfull"
EVAL_SCRIPT="${ROOT_DIR}/tools/run_custom_eval.py"
CSV_PATH="${ROOT_DIR}/results/benchmark_round_robin.csv"
: "${CARLA_ROOT:=${ROOT_DIR}/carla912}"
export CARLA_ROOT

# Force evaluations onto GPU 4 as requested.
export CUDA_VISIBLE_DEVICES=4

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but not found in PATH" >&2
  exit 1
fi

if [[ ! -x "${EVAL_SCRIPT}" ]]; then
  echo "run_custom_eval.py not found or not executable at ${EVAL_SCRIPT}" >&2
  exit 1
fi

mkdir -p "$(dirname "${CSV_PATH}")"
if [[ ! -f "${CSV_PATH}" ]]; then
  echo "global_run,category,category_run,scenario,planner,driving_score,route_completion,results_dir,timestamp" > "${CSV_PATH}"
fi

# Collect scenarios grouped by category prefix (portion before _v*).
declare -A scenario_lists
categories=()

while IFS= read -r name; do
  category="${name%%_v*}"
  if [[ -v scenario_lists[$category] ]]; then
    scenario_lists[$category]+=" ${name}"
  else
    scenario_lists[$category]="${name}"
    categories+=("${category}")
  fi
done < <(find "${BENCH_DIR}" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort)

if [[ ${#categories[@]} -eq 0 ]]; then
  echo "No scenarios found under ${BENCH_DIR}" >&2
  exit 1
fi

IFS=$'\n' categories=($(printf "%s\n" "${categories[@]}" | sort))
IFS=$' \t\n'  # reset to default for word splitting in loops
declare -A next_idx
for cat in "${categories[@]}"; do
  next_idx[$cat]=0
done

PLANNERS_COLM=(colmdriver colmdriver_rulebase codriving tcp)
PLANNERS_B2D=(vad uniad)

metrics_from_result_dir() {
  local result_dir=$1
  python - "$result_dir" <<'PY'
import json, sys, pathlib
root = pathlib.Path(sys.argv[1])
scores = []
for p in sorted(root.glob("ego_vehicle_*/results.json")):
    try:
        data = json.loads(p.read_text())
    except Exception:
        continue
    label_map = dict(zip(data.get("labels", []), data.get("values", [])))
    ds = label_map.get("Avg. driving score")
    rc = label_map.get("Avg. route completion")
    if ds is None or rc is None:
        continue
    try:
        ds = float(ds)
        rc = float(rc)
    except Exception:
        continue
    scores.append((ds, rc))

if not scores:
    print("NA,NA")
else:
    avg_ds = sum(s[0] for s in scores) / len(scores)
    avg_rc = sum(s[1] for s in scores) / len(scores)
    print(f"{avg_ds:.3f},{avg_rc:.3f}")
PY
}

append_csv_row() {
  local global_run=$1
  local category=$2
  local category_run=$3
  local scenario=$4
  local planner=$5
  local result_dir=$6
  local driving_score=$7
  local route_completion=$8
  local ts
  ts=$(date -Iseconds)
  echo "${global_run},${category},${category_run},${scenario},${planner},${driving_score},${route_completion},${result_dir},${ts}" >> "${CSV_PATH}"
}

# Check if a scenario+planner combination already has valid results in the CSV.
# Returns 0 (success) if valid results exist and should be skipped, 1 otherwise.
has_valid_results() {
  local scenario=$1
  local planner=$2
  
  # Check if CSV exists and has more than header
  if [[ ! -f "${CSV_PATH}" ]]; then
    return 1
  fi
  
  # Look for rows matching scenario,planner with non-NA driving_score and route_completion
  # CSV format: global_run,category,category_run,scenario,planner,driving_score,route_completion,results_dir,timestamp
  # Fields: $4=scenario, $5=planner, $6=driving_score, $7=route_completion
  local found
  found=$(awk -F',' -v scen="$scenario" -v plan="$planner" '
    NR > 1 && $4 == scen && $5 == plan && $6 != "NA" && $7 != "NA" && $6 != "" && $7 != "" {
      print "found"; exit
    }
  ' "${CSV_PATH}")
  
  if [[ "${found}" == "found" ]]; then
    return 0
  fi
  return 1
}

run_planner() {
  local env_name=$1
  local planner=$2
  local routes_dir=$3
  local llm_port=$4
  local result_tag=$5
  local category=$6
  local scenario=$7
  local category_run=$8
  local global_run=$9

  local result_dir="${ROOT_DIR}/results/results_driving_custom/${result_tag}"

  echo ">>> [$(date '+%F %T')] planner=${planner} env=${env_name} routes=${routes_dir} llm_port=${llm_port} results_tag=${result_tag}"
  conda run -n "${env_name}" python "${EVAL_SCRIPT}" \
    --routes-dir "${routes_dir}" \
    --planner "${planner}" \
    --llm-port "${llm_port}" \
    --results-tag "${result_tag}"

  local metric_line
  metric_line=$(metrics_from_result_dir "${result_dir}")
  IFS=',' read -r driving_score route_completion <<< "${metric_line}"
  append_csv_row "${global_run}" "${category}" "${category_run}" "${scenario}" "${planner}" "${result_dir}" "${driving_score}" "${route_completion}"
}

run_scenario() {
  local scenario=$1
  local category=$2
  local category_run=$3
  local global_run=$4
  local routes_dir="${BENCH_DIR}/${scenario}/routes"

  if [[ ! -d "${routes_dir}" ]]; then
    echo "Routes directory missing: ${routes_dir}" >&2
    return 1
  fi

  for planner in "${PLANNERS_COLM[@]}"; do
    if has_valid_results "${scenario}" "${planner}"; then
      echo "    [SKIP] ${scenario} + ${planner}: already has valid results in CSV"
      continue
    fi
    local port=8888
    [[ "${planner}" == "colmdriver" ]] && port=8889
    local tag="${scenario}__run${global_run}__${planner}"
    run_planner colmdrivermarco2 "${planner}" "${routes_dir}" "${port}" "${tag}" "${category}" "${scenario}" "${category_run}" "${global_run}"
  done

  for planner in "${PLANNERS_B2D[@]}"; do
    if has_valid_results "${scenario}" "${planner}"; then
      echo "    [SKIP] ${scenario} + ${planner}: already has valid results in CSV"
      continue
    fi
    local tag="${scenario}__run${global_run}__${planner}"
    run_planner b2d_zoo "${planner}" "${routes_dir}" 8888 "${tag}" "${category}" "${scenario}" "${category_run}" "${global_run}"
  done
}

echo "Using CARLA_ROOT=${CARLA_ROOT}"
echo "Discovered categories (${#categories[@]}): ${categories[*]}"

global_run_counter=0

while :; do
  had_run=false
  for cat in "${categories[@]}"; do
    read -ra list <<< "${scenario_lists[$cat]}"
    idx=${next_idx[$cat]}
    if (( idx < ${#list[@]} )); then
      scenario=${list[$idx]}
      next_idx[$cat]=$((idx + 1))
      global_run_counter=$((global_run_counter + 1))
      category_run=$((idx + 1))
      had_run=true
      printf "\n==== Running %s (category %s, category run %d, global run %d) ====\n" "${scenario}" "${cat}" "${category_run}" "${global_run_counter}"
      run_scenario "${scenario}" "${cat}" "${category_run}" "${global_run_counter}"
    fi
  done

  [[ "${had_run}" == true ]] || break
done

echo "All scenarios completed."
