# Scenario Generator

This package is streamlined around JSON schema generation and the core scenario pipeline.
Natural-language entry is still supported for running a single description through the pipeline.

## How to run

- Benchmark/categories (schema → pipeline → CARLA): `python tools/run_audit_benchmark.py ...`
- Single scenario from natural language: `python scenario_generator/run_scenario_pipeline.py --description "Vehicle 1 ..."`

## Where capabilities and categories live

- All topology types, map requirements, and category definitions are in `scenario_generator/scenario_generator/capabilities.py`. Add new scenario categories there.

## Pipeline stages (what actually runs)

1. **Crop selection** (`run_crop_region_picker.py`): choose a crop in the target town. Roundabout category forces the hardcoded Town03 roundabout crop.
2. **Legal paths** (`generate_legal_paths`): enumerate candidate paths inside the crop. Roundabout runs in roundabout_mode with deeper search; others use standard settings.
3. **Path picking** (`run_path_picker.py`): LLM picks one path per ego from the candidates using the scenario description/schema.
4. **Repair/refine**:
   - Audit runner (`tools/run_audit_benchmark.py`) retries schema generation on validation errors and can rerun pipeline attempts.
   - Path refinement (`run_path_refiner.py`) adjusts spawn/timing/lane changes; **skipped for roundabout** to avoid synthetic cuts.
5. **Object placement** (`run_object_placer.py`): place static props/NPCs per schema (or inferred) and category rules.
6. **Routes/CARLA (optional)**: `convert_scene_to_routes.py` emits CARLA route XMLs; audit runner will execute CARLA when configured.
