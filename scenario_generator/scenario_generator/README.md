# Scenario Generator (Schema-First)

This package is streamlined around JSON schema generation and the core scenario pipeline.
It does not include the legacy natural-language auto-generation stack.

## Entry points

- Schema generation + pipeline:
  - `python scenario_generator/generate_schema_scenarios.py --output-dir log_schema`
  - `python scenario_generator/generate_schema_scenarios.py --categories "Unprotected Left Turn" --template-only`

- Direct pipeline run (existing scenario text files or single description):
  - `python scenario_generator/run_scenario_pipeline.py --scenarios-file scenario_generator/scenarios.txt --out-dir log`
  - `python scenario_generator/run_scenario_pipeline.py --town Town05 --description "Vehicle 1 ..." --out-dir log_single`

## What the pipeline does

1. Crop region selection from town nodes
2. Legal path generation in the crop
3. Path picking with the scenario description
4. Path refinement (spawn points, lane changes, timing)
5. Object placement (static props, pedestrians, etc.)
6. Optional route conversion and CARLA alignment

## Outputs

Per scenario output directory contains:
- `scenario_input.txt`
- `legal_paths_prompt.txt`
- `legal_paths_detailed.json`
- `path_picker_prompt.txt`
- `picked_paths_detailed.json`
- `picked_paths_refined.json`
- `scene_objects.json`
- `scene_objects.png`

Schema generation additionally writes:
- `scenario_spec.json`
- `scenario_description.txt`

## Programmatic usage

If you need to run the pipeline from Python, use:
- `scenario_generator/scenario_generator/pipeline_runner.py`

This provides `PipelineRunner` (shared model, in-process run) and `GenerationLogger`.
