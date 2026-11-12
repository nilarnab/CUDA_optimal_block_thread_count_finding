# 1D pilot results

COMPLETE

- status: COMPLETE
- heavy_protocol_revision: 1
- planned_trials: 5120
- recorded_trials: 5120
- failed_trials: 0
- missing_trials: 0
- contexts: 32
- actions: 512
- valid_actions: 512
- usable_reward_rows: 512
- reference_seconds_contexts: 12
- complete_contexts: 32
- noisy_actions_over_5pct: 3
- successful_calibrations: 92
- failed_or_memory_limited_calibrations: 12
- kernels_without_selected_shapes: []
- seconds_target_unreached: ['vector_add', 'periodic_tile', 'repeat_vector', 'matrix_transpose', 'variance']

Single-launch CUDA event times; different protocol from v1/v2. Large outputs use 64 CPU samples plus all-output finite and guard checks. Invalid replay latencies are null. Net speedups use an assumed prediction cost, not an executed model. Seconds contexts are counted using measured reference medians. Inspect calibration.json for unreached targets and probe failures.
