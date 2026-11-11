# Heavy pilot results

COMPLETE

- status: COMPLETE
- heavy_protocol_revision: 1
- planned_trials: 2070
- recorded_trials: 2070
- failed_trials: 0
- missing_trials: 0
- contexts: 16
- actions: 207
- valid_actions: 207
- usable_reward_rows: 207
- reference_seconds_contexts: 2
- complete_contexts: 16
- noisy_actions_over_5pct: 0
- successful_calibrations: 38
- failed_or_memory_limited_calibrations: 1
- kernels_without_selected_shapes: []
- seconds_target_unreached: ['gemm', 'segmented_sum', 'mean', 'variance']

Single-launch CUDA event times; different protocol from v1/v2. Large outputs use 64 CPU samples plus all-output finite and guard checks. Invalid replay latencies are null. Net speedups use an assumed prediction cost, not an executed model. Seconds contexts are counted using measured reference medians. Inspect calibration.json for unreached targets and probe failures.
