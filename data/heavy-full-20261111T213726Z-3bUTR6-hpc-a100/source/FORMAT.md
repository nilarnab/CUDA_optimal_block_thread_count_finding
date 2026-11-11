# Compatibility with the first ls_cat_pilot

The three main CSV headers below are exactly the originals. There are no appended main-table columns. Row count and kernel IDs change because the workloads change.

## trials.csv

```text
kernel_id,repo_id,source_sha256,size,block_x,block_y,round,seed,status,kernel_ms,batch_ms,iterations,grid_x,grid_y,grid_z,input_elements,output_elements,max_abs_error,warmup_launches,elapsed_wall_s,error,detail_json
```

- kernel_id: base source family plus a fixed K suffix where needed, rectangular N suffix if different from size, and optional grid/seed suffixes. These are identifiers, not independent programs.
- repo_id: original numeric LS-CAT repository ID; polybench_acc or nvidia_adapted for non-LS-CAT sources. Do not cast the entire column to integers.
- source_sha256: the executed source function file (or N-body adaptation), not the binary hash. Full provenance and source snapshots are retained.
- size: actual M/output-row count, or body count. K is not replaced with an arbitrary size ID.
- block/grid: actual launch dimensions, with grid_z=1.
- kernel_ms, batch_ms: milliseconds for the SAME single timed invocation; iterations=1. The normalization identity is unchanged.
- seed: actual input seed, constant within each shape/seed context.
- status: ok for accepted observations; failed for timeout/missing/incorrect/protocol observations. Empty timing fields are missing data, never zero-time results.
- max_abs_error: maximum discrepancy among outputs checked against the independent CPU reference. It is not a guarantee about every large-output element.
- warmup_launches: 1 on first use of an action within that worker; 0 thereafter.
- elapsed_wall_s: per-observation reset/warmup/timing/validation time in seconds. Initial shared setup and worker process startup are excluded.
- detail_json: typed M/N/K, base_kernel_id, context and trial IDs, input-seed index, measured runtime tier, correctness scope, checked-output count, scaled error, single-launch timing label, and query/reference costs.

## measurements.csv

```text
kernel_id,size,block_x,block_y,repeats_ok,repeats_failed,median_ms,mad_ms,relative_mad,status
```

Grouping key and full/failed/incomplete meanings are the first version's. relative_mad is a ratio, not a percentage.

## action_table.csv

```text
kernel_id,size,block_x,block_y,median_ms,relative_mad,baseline_block_x,baseline_block_y,reward_log_vs_default,regret
```

The baseline is the fully valid largest block, with the same tie-breaking as the first version. reward_log_vs_default=log(default_median/action_median); regret=action_median/best_valid_median-1. Only groups with every planned repetition present and successful are included.

## Additional metadata, outside the original schema

contexts.json is a list keyed by (kernel_id,size). It supplies base_kernel_id, M/N/K, split_group, input seed/index, grid policy, runtime tier, and source location. All aliases of a source family must stay together when evaluating unseen code.

heavy_trials.csv, heavy_planned.json, heavy_measurements.csv, heavy_action_table.csv, and heavy_summary.* are richer companion records. They preserve joint block/grid contexts and compare to a static coverage reference. replay.jsonl uses those base contexts and nulls invalid action latencies. They do not replace the primary first-format files.

The original CSV format does not imply the original measurement protocol: these heavy workloads use single launches, and large outputs use sampled CPU checks. Retain metadata and detail_json when combining runs.
