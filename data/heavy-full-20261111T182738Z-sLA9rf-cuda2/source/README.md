# LS-CAT pilot: heavy workloads

Run it like the first ls_cat_pilot. The three primary CSV files have exactly the first pilot's headers, column order, millisecond units, and aggregation/reward rules. Extra workload details live in detail_json, contexts.json, and optional heavy_* diagnostics.

This is a separate package: extracting it creates ls_cat_pilot_heavy and does not overwrite your first package or its results. No Python packages, downloads, graphics libraries, or administrator changes are needed on the GPU machine.

## Run

Requirements: Linux NVIDIA compute host, compatible CUDA toolkit/driver, nvcc, a C++14 compiler, Python 3.9+, bash, tar, tee. Load the CUDA/compiler/Python environment already available on your machine.

```bash
tar -xzf ~/ls-cat-pilot-heavy.tar.gz -C ~
cd ~/ls_cat_pilot_heavy
bash run.sh smoke 0
```

The last argument is the CUDA logical device index. CUDA_VISIBLE_DEVICES is preserved; on a scheduler-allocated single GPU, use 0.

Inspect summary.md, COMPLETE, host_checks.log, gpu_selftest.log, build.log, and progress.log. Once smoke succeeds:

```bash
bash run.sh quick 0
bash run.sh full 0
```

Return the entire generated runs/heavy-*.tar.gz archive. The launcher snapshots sources, records the environment, compiles for the actual GPU, and archives complete or partial results on normal shell exit.

Full collection can take hours. A reference duration is not a worst-case action duration. Do not start full until quick succeeds.

## Workloads

| Kernel ID | Work | Input dimensions |
|---|---|---|
| gemm | Unchanged LS-CAT matrixMultiply | M-by-K times K-by-N; K is no longer fixed at 32 |
| segmented_sum | Unchanged LS-CAT gpuSum | M independent rows, K integers per row |
| mean | Unchanged LS-CAT mean_kernel | M filters, K values each, batch 1 |
| variance | Unchanged LS-CAT variance_kernel | M filters, K values each; precomputed mean, batch 1 |
| syrk | Extracted original PolyBench syrk_kernel | C = A A^T + .5 C; M=N and reduction K |
| nbody | Authored adaptation of NVIDIA's tiled all-pairs algorithm | M particles; fixed-input force calculation, not a full simulation step |

provenance.json records original file SHA256 and Git blob identities. Full upstream files are retained in upstream/. The N-body adaptation uses shared-memory tiles, supports partial tiles and grid-stride target coverage, and does not integrate positions or mutate velocity. It is explicitly not an unchanged NVIDIA sample. Monte Carlo is not part of this package.

Matrix ladders grow through M=N=8192 and K=16384. Reduction ladders grow K through 1048576. N-body doubles progressively from 8192 through at most 4194304 bodies, stopping when calibration reaches the target or a limit. These maxima are candidate bounds, not a promise that collection will use every shape.

## Modes

| Mode | Shape selection | Input seeds | Repetitions per action |
|---|---|---:|---:|
| smoke | Tiny correctness workloads for all six kernels | 1 | 1 |
| quick | First calibrated shape at least 5 ms per kernel, if reachable | 1 | 3 |
| full | Up to four shapes per kernel: first reaching 5, 50, 500, and 1000 ms | 1 | 10 |
| seconds | First calibrated shape in 1000–5000 ms per kernel, if reachable | 1 | 3 |
| calibrate | Calibration only; no block sweep | 1 | No trial dataset |

Full deduplicates shapes that satisfy several targets simultaneously. Reference configurations are coverage grids with block 256x1 for linear kernels or 16x16 for matrix kernels, with a nearest legal coverage block if the device excludes that shape. Baseline selection for the first-version action table remains the original largest-block rule; calibration does not change that reward definition.

Actual runtime determines the tier. Cases below target are recorded in calibration.json, not silently presented as seconds-long kernels. Calibration stops growing a family above 5 seconds or after its ladder is exhausted. A family may never reach one second within the reviewed bounds. seconds exits incomplete if no family supplies a suitable shape; inspect calibration.json and heavy_summary.json.

Run only the longer candidates, or change the sample count explicitly:

```bash
bash run.sh seconds 0 --kernels gemm,syrk,nbody
bash run.sh full 0 --kernels gemm,variance,nbody --repeats 3
bash run.sh full 0 --seeds 3 --repeats 3
```

Options are passed directly to collect.py. --kernels takes comma-separated IDs without spaces. --seeds is 1–10; --repeats is 1–20. Every run gets its own directory.

## Block/grid sweep

The default mirrors the first pilot: vary blocks and derive coverage grids. Linear kernels use (64,1), (128,1), ... (1024,1), filtered by actual GPU/kernel limits. GEMM and SYRK use nine valid 2D shapes: (8,8), (16,8), (8,16), (16,16), (32,8), (8,32), (32,16), (16,32), (32,32).

There are 82 planned smoke trials on a device accepting every shape. Full counts depend on calibration and GPU resource limits; the collector prints the actual plan before measurement.

Optional expanded grid search:

```bash
bash run.sh quick 0 --grid-search expanded
```

This adds overcoverage grids and legal reshapes; N-body also tests grids capped at 1, 2, 4, or 8 times the GPU SM count. Actual duplicate tuples are removed. Only N-body can use reduced grids because its loop still covers all particles. Expanded smoke includes a one-block N-body test action. Source legality checks reject skipped work.

## Timing and correctness

Each observation times ONE kernel invocation using CUDA events on a private stream. kernel_ms and batch_ms are equal and iterations=1. There is one untimed warmup the first time an action is encountered in that worker. Action order and repetitions are randomized. Allocations and inputs are reused within a kernel/shape/seed worker.

Output resets, input/output copies, CPU setup/reference generation, and validation are outside the CUDA event interval. SYRK starts each invocation from C=.5, with alpha=1 and beta=.5, so repeated runs do not accumulate a different task. N-body uses fixed positions, mass 1/M, and softening-squared .01.

Small workloads with at most 4096 output elements receive a FULL independent CPU reference check. Larger workloads receive 64 deterministic CPU reference samples, including boundary positions, plus a complete output scan for nonfinite/unwritten values and sixteen guard elements on each side. Sampling does not establish correctness of every large-output value. correctness_scope and checked_outputs are recorded explicitly. SYRK's initialized interior means its full finite scan alone cannot detect an unwritten element; sampled/full reference checks provide the numerical check.

The launcher runs CPU-emulated checks and a separate GPU self-test BEFORE collection. The GPU self-test fully checks all small outputs for each function, including a capped and partially populated N-body tile. It stops collection on failure.

Floating comparisons use absolute-plus-relative tolerances: 3e-5 + 1e-3*abs(reference), or 2e-4 + 2e-3*abs(reference) for FP32 N-body. Integer sums are exact. max_abs_error refers to the checked CPU reference outputs. The N-body interaction arithmetic has also been checked on CPU at 1048576 and 4194304 bodies; this is not a GPU benchmark.

These are warm single-launch measurements. The first pilot used repeated host-submitted launch batches. The CSV format is compatible; the timing methods are different and must remain labeled in metadata when combining datasets. elapsed_wall_s is per-observation worker time, including its reset/validation and any first-use warmup, but excluding worker startup and shared initial setup. It is not model inference latency.

## Limits and incomplete runs

Device buffers are capped at 2 GiB by default and 80% of currently free GPU memory. Limits are checked before allocations. Host memory also stores inputs and output buffers; the scheduler example requests 16 GiB.

```bash
bash run.sh full 0 --max-device-gib 3 --worker-timeout 7200
```

Increase memory only if available. --worker-timeout limits a WHOLE shape/seed context, not one kernel. It defaults to 3600 seconds; --probe-timeout defaults to 180 seconds. Very slow block choices can exceed the context limit even when the reference is under five seconds. Completed worker records are flushed immediately. Failures remain in CSV; incomplete actions do not become usable reward labels. No automatic retry/resume is implemented.

Optional one-second nvidia-smi telemetry observes the selected physical GPU. It is diagnostic, may include other users, and is not a per-kernel energy measurement. Model-free calibration and benchmark overhead are not the selector's deployment latency.

Graceful completion archives the run. Hard termination can leave an unarchived directory or records present only in workers/*.jsonl; preserve them. Do not delete partial data.

## First-version data compatibility

The primary files are:

- trials.csv: EXACT first-version columns and column order.
- measurements.csv: EXACT first-version columns and full/failed/incomplete status semantics.
- action_table.csv: EXACT first-version columns, largest-block baseline, reward_log_vs_default and regret definitions.
- planned.json, metadata.json, summary.json, summary.md, COMPLETE/INCOMPLETE, environment/build logs, and source snapshot.

The original first-pilot analyzer is included unchanged as legacy_analyze.py. It produces the primary aggregate files. FORMAT.md lists every column and explains identifiers. The package tests also run that analyzer against synthetic end-to-end output.

Because the first format has only one size field, size remains a REAL dimension: M/output rows for matrix/reduction kernels, or body count for N-body. It is never an opaque context number. Additional dimensions are encoded in the kernel ID suffix where necessary and stored explicitly in detail_json and contexts.json. For example, gemm__k4096 at size=2048 denotes a square 2048-by-4096 times 4096-by-2048 multiply. Smoke's rectangular example is gemm__k23__n19, size=17.

With optional multiple seeds or expanded grids, kernel IDs get __sINDEX and __grid_POLICY suffixes. This avoids duplicate first-version grouping keys. Treat these aliases as the SAME source family for dataset splitting. Join contexts.json on (kernel_id,size) to find base_kernel_id, M/N/K, actual seed, source path/hash, split_group, and grid policy.

Additional heavy_* CSV/JSON diagnostics retain all shapes, seeds, and grids together under base IDs. heavy_action_table.csv uses the stronger fixed coverage reference and includes net-speedup estimates with an ASSUMED 0.8 ms prediction cost. No model actually runs during collection. replay.jsonl contains the base-context action sets and null latency for invalid actions. Use these optional files if evaluating joint grid search; v1 action_table.csv compares blocks within each grid-policy alias.

CSV readers can keep the first-version column selection. A trained model or feature extractor still needs to understand the new source functions and M/N/K contracts. Column compatibility does not make new kernels identical to the old six, or make short and long datasets statistically interchangeable.

## NYU Torch

Extract on compute-accessible storage, load the CUDA/compiler/Python environment, and submit from this directory using your real allocation account:

```bash
sbatch --account=YOUR_ACTUAL_ACCOUNT --export=ALL,MODE=smoke submit.slurm
sbatch --account=YOUR_ACTUAL_ACCOUNT --export=ALL,MODE=full submit.slurm
```

The template requests one GPU, two CPU cores, 16 GiB RAM, and twelve hours. Adjust time/account/constraints to the resources available to you. It does not guess module names or a partition.

## Validation status

CPU-emulated original-function checks, reference/guard failure checks, and address/undefined-behavior sanitizer checks have passed locally. Python tests cover provenance, dimensions, legal actions, invalid protocol records, single-launch timing, first-version header/key compatibility, partial/missing observations, and end-to-end smoke/seconds collection through a simulated worker.

The CUDA worker has NOT been compiled or run on a GPU in this local session. Run smoke on the actual GPU first; its build and GPU self-test logs are retained for diagnosing any toolkit/device issue. No seconds-scale duration is guaranteed before calibration.
