# 1D LS-CAT collection: twelve kernels, sixteen block sizes

This standalone package combines the 1D kernels in `ls_cat_pilot` and
`ls_cat_pilot_heavy`. It reuses their checked sources and single-launch worker.

| Family | Original package | Input contract |
|---|---|---|
| vector_add | pilot | Two M*M FP32 arrays; M*M outputs |
| segmented_sum | both | M integer rows of K elements; M outputs |
| periodic_tile | pilot | Tile a 17*13 FP32 input to M*M outputs |
| repeat_vector | pilot | Repeat M FP32 values M times |
| matrix_transpose | pilot | Permute FP32 [1,M,M]; 1D thread indexing |
| mean | heavy | M FP32 channels of K values; batch 1 |
| variance | heavy | M FP32 channels of K values; means prepared before timing |
| nbody | heavy | M particles; force-only adaptation of NVIDIA all-pairs N-body |
| binarize_weights | LS-CAT 10194/2 | M filters of K weights; magnitude reduction then M*K signed outputs |
| binarize_input | LS-CAT 10194/1 | K channels by M positions; reduce channels then write K*M signed outputs |
| softmax | LS-CAT 10194/41 | M independent vectors of K values; temperature 1 |
| accumulate | LS-CAT 10194/15 | K groups of M values; strided reduction to M sums |

There are twelve distinct kernel functions: segmented_sum is shared. Matrix_add,
GEMM and SYRK have 2D thread indexing and are excluded. N-body is the existing
authored adaptation; the eleven LS-CAT kernel files are copied byte for byte.
`provenance.json` records their hashes and origin URLs. No extra kernels are
invented to increase the count. All four additions come from LS-CAT repository
group 10194, which also contains transpose, mean and variance; this is not
twelve independent source repositories. Keep that group together in splits.

## Exactly sixteen actions

Every context uses `(64,1,1), (128,1,1), ... (1024,1,1)`. The grid covers all
outputs or channels, or all particles for N-body. No grid search is performed.
The collector refuses a device/kernel that cannot support all sixteen blocks.
The GPU self-test exercises all twelve functions at every block size before
collection. Problem shape, precision and input data stay fixed across actions.

## Sizes and duration targets

The four elementwise/permutation kernels grow M through
`240,1024,2048,4096,8192,12288,16384,22528,32768`, subject to memory limits.
For example, vector_add at M=12288 needs 1.6875 GiB of device buffers, plus
128 guard bytes; at M=16384 it needs 3 GiB. These are candidate bounds.

Segmented sum, mean and variance vary M and K independently. K grows from
1024 through 16,777,216. At the larger K values M decreases, keeping input
storage near 1 GiB while increasing the length of each real reduction.
Accumulate uses the same M/K ladder but reads the input with a different stride.
Binarize weights, binarize input and softmax also grow K through 16,777,216,
with up to 134,217,728 input and output values each (1 GiB total plus guards at
the largest listed cases). They launch M logical threads, not M*K threads:
each thread performs its entire reduction and output passes. The CPU checks
use the corresponding contiguous or strided layouts. Softmax uses bounded
inputs in [-0.5,0.5] and a probability-relative tolerance; its original helper's
integer maximum temporary is preserved. Long positive FP32 reductions in the
new kernels can also fail numerical checks; those probes remain excluded.
N-body grows from 8192 to at most 4,194,304 particles, with extra intermediate
sizes around its previously measured seconds range. All indexing/allocation
limits are checked in both Python and C++.

`full` selects the first reference shapes reaching 5, 50, 500 and 1000 ms,
deduplicating shapes that reach several targets. `seconds` selects the first
reference in 500–2000 ms. Calibration stops above 2000 ms. The reference is
always block 256 with its coverage grid. All sixteen actions are then measured,
including actions outside the reference's duration range.

`quick` targets 5 ms. In `quick`/`full`, a family that never reaches 5 ms retains
its slowest successful bounded probe as a labeled fast control. Memory-bound
kernels may stay below 5 ms on A100; larger sizes cannot guarantee seconds
within available memory. No dummy work, launch batches, or slower reference
blocks are used to manufacture long durations. Large variance reductions can
fail the existing numerical tolerance; these failures remain recorded and are
not relabeled as usable timings.

`duration_targets.json` reports targets reached using the final measured
reference medians, separately from calibration. `COMPLETE` certifies the
planned data and requested-family coverage, not that every family reached
one second. Inspect `seconds_target_unreached` and `fast_control_only`.

## Run separately on CUDA2 and A100

Requirements: Linux NVIDIA compute host, `nvcc`, compatible C++14 compiler,
Python 3.9+, bash and tar. No Python packages or downloads are needed on the
GPU host. Every invocation creates a separate source snapshot, result directory
and archive. GPU visibility is preserved; the device argument is a logical ID.

```bash
tar -xzf ls-cat-pilot-1d.tar.gz
cd ls_cat_pilot_1d
bash run.sh smoke 0
```

Once smoke passes on that GPU:

```bash
bash run.sh quick 0
bash run.sh full 0 --seeds 3 --repeats 3
```

Calibration runs independently on each GPU. Do not reuse CUDA2's selected sizes
as if they also met the timing target on A100. For only seconds-scale cases:

```bash
bash run.sh seconds 0 --kernels segmented_sum,mean,variance,nbody,binarize_weights,binarize_input,softmax,accumulate --repeats 3
```

Seconds mode returns INCOMPLETE when any requested family has no qualifying
shape; valid data and calibration failures are retained. `calibrate` saves only
probes and no block sweep. `--kernels` accepts any comma-separated subset of
the twelve IDs. Each probe has a 180-second worker limit; a complete context
has a 3600-second limit, configurable with `--probe-timeout`/`--worker-timeout`.

Default device memory cap is 2 GiB and at most 80% of current free memory.
On a GPU with sufficient free memory you can explicitly allow larger inputs:

```bash
bash run.sh full 0 --max-device-gib 4 --seeds 3 --repeats 3
```

Host buffers also use several GiB. The included Slurm template requests 16 GiB
host memory and twelve hours. From the allocated environment, with your actual
account and available modules:

```bash
sbatch --account=YOUR_ACTUAL_ACCOUNT --export=ALL,MODE=smoke submit.slurm
sbatch --account=YOUR_ACTUAL_ACCOUNT --export=ALL,MODE=full,SEEDS=3,REPEATS=3 submit.slurm
```

## Timing, validation and model cost

Each observation times exactly one kernel invocation with CUDA events, with
one first-use warmup per action. Allocation, transfers, CPU references and
output checks are excluded. Small outputs receive full CPU comparison;
larger outputs receive 64 deterministic sampled references plus a complete
finite/unwritten scan and guard checks. Sampling does not certify every large
output's numerical value. Timings are warm single launches, unlike the original
pilot's host-submitted launch batches.

The assumed prediction cost is 0.8 ms. A selection pays off only when
`reference_ms - selected_ms > 0.8`; a long duration alone does not establish
a tuning benefit. `heavy_action_table.csv` records net speedup and saved time
after this assumed cost. The model is not executed during collection.

## Output and training use

Return the entire printed result archive, including partial runs.
`trials.csv`, `measurements.csv` and `action_table.csv` preserve the original
pilot's columns and units. `heavy_*` files retain explicit M/N/K, stable base
family IDs and split groups. `contexts.json` maps exported aliases to original
sources. The legacy `size` is M, not an opaque context number. K and seed suffixes
avoid key collisions and do not create additional source families.

Use base source families/repository groups for data splits. In particular,
transpose, mean and variance share the original repository group. Keep GPU
identities separate unless hardware becomes an explicit model input. The
existing model's pilot-preparation adapter needs an extension to understand
these twelve source contracts and K; CSV header compatibility alone does not
make its original six-kernel preparation script applicable.

## Local checks

```bash
python3 -m unittest discover -s . -p 'test_*.py' -v
c++ -O2 -std=c++14 test_reference.cpp -o /tmp/lscat-1d-reference
/tmp/lscat-1d-reference
```

GPU compilation, full sixteen-action GPU self-tests and duration calibration
must run on CUDA2 and A100. The package does not claim GPU measurements from
local host checks or simulated-worker tests.
