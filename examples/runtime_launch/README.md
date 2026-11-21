# Build and run a CUDA kernel with `toolcall`

`toolcall` encodes the kernel at build time and links a block-size selector
into the CUDA program. At runtime, the default native C++ scorer runs on the
CPU. It selects a block size, calculates the grid, and returns both to the
host launcher. The GPU runs the user’s CUDA kernel.

## Build and run vector add

From the repository root on Linux, activate the project’s Python environment
with PyTorch installed. You also need a host C++ compiler, Clang with CUDA
support, NVCC, and the original vocabulary that matches the adaptive
checkpoint.

The defaults use the CUDA2 target (`sm_75`),
`checkpoints/lscat_adaptive_h1_lambda1e-8_no-kl_ep120.best.pt`, and
`data/vocabulary.json`. Use the original training vocabulary for your
checkpoint. Pass `--checkpoint`, `--vocabulary`, or `--gpu-arch` to override
the defaults.

```bash
./toolcall examples/runtime_launch/toolcall_vector_add.cu -o build/vector_add

./build/vector_add 512
```

The argument `512` sets `M`. This example processes `M × M` elements and sends
`(M, 1)` to the default scorer. It prints the chosen grid and block and verifies
the output. [Example source](toolcall_vector_add.cu)

## Build and run accumulate

This example sums each of `M` groups across `N` input rows. It reads
`M × N` values, launches one thread per group, and checks the `M` results
against a CPU reference. The scorer receives `(M, N)` and the grid covers
`M` work items. [Example source](toolcall_accumulate.cu)

```bash
./toolcall examples/runtime_launch/toolcall_accumulate.cu -o build/accumulate
./build/accumulate 257 67
```

## Build and run softmax

This example takes `M` vectors of `N` logits and computes a separate
softmax for each vector. It subtracts the vector maximum before exponentiation
and checks the probabilities against a CPU reference. One GPU thread processes
one vector, so the scorer receives `(M, N)` and the grid covers `M` work items.
[Example source](toolcall_softmax.cu)

```bash
./toolcall examples/runtime_launch/toolcall_softmax.cu -o build/softmax
./build/softmax 257 67
```

## Use the API in your own kernel

Include [`toolcall.hpp`](../../src/launch/toolcall.hpp) and use
`model_output()` in the host code before launching your kernel.

```cpp
const auto [grid, block] =
    toolcall::model_output(my_kernel, {m, n, k, work_items});
my_kernel<<<grid, block>>>(inputs, output, work_items);
```

Pass the workload dimensions using the same meanings and order as the
checkpoint’s training data. The API accepts `{m, n, k, work_items}`. A
two-input checkpoint reads `m` and `n`. A three-input checkpoint also reads
`k`. Pass the actual number of elements to cover as `work_items`.
The current interface supports one CUDA kernel per input file, a host
`main()`, and a one-dimensional launch with one work item per thread.

Build any such file with `./toolcall my_kernel.cu -o my_program`. Use your
checkout’s defaults or pass the same options as above. Your host `main()`
defines the arguments to the resulting executable.

## How the build and launch fit together

1. **Build time.** Clang emits device LLVM IR. Python tokenizes the IR and uses
   the checkpoint’s FastText embeddings, LSTM, and learned gate to select
   which hidden states to retain. `toolcall` embeds those states and their
   gate values in the executable. It compiles the context encoder, dense
   scorer, attention, and their trained weights into `libtoolcall_model.so`.
2. **Runtime.** `model_output()` gets the active CUDA device and kernel launch
   limits. The CPU scorer combines the checkpoint’s workload dimensions with
   the embedded states and picks the highest-scoring legal block. For
   one retained state, the scorer uses that row’s scores directly. It
   calculates the grid as `ceil(work_items / block)`.

The build prints a unique artifact directory beside the executable.
Preserve this relative layout when moving the executable.

```text
build/
├── vector_add
└── .toolcall/
    └── kernel-…/
        ├── libtoolcall_model.so
        └── build.json
```

`build.json` records the checkpoint, GPU target, scorer backend, workload
columns, and `retained_states` for troubleshooting and benchmarking. The native
runtime uses the executable and library with the usual CUDA and system
libraries. The generated LLVM IR and headers are build artifacts.

## Benchmark against a fixed block

Build both programs for the same GPU target and run them with the same `M`.
The fixed program takes the block size as its second argument.

```bash
mkdir -p build
nvcc -std=c++17 -O3 -arch=sm_75 \
  examples/runtime_launch/bench_vector_add_fixed.cu -o build/vector_add_fixed

./toolcall examples/runtime_launch/bench_vector_add_model.cu -o build/vector_add_model

./build/vector_add_fixed 512 256
./build/vector_add_model 512
```

Both programs verify the output and report medians over 51 trials after
warm-up. With device buffers already allocated and populated, `gpu_us` times
the GPU kernel, `selection_us` times the CPU launch decision, and `total_us`
measures host time from the start of that decision through kernel completion.
Divide the fixed program’s `total_us` by the model program’s `total_us` to
measure speedup including selection.

### Accumulate and softmax

Each pair uses the same CUDA kernel, input data, and timing code. Build the
fixed-block baselines directly with NVCC. Build the model variants with
`toolcall` so it can encode the kernel and link the scorer before NVCC runs.

```bash
mkdir -p build
nvcc -std=c++17 -O3 -arch=sm_75 examples/runtime_launch/bench_accumulate_fixed.cu -o build/accumulate_fixed
nvcc -std=c++17 -O3 -arch=sm_75 examples/runtime_launch/bench_softmax_fixed.cu -o build/softmax_fixed
./toolcall examples/runtime_launch/bench_accumulate.cu -o build/accumulate_model
./toolcall examples/runtime_launch/bench_softmax.cu -o build/softmax_model

./build/accumulate_fixed 2051 67 256
./build/accumulate_model 2051 67
./build/softmax_fixed 2051 67 256
./build/softmax_model 2051 67
```

Use the same `M` and `N` for both programs in each pair. The fixed
executables take the block size as a third argument. Compare `gpu_us` for
kernel execution and `total_us` for latency including block selection.

### Collect a block-size sweep

Run from the repository root after building the two softmax benchmarks above.

```bash
python -m pip install wandb matplotlib
wandb login
python scripts/collect_launch_benchmarks.py --model-bin build/softmax_model \
  --wandb --plot results/launch_benchmarks/softmax_fine.png
```

The defaults cover `150051 1500`, `180051 1500`, `150051 2000`, and
`200051 2000`. Softmax allocates two float arrays on the GPU and two on
the host, up to about 3.2 GB of array storage on each for the largest shape.
Each shape runs the model once and each fixed block once by default.
The fixed sweep covers 64 through 1024 threads in steps of 64 and adds
32-thread steps within 128 threads of the model's choice. For example,
a prediction of 576 adds 480, 544, 608, and 672 to the sweep.
Use `--blocks` to test a custom list plus the model's chosen block.

The timestamped CSV under `results/launch_benchmarks/` contains one row
per completed binary run. `total_us`, `gpu_us`, and `selection_us`
are the benchmark's separate medians over 51 trials. `process_wall_s`
includes startup, allocations, copies, warm-up, all trials, and verification.
Local diamonds mark model choices at their actual block sizes. W&B includes
a measurement table, the CSV, and charts with labeled model reference lines.

For toolcall executables, `retained_states` comes from the matching
`build.json` and `full_states` counts the pre-gate token states in
`kernel.ll`. `state_reduction_pct` is the percentage of rows removed.
If the collector cannot find the artifact directory from the executable,
pass `--build-json /path/to/.toolcall/kernel-ID/build.json`.

To use other binaries, supply the argument list for each size. The
collector appends the block size only for the fixed binary. Both programs
must print the verified summary and timing fields used by these examples.

```bash
python scripts/collect_launch_benchmarks.py --kernel vector_add \
  --model-bin build/vector_add_model --fixed-bin build/vector_add_fixed \
  --size 8192 --size 10240 --size 12288 --blocks 256 512 576 640 \
  --csv results/launch_benchmarks/vector_add.csv --wandb
```

Run without `--wandb` or `--plot` for a CSV-only collection with no
additional Python packages. Use `--wandb-project` and `--wandb-entity`
to change the upload destination.

### Linear layer

The [linear layer benchmark](bench_linear_layer.cu) uses the
[LS-CAT kernel](https://github.com/NTNU-HPC-Lab/LS-CAT/blob/83a50786eee2de633be477819c044d2c9a1ad076/data/kernels/6580/20/linearLayerForward.cu)
to compute an `M × N` output with reduction width `K`. The model receives
`(M, N)` and selects the x block size. The launch has `ceil(N / block)`
blocks in x and `M` rows in y for both binaries.

```bash
mkdir -p build
nvcc -std=c++17 -O3 -arch=sm_75 examples/runtime_launch/bench_linear_layer_fixed.cu -o build/linear_layer_fixed
./toolcall --aot examples/runtime_launch/bench_linear_layer.cu -o build/linear_layer_aot

./build/linear_layer_aot 1600 1600 8192
./build/linear_layer_fixed 1600 1600 8192 256
```

The model build uses learned gating with the default checkpoint.
Run the same block sweep as for softmax with three arguments per size.

```bash
python scripts/collect_launch_benchmarks.py --kernel linear_layer \
  --model-bin build/linear_layer_aot --fixed-bin build/linear_layer_fixed \
  --size 1600 1600 8192 --size 2240 2240 8192 \
  --size 4096 4096 8192 --size 6144 6144 6144 \
  --wandb --plot results/launch_benchmarks/linear_layer.png
```

The largest shape uses about 432 MiB of GPU buffer space. The benchmark
checks all outputs for finite values, compares small cases fully against
a CPU reference, and checks up to 64 CPU reference values in large cases.
The timing and gate-count columns are the same as in the other benchmarks.
`grid` reports the x dimension in the CSV.

### ComputeOrientations

The [orientation benchmark](bench_compute_orientations.cu) adapts the
[LS-CAT kernel](https://github.com/NTNU-HPC-Lab/LS-CAT/blob/83a50786eee2de633be477819c044d2c9a1ad076/data/kernels/10319/29/ComputeOrientations.cu)
to compute 32-bin orientation histograms from a 15 × 15 image patch per
keypoint. Each warp owns one keypoint and its shared memory. The arguments
are `KEYPOINTS IMAGE_SIDE`; the fixed binary also takes `FIXED_BLOCK`.
The model sees `(KEYPOINTS, IMAGE_SIDE)`. The grid covers one warp per
keypoint, including a partial last block when needed.

```bash
mkdir -p build
nvcc -std=c++17 -O3 -arch=sm_75 \
  examples/runtime_launch/bench_compute_orientations_fixed.cu -o build/compute_orientations_fixed
./toolcall --aot examples/runtime_launch/bench_compute_orientations.cu -o build/compute_orientations_aot

python scripts/collect_launch_benchmarks.py --kernel compute_orientations \
  --model-bin build/compute_orientations_aot --fixed-bin build/compute_orientations_fixed \
  --size 524288 4096 --size 1048576 4096 --size 2097152 4096 --size 4194304 4096 \
  --blocks 64 96 128 160 192 224 256 288 320 352 384 416 448 480 512 544 576 \
  --wandb --plot results/launch_benchmarks/compute_orientations.png
```

The explicit block list keeps the sweep within the kernel's launch limit.
A 576-thread block uses 47.25 KiB of shared memory. The largest input
uses 112 MiB of GPU buffers. Each run checks both angles against a CPU
reference for up to 64 keypoints and scans every output for invalid angles.
Build with the default learned gates to measure natural pruning in the CSV.

### Test a smaller kernel representation with the same checkpoint

```bash
./toolcall --retain-fraction 0.75 examples/runtime_launch/bench_softmax.cu \
  -o build/softmax_model_keep75
```

The build applies the checkpoint's learned gate first. The optional cap
then keeps the highest-gated rows up to 75% of that count, rounded down,
with the final LSTM state always retained and counted toward the limit.
The selected states stay in their original order. The cap works with
both the native scorer and `--aot`.

The printed artifact directory contains `build.json` with
`learned_retained_states` before the cap and `retained_states` after it.
The collector uses the latter as the number of rows scored at runtime.
Compare capped and uncapped binaries built with the same checkpoint and
backend using the same inputs.

## Optional PyTorch-compiled scorer

Add `--aot` to compile the existing Python scorer for the CPU with
AOTInductor. The CUDA program still calls the same `model_output()` function.

```bash
./toolcall --aot examples/runtime_launch/bench_vector_add_model.cu -o build/vector_add_aot

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ./build/vector_add_aot 512
```

Keep `scorer.pt2` beside `libtoolcall_model.so` in the printed artifact
directory and keep the build environment’s LibTorch libraries available.
The first prediction loads the package. Compare this run with the native
model benchmark at the same size. The benchmark warms up both scorers before
recording selection latency.
