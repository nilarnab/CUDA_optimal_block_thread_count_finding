# Build and run a CUDA kernel with `toolcall`

`toolcall` encodes the kernel at build time and links a block-size selector
into the CUDA program. By default, it compiles the CPU scorer with PyTorch
AOTInductor. The scorer selects a block size, and the host launcher uses
the returned grid and block dimensions to run the CUDA kernel on the GPU.
Pass `--native` at build time to use the C++ CPU scorer.

## Build and run softmax

From the repository root on Linux, activate the project’s Python environment
with PyTorch installed. The default build needs PyTorch with AOTInductor
packaging support. You also need a host C++ compiler, Clang with CUDA support,
NVCC, and the original vocabulary that matches the adaptive checkpoint.

The defaults use the CUDA2 target (`sm_75`),
`checkpoints/lscat_adaptive_h1_lambda1e-8_no-kl_ep120.best.pt`, and
`data/vocabulary.json`. Use the original training vocabulary for your
checkpoint. Pass `--checkpoint`, `--vocabulary`, or `--gpu-arch` to override
the defaults.

This example takes `M` vectors of `N` logits and computes a separate
softmax for each vector. It subtracts the vector maximum before exponentiation
and checks the probabilities against a CPU reference. One GPU thread processes
one vector, so the scorer receives `(M, N)` and the grid covers `M` work items.
[Standalone entry point](softmax/toolcall_softmax.cu)

```bash
softmax=examples/runtime_launch/softmax
./toolcall "$softmax/softmax_model.cu" \
  --link-source "$softmax/toolcall_softmax.cu" -o build/softmax
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
The current interface encodes one CUDA kernel from the primary source file.
The program needs a host `main()` and can use the returned configuration for
a one-dimensional launch with one work item per thread.

Build a single-file program with `./toolcall my_kernel.cu -o my_program`.
If your host `main()` lives in a separate file, pass it with
`--link-source main.cu`. `toolcall` links that file into the program and
uses only the primary source for LLVM IR and model input. Your host
`main()` defines the executable's arguments.

## How the build and launch fit together

1. **Build time.** Clang emits device LLVM IR. Python tokenizes the IR and uses
   the checkpoint’s FastText embeddings, LSTM, and learned gate to select
   which hidden states to retain. `toolcall` embeds every retained state and
   its gate value in the executable. By default, it compiles the context
   encoder, dense scorer, attention, and their trained weights into
   `scorer.pt2` and builds `libtoolcall_model.so` to load it.
2. **Runtime.** `model_output()` gets the active CUDA device and kernel launch
   limits. The CPU scorer combines the checkpoint’s workload dimensions with
   the embedded states and picks the highest-scoring legal block. For
   one retained state, the scorer uses that row’s scores directly. It
   calculates the grid as `ceil(work_items / block)`.

The build prints a unique artifact directory beside the executable.
Preserve this relative layout when moving the executable.

```text
build/
├── softmax
└── .toolcall/
    └── kernel-…/
        ├── libtoolcall_model.so
        ├── scorer.pt2
        └── build.json
```

`build.json` records the checkpoint, GPU target, scorer backend, workload
columns, and `retained_states`, the count selected by the checkpoint's
learned gate. Keep `scorer.pt2` beside `libtoolcall_model.so` and keep the
build environment’s LibTorch libraries available. The first prediction loads
the package. The generated LLVM IR and headers are build artifacts.

## Benchmark source layout

Each kernel has its own folder and separate model and fixed executables.

| Kernel | CUDA implementation | Model launcher | Model benchmark | Fixed launcher | Fixed benchmark |
| --- | --- | --- | --- | --- | --- |
| Softmax | [Kernel](softmax/kernel.cuh) | [Launcher](softmax/softmax_model.cu) | [Benchmark](softmax/bench_softmax_model.cu) | [Launcher](softmax/softmax_fixed.cu) | [Benchmark](softmax/bench_softmax_fixed.cu) |
| ComputeOrientations | [Kernel](compute_orientations/kernel.cuh) | [Launcher](compute_orientations/compute_orientations_model.cu) | [Benchmark](compute_orientations/bench_compute_orientations_model.cu) | [Launcher](compute_orientations/compute_orientations_fixed.cu) | [Benchmark](compute_orientations/bench_compute_orientations_fixed.cu) |
| Matrix multiply and ReLU | [Kernel](matrix_multiply_and_relu/kernel.cuh) | [Launcher](matrix_multiply_and_relu/matrix_multiply_and_relu_model.cu) | [Benchmark](matrix_multiply_and_relu/bench_matrix_multiply_and_relu_model.cu) | [Launcher](matrix_multiply_and_relu/matrix_multiply_and_relu_fixed.cu) | [Benchmark](matrix_multiply_and_relu/bench_matrix_multiply_and_relu_fixed.cu) |

The model launcher calls `toolcall::model_output()` and launches the kernel
with the selected configuration. The fixed launcher validates and uses
`FIXED_BLOCK`, the last argument to the fixed benchmark. Each folder's
`launch.hpp` declares these functions and its `benchmark.hpp` shares input
setup, timing, and output checks between the two benchmarks.

Build the model launcher with `toolcall` and pass its benchmark with
`--link-source`. Build the fixed launcher and its benchmark together with
`nvcc`.

## Benchmark against a fixed block

Build the softmax benchmarks for the same GPU target and use the same
`M` and `N` for both runs. The fixed binary takes the block size as a third
argument.

```bash
mkdir -p build
softmax=examples/runtime_launch/softmax
nvcc -std=c++17 -O3 -arch=sm_75 \
  "$softmax/softmax_fixed.cu" "$softmax/bench_softmax_fixed.cu" -o build/softmax_fixed
./toolcall "$softmax/softmax_model.cu" \
  --link-source "$softmax/bench_softmax_model.cu" -o build/softmax_model

./build/softmax_fixed 2051 67 256
./build/softmax_model 2051 67
```

Both programs verify the output and report medians over 51 trials after
warm-up. With device buffers already allocated and populated, `gpu_us` times
the GPU kernel, `selection_us` times the CPU launch decision, and `total_us`
measures host time from the start of that decision through kernel completion.
Divide the fixed program’s `total_us` by the model program’s `total_us` to
measure speedup including selection.

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

Run without `--wandb` or `--plot` for a CSV-only collection with no
additional Python packages. Use `--wandb-project` and `--wandb-entity`
to change the upload destination.

### Matrix multiply and ReLU

The [ReLU benchmark](matrix_multiply_and_relu/bench_matrix_multiply_and_relu_model.cu) adapts the
[LS-CAT kernel](https://github.com/NTNU-HPC-Lab/LS-CAT/blob/83a50786eee2de633be477819c044d2c9a1ad076/data/kernels/6580/8/matrixMultiplyAndRelu.cu)
to compute a matrix product with row bias and ReLU. It accepts `M N K`
and uses 32-column tiles. Its two outputs are the preactivation and
ReLU result. The largest example uses about 576 MiB of GPU buffers.

```bash
mkdir -p build
relu=examples/runtime_launch/matrix_multiply_and_relu
nvcc -std=c++17 -O3 -arch=sm_75 \
  "$relu/matrix_multiply_and_relu_fixed.cu" \
  "$relu/bench_matrix_multiply_and_relu_fixed.cu" -o build/matrix_multiply_and_relu_fixed
./toolcall "$relu/matrix_multiply_and_relu_model.cu" \
  --link-source "$relu/bench_matrix_multiply_and_relu_model.cu" -o build/matrix_multiply_and_relu_aot

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/collect_launch_benchmarks.py --kernel matrix_multiply_and_relu \
  --model-bin build/matrix_multiply_and_relu_aot \
  --fixed-bin build/matrix_multiply_and_relu_fixed \
  --size 1600 1600 8192 --size 2240 2240 8192 \
  --size 4096 4096 8192 --size 6144 6144 6144 \
  --wandb --plot results/launch_benchmarks/matrix_multiply_and_relu.png
```

### ComputeOrientations

The [orientation benchmark](compute_orientations/bench_compute_orientations_model.cu) adapts the
[LS-CAT kernel](https://github.com/NTNU-HPC-Lab/LS-CAT/blob/83a50786eee2de633be477819c044d2c9a1ad076/data/kernels/10319/29/ComputeOrientations.cu)
to compute 32-bin orientation histograms from a 15 × 15 image patch per
keypoint. Each warp owns one keypoint and its shared memory. The arguments
are `KEYPOINTS IMAGE_SIDE`; the fixed binary also takes `FIXED_BLOCK`.
The model sees `(KEYPOINTS, IMAGE_SIDE)`. The grid covers one warp per
keypoint, including a partial last block when needed.

```bash
mkdir -p build
orient=examples/runtime_launch/compute_orientations
nvcc -std=c++17 -O3 -arch=sm_75 \
  "$orient/compute_orientations_fixed.cu" \
  "$orient/bench_compute_orientations_fixed.cu" -o build/compute_orientations_fixed
./toolcall "$orient/compute_orientations_model.cu" \
  --link-source "$orient/bench_compute_orientations_model.cu" -o build/compute_orientations_aot

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

## Use the native C++ scorer

Use `--native` to compile the C++ scorer and checkpoint weights directly into
`libtoolcall_model.so`. The native executable needs the model library and the
usual CUDA and system libraries at runtime. Both backends use the same
`model_output()` API and the states retained by the checkpoint's learned gate.

```bash
softmax=examples/runtime_launch/softmax
./toolcall --native "$softmax/softmax_model.cu" \
  --link-source "$softmax/bench_softmax_model.cu" -o build/softmax_native
./build/softmax_native 2051 67
```

Compare with the default AOT build at the same size. The benchmark warms
up the scorer before recording selection latency. Existing commands that
pass `--aot` explicitly still select AOT.
