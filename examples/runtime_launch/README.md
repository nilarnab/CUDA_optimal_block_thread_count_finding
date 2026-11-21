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
