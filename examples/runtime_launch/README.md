# Use the trained selector to launch a CUDA kernel

## C++ launcher with compile-time kernel encoding

Include `toolcall.hpp` in the CUDA file and call `model_output` from
host code before launching the kernel.

```cpp
const auto [grid, block] = toolcall::model_output(my_kernel, {m, n, k, work_items});
my_kernel<<<grid, block>>>(inputs, output, work_items);
```

Give the checkpoint's workload dimensions and the actual number of work
items. This interface supports one kernel per source file and a
one-dimensional launch with one item per thread.

Set the checkpoint, its original vocabulary, and the GPU target once in
your build environment. For example, from the repository root on CUDA2:

```bash
export TOOLCALL_CHECKPOINT="$PWD/checkpoints/retention_h8_split2_groups3_seed1/adaptive_w8_lambda1e-05_seed1.best.pt"
export TOOLCALL_VOCABULARY="$PWD/data/processed_1d/vocabulary.json"
export TOOLCALL_GPU_ARCH=sm_75

./toolcall examples/runtime_launch/toolcall_vector_add.cu -o build/vector_add
./build/vector_add 512
```

With those settings, building from any working directory only needs your
CUDA filename and output path. The corresponding command-line flags can
override the environment.

The build embeds the retained kernel states in the executable and the
scorer weights in `libtoolcall_model.so`. Keep the printed `.toolcall` artifact
directory alongside the executable so the Linux loader can find the library.
The launcher predicts the block size on the CPU, calculates the grid size,
and checks the CUDA output. PyTorch and the vocabulary are build-time inputs.

## Benchmark the model and a fixed block

Build two executables and pass the same `M` to both. The fixed benchmark
takes the baseline block size as its second argument.

```bash
mkdir -p build
nvcc -std=c++17 -O3 -arch="$TOOLCALL_GPU_ARCH" examples/runtime_launch/bench_vector_add_fixed.cu -o build/vector_add_fixed
./toolcall examples/runtime_launch/bench_vector_add_model.cu -o build/vector_add_model

./build/vector_add_fixed 512 256
./build/vector_add_model 512
```

Both programs verify the output and report median `gpu_us`, `selection_us`,
and `total_us` after warm-up with the input buffers already on the GPU.
Divide the fixed program's `total_us` by the model program's `total_us` to
measure speedup including prediction.

## Python launcher

Build the kernel and its host launcher once. Clang emits device LLVM IR for the
selector, and NVCC emits a shared library for the GPU launch. The Python host
loads the checkpoint and encodes the LLVM IR once at startup. For each workload,
it predicts a block size, computes the grid using the kernel's launch rule, and
calls the compiled launcher without leaving the process.

The example uses the one-dimensional M/N/K model and an elementwise vector add
with M×M elements. It verifies every output element on CUDA. From the repository
root on CUDA2, with the project's Python environment active and a compatible
Clang CUDA toolchain and NVCC available:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

python scripts/build_cuda_kernel.py \
  --kernel examples/runtime_launch/vector_add.cu \
  --launcher examples/runtime_launch/vector_add_launcher.cu \
  --gpu-arch sm_75 \
  --output-dir build/vector-add

python examples/runtime_launch/run_vector_add.py \
  --build-dir build/vector-add \
  --checkpoint checkpoints/retention_h8_split2_groups3_seed1/baseline_w8_lambda0_seed1.best.pt \
  --vocabulary data/processed_1d/vocabulary.json \
  --side 512 1024
```

The build writes `kernel.ll` and `kernel.so`. Use a fresh output directory when
rebuilding. Pass `--clang`, `--nvcc`, or `--cuda-path` if the toolchain is
installed outside the usual paths. Use a GPU target supported by your own
device and toolchain. Keep the LLVM preparation comparable to training.

The example uses the pooled baseline checkpoint because it had slightly lower
validation regret and lower measured prediction overhead on this split. You can
substitute an adaptive or sequence checkpoint with the same M/N/K schema. The
checkpoint contains the embedding weights. Supply the original vocabulary used
to assign embedding row IDs. A different vocabulary of the same size can pass
the loader's structural check and silently change predictions.

For your own kernel:

1. Write a kernel-only `.cu` file and an `extern "C"` host launcher that includes
   it. Adapt the launcher's parameter types and grid rule for your operation.
2. Build both artifacts with `build_cuda_kernel.py`. Keep the source being
   encoded focused on the one kernel and its device helpers.
3. Load the model and cache the encoding once:

   ```python
   from pathlib import Path
   from inference import InferenceEngine

   engine = InferenceEngine.from_checkpoint(checkpoint, vocabulary_path, device="cpu")
   cached = engine.prepare_kernel(Path("build/my-kernel/kernel.ll").read_text())
   ```

4. Filter the checkpoint's catalog to configurations your kernel supports on
   the actual GPU. Include any algorithm restrictions, such as fixed block
   sizes, required warp counts, and shared-memory limits, alongside the device's
   limits. Select the highest scoring legal block with
   `engine.predict_cached(cached, (m, n, k), allowed=allowed).configuration`.
   Keep dimension order and meaning aligned with training. `engine.problem_columns`
   gives the expected order. These dimensions can differ from work-item count.
5. Pass the chosen block, your CUDA device pointers, and the caller's CUDA
   stream to your launcher. Compute the grid per kernel. The example shows
   one-dimensional coverage and uses a bounds check for the final partial
   block. In production keep your existing known-safe launch path for callers
   with no legal model action or failed model setup.

The Python example uses PyTorch tensors to own the buffers and passes their
device pointers through `ctypes`. It synchronizes to verify correctness. Your
integration can keep normal asynchronous stream semantics by retaining the
buffers until the launch finishes. The predictor defaults to CPU and can run
on CUDA with `--predictor-device cuda:0`. Build and tokenization are outside the
launch path. When the same kernel and workload recur, reuse the selected block
if GPU and launch constraints are unchanged. Measure predictor overhead plus
kernel runtime before enabling the selector for short kernels.

This is a Linux CUDA wrapper for a Python host. The trained models predict block
shapes from their catalog, and their speed on new kernels or GPU models needs
measurement.
