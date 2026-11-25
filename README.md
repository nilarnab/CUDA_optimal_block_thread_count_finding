# CUDA Thread-Block Selector Quick Start

## Setup

Run from the repo root on CIMS CUDA Linux with Python, an NVIDIA GPU, NVCC, CUDA-capable clang++, and a C++17 host compiler.

1. Create the environment:

   ```bash
   python -m venv .venv
   # or
   python3.11 -m venv .venv
   ```

2. Activate it:

   ```bash
   . .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Make the project modules available:

   ```bash
   export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
   ```

5. Load the CUDA module:

   ```bash
   module load cuda-13.0
   ```

6. Create the output directory:

   ```bash
   mkdir -p build
   ```

Notes:

- Make sure you have at least **6.7 GB** space in your disk to create the environment and run the tool. If you want to run training and create your own checkpoint, you need more disk space quota so if you run into space problems, please clear your partition and try again.

## Running the Tool

1. Set the target for the GPU in cuda2 (or use the architecture of your GPU):

   ```bash
   arch=sm_75
   ```

2. Select the example:

   ```bash
   kernel=matrix_multiply_and_relu
   ```

3. Set the source folder:

   ```bash
   example=examples/runtime_launch/$kernel
   ```

4. Build the model binary:

   ```bash
   ./toolcall --gpu-arch "$arch" "$example/${kernel}_model.cu" -o "build/${kernel}_model"
   ```

5. Build the fixed binary (kernel with block size given as input by the user):

   ```bash
   nvcc -std=c++17 -O3 -arch="$arch" "$example/${kernel}_fixed.cu" -o "build/${kernel}_fixed"
   ```

6. Run the model binary, where the three numbers are the input problem size `M`, `N`, and `K`:

   ```bash
   "./build/${kernel}_model" 6144 6144 6144
   ```

7. Run the same input at 256 threads per block:

   ```bash
   "./build/${kernel}_fixed" 6144 6144 6144 256
   ```

8. Save a sweep CSV under `results/launch_benchmarks/`:

   ```bash
   python scripts/collect_launch_benchmarks.py \
     --kernel "$kernel" \
     --model-bin "build/${kernel}_model" \
     --fixed-bin "build/${kernel}_fixed" \
     --size 6144 6144 6144 \
     --blocks 64 128 256 512
   ```

Notes:

- Read `gpu_us`, `selection_us`, and `total_us` as medians over 51 timed trials. Divide fixed `total_us` by model `total_us` for speedup including selection.
- See the `examples/runtime_launch` folder for other example kernels to run.

## Running the Tool on a Custom Kernel

- Start from `matrix_multiply_and_relu_model.cu` for timing and output checks. Keep one `__global__` kernel in the primary `.cu` file, as the tool currently supports only one kernel per `.cu` file.
- Include `toolcall.hpp` in the host launcher and get launch dimensions with:

  ```cpp
  const auto [grid, block] = toolcall::model_output(my_kernel, {m, n, k, work_items});
  ```

- Match `m`, `n`, and `k` to the checkpoint inputs, and set `work_items` to the number of items the 1D grid covers.
- For one item per thread, launch with the call below and guard the item index inside the kernel:

  ```cpp
  my_kernel<<<grid, block>>>(inputs, output, work_items);
  ```

- For a tiled kernel, use the selected thread count to construct the matching block and grid, as the matrix example does.
- Build with `./toolcall my_kernel.cu -o build/my_kernel`. Add `--link-source main.cu` if `main()` is in a separate file.
- Run `./build/my_kernel` with the arguments your `main()` expects.
- Use `--native` to select the native C++ CPU scorer, or use the default AOTInductor backend.
- Pass `--checkpoint` and the matching `--vocabulary` to deploy your own adaptive model. Steps to get your own checkpoint and vocabulary are given below.

## Training

1. Set locations for the LS-CAT checkout and processed dataset:

   ```bash
   lscat_repo=data/raw/ls-cat-official
   lscat=data/processed/lscat_t4_1d
   ```

2. Download the public CUDA sources and archived timings:

   ```bash
   git clone --depth 1 https://github.com/NTNU-HPC-Lab/LS-CAT.git "$lscat_repo"
   ```

3. Import all T4 workload cases with a complete sweep of the selector's 16 one-dimensional block sizes into a new or empty output directory:

   ```bash
   python scripts/import_lscat.py --repository "$lscat_repo" --output-dir "$lscat"
   ```

4. Compile the imported kernel sources to LLVM IR and generate the vocabulary, token IDs, and FastText corpus:

   ```bash
   python scripts/prepare_frontend.py \
     --source-manifest "$lscat/sources.json" \
     --source-root "$lscat_repo" \
     --output-dir "$lscat"
   ```

   > This step takes several hours as there are a lot of kernels to process. To run it quicker, remove some kernels from the dataset to use a subset.

5. Generate the filtered runtime and split-group files for successfully compiled kernels:

   ```bash
   python scripts/prepare_frontend_from_ir.py \
     --ir-dir "$lscat/llvm_ir" \
     --output-dir "$lscat" \
     --runtimes "$lscat/runtimes.csv" \
     --split-groups "$lscat/split_groups.json"
   ```

6. Train FastText and export the embedding matrix:

   ```bash
   python scripts/train_fasttext.py \
     --corpus "$lscat/fasttext_corpus.txt" \
     --vocabulary "$lscat/vocabulary.json" \
     --output-model "$lscat/fasttext.bin" \
     --output-matrix "$lscat/fasttext_embeddings.pt"
   ```

7. Train the adaptive selector on CUDA with 50 held-out source groups and all remaining groups for training:

   ```bash
   python scripts/train_adaptive.py \
     --runtimes "$lscat/runtimes_frontend.csv" \
     --token-ids "$lscat/token_ids.json" \
     --embedding-matrix "$lscat/fasttext_embeddings.pt" \
     --problem-columns m n \
     --split-groups "$lscat/split_groups_frontend.json" \
     --validation-groups 50 \
     --seed 2 \
     --device cuda \
     --epochs 120 \
     --batch-size 8 \
     --loss bce \
     --architecture adaptive \
     --lstm-hidden-size 1 \
     --length-penalty 1e-8 \
     --kl-weight 0 \
     --checkpoint checkpoints/lscat_adaptive.pt \
     --save-best
   ```

Options:

- Add `--training-groups 400` to use the report's training group cap.
- Use `--split-seed` to set the split independently of the training seed.