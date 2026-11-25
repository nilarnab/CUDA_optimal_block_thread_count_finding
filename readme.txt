CUDA THREAD-BLOCK SELECTOR QUICK START
---------------------------------------

SETUP:

- Run from the repo root on CIMS CUDA Linux with Python, an NVIDIA GPU, NVCC, CUDA-capable clang++, and a C++17 host compiler.

- Create the environment with 
`
python -m venv .venv
`
or
`
python3.11 -m venv .venv
`

- Activate it with 
`
. .venv/bin/activate
`

- Install dependencies with 
`
python -m pip install -r requirements.txt
`

- Make the project modules available with 
`
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
`

- Load the cuda module with
`
module load nvcc-13.0
`

- Create the output directory with 
`
mkdir -p build
`

RUNNING THE TOOL:

- Set the target for the GPU in cuda2 with 
`
arch=sm_75
`
or use the architecture of your GPU.

- Select the example with 
`
kernel=matrix_multiply_and_relu
`

- Set the source folder with 
`
example=examples/runtime_launch/$kernel
`

- Build the model binary with 
`
./toolcall --gpu-arch "$arch" "$example/${kernel}_model.cu" -o "build/${kernel}_model"
`

- Build the fixed binary (kernel with block size given as input by the user) with
`
nvcc -std=c++17 -O3 -arch="$arch" "$example/${kernel}_fixed.cu" -o "build/${kernel}_fixed"
`

- Run the model binary with 
`
"./build/${kernel}_model" 6144 6144 6144
`
where the three numbers represent input problem size M, N, and K respectively.

- Run the same input at 256 threads per block with 
`
"./build/${kernel}_fixed" 6144 6144 6144 256
`

- Save a sweep CSV under `results/launch_benchmarks/` with 
`
python scripts/collect_launch_benchmarks.py --kernel "$kernel" --model-bin "build/${kernel}_model" --fixed-bin "build/${kernel}_fixed" --size 6144 6144 6144 --blocks 64 128 256 512
`

- Read `gpu_us`, `selection_us`, and `total_us` as medians over 51 timed trials and divide fixed `total_us` by model `total_us` for speedup including selection.
- See `examples/runtime_launch` folder for other example kernels to run

RUNNING THE TOOL ON A CUSTOM KERNEL:

- Start from `matrix_multiply_and_relu_model.cu` for timing and output checks, and keep one `__global__` kernel in the primary `.cu` file as the tool currently supports only one kernel per .cu file.

- Include `toolcall.hpp` in the host launcher and get launch dimensions with `const auto [grid, block] = toolcall::model_output(my_kernel, {m, n, k, work_items});`.

- Match `m`, `n`, and `k` to the checkpoint inputs and set `work_items` to the number of items the 1D grid covers.

- For one item per thread, launch with `my_kernel<<<grid, block>>>(inputs, output, work_items);` and guard the item index inside the kernel.

- For a tiled kernel, use the selected thread count to construct the matching block and grid as the matrix example does.

- Build with `./toolcall my_kernel.cu -o build/my_kernel` and add `--link-source main.cu` if `main()` is in a separate file.

- Run `./build/my_kernel` with the arguments your `main()` expects.

- Use `--native` to select the native C++ CPU scorer or use the default AOTInductor backend.

- Pass `--checkpoint` and the matching `--vocabulary` to deploy your own adaptive model. Steps to get your own checkpoint and vocabulary is given below.

TRAINING:

- Set locations for the LS-CAT checkout and processed dataset with 
`
lscat_repo=data/raw/ls-cat-official
`
and 
`
lscat=data/processed/lscat_t4_1d
`

- Download the public CUDA sources and archived timings with
`
git clone --depth 1 https://github.com/NTNU-HPC-Lab/LS-CAT.git "$lscat_repo"
`

- Import all T4 workload cases with a complete sweep of the selector's 16 one-dimensional block sizes into a new or empty output directory with
`
python scripts/import_lscat.py --repository "$lscat_repo" --output-dir "$lscat"
`

- Compile the imported kernel sources to LLVM IR and generate the vocabulary, token IDs, and FastText corpus with
`
python scripts/prepare_frontend.py --source-manifest "$lscat/sources.json" --source-root "$lscat_repo" --output-dir "$lscat"
`
This step will take sometime (several hours) as there are a lot of kernels to process. You can get a subset of the kernels if you needby removing some from the dataset if you want this script to run quicker.

- Generate the filtered runtime and split-group files for successfully compiled kernels with
`
python scripts/prepare_frontend_from_ir.py --ir-dir "$lscat/llvm_ir" --output-dir "$lscat" --runtimes "$lscat/runtimes.csv" --split-groups "$lscat/split_groups.json"
`

- Train FastText and export the embedding matrix with
`
python scripts/train_fasttext.py --corpus "$lscat/fasttext_corpus.txt" --vocabulary "$lscat/vocabulary.json" --output-model "$lscat/fasttext.bin" --output-matrix "$lscat/fasttext_embeddings.pt"
`

- Train the adaptive selector on CUDA with 50 held-out source groups and all remaining groups for training with
`
python scripts/train_adaptive.py --runtimes "$lscat/runtimes_frontend.csv" --token-ids "$lscat/token_ids.json" --embedding-matrix "$lscat/fasttext_embeddings.pt" --problem-columns m n --split-groups "$lscat/split_groups_frontend.json" --validation-groups 50 --seed 2 --device cuda --epochs 120 --batch-size 8 --loss bce --architecture adaptive --lstm-hidden-size 1 --length-penalty 1e-8 --kl-weight 0 --checkpoint build/lscat_adaptive.pt --save-best
`

- Add `--training-groups 400` to use the report's training group cap.

- Use `--split-seed` to set the split independently of the training seed.


