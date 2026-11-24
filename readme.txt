CUDA THREAD-BLOCK SELECTOR QUICK START

SETUP
- Run from the repo root on CIMS Linux with Python 3.11, an NVIDIA GPU, NVCC, CUDA-capable clang++, and a C++17 host compiler.
- Create the environment with `python -m venv .venv`, or use `python3.11 -m venv .venv` if needed.
- Activate it with `. .venv/bin/activate`.
- Install dependencies with `python -m pip install -r requirements.txt`.
- Make the project modules available with `export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"`.
- Create the output directory with `mkdir -p build`.

TRAINING
- Point to the bundled 12-kernel pilot with `data=data/processed_1d`.
- Train for two CPU epochs with 
`
python scripts/train_adaptive.py --runtimes "$data/runtimes.csv" --token-ids "$data/token_ids.json" --embedding-matrix "$data/fasttext_embeddings.pt" --problem-columns m n k --split-groups "$data/split_groups.json" --validation-groups 2 --seed 1 --device cpu --epochs 2 --batch-size 8 --loss bce --architecture adaptive --lstm-hidden-size 1 --length-penalty 1e-8 --kl-weight 0 --checkpoint build/pilot_adaptive.pt --save-best --verbose
`
- Choose `--architecture adaptive` to learn retention, `final` for the last state, `sequence` for all states, `context` for dimensions only, or `baseline` for pooled LSTM states.
- Set `--lstm-hidden-size` for features per state and `--length-penalty` to encourage fewer retained states.
- Set a positive `--kl-weight` to enable EMA-teacher regularization and use `--loss` to select `bce`, `mse`, or `regret`.
- Use `--split-groups` and `--validation-groups` for source-group holdouts, `--training-groups` to cap training groups, and `--split-seed` to keep the split fixed.
- Use `--device cuda` for GPU training and adjust `--epochs` and `--batch-size` for the run.
- Use `--save-best` to save a `.best.pt` checkpoint selected by validation regret, with task loss as the tiebreaker.
- For public LS-CAT T4 data, follow the preparation steps in `ML_README.md` with `python` from the activated environment.
- If any kernels fail to compile, use `scripts/prepare_frontend_from_ir.py --help` for the command that filters runtimes and groups to the successful LLVM IR files.
- For the report settings, use the prepared T4 data paths, a new checkpoint name, and `--problem-columns m n --training-groups 400 --validation-groups 50 --seed 2 --split-seed 2 --epochs 120 --device cuda`.

MATRIX MULTIPLY + RELU BENCHMARK
- Set the target for an RTX 2080 Ti with `arch=sm_75` or use the architecture of your GPU.
- Select the example with `kernel=matrix_multiply_and_relu`.
- Set the source folder with `example=examples/runtime_launch/$kernel`.
- Build the model binary with 
`
./toolcall --gpu-arch "$arch" "$example/${kernel}_model.cu" -o "build/${kernel}_model"
`
- Build the fixed binary with
`
nvcc -std=c++17 -O3 -arch="$arch" "$example/${kernel}_fixed.cu" -o "build/${kernel}_fixed"
`
- Run the model with 
`
"./build/${kernel}_model" 6144 6144 6144
`, passing M, N, and K in that order.
- Run the same input at 256 threads per block with 
`
"./build/${kernel}_fixed" 6144 6144 6144 256
`
- Save a sweep CSV under `results/launch_benchmarks/` with 
`
python scripts/collect_launch_benchmarks.py --kernel "$kernel" --model-bin "build/${kernel}_model" --fixed-bin "build/${kernel}_fixed" --size 6144 6144 6144 --blocks 64 128 256 512
`
- Read `gpu_us`, `selection_us`, and `total_us` as medians over 51 timed trials and divide fixed `total_us` by model `total_us` for speedup including selection.
- See `examples/runtime_launch/README.md` for larger inputs and other kernels.

YOUR OWN KERNEL
- Start from `matrix_multiply_and_relu_model.cu` for timing and output checks, and keep one `__global__` kernel in the primary `.cu` file.
- Include `toolcall.hpp` in the host launcher and get launch dimensions with `const auto [grid, block] = toolcall::model_output(my_kernel, {m, n, k, work_items});`.
- Match `m`, `n`, and `k` to the checkpoint inputs and set `work_items` to the number of items the 1D grid covers.
- For one item per thread, launch with `my_kernel<<<grid, block>>>(inputs, output, work_items);` and guard the item index inside the kernel.
- For a tiled kernel, use the selected thread count to construct the matching block and grid as the matrix example does.
- Build with `./toolcall my_kernel.cu -o build/my_kernel` and add `--link-source main.cu` if `main()` is in a separate file.
- Run `./build/my_kernel` with the arguments your `main()` expects.
- Use `--native` to select the C++ CPU scorer or use the default AOTInductor backend.
- Pass `--checkpoint` and the matching `--vocabulary` to deploy your own adaptive model.
- Keep the executable beside its `.toolcall/` directory and keep LibTorch available for the default AOTInductor build.
