# LS-CAT kernel data pilot

Run `smoke` on one CIMS GPU, inspect its summary, then run `full` in three separate sessions on the same GPU model. Repeat on a second GPU model only after the first works. Each invocation creates a new run directory and results archive.

This package bundles six original CUDA kernels from [NTNU-HPC-Lab/LS-CAT](https://github.com/NTNU-HPC-Lab/LS-CAT), pinned to commit `83a50786eee2de633be477819c044d2c9a1ad076`. Kernel bytes and original empty `includes.h` files are preserved. `kernels/manifest.json` records paths, repository IDs, Git blob IDs, SHA256 checksums, and source URLs. The input wrappers, CPU references, timing, and analysis are new pilot code. No dataset download or Python packages are needed on the server.

| Pilot ID | Original function | Workload for dimension N |
| --- | --- | --- |
| vector_add | add | N*N float vector addition |
| segmented_sum | gpuSum | N integer segments, each of length N |
| periodic_tile | kTile | Tile a 17*13 float input to N*N output |
| repeat_vector | times | Repeat an N-element float input N times |
| matrix_transpose | flatten_kernel | Permute float [1,N,N] from channel-major to spatial-major |
| matrix_add | sumMat | N*N double matrix addition |

Five kernels index threads along x, so their valid launch actions are `(64,1)` through `(1024,1)` in steps of 64. `matrix_add` supports the historical LS-CAT shapes `(1,64)` through `(1,1024)` and `(8,8)`, `(16,16)`, `(24,24)`, `(32,32)`. Device and kernel limits can remove unsupported shapes. The changed 1D orientation is deliberate: using `(1,b)` for x-only indexing would repeat work or introduce races. This is a correctness-checked subset study, not an exact replay of the paper's generated workloads or score.

## 1. Requirements

- Linux GPU compute host, NVIDIA GPU, and a CUDA toolkit compatible with that GPU and driver.
- `nvcc`, a compatible C++ compiler exposed as `c++`, Python **3.9 or later**, `bash`, and `tar`.
- One GPU assigned/available under your normal CIMS or HPC access rules. The launcher preserves `CUDA_VISIBLE_DEVICES`; the device argument is a CUDA **logical** index inside that assignment, normally `0` under Slurm.
- No administrator commands, clock changes, containers, or pip installs.

The launcher detects the GPU and compiles for its exact architecture. A recent toolkit may omit older GPUs: CUDA 13 removed offline support for Maxwell, Pascal, and Volta. Use an appropriate earlier toolkit for those devices, and check older Kepler devices separately. The discovery/build logs are retained if compilation fails. [NVIDIA release notes](https://docs.nvidia.com/cuda/archive/13.0.3/cuda-toolkit-release-notes/index.html).

## 2. CIMS CUDA server

Transfer `ls-cat-pilot.tar.gz` to your CIMS home directory using your normal transfer method, then connect through the CIMS access server to an authorized CUDA host. The examples use `cuda2`; another compatible CUDA host works. Replace `CIMS_USER` with your CIMS username.

```bash
# On your laptop, in the directory containing the archive:
scp ls-cat-pilot.tar.gz CIMS_USER@access.cims.nyu.edu:~/
ssh CIMS_USER@access.cims.nyu.edu

# On access:
ssh cuda2.cims.nyu.edu

# On the CUDA host:
tar -xzf ~/ls-cat-pilot.tar.gz -C ~
cd ~/ls_cat_pilot
module avail cuda
```

Load a CUDA/compiler module shown by your server, using `module load ACTUAL_MODULE_NAME`. Then:

```bash
nvcc --version
python3 --version
nvidia-smi
bash run.sh smoke 0
```

The launcher prints the unique result directory and archive path. Read that run's `summary.md`. If it says `COMPLETE`, run the larger collection:

```bash
bash run.sh quick 0
bash run.sh full 0
```

Use the actual logical device index available to you. For an unallocated multi-GPU CIMS host, inspect current use and choose a GPU according to the host's rules. Do not overwrite scheduler visibility settings to reach another GPU. [Official CIMS instructions](https://cims.nyu.edu/dynamic/systems/resources/computeservers/cuda/).

## 3. HPC / Slurm

Your confirmed HPC is **NYU Torch**. On its login host, run `my_slurm_accounts` to find your project account, then `module avail` to find the available CUDA/compiler/Python modules. Load the modules you will use, and check `nvcc --version`, `c++ --version`, and `python3 --version` before submitting. Replace `YOUR_ACTUAL_ACCOUNT` below with an account returned for your allocation. GPU discovery and benchmark compilation happen inside the allocated job, not on the login host. [Torch modules](https://services.rt.nyu.edu/docs/hpc/tutorial_intro_hpc/modules/).

Transfer and extract the package into storage visible from your compute job. Load the available CUDA/compiler/Python environment first. Submit from inside `ls_cat_pilot`:

```bash
sbatch --account=YOUR_ACTUAL_ACCOUNT --export=ALL,MODE=smoke submit.slurm
```

After the smoke run is `COMPLETE`:

```bash
sbatch --account=YOUR_ACTUAL_ACCOUNT --export=ALL,MODE=full submit.slurm
```

Use `squeue -u "$USER"` to follow scheduling, then inspect `lscat-JOBID.out` and the printed run directory. The three full sessions should use the same GPU model: once the smoke run records its GPU, select the corresponding available Torch GPU-type constraint for later submissions, following the submission guide. Do not combine measurements from different models without labeling them.

`submit.slurm` requests one GPU, two CPU cores, 8 GB memory, and two hours. Change those resource requests if your HPC's rules require it. It leaves the account, partition, GPU type, and module names to your environment. On NYU Torch, `my_slurm_accounts` lists available accounts; current guidance requires an account and says to omit manual partition selection. [Torch submission guide](https://services.rt.nyu.edu/docs/hpc/submitting_jobs/slurm_submitting_jobs/).

An interactive GPU allocation also works:

```bash
srun --account=YOUR_ACTUAL_ACCOUNT --gres=gpu:1 --cpus-per-task=2 --mem=8G --time=00:30:00 --pty bash
# Inside the allocated GPU session, after loading the toolkit:
cd /your/shared/path/ls_cat_pilot
bash run.sh smoke 0
```

Torch currently enforces minimum GPU utilization. Isolated short kernel runs and CPU correctness checks may fall below that threshold, so CIMS is the best first target. If the scheduler cancels a benchmarking job, retain the logs and use a suitable approved benchmarking allocation or host; do not add artificial GPU work.

## 4. Collection size and protocol

| Mode | Square dimensions | Independent process rounds per action | Planned trials before device filtering |
| --- | --- | --- | --- |
| smoke | 240 | 1 | 100 |
| quick | 240, 784, 2024 | 3 | 900 |
| full | 240, 496, 784, 1016, 1232, 1680, 2024 | 10 | 7,000 |

Timing depends on the GPU, process startup, CPU reference generation, and shared-host activity; there is no fixed completion-time promise. The script streams progress and preserves completed rows immediately. Each configuration gets its own process and a 60-second timeout. A run interrupted by a time limit retains flushed rows; a hard kill can prevent automatic archiving, so the run directory itself is the primary record.

Each trial uses initialized deterministic inputs, checks all output elements against a CPU reference after three warm-up launches, calibrates a batch targeting about 10 ms capped at 1,000 launches, resets the output, times a batch with CUDA events and end-event synchronization, then checks all outputs again. Allocation, input/output copies, CPU reference work, and correctness checks are excluded from GPU timing. Kernel inputs are immutable across repeated launches. Timings describe repeated warm launches and may include host submission gaps; they are not cold-cache, application end-to-end, or energy measurements. [NVIDIA timing guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#timing).

The plan shuffles all kernel/size/action/round trials using a recorded random session seed. Within a session, each kernel/size uses the same initialized input across actions and rounds. Separate invocations use new inputs and trial orders. GPU allocation, compiler settings, and input seed remain part of the experimental context.

## 5. What to keep and return

Keep the generated **entire results archive**, including partial runs. Important members are:

- `trials.csv`: raw per-process timings, status/error, block and grid shape, iterations, seed, source hash, and validation error.
- `planned.json`: exact trial plan, including configurations not completed.
- `measurements.csv`: median/MAD per kernel/size/action; failed or missing groups remain visible.
- `action_table.csv`: only fully successful action groups, with a documented log-runtime reward and tested-set regret for replay experiments.
- `metadata.json`, `environment.txt`, and GPU inventory: actual GPU properties, per-kernel resources, runtime/driver/compiler details, host, and allocation context.
- `source/`, `source_sha256.txt`: exact package sources used for that run.
- `build.log`, `host_checks.log`, `stdout.log`, `progress.log`, `exit_status.txt`: build, host self-check, and collection diagnostics.
- `summary.md`, `summary.json`, and `COMPLETE` or `INCOMPLETE`: coverage and descriptive default-versus-best comparisons.

`COMPLETE` means every planned trial is present exactly once and has an accepted finite timing with passed output checks. It does not certify statistical significance, isolation from other users, or model performance. Failed, missing, and non-finite timings never become zero-runtime labels. `action_table.csv` can contain valid actions from an incomplete run, but retain the failure records to avoid hiding selection bias.

The descriptive default is the fully valid action with the largest thread count (ties prefer larger x, then y). `reward_log_vs_default = log(default_median_ms / action_median_ms)` and `regret = action_median_ms / best_measured_median_ms - 1`. These are explicit pilot definitions, not the paper's historical normalized score. Future model evaluation must select defaults and tune models using training groups, then evaluate on held-out kernels/sessions.

For CIMS shared home directories, you can copy the printed run archive back through `access.cims.nyu.edu` using `scp`. For HPC, use your usual transfer route. Return the archive from smoke first if it fails; its logs should identify the toolkit/device/compile/runtime issue.

## 6. Next data milestone

Collect three independent `full` sessions on one GPU model. Inspect correctness, coverage, timing spread, and whether block choice has repeatable headroom over a fixed default. Then add independently sourced kernels with reviewed argument adapters and CPU references before training or testing code-based generalization. Six selected kernels are enough to validate collection and build a tabular RL replay prototype; they are too few for a credible learned code model.

Keep every size, block configuration, and repetition of a kernel together when splitting. Group by the originating LS-CAT repository ID and source similarity as the collection grows. Keep GPU models separate unless hardware is an explicit model input and the experiment includes a defined transfer protocol. Do not pool T4 archive runtimes with new hardware measurements as if they were interchangeable labels.

## 7. Local checks and optional device checks

The launcher builds and runs the CPU reference self-check before compiling the CUDA harness. To check the data-processing code without a GPU:

```bash
python3 -m unittest discover -s . -p 'test_*.py' -v
c++ -O2 -std=c++14 test_reference.cpp -o /tmp/lscat-reference-test
/tmp/lscat-reference-test
```

On a compute host with Compute Sanitizer available, use the **printed run directory**, whose binary matches its saved sources, for an additional memory check outside timing collection:

```bash
compute-sanitizer --tool memcheck --error-exitcode 1 /actual/run/directory/benchmark --device 0 --kernel matrix_add --size 240 --block-x 16 --block-y 16 --seed 1
```

The ordinary output checks validate results; a sanitizer separately checks device memory accesses. Never use sanitizer-instrumented durations as performance labels.

The package intentionally contains no model training, generic arbitrary-kernel execution, or automatic modification of upstream sources. Extending the dataset requires an input contract, a CPU reference, and a valid launch recipe per added kernel.
