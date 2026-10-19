# Thermal launch-configuration experiment

This directory contains a separate experiment harness. It does not modify or
replace the root `gpu_heat_monitor` program.

The harness tests whether the fastest CUDA launch configuration changes with
pre-launch GPU temperature. It uses a fixed grid-stride compute problem so
threads per block and block count can vary independently.

For every target temperature it:

1. Heats or passively cools the GPU into a target band.
2. Keeps it in that band for a stabilization period.
3. Records pre-launch NVML telemetry.
4. Times several launches with CUDA events.
5. Records post-launch telemetry.
6. Repeats all configurations in randomized order.

## What configurations are tested?

The default configuration search is the cross-product of:

```text
Threads per block: 64, 128, 256, 512
Blocks per SM:     2, 4
```

This produces eight launch configurations at every target temperature:

| Configuration | Threads per block | Blocks per SM |
| --- | ---: | ---: |
| 1 | 64 | 2 |
| 2 | 64 | 4 |
| 3 | 128 | 2 |
| 4 | 128 | 4 |
| 5 | 256 | 2 |
| 6 | 256 | 4 |
| 7 | 512 | 2 |
| 8 | 512 | 4 |

The actual CUDA grid size is calculated as:

```text
block count = number of GPU SMs x blocks per SM
```

For example, on a GPU with 46 SMs, `2` blocks/SM produces 92 blocks and `4`
blocks/SM produces 184 blocks. Because the test kernel uses a grid-stride loop,
all configurations process the same number of elements.

## Temperature controls

`--temperatures` lists the pre-launch temperature targets. For example:

```bash
--temperatures 45,55,65
```

`--temp-band` defines the acceptable deviation around each target. A target of
55 C with `--temp-band 1` permits temperatures from 54 C through 56 C.

`--stabilize-seconds` specifies how long the temperature must remain
continuously inside that band before a trial can start. With:

```bash
--temperatures 55 --temp-band 1 --stabilize-seconds 3
```

the sequence is:

```text
Heat or passively cool toward 55 C
              |
Enter the 54-56 C acceptable band
              |
Stay inside the band continuously for 3 seconds
              |
Record pre-launch telemetry and run one trial
```

If the temperature leaves the band, the stabilization timer resets. After a
trial, the program restores and stabilizes the target temperature again before
testing the next randomized configuration.

`--max-temp` is an emergency stop, not a target. Every target plus its band
must be below this value. `--prepare-timeout` controls how long the program will
try to reach a target before reporting that the target may be unreachable.

## Build on a CIMS CUDA machine

From `access2`, connect to an available CUDA host and inspect usage first:

```bash
ssh cuda5.cims.nyu.edu
nvidia-smi
```

Do not run the experiment if another user is using the selected GPU. Load an
available CUDA module, enter this directory, and build:

```bash
module avail cuda
module load cuda-<version-listed-by-module-avail>
cd ~/CUDA_optimal_block_thread_count_finding/experiment
make clean
make
```

Adjust the `cd` path if the repository is stored somewhere else.

## First smoke test

Use one reachable temperature and one repetition before starting a larger run.
If the GPU already idles above 40 C, select a target a few degrees above its
idle temperature.

```bash
./thermal_config_experiment \
  --temperatures 45 \
  --threads 64,128,256,512 \
  --blocks-per-sm 2,4 \
  --repetitions 1 \
  --launches 3 \
  --stabilize-seconds 2 \
  --max-temp 70 \
  --csv smoke.csv
```

This smoke test runs eight trials: one for each configuration. If 45 C is below
the GPU's idle temperature, replace it with a reachable value a few degrees
above the observed idle temperature. For a faster initial check, a 2 C band can
be used:

```bash
--temp-band 2 --stabilize-seconds 1
```

The terminal prints each configuration's time and its pre/post temperature.
The full measurements are written to `smoke.csv`.

Keep a second terminal open with `watch -n 1 nvidia-smi`. Stop with Ctrl-C if
another user starts a workload, cooling behaves abnormally, or the machine
becomes unstable.

## Initial experiment

After the smoke test succeeds, start with:

```bash
./thermal_config_experiment \
  --temperatures 45,55,65 \
  --threads 64,128,256,512 \
  --blocks-per-sm 2,4 \
  --repetitions 5 \
  --launches 10 \
  --temp-band 1 \
  --stabilize-seconds 3 \
  --max-temp 75 \
  --csv results.csv
```

This produces 120 randomized trials: 3 temperatures x 8 configurations x 5
repetitions. The program restores the target band before each trial. Cooling to
a lower temperature can take several minutes, so temperatures should normally
be supplied in ascending order.

`--launches 10` means that every trial launches the same configuration ten
times inside one CUDA-event timing interval. The CSV contains both total time
and average time per launch. If the pre/post temperature changes by more than
about 1 C, reduce `--launches` so the trial better represents its pre-launch
state.

To select a different GPU on a multi-GPU host, add `--device 1`.

## Analyze results

```bash
python3 analyze_results.py results.csv --summary summary.csv
```

The analyzer ranks configurations by median time and reports regret relative to
the fastest point estimate at each temperature. A changed point-estimate winner
is not automatically a statistically meaningful change: compare the runtime
difference with the reported variability and collect more repetitions before
drawing a conclusion.

The raw results contain one row per trial, including:

- target, pre-launch, and post-launch temperature;
- threads per block, blocks per SM, and actual block count;
- total trial time and average kernel-launch time;
- power, utilization, SM clock, and memory clock before and after the trial;
- NVML clock-throttling reason masks.

If the winning configuration changes across temperature targets, first verify
that its improvement is larger than the run-to-run variability. Also inspect
the clock and throttling columns: temperature may affect performance through
clock or power throttling rather than directly.

## Command reference

Run `./thermal_config_experiment --help` to see all options. The most useful
controls are:

| Option | Meaning |
| --- | --- |
| `--device N` | CUDA device index on a multi-GPU host |
| `--temperatures A,B,C` | Target pre-launch temperatures |
| `--threads A,B,C` | Candidate thread counts per block |
| `--blocks-per-sm A,B` | Candidate grid sizes, expressed per SM |
| `--repetitions N` | Trials per configuration at each temperature |
| `--launches N` | Timed kernel launches inside each trial |
| `--temp-band C` | Allowed temperature deviation around a target |
| `--stabilize-seconds S` | Continuous time required inside the band |
| `--prepare-timeout S` | Maximum time allowed to reach/stabilize a target |
| `--max-temp C` | Emergency temperature cutoff |
| `--csv PATH` | Raw output CSV path |

## Important limitations

- The first version contains one compute-bound kernel. Later experiments should
  add memory-bound and synchronization-heavy kernels before making broader
  claims.
- Temperature may affect performance through clock or power throttling rather
  than directly. The raw CSV records clocks, power, and throttle reasons.
- NVML readings may be averaged over device-dependent intervals.
- The benchmark itself can change temperature. Keep `--launches` low enough
  that pre- and post-temperature normally differ by no more than about 1 C.
- A target below idle temperature or above the cooling system's reachable
  steady state will time out rather than run an uncontrolled trial.
- Never disable fans, power limits, or NVIDIA thermal safeguards.
