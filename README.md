# Controlled CUDA GPU heater and monitor

`gpu_heat_monitor` applies a compute-heavy CUDA workload and periodically
reports GPU temperature, board power, utilization, clocks, and active clock
throttling reasons through NVIDIA NVML. It stops when it reaches the requested
target temperature, the hard temperature limit, the time limit, or Ctrl-C.

## Requirements

- Linux system with an NVIDIA GPU and driver
- CUDA toolkit (`nvcc`)
- NVML headers and library; these are normally included with CUDA/the driver

This cannot run on an Apple GPU or a machine without NVIDIA CUDA support.

## Build

```bash
make
```

If the linker cannot locate NVML, install the NVIDIA development package for
your distribution or add its library directory to the `nvcc` command.

## Run

Start conservatively:

```bash
./gpu_heat_monitor --target-temp 70 --max-temp 80 --duration 300 \
  --sample-ms 1000 --csv warmup.csv
```

After confirming that cooling works and the readings look reasonable, a higher
target below the hard limit can be requested:

```bash
./gpu_heat_monitor --target-temp 75 --max-temp 80 --duration 300 \
  --csv warmup-75c.csv
```

Run `./gpu_heat_monitor --help` for workload controls. If a kernel chunk takes
so long that monitoring is sluggish, reduce `--iterations` (for example, to
`5000`). If utilization is unexpectedly low, increase it gradually.

## Safety and experimental limitations

- Do not disable the fan, firmware safeguards, power limits, or thermal
  throttling. Monitor the first runs in person.
- The default 80 C cutoff is deliberately conservative, but it is not a
  universal guarantee. Use a lower limit if required by the GPU/system owner.
- Stop immediately if fans fail, temperature rises unusually quickly, the
  driver reports errors, or the system becomes unstable.
- A GPU cannot be heated to an arbitrary temperature. Firmware, cooling,
  ambient temperature, workload type, power limits, and throttling establish
  its steady state.
- NVML power and utilization values can be averaged over a device-dependent
  interval. They should not be treated as perfectly instantaneous samples.
- Reaching a target once is not the same as maintaining a stable thermal state.
  For repeatable experiments, define a temperature band and stabilization time
  in the later benchmark harness.
