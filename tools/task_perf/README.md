# Task performance counters

Tools to measure how much work each SWIFT task does and how fast it does it,
to tell compute-bound from bandwidth-bound tasks (e.g. before porting them to
GPUs).

## Recording

Configure with `--enable-task-perf` (implies `--enable-task-debugging`) and
run with task dumps, e.g. `swift ... -y 1` for every step. In addition to the
usual `thread_info-stepN.dat`, each dump writes `task_perf-stepN.dat` with,
for every task run in the step: its type, thread, start and end ticks, the
particle counts of its cells and the counters of `src/task_perf.h`:

| counter              | incremented by                                        |
|----------------------|-------------------------------------------------------|
| `hydro_iact_sym`     | each call of a symmetric hydro kernel `runner_iact_*` |
| `hydro_iact_nonsym`  | each call of a `runner_iact_nonsym_*` hydro kernel    |
| `grav_p2p_full`      | each Newtonian P2P evaluation (incl. SIMD padding)    |
| `grav_p2p_truncated` | each mesh-truncated P2P evaluation                    |
| `grav_m2p_full`      | each Newtonian M2P evaluation                         |
| `grav_m2p_truncated` | each mesh-truncated M2P evaluation                    |
| `grav_m2l`           | each M2L (multipole to field tensor) evaluation       |
| `grav_parts_loaded`  | each gpart read into the P2P/M2P loops                |

The counters are plain per-runner integer increments; the counter overhead is
measured by comparing the step times with a build without them.

## Cost of one kernel evaluation

    tools/task_perf/kernel_flops.py <build_dir> --json kernels_compiled.json
    tools/task_perf/source_flops.py <build_dir> --json kernels_source.json

Both work on the wrappers of `kernel_flops.c`, compiled for the configured
build (same flags, same hydro/gravity schemes):

- `kernel_flops.py` counts the FP instructions the compiler generated
  (empirical FLOPs; AArch64 and x86-64, scalar and SIMD).
- `source_flops.py` counts the FP operations written in the C source by
  walking clang's AST (theoretical FLOPs; reports the cheapest and the most
  expensive path through branches).

The two should agree to within what the compiler eliminates (common
sub-expressions, constant folding).

## Machine ceilings

    clang -O2 -ffp-contract=fast -fno-vectorize -fno-slp-vectorize -mcpu=native \
          -pthread tools/task_perf/machine_peak.c -o machine_peak
    ./machine_peak <threads> > machine.json

measures the FP32 FMA peak (SIMD and scalar, one core and all threads) and the
STREAM-triad bandwidth (one thread and all threads).

## Analysis

    tools/task_perf/analyse_task_perf.py --kernels kernels_compiled.json \
        --machine machine.json --out eagle12 'task_perf-step*.dat'

(run through `uv`, which provides numpy and matplotlib) prints and writes, per
task type: count, time, interactions, GFLOP, GFLOP/s, GB/s, arithmetic
intensity with respect to the unique data touched by the task and with
respect to the data streamed through the kernels, and draws a roofline
(`*_roofline.png`) and the cost and thread utilisation of each step
(`*_steps.png`). See the docstring of the script for the byte models.

Task timelines are drawn by the existing `tools/task_plots/plot_tasks.py`
from the `thread_info-stepN.dat` files of the same run.
