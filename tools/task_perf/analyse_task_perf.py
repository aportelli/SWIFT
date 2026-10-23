#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy", "matplotlib"]
# ///
"""
Turn the task_perf-step*.dat dumps of a SWIFT run configured with
--enable-task-perf into performance figures per task type: time, number of
interactions, FLOPs, FLOP rate, working-set bytes and arithmetic intensity.

FLOPs = sum over counters of (count x FLOPs per evaluation). The FLOPs per
kernel evaluation come from kernel_flops.py (compiler-generated) or
source_flops.py (counted in the C source); to these we add the FLOPs of the
loop body around the kernel call (distance computation, accumulation), which
were counted by hand in the source (see LOOP_FLOPS below).

Two byte counts are modelled for each task:

  - unique: the particle data the task touches, read once and written back
    once. For hydro this is bounded by the particles of the cells and by the
    number of interactions, for gravity by the particles of the cells and
    the number of particles loaded in the gravity caches; M2L reads one
    multipole per interaction. This is the data that would have to be moved
    to an accelerator, and the DRAM traffic if nothing is reused across
    tasks. FLOP / unique byte is the arithmetic intensity (AI) to compare
    with the machine balance (peak FLOP/s / memory bandwidth).
  - streamed: the data read by the kernels if nothing were reused between
    kernel evaluations (one particle per hydro interaction, one per particle
    loaded in a gravity cache). FLOP / streamed byte tells whether the kernel
    could be fed from the caches.

Usage:
    analyse_task_perf.py --kernels kernels.json [--machine machine.json]
                         [--out prefix] task_perf-step*.dat
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "task_plots"))
from swift_hardcoded_data import TASKTYPES, SUBTYPES  # noqa: E402

# FLOPs of the loop body around each kernel call, counted in the source:
#
# hydro (runner_doiact_functions_hydro.h, e.g. DOPAIR1): pj position relative
#   to the cell (3 sub), dx (3 sub), r2 (3 mul + 2 add) = 11. Only the pairs
#   that pass the r2 < h^2 test are counted, so the distance checks of the
#   pairs that fail it are not included.
# grav P2P (runner_doiact_grav.c, runner_dopair_grav_pp_full/truncated):
#   dx (3 sub), periodic wrapping (3 add), r2 (3 mul + 2 add), h^2, 1/h, 1/h^3
#   (1 + 1 + 2), accumulation of a and pot (3 mul + 4 add) = 22.
# grav M2P (runner_dopair_grav_pm_full/truncated): dx (3 sub), periodic
#   wrapping (3 add), r2 (3 mul + 2 add), accumulation (4 add) = 15.
LOOP_FLOPS = {
    "hydro": 11,
    "grav_p2p": 22,
    "grav_m2p": 15,
    "grav_m2l": 0,
}

COUNTERS = [
    "hydro_iact_sym",
    "hydro_iact_nonsym",
    "grav_p2p_full",
    "grav_p2p_truncated",
    "grav_m2p_full",
    "grav_m2p_truncated",
    "grav_m2l",
    "grav_parts_loaded",
]

# Counters that are not kernel evaluations
NOT_INTERACTIONS = ("grav_parts_loaded",)


def read_dump(fname):
    """Read one dump: header dict and data array."""
    header = {}
    with open(fname) as f:
        for line in f:
            if not line.startswith("#"):
                break
            words = line[1:].split()
            if words[0] == "sizeof":
                header["sizeof"] = {
                    words[i]: int(words[i + 1]) for i in range(1, len(words), 2)
                }
            elif words[0] == "rid":
                header["columns"] = words
            else:
                for i in range(0, len(words) - 1, 2):
                    header[words[i]] = int(words[i + 1])
    data = np.loadtxt(fname, ndmin=2)
    return header, data


def kernel_flops(kernels, counter, subtype):
    """FLOPs of one evaluation of the kernel behind a counter, or None."""
    k = kernels["kernels"]

    def get(name):
        if name not in k:
            return None
        v = k[name]
        return v["flops"] if "flops" in v else None

    if counter.startswith("hydro"):
        sym = "sym" if counter == "hydro_iact_sym" else "nonsym"
        if subtype == "none":
            # The ghost tasks re-run the density loop on unconverged particles
            subtype = "density"
        f = get("hydro_%s_%s" % (subtype, sym))
        if f is None:
            return None
        # A symmetric call does the work of two non-symmetric ones, including
        # the shared distance computation only once
        return f + LOOP_FLOPS["hydro"]
    if counter.startswith("grav_p2p"):
        # The Newtonian branch is taken by all but the closest pairs
        return get("%s_newton" % counter) + LOOP_FLOPS["grav_p2p"]
    if counter.startswith("grav_m2p"):
        return get("%s_newton" % counter) + LOOP_FLOPS["grav_m2p"]
    if counter == "grav_m2l":
        return get("grav_m2l_periodic")
    return None


def task_bytes(row, col, sizes, name):
    """(unique, streamed) bytes of a task (see module docstring)."""

    def get(c):
        return row[col[c]] if c in col else 0.0

    hcells = get("ci_hcount") + get("cj_hcount")
    gcells = get("ci_gcount") + get("cj_gcount")
    hits = get("hydro_iact_sym") * 2 + get("hydro_iact_nonsym")
    loaded = get("grav_parts_loaded")
    m2l = get("grav_m2l")

    unique = streamed = 0.0
    if hits > 0:
        unique += 2.0 * min(hcells, hits) * sizes["part"]
        unique += min(hcells, hits) * sizes["sort_entry"]
        streamed += hits * sizes["part"]
    if loaded > 0:
        unique += 2.0 * min(gcells, loaded) * sizes["gpart"]
        streamed += loaded * sizes["gpart"]
    if m2l > 0:
        unique += m2l * sizes["gravity_tensors"]
        streamed += m2l * sizes["gravity_tensors"]
    return unique, streamed


def task_name(ttype, subtype):
    t = TASKTYPES[ttype]
    s = SUBTYPES[subtype]
    return t if s == "none" else "%s/%s" % (t, s)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("files", nargs="+", help="task_perf-step*.dat files")
    parser.add_argument("--kernels", required=True, help="kernel_flops.py JSON")
    parser.add_argument("--machine", help="machine_peak JSON (for the roofline)")
    parser.add_argument("--out", default="task_perf", help="Output file prefix")
    args = parser.parse_args()

    with open(args.kernels) as f:
        kernels = json.load(f)
    machine = None
    if args.machine:
        with open(args.machine) as f:
            machine = json.load(f)

    files = sorted(
        sum((glob.glob(f) for f in args.files), []),
        key=lambda s: int(s.split("step")[-1].split(".")[0]),
    )

    stats = {}
    steps = []
    sizes = None
    for fname in files:
        header, data = read_dump(fname)
        sizes = header["sizeof"]
        col = {c: i for i, c in enumerate(header["columns"])}
        cpufreq = float(header["cpufreq"])
        wall = (header["toc_step"] - header["tic_step"]) / cpufreq
        busy = 0.0
        nthreads = int(data[:, col["rid"]].max()) + 1 if len(data) else 0
        for row in data:
            ttype, subtype = int(row[col["type"]]), int(row[col["subtype"]])
            name = task_name(ttype, subtype)
            dt = (row[col["toc"]] - row[col["tic"]]) / cpufreq
            busy += dt
            flops, unknown = 0.0, False
            sub = SUBTYPES[subtype]
            for c in COUNTERS:
                n = row[col[c]] if c in col else 0
                if n == 0 or c in NOT_INTERACTIONS:
                    continue
                f = kernel_flops(kernels, c, sub)
                if f is None:
                    unknown = True
                else:
                    flops += n * f
            s = stats.setdefault(
                name,
                {
                    "tasks": 0,
                    "time": 0.0,
                    "flops": 0.0,
                    "bytes": 0.0,
                    "streamed": 0.0,
                    "unknown": False,
                    **{c: 0 for c in COUNTERS},
                },
            )
            s["tasks"] += 1
            s["time"] += dt
            s["flops"] += flops
            unique, streamed = task_bytes(row, col, sizes, name)
            s["bytes"] += unique
            s["streamed"] += streamed
            s["unknown"] |= unknown
            for c in COUNTERS:
                s[c] += row[col[c]] if c in col else 0
        steps.append(
            {
                "file": os.path.basename(fname),
                "wall": wall,
                "busy": busy,
                "threads": nthreads,
                "updates": header.get("updates", 0),
                "g_updates": header.get("g_updates", 0),
            }
        )

    total = sum(s["time"] for s in stats.values())
    wall = sum(s["wall"] for s in steps)
    nthreads = max(s["threads"] for s in steps)

    lines = []
    lines.append("# %d steps, wall-clock %.3f s, %d threads" % (len(steps), wall, nthreads))
    lines.append(
        "# time in tasks %.3f s = %.1f%% of threads x wall-clock"
        % (total, 100.0 * total / (nthreads * wall))
    )
    lines.append(
        "# sizeof part %d B, gpart %d B, gravity_tensors %d B"
        % (sizes["part"], sizes["gpart"], sizes["gravity_tensors"])
    )
    lines.append(
        "%-34s %8s %9s %6s %12s %10s %9s %9s %8s %9s"
        % (
            "# task",
            "count",
            "time[s]",
            "%time",
            "interactions",
            "GFLOP",
            "GFLOP/s",
            "GB/s",
            "AI",
            "AI-strm",
        )
    )
    table = []
    for name, s in sorted(stats.items(), key=lambda x: -x[1]["time"]):
        inter = sum(s[c] for c in COUNTERS if c not in NOT_INTERACTIONS)
        gflop = s["flops"] / 1e9
        rate = gflop / s["time"] if s["time"] > 0 else 0.0
        bw = s["bytes"] / 1e9 / s["time"] if s["time"] > 0 else 0.0
        ai = s["flops"] / s["bytes"] if s["bytes"] > 0 else float("nan")
        ai_stream = (
            s["flops"] / s["streamed"] if s["streamed"] > 0 else float("nan")
        )
        table.append(
            {
                "task": name,
                "count": s["tasks"],
                "time": s["time"],
                "fraction": s["time"] / total,
                "interactions": inter,
                "gflop": gflop,
                "gflops_per_thread": rate,
                "gbytes_per_s_per_thread": bw,
                "flop_per_byte": ai,
                "flop_per_streamed_byte": ai_stream,
                "flops_incomplete": s["unknown"],
                **{c: s[c] for c in COUNTERS},
            }
        )
        if s["time"] / total < 1e-3:
            continue
        lines.append(
            "%-34s %8d %9.3f %6.2f %12.4g %10.3f %9.2f %9.2f %8.2f %9.2f%s"
            % (
                name,
                s["tasks"],
                s["time"],
                100.0 * s["time"] / total,
                inter,
                gflop,
                rate,
                bw,
                ai,
                ai_stream,
                " *" if s["unknown"] else "",
            )
        )
    lines.append("# GFLOP/s and GB/s (unique bytes) are per thread: FLOPs or bytes / time in task")
    lines.append("# AI = FLOP / unique byte, AI-strm = FLOP / streamed byte")
    lines.append("# * : some interactions have no FLOP count (kernel not in --kernels)")
    print("\n".join(lines))

    with open(args.out + "_summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(args.out + "_summary.json", "w") as f:
        json.dump({"steps": steps, "tasks": table, "sizeof": sizes}, f, indent=1)

    # Roofline
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(8, 5.5))
    pts = [t for t in table if t["gflop"] > 0 and t["fraction"] > 1e-3]
    for t in pts:
        ax.scatter(t["flop_per_byte"], t["gflops_per_thread"], s=20 + 600 * t["fraction"])
        ax.annotate(
            "%s (%.0f%%)" % (t["task"], 100 * t["fraction"]),
            (t["flop_per_byte"], t["gflops_per_thread"]),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )
    if machine:
        x = np.logspace(-2, 3, 200)
        peak = machine["gflops_per_core"]
        bw = machine["gbytes_per_s_per_core"]
        ax.plot(x, np.minimum(peak, bw * x), "k-", lw=1, label="1 core roofline (SIMD FMA)")
        if "gflops_per_core_scalar" in machine:
            ax.axhline(machine["gflops_per_core_scalar"], color="k", ls=":", lw=1,
                       label="1 core scalar FMA peak")
        if "gbytes_per_s_node" in machine and nthreads:
            share = machine["gbytes_per_s_node"] / nthreads
            ax.plot(x, np.minimum(peak, share * x), "k--", lw=1,
                    label="node bandwidth / %d threads" % nthreads)
        ax.legend(loc="lower right", fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity [FLOP / unique byte]")
    ax.set_ylabel("GFLOP/s per thread")
    ax.set_title("SWIFT tasks (marker area ~ fraction of task time)")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out + "_roofline.png", dpi=150)

    # Cost and thread utilisation of each step versus its number of updates
    upd = np.array([max(st["updates"], st["g_updates"], 1) for st in steps])
    wall_s = np.array([st["wall"] for st in steps])
    util = np.array(
        [st["busy"] / (st["threads"] * st["wall"]) if st["wall"] > 0 and st["threads"] else 0
         for st in steps]
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.scatter(upd, 1e3 * wall_s, s=8)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Particles updated in the step (max of gas, gravity)")
    ax1.set_ylabel("Step wall-clock [ms]")
    ax1.grid(True, which="both", alpha=0.3)
    ax2.scatter(upd, 100 * util, s=8)
    ax2.set_xscale("log")
    ax2.set_ylim(0, 100)
    ax2.set_xlabel("Particles updated in the step (max of gas, gravity)")
    ax2.set_ylabel("Time in tasks / (threads x wall-clock) [%]")
    ax2.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out + "_steps.png", dpi=150)


if __name__ == "__main__":
    sys.exit(main())
