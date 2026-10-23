#!/usr/bin/env python3
"""
Count the floating-point operations of one evaluation of each SWIFT
interaction kernel, as compiled for the configured build.

The kernels are wrapped in kernel_flops.c, compiled with the same flags as
the build, and the disassembly of each wrapper is classified instruction by
instruction. Scalar and SIMD instructions are both handled (SIMD ones are
weighted by their number of lanes) for AArch64 and x86-64.

FLOP weights follow the usual convention: add, sub, mul, div, sqrt, min, max
and reciprocal (square root) estimates count as 1, fused multiply-adds and
the AArch64 Newton-Raphson step instructions (frecps/frsqrts) count as 2.
Moves, conversions, comparisons, selects, abs and neg count as 0. Calls to
external maths functions (e.g. expf) are reported separately.

A wrapper that still contains conditional branches after the optimiser has
removed the paths excluded by its guard is flagged: its count is then the
sum over all remaining paths (an upper bound).

Usage:
    kernel_flops.py <build_dir> [--json out.json]
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "src"))

# AArch64 scalar/vector FP mnemonics -> FLOP weight per lane
A64 = {
    "fadd": 1, "fsub": 1, "fmul": 1, "fnmul": 1, "fdiv": 1, "fsqrt": 1,
    "fmax": 1, "fmin": 1, "fmaxnm": 1, "fminnm": 1, "fabd": 1,
    "fmadd": 2, "fmsub": 2, "fnmadd": 2, "fnmsub": 2, "fmla": 2, "fmls": 2,
    "frecpe": 1, "frsqrte": 1, "frecps": 2, "frsqrts": 2,
    "faddp": 1,
}
A64_ZERO = re.compile(
    r"^(fmov|fcvt\w*|scvtf|ucvtf|fcmp\w*|fccmp\w*|fcsel|fabs|fneg|frint\w*|fcm\w+)$"
)

# x86-64: strip the leading 'v' and the ss/sd/ps/pd suffix
X86 = {
    "add": 1, "sub": 1, "mul": 1, "div": 1, "sqrt": 1, "min": 1, "max": 1,
    "rcp": 1, "rsqrt": 1,
}
X86_FMA = re.compile(r"^v?f(n?)m(add|sub)(132|213|231)(ss|sd|ps|pd)$")
X86_OP = re.compile(r"^v?(add|sub|mul|div|sqrt|min|max|rcp|rsqrt)(14)?(ss|sd|ps|pd)$")


def split_a64(mnemonic, operands):
    """Split an AArch64 FP instruction into its base mnemonic and lanes.

    Handles both the GNU syntax (fmul v0.4s, ...) and the Apple one, where
    the arrangement is a suffix of the mnemonic (fmul.4s v0, ...)."""
    base, _, suffix = mnemonic.partition(".")
    m = re.match(r"^(\d+)[sdh]$", suffix)
    if m:
        return base, int(m.group(1))
    if suffix:
        # Scalar or by-element form (fmul.s s0, s1, v2[1])
        return base, 1
    m = re.search(r"\bv\d+\.(\d+)[sdh]\b", operands)
    return base, int(m.group(1)) if m else 1


def lanes_x86(mnemonic, operands):
    if mnemonic.endswith(("ss", "sd")):
        return 1
    width = 16 if "%xmm" in operands else 32 if "%ymm" in operands else 64
    width = 32 if "%ymm" in operands else width
    width = 64 if "%zmm" in operands else width
    per = 4 if mnemonic.endswith("ps") else 8
    return width // per


def classify(mnemonic, operands, arch):
    """Return (flops, is_fp_instruction)."""
    if arch == "aarch64":
        base, lanes = split_a64(mnemonic, operands)
        if base in A64:
            return A64[base] * lanes, True
        if A64_ZERO.match(base):
            return 0, True
        return 0, False
    m = X86_FMA.match(mnemonic)
    if m:
        return 2 * lanes_x86(mnemonic, operands), True
    m = X86_OP.match(mnemonic)
    if m:
        return X86[m.group(1)] * lanes_x86(mnemonic, operands), True
    return 0, False


def build_flags(build_dir):
    """Recover the compiler and flags used by the build."""
    cc, cflags, incs = "cc", "", []
    with open(os.path.join(build_dir, "src", "Makefile")) as f:
        for line in f:
            m = re.match(r"^(\w+) = (.*)$", line.rstrip("\n"))
            if not m:
                continue
            var, val = m.groups()
            if var == "CC":
                cc = val.strip()
            elif var == "CFLAGS":
                cflags = val.strip()
            elif var == "CPPFLAGS" or var.endswith(("_CPPFLAGS", "_INCS")):
                # Third-party include paths (HDF5, GSL, FFTW, ...)
                incs += [x for x in shlex.split(val) if x.startswith("-I")]
    # LTO would leave bitcode in the object file and -Werror is not our business
    flags = [x for x in shlex.split(cflags) if x not in ("-flto", "-Werror")]
    return cc, incs + flags


def disassemble(obj):
    """Return {function name: [(mnemonic, operands, call target)]}."""
    out = subprocess.run(
        ["objdump", "-dr", "--no-show-raw-insn", obj],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    funcs, name = {}, None
    for line in out.splitlines():
        m = re.match(r"^[0-9a-f]+ <_?([\w.]+)>:", line)
        if m:
            name = m.group(1)
            funcs[name] = []
            continue
        if name is None:
            continue
        # Relocation (GNU: R_AARCH64_CALL26, Apple: ARM64_RELOC_BRANCH26)
        reloc = re.match(
            r"^\s+[0-9a-f]+:\s+(?:R_|ARM64_RELOC_|X86_64_RELOC_)\w+\s+_?([\w.]+)",
            line,
        )
        if reloc and funcs[name]:
            mnemonic, operands, _ = funcs[name][-1]
            funcs[name][-1] = (mnemonic, operands, reloc.group(1))
            continue
        m = re.match(r"^\s+[0-9a-f]+:\s+(\S+)\s*(.*)$", line)
        if m:
            mnemonic, operands = m.groups()
            # Direct call or tail call to another function of this object
            target = re.search(r"<_?([\w.]+)>$", operands)
            target = target.group(1) if target else None
            funcs[name].append((mnemonic, operands, target))
    return funcs


CALLS = ("bl", "call", "callq", "b", "jmp", "jmpq")
BRANCHES = re.compile(r"^(b\.\w+|cbn?z|tbn?z|j(?!mp)\w+)$")


def analyse(name, funcs, arch, stack=()):
    """Count the operations of a function, including the local functions it
    calls (the compiler sometimes refuses to inline large ones)."""
    res = {
        "flops": 0,
        "fp_instructions": 0,
        "instructions": 0,
        "calls": set(),
        "conditional_branches": 0,
    }
    for mnemonic, operands, target in funcs[name]:
        res["instructions"] += 1
        f, is_fp = classify(mnemonic, operands, arch)
        res["flops"] += f
        res["fp_instructions"] += is_fp
        if BRANCHES.match(mnemonic):
            res["conditional_branches"] += 1
        if mnemonic in CALLS and target and target != name:
            if target in funcs and target not in stack:
                sub = analyse(target, funcs, arch, stack + (name,))
                for k in ("flops", "fp_instructions", "instructions",
                          "conditional_branches"):
                    res[k] += sub[k]
                res["calls"] |= set(sub["calls"])
            elif target not in funcs:
                res["calls"].add(target)
    res["calls"] = sorted(res["calls"])
    return res


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("build_dir", help="A configured and built SWIFT build directory")
    parser.add_argument("--json", help="Write the results to this JSON file")
    args = parser.parse_args()

    build_dir = os.path.abspath(args.build_dir)
    cc, flags = build_flags(build_dir)
    arch = os.uname().machine
    arch = "aarch64" if arch in ("arm64", "aarch64") else "x86_64"

    with tempfile.TemporaryDirectory() as tmp:
        obj = os.path.join(tmp, "kernel_flops.o")
        cmd = (
            [cc, "-DHAVE_CONFIG_H", "-I" + build_dir, "-I" + SRC]
            + flags
            + ["-w", "-c", os.path.join(HERE, "kernel_flops.c"), "-o", obj]
        )
        subprocess.run(cmd, check=True)
        funcs = disassemble(obj)

    results = {
        name[3:]: analyse(name, funcs, arch)
        for name in funcs
        if name.startswith("kf_")
    }

    print("# arch: %s, compiler: %s" % (arch, cc))
    print("# flags: %s" % " ".join(flags))
    print(
        "%-28s %7s %7s %7s %8s  %s"
        % ("# kernel", "flops", "fp-ins", "ins", "branches", "calls")
    )
    for name, r in results.items():
        print(
            "%-28s %7d %7d %7d %8d  %s"
            % (
                name,
                r["flops"],
                r["fp_instructions"],
                r["instructions"],
                r["conditional_branches"],
                ",".join(r["calls"]),
            )
        )

    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {"arch": arch, "compiler": cc, "flags": flags, "kernels": results},
                f,
                indent=2,
            )


if __name__ == "__main__":
    sys.exit(main())
