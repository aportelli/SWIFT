#!/usr/bin/env python3
"""
Count the floating-point operations of one evaluation of each SWIFT
interaction kernel directly from the C source ("theoretical" FLOPs), to be
compared with the compiler-generated ones counted by kernel_flops.py.

The wrappers of kernel_flops.c are parsed by clang, and the resulting AST is
walked from each wrapper, following the calls to the (inline) functions
defined in SWIFT. Only operations on float and double values are counted:

  - add/sub: binary + and -, compound += and -=
  - mul:     binary *, compound *=
  - div:     binary /, compound /=
  - cmp:     comparisons (they become min/max or selects in the binary)
  - sqrt:    calls to sqrt/sqrtf
  - math:    calls to other maths functions (expf, powf, ...)

Sub-expressions whose operands are all compile-time constants are not
counted, as the compiler folds them. Branches (if/else and ?:) are handled by
reporting the cheapest (min) and the most expensive (max) path. The guard of
the wrappers (the top-level if statement) is assumed to be taken. Loops with
constant bounds are unrolled, others are counted once and flagged.

FLOPs = add/sub + mul + div + sqrt (+ cmp, reported separately).

Usage:
    source_flops.py <build_dir> [--json out.json]
"""

import argparse
import json
import os
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "src"))

CATEGORIES = ("addsub", "mul", "div", "sqrt", "cmp", "math")
FLOAT_TYPES = ("float", "double", "const float", "const double")
SQRT = ("sqrt", "sqrtf")
MATHS = (
    "exp", "expf", "pow", "powf", "cbrt", "cbrtf", "log", "logf", "log10",
    "log10f", "erfc", "erfcf", "sin", "sinf", "cos", "cosf", "sinh", "sinhf",
    "exp10", "exp10f", "tanh", "tanhf",
)
# Calls that are free or are not maths (fabs, fmin, ... count as compares)
FREE = ("fabs", "fabsf")
MINMAX = ("fmin", "fminf", "fmax", "fmaxf")


class Cost:
    """A vector of operation counts."""

    def __init__(self, **kw):
        self.c = {k: kw.get(k, 0) for k in CATEGORIES}

    def __add__(self, other):
        r = Cost()
        r.c = {k: self.c[k] + other.c[k] for k in CATEGORIES}
        return r

    def __mul__(self, n):
        r = Cost()
        r.c = {k: self.c[k] * n for k in CATEGORIES}
        return r

    def flops(self):
        return sum(self.c[k] for k in ("addsub", "mul", "div", "sqrt"))


class Range:
    """The [min, max] cost over the paths through some code."""

    def __init__(self, lo=None, hi=None):
        self.lo = lo if lo is not None else Cost()
        self.hi = hi if hi is not None else Cost()

    def __add__(self, other):
        return Range(self.lo + other.lo, self.hi + other.hi)

    def __mul__(self, n):
        return Range(self.lo * n, self.hi * n)

    @staticmethod
    def either(a, b):
        lo = a.lo if a.lo.flops() <= b.lo.flops() else b.lo
        hi = a.hi if a.hi.flops() >= b.hi.flops() else b.hi
        return Range(lo, hi)


def qual_type(node):
    t = node.get("type", {})
    return t.get("desugaredQualType", t.get("qualType", ""))


def is_float(node):
    return qual_type(node) in FLOAT_TYPES


def is_const(node):
    """Is this expression a compile-time constant?"""
    kind = node.get("kind")
    if kind in ("IntegerLiteral", "FloatingLiteral", "CharacterLiteral"):
        return True
    if kind == "DeclRefExpr":
        return node.get("referencedDecl", {}).get("kind") == "EnumConstantDecl"
    if kind in (
        "ImplicitCastExpr",
        "CStyleCastExpr",
        "ParenExpr",
        "UnaryOperator",
        "BinaryOperator",
        "ConstantExpr",
    ):
        return all(is_const(c) for c in node.get("inner", []))
    return False


def const_value(node):
    """Evaluate an integer constant expression (or return None)."""
    kind = node.get("kind")
    if kind == "IntegerLiteral":
        return int(node["value"])
    if kind == "ConstantExpr" and "value" in node:
        return int(node["value"])
    if kind in ("ImplicitCastExpr", "CStyleCastExpr", "ParenExpr", "ConstantExpr"):
        return const_value(node["inner"][0])
    if kind == "BinaryOperator":
        a, b = (const_value(c) for c in node["inner"])
        if a is None or b is None:
            return None
        op = node["opcode"]
        ops = {"+": a + b, "-": a - b, "*": a * b}
        if op in ops:
            return ops[op]
        if op == "/" and b:
            return a // b
    if kind == "UnaryOperator" and node["opcode"] == "-":
        v = const_value(node["inner"][0])
        return -v if v is not None else None
    return None


class Walker:
    def __init__(self, ast):
        self.functions = {}
        for decl in ast["inner"]:
            if decl.get("kind") == "FunctionDecl" and self.body(decl):
                self.functions[decl["name"]] = decl
        self.cache = {}
        self.warnings = set()

    @staticmethod
    def body(decl):
        for c in decl.get("inner", []):
            if c.get("kind") == "CompoundStmt":
                return c
        return None

    def function(self, name):
        if name not in self.cache:
            self.cache[name] = Range()  # Guard against recursion
            self.cache[name] = self.walk(self.body(self.functions[name]))
        return self.cache[name]

    def callee(self, call):
        node = call["inner"][0]
        while node.get("kind") in ("ImplicitCastExpr", "ParenExpr"):
            node = node["inner"][0]
        if node.get("kind") == "DeclRefExpr":
            return node["referencedDecl"]["name"]
        return None

    def loop_trips(self, node):
        """Trip count of for (int k = a; k < b (or <= b); k++)."""
        try:
            init, _, cond, inc, _ = node["inner"]
            var = init["inner"][0]
            start = const_value(var["inner"][0])
            op = cond["opcode"]
            end = const_value(cond["inner"][1])
            if inc.get("opcode") not in ("++", "postfix ++") and not inc.get(
                "isPostfix", False
            ):
                return None
            if start is None or end is None:
                return None
            if op == "<":
                return max(end - start, 0)
            if op == "<=":
                return max(end - start + 1, 0)
        except (KeyError, ValueError, TypeError, IndexError):
            pass
        return None

    def walk(self, node, guard=False):
        if node is None:
            return Range()
        kind = node.get("kind")
        inner = node.get("inner", [])

        if kind == "IfStmt":
            cond = self.walk(inner[0])
            then = self.walk(inner[1])
            if guard:
                return cond + then
            other = self.walk(inner[2]) if len(inner) > 2 else Range()
            return cond + Range.either(then, other)

        if kind == "ConditionalOperator":
            cond = self.walk(inner[0])
            return cond + Range.either(self.walk(inner[1]), self.walk(inner[2]))

        if kind == "ForStmt":
            trips = self.loop_trips(node)
            if trips is None:
                self.warnings.add("loop with unknown trip count counted once")
                trips = 1
            return self.walk(inner[2]) + self.walk(inner[4]) * trips

        if kind in ("WhileStmt", "DoStmt"):
            self.warnings.add("while loop counted once")

        if kind in ("BinaryOperator", "CompoundAssignOperator"):
            op = node["opcode"]
            cost = sum((self.walk(c) for c in inner), Range())
            if kind == "BinaryOperator" and is_const(node):
                return Range()
            if op in ("<", ">", "<=", ">="):
                # Comparisons have an int type: look at the operands
                is_fp = any(is_float(self.strip(c)) for c in inner)
            else:
                is_fp = is_float(node)
            if not is_fp:
                return cost
            cat = {
                "+": "addsub", "-": "addsub", "*": "mul", "/": "div",
                "+=": "addsub", "-=": "addsub", "*=": "mul", "/=": "div",
                "<": "cmp", ">": "cmp", "<=": "cmp", ">=": "cmp",
            }.get(op)
            if cat:
                cost = cost + Range(Cost(**{cat: 1}), Cost(**{cat: 1}))
            return cost

        if kind == "CallExpr":
            name = self.callee(node)
            args = sum((self.walk(c) for c in inner[1:]), Range())
            if name in SQRT:
                return args + Range(Cost(sqrt=1), Cost(sqrt=1))
            if name in MINMAX:
                return args + Range(Cost(cmp=1), Cost(cmp=1))
            if name in FREE:
                return args
            if name in MATHS:
                self.warnings.add("calls to %s" % name)
                return args + Range(Cost(math=1), Cost(math=1))
            if name in self.functions:
                return args + self.function(name)
            return args

        return sum((self.walk(c) for c in inner), Range())

    @staticmethod
    def strip(node):
        while node.get("kind") in ("ImplicitCastExpr", "ParenExpr") and node.get(
            "inner"
        ):
            node = node["inner"][0]
        return node

    def wrapper(self, name):
        """Cost of a wrapper, whose top-level if statement is a guard."""
        body = self.body(self.functions[name])
        return sum((self.walk(s, guard=True) for s in body.get("inner", [])), Range())


def build_flags(build_dir):
    sys.path.insert(0, HERE)
    from kernel_flops import build_flags as kf_build_flags

    cc, flags = kf_build_flags(build_dir)
    # Only the pre-processor flags matter for the AST
    return [f for f in flags if f.startswith(("-I", "-D", "-ffast-math"))]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("build_dir", help="A configured SWIFT build directory")
    parser.add_argument("--json", help="Write the results to this JSON file")
    args = parser.parse_args()

    build_dir = os.path.abspath(args.build_dir)
    cmd = (
        ["clang", "-DHAVE_CONFIG_H", "-I" + build_dir, "-I" + SRC]
        + build_flags(build_dir)
        + ["-w", "-fsyntax-only", "-Xclang", "-ast-dump=json"]
        + [os.path.join(HERE, "kernel_flops.c")]
    )
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    walker = Walker(json.loads(out))

    results = {}
    for name in walker.functions:
        if not name.startswith("kf_"):
            continue
        r = walker.wrapper(name)
        results[name[3:]] = {"min": r.lo.c, "max": r.hi.c}

    print(
        "%-28s %9s %9s   %s"
        % ("# kernel", "flops-min", "flops-max", "max path: addsub mul div sqrt cmp math")
    )
    for name, r in results.items():
        lo = sum(r["min"][k] for k in ("addsub", "mul", "div", "sqrt"))
        hi = sum(r["max"][k] for k in ("addsub", "mul", "div", "sqrt"))
        print(
            "%-28s %9d %9d   %s"
            % (name, lo, hi, " ".join("%d" % r["max"][k] for k in CATEGORIES))
        )
    for w in sorted(walker.warnings):
        print("# warning:", w)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    sys.exit(main())
