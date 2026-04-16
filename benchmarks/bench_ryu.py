"""bench_ryu.py - Float-to-string conversion benchmark for CPython Ryu integration.

Measures ns/op for all float formatting paths:
  repr/str, %e, %f, %g, f-strings, round()

Usage:
    python bench_ryu.py                     # human-readable table
    python bench_ryu.py --label main        # table + JSON line tagged "main"
    python bench_ryu.py --json              # JSON only (for bench_ryu_compare.py)

Output JSON line format:
    {"label": "<label>", "results": {<case>: {"ns_per_op": ..., "min_ms": ...}, ...}}
"""

import timeit
import math
import sys
import json
import argparse

# ---------------------------------------------------------------------------
# Test values: a representative mix of float categories
# ---------------------------------------------------------------------------
FLOATS = [
    # small integers
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
    11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
    # common fractions
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
    # well-known constants
    math.pi, math.e, 1/3, 2/3, math.sqrt(2),
    # large values
    1e100, 9.9e307, 1.23456789e200,
    # small values
    1e-100, 5e-324,  # min subnormal
    2.2250738585072014e-308,  # min normal
    # negative
    -1.5, -math.pi, -1e100,
    # special
    float('inf'), float('-inf'), float('nan'),
]

N_FLOATS = len(FLOATS)

# ---------------------------------------------------------------------------
# Benchmark cases: (label, setup_extra, stmt)
# setup always has 'floats' available.
# ---------------------------------------------------------------------------
CASES = [
    ("repr(x)  [shortest]",      "", "[repr(x) for x in floats]"),
    ("str(x)   [shortest]",      "", "[str(x) for x in floats]"),
    ("'%.6e' % x",               "", "['%.6e' % x for x in floats]"),
    ("'%.2e' % x",               "", "['%.2e' % x for x in floats]"),
    ("'%.3f' % x",               "", "['%.3f' % x for x in floats]"),
    ("'%.6f' % x",               "", "['%.6f' % x for x in floats]"),
    ("'%.10f' % x",              "", "['%.10f' % x for x in floats]"),
    ("'%g' % x",                 "", "['%g' % x for x in floats]"),
    ("'%.4g' % x",               "", "['%.4g' % x for x in floats]"),
    ("f'{x:.3f}'",               "", "[f'{x:.3f}' for x in floats]"),
    ("f'{x:.6g}'",               "", "[f'{x:.6g}' for x in floats]"),
    ("f'{x!r}'",                 "", "[f'{x!r}' for x in floats]"),
    ("f'{x}'",                   "", "[f'{x}' for x in floats]"),
    ("round(x, 2)",              "", "[round(x, 2) for x in floats]"),
    ("round(x, 6)",              "", "[round(x, 6) for x in floats]"),
    ("round(x, -2)  [Gay fallback]", "", "[round(x, -2) for x in floats]"),
    ("repr(inf/nan)",
     "specials = [float('inf'), float('-inf'), float('nan')]",
     "[repr(x) for x in specials]"),
]

REPEATS = 7       # timeit.repeat count
NUMBER  = 500     # iterations per repeat


def run_benchmarks():
    results = {}
    col = max(len(c[0]) for c in CASES) + 2
    print(f"Python {sys.version}")
    print(f"{'Case':<{col}} {'ns/op':>10}  {'min ms':>8}")
    print("-" * (col + 22))

    for label, setup_extra, stmt in CASES:
        # Build floats list in setup via import+literal to avoid repr issues (inf, nan)
        setup = "import math; floats = " + repr(FLOATS).replace("inf", "float('inf')").replace("nan", "float('nan')") + "\n"
        if setup_extra:
            setup += setup_extra + "\n"

        times = timeit.repeat(stmt, setup=setup, repeat=REPEATS, number=NUMBER)
        # Each time is total seconds for NUMBER iterations over N_FLOATS values.
        # ns per individual float operation:
        min_ms = min(times) * 1000
        ns_per_op = min(times) / (NUMBER * N_FLOATS) * 1e9

        print(f"  {label:<{col}} {ns_per_op:>9.1f}  {min_ms:>8.1f}")
        results[label] = {"ns_per_op": ns_per_op, "min_ms": min_ms}

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", default="", help="Tag for JSON output (e.g. 'main' or 'ryu')")
    parser.add_argument("--json", action="store_true", help="Suppress table, emit only JSON")
    args = parser.parse_args()

    if args.json:
        # Suppress table output
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            results = run_benchmarks()
    else:
        results = run_benchmarks()

    if args.label or args.json:
        record = {"label": args.label or "unlabeled", "results": results}
        print(json.dumps(record))


if __name__ == "__main__":
    main()
