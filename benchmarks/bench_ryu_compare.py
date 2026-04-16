"""bench_ryu_compare.py - Compare two bench_ryu.py JSON result files side by side.

Usage:
    # Run benchmarks on two builds, save JSON lines:
    path/to/main/python bench_ryu.py --label main --json > main.json
    path/to/ryu/python  bench_ryu.py --label ryu  --json > ryu.json

    # Compare:
    python bench_ryu_compare.py main.json ryu.json

    # Or compare using a combined file (two JSON lines):
    cat main.json ryu.json > combined.json
    python bench_ryu_compare.py combined.json

Options:
    --threshold FLOAT   Flag regressions worse than this ratio (default: 0.95)
    --md                Emit a Markdown table instead of plain text
"""

import json
import sys
import argparse
import math


def load_result(path):
    """Load one or more JSON-line result records from a file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def geomean(values):
    if not values:
        return float("nan")
    log_sum = sum(math.log(v) for v in values if v > 0)
    return math.exp(log_sum / len(values))


def compare(base_results, new_results, threshold=0.95, md=False):
    all_cases = list(base_results.keys())

    if md:
        header = f"| {'Case':<45} | {'base (ns/op)':>13} | {'new (ns/op)':>13} | {'Speedup':>9} |"
        sep    = f"|{'-'*47}|{'-'*15}|{'-'*15}|{'-'*11}|"
        print(header)
        print(sep)
    else:
        col = max(len(c) for c in all_cases) + 2
        print(f"  {'Case':<{col}} {'base':>10}  {'new':>10}  {'speedup':>9}  {'flag'}")
        print("-" * (col + 38))

    speedups = []
    regressions = []

    for case in all_cases:
        base_ns = base_results.get(case, {}).get("ns_per_op")
        new_ns  = new_results.get(case, {}).get("ns_per_op")
        if base_ns is None or new_ns is None:
            continue

        speedup = base_ns / new_ns
        speedups.append(speedup)

        flag = ""
        if speedup < threshold:
            flag = "REGRESSION"
            regressions.append((case, speedup))

        if md:
            marker = " **<--**" if speedup < threshold else ""
            print(f"| {case:<45} | {base_ns:>13.1f} | {new_ns:>13.1f} | {speedup:>8.2f}x{marker} |")
        else:
            col = max(len(c) for c in all_cases) + 2
            print(f"  {case:<{col}} {base_ns:>10.1f}  {new_ns:>10.1f}  {speedup:>8.2f}x  {flag}")

    gm = geomean(speedups)
    if md:
        print(f"|                                                | {'':>13} | {'':>13} | {'':>9} |")
        print(f"| **Geomean**                                    | {'':>13} | {'':>13} | **{gm:.2f}x** |")
    else:
        print()
        print(f"  Geomean speedup: {gm:.2f}x")

    if regressions:
        print()
        print("Regressions (speedup < threshold):")
        for case, sp in regressions:
            print(f"  {case}: {sp:.2f}x")
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help="One or two JSON result files")
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="Flag speedup < threshold as regression (default: 0.95)")
    parser.add_argument("--md", action="store_true", help="Emit Markdown table")
    args = parser.parse_args()

    records = []
    for path in args.files:
        records.extend(load_result(path))

    if len(records) < 2:
        print("Error: need at least 2 result records (two JSON lines or two files)", file=sys.stderr)
        sys.exit(1)

    # Use the first two records
    base, new = records[0], records[1]
    base_label = base.get("label", "base")
    new_label  = new.get("label", "new")

    print(f"Comparing: {base_label!r} (base) vs {new_label!r} (new)")
    print()

    rc = compare(base["results"], new["results"], threshold=args.threshold, md=args.md)
    sys.exit(rc)


if __name__ == "__main__":
    main()
