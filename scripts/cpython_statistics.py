"""Gather statistics from pystats files"""

import glob
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from rich import print as rprint


def strip_lead(d, split="#"):
    def strip(k):
        return k.split(split)[1]

    return {strip(k): v for k, v in d.items()}


def to_markdown(
    d,
    header,
):
    txt = f"| {header} | Counts|\n"
    txt += "| ---------| -------|\n"
    for k, v in d.items():
        if v > 1e6:
            vs = f"{v / 1e6:.1f}M"
        else:
            vs = v
        txt += f"| {k} | {vs} |\n"
    return txt


if sys.platform == "linux":
    stats_folder = Path(r"/tmp/py_stats")
    stats_folder = Path(r"/home/eendebakpt/py_stats")
else:
    stats_folder = Path(r"c:\temp\py_stats")

files = glob.glob("*txt", root_dir=stats_folder)

# print(files)
print(f"found {len(files)} files to analyze")

# %%
# for f in files:
#     with open(stats_folder / f) as fid:
#         l = fid.readlines()


def gather_statistics(f, results=None):
    with open(stats_folder / f) as fid:
        lines = fid.readlines()
    if results is None:
        results = {}
    for line in lines:
        line = line.strip()
        tag, count = line.split(":")
        count = int(count)
        results[tag] = results.get(tag, 0) + count
    return results


results = {}
for f in files:
    gather_statistics(f, results)

# results = gather_statistics(f, results = results)
# results


# %%
def sort_values(d):
    return dict(sorted(d.items(), key=lambda x: x[1], reverse=True))


def get_subset(results, start: str, minimal_count=0, not_contains=None):
    subset = {k: v for k, v in results.items() if k.startswith(start) and v >= minimal_count}
    if not_contains:
        subset = {k: v for k, v in subset.items() if not_contains not in k}
    subset = sort_values(subset)

    return subset


tap = results["Object allocations"]

alloc_size_results = get_subset(results, "Object allocations of size")
rprint(alloc_size_results)

object_results = get_subset(results, "Object", minimal_count=0, not_contains="of size")

freelist_results = get_subset(results, "Freelist")
alloc_results = get_subset(results, "Alloc", minimal_count=tap // 5000)

bfm = get_subset(results, "PyCMethod_New", minimal_count=1200)


rprint(alloc_results)
rprint(freelist_results)

ta = sum(alloc_results.values())
print(f"total allocations tracked: {ta // 1e6}M/{tap // 1e6}M")

# %%

rprint(bfm)


def strip_lead_ml(d):
    def strip(k):
        z = k.split(" ml ")[1]
        z = z.split(" self ")
        q = z[1]
        return q + "." + z[0]

    return {strip(k): v for k, v in d.items()}


bfm = get_subset(results, "PyCMethod_New", minimal_count=500_000)
s = to_markdown(strip_lead_ml(bfm), "Allocations via PyCMethod_New")
print(s)
# %%
rprint({k: v for k, v in alloc_results.items() if "tuple" in k})

# %%
w = get_subset(results, "tuple")
rprint(w)
w = get_subset(results, "list", minimal_count=100_00)

rprint(w)

plt.rcParams["axes.labelsize"] = 14
plt.rcParams["font.size"] = 12
plt.figure(10)
plt.clf()
for sz in range(0, 100):
    v = results.get(f"list_new_prealloc_size_{sz}", 0)
    plt.semilogy(sz, v, ".C0")
plt.xlabel("List preallocation size")
plt.ylabel("Frequency (log scale)")


alloc_results = get_subset(results, "Alloc", minimal_count=1_000_000)


s = ""
s += to_markdown(strip_lead(alloc_results), "Allocations")
s += to_markdown(strip_lead(freelist_results), "Freelist allocations")
print(s)
