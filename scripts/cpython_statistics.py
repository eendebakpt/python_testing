"""Gather statistics from pystats files"""

import glob
from pathlib import Path

from rich import print as rprint

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


def get_subset(results, start: str):
    subset = {k: v for k, v in results.items() if k.startswith(start)}
    subset = sort_values(subset)

    return subset


alloc_size_results = get_subset(results, "Object allocations of size")
rprint(alloc_size_results)


freelist_results = get_subset(results, "Freelist")
alloc_results = get_subset(results, "Alloc")

rprint(alloc_results)
rprint(freelist_results)
