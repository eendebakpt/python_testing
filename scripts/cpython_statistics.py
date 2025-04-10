"""Gather statistics from pystats files"""

import glob
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from rich import print as rprint


def strip_lead(d, split="#"):
    def strip(k):
        return k.split(split)[1]

    return {strip(k): v for k, v in d.items()}


def sort_values(d):
    return dict(sorted(d.items(), key=lambda x: x[1], reverse=True))


def plot_hist(h, label, **kwargs):
    yy = list(h.values())
    xx = list(h.keys())
    kwargs = {"markersize": 10} | kwargs
    plt.plot(xx, yy, ".", label=label, **kwargs)


def sorted_dictionary(d):
    return {k: d[k] for k in sorted(d)}


def get_subset(results, start: str, minimal_count=0, not_contains=None):
    subset = {k: v for k, v in results.items() if k.startswith(start) and v >= minimal_count}
    if not_contains:
        subset = {k: v for k, v in subset.items() if not_contains not in k}
    subset = sort_values(subset)

    return subset


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
    # stats_folder = Path(r"/home/eendebakpt/py_stats")
else:
    stats_folder = Path(r"c:\temp\py_stats")

stats_folder = Path(r"/home/eendebakpt/freelist_py_stats")


files = glob.glob("*txt", root_dir=stats_folder)

# print(files)
if 0:
    latest_file = max([stats_folder / f for f in files], key=os.path.getctime)
    print(f"  {latest_file=}")
    files = [latest_file]

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
        count, tag = (line)[::-1].split(":", maxsplit=1)
        count = count[::-1]
        tag = tag[::-1]
        # tag, count = line.split(":")
        count = int(count)
        results[tag] = results.get(tag, 0) + count
    return results


results = {}
for f in files:
    gather_statistics(f, results)

# results = gather_statistics(f, results = results)
# results

# %%
from collections import Counter

w = get_subset(results, "small_list")
# rprint(w)


def make_hist(results, tag, maxsize=400):
    w = get_subset(results, tag)
    c = Counter()
    for k, v in w.items():
        data = k.removeprefix(tag)
        num = int(data)
        c[num] += v
    c = {k: v for k, v in c.items() if k <= maxsize}
    c = sorted_dictionary(c)
    return c


dealloc_hist = make_hist(results, tag="small_list_freelist normal/freelist deallocate")
small_dealloc_hist = make_hist(results, tag="small_list_freelist small deallocate")

normal_allocation_hist = make_hist(results, tag="small_list_freelist normal allocate")
freelist_allocation_hist = make_hist(results, tag="small_list_freelist freelist allocate")
small_allocation_hist = make_hist(results, tag="small_list_freelist small allocate")
resizes_hist = make_hist(results, tag="Resize list to")

# %% Freelist histograms
import numpy as np
from natsort import natsorted

fresults = {k: v for k, v in results.items() if "freelistsize" in k and "_PyFreeList_Pop" in k}
list(fresults)


def freelist_names(fresults):
    header = "_PyFreeList_Pop"
    names = set()
    for k in fresults:
        if not k.startswith(header):
            continue
        #        try:
        #            head, tail = k.split('[', 1)
        #        except:
        head, tail = k.split(":", 1)
        head = head[len(header) + 1 :]
        names.add(head)
    names = list(names)
    return natsorted(names)


names = freelist_names(fresults)
data = {}
for name in names:
    tag = "_PyFreeList_Pop " + name + ": freelistsize"
    h = make_hist(results, tag=tag)
    #    print(h)
    data[name] = h


def plot_barhist(h, label, **kwargs):
    yy = list(h.values())
    xx = list(h.keys())
    bins = np.arange(-0.5, max(yy) + 0.5 + 1e-6, 1)
    kwargs = {} | kwargs
    # np.hist()
    plt.bar(xx, yy, label=label, **kwargs)


pdir = Path(r"/home/eendebakpt/eendebakpt.github.io/posts")


plt.figure(10)
plt.clf()


name = "floats"
# name='dicts'
# name='ranges'
# name=f'tuples[5]'
# name='pymethodobjects'
plot_barhist(
    data[name],
    label=f"{name} freelist occupation",
)
plt.xlabel("Freelist occupation")
plt.ylabel("Frequency (log scale)")
plt.yscale("log")
plt.legend()
plt.title(f"Allocations for freelist {name}")

# plt.savefig(pdir / 'images' / f'freelist_allocations_{name}.png')

# %% All figures
for name in names:
    print(name)
    plt.figure(10)
    plt.clf()

    plot_barhist(
        data[name],
        label=f"{name} freelist occupation",
    )
    plt.xlabel("Freelist occupation")
    plt.ylabel("Frequency (log scale)")
    plt.yscale("log")
    plt.legend()
    plt.title(f"Allocations for freelist {name}")
    plt.draw()
    plt.savefig(pdir / "images" / f"freelist_allocations_{name}.png")

# %%

plt.figure(10)
plt.clf()


for ii in range(1, 21):
    name = f"tuples[{ii}]"
    xx = list(data[name].keys())
    yy = list(data[name].values())
    plt.plot(
        xx,
        yy,
        ".-",
        label=f"{name} freelist occupation",
    )
plt.xlabel("Freelist size")
plt.ylabel("Frequency (log scale)")
plt.yscale("log")
plt.legend()
plt.title("Allocations for tuples[...]")


# %% Create markdown
s = "# Freelist allocation statistics\n"
s += "\nStatistics obtained from the pyperformance benchmark using branch [small_list_freelist_statistics](https://github.com/eendebakpt/cpython/tree/small_list_freelist_statistics).\n"

for name, d in data.items():
    if 0:
        s += f"\n**Freelist allocations for {name}**\n\n"
        s += to_markdown(d, "Freelist size")
        s += "\n\n"
    else:
        # s+= f'\n**Freelist allocations for {name}**\n\n'
        path = "images/" + f"freelist_allocations_{name}.png"
        s += f"![image info]({path})"
        s += "\n\n"

# print(s)

from pathlib import Path

pdir = Path(r"/home/eendebakpt/eendebakpt.github.io/posts")
with open(pdir / "freelist_stats.md", "w") as fid:
    fid.write(s)

# %% List allocations

plt.rcParams["axes.labelsize"] = 14
plt.rcParams["font.size"] = 12

plt.figure(10)
plt.clf()


plot_hist(normal_allocation_hist, label="Normal allocation")
plot_hist(freelist_allocation_hist, label="Normal freelist allocation")
plot_hist(small_allocation_hist, markersize=12, label="Small freelist allocation")

plt.xlabel("List size")
plt.ylabel("Frequency (log scale)")
plt.yscale("log")
plt.legend()
plt.title("Allocations")

plt.figure(11)
plt.clf()

plot_hist(dealloc_hist, label="Normal deallocation")
plot_hist(small_dealloc_hist, label="Deallocation to small")
plt.yscale("log")
plt.xlabel("List size")
plt.ylabel("Frequency (log scale)")
plt.legend()
plt.title("Deallocations")

plt.figure(12)
plt.clf()

plot_hist(resizes_hist, label="List resize")
plt.yscale("log")
plt.xlabel("List size")
plt.ylabel("Frequency (log scale)")
plt.yscale("log")
plt.legend()
plt.title("Resize list to ")

# plt.ylim([0, max(yy)])

# alloc_results = get_subset(results, "Alloc", minimal_count=1_000_000)

print(f"small list: alloc/dealloc {sum(small_allocation_hist.values())}/{sum(small_dealloc_hist.values())} ")

# %%
STOP

# %%
s = ""
s += to_markdown(strip_lead(alloc_results), "Allocations")
s += to_markdown(strip_lead(freelist_results), "Freelist allocations")
print(s)
# %%


tap = results["Object allocations"]

alloc_size_results = get_subset(results, "Object allocations of size")
# rprint(alloc_size_results)

object_results = get_subset(results, "Object", minimal_count=0, not_contains="of size")

freelist_results = get_subset(results, "Freelist")
alloc_results = get_subset(results, "Alloc", minimal_count=tap // 5000)

bfm = get_subset(results, "PyCMethod_New", minimal_count=1200)

if __name__ == "__main__":
    pass
    # rprint(alloc_results)
    # rprint(freelist_results)

ta = sum(alloc_results.values())
print(f"total allocations tracked: {ta // 1e6}M/{tap // 1e6}M")

# %%

if __name__ == "__main__":
    pass
#    rprint(bfm)


def strip_lead_ml(d):
    def strip(k):
        z = k.split(" ml ")[1]
        z = z.split(" self ")
        q = z[1]
        return q + "." + z[0]

    return {strip(k): v for k, v in d.items()}


if 0:
    bfm = get_subset(results, "PyCMethod_New", minimal_count=500_000)
    s = to_markdown(strip_lead_ml(bfm), "Allocations via PyCMethod_New")
    print(s)
# %%
# rprint({k: v for k, v in alloc_results.items() if "tuple" in k})

# %%
w = get_subset(results, "tuple")
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
