# /// script
# requires-python = "==3.13"
# dependencies = ["matplotlib", "numpy"]
# [tool.uv]
# exclude-newer = "2026-01-01T00:00:00Z"
# ///

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib._histograms_impl import _hist_bin_auto, _hist_bin_fd, _hist_bin_sqrt, _hist_bin_sturges


def uniform_dataset(size):
    return np.random.rand(size)


def poisson_dataset(size):
    return np.random.poisson(size=size)


def normal_dataset(size):
    return np.abs(np.random.normal(size=size))


def abs_normal_dataset(size):
    return np.random.normal(size=size)


def double_gaussian_dataset(size):
    x = np.random.normal(size=size)
    x[: x.size // 2] += 10
    return x


def iqr(x):
    return np.diff(np.percentile(x, [25, 75])).item()


x = normal_dataset(10_000)
print(f"IQR for normal dataset: {iqr(x)}/{np.ptp(x)}")


def zero_iqr_dataset(size):
    x = np.random.rand(size)
    x[: (4 * size) // 5] = 0.5
    return x


def small_iqr_dataset(size):
    x = np.random.rand(size)
    x[0] = 0
    x[1] = 1
    x[: size // 3] = 0.5
    x[size // 3 : (2 * size) // 3] = 0.5 + 1e-3
    return x


def _hist_bin_pr(x, range):
    # properties: continuous, behavoir unchanges on several distributions, no out-of-memory
    fd_bw = _hist_bin_fd(x, range)
    sturges_bw = _hist_bin_sturges(x, range)
    sqrt_bw = _hist_bin_sqrt(x, range)
    fd_bw_corrected = max(fd_bw, sqrt_bw / 2)
    return min(fd_bw_corrected, sturges_bw)


iterations = 400
nn = [2, 6, 10, 20, 40, 60, 100, 200, 500, 1_000, 10_000, 100_000, 1_000_000]  # , 10_000_000]
nn = [10, 20, 40, 60, 100, 200, 500, 1_000, 10_000]
sturges = []
sqrt = []
fd_normal = []
fd_uniform = []
fd_iqr = []
current_main = []
pr = []

methods = {
    "Sturges": _hist_bin_sturges,
    "FD": _hist_bin_fd,
    "Sqrt": _hist_bin_sqrt,
    "Main": _hist_bin_auto,
    "PR": _hist_bin_pr,
}
datasets = {
    "Uniform": uniform_dataset,
    "Poisson": poisson_dataset,
    "Normal": normal_dataset,
    "Double Gaussian": double_gaussian_dataset,
    "Zero IQR": zero_iqr_dataset,
    "Small IQR": small_iqr_dataset,
}

range_arg = None  # unused
results = defaultdict(dict)

for dataset, dataset_method in datasets.items():
    print(f"Generating data for {dataset}")
    for method_name in methods:
        results[dataset][method_name] = np.zeros(len(nn))
    for idx, n in enumerate(nn):
        print(n)
        for it in range(iterations):
            x = dataset_method(n)
            for method_name, method in methods.items():
                bw = method(x, range_arg)
                if bw == 0:
                    bins = 1
                else:
                    bins = np.ptp(x) / bw
                results[dataset][method_name][idx] += bins
    for method_name in methods:
        results[dataset][method_name] /= iterations


# %%
markers = defaultdict(lambda: ".", {"Main": "d", "PR": "."})
linewidth = defaultdict(lambda: 1, {"Main": 2, "PR": 2})
markersize = defaultdict(lambda: 8, {"Main": 8})
for idx, dataset in enumerate(datasets):
    print(f"Plotting data for {dataset}")

    r = results[dataset]
    plt.figure(100 + idx)
    plt.clf()
    for name, counts in r.items():
        marker = markers[name]
        plt.plot(
            nn, counts, "-", marker=marker, markersize=markersize[name], linewidth=linewidth[name], label=f"{name}"
        )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of data points")
    plt.ylabel("Number of bins")
    plt.legend()
    plt.title(f"Dataset {dataset}")

# %%
