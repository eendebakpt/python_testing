# This script runs a set of small benchmarks to help identify scaling
# bottlenecks in the free-threaded interpreter. The benchmarks consist
# of patterns that ought to scale well, but haven't in the past. This is
# typically due to reference count contention or lock contention.
#
# This is not intended to be a general multithreading benchmark suite, nor
# are the benchmarks intended to be representative of real-world workloads.
#
# On Linux, to avoid confounding hardware effects, the script attempts to:
# * Use a single CPU socket (to avoid NUMA effects)
# * Use distinct physical cores (to avoid hyperthreading/SMT effects)
# * Use "performance" cores (Intel, ARM) on CPUs that have performance and
#   efficiency cores
#
# It also helps to disable dynamic frequency scaling (i.e., "Turbo Boost")
#
# Intel:
# > echo "1" | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
#
# AMD:
# > echo "0" | sudo tee /sys/devices/system/cpu/cpufreq/boost
#

import copy
import os
import queue
import sys
import threading
import time

# The iterations in individual benchmarks are scaled by this factor.
WORK_SCALE = 100

ALL_BENCHMARKS = {}

threads = []
in_queues = []
out_queues = []


def register_benchmark(func):
    ALL_BENCHMARKS[func.__name__] = func
    return func


small_tuple = (1, 2, 3, 4)
small_list = [1, 2, 3, 4]
small_set = {1, 2, 3, 4}
small_frozenset = frozenset({1, 2, 3, 4})


@register_benchmark
def tuple_contains():
    z = 0
    for i in range(500 * WORK_SCALE):
        z in small_tuple


@register_benchmark
def list_contains():
    z = 0
    for i in range(500 * WORK_SCALE):
        z in small_list


@register_benchmark
def frozenset_contains():
    z = 1
    for i in range(500 * WORK_SCALE):
        z in small_frozenset


@register_benchmark
def frozenset_contains_dunder():
    z = 1
    w = small_frozenset.__contains__
    for i in range(500 * WORK_SCALE):
        w(z)


@register_benchmark
def set_contains():
    z = 1
    w = small_set.__contains__
    for i in range(500 * WORK_SCALE):
        w(z)


@register_benchmark
def set_contains_alt():
    z = 0
    for i in range(500 * WORK_SCALE):
        z in tuple(small_set)


@register_benchmark
def shallow_copy():
    x = [1, 2, 3]
    shallow_copy = copy.copy
    for i in range(400 * WORK_SCALE):
        shallow_copy(x)


@register_benchmark
def deepcopy():
    x = {"list": [1, 2], "tuple": (1, None)}
    deepcopy = copy.deepcopy
    for i in range(80 * WORK_SCALE):
        deepcopy(x)


module = sys.modules[__name__]

thread_local = threading.local()


def bench_one_thread(func):
    t0 = time.perf_counter_ns()
    func()
    t1 = time.perf_counter_ns()
    return t1 - t0


def bench_parallel(func):
    t0 = time.perf_counter_ns()
    for inq in in_queues:
        inq.put(func)
    for outq in out_queues:
        outq.get()
    t1 = time.perf_counter_ns()
    return t1 - t0


def benchmark(func):
    delta_one_thread = bench_one_thread(func)
    delta_many_threads = bench_parallel(func)

    speedup = delta_one_thread * len(threads) / delta_many_threads
    if speedup >= 1:
        factor = speedup
        direction = "faster"
    else:
        factor = 1 / speedup
        direction = "slower"

    use_color = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    color = reset_color = ""
    if use_color:
        if speedup <= 1.1:
            color = "\x1b[31m"  # red
        elif speedup < len(threads) / 2:
            color = "\x1b[33m"  # yellow
        reset_color = "\x1b[0m"

    print(f"{color}{func.__name__:<25} {round(factor, 1):>4}x {direction}{reset_color}")


def determine_num_threads_and_affinity():
    if sys.platform != "linux":
        return [None] * os.cpu_count()

    # Try to use `lscpu -p` on Linux
    import subprocess

    try:
        output = subprocess.check_output(["lscpu", "-p=cpu,node,core,MAXMHZ"], text=True, env={"LC_NUMERIC": "C"})
    except (FileNotFoundError, subprocess.CalledProcessError):
        return [None] * os.cpu_count()

    table = []
    for line in output.splitlines():
        if line.startswith("#"):
            continue
        cpu, node, core, maxhz = line.split(",")
        if maxhz == "":
            maxhz = "0"
        table.append((int(cpu), int(node), int(core), float(maxhz)))

    cpus = []
    cores = set()
    max_mhz_all = max(row[3] for row in table)
    for cpu, node, core, maxmhz in table:
        # Choose only CPUs on the same node, unique cores, and try to avoid
        # "efficiency" cores.
        if node == 0 and core not in cores and maxmhz == max_mhz_all:
            cpus.append(cpu)
            cores.add(core)
    return cpus


def thread_run(cpu, in_queue, out_queue):
    if cpu is not None and hasattr(os, "sched_setaffinity"):
        # Set the affinity for the current thread
        os.sched_setaffinity(0, (cpu,))

    while True:
        func = in_queue.get()
        if func is None:
            break
        func()
        out_queue.put(None)


def initialize_threads(opts):
    if opts.threads == -1:
        cpus = determine_num_threads_and_affinity()
    else:
        cpus = [None] * opts.threads  # don't set affinity

    print(f"Running benchmarks with {len(cpus)} threads")
    for cpu in cpus:
        inq = queue.Queue()
        outq = queue.Queue()
        in_queues.append(inq)
        out_queues.append(outq)
        t = threading.Thread(target=thread_run, args=(cpu, inq, outq), daemon=True)
        threads.append(t)
        t.start()


def main(opts):
    global WORK_SCALE
    if not hasattr(sys, "_is_gil_enabled") or sys._is_gil_enabled():
        sys.stderr.write("expected to be run with the  GIL disabled\n")

    benchmark_names = opts.benchmarks
    if benchmark_names:
        for name in benchmark_names:
            if name not in ALL_BENCHMARKS:
                sys.stderr.write(f"Unknown benchmark: {name}\n")
                sys.exit(1)
    else:
        benchmark_names = ALL_BENCHMARKS.keys()

    WORK_SCALE = opts.scale

    if not opts.baseline_only:
        initialize_threads(opts)

    do_bench = not opts.baseline_only and not opts.parallel_only
    for name in benchmark_names:
        func = ALL_BENCHMARKS[name]
        if do_bench:
            benchmark(func)
            continue

        if opts.parallel_only:
            delta_ns = bench_parallel(func)
        else:
            delta_ns = bench_one_thread(func)

        time_ms = delta_ns / 1_000_000
        print(f"{func.__name__:<18} {time_ms:.1f} ms")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--threads", type=int, default=-1, help="number of threads to use")
    parser.add_argument("--scale", type=int, default=100, help="work scale factor for the benchmark (default=100)")
    parser.add_argument(
        "--baseline-only", default=False, action="store_true", help="only run the baseline benchmarks (single thread)"
    )
    parser.add_argument(
        "--parallel-only", default=False, action="store_true", help="only run the parallel benchmark (many threads)"
    )
    parser.add_argument("benchmarks", nargs="*", help="benchmarks to run")
    options = parser.parse_args()
    main(options)
