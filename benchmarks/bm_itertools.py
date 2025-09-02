# Quick benchmark for cpython long objects

import pyperf
from itertools import cycle, product


def bench_cycle(loops):
    range_it = range(loops)
    t0 = pyperf.perf_counter()

    c = cycle((1, 2, 3, 4))
    for ii in range_it:
        value = next(c)
    return pyperf.perf_counter() - t0


def bench_product(loops):
    range_it = range(loops)
    t0 = pyperf.perf_counter()

    for ii in range_it:
        it = product((1, 2, 3), (4, 5, 6), (7, 8, 9))
        for p in it:
            sum(p)  # minimal amount of work
    return pyperf.perf_counter() - t0


if __name__ == "__main__":
    runner = pyperf.Runner()
    runner.bench_time_func("bench_product", bench_product)
    runner.bench_time_func("bench_cycle", bench_cycle)
