# Quick benchmark for cpython long objects

import pyperf


def bench_list(loops):
    range_it = iter(range(loops))

    lst = list(range(5))
    t0 = pyperf.perf_counter()
    for ii in range_it:
        x = 0
        for ii in lst:
            x += ii
    return pyperf.perf_counter() - t0


def bench_tuple(loops):
    range_it = iter(range(loops))

    tpl = tuple(range(5))
    t0 = pyperf.perf_counter()
    for ii in range_it:
        x = 0
        for ii in tpl:
            x += ii
    return pyperf.perf_counter() - t0


def bench_range(loops):
    range_it = iter(range(loops))

    r = range(5)
    t0 = pyperf.perf_counter()
    for ii in range_it:
        x = 0
        for ii in r:
            x += ii
    return pyperf.perf_counter() - t0


# %timeit bench_list(1000)

if __name__ == "__main__":
    runner = pyperf.Runner()
    runner.bench_time_func("bench_list", bench_list)
    runner.bench_time_func("bench_tuple", bench_tuple)
    runner.bench_time_func("bench_range", bench_range)
