# Quick benchmark for cpython long objects

import pyperf


def collatz(a):
    while a > 1:
        if a % 2 == 0:
            a = a // 2
        else:
            a = 3 * a + 1


def bench_collatz(loops):
    range_it = range(loops)
    t0 = pyperf.perf_counter()
    for ii in range_it:
        collatz(ii)
    return pyperf.perf_counter() - t0


def bench_long(loops):
    range_it = range(loops)
    t0 = pyperf.perf_counter()
    x = 10
    for ii in range_it:
        x = x * x
        y = x // 2
        x = y + ii + x
        if x > 10**10:
            x = x % 1000
    return pyperf.perf_counter() - t0


def bench_alloc(loops):
    range_it = range(loops)
    t0 = pyperf.perf_counter()
    for ii in range_it:
        for kk in range(20_000):
            del kk
    return pyperf.perf_counter() - t0


def bench_lshift(loops):
    range_it = range(loops)
    t0 = pyperf.perf_counter()
    for ii in range_it:
        _ = ii << 2
    return pyperf.perf_counter() - t0


# %timeit bench_long(1000)

if __name__ == "__main__":
    runner = pyperf.Runner()
    runner.bench_time_func("bench_collatz", bench_collatz)
    runner.bench_time_func("bench_long", bench_long)
    runner.bench_time_func("bench_alloc", bench_alloc)
    runner.bench_time_func("bench_lshift", bench_lshift)
