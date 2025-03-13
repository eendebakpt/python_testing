# Quick benchmark for cpython binary op specialization

import pyperf


def bench_rshift_compactint(loops):
    range_it = iter(range(loops))
    t0 = pyperf.perf_counter()
    for ii in range_it:
        for x in range(0, 2**13):
            x = x >> 4
    return pyperf.perf_counter() - t0


def bench_rshift_largeint(loops):
    offset = 2**40
    range_it = iter(range(loops))
    t0 = pyperf.perf_counter()
    for ii in range_it:
        for x in range(offset, offset + 2**13):
            x = x >> 4
    return pyperf.perf_counter() - t0


def bench_lshift_compactint(loops):
    range_it = iter(range(loops))
    t0 = pyperf.perf_counter()
    for ii in range_it:
        for x in range(0, 2**13):
            x = x << 4
    return pyperf.perf_counter() - t0


if __name__ == "__main__":
    runner = pyperf.Runner()
    runner.bench_time_func("bench_lshift_compactint", bench_lshift_compactint)
    runner.bench_time_func("bench_rshift_compactint", bench_rshift_compactint)
    runner.bench_time_func("bench_rshift_largeint", bench_rshift_largeint)
