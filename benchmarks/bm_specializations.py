# Quick benchmark for cpython binary op specialization

import pyperf


def bench_x_add_int(loops):
    range_it = iter(range(loops))
    t0 = pyperf.perf_counter()
    x = 1.1
    for ii in range_it:
        _ = x + 4
    return pyperf.perf_counter() - t0


def bench_x_pow_int(loops):
    range_it = iter(range(loops))
    t0 = pyperf.perf_counter()
    x = 1.1
    for ii in range_it:
        _ = x**2
    return pyperf.perf_counter() - t0


def bench_list_list_add(loops):
    range_it = iter(range(loops))
    t0 = pyperf.perf_counter()
    x = [1, 2, 3]
    y = [None, "so", "far"]
    for ii in range_it:
        _ = x + y
    return pyperf.perf_counter() - t0


def bench_tuple_tuple_add(loops):
    range_it = iter(range(loops))
    t0 = pyperf.perf_counter()
    x = (1, 2, 3)
    y = (None, "so", "far")
    for ii in range_it:
        _ = x + y
    return pyperf.perf_counter() - t0


def bench_long_True_add(loops):
    range_it = iter(range(loops))
    t0 = pyperf.perf_counter()
    x = 123
    y = True
    for ii in range_it:
        _ = x + y
    return pyperf.perf_counter() - t0


def bench_long_False_add(loops):
    range_it = iter(range(loops))
    t0 = pyperf.perf_counter()
    x = 123
    y = False
    for ii in range_it:
        _ = x + y
    return pyperf.perf_counter() - t0


if __name__ == "__main__":
    runner = pyperf.Runner()
    # runner.bench_time_func("bench_x_add_int", bench_x_add_int)
    # runner.bench_time_func("bench_x_pow_int", bench_x_pow_int)
    # runner.bench_time_func("bench_list_list_add", bench_list_list_add)
    # runner.bench_time_func("bench_tuple_tuple_add", bench_tuple_tuple_add)
    runner.bench_time_func("bench_long_True_add", bench_tuple_tuple_add)
    runner.bench_time_func("bench_long_False_add", bench_tuple_tuple_add)
