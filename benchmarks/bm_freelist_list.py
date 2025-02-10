# Quick benchmark for cpython freelists for small lists

import pyperf

def bench_list_from_tuple(loops):
    range_it = range(loops)
   
    x = (1, 2)
    t0 = pyperf.perf_counter()
    for ii in range_it:
        _ = list(x)
    return pyperf.perf_counter() - t0

def bench_list_create(loops):
    range_it = range(loops)
   
    x = [1, 2]
    t0 = pyperf.perf_counter()
    for ii in range_it:
        x = [1, ii]
    return pyperf.perf_counter() - t0

def bench_list_create_const(loops):
    range_it = range(loops)
   
    x = [1, 2]
    t0 = pyperf.perf_counter()
    for ii in range_it:
        x = [1, 3]
    return pyperf.perf_counter() - t0

def bench_list_copy(loops):
    range_it = range(loops)
   
    x = [1, 2]
    t0 = pyperf.perf_counter()
    for ii in range_it:
        _ = x.copy()
    return pyperf.perf_counter() - t0

def bench_list_repeat(loops):
    range_it = range(loops)
   
    x = [1, 2]
    t0 = pyperf.perf_counter()
    for ii in range_it:
        _ = x * 2
    return pyperf.perf_counter() - t0


if __name__ == "__main__":
    runner = pyperf.Runner()
    runner.bench_time_func("bench_list_create", bench_list_create)
    runner.bench_time_func("bench_list_create_const", bench_list_create_const)
    runner.bench_time_func("bench_list_from_tuple", bench_list_from_tuple)
    runner.bench_time_func("bench_list_copy", bench_list_copy)
    runner.bench_time_func("bench_list_repeat", bench_list_repeat)