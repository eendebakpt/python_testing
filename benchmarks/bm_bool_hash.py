# Quick benchmark for bool hash

from itertools import repeat

import pyperf


def bench_bool_hash_true(loops):
    loop_iterator = repeat(None, loops)

    t0 = pyperf.perf_counter()
    for ii in loop_iterator:
        hash(True)
    return pyperf.perf_counter() - t0


def bench_bool_hash_False(loops):
    loop_iterator = repeat(None, loops)

    t0 = pyperf.perf_counter()
    for ii in loop_iterator:
        hash(False)
    return pyperf.perf_counter() - t0


def bench_hash_tuple(loops):
    loop_iterator = repeat(None, loops)

    t0 = pyperf.perf_counter()
    x = (True, False, True, False, True, False, True, False)
    for ii in loop_iterator:
        hash(x)
    return pyperf.perf_counter() - t0


def bench_bool_in_set(loops):
    loop_iterator = repeat(None, loops)

    t0 = pyperf.perf_counter()
    d = {"a": None, "b": None, "c": None, True: None}
    for ii in loop_iterator:
        True in d
    return pyperf.perf_counter() - t0


# %timeit bench_list(1000)

if __name__ == "__main__":
    runner = pyperf.Runner()
    runner.bench_time_func("bench_bool_hash_true", bench_bool_hash_true)
    runner.bench_time_func("bench_bool_hash_False", bench_bool_hash_False)
    runner.bench_time_func("bench_hash_tuple", bench_hash_tuple)
    runner.bench_time_func("bench_bool_in_set", bench_bool_in_set)
