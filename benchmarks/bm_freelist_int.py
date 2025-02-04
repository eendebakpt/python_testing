# Quick benchmark for cpython freelists for int

from uuid import uuid8
import pyperf


def collatz(a):
    while a > 1:
        if a % 2 == 0:
            a = a // 2
        else:
            a = 3 * a + 1

def bench_int(loops):
    range_it = range(loops)
    tpl = tuple(range(200, 300))

    t0 = pyperf.perf_counter()
    for ii in range_it:
        for jj in tpl:
            collatz(jj)
    return pyperf.perf_counter() - t0

def gcd(x, y):
    while y != 0:
        (x, y) = (y, x % y)
    return x

def bench_gcd(loops):
    range_it = range(loops)
    tpl = tuple(range(100))

    t0 = pyperf.perf_counter()
    for ii in range_it:
        for a in tpl:
            gcd( (2<<120) + 1231231232131, 2<<30+ a)
    return pyperf.perf_counter() - t0

def bench_uuid8(loops):
    range_it = iter(range(loops))

    t0 = pyperf.perf_counter()
    for ii in range_it:
        _ = uuid8()
    return pyperf.perf_counter() - t0

def bench_id(loops):
    range_it = iter(range(loops))

    t0 = pyperf.perf_counter()
    for ii in range_it:
        for jj in range(1000):
            _ = id(jj)
    return pyperf.perf_counter() - t0


if __name__ == "__main__":
    runner = pyperf.Runner()
    runner.bench_time_func("bench_int", bench_int)
    runner.bench_time_func("bench_gcd", bench_gcd)
    runner.bench_time_func("bench_uuid8", bench_uuid8)
    runner.bench_time_func("bench_id", bench_id)
