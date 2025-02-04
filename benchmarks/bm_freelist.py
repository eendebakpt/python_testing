# Quick benchmark for cpython freelists

import pyperf


def bench_list(loops):
    range_it = iter(range(loops))

    t0 = pyperf.perf_counter()
    for ii in range_it:
        _ = [ii]
        _ = [ii, ii + 1]
        _ = [ii, ii + 1, ii]
    return pyperf.perf_counter() - t0



def bench_float(loops):
    range_it = iter(range(loops))
    tpl = tuple(range(500))

    t0 = pyperf.perf_counter()
    for ii in range_it:
        x = 0
        for ii in tpl:
            x += float(ii + 1) ** 2 - float(ii + 1) ** 2
    return pyperf.perf_counter() - t0


def bench_builtin_or_method(loops):
    range_it = iter(range(loops))
    tpl = tuple(range(50))

    lst = []
    it = iter(set([2, 3, 4]))
    t0 = pyperf.perf_counter()
    for ii in range_it:
        for ii in tpl:
            lst.append
            it.__length_hint__
    return pyperf.perf_counter() - t0


class A:
    def __init__(self, value):
        self.value = value

    def x(self):
        return self.value

    @property
    def v(self):
        return self.value


def bench_property(loops):
    range_it = iter(range(loops))
    tpl = tuple(range(50))

    t0 = pyperf.perf_counter()
    for ii in range_it:
        a = A(ii)
        for ii in tpl:
            _ = a.v
    return pyperf.perf_counter() - t0


def bench_class_method(loops):
    range_it = iter(range(loops))
    tpl = tuple(range(50))

    t0 = pyperf.perf_counter()
    for ii in range_it:
        a = A(ii)
        for ii in tpl:
            _ = a.x()
    return pyperf.perf_counter() - t0


def bench_class_method_create(loops):
    range_it = iter(range(loops))
    tpl = tuple(range(50))

    t0 = pyperf.perf_counter()
    for ii in range_it:
        a = A(ii)
        for ii in tpl:
            _ = a.x
    return pyperf.perf_counter() - t0


def bench_list_iter(loops):
    range_it = iter(range(loops))

    lst = list(range(5))
    t0 = pyperf.perf_counter()
    for ii in range_it:
        x = 0
        for ii in lst:
            x += ii
    return pyperf.perf_counter() - t0


def bench_tuple_iter(loops):
    range_it = iter(range(loops))

    tpl = tuple(range(5))
    t0 = pyperf.perf_counter()
    for ii in range_it:
        x = 0
        for ii in tpl:
            x += ii
    return pyperf.perf_counter() - t0


def bench_range_iter(loops):
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
    runner.bench_time_func("bench_float", bench_float)
    runner.bench_time_func("bench_builtin_or_method", bench_builtin_or_method)
    runner.bench_time_func("bench_list_iter", bench_list_iter)
    runner.bench_time_func("bench_tuple_iter", bench_tuple_iter)
    runner.bench_time_func("bench_range_iter", bench_range_iter)
    runner.bench_time_func("bench_property", bench_property)
    runner.bench_time_func("bench_class_method", bench_class_method)
    runner.bench_time_func("bench_class_method_create", bench_class_method_create)
