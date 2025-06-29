import itertools
import statistics
import time


def generator_chain(*iterables):
    # chain('ABC', 'DEF') → A B C D E F
    for iterable in iterables:
        yield from iterable


class python_chain:
    def __init__(self, *iterables):
        self.iterables = iter(iterables)
        self.active = None

    def __next__(self):
        if self.active is None:
            self.active = iter(next(self.iterables))

        while True:
            try:
                return next(self.active)
            except StopIteration:
                self.active = iter(next(self.iterables))

    def __iter__(self):
        return self


data = [(1, 2, 3)] * 2
N = 10_000

times = [[], [], []]
options = [itertools.chain, generator_chain, python_chain]

for _ in range(60):
    for jj, chain_method in enumerate(options):
        t0 = time.perf_counter()
        for _ in range(N):
            s = 0
            for x in chain_method(*data):
                s += x**2
        times[jj].append(time.perf_counter() - t0)


for jj, chain_method in enumerate(options):
    n = len(times[jj])
    w = sum(times[jj])
    m = max(times[jj]) * n

    v = statistics.median(times[jj]) * n

    if jj == 0:
        w0 = w
    ratio = w / w0
    print(f"{chain_method} mean {w:.3f} / max {m:.3f} / median {v:.3f} ratio {ratio:.3f}")

# FT build: python recipe 10% slower
# normal build: python recipe 18% slower
