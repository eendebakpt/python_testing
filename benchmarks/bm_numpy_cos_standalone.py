import time
import sys
import statistics

# about 100 ms for python start and numpy import
import numpy as np

x = np.array([1.0, 2.0])
# y = np.array([0., 0.])

x = 10.1
# x = np.float64(10.1)
outer = 100
inner = 3
iterations = 200_000


def bench(iterations, x):
    for _ in range(iterations):
        # np.sign(x)
        np.abs(x)
        # x + x
        # np.conjugate(x)
        # np.sin(x, dtype=x.dtype)
        # np.add.reduce(x, axis=0)


times = []
for _ in range(outer):
    tt = []
    for _ in range(inner):
        t0 = time.perf_counter()
        bench(iterations, x)
        dt = time.perf_counter() - t0
        tt.append(dt)
    t00 = 1e9 * sum(tt) / (inner * iterations)
    times.append(t00)
    print(f"{t00:.2f} [ns] / operation")
q10 = statistics.quantiles(times, n=10)[0]
print(f"q10 {q10:.2f} [ns] / operation")
print(sys.executable)
print(np)

import numpy as np
import numpy._core._multiarray_umath as m
# np.show_config()
# Extensions built with limited API have filenames like:
#   module.abi3.so (Linux) or module.pyd (Windows)
# print(m.__file__)
