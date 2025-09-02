import timeit
import numpy as np
import time
from itertools import repeat

x = np.array([1.0, 2.0])

print(np, np.__version__)

print(np.cos(x))

n = 10_000_000
t0 = time.perf_counter()
for _ in repeat(None, n):
    np.cos(x)
dt = time.perf_counter() - t0
#dt = timeit.timeit("np.cos(x)", number=n, globals=globals())
print(f"dt: {1e9*dt/n:.1f} [ns/it] (total {dt:.1f} [s])")
