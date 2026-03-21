import time

import numpy as np

start = np.float64(5.0)
newdtype = np.float64(15.0).dtype
newdtype = np.float64

t0 = time.perf_counter()
nn = 1_000_000
for ii in range(nn):
    start.astype(newdtype)
dt = time.perf_counter() - t0

print(f"dt {1e9 * dt / nn:.3f} [ns/iter]")
