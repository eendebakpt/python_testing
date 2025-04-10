# Quick benchmark for itertools.cycle
import gc
import time
from itertools import cycle

t0 = time.perf_counter()
for ii in range(100):
    c = cycle((1, 2, 3, 4))

    for _ in range(200):
        next(c)

gc.collect()  # make sure that in both the normal and free-threading build we clean up the constructed objects
dt = time.perf_counter() - t0
print(dt)
