# /// script
# requires-python = ">=3.9"
# dependencies = [ "numpy==2.2"]
# [tool.uv]
# exclude-newer = "2026-01-01T00:00:00Z"
# ///

import numpy as np
from numpy.polynomial import Polynomial
import timeit
sz=32
d = np.random.random((4, sz, sz))
P = Polynomial([1., 2., 3., 4.])
old_p = np.poly1d([4., 3., 2., 1.])

number=400
dt1=0
dt2=0
for ii in range(38):
    dt1 += timeit.timeit("P(d)", globals={"P":P, "d": d}, number=number)
   # print(f'dt1 {dt*1e3:.1f} [ms]')
    dt2 +=timeit.timeit("old_p(d)", globals={"old_p":old_p, "d": d}, number=number)
print(f'dt1 {dt1*1e3:.1f} [ms]')
print(f'dt2 {dt2*1e3:.1f} [ms]')

