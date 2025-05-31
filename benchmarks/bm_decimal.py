import math
import platform
import sys
import time

if platform.python_implementation() == "CPython":
    import _decimal as C
else:  # PyPy
    import __decimal as C


def pi(D):
    lasts, t, s, n, na, d, da = D(0), D(3), D(3), D(1), D(0), D(0), D(24)
    while s != lasts:
        lasts = s
        n, na = n + na, na + 8
        d, da = d + da, da + 32
        t = (t * n) / d
        s += t
    return s


def pi(D):
    x = D(math.pi)
    for ii in range(100):
        x = x + 1


niter = 40_000


prec = 9
start = time.time()
C.getcontext().prec = prec
for i in range(niter):
    x = pi(C.Decimal)
print("niter %s, prec %d, result: %s" % (niter, prec, str(x)))
print("%s: time: %fs\n" % (sys.version, time.time() - start))
