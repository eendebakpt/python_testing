import pyperf

runner = pyperf.Runner()

setup = """
import numpy as np
x = np.array([1., 2.])
f = np.float64(2.2)
"""


if 1:
    runner.timeit(name="np.cos(x)", stmt="np.cos(x)", setup=setup)
    runner.timeit(name="np.cos(float64)", stmt="np.cos(f)", setup=setup)
    runner.timeit(name="np.add.reduce(x)", stmt="np.add.reduce(x)", setup=setup)
    runner.timeit(name="np.add.accumulate(x)", stmt="np.add.accumulate(x)", setup=setup)
