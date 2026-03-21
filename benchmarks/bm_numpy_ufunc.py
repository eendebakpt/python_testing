import pyperf

runner = pyperf.Runner()

setup = """
import numpy as np
x = np.array([1., 2.])
y = np.array([-1., 12.])
"""

runner.timeit(name="x + x", stmt="x + y", setup=setup)
runner.timeit(name="np.abs(x)", stmt="np.abs(x)", setup=setup)
runner.timeit(name="np.cos(x)", stmt="np.cos(x)", setup=setup)
runner.timeit(name="np.add.reduce(x)", stmt="np.add.reduce(x)", setup=setup)
runner.timeit(name="np.add.accumulate(x)", stmt="np.add.accumulate(x)", setup=setup)
