# /// script
# requires-python = ">=3.10"
# dependencies = ['numpy', 'pyperf']
# ///

import pyperf

setup = """
import numpy as np
from numpy import result_type

x = np.array([1., 2.])
y = np.array([1., 2., 4.])

x32 = np.array([1., 2.], dtype=np.float32)

d = x.dtype
d32 = x32.dtype
"""

runner = pyperf.Runner()
runner.timeit(name="result_type(x)", stmt="result_type(x)", setup=setup)
runner.timeit(name="result_type(x, x)", stmt="result_type(x, x)", setup=setup)
runner.timeit(name="result_type(x, y)", stmt="result_type(x, y)", setup=setup)
runner.timeit(name="result_type(x, x32)", stmt="result_type(x, x32)", setup=setup)
runner.timeit(name="result_type(x, x32, x)", stmt="result_type(x, x32, x)", setup=setup)
runner.timeit(name="result_type(d, d)", stmt="result_type(d, d)", setup=setup)
runner.timeit(name="result_type(d, d32)", stmt="result_type(d, d32)", setup=setup)
runner.timeit(name="result_type(d, d32, x)", stmt="result_type(d, d32, x)", setup=setup)
