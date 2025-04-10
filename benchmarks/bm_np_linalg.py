# /// script
# requires-python = ">=3.10"
# dependencies = ['numpy', 'pyperf']
# ///

import pyperf

setup = """
import numpy as np
x22 = np.arange(4.).reshape( (2,2) ) + np.eye(2)
x33 = np.arange(9.).reshape( (3,3) ) + np.eye(3)
"""

runner = pyperf.Runner()
runner.timeit(name="np.linalg.det(x22)", stmt="np.linalg.det(x22)", setup=setup)
runner.timeit(name="np.linalg.det(x33)", stmt="np.linalg.det(x33)", setup=setup)
runner.timeit(name="np.linalg.inv(x22)", stmt="np.linalg.inv(x22)", setup=setup)
runner.timeit(name="np.linalg.inv(x33)", stmt="np.linalg.inv(x33)", setup=setup)
runner.timeit(name="np.linalg.eig(x22)", stmt="np.linalg.eig(x22)", setup=setup)
