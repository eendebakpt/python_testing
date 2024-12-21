import pyperf
import numpy as np
#print(np, np.__version__)

#np._set_promotion_state('weak')

setup="""
import numpy as np
#np._set_promotion_state('weak')

x = np.array([1., 2., 3.])
x1000= np.arange(1000,)
i = 3
f = 1.1

s = np.float64(20.)
phi = np.pi/4
"""

runner = pyperf.Runner()
runner.timeit(name="f + x", stmt="_ = f + x",setup=setup)
runner.timeit(name="np.sin(f)", stmt="_ = np.sin(f)",setup=setup)
runner.timeit(name="np.sin(s)", stmt="_ = np.sin(s)",setup=setup)
runner.timeit(name="i + x", stmt="_ = f + x",setup=setup)
runner.timeit(name="np.sin(2*np.pi*x + phi)", stmt="_ = np.sin(2*np.pi*x + phi)",setup=setup)
runner.timeit(name="s + x", stmt="_ = s + x",setup=setup)
runner.timeit(name="x + x", stmt="_ = s + x",setup=setup)
runner.timeit(name="x**2", stmt="_ = x**2",setup=setup)
runner.timeit(name="f + x1000", stmt="_ = f + x1000",setup=setup)