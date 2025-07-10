# /// script
# requires-python = ">=3.10"
# dependencies = ['numpy', 'pyperf']
# ///

import pyperf

setup = """

def lh():
    for ii in range(2**34, 2**34 + 1000):
        hash(ii)

def small_longhash():
    for ii in range(0,1000):
        hash(ii)

"""

runner = pyperf.Runner()
runner.timeit(name="long_hash(small int)", stmt="small_longhash()", setup=setup)
runner.timeit(name="long_hash", stmt="lh()", setup=setup)
