# /// script
# requires-python = ">=3.10"
# dependencies = ['numpy', 'pyperf']
# ///

import pyperf

setup = """

z = 2 << 30 << 30
two_digit_ints = list(range(z, z + 2000))

def lh():
    z = 2 << 30 << 30
    for ii in range(z, z + 1000):
        hash(ii)

def long_hash_multi_digit():
    z = 1 << 30 << 30 << 30 << 30
    for ii in range(2**61, 2**61 + 1000):
        hash(ii)

def small_longhash():
    for ii in range(0,1000):
        hash(ii)

"""

runner = pyperf.Runner()
runner.timeit(name="long_hash(small int)", stmt="small_longhash()", setup=setup)
runner.timeit(name="long_hash", stmt="lh()", setup=setup)
runner.timeit(name="long_hash_multi_digit", stmt="long_hash_multi_digit()", setup=setup)
runner.timeit(name="set(ints)", stmt="set(two_digit_ints)", setup=setup)
