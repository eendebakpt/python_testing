# /// script
# requires-python = ">=3.10"
# dependencies = ['numpy', 'pyperf']
# ///

import pyperf

setup = """

z = 2 << 30
two_digit_ints = list(range(z, z + 2000))

def long_hash_small_int():
    for _ in range(4):
        for ii in range(0,250):
            hash(ii)

def long_hash_one_digit():
    z = 1000
    for ii in range(z, z + 1000):
        hash(ii)

def long_hash_two_digit():
    z = 2 << 30
    for ii in range(z, z + 1000):
        hash(ii)

def long_hash_multi_digit():
    z = 1 << 30 << 30 << 30 << 30
    for ii in range(z, z + 1000):
        hash(ii)
"""

runner = pyperf.Runner()
runner.timeit(name="long_hash_small_int", stmt="long_hash_small_int()", setup=setup)
runner.timeit(name="long_hash_one_digit", stmt="long_hash_one_digit()", setup=setup)
runner.timeit(name="long_hash_two_digit", stmt="long_hash_two_digit()", setup=setup)
runner.timeit(name="long_hash_multi_digit", stmt="long_hash_multi_digit()", setup=setup)
runner.timeit(name="set(ints)", stmt="set(two_digit_ints)", setup=setup)
