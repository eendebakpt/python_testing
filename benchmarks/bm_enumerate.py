# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///

import pyperf

runner = pyperf.Runner()

setup = """
import copy

it_small = tuple(range(10))
it = tuple(range(10_000))



def enumerate_tuple_pass(it):
    for ii, value in enumerate(it):
        pass

def enumerate_tuple(it):
    delta = 0
    for ii, value in enumerate(it):
        delta += ii - value

def enumerate_tuple_hold_tuple(it):
    prev_pair = None
    for pair in enumerate(it):
        prev_pair = pair

        
"""

runner.timeit(name="enumerate_tuple_pass small", stmt="enumerate_tuple_pass(it_small)", setup=setup)
runner.timeit(name="enumerate_tuple_pass", stmt="enumerate_tuple_pass(it)", setup=setup)
runner.timeit(name="enumerate_tuple", stmt="enumerate_tuple(it)", setup=setup)
runner.timeit(name="enumerate_tuple_hold_tuple", stmt="enumerate_tuple_hold_tuple(it)", setup=setup)
