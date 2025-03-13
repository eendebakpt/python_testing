# Quick benchmark for cpython range objects

import pyperf

import pyperf
runner = pyperf.Runner()

loop = """
def g(n):
    x=0
    for ii in range(n):
        x += 1
"""
        
for s in [1, 10, 100, 400]:
	time = runner.timeit(name=f'range({s})', stmt=f"range({s})")
	time = runner.timeit(name=f'iter(range({s}))', stmt=f"iter(range({s}))")
	time = runner.timeit(name=f'list(range({s}))', stmt=f"list(range({s}))")
	time = runner.timeit(name=f'range(2, {s})', stmt=f"range(2, {s})")
	time = runner.timeit(name=f'iter(range(2, {s}))', stmt=f"iter(range(2, {s}))")

	time = runner.timeit(name=f'for loop length {s}', stmt=f"g({s})", setup=loop)
    


