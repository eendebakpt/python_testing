"""
Created on Thu Mar 27 22:15:15 2025

@author: eendebakpt
"""

# python -c "import ptetools.tools; x = '   '; ptetools.tools.interleaved_benchmark(x.lstrip, x.strip)"

import timeit

x = " hello_world"

left, right, both = 0.0, 0.0, 0.0
number = 10_000
for ii in range(1000):
    # interleaved measurement to counter any thermal throttling
    left += timeit.timeit(x.lstrip, number=number)
    both += timeit.timeit(x.strip, number=number)
    right += timeit.timeit(x.rstrip, number=number)

print(f"{left=:.3f}, {right=:.3f}, {both=:.3f}")
