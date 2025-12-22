# Quick benchmark for cpython binary op specialization

import pyperf

setup = """

def swap_bits(x: int) -> int:
    # Swaps the even and odd bits of a given number.
    # Get all even bits of x
    even_bits = x & 0xAAAAAAAA
    # Get all odd bits of x
    odd_bits = x & 0x55555555
    # Right shift even bits
    even_bits >>= 1
    # Left shift odd bits
    odd_bits <<= 1
    # Combine even and odd bits
    return (even_bits | odd_bits)

def nth_magic_number(n: int) -> int:
    # Finds the nth magic number. 
    # A magic number is defined as a number which can be expressed as a power of 5 or sum of unique powers of 5.
    pow_of_5 = 1
    magic_number = 0

    # Go through every bit of n
    while n:
        pow_of_5 *= 5

         # If last bit of n is set
        if n & 1:
            magic_number += pow_of_5

        # proceed to next bit
        n >>= 1
    return magic_number

def bit_ops(n):
    start = 2**47
    prev = 0
    value = 0
    for ii in range(start, start + n):
        value = (value) | ii & prev
        prev = ii
"""

runner = pyperf.Runner()
runner.timeit(name="bit ops", stmt="bit_ops(10_000)", setup=setup)
runner.timeit(name="swap_bits", stmt="[swap_bits(ii) for ii in range(2**34, 2**34+int(1e5))]", setup=setup)
runner.timeit(
    name="nth_magic_number", stmt="[nth_magic_number(ii) for ii in range(2**34, 2**34+int(1e3))]", setup=setup
)
