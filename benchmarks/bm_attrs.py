# /// script
# requires-python = ">=3.10"
# dependencies = [attrs, pyperf]
# ///

import pyperf

setup = """
from attrs import define, asdict

@define
class Simple:
     i : int
     s : str
     l : list

s = Simple(10, 'hi', [3, 1, 4])

"""

runner = pyperf.Runner()
runner.timeit(name="instance creation", stmt="Simple(10, 'hi', [3, 1, 4])", setup=setup)
runner.timeit(name="asdict", stmt="asdict(s)", setup=setup)
