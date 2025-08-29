# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///

import pyperf

setup = """
from dataclasses import dataclass, asdict, astuple
from pickle import dumps

@dataclass
class Simple:
     i : int
     s : str
     l : list

s = Simple(10, 'hi', [3, 1, 4, 1])

@dataclass(frozen=True, slots=True)
class Frozen:
     i : int
     s : str
     l : list

f = Frozen(10, 'hi', [3, 1, 4, 1])
f.__getstate__()

"""

runner = pyperf.Runner()
runner.timeit(name="instance creation", stmt="Simple(10, 'hi', [3, 1, 4, 1])", setup=setup)
runner.timeit(name="asdict", stmt="asdict(s)", setup=setup)
runner.timeit(name="astuple", stmt="astuple(s)", setup=setup)
runner.timeit(name="f.__getstate__()", stmt="f.__getstate__()", setup=setup)
