# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///

import pyperf

runner = pyperf.Runner()

setup = """
import copy

a={'list': [1,2,3,43], 't': (1,2,3), 'str': 'hello', 'subdict': {'a': True}}

from dataclasses import dataclass

lst = [1, 's']
tpl  =('a', 'b', 3)

i = 123123123
sl = slice(1,2,3)

@dataclass
class A:
    a : int
    
dc = A(123)
list_dc = [A(1), A(2), A(3), A(4)]
"""

runner.timeit(name="deepcopy int", stmt="b=copy.deepcopy(i)", setup=setup)
runner.timeit(name="deepcopy dict", stmt=f"b=copy.deepcopy(a)", setup=setup)
runner.timeit(name="deepcopy dataclass", stmt=f"b=copy.deepcopy(dc)", setup=setup)
runner.timeit(name="deepcopy small list", stmt=f"b=copy.deepcopy(lst)", setup=setup)
runner.timeit(name="deepcopy list of dataclasses", stmt=f"b=copy.deepcopy(list_dc)", setup=setup)
