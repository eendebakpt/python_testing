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

runner.timeit(name="copy int", stmt="b=copy.copy(i)", setup=setup)
runner.timeit(name="copy slice", stmt="b=copy.copy(sl)", setup=setup)
runner.timeit(name="copy dict", stmt="b=copy.copy(a)", setup=setup)
runner.timeit(name="copy dataclass", stmt="b=copy.copy(dc)", setup=setup)
runner.timeit(name="copy small list", stmt="b=copy.copy(lst)", setup=setup)
runner.timeit(name="copy small tuple", stmt="b=copy.copy(tpl)", setup=setup)
runner.timeit(name="copy list dataclasses", stmt="b=copy.copy(list_dc)", setup=setup)
