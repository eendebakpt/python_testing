# /// script
# requires-python = ">=3.10"
# dependencies = ['numpy', 'pyperf']
# ///

import pyperf

setup = """

empty = {}
d = {'a': 1, 'b': 2}
"""

runner = pyperf.Runner()
runner.timeit(name="create empty", stmt="x = {}", setup=setup)
runner.timeit(name="create small dict", stmt="x = {'a' : 1}", setup=setup)
runner.timeit(name="copy empty", stmt="empty.copy()", setup=setup)
runner.timeit(name="union of two small dicts", stmt="x = empty | d", setup=setup)
