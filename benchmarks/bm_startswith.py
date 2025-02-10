import pyperf
runner = pyperf.Runner()

setup="""
x = 'abcdefghijklmnop'
y = 'abcdefghijklmnop_bbbbbbbbbbbbbbbbbb'

xu = x + '\u1234'
yu = y + '\u1234'
l = 'a' * 1000 + 'b'
x_startswith = x.startswith
"""

# Tested with ./python sw.py --rigorous -o main_startswith.json

if 1:
    runner.timeit(name="empty: x.startswith('')", stmt="x.startswith(''); y.startswith('')", setup=setup)
    runner.timeit(name="single-character match: x.startswith('a')", stmt="x.startswith('a'); y.startswith('a')", setup=setup)
    runner.timeit(name="single-character fail: x.startswith('q')", stmt="x.startswith('q'); y.startswith('q')", setup=setup)
    
    runner.timeit(name="two-character match: x.startswith('ab')", stmt="x.startswith('ab'); y.startswith('ab')", setup=setup)
    runner.timeit(name="two-character fail head: x.startswith('qb')", stmt="x.startswith('qb'); y.startswith('qb')", setup=setup)
    runner.timeit(name="two-character fail tail: x.startswith('aq')", stmt="x.startswith('aq'); y.startswith('aq')", setup=setup)
    runner.timeit(name="multi-character match: x.startswith('abcdefghijkl')", stmt="x.startswith('abcdefghijkl'); y.startswith('abcdefghijkl')", setup=setup)
    runner.timeit(name="multi-character fail midle: x.startswith('abcdef_hijkl')", stmt="x.startswith('abcdef_hijkl'); y.startswith('abcdef_hijkl')", setup=setup)

    runner.timeit(name="multi-character different kind match: xu.startswith('abcdefghijkl')", stmt="xu.startswith('abcdefghijkl'); yu.startswith('abcdefghijkl')", setup=setup)
    runner.timeit(name="multi-character different kind fail: xu.startswith('abcdef_hijkl')", stmt="xu.startswith('abcdefghijkl'); yu.startswith('abcdefghijkl')", setup=setup)
    
    runner.timeit(name="endswith single-character match: x.endswith('p')", stmt="x.startswith('a'); y.endswith('p')", setup=setup)
