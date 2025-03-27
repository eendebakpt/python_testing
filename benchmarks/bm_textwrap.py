# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///

import pyperf

runner = pyperf.Runner()

setup = """
import textwrap
newline=chr(10)

single_line = '   Hello world!'
dedent_docstring = textwrap.dedent.__doc__
no_indent = textwrap.dedent(dedent_docstring)
spaces = newline.join(['   ' + l for l in no_indent.split(newline)])
mixed = newline.join(['   Hello space', '\tHello tab', 'Hello'] * 20)
large_text = newline.join([dedent_docstring] * 40)


whitespace_only = newline.join(['  ', '\t', ''])
"""

runner.timeit(name="textwrap.dedent(single_line)", stmt="textwrap.dedent(single_line)", setup=setup)
runner.timeit(name="textwrap.dedent(no_indent)", stmt="textwrap.dedent(no_indent)", setup=setup)
runner.timeit(name="textwrap.dedent(spaces)", stmt="textwrap.dedent(spaces)", setup=setup)
runner.timeit(name="textwrap.dedent(mixed)", stmt="textwrap.dedent(mixed)", setup=setup)
runner.timeit(name="textwrap.dedent(large_text)", stmt="textwrap.dedent(large_text)", setup=setup)
