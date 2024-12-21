import pyperf

setup = """
from operator import methodcaller as mc
arr = []
arr2 = [1, 2]
call = mc('sort')
call_positional_arg = mc('index', 1)
call_kwarg = mc('sort', reverse=True)


class A:
	i = 0
	def work(self, *args, **kwargs):
		self.i = self.i + 2
a = A()
kw = {str(i): i for i in range(10)}
call_many_kwargs = mc('work', **kw)

call_mixed_arg_kwarg = mc('work', 2, x=3)
"""

runner = pyperf.Runner()
runner.timeit(name="call", stmt="call(arr)", setup=setup)
runner.timeit(name="creation", stmt="call = mc('sort')", setup=setup)
runner.timeit(name="creation+call", stmt="call = mc('sort'); call(arr)", setup=setup)
runner.timeit(name="call_positional_arg", stmt="call_positional_arg(arr2)", setup=setup)
runner.timeit(name="call kwarg", stmt="call_kwarg(arr)", setup=setup)
runner.timeit(name="creation kwarg", stmt="call = mc('sort', reverse=True)", setup=setup)
runner.timeit(name="creation+call kwarg", stmt="call = mc('sort', reverse=True); call(arr)", setup=setup)
runner.timeit(name="call mixed position and kwarg", stmt="call_mixed_arg_kwarg(a)", setup=setup)
runner.timeit(name="call many kwarg", stmt="call_many_kwargs(a)", setup=setup)

