import timeit

import numpy as np

print(np)

x = np.array([1.0])
y = np.array([1.0], dtype=object)

number = 1_000_000
for alist in [
    [
        x,
    ],
    [x, x],
    [x, y],
]:
    dt = timeit.timeit("as_series(alist, trim=True)", globals=globals(), number=number)
    print(f"as_series({alist}): {1e6 * dt / number:.3f} [us]")


# python -m timeit -s "import numpy as np; x=np.array([1.]); y=np.array([1.], dtype=object); alist=[x,]" "as_series(alist, trim=True)" # noqa

"""
as_series([array([1.])]): 2.234 [us]
as_series([array([1.]), array([1.])]): 3.065 [us]
as_series([array([1.]), array([1.0], dtype=object)]): 2.655 [us]
"""
