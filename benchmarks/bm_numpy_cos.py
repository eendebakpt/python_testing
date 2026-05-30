import pyperf

runner = pyperf.Runner()

setup = """
import numpy as np
x = np.array([1., 2.])
f = -np.float64(2.2)
p = -2.2
"""

exec(setup)

if 1:
    runner.timeit(name=f"np.abs({repr(p)})", stmt="np.abs(p)", setup=setup)
    runner.timeit(name=f"np.cos({repr(p)})", stmt="np.cos(p)", setup=setup)
if 1:
    runner.timeit(name=f"np.abs({repr(f)})", stmt="np.abs(f)", setup=setup)
    runner.timeit(name=f"np.cos({repr(f)})", stmt="np.cos(f)", setup=setup)
if 1:
    runner.timeit(name="np.abs(x)", stmt="np.abs(x)", setup=setup)
    runner.timeit(name="np.cos(x)", stmt="np.cos(x)", setup=setup)
    runner.timeit(name="np.add.reduce(x)", stmt="np.add.reduce(x)", setup=setup)
    runner.timeit(name="np.add.accumulate(x)", stmt="np.add.accumulate(x)", setup=setup)


""" 
Results from: commit c3b55c964b104b56c57ac1ab1a0e53f9df0d41b0 (HEAD -> ufunc_loop_cache, pte/ufunc_loop_cache)

    
### PR summary

The `PyUFunc_DefaultLegacyInnerLoopSelector` is quite slow and called for every ufunc execution (for details see  https://github.com/numpy/numpy/pull/31018).

In this PR we cache the selected loop on the `PyArrayMethodObject`. Benchmark results:
```
np.abs(-2.2): Mean +- std dev: [main_fix] 129 ns +- 2 ns -> [pr_fix] 110 ns +- 6 ns: 1.18x faster
np.cos(-2.2): Mean +- std dev: [main_fix] 121 ns +- 9 ns -> [pr_fix] 108 ns +- 8 ns: 1.12x faster
np.abs(np.float64(-2.2)): Mean +- std dev: [main_fix] 173 ns +- 5 ns -> [pr_fix] 155 ns +- 8 ns: 1.12x faster
np.cos(np.float64(-2.2)): Mean +- std dev: [main_fix] 160 ns +- 7 ns -> [pr_fix] 148 ns +- 7 ns: 1.09x faster
np.abs(x): Mean +- std dev: [main_fix] 445 ns +- 3 ns -> [pr_fix] 420 ns +- 4 ns: 1.06x faster
np.cos(x): Mean +- std dev: [main_fix] 442 ns +- 12 ns -> [pr_fix] 426 ns +- 12 ns: 1.04x faster
np.add.reduce(x): Mean +- std dev: [main_fix] 1.11 us +- 0.03 us -> [pr_fix] 1.10 us +- 0.04 us: 1.01x faster
np.add.accumulate(x): Mean +- std dev: [main_fix] 612 ns +- 26 ns -> [pr_fix] 579 ns +- 17 ns: 1.06x faster

Geometric mean: 1.08x faster
```

For reviewers: in the first commit legacy ufunc loops and cached (core of this PR). In the second commit we move the caching to the level of strided loops (not necessary for performance, but seems nicer). In the third commit the freelist for `NpyAuxData` is removed to simplify code (it is no longer needed).

We could probably refactor `generate_umath.py` to generate strided loops directly. In what way we could avoid the additional fields on `PyArrayMethodObject`. The amount of changes would be much larger though, and we maybe would have to maintain two sets of generation (for backwards compatibility reasons). If desired, I can explore this option.

<!-- Please take some time to make it easier for us maintainers to understand
  and review your PR. Describe the pull request, using the questions below as
  guidance, and link to any relevant issues and PRs.

  
  Also, have you hit all [the guidelines](https://numpy.org/devdocs/dev/index.html#guidelines)?
  And have you filled out the disclosure section below?

-->


#### AI Disclosure

Claude code was used in analysing the performance patterns and suggested to cache the ufunc loops. Significant parts of the code in the PR have been written by the tools.
i
<!-- If AI was used in the preparation of this pull request, please disclose
the tool(s) used, how they were used, and specify what code or text is AI generated.
If no AI tools were used, please write "No AI tools used" in this section. Read our
policy on AI generated code at
https://numpy.org/devdocs/dev/ai_policy.html.

In particular, all interaction is to be done by humans, including submission of PRs.
-->


"""
