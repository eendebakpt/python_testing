#!/usr/bin/env python3
"""Interleaved benchmark for itertools.cycle

@author: eendebakpt
"""

import subprocess

import matplotlib.pyplot as plt
import numpy as np

test_script = "/home/eendebakpt/python_testing/benchmarks/bm_itertools_cycle.py"
cmds = ["/home/eendebakpt/cpython0/python {test_script", "/home/eendebakpt/cpython/python {test_script}"]

verbose = False
tt = []
for ii in range(600):
    print(f"run {ii}")
    for cmd in cmds:
        p = subprocess.run(cmd, shell=True, check=True, capture_output=True, encoding="utf-8")
        if verbose:
            print(f"Command {p.args} exited with {p.returncode} code, output: \n{p.stdout}")
        tt.append(float(p.stdout))

tt_main = tt[::2]
tt_pr = tt[1::2]

# %% Show results
plt.figure(10)
plt.clf()
plt.plot(tt_main[::2], ".", label="Main")
plt.plot(tt_pr[1::2], ".", label="PR")
plt.axhline(np.mean(tt_main), color="C0", label="mean for main")
plt.axhline(np.mean(tt_pr), color="C1", label="mean for PR")
plt.ylabel("Execution time [s]")
plt.legend()

gain = np.mean(tt_main) / np.mean(tt_pr)
plt.title(f"Performance gain: {gain:.3f}")
