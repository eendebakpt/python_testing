# /// script
# requires-python = "==3.13"
# dependencies = ["matplotlib", "numpy", "quantify_core"]
# [tool.uv]
# exclude-newer = "2026-01-01T00:00:00Z"
# ///

# ruff: noqa: E402
import time

t0 = time.time()  # noqa
from math import pi

import matplotlib.pyplot as plt
import numpy as np
from quantify_core.visualization.SI_utilities import set_xlabel

dt = time.time() - t0
print(f"import time {dt:.2f} [s]")

rf_frequency = 123.43e6
sideband = 2e6

timestep = 25e-9
duration = 10e-6
tt = np.arange(0, duration, timestep)
signal = np.sin(2 * pi * sideband * tt) * np.sign(tt)
signal += 0.15 * np.random.rand(signal.size)
for kk in range(5):
    signal = np.convolve(signal, [1, 1, 1])

print(f"{signal.size=}")
F = np.fft.fft(signal)
freq = np.fft.fftfreq(signal.size, d=timestep)


fourier = np.fft.fft(signal)
n = signal.size
freq = np.fft.fftfreq(n, d=timestep)
idx = np.abs(freq) < 0.01e9
plt.figure(10)
plt.clf()
plt.plot(freq[idx], np.abs(F)[idx], ".b")
set_xlabel("Frequency", "Hz")
print("Close plot to continue")
plt.show()
