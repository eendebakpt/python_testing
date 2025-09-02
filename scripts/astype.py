#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 23:43:42 2025

@author: eendebakpt
"""

import numpy as np
import timeit
n=10_000
#n=2

f=np.float32(1.2)
print('--')
f.astype(float)

f=np.float64(1.2)
d=float
print('--')
f.astype(d)

dt=timeit.timeit('f.astype(d)', number=n, globals=globals())
print(f'{1e6*dt/n} [us/loop]')

d=f.dtype
print('--')
f.astype(d)

dt=timeit.timeit('f.astype(d)', number=n, globals=globals())
print(f'{1e6*dt/n} [us/loop]')