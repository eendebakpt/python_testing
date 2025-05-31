import numpy as np
import numpy.linalg._linalg
from numpy.linalg._linalg import array_function_dispatch, single, double, inexact, _realType, isComplexType, _unary_dispatcher
from numpy.linalg._linalg import LinAlgError, _assert_stacked_2d, _assert_stacked_square, _umath_linalg, asarray, _commonType
from numpy import  promote_types

A=np.random.rand( 3,3)
A
from functools import lru_cache
#import scipy.linalg

def _assert_stacked_2d(*arrays):
    for a in arrays:
        if a.ndim < 2:
            raise LinAlgError('%d-dimensional array given. Array must be '
                    'at least two-dimensional' % a.ndim)


def _assert_stacked_square(*arrays):
    for a in arrays:
        if a.ndim < 2:
            raise LinAlgError('%d-dimensional array given. Array must be '
                    'at least two-dimensional' % a.ndim)
        m, n = a.shape[-2:]
        if m != n:
           raise LinAlgError('Last 2 dimensions of the array must be square')

@lru_cache
def commonType(*dtypes):
    arrays=[np.array([1], dtype=d) for d in dtypes]
    return _commonType(*arrays)

#A=np.random.rand(20,20)

@array_function_dispatch(_unary_dispatcher)
def det(a):
    a = asarray(a)
    #return None # 108 ns 
    #_assert_stacked_2d(a)
    #return None # 186 ns 
    _assert_stacked_square(a)
    #return None # 316 ns 
    #t, result_t = _commonType(a)
    t, result_t = commonType(a.dtype)
    signature = 'D->D' if isComplexType(t) else 'd->d'
    #return None # 500 ns 
    r = _umath_linalg.det(a, signature=signature)
    #return r, result_t # 1.5 us 
    #print(r.dtype, result_t)
    #r = r.astype(result_t, copy=False)
    if r.dtype.type is not result_t:
        # needed at all???
        r = r.astype(result_t, copy=False)
        # 1.9 us
    return r # 1.55 us

@array_function_dispatch(_unary_dispatcher)
def det(a):
    a = asarray(a)
    _assert_stacked_square(a)
    t, result_t = _commonType(a)
    signature = 'D->D' if isComplexType(t) else 'd->d'
    r = _umath_linalg.det(a, signature=signature)
    r = r.astype(result_t, copy=False)
    return r

@array_function_dispatch(_unary_dispatcher)
def det2(a):
    a = asarray(a)
    _assert_stacked_square(a)
    #t, result_t = _commonType(a)
    #signature = 'D->D' if isComplexType(t) else 'd->d'
    d = promote_types(a.dtype, double)
    r = _umath_linalg.det(a, dtype=d)
    r = r.astype(d, copy=False)
    return r


r=det(A)
r=det2(A)
r=det(A.astype(np.float32))
#r=det2(A)

%timeit det2(A)


#%%
#det=np.linalg.det
r=det(A)
r=det2(A)
%timeit det(A)
%timeit det2(A)

#%%

#%timeit scipy.linalg.det(A)

""" To optimize

# 2.2 -> 1.8 -> 1.55 -> 1.47
# so over 40% reduction in computation time for 3x3 array. 
# others such as inv, qt, cond will also gain performance for small arrays
#
# we still have a measureable performance gain for 8x8 arrays or 20x20 arrays!


lru_cache on _commonType (move to dtype args)

r.astype for a scalar with copy=False is _slow_ (internal convertion to np.ndarray?)

see: https://github.com/numpy/numpy/blob/4542c5f241055dcef3daf7a62d9578623305b390/numpy/_core/src/multiarray/scalartypes.c.src#L2599 ?
see: https://github.com/numpy/numpy/blob/4542c5f241055dcef3daf7a62d9578623305b390/numpy/_core/src/multiarray/scalartypes.c.src#L2134''

ends up in slow path: https://github.com/numpy/numpy/blob/4542c5f241055dcef3daf7a62d9578623305b390/numpy/_core/src/multiarray/scalartypes.c.src#L127
https://github.com/numpy/numpy/blob/main/numpy/_core/src/multiarray/scalarapi.c#L650

easy solution: fast path check

_assert_stacked_2d/_assert_stacked_square are _always_ in pairs, we should combine them

the siganture is _always_ computed from the commonType results, lets include that in the cache as well

_convertarray is unused, can be removed (it is in _linalg which is recent, although from numpy.linalg.linalg import _convertarray works? )

These seem all safe, and adding hardly any complexity
"""

#%%
a=np.float64(3.3)
print(a.astype(float, copy=False) is a)
%timeit a.astype(float, copy=False)

#%%    
@lru_cache
def commonType(*arrays):
    return _commonType(*arrays)

def _commonType(*arrays):
    # in lite version, use higher precision (always double or cdouble)
    result_type = single
    is_complex = False
    for a in arrays:
        type_ = a.dtype.type
        if issubclass(type_, inexact):
            if isComplexType(type_):
                is_complex = True
            rt = _realType(type_, default=None)
            if rt is double:
                result_type = double
            elif rt is None:
                # unsupported inexact scalar
                raise TypeError("array type %s is unsupported in linalg" %
                        (a.dtype.name,))
        else:
            result_type = double
    if is_complex:
        result_type = _complex_types_map[result_type]
        return cdouble, result_type
    else:
        return double, result_type
_commonType(np.zeros( (3,3), dtype=int))

%timeit _commonType(A)

#%%
from numpy._core import _multiarray_umath
def _commonType(a):
    return _multiarray_umath.result_type(a.dtype, double), _multiarray_umath.result_type(a.dtype, single)
    #return np.promote_types(a.dtype, double), np.promote_types(a.dtype, single)
    return np.result_type(a.dtype, double), np.result_type(a.dtype, single)

_commonType(np.zeros( (3,3), dtype=int))

%timeit _commonType(A)

#%%
A=np.random.rand( 2,2)
%timeit np.linalg.det(A)
%timeit np.linalg.qr(A)
%timeit np.linalg.inv(A)
%timeit np.linalg.cond(A)

# 2.19 μs ± 84.6 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)

#%%
import numpy as np
r = np.float64(.1)
tp = r.dtype.type

s = r.astype(tp)
print(s is r) # False
help(r.astype) # Scalar method identical to the corresponding array attribute.
help(np.ndarray.astype) # copy : bool, ... If set to false .... the input array is returned instead of a copy.


s = r.astype(float, copy=True)
print(s is r) # False

#%%
import copy
import numpy as np
r = np.float64(.1)
x=np.array([1., 2.])
x32=np.array([1., 2.], dtype=np.float32)
double=np.float64.dtype

#for x in [1.2, r, r.dtype]:
#    print(copy.copy(x) is x)
    
%timeit np.result_type(x, x32)
%timeit np.result_type(x, x)
