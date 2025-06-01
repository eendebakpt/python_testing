import collections.abc
import functools
import warnings

import numpy as np
import numpy._core.numeric as _nx
from numpy._core import overrides
from numpy._core.multiarray import bincount
from numpy._core.numeric import (
    array,
    asanyarray,
    asarray,
    concatenate,
    isscalar,
    ndarray,
    take,
    zeros_like,
)
from numpy._core.numerictypes import typecodes
from numpy._core.umath import (
    add,
    minimum,
    subtract,
)
from numpy._utils import set_module

# needed in this module for compatibility
from numpy.lib._histograms_impl import histogram, histogramdd  # noqa: F401
from numpy.lib.tests.test_function_base import assert_array_equal

array_function_dispatch = functools.partial(overrides.array_function_dispatch, module="numpy")


__all__ = [
    "select",
    "piecewise",
    "trim_zeros",
    "copy",
    "iterable",
    "percentile",
    "diff",
    "gradient",
    "angle",
    "unwrap",
    "sort_complex",
    "flip",
    "rot90",
    "extract",
    "place",
    "vectorize",
    "asarray_chkfinite",
    "average",
    "bincount",
    "digitize",
    "cov",
    "corrcoef",
    "median",
    "sinc",
    "hamming",
    "hanning",
    "bartlett",
    "blackman",
    "kaiser",
    "trapezoid",
    "trapz",
    "i0",
    "meshgrid",
    "delete",
    "insert",
    "append",
    "interp",
    "quantile",
]

_QuantileMethods = {
    # --- HYNDMAN and FAN METHODS
    # Discrete methods
    "inverted_cdf": {
        "get_virtual_index": lambda n, quantiles: _inverted_cdf(n, quantiles),  # noqa: PLW0108
        "fix_gamma": None,  # should never be called
    },
    "averaged_inverted_cdf": {
        "get_virtual_index": lambda n, quantiles: (n * quantiles) - 1,
        "fix_gamma": lambda gamma, _: _get_gamma_mask(
            shape=gamma.shape, default_value=1.0, conditioned_value=0.5, where=gamma == 0
        ),
    },
    "closest_observation": {
        "get_virtual_index": lambda n, quantiles: _closest_observation(n, quantiles),  # noqa: PLW0108
        "fix_gamma": None,  # should never be called
    },
    # Continuous methods
    "interpolated_inverted_cdf": {
        "get_virtual_index": lambda n, quantiles: _compute_virtual_index(n, quantiles, 0, 1),
        "fix_gamma": lambda gamma, _: gamma,
    },
    "hazen": {
        "get_virtual_index": lambda n, quantiles: _compute_virtual_index(n, quantiles, 0.5, 0.5),
        "fix_gamma": lambda gamma, _: gamma,
    },
    "weibull": {
        "get_virtual_index": lambda n, quantiles: _compute_virtual_index(n, quantiles, 0, 0),
        "fix_gamma": lambda gamma, _: gamma,
    },
    # Default method.
    # To avoid some rounding issues, `(n-1) * quantiles` is preferred to
    # `_compute_virtual_index(n, quantiles, 1, 1)`.
    # They are mathematically equivalent.
    "linear": {
        "get_virtual_index": lambda n, quantiles: (n - np.int64(1)) * quantiles,
        "fix_gamma": lambda gamma, _: gamma,
    },
    "median_unbiased": {
        "get_virtual_index": lambda n, quantiles: _compute_virtual_index(n, quantiles, 1 / 3.0, 1 / 3.0),
        "fix_gamma": lambda gamma, _: gamma,
    },
    "normal_unbiased": {
        "get_virtual_index": lambda n, quantiles: _compute_virtual_index(n, quantiles, 3 / 8.0, 3 / 8.0),
        "fix_gamma": lambda gamma, _: gamma,
    },
    # --- OTHER METHODS
    "lower": {
        "get_virtual_index": lambda n, quantiles: np.floor((n - 1) * quantiles).astype(np.intp),
        "fix_gamma": None,  # should never be called, index dtype is int
    },
    "higher": {
        "get_virtual_index": lambda n, quantiles: np.ceil((n - 1) * quantiles).astype(np.intp),
        "fix_gamma": None,  # should never be called, index dtype is int
    },
    "midpoint": {
        "get_virtual_index": lambda n, quantiles: 0.5 * (np.floor((n - 1) * quantiles) + np.ceil((n - 1) * quantiles)),
        "fix_gamma": lambda gamma, index: _get_gamma_mask(
            shape=gamma.shape, default_value=0.5, conditioned_value=0.0, where=index % 1 == 0
        ),
    },
    "nearest": {
        "get_virtual_index": lambda n, quantiles: np.around((n - 1) * quantiles).astype(np.intp),
        "fix_gamma": None,
        # should never be called, index dtype is int
    },
}


def _ureduce(a, func, keepdims=False, **kwargs):
    """
    Internal Function.
    Call `func` with `a` as first argument swapping the axes to use extended
    axis on functions that don't support it natively.

    Returns result and a.shape with axis dims set to 1.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    func : callable
        Reduction function capable of receiving a single axis argument.
        It is called with `a` as first argument followed by `kwargs`.
    kwargs : keyword arguments
        additional keyword arguments to pass to `func`.

    Returns
    -------
    result : tuple
        Result of func(a, **kwargs) and a.shape with axis dims set to 1
        which can be used to reshape the result to the same shape a ufunc with
        keepdims=True would produce.

    """
    a = np.asanyarray(a)
    axis = kwargs.get("axis")
    out = kwargs.get("out")

    if keepdims is np._NoValue:
        keepdims = False

    nd = a.ndim
    if axis is not None:
        axis = _nx.normalize_axis_tuple(axis, nd)

        if keepdims and out is not None:
            index_out = tuple(0 if i in axis else slice(None) for i in range(nd))
            kwargs["out"] = out[(Ellipsis,) + index_out]

        if len(axis) == 1:
            kwargs["axis"] = axis[0]
        else:
            keep = set(range(nd)) - set(axis)
            nkeep = len(keep)
            # swap axis that should not be reduced to front
            for i, s in enumerate(sorted(keep)):
                a = a.swapaxes(i, s)
            # merge reduced axis
            a = a.reshape(a.shape[:nkeep] + (-1,))
            kwargs["axis"] = -1
    elif keepdims and out is not None:
        index_out = (0,) * nd
        kwargs["out"] = out[(Ellipsis,) + index_out]

    r = func(a, **kwargs)

    if out is not None:
        return out

    if keepdims:
        if axis is None:
            index_r = (np.newaxis,) * nd
        else:
            index_r = tuple(np.newaxis if i in axis else slice(None) for i in range(nd))
        r = r[(Ellipsis,) + index_r]

    return r


def _weights_are_valid(weights, a, axis):
    """Validate weights array.

    We assume, weights is not None.
    """
    wgt = np.asanyarray(weights)

    # Sanity checks
    if a.shape != wgt.shape:
        if axis is None:
            raise TypeError("Axis must be specified when shapes of a and weights differ.")
        if wgt.shape != tuple(a.shape[ax] for ax in axis):
            raise ValueError("Shape of weights must be consistent with shape of a along specified axis.")

        # setup wgt to broadcast along axis
        wgt = wgt.transpose(np.argsort(axis))
        wgt = wgt.reshape(tuple((s if ax in axis else 1) for ax, s in enumerate(a.shape)))
    return wgt


def _average_dispatcher(a, axis=None, weights=None, returned=None, *, keepdims=None):
    return (a, weights)


@array_function_dispatch(_average_dispatcher)
def average(a, axis=None, weights=None, returned=False, *, keepdims=np._NoValue):
    """
    Compute the weighted average along the specified axis.

    Parameters
    ----------
    a : array_like
        Array containing data to be averaged. If `a` is not an array, a
        conversion is attempted.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to average `a`.  The default,
        `axis=None`, will average over all of the elements of the input array.
        If axis is negative it counts from the last to the first axis.
        If axis is a tuple of ints, averaging is performed on all of the axes
        specified in the tuple instead of a single axis or all the axes as
        before.
    weights : array_like, optional
        An array of weights associated with the values in `a`. Each value in
        `a` contributes to the average according to its associated weight.
        The array of weights must be the same shape as `a` if no axis is
        specified, otherwise the weights must have dimensions and shape
        consistent with `a` along the specified axis.
        If `weights=None`, then all data in `a` are assumed to have a
        weight equal to one.
        The calculation is::

            avg = sum(a * weights) / sum(weights)

        where the sum is over all included elements.
        The only constraint on the values of `weights` is that `sum(weights)`
        must not be 0.
    returned : bool, optional
        Default is `False`. If `True`, the tuple (`average`, `sum_of_weights`)
        is returned, otherwise only the average is returned.
        If `weights=None`, `sum_of_weights` is equivalent to the number of
        elements over which the average is taken.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `a`.
        *Note:* `keepdims` will not work with instances of `numpy.matrix`
        or other classes whose methods do not support `keepdims`.

        .. versionadded:: 1.23.0

    Returns
    -------
    retval, [sum_of_weights] : array_type or double
        Return the average along the specified axis. When `returned` is `True`,
        return a tuple with the average as the first element and the sum
        of the weights as the second element. `sum_of_weights` is of the
        same type as `retval`. The result dtype follows a general pattern.
        If `weights` is None, the result dtype will be that of `a` , or ``float64``
        if `a` is integral. Otherwise, if `weights` is not None and `a` is non-
        integral, the result type will be the type of lowest precision capable of
        representing values of both `a` and `weights`. If `a` happens to be
        integral, the previous rules still applies but the result dtype will
        at least be ``float64``.

    Raises
    ------
    ZeroDivisionError
        When all weights along axis are zero. See `numpy.ma.average` for a
        version robust to this type of error.
    TypeError
        When `weights` does not have the same shape as `a`, and `axis=None`.
    ValueError
        When `weights` does not have dimensions and shape consistent with `a`
        along specified `axis`.

    See Also
    --------
    mean

    ma.average : average for masked arrays -- useful if your data contains
                 "missing" values
    numpy.result_type : Returns the type that results from applying the
                        numpy type promotion rules to the arguments.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.arange(1, 5)
    >>> data
    array([1, 2, 3, 4])
    >>> np.average(data)
    2.5
    >>> np.average(np.arange(1, 11), weights=np.arange(10, 0, -1))
    4.0

    >>> data = np.arange(6).reshape((3, 2))
    >>> data
    array([[0, 1],
           [2, 3],
           [4, 5]])
    >>> np.average(data, axis=1, weights=[1./4, 3./4])
    array([0.75, 2.75, 4.75])
    >>> np.average(data, weights=[1./4, 3./4])
    Traceback (most recent call last):
        ...
    TypeError: Axis must be specified when shapes of a and weights differ.

    With ``keepdims=True``, the following result has shape (3, 1).

    >>> np.average(data, axis=1, keepdims=True)
    array([[0.5],
           [2.5],
           [4.5]])

    >>> data = np.arange(8).reshape((2, 2, 2))
    >>> data
    array([[[0, 1],
            [2, 3]],
           [[4, 5],
            [6, 7]]])
    >>> np.average(data, axis=(0, 1), weights=[[1./4, 3./4], [1., 1./2]])
    array([3.4, 4.4])
    >>> np.average(data, axis=0, weights=[[1./4, 3./4], [1., 1./2]])
    Traceback (most recent call last):
        ...
    ValueError: Shape of weights must be consistent
    with shape of a along specified axis.
    """
    a = np.asanyarray(a)

    if axis is not None:
        axis = _nx.normalize_axis_tuple(axis, a.ndim, argname="axis")

    if keepdims is np._NoValue:
        # Don't pass on the keepdims argument if one wasn't given.
        keepdims_kw = {}
    else:
        keepdims_kw = {"keepdims": keepdims}

    if weights is None:
        avg = a.mean(axis, **keepdims_kw)
        avg_as_array = np.asanyarray(avg)
        scl = avg_as_array.dtype.type(a.size / avg_as_array.size)
    else:
        wgt = _weights_are_valid(weights=weights, a=a, axis=axis)

        if issubclass(a.dtype.type, (np.integer, np.bool)):
            result_dtype = np.result_type(a.dtype, wgt.dtype, "f8")
        else:
            result_dtype = np.result_type(a.dtype, wgt.dtype)

        scl = wgt.sum(axis=axis, dtype=result_dtype, **keepdims_kw)
        if np.any(scl == 0.0):
            raise ZeroDivisionError("Weights sum to zero, can't be normalized")

        avg = avg_as_array = np.multiply(a, wgt, dtype=result_dtype).sum(axis, **keepdims_kw) / scl

    if returned:
        if scl.shape != avg_as_array.shape:
            scl = np.broadcast_to(scl, avg_as_array.shape).copy()
        return avg, scl
    else:
        return avg


@set_module("numpy")
def asarray_chkfinite(a, dtype=None, order=None):
    """Convert the input to an array, checking for NaNs or Infs.

    Parameters
    ----------
    a : array_like
        Input data, in any form that can be converted to an array.  This
        includes lists, lists of tuples, tuples, tuples of tuples, tuples
        of lists and ndarrays.  Success requires no NaNs or Infs.
    dtype : data-type, optional
        By default, the data-type is inferred from the input data.
    order : {'C', 'F', 'A', 'K'}, optional
        Memory layout.  'A' and 'K' depend on the order of input array a.
        'C' row-major (C-style),
        'F' column-major (Fortran-style) memory representation.
        'A' (any) means 'F' if `a` is Fortran contiguous, 'C' otherwise
        'K' (keep) preserve input order
        Defaults to 'C'.

    Returns
    -------
    out : ndarray
        Array interpretation of `a`.  No copy is performed if the input
        is already an ndarray.  If `a` is a subclass of ndarray, a base
        class ndarray is returned.

    Raises
    ------
    ValueError
        Raises ValueError if `a` contains NaN (Not a Number) or Inf (Infinity).

    See Also
    --------
    asarray : Create and array.
    asanyarray : Similar function which passes through subclasses.
    ascontiguousarray : Convert input to a contiguous array.
    asfortranarray : Convert input to an ndarray with column-major
                     memory order.
    fromiter : Create an array from an iterator.
    fromfunction : Construct an array by executing a function on grid
                   positions.

    Examples
    --------
    >>> import numpy as np

    Convert a list into an array. If all elements are finite, then
    ``asarray_chkfinite`` is identical to ``asarray``.

    >>> a = [1, 2]
    >>> np.asarray_chkfinite(a, dtype=float)
    array([1., 2.])

    Raises ValueError if array_like contains Nans or Infs.

    >>> a = [1, 2, np.inf]
    >>> try:
    ...     np.asarray_chkfinite(a)
    ... except ValueError:
    ...     print('ValueError')
    ...
    ValueError

    """
    a = asarray(a, dtype=dtype, order=order)
    if a.dtype.char in typecodes["AllFloat"] and not np.isfinite(a).all():
        raise ValueError("array must not contain infs or NaNs")
    return a


def _piecewise_dispatcher(x, condlist, funclist, *args, **kw):
    yield x
    # support the undocumented behavior of allowing scalars
    if np.iterable(condlist):
        yield from condlist


@array_function_dispatch(_piecewise_dispatcher)
def piecewise(x, condlist, funclist, *args, **kw):
    """
    Evaluate a piecewise-defined function.

    Given a set of conditions and corresponding functions, evaluate each
    function on the input data wherever its condition is true.

    Parameters
    ----------
    x : ndarray or scalar
        The input domain.
    condlist : list of bool arrays or bool scalars
        Each boolean array corresponds to a function in `funclist`.  Wherever
        `condlist[i]` is True, `funclist[i](x)` is used as the output value.

        Each boolean array in `condlist` selects a piece of `x`,
        and should therefore be of the same shape as `x`.

        The length of `condlist` must correspond to that of `funclist`.
        If one extra function is given, i.e. if
        ``len(funclist) == len(condlist) + 1``, then that extra function
        is the default value, used wherever all conditions are false.
    funclist : list of callables, f(x,*args,**kw), or scalars
        Each function is evaluated over `x` wherever its corresponding
        condition is True.  It should take a 1d array as input and give an 1d
        array or a scalar value as output.  If, instead of a callable,
        a scalar is provided then a constant function (``lambda x: scalar``) is
        assumed.
    args : tuple, optional
        Any further arguments given to `piecewise` are passed to the functions
        upon execution, i.e., if called ``piecewise(..., ..., 1, 'a')``, then
        each function is called as ``f(x, 1, 'a')``.
    kw : dict, optional
        Keyword arguments used in calling `piecewise` are passed to the
        functions upon execution, i.e., if called
        ``piecewise(..., ..., alpha=1)``, then each function is called as
        ``f(x, alpha=1)``.

    Returns
    -------
    out : ndarray
        The output is the same shape and type as x and is found by
        calling the functions in `funclist` on the appropriate portions of `x`,
        as defined by the boolean arrays in `condlist`.  Portions not covered
        by any condition have a default value of 0.


    See Also
    --------
    choose, select, where

    Notes
    -----
    This is similar to choose or select, except that functions are
    evaluated on elements of `x` that satisfy the corresponding condition from
    `condlist`.

    The result is::

            |--
            |funclist[0](x[condlist[0]])
      out = |funclist[1](x[condlist[1]])
            |...
            |funclist[n2](x[condlist[n2]])
            |--

    Examples
    --------
    >>> import numpy as np

    Define the signum function, which is -1 for ``x < 0`` and +1 for ``x >= 0``.

    >>> x = np.linspace(-2.5, 2.5, 6)
    >>> np.piecewise(x, [x < 0, x >= 0], [-1, 1])
    array([-1., -1., -1.,  1.,  1.,  1.])

    Define the absolute value, which is ``-x`` for ``x <0`` and ``x`` for
    ``x >= 0``.

    >>> np.piecewise(x, [x < 0, x >= 0], [lambda x: -x, lambda x: x])
    array([2.5,  1.5,  0.5,  0.5,  1.5,  2.5])

    Apply the same function to a scalar value.

    >>> y = -2
    >>> np.piecewise(y, [y < 0, y >= 0], [lambda x: -x, lambda x: x])
    array(2)

    """
    x = asanyarray(x)
    n2 = len(funclist)

    # undocumented: single condition is promoted to a list of one condition
    if isscalar(condlist) or (not isinstance(condlist[0], (list, ndarray)) and x.ndim != 0):
        condlist = [condlist]

    condlist = asarray(condlist, dtype=bool)
    n = len(condlist)

    if n == n2 - 1:  # compute the "otherwise" condition.
        condelse = ~np.any(condlist, axis=0, keepdims=True)
        condlist = np.concatenate([condlist, condelse], axis=0)
        n += 1
    elif n != n2:
        raise ValueError(f"with {n} condition(s), either {n} or {n + 1} functions are expected")

    y = zeros_like(x)
    for cond, func in zip(condlist, funclist):
        if not isinstance(func, collections.abc.Callable):
            y[cond] = func
        else:
            vals = x[cond]
            if vals.size > 0:
                y[cond] = func(vals, *args, **kw)

    return y


def _select_dispatcher(condlist, choicelist, default=None):
    yield from condlist
    yield from choicelist


@array_function_dispatch(_select_dispatcher)
def select(condlist, choicelist, default=0):
    """
    Return an array drawn from elements in choicelist, depending on conditions.

    Parameters
    ----------
    condlist : list of bool ndarrays
        The list of conditions which determine from which array in `choicelist`
        the output elements are taken. When multiple conditions are satisfied,
        the first one encountered in `condlist` is used.
    choicelist : list of ndarrays
        The list of arrays from which the output elements are taken. It has
        to be of the same length as `condlist`.
    default : scalar, optional
        The element inserted in `output` when all conditions evaluate to False.

    Returns
    -------
    output : ndarray
        The output at position m is the m-th element of the array in
        `choicelist` where the m-th element of the corresponding array in
        `condlist` is True.

    See Also
    --------
    where : Return elements from one of two arrays depending on condition.
    take, choose, compress, diag, diagonal

    Examples
    --------
    >>> import numpy as np

    Beginning with an array of integers from 0 to 5 (inclusive),
    elements less than ``3`` are negated, elements greater than ``3``
    are squared, and elements not meeting either of these conditions
    (exactly ``3``) are replaced with a `default` value of ``42``.

    >>> x = np.arange(6)
    >>> condlist = [x<3, x>3]
    >>> choicelist = [-x, x**2]
    >>> np.select(condlist, choicelist, 42)
    array([ 0,  -1,  -2, 42, 16, 25])

    When multiple conditions are satisfied, the first one encountered in
    `condlist` is used.

    >>> condlist = [x<=4, x>3]
    >>> choicelist = [x, x**2]
    >>> np.select(condlist, choicelist, 55)
    array([ 0,  1,  2,  3,  4, 25])

    """
    # Check the size of condlist and choicelist are the same, or abort.
    if len(condlist) != len(choicelist):
        raise ValueError("list of cases must be same length as list of conditions")

    # Now that the dtype is known, handle the deprecated select([], []) case
    if len(condlist) == 0:
        raise ValueError("select with an empty condition list is not possible")

    # TODO: This preserves the Python int, float, complex manually to get the
    #       right `result_type` with NEP 50.  Most likely we will grow a better
    #       way to spell this (and this can be replaced).
    choicelist = [choice if type(choice) in (int, float, complex) else np.asarray(choice) for choice in choicelist]
    choicelist.append(default if type(default) in (int, float, complex) else np.asarray(default))

    try:
        dtype = np.result_type(*choicelist)
    except TypeError as e:
        msg = f"Choicelist and default value do not have a common dtype: {e}"
        raise TypeError(msg) from None

    # Convert conditions to arrays and broadcast conditions and choices
    # as the shape is needed for the result. Doing it separately optimizes
    # for example when all choices are scalars.
    condlist = np.broadcast_arrays(*condlist)
    choicelist = np.broadcast_arrays(*choicelist)

    # If cond array is not an ndarray in boolean format or scalar bool, abort.
    for i, cond in enumerate(condlist):
        if cond.dtype.type is not np.bool:
            raise TypeError(f"invalid entry {i} in condlist: should be boolean ndarray")

    if choicelist[0].ndim == 0:
        # This may be common, so avoid the call.
        result_shape = condlist[0].shape
    else:
        result_shape = np.broadcast_arrays(condlist[0], choicelist[0])[0].shape

    result = np.full(result_shape, choicelist[-1], dtype)

    # Use np.copyto to burn each choicelist array onto result, using the
    # corresponding condlist as a boolean mask. This is done in reverse
    # order since the first choice should take precedence.
    choicelist = choicelist[-2::-1]
    condlist = condlist[::-1]
    for choice, cond in zip(choicelist, condlist):
        np.copyto(result, choice, where=cond)

    return result


def _copy_dispatcher(a, order=None, subok=None):
    return (a,)


@array_function_dispatch(_copy_dispatcher)
def copy(a, order="K", subok=False):
    """
    Return an array copy of the given object.

    Parameters
    ----------
    a : array_like
        Input data.
    order : {'C', 'F', 'A', 'K'}, optional
        Controls the memory layout of the copy. 'C' means C-order,
        'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous,
        'C' otherwise. 'K' means match the layout of `a` as closely
        as possible. (Note that this function and :meth:`ndarray.copy` are very
        similar, but have different default values for their order=
        arguments.)
    subok : bool, optional
        If True, then sub-classes will be passed-through, otherwise the
        returned array will be forced to be a base-class array (defaults to False).

    Returns
    -------
    arr : ndarray
        Array interpretation of `a`.

    See Also
    --------
    ndarray.copy : Preferred method for creating an array copy

    Notes
    -----
    This is equivalent to:

    >>> np.array(a, copy=True)  #doctest: +SKIP

    The copy made of the data is shallow, i.e., for arrays with object dtype,
    the new array will point to the same objects.
    See Examples from `ndarray.copy`.

    Examples
    --------
    >>> import numpy as np

    Create an array x, with a reference y and a copy z:

    >>> x = np.array([1, 2, 3])
    >>> y = x
    >>> z = np.copy(x)

    Note that, when we modify x, y changes, but not z:

    >>> x[0] = 10
    >>> x[0] == y[0]
    True
    >>> x[0] == z[0]
    False

    Note that, np.copy clears previously set WRITEABLE=False flag.

    >>> a = np.array([1, 2, 3])
    >>> a.flags["WRITEABLE"] = False
    >>> b = np.copy(a)
    >>> b.flags["WRITEABLE"]
    True
    >>> b[0] = 3
    >>> b
    array([3, 2, 3])
    """
    return array(a, order=order, subok=subok, copy=True)


# Basic operations


def _percentile_dispatcher(
    a, q, axis=None, out=None, overwrite_input=None, method=None, keepdims=None, *, weights=None, interpolation=None
):
    return (a, q, out, weights)


@array_function_dispatch(_percentile_dispatcher)
def percentile(
    a,
    q,
    axis=None,
    out=None,
    overwrite_input=False,
    method="linear",
    keepdims=False,
    *,
    weights=None,
    interpolation=None,
):
    if interpolation is not None:
        method = _check_interpolation_as_method(method, interpolation, "percentile")

    a = np.asanyarray(a)
    if a.dtype.kind == "c":
        raise TypeError("a must be an array of real numbers")

    # Use dtype of array if possible (e.g., if q is a python int or float)
    # by making the divisor have the dtype of the data array.
    q = np.true_divide(q, 100 if a.dtype.kind == "f" else 100, out=...)
    # q = np.true_divide(q, a.dtype.type(100) if a.dtype.kind == "f" else 100, out=...)
    if not _quantile_is_valid(q):
        raise ValueError("Percentiles must be in the range [0, 100]")

    if weights is not None:
        if method != "inverted_cdf":
            msg = f"Only method 'inverted_cdf' supports weights. Got: {method}."
            raise ValueError(msg)
        if axis is not None:
            axis = _nx.normalize_axis_tuple(axis, a.ndim, argname="axis")
        weights = _weights_are_valid(weights=weights, a=a, axis=axis)
        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative.")

    return _quantile_unchecked(a, q, axis, out, overwrite_input, method, keepdims, weights)


def _quantile_dispatcher(
    a, q, axis=None, out=None, overwrite_input=None, method=None, keepdims=None, *, weights=None, interpolation=None
):
    return (a, q, out, weights)


@array_function_dispatch(_quantile_dispatcher)
def quantile(
    a,
    q,
    axis=None,
    out=None,
    overwrite_input=False,
    method="linear",
    keepdims=False,
    *,
    weights=None,
    interpolation=None,
):
    if interpolation is not None:
        method = _check_interpolation_as_method(method, interpolation, "quantile")

    a = np.asanyarray(a)
    if a.dtype.kind == "c":
        raise TypeError("a must be an array of real numbers")

    # Use dtype of array if possible (e.g., if q is a python int or float).
    if isinstance(q, (int, float)) and a.dtype.kind == "f":
        #        q_dtype = np.result_type(a.dtype)
        #       q = np.asanyarray(q, dtype=q_dtype)
        q = np.asanyarray(q)
    else:
        q = np.asanyarray(q)

    if not _quantile_is_valid(q):
        raise ValueError("Quantiles must be in the range [0, 1]")

    if weights is not None:
        if method != "inverted_cdf":
            msg = f"Only method 'inverted_cdf' supports weights. Got: {method}."
            raise ValueError(msg)
        if axis is not None:
            axis = _nx.normalize_axis_tuple(axis, a.ndim, argname="axis")
        weights = _weights_are_valid(weights=weights, a=a, axis=axis)
        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative.")

    return _quantile_unchecked(a, q, axis, out, overwrite_input, method, keepdims, weights)


def _quantile_unchecked(
    a, q, axis=None, out=None, overwrite_input=False, method="linear", keepdims=False, weights=None
):
    """Assumes that q is in [0, 1], and is an ndarray"""
    return _ureduce(
        a,
        func=_quantile_ureduce_func,
        q=q,
        weights=weights,
        keepdims=keepdims,
        axis=axis,
        out=out,
        overwrite_input=overwrite_input,
        method=method,
    )


def _quantile_is_valid(q):
    # avoid expensive reductions, relevant for arrays with < O(1000) elements
    if q.ndim == 1 and q.size < 10:
        for i in range(q.size):
            if not (0.0 <= q[i] <= 1.0):
                return False
    elif not (q.min() >= 0 and q.max() <= 1):
        return False
    return True


def _check_interpolation_as_method(method, interpolation, fname):
    # Deprecated NumPy 1.22, 2021-11-08
    warnings.warn(
        f"the `interpolation=` argument to {fname} was renamed to "
        "`method=`, which has additional options.\n"
        "Users of the modes 'nearest', 'lower', 'higher', or "
        "'midpoint' are encouraged to review the method they used. "
        "(Deprecated NumPy 1.22)",
        DeprecationWarning,
        stacklevel=4,
    )
    if method != "linear":
        # sanity check, we assume this basically never happens
        raise TypeError(
            "You shall not pass both `method` and `interpolation`!\n"
            "(`interpolation` is Deprecated in favor of `method`)"
        )
    return interpolation


def _compute_virtual_index(n, quantiles, alpha: float, beta: float):
    """
    Compute the floating point indexes of an array for the linear
    interpolation of quantiles.
    n : array_like
        The sample sizes.
    quantiles : array_like
        The quantiles values.
    alpha : float
        A constant used to correct the index computed.
    beta : float
        A constant used to correct the index computed.

    alpha and beta values depend on the chosen method
    (see quantile documentation)

    Reference:
    Hyndman&Fan paper "Sample Quantiles in Statistical Packages",
    DOI: 10.1080/00031305.1996.10473566
    """
    return n * quantiles + (alpha + quantiles * (1 - alpha - beta)) - 1


def _get_gamma(virtual_indexes, previous_indexes, method, dtype=None):
    """ """
    gamma = np.asanyarray(virtual_indexes - previous_indexes)
    gamma = method["fix_gamma"](gamma, virtual_indexes)
    # Ensure both that we have an array, and that we keep the dtype
    # (which may have been matched to the input array).
    if dtype is None:
        dtype = virtual_indexes.dtype

    return np.asanyarray(gamma, dtype=dtype)


def _lerp(a, b, t, out=None):
    diff_b_a = subtract(b, a)
    # asanyarray is a stop-gap until gh-13105
    lerp_interpolation = asanyarray(add(a, diff_b_a * t, out=out))
    subtract(
        b,
        diff_b_a * (1 - t),
        out=lerp_interpolation,
        where=t >= 0.5,
        casting="unsafe",
        dtype=type(lerp_interpolation.dtype),
    )
    if lerp_interpolation.ndim == 0 and out is None:
        lerp_interpolation = lerp_interpolation[()]  # unpack 0d arrays
    return lerp_interpolation


def _get_gamma_mask(shape, default_value, conditioned_value, where):
    out = np.full(shape, default_value)
    np.copyto(out, conditioned_value, where=where, casting="unsafe")
    return out


def _discrete_interpolation_to_boundaries(index, gamma_condition_fun):
    previous = np.floor(index)
    next = previous + 1
    gamma = index - previous
    res = _get_gamma_mask(
        shape=index.shape, default_value=next, conditioned_value=previous, where=gamma_condition_fun(gamma, index)
    ).astype(np.intp)
    # Some methods can lead to out-of-bound integers, clip them:
    res[res < 0] = 0
    return res


def _closest_observation(n, quantiles):
    # "choose the nearest even order statistic at g=0" (H&F (1996) pp. 362).
    # Order is 1-based so for zero-based indexing round to nearest odd index.
    gamma_fun = lambda gamma, index: (gamma == 0) & (np.floor(index) % 2 == 1)
    return _discrete_interpolation_to_boundaries((n * quantiles) - 1 - 0.5, gamma_fun)


def _inverted_cdf(n, quantiles):
    gamma_fun = lambda gamma, _: (gamma == 0)
    return _discrete_interpolation_to_boundaries((n * quantiles) - 1, gamma_fun)


def _quantile_ureduce_func(
    a: np.array,
    q: np.array,
    weights: np.array,
    axis: int | None = None,
    out=None,
    overwrite_input: bool = False,
    method="linear",
) -> np.array:
    if q.ndim > 2:
        # The code below works fine for nd, but it might not have useful
        # semantics. For now, keep the supported dimensions the same as it was
        # before.
        raise ValueError("q must be a scalar or 1d")
    if overwrite_input:
        if axis is None:
            axis = 0
            arr = a.ravel()
            wgt = None if weights is None else weights.ravel()
        else:
            arr = a
            wgt = weights
    elif axis is None:
        axis = 0
        arr = a.flatten()
        wgt = None if weights is None else weights.flatten()
    else:
        arr = a.copy()
        wgt = weights
    result = _quantile(arr, quantiles=q, axis=axis, method=method, out=out, weights=wgt)
    return result


def _get_indexes(arr, virtual_indexes, valid_values_count):
    """
    Get the valid indexes of arr neighbouring virtual_indexes.
    Note
    This is a companion function to linear interpolation of
    Quantiles

    Returns
    -------
    (previous_indexes, next_indexes): Tuple
        A Tuple of virtual_indexes neighbouring indexes
    """
    previous_indexes = np.asanyarray(np.floor(virtual_indexes))
    next_indexes = np.asanyarray(previous_indexes + 1)
    indexes_above_bounds = virtual_indexes >= valid_values_count - 1
    # When indexes is above max index, take the max value of the array
    if indexes_above_bounds.any():
        previous_indexes[indexes_above_bounds] = -1
        next_indexes[indexes_above_bounds] = -1
    # When indexes is below min index, take the min value of the array
    indexes_below_bounds = virtual_indexes < 0
    if indexes_below_bounds.any():
        previous_indexes[indexes_below_bounds] = 0
        next_indexes[indexes_below_bounds] = 0
    if np.issubdtype(arr.dtype, np.inexact):
        # After the sort, slices having NaNs will have for last element a NaN
        virtual_indexes_nans = np.isnan(virtual_indexes)
        if virtual_indexes_nans.any():
            previous_indexes[virtual_indexes_nans] = -1
            next_indexes[virtual_indexes_nans] = -1
    previous_indexes = previous_indexes.astype(np.intp)
    next_indexes = next_indexes.astype(np.intp)
    return previous_indexes, next_indexes


def _quantile(
    arr: np.array,
    quantiles: np.array,
    axis: int = -1,
    method="linear",
    out=None,
    weights=None,
):
    """
    Private function that doesn't support extended axis or keepdims.
    These methods are extended to this function using _ureduce
    See nanpercentile for parameter usage
    It computes the quantiles of the array for the given axis.
    A linear interpolation is performed based on the `interpolation`.

    By default, the method is "linear" where alpha == beta == 1 which
    performs the 7th method of Hyndman&Fan.
    With "median_unbiased" we get alpha == beta == 1/3
    thus the 8th method of Hyndman&Fan.
    """
    # --- Setup
    arr = np.asanyarray(arr)
    values_count = arr.shape[axis]
    # The dimensions of `q` are prepended to the output shape, so we need the
    # axis being sampled from `arr` to be last.
    if axis != 0:  # But moveaxis is slow, so only call it if necessary.
        arr = np.moveaxis(arr, axis, destination=0)
    supports_nans = np.issubdtype(arr.dtype, np.inexact) or arr.dtype.kind in "Mm"

    if weights is None:
        # --- Computation of indexes
        # Index where to find the value in the sorted array.
        # Virtual because it is a floating point value, not an valid index.
        # The nearest neighbours are used for interpolation
        try:
            method_props = _QuantileMethods[method]
        except KeyError:
            raise ValueError(f"{method!r} is not a valid method. Use one of: {_QuantileMethods.keys()}") from None
        virtual_indexes = method_props["get_virtual_index"](values_count, quantiles)
        virtual_indexes = np.asanyarray(virtual_indexes)
        # print(f'_quantile {virtual_indexes=} {virtual_indexes.dtype=}')

        if method_props["fix_gamma"] is None:
            supports_integers = True
        else:
            int_virtual_indices = np.issubdtype(virtual_indexes.dtype, np.integer)
            supports_integers = method == "linear" and int_virtual_indices

        if supports_integers:
            # No interpolation needed, take the points along axis
            if supports_nans:
                # may contain nan, which would sort to the end
                arr.partition(
                    concatenate((virtual_indexes.ravel(), [-1])),
                    axis=0,
                )
                slices_having_nans = np.isnan(arr[-1, ...])
            else:
                # cannot contain nan
                arr.partition(virtual_indexes.ravel(), axis=0)
                slices_having_nans = np.array(False, dtype=bool)
            result = take(arr, virtual_indexes, axis=0, out=out)
        else:
            previous_indexes, next_indexes = _get_indexes(arr, virtual_indexes, values_count)
            # --- Sorting
            arr.partition(
                np.unique(
                    np.concatenate(
                        (
                            [0, -1],
                            previous_indexes.ravel(),
                            next_indexes.ravel(),
                        )
                    )
                ),
                axis=0,
            )
            if supports_nans:
                slices_having_nans = np.isnan(arr[-1, ...])
            else:
                slices_having_nans = None
            # --- Get values from indexes
            previous = arr[previous_indexes]
            next = arr[next_indexes]
            # --- Linear interpolation
            if arr.dtype.kind in "iu":
                gtype = None  # np.result_type(1., arr.dtype)
            elif arr.dtype.kind == "f":
                # make sure to the return value matches the input array type
                gtype = arr.dtype
            else:
                gtype = virtual_indexes.dtype
            gamma = _get_gamma(virtual_indexes, previous_indexes, method_props, dtype=gtype)
            print(f"_get_gamma -> {gamma} {gamma.dtype=} {arr.dtype=} {virtual_indexes.dtype=}")
            result_shape = virtual_indexes.shape + (1,) * (arr.ndim - 1)
            gamma = gamma.reshape(result_shape)
            result = _lerp(previous, next, gamma, out=out)
    else:
        # Weighted case
        # This implements method="inverted_cdf", the only supported weighted
        # method, which needs to sort anyway.
        weights = np.asanyarray(weights)
        if axis != 0:
            weights = np.moveaxis(weights, axis, destination=0)
        index_array = np.argsort(arr, axis=0, kind="stable")

        # arr = arr[index_array, ...]  # but this adds trailing dimensions of
        # 1.
        arr = np.take_along_axis(arr, index_array, axis=0)
        if weights.shape == arr.shape:
            weights = np.take_along_axis(weights, index_array, axis=0)
        else:
            # weights is 1d
            weights = weights.reshape(-1)[index_array, ...]

        if supports_nans:
            # may contain nan, which would sort to the end
            slices_having_nans = np.isnan(arr[-1, ...])
        else:
            # cannot contain nan
            slices_having_nans = np.array(False, dtype=bool)

        # We use the weights to calculate the empirical cumulative
        # distribution function cdf
        cdf = weights.cumsum(axis=0, dtype=np.float64)
        cdf /= cdf[-1, ...]  # normalization to 1
        # Search index i such that
        #   sum(weights[j], j=0..i-1) < quantile <= sum(weights[j], j=0..i)
        # is then equivalent to
        #   cdf[i-1] < quantile <= cdf[i]
        # Unfortunately, searchsorted only accepts 1-d arrays as first
        # argument, so we will need to iterate over dimensions.

        # Without the following cast, searchsorted can return surprising
        # results, e.g.
        #   np.searchsorted(np.array([0.2, 0.4, 0.6, 0.8, 1.]),
        #                   np.array(0.4, dtype=np.float32), side="left")
        # returns 2 instead of 1 because 0.4 is not binary representable.
        if quantiles.dtype.kind == "f":
            cdf = cdf.astype(quantiles.dtype)
        # Weights must be non-negative, so we might have zero weights at the
        # beginning leading to some leading zeros in cdf. The call to
        # np.searchsorted for quantiles=0 will then pick the first element,
        # but should pick the first one larger than zero. We
        # therefore simply set 0 values in cdf to -1.
        if np.any(cdf[0, ...] == 0):
            cdf[cdf == 0] = -1

        def find_cdf_1d(arr, cdf):
            indices = np.searchsorted(cdf, quantiles, side="left")
            # We might have reached the maximum with i = len(arr), e.g. for
            # quantiles = 1, and need to cut it to len(arr) - 1.
            indices = minimum(indices, values_count - 1)
            result = take(arr, indices, axis=0)
            return result

        r_shape = arr.shape[1:]
        if quantiles.ndim > 0:
            r_shape = quantiles.shape + r_shape
        if out is None:
            result = np.empty_like(arr, shape=r_shape)
        else:
            if out.shape != r_shape:
                msg = f"Wrong shape of argument 'out', shape={r_shape} is required; got shape={out.shape}."
                raise ValueError(msg)
            result = out

        # See apply_along_axis, which we do for axis=0. Note that Ni = (,)
        # always, so we remove it here.
        Nk = arr.shape[1:]
        for kk in np.ndindex(Nk):
            result[(...,) + kk] = find_cdf_1d(arr[np.s_[:,] + kk], cdf[np.s_[:,] + kk])

        # Make result the same as in unweighted inverted_cdf.
        if result.shape == () and result.dtype == np.dtype("O"):
            result = result.item()

    if np.any(slices_having_nans):
        if result.ndim == 0 and out is None:
            # can't write to a scalar, but indexing will be correct
            result = arr[-1]
        else:
            np.copyto(result, arr[-1, ...], where=slices_having_nans)
    return result


if 0:
    a = arr = np.zeros(65521, dtype=np.float16)
    arr[:10] = 1
    z = percentile(arr, 50)
    print(z)
    assert not np.isnan(z)
    assert z == 0

    q = 0.5
    z = quantile(arr, q)
    print(z)
    assert not np.isnan(z)

# %%
input_dtype = np.float16
# linear-False-29-float32-float32-percentile-40.0
if 1:
    input_dtype = np.float32
    output_dtype = np.float32
    method = "linear"
    expected = 29
    weights = False

    q = 40.0

# interpolated_inverted_cdf-False-20-n-float64-percentile-40.0]
if 0:
    input_dtype = "n"
    output_dtype = np.float64
    method = "interpolated_inverted_cdf"
    expected = 20
    weights = False

    q = 40.0


# linear-False-29-P-float64-quantile-0.4
if 1:
    method = "linear"
    weights = False
    expected = 29
    q = 0.4
    input_dtype = "N"
    output_dtype = np.float64
#    quantiles = np.array(q, dtype=arr.dtype)

if 0:
    arr = np.asarray([15.0, 20.0, 35.0, 40.0, 50.0], dtype=input_dtype)
    n = arr.size

    # v = percentile(arr, q)
    v = np.quantile(arr, q)
    assert v.dtype == output_dtype, f"return type {v.dtype} expected {output_dtype}"
    assert v == expected, f"value {v} {expected=}"

    print(v)
    print(v - 29)

# %%

import numpy as np

arr = np.zeros(65521, dtype=np.float16)
arr[:10] = 1
methods = [
    "inverted_cdf",
    "averaged_inverted_cdf",
    "closest_observation",
    "interpolated_inverted_cdf",
    "hazen",
    "weibull",
    "linear",
    "median_unbiased",
    "normal_unbiased",
]

if 0:
    for method in methods:
        print()

        z = np.percentile(arr, 50, method=method)
        assert z.dtype == arr.dtype
        assert z == 0
        # print(f'{method=} output={z} {z.dtype=}')

        z = percentile(arr, 50, method=method)
        # print(f'{method=} output={z} {z.dtype=}')


# %%

x = quantile([1, 2, 3], 0.5, method="nearest")
x


arr = np.array([1, 2])
q = Fraction(1)
r = quantile(arr, q=q)
print(f"{arr=} {q=} {r=}")
# %%

#    @pytest.mark.parametrize("dtype", ["m8[D]", "M8[s]"])
#    @pytest.mark.parametrize("pos", [0, 23, 10])
#    def test_nat_basic(self, dtype, pos):

dtype = "m8[D]"
pos = 23
if 0:
    # TODO: Note that times have dubious rounding as of fixing NaTs!
    # NaT and NaN should behave the same, do basic tests for NaT:
    a = np.arange(0, 24, dtype=dtype)
    a[pos] = "NaT"
    res = percentile(a, 30)
    assert res.dtype == dtype
    assert np.isnat(res)
    res = percentile(a, [30, 60])
    assert res.dtype == dtype
    assert np.isnat(res).all()

    a = np.arange(0, 24 * 3, dtype=dtype).reshape(-1, 3)
    a[pos, 1] = "NaT"
    res = percentile(a, 30, axis=0)
    assert_array_equal(np.isnat(res), [False, True, False])
# %%
#    @pytest.mark.parametrize("qtype", [np.float16, Fraction])
#    def test_percentile_gh_29003(self, qtype, ):
from fractions import Fraction

qtype = Fraction
qtype = np.float16

# failure mechnanism: quantiles is not cast to arr.dtype by default any longer, then input q=.4 ends up being float64
# casting would cast to Fraction type, which is the desired behavour.

if 0:
    zero = qtype(0)
    one = qtype(0)
    data = [zero] * 65521
    a = np.array(data)
    a[:20_000] = one
    z = percentile(a, qtype(50))
    assert z == zero
    assert np.array(z).dtype == a.dtype, f"result dtype {z.dtype} does not match {a.dtype}"

    z = percentile(a, 50)
    assert z == zero
    assert z.dtype == a.dtype, f"result dtype {z.dtype} does not match {a.dtype}"

    z = quantile(a, 0.9)
    assert z == one
    assert z.dtype == a.dtype

# %%

if 0:
    n = arr.size

    np.int64(n - 1) * np.float16(1.6)

    w = (n - np.int64(1)) * q
    print(f" {w=}")

    q = np.array(q, dtype=np.float16)
    w = (n - np.int64(1)) * q
    print(f" {w=}")

    20 * 0.4 + 35 * 0.6

    # %%
    import numpy as np

    print(f"{np.float16(0.4):.41f}")
    print(f"{np.float64(0.4):.41f}")

    # %%
    import numpy as np

    print(f"{np.__version__=}")

    f16 = np.float16(0.4)
    print(f" {f16=}")
    print(f" {4*f16=} {4*f16=:.10f}")
    print(f" {np.int64(4)*f16=}")

    a16 = np.array([0.4], dtype=np.float16)
    print(f" {a16=}")
    print(f" {4*a16=}")
    print(f" {np.int64(4)*a16=}")  # not equal to 1.6
    print(f" {a16*np.int64(4)=}")  # not equal to 1.6

    print(f"{4 * f16:.41f}")
