"""Validated synchronous boundary. Array inputs are borrowed, never coerced/copied."""
import math
import numpy as np
from . import _native
from .types import Mode

Counter = _native.Counter

def array(a, *, ndim=None, shape=None, dtype=np.float64, writable=False):
    if not isinstance(a, np.ndarray):
        raise TypeError("expected a NumPy ndarray; convert explicitly")
    if a.dtype != np.dtype(dtype) or not a.dtype.isnative:
        raise TypeError(f"expected native {np.dtype(dtype)}")
    if ndim is not None and a.ndim != ndim:
        raise ValueError(f"expected rank {ndim}")
    if shape is not None and a.shape != tuple(shape):
        raise ValueError(f"expected shape {tuple(shape)}, got {a.shape}")
    if a.ndim < 1 or a.ndim > 4:
        raise ValueError("supported rank is 1 through 4")
    if not a.flags.aligned or any(s % a.itemsize for s in a.strides):
        raise ValueError("unaligned storage or fractional element stride")
    # Check the actual storage owner, including views made by as_strided.
    owner = a
    seen = set()
    while getattr(owner, 'base', None) is not None and id(owner) not in seen:
        seen.add(id(owner))
        owner = owner.base
    if a.size:
        lo, hi = np.lib.array_utils.byte_bounds(a)
        if isinstance(owner, np.ndarray):
            low, high = np.lib.array_utils.byte_bounds(owner)
        else:
            try:
                backing = np.frombuffer(owner, dtype=np.uint8)
                low, high = np.lib.array_utils.byte_bounds(backing)
            except (TypeError, BufferError, ValueError):
                raise ValueError("cannot establish backing storage bounds") from None
        if lo < low or hi > high:
            raise ValueError("view extends outside backing storage")
    if writable:
        if not a.flags.writeable:
            raise ValueError("output array is read-only")
        # Conservative proof that no two logical elements overlap.
        extent = a.itemsize
        for stride, n in sorted((abs(s), n) for s, n in zip(a.strides, a.shape) if n > 1):
            if stride < extent:
                raise ValueError("overlapping output strides")
            extent += (n - 1) * stride
    return a

def output(out, shape, *inputs):
    if out is None:
        return np.empty(shape, dtype=np.float64)
    array(out, shape=shape, writable=True)
    for a in inputs:
        if np.shares_memory(out, a):
            raise ValueError("output overlaps an input")
    return out

def scalar_roundtrip(integer, real, flag, text, optional=None):
    if type(integer) is not int or not -(2**31) <= integer < 2**31:
        raise OverflowError("integer must fit signed Int32 (also mapped to native 64-bit Int)")
    if type(flag) is not bool or not isinstance(text, str):
        raise TypeError("expected bool and str")
    if not isinstance(real, (int, float)) or not math.isfinite(real):
        raise ValueError("real must be finite")
    if optional is not None and not isinstance(optional, (float, int)):
        raise TypeError("optional must be a number or None")
    return _native.scalars((integer, real, flag, text, optional))

def collection_roundtrip(items, coordinate, settings, mode=Mode.ADD):
    if not isinstance(items, list) or any(type(x) not in (int, float) for x in items):
        raise TypeError("items must be a numeric list")
    if not isinstance(coordinate, tuple) or len(coordinate) != 2 or any(type(x) not in (int, float) for x in coordinate):
        raise TypeError("coordinate must be a numeric pair")
    if not isinstance(settings, dict) or set(settings) != {'scale'} or type(settings['scale']) not in (int, float):
        raise TypeError("settings must contain one numeric 'scale'")
    if not isinstance(mode, Mode):
        raise TypeError("mode must be a Mode enum")
    return _native.collections((items, coordinate, settings, int(mode)))

def scale_inplace(a, factor=2):
    if not isinstance(a, np.ndarray) or a.dtype not in (np.dtype('float64'), np.dtype('float32'), np.dtype('int64')):
        raise TypeError("supported dtypes: float64, float32, int64")
    array(a, dtype=a.dtype, writable=True)
    if not math.isfinite(factor):
        raise ValueError("factor must be finite")
    if a.dtype == np.int64:
        if int(factor) != factor:
            raise ValueError("integer scaling requires an integer factor")
        if not -(2**63) <= factor < 2**63:
            raise OverflowError("factor does not fit Int64")
        if a.size:
            low, high = int(a.min()) * int(factor), int(a.max()) * int(factor)
            if min(low, high) < -(2**63) or max(low, high) >= 2**63:
                raise OverflowError("integer product does not fit Int64")
    return _native.scale(a, factor)

def native_error(message="native demonstration error"):
    return _native.fail(message)
