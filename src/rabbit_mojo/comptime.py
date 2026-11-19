"""Finite runtime dispatch to separately compiled fixed-type/fixed-shape kernels."""
import numpy as np
from . import _native
from .bindings import array

def polynomial(x,*,degree=2,unroll=True):
    if degree not in (2,3): raise ValueError("compiled degrees are 2 and 3")
    return _native.evaluate(float(x),(degree,bool(unroll)))

def generic_dot(a,b):
    if not isinstance(a,np.ndarray) or a.dtype not in (np.dtype('float32'),np.dtype('float64')): raise TypeError("expected float32 or float64")
    array(a,ndim=1,dtype=a.dtype); array(b,shape=a.shape,dtype=a.dtype)
    if a.size not in (2,3,4): raise ValueError("compiled lengths are 2, 3 and 4")
    return _native.generic_dot(a,b)
