"""Small native vector and matrix operations, float64, strided inputs allowed."""

import numpy as np
from . import _native
from .bindings import array, output


def dot(a, b, *, simd=False):
    array(a, ndim=1)
    array(b, shape=a.shape)
    if simd and (not a.flags.c_contiguous or not b.flags.c_contiguous):
        raise ValueError("SIMD reduction requires contiguous inputs")
    return _native.dot(a, b, simd)


def norm(a):
    return dot(a, a) ** 0.5


def _vector(a, b=None, op=0, factor=1):
    array(a, ndim=1)
    if a.size not in (2, 3, 4):
        raise ValueError("vector size must be 2, 3 or 4")
    if b is None:
        b = a
    array(b, shape=a.shape)
    out = output(None, a.shape)
    return _native.vector(a, b, out, op, factor)


def add(a, b):
    return _vector(a, b)


def scale_vector(a, factor):
    return _vector(a, op=1, factor=float(factor))


def normalize(a):
    return _vector(a, op=2)


def cross(a, b):
    array(a, shape=(3,))
    array(b, shape=(3,))
    return _vector(a, b, op=3)


def _matrix(a, b=None, op=0):
    array(a, ndim=2)
    if a.shape not in ((2, 2), (3, 3)):
        raise ValueError("expected 2x2 or 3x3 matrix")
    if not np.isfinite(a).all():
        raise ValueError("matrix must be finite")
    if b is None:
        b = a
    array(b)
    if op == 0:
        array(b, shape=a.shape)
    if op == 2:
        array(b, shape=(a.shape[0],))
    out = output(None, (a.shape[0],) if op == 2 else a.shape)
    return _native.matrix(a, b, out, op)


def matmul(a, b):
    return _matrix(a, b)


def transpose(a):
    return _matrix(a, op=1)


def matvec(a, b):
    return _matrix(a, b, 2)


def determinant(a):
    return _matrix(a, op=3)


def inverse(a):
    return _matrix(a, op=4)
