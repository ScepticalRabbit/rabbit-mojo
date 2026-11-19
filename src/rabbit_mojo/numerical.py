"""Native stencils, reductions, selected-function Newton and Householder QR."""

import math
import numpy as np
from . import _native
from .bindings import array
from .types import NewtonOptions, NewtonResult


def _axis(axis, ndim):
    if type(axis) is not int or not -ndim <= axis < ndim:
        raise ValueError("invalid axis")
    return axis % ndim


def differentiate(y, spacing, *, axis=-1, method="central"):
    array(y)
    axis = _axis(axis, y.ndim)
    methods = {"forward": 0, "backward": 1, "central": 2, "second": 3}
    if method not in methods:
        raise ValueError("unknown difference method")
    if y.shape[axis] < (4 if method == "second" else 3):
        raise ValueError("too few samples")
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing must be finite and positive")
    out = np.empty(y.shape)
    return _native.differentiate(y, out, (spacing, axis, methods[method]))


def gradient2d(y, spacing=(1.0, 1.0)):
    array(y, ndim=2)
    if len(spacing) != 2:
        raise ValueError("expected two spacings")
    return tuple(differentiate(y, h, axis=i) for i, h in enumerate(spacing))


def integrate(y, x=None, *, spacing=1.0, axis=-1, method="trapezoid"):
    array(y)
    axis = _axis(axis, y.ndim)
    n = y.shape[axis]
    if n < 2:
        raise ValueError("integration needs at least two points")
    if x is None:
        if not math.isfinite(spacing) or spacing <= 0:
            raise ValueError("spacing must be finite and positive")
        x = np.arange(n, dtype=float) * spacing
    array(x, shape=(n,))
    if not np.isfinite(x).all() or not np.all(np.diff(x) > 0):
        raise ValueError("x must be finite and strictly increasing")
    if method not in ("trapezoid", "simpson"):
        raise ValueError("unknown integration method")
    if method == "simpson" and (
        n % 2 != 1 or not np.allclose(np.diff(x), x[1] - x[0], rtol=1e-12, atol=0)
    ):
        raise ValueError("Simpson requires odd sample count and a uniform grid")
    shape = y.shape[:axis] + y.shape[axis + 1 :]
    out = np.empty(shape or (1,))
    _native.integrate(y, x, out, (axis, method == "simpson"))
    return out if shape else float(out[0])


def newton(
    initial, options=None, *, function="square_minus_two", raise_on_failure=False
):
    options = options or NewtonOptions()
    if not isinstance(options, NewtonOptions):
        raise TypeError("expected NewtonOptions")
    if not math.isfinite(initial):
        raise ValueError("initial value must be finite")
    if (
        not math.isfinite(options.atol)
        or not math.isfinite(options.rtol)
        or options.atol <= 0
        or options.rtol < 0
        or type(options.max_iterations) is not int
        or options.max_iterations < 1
    ):
        raise ValueError("invalid Newton tolerances or iteration count")
    functions = {"square_minus_two": 0, "cos_minus_x": 1, "cube_minus_one": 2}
    if function not in functions:
        raise ValueError("unknown native function")
    result = NewtonResult(
        *_native.newton(
            float(initial),
            (options.atol, options.rtol, options.max_iterations, functions[function]),
        )
    )
    if raise_on_failure and not result.converged:
        raise RuntimeError("Newton did not converge")
    return result


def least_squares(a, b):
    array(a, ndim=2)
    array(b, shape=(a.shape[0],))
    if (
        not 1 <= a.shape[1] <= a.shape[0]
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("expected finite overdetermined design matrix and response")
    out = np.empty(a.shape[1])
    return _native.least_squares(a, b, out)
