"""Regular grids: coordinates follow array axes. Images: (x,y), integer pixel centres."""

import math
import numpy as np
from . import _native
from .bindings import array


def interpolate(
    field, coordinates, *, origin=None, spacing=None, policy="raise", method="linear"
):
    array(field)
    array(coordinates, ndim=2)
    if field.ndim not in (1, 2, 3) or min(field.shape) < 2:
        raise ValueError("grid needs 2+ samples on 1 to 3 axes")
    if coordinates.shape[1] != field.ndim:
        raise ValueError("coordinate dimension must match grid rank")
    if not np.isfinite(coordinates).all():
        raise ValueError("coordinates must be finite")
    if policy not in ("raise", "clamp") or method not in ("linear", "nearest"):
        raise ValueError("invalid interpolation policy/method")
    origin = tuple(origin) if origin is not None else (0.0,) * field.ndim
    spacing = tuple(spacing) if spacing is not None else (1.0,) * field.ndim
    if (
        len(origin) != field.ndim
        or len(spacing) != field.ndim
        or any(not math.isfinite(v) for v in origin)
        or any(not math.isfinite(v) or v <= 0 for v in spacing)
    ):
        raise ValueError("expected finite origin and positive spacing per axis")
    out = np.empty(len(coordinates))
    return _native.interpolate(
        field,
        coordinates,
        out,
        (method == "nearest", policy == "clamp", origin, spacing),
    )


def sample_image(image, xy, *, method="bilinear", policy="raise"):
    array(image, ndim=2)
    array(xy, ndim=2)
    if xy.shape[1] != 2:
        raise ValueError("expected pixel coordinates (N,2)")
    if method not in ("nearest", "bilinear"):
        raise ValueError("unknown image sampling method")
    return interpolate(
        image,
        xy[:, ::-1],
        method="linear" if method == "bilinear" else "nearest",
        policy=policy,
    )


def q4_shape(xi, eta):
    if not math.isfinite(xi) or not math.isfinite(eta):
        raise ValueError("coordinates must be finite")
    n = np.empty(4)
    gradients = np.empty((4, 2))
    _native.q4_shape((xi, eta), n, gradients)
    return n, gradients


def q4_integrate(vertices, *, powers=(0, 0)):
    array(vertices, shape=(4, 2))
    if not np.isfinite(vertices).all():
        raise ValueError("vertices must be finite")
    if len(powers) != 2 or any(type(p) is not int or not 0 <= p <= 3 for p in powers):
        raise ValueError("powers must be two integers in 0..3")
    # Check all corners as well: positive Gauss determinants alone do not prove validity.
    for xi, eta in ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)):
        _, derivatives = q4_shape(xi, eta)
        if np.linalg.det(vertices.T @ derivatives) <= 0:
            raise ValueError("inverted or degenerate Q4 element")
    samples = np.empty((4, 7))
    integral = _native.q4_integrate(vertices, samples, powers)
    return integral, {
        "coordinates": samples[:, :2],
        "determinants": samples[:, 2],
        "jacobians": samples[:, 3:].reshape(4, 2, 2),
    }
