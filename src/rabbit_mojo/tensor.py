"""Symmetric tensors use tensorial shear; principal values descend, directions are columns."""

import numpy as np
from . import _native
from .bindings import array
from .types import IsotropicElastic


def symmetric(a, dimensions=(3,)):
    array(a)
    if a.ndim < 2 or a.shape[-1] not in dimensions or a.shape[-2] != a.shape[-1]:
        raise ValueError("expected square symmetric tensors of supported dimension")
    if not np.isfinite(a).all() or not np.allclose(
        a, np.swapaxes(a, -1, -2), rtol=1e-13, atol=0
    ):
        raise ValueError("tensor must be finite and symmetric")
    return a


def tensor_invariants(a):
    symmetric(a)
    stats = np.empty(a.shape[:-2] + (6,))
    dev = np.empty(a.shape)
    _native.invariants(a, stats, dev)
    return {
        "I1": stats[..., 0],
        "I2": stats[..., 1],
        "I3": stats[..., 2],
        "volumetric": stats[..., 3],
        "norm": stats[..., 4],
        "J2": stats[..., 5],
        "deviator": dev,
    }


def principal_strains(a, *, directions=False):
    symmetric(a, (2, 3))
    n = a.shape[-1]
    values = np.empty(a.shape[:-1])
    axes = np.empty(a.shape)
    _native.principal(a, values, axes, n)
    return (values, axes) if directions else values


def maximum_engineering_shear(a):
    values = principal_strains(a)
    return values[..., 0] - values[..., -1]


def lame_parameters(material):
    if not isinstance(material, IsotropicElastic):
        raise TypeError("expected IsotropicElastic")
    material.__post_init__()
    return _native.material(material)


def elastic_stress(strain, material):
    symmetric(strain)
    lame_parameters(material)
    stress = np.empty(strain.shape)
    energy = np.empty(strain.shape[:-2] or (1,))
    _native.elastic(strain, material, stress, energy)
    return stress, energy if strain.ndim > 2 else float(energy[0])


def elastic_stiffness(material):
    """6x6 Voigt stiffness for engineering shear [xx,yy,zz,yz,xz,xy]."""
    lame_parameters(material)
    return _native.stiffness(material, np.empty((6, 6)))


def material_roundtrip(material):
    """Copy dataclass fields to native Material and construct a new Python dataclass."""
    lame_parameters(material)
    return IsotropicElastic(*_native.material_roundtrip(material))
