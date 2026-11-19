"""Scientific kernels implemented in native Mojo; no Python numerical fallback."""
# ruff: noqa: E402
# The native loader must select the host interpreter before importing wrappers.

from ._bootstrap import load_native

_native = load_native()
del load_native
from .types import IsotropicElastic, NewtonOptions, NewtonResult, Transform3D, Mode
from .bindings import (
    Counter,
    scalar_roundtrip,
    collection_roundtrip,
    scale_inplace,
    native_error,
    native_affine,
)
from .linalg import (
    dot,
    norm,
    add,
    scale_vector,
    normalize,
    cross,
    matmul,
    matvec,
    transpose,
    determinant,
    inverse,
)
from .geometry import (
    transform_points,
    transform_vectors,
    rotate_vectors,
    rotate_tensors,
    parallel_transform,
)
from .tensor import (
    tensor_invariants,
    principal_strains,
    maximum_engineering_shear,
    lame_parameters,
    elastic_stress,
    elastic_stiffness,
    material_roundtrip,
)
from .numerical import differentiate, gradient2d, integrate, newton, least_squares
from .interpolation import interpolate, sample_image, q4_shape, q4_integrate
from .comptime import polynomial, generic_dot

__version__ = "2026.11.0"


__all__ = [
    "IsotropicElastic",
    "NewtonOptions",
    "NewtonResult",
    "Transform3D",
    "Mode",
    "Counter",
    "scalar_roundtrip",
    "collection_roundtrip",
    "scale_inplace",
    "native_error",
    "native_affine",
    "dot",
    "norm",
    "add",
    "scale_vector",
    "normalize",
    "cross",
    "matmul",
    "matvec",
    "transpose",
    "determinant",
    "inverse",
    "transform_points",
    "transform_vectors",
    "rotate_vectors",
    "rotate_tensors",
    "parallel_transform",
    "tensor_invariants",
    "principal_strains",
    "maximum_engineering_shear",
    "lame_parameters",
    "elastic_stress",
    "elastic_stiffness",
    "material_roundtrip",
    "differentiate",
    "gradient2d",
    "integrate",
    "newton",
    "least_squares",
    "interpolate",
    "sample_image",
    "q4_shape",
    "q4_integrate",
    "polynomial",
    "generic_dot",
    "__version__",
]
