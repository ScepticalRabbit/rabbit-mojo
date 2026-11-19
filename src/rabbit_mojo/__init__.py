"""Scientific kernels implemented in native Mojo; no Python numerical fallback."""
from . import _native
from .types import IsotropicElastic, NewtonOptions, NewtonResult, Transform3D, Mode
from .bindings import Counter, scalar_roundtrip, collection_roundtrip, scale_inplace, native_error
from .linalg import dot, norm, add, scale_vector, normalize, cross, matmul, matvec, transpose, determinant, inverse
from .geometry import transform_points, transform_vectors, rotate_vectors, rotate_tensors, parallel_transform
from .tensor import tensor_invariants, principal_strains, maximum_engineering_shear, lame_parameters, elastic_stress, elastic_stiffness
from .numerical import differentiate, gradient2d, integrate, newton, least_squares
from .interpolation import interpolate, sample_image, q4_shape, q4_integrate
from .comptime import polynomial, generic_dot
