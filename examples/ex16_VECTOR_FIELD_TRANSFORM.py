"""Coordinates include translation, field vectors do not; scalar and explicit SIMD agree.

Implementation: src/rabbit_mojo/_mojo/geometry.mojo; tests/test_geometry.py.
Inputs below are deterministic; assertions specify the expected numerical output.
"""

import numpy as np
import rabbit_mojo as rm


def main():
    field = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 2.0, 3.0]]
    )
    t = np.eye(4)
    t[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    t[:3, 3] = [10, 20, 30]
    scalar = rm.transform_vectors(field, t)
    simd = rm.transform_vectors(field, t, simd=True)
    np.testing.assert_allclose(scalar, field @ t[:3, :3].T)
    np.testing.assert_allclose(simd, scalar)
    np.testing.assert_allclose(
        rm.transform_points(field, t) - scalar, np.broadcast_to(t[:3, 3], field.shape)
    )
    print(
        "samples",
        field,
        "vector field rotation",
        scalar,
        "coordinate transform",
        rm.transform_points(field, t),
    )


if __name__ == "__main__":
    main()
