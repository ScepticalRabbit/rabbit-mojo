"""Affine column-vector convention; exact in-place operation reads a full point before writing.

Implementation: src/rabbit_mojo/_mojo/geometry.mojo; tests/test_geometry.py.
Inputs below are deterministic; assertions specify the expected numerical output.
"""

import numpy as np
import rabbit_mojo as rm


def main():
    points = np.arange(18.0).reshape(6, 3)[::2]
    transform = np.eye(4)
    transform[:3, 3] = [10.0, 20.0, 30.0]
    expected = points + transform[:3, 3]
    out = np.empty_like(points)
    rm.transform_points(points, transform, out=out)
    np.testing.assert_allclose(out, expected)
    print("strided points", points, "affine transform", transform, "output", out)
    rm.transform_points(points, transform, out=points)
    np.testing.assert_allclose(points, expected)
    print(
        "in-place",
        points,
        "; bottom row must be [0,0,0,1], partial aliasing is rejected",
    )


if __name__ == "__main__":
    main()
