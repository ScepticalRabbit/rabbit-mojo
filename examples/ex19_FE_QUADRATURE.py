"""Q4 native shape functions and 2x2 Gauss integration use counter-clockwise vertices.

Implementation: src/rabbit_mojo/_mojo/interpolation.mojo; tests/test_interpolation.py.
Inputs below are deterministic; assertions specify the expected numerical output.
"""

import numpy as np
import rabbit_mojo as rm


def main():
    vertices = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 3.0], [0.0, 3.0]])
    n, derivatives = rm.q4_shape(0.0, 0.0)
    np.testing.assert_allclose(n, 0.25)
    np.testing.assert_allclose(derivatives.sum(axis=0), 0.0)
    area, details = rm.q4_integrate(vertices)
    xy, _ = rm.q4_integrate(vertices, powers=(1, 1))
    np.testing.assert_allclose([area, xy], [6.0, 9.0])
    print("vertices", vertices, "centre N", n, "dN", derivatives)
    print(
        "area",
        area,
        "integral x*y",
        xy,
        "Gauss coordinates/Jacobians/determinants",
        details,
    )


if __name__ == "__main__":
    main()
