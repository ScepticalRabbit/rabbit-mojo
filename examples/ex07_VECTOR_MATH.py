"""Native fixed-size SIMD values implement small vectors; float64 inputs are borrowed.

Implementation: src/rabbit_mojo/_mojo/linalg.mojo; tests/test_linalg.py.
Inputs below are deterministic; assertions specify the expected numerical output.
"""

import numpy as np
import rabbit_mojo as rm


def main():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([3.0, 1.0, 2.0])
    print("a", a, "b", b, "dot", rm.dot(a, b), "cross", rm.cross(a, b))
    assert rm.dot(a, b) == np.dot(a, b)
    np.testing.assert_allclose(rm.cross(a, b), np.cross(a, b))
    np.testing.assert_allclose(rm.normalize(a), a / np.linalg.norm(a))
    np.testing.assert_allclose(rm.add(a, b), a + b)
    np.testing.assert_allclose(rm.scale_vector(a, 2.0), a * 2)
    print("norm", rm.norm(a), "normalised", rm.normalize(a))


if __name__ == "__main__":
    main()
