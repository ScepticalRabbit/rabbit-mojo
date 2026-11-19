"""R A R-transpose is native; rotation matrices use column-vector mathematical convention.

Implementation: src/rabbit_mojo/_mojo/geometry.mojo; tests/test_geometry.py.
Inputs below are deterministic; assertions specify the expected numerical output.
"""

import numpy as np
import rabbit_mojo as rm


def main():
    r = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    a = np.array([[[2.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 3.0]]])
    rotated = rm.rotate_tensors(a, r)
    np.testing.assert_allclose(rotated, r @ a @ r.T)
    np.testing.assert_allclose(rm.rotate_tensors(rotated, r.T), a)
    for key in ("I1", "I2", "I3"):
        np.testing.assert_allclose(
            rm.tensor_invariants(a)[key], rm.tensor_invariants(rotated)[key]
        )
    print("rotation", r, "tensors", a, "rotated", rotated)
    print("rotated vectors", rm.rotate_vectors(np.array([[1.0, 0.0, 0.0]]), r))


if __name__ == "__main__":
    main()
