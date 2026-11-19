"""Borrowed symmetric tensors produce I1/I2/I3, J2, volumetric and deviatoric parts.

Implementation: src/rabbit_mojo/_mojo/tensor.mojo; tests/test_tensor.py.
Inputs below are deterministic; assertions specify the expected numerical output.
"""

import numpy as np
import rabbit_mojo as rm


def main():
    a = np.array([[[2.0, 1.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 4.0]], np.eye(3)])
    result = rm.tensor_invariants(a)
    print("tensor batch", a, "invariants", result)
    np.testing.assert_allclose(result["I1"], np.trace(a, axis1=-2, axis2=-1))
    np.testing.assert_allclose(result["I3"], np.linalg.det(a))
    np.testing.assert_allclose(
        np.trace(result["deviator"], axis1=-2, axis2=-1), 0, atol=1e-14
    )
    np.testing.assert_allclose(
        result["deviator"] + result["volumetric"][:, None, None] * np.eye(3), a
    )
    print("J2 = 0.5 dev:dev; norm = sqrt(A:A); strain shear is tensorial")


if __name__ == "__main__":
    main()
