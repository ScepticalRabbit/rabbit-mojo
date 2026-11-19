"""Compile-time dimensions select native matrix products; inversion uses partial pivoting.

Implementation: src/rabbit_mojo/_mojo/linalg.mojo; tests/test_linalg.py.
Inputs below are deterministic; assertions specify the expected numerical output.
"""

import numpy as np
import rabbit_mojo as rm


def main():
    for n in (2, 3):
        a = np.arange(n * n, dtype=float).reshape(n, n) + 4 * np.eye(n)
        b = np.eye(n) * 2
        np.testing.assert_allclose(rm.matmul(a, b), a @ b)
        np.testing.assert_allclose(rm.matvec(a, b[0]), a @ b[0])
        np.testing.assert_allclose(rm.transpose(a), a.T)
        np.testing.assert_allclose(rm.inverse(a), np.linalg.inv(a))
        np.testing.assert_allclose(rm.determinant(a), np.linalg.det(a))
        print("matrix", a, "determinant", rm.determinant(a), "inverse", rm.inverse(a))
    try:
        rm.inverse(np.zeros((2, 2)))
    except Exception as error:
        print("Native singularity error:", error)
    else:
        raise AssertionError("singular matrix accepted")


if __name__ == "__main__":
    main()
