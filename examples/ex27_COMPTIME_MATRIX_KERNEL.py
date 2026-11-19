"""multiply[N] generates statically shaped 2x2/3x3 loops; matrix entries remain runtime.

Implementation: src/rabbit_mojo/_mojo/linalg.mojo; tests/test_linalg.py.
Inputs below are deterministic; assertions specify the expected numerical output.
"""

import numpy as np
import rabbit_mojo as rm


def main():
    for n in (2, 3):
        a = np.arange(1, n * n + 1, dtype=float).reshape(n, n)
        b = np.eye(n) * 2
        result = rm.matmul(a, b)
        np.testing.assert_allclose(result, a @ b)
        print("compiled N", n, "runtime matrices", a, b, "product", result)
    print("Zig pair ex27 uses the same row-major values and diagonal multiplier 2.")


if __name__ == "__main__":
    main()
