"""Explicit SIMD[float64,4] loads/reduction use scalar tails and element-aligned storage.

Implementation: src/rabbit_mojo/_mojo/linalg.mojo; tests/test_linalg.py.
Inputs below are deterministic; assertions specify the expected numerical output.
"""

import numpy as np
import rabbit_mojo as rm


def main():
    for count in (0, 1, 3, 4, 7, 17):
        a = np.linspace(-1.0, 1.0, count)
        b = np.linspace(2.0, 3.0, count)
        scalar = rm.dot(a, b)
        simd = rm.dot(a, b, simd=True)
        np.testing.assert_allclose([scalar, simd], np.dot(a, b), atol=1e-13)
        print(
            "length", count, "scalar", scalar, "SIMD", simd, "tail elements", count % 4
        )
    print(
        "Width is compiled as 4. SIMD does not imply a measured speedup; see benchmarks/."
    )


if __name__ == "__main__":
    main()
