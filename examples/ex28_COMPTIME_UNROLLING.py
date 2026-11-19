"""Compile-time Horner loop extent is unrolled; runtime-loop reference uses identical coefficients.

Implementation: src/rabbit_mojo/_mojo/comptime.mojo; tests/test_comptime.py.
Inputs below are deterministic; assertions specify the expected numerical output.
"""

import numpy as np
import rabbit_mojo as rm


def main():
    for x in (-1.0, 0.5, 2.0):
        unrolled = rm.polynomial(x, degree=3, unroll=True)
        loop = rm.polynomial(x, degree=3, unroll=False)
        np.testing.assert_allclose(unrolled, loop, atol=1e-14)
        np.testing.assert_allclose(unrolled, 1 + 2 * x + 3 * x * x + 4 * x * x * x)
        print("runtime x", x, "unrolled cubic", unrolled, "runtime-loop cubic", loop)
    print("Zig pair ex28: results -2, 3.25, 49; unrolling alone promises no speedup.")


if __name__ == "__main__":
    main()
