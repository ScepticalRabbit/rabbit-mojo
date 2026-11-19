"""Vec[dtype,N] owns a fixed-size native Array; six type/length instantiations copy tiny inputs.

Implementation: src/rabbit_mojo/_mojo/comptime.mojo; tests/test_comptime.py.
Inputs below are deterministic; assertions specify the expected numerical output.
"""

import numpy as np
import rabbit_mojo as rm


def main():
    for dtype in (np.float32, np.float64):
        for n in (2, 3, 4):
            a = np.arange(1, n + 1, dtype=dtype)
            b = np.arange(n, 0, -1, dtype=dtype)
            result = rm.generic_dot(a, b)
            np.testing.assert_allclose(result, np.dot(a, b), rtol=1e-6)
            print(
                "compiled dtype", dtype, "N", n, "runtime vectors", a, b, "dot", result
            )
    print("Zig pair ex26: corresponding results 4, 10, 20 in both precisions.")


if __name__ == "__main__":
    main()
