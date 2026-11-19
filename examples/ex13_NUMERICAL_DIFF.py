"""Native uniform-grid stencils use one-sided boundaries and central interior differences.

Implementation: src/rabbit_mojo/_mojo/numerical.mojo; tests/test_numerical.py.
Inputs below are deterministic; assertions specify the expected numerical output.
"""

import numpy as np
import rabbit_mojo as rm


def main():
    errors = []
    for n in (21, 41):
        x = np.linspace(0.0, 1.0, n)
        h = x[1] - x[0]
        derivative = rm.differentiate(np.exp(x), h)
        errors.append(np.max(abs(derivative - np.exp(x))))
        print("grid points", n, "central max error", errors[-1])
    assert errors[0] / errors[1] > 3.5
    x = np.linspace(0.0, 1.0, 9)
    h = x[1] - x[0]
    for method in ("forward", "backward", "central", "second"):
        print(method, rm.differentiate(x * x, h, method=method))
    np.testing.assert_allclose(rm.differentiate(x * x, h, method="second"), 2.0)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    grad = rm.gradient2d(xx**2 + 3 * yy, (h, h))
    np.testing.assert_allclose(grad[0], 2 * xx, atol=1e-13)
    np.testing.assert_allclose(grad[1], 3, atol=1e-13)


if __name__ == "__main__":
    main()
