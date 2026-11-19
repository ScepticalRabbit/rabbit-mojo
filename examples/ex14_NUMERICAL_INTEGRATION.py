"""Native reductions integrate an axis; Simpson requires uniform spacing and an odd sample count.

Implementation: src/rabbit_mojo/_mojo/numerical.mojo; tests/test_numerical.py.
Inputs below are deterministic; assertions specify the expected numerical output.
"""

import numpy as np
import rabbit_mojo as rm


def main():
    x = np.linspace(0.0, 1.0, 9)
    result = rm.integrate(x**3, x, method="simpson")
    np.testing.assert_allclose(result, 0.25)
    nonuniform = np.array([0.0, 0.1, 0.4, 1.0])
    np.testing.assert_allclose(rm.integrate(2 * nonuniform + 1, nonuniform), 2.0)
    y = np.outer(x, x * x)
    np.testing.assert_allclose(rm.integrate(y, x, axis=1, method="simpson"), x / 3)
    np.testing.assert_allclose(rm.integrate(y, x, axis=0), np.trapezoid(y, x, axis=0))
    print(
        "x",
        x,
        "Simpson integral x^3",
        result,
        "axis integral",
        rm.integrate(y, x, axis=1),
    )


if __name__ == "__main__":
    main()
