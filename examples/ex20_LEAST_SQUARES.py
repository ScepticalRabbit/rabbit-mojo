"""Native Householder QR avoids normal equations; rank deficiency fails explicitly.

Implementation: src/rabbit_mojo/_mojo/numerical.mojo; tests/test_numerical.py.
Inputs below are deterministic; assertions specify the expected numerical output.
"""

import numpy as np
import rabbit_mojo as rm


def main():
    x = np.linspace(-1.0, 1.0, 9)
    a = np.column_stack((np.ones_like(x), x))
    b = 2 + 3 * x + 0.01 * x * x
    fit = rm.least_squares(a, b)
    reference = np.linalg.lstsq(a, b, rcond=None)[0]
    np.testing.assert_allclose(fit, reference)
    print(
        "design",
        a,
        "response",
        b,
        "coefficients",
        fit,
        "residual norm",
        np.linalg.norm(a @ fit - b),
    )
    print(
        "condition number",
        np.linalg.cond(a),
        "; QR is unpivoted, limited to well-conditioned small problems",
    )
    try:
        rm.least_squares(np.ones((5, 2)), np.ones(5))
    except Exception as error:
        print("Rank-deficient design rejected:", error)
    else:
        raise AssertionError("rank deficiency accepted")


if __name__ == "__main__":
    main()
