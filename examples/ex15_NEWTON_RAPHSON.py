"""Dataclass options configure a native selected function and structured convergence result.

Implementation: src/rabbit_mojo/_mojo/numerical.mojo; tests/test_numerical.py.
Inputs below are deterministic; assertions specify the expected numerical output.
"""

import numpy as np
import rabbit_mojo as rm


def main():
    options = rm.NewtonOptions(atol=1e-12, max_iterations=30)
    result = rm.newton(1.0, options)
    assert result.converged and abs(result.root - np.sqrt(2)) < 1e-12
    print("sqrt(2) from x^2-2, options", options, "result", result)
    print("cos(x)-x root", rm.newton(0.5, function="cos_minus_x"))
    failed = rm.newton(10.0, rm.NewtonOptions(max_iterations=1))
    assert not failed.converged
    print("Nonconvergence is explicit:", failed)
    try:
        rm.newton(0.0)
    except Exception as error:
        print("Native zero derivative error:", error)
    else:
        raise AssertionError("zero derivative accepted")


if __name__ == "__main__":
    main()
