"""Native Jacobi rotations return descending eigenvalues and optional column directions.

Implementation: src/rabbit_mojo/_mojo/tensor.mojo; tests/test_tensor.py.
Inputs below are deterministic; assertions specify the expected numerical output.
"""

import numpy as np
import rabbit_mojo as rm


def main():
    for n in (2, 3):
        a = np.eye(n) * 2 + np.ones((n, n)) * 0.1
        values, axes = rm.principal_strains(a, directions=True)
        np.testing.assert_allclose(values, np.linalg.eigvalsh(a)[::-1])
        np.testing.assert_allclose(a @ axes, axes * values, atol=1e-13)
        assert rm.maximum_engineering_shear(a) == values[0] - values[-1]
        print(
            "strain",
            a,
            "principal strains",
            values,
            "directions",
            axes,
            "max engineering shear",
            rm.maximum_engineering_shear(a),
        )


if __name__ == "__main__":
    main()
