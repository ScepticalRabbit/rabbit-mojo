"""Native multilinear interpolation borrows signed-stride grids; coordinates follow array axes.

Implementation: src/rabbit_mojo/_mojo/interpolation.mojo; tests/test_interpolation.py.
Inputs below are deterministic; assertions specify the expected numerical output.
"""

import numpy as np
import rabbit_mojo as rm


def main():
    for rank in (1, 2, 3):
        indices = np.indices((4,) * rank, dtype=float)
        field = sum((axis + 1) * indices[axis] for axis in range(rank))
        queries = np.full((2, rank), 0.5)
        queries[1] = 2.25
        values = rm.interpolate(field, queries)
        np.testing.assert_allclose(values, queries @ np.arange(1, rank + 1))
        print("rank", rank, "query coordinates", queries, "interpolated", values)
        np.testing.assert_allclose(
            rm.interpolate(field, np.full((1, rank), -1.0), policy="clamp"), 0.0
        )
    print("Outside coordinates raise by default; clamp is an explicit alternative.")


if __name__ == "__main__":
    main()
