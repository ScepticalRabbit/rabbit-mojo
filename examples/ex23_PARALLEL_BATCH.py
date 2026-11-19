"""Native pthread workers own disjoint row ranges; caller retains owners and joins every worker.

Implementation: src/rabbit_mojo/_mojo/parallel.mojo; tests/test_geometry.py.
Inputs below are deterministic; assertions specify the expected numerical output.
"""

import numpy as np
import rabbit_mojo as rm


def main():
    points = np.arange(300.0).reshape(100, 3)[::-1]
    t = np.eye(4)
    t[:3, 3] = [1.0, 2.0, 3.0]
    sequential = rm.transform_points(points, t)
    parallel = rm.parallel_transform(points, t, workers=2)
    np.testing.assert_array_equal(parallel, sequential)
    np.testing.assert_array_equal(parallel, rm.parallel_transform(points, t, workers=3))
    print(
        "points first/last",
        points[[0, -1]],
        "parallel output first/last",
        parallel[[0, -1]],
    )
    print(
        "100 rows, 2/3 native POSIX threads, exact deterministic agreement; startup is included in benchmarks."
    )


if __name__ == "__main__":
    main()
