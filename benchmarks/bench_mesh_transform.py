"""Scalar/SIMD/parallel native transforms versus vectorised NumPy, including binding costs."""

import numpy as np
import rabbit_mojo as rm
from _common import measure, report


def run():
    data = []
    rng = np.random.default_rng(42)
    for n in (4, 100_000):
        points = rng.normal(size=(n, 3))
        t = np.eye(4)
        t[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        t[:3, 3] = [1, 2, 3]
        expected = points @ t[:3, :3].T + t[:3, 3]
        out = np.empty_like(points)
        calls = {
            "numpy_vectorised": lambda: points @ t[:3, :3].T + t[:3, 3],
            "mojo_scalar_public": lambda: rm.transform_points(points, t),
            "mojo_simd_public": lambda: rm.transform_points(points, t, simd=True),
            "mojo_pthread_1": lambda: rm.parallel_transform(points, t, workers=1),
            "mojo_parallel_2": lambda: rm.parallel_transform(points, t, workers=2),
            "mojo_caller_output": lambda: rm.transform_points(points, t, out=out),
            "mojo_binding_and_kernel_only": lambda: rm._native.transform(
                points, t, out, (True, False)
            ),
        }
        for name, function in calls.items():
            np.testing.assert_allclose(function(), expected, rtol=1e-13, atol=1e-13)
            data.append({"n": n, "dtype": "float64", "case": name, **measure(function)})
        f32 = points.astype(np.float32)
        np.testing.assert_allclose(
            rm.transform_points(f32.astype(np.float64), t),
            f32.astype(np.float64) @ t[:3, :3].T + t[:3, 3],
        )
        data.append(
            {
                "n": n,
                "case": "explicit_float32_to_float64_copy_only",
                **measure(lambda: f32.astype(np.float64)),
            }
        )
        data.append(
            {
                "n": n,
                "case": "explicit_conversion_and_public_transform",
                **measure(lambda: rm.transform_points(f32.astype(np.float64), t)),
            }
        )
    return data


if __name__ == "__main__":
    report("mesh_transform", run)
