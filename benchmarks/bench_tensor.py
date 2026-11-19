"""Batched invariants and principal strains versus vectorised NumPy equivalents."""

import numpy as np
import rabbit_mojo as rm
from _common import measure, report


def numpy_invariants(a):
    trace = np.trace(a, axis1=-2, axis2=-1)
    dev = a - trace[:, None, None] * np.eye(3) / 3
    return {
        "I1": trace,
        "I2": (trace**2 - np.einsum("bij,bji->b", a, a)) / 2,
        "I3": np.linalg.det(a),
        "volumetric": trace / 3,
        "norm": np.linalg.norm(a, axis=(1, 2)),
        "J2": np.sum(dev * dev, axis=(1, 2)) / 2,
        "deviator": dev,
    }


def run():
    data = []
    rng = np.random.default_rng(4)
    for n in (4, 10_000):
        a = rng.normal(size=(n, 3, 3))
        a = a + a.swapaxes(-1, -2)
        expected = numpy_invariants(a)
        actual = rm.tensor_invariants(a)
        for key in expected:
            np.testing.assert_allclose(
                actual[key], expected[key], rtol=1e-10, atol=1e-12
            )
        np.testing.assert_allclose(
            rm.principal_strains(a),
            np.linalg.eigvalsh(a)[:, ::-1],
            rtol=1e-10,
            atol=1e-12,
        )
        calls = {
            "numpy_all_invariants": lambda: numpy_invariants(a),
            "mojo_all_invariants": lambda: rm.tensor_invariants(a),
            "numpy_eigvalsh": lambda: np.linalg.eigvalsh(a)[:, ::-1],
            "mojo_jacobi_principal": lambda: rm.principal_strains(a),
        }
        for name, function in calls.items():
            data.append({"n": n, "dtype": "float64", "case": name, **measure(function)})
    return data


if __name__ == "__main__":
    report("tensor", run)
