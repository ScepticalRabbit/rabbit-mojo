import numpy as np
import pytest
import rabbit_mojo as r


@pytest.mark.parametrize("degree", [2, 3])
@pytest.mark.parametrize("x", [-2.0, 0.0, 0.5, 2.0])
def test_poly(degree, x):
    expected = sum((i + 1) * x**i for i in range(degree + 1))
    assert r.polynomial(x, degree=degree) == pytest.approx(expected)
    assert r.polynomial(x, degree=degree, unroll=False) == pytest.approx(expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n", [2, 3, 4])
def test_generic(dtype, n):
    a = np.arange(1, n + 1, dtype=dtype)
    b = np.arange(n, 0, -1, dtype=dtype)
    assert r.generic_dot(a, b) == pytest.approx(
        np.dot(a, b), rel=1e-6 if dtype == np.float32 else 1e-14
    )
