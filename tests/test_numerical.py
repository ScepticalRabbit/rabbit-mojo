import numpy as np
import pytest
import rabbit_mojo as r


def test_difference_convergence():
    for method, order in [
        ("forward", 1),
        ("backward", 1),
        ("central", 2),
        ("second", 2),
    ]:
        errors = []
        for n in (41, 81):
            x = np.linspace(0, 1, n)
            y = np.exp(x)
            actual = r.differentiate(y, x[1] - x[0], method=method)
            errors.append(np.max(abs(actual[1:-1] - y[1:-1])))
        assert 2**order * 0.85 < errors[0] / errors[1] < 2**order * 1.15
    x = np.linspace(0, 1, 8)
    y = x * x
    np.testing.assert_allclose(r.differentiate(y, x[1] - x[0]), 2 * x, atol=1e-14)
    np.testing.assert_allclose(
        r.differentiate(y, x[1] - x[0], method="second"), 2, atol=1e-12
    )
    xx, yy = np.meshgrid(x, x, indexing="ij")
    gy, gx = r.gradient2d(xx**2 + 3 * yy, (x[1] - x[0],) * 2)
    np.testing.assert_allclose(gy, 2 * xx, atol=1e-13)
    np.testing.assert_allclose(gx, 3, atol=1e-13)


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_integrate_axis(axis):
    x = np.linspace(0, 1, 9)
    y = np.outer(x, x * x)
    actual = r.integrate(y, x, axis=axis)
    np.testing.assert_allclose(actual, np.trapezoid(y, x, axis=axis))
    actual = r.integrate(y, x, axis=axis, method="simpson")
    np.testing.assert_allclose(actual, x * x / 2 if axis == 0 else x / 3)


def test_integrate_nonuniform():
    x = np.array([0.0, 0.1, 0.3, 1.0])
    y = 2 * x + 1
    assert r.integrate(y, x) == pytest.approx(2)
    with pytest.raises(ValueError):
        r.integrate(y, x, method="simpson")
    with pytest.raises(ValueError):
        r.integrate(y, x[::-1])
    with pytest.raises(ValueError):
        r.differentiate(y, 0)


def test_newton():
    result = r.newton(1.0)
    assert (
        result.converged
        and result.root == pytest.approx(np.sqrt(2))
        and result.residual < 1e-12
    )
    assert r.newton(0.5, function="cos_minus_x").root == pytest.approx(0.739085133215)
    assert r.newton(2.0, function="cube_minus_one").root == pytest.approx(1.0)
    with pytest.raises(Exception, match="derivative"):
        r.newton(0.0)
    assert not r.newton(10.0, r.NewtonOptions(max_iterations=1)).converged
    with pytest.raises(RuntimeError):
        r.newton(10.0, r.NewtonOptions(max_iterations=1), raise_on_failure=True)
    with pytest.raises(ValueError):
        r.newton(1.0, r.NewtonOptions(atol=-1))


def test_qr():
    rng = np.random.default_rng(12)
    a = rng.normal(size=(20, 4))
    b = rng.normal(size=20)
    actual = r.least_squares(a, b)
    np.testing.assert_allclose(
        actual, np.linalg.lstsq(a, b, rcond=None)[0], rtol=1e-12, atol=1e-13
    )
    np.testing.assert_allclose(a.T @ (a @ actual - b), 0, atol=1e-13)
    x = np.linspace(-1, 1, 10)
    a = np.column_stack((np.ones(10), x))
    b = 2 + 3 * x
    np.testing.assert_allclose(r.least_squares(a, b), [2, 3])
    with pytest.raises(Exception, match="rank"):
        r.least_squares(np.ones((5, 2)), np.ones(5))
