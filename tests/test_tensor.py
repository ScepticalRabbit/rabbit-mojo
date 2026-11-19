import numpy as np
import pytest
import rabbit_mojo as r


def test_invariants():
    rng = np.random.default_rng(23)
    a = rng.normal(size=(7, 3, 3))
    a = a + a.swapaxes(-1, -2)
    result = r.tensor_invariants(a)
    trace = np.trace(a, axis1=-2, axis2=-1)
    np.testing.assert_allclose(result["I1"], trace)
    np.testing.assert_allclose(
        result["I2"], (trace**2 - np.einsum("...ij,...ji->...", a, a)) / 2
    )
    np.testing.assert_allclose(result["I3"], np.linalg.det(a))
    np.testing.assert_allclose(
        np.trace(result["deviator"], axis1=-2, axis2=-1), 0, atol=1e-14
    )
    np.testing.assert_allclose(
        result["volumetric"][..., None, None] * np.eye(3) + result["deviator"], a
    )
    np.testing.assert_allclose(result["norm"], np.linalg.norm(a, axis=(-2, -1)))
    np.testing.assert_allclose(
        result["J2"], np.sum(result["deviator"] ** 2, axis=(-2, -1)) / 2
    )


@pytest.mark.parametrize("n", [2, 3])
@pytest.mark.parametrize("scale", [1.0, 1e-15, 1e15])
def test_principal(n, scale):
    rng = np.random.default_rng(29)
    a = rng.normal(size=(10, n, n))
    a = (a + a.swapaxes(-1, -2)) * scale
    repeated = np.eye(n) * scale
    a = np.concatenate(
        (a, repeated[None], (repeated + np.ones((n, n)) * 1e-14 * scale)[None])
    )
    values, directions = r.principal_strains(a, directions=True)
    np.testing.assert_allclose(
        values, np.linalg.eigvalsh(a)[:, ::-1], rtol=1e-12, atol=1e-13 * scale
    )
    np.testing.assert_allclose(
        a @ directions, directions * values[:, None, :], rtol=1e-11, atol=1e-13 * scale
    )
    np.testing.assert_allclose(
        directions.swapaxes(-1, -2) @ directions,
        np.broadcast_to(np.eye(n), a.shape),
        atol=1e-13,
    )
    np.testing.assert_allclose(
        r.maximum_engineering_shear(a), values[:, 0] - values[:, -1]
    )


def test_elastic():
    mat = r.IsotropicElastic(200.0, 0.25)
    strain = np.array([[0.01, 0.02, 0], [0.02, -0.01, 0], [0, 0, 0.03]])
    sigma, energy = r.elastic_stress(strain, mat)
    lam, mu = r.lame_parameters(mat)
    expected = lam * np.trace(strain) * np.eye(3) + 2 * mu * strain
    np.testing.assert_allclose(sigma, expected)
    assert energy == pytest.approx(np.sum(expected * strain) / 2)
    c = r.elastic_stiffness(mat)
    voigt = np.array([0.01, -0.01, 0.03, 0, 0, 0.04])
    np.testing.assert_allclose(
        c @ voigt, [sigma[0, 0], sigma[1, 1], sigma[2, 2], 0, 0, sigma[0, 1]]
    )
    with pytest.raises(ValueError):
        r.principal_strains(np.ones((3, 3)) + np.triu(np.ones((3, 3))))
