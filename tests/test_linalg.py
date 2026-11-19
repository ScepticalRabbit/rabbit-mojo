import numpy as np
import pytest
import rabbit_mojo as r


@pytest.mark.parametrize("n", [2, 3, 4])
def test_vectors(n):
    a = np.arange(1, n + 1, dtype=float)
    b = a[::-1]
    assert r.dot(a, b) == np.dot(a, b)
    assert r.norm(a) == np.linalg.norm(a)
    np.testing.assert_allclose(r.normalize(a), a / np.linalg.norm(a))
    np.testing.assert_allclose(r.add(a, b), a + b)
    np.testing.assert_allclose(r.scale_vector(a, 3), a * 3)
    if n == 3:
        np.testing.assert_allclose(r.cross(a, b), np.cross(a, b))


@pytest.mark.parametrize("n", [2, 3])
def test_matrices(n):
    rng = np.random.default_rng(3)
    for _ in range(10):
        a = rng.normal(size=(n, n)) + np.eye(n) * 3
        b = rng.normal(size=(n, n))
        np.testing.assert_allclose(r.matmul(a, b), a @ b, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(r.transpose(a), a.T)
        np.testing.assert_allclose(r.matvec(a, b[0]), a @ b[0])
        np.testing.assert_allclose(
            r.inverse(a), np.linalg.inv(a), rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(r.determinant(a), np.linalg.det(a), rtol=1e-12)
    assert r.determinant(np.zeros((n, n))) == 0
    with pytest.raises(Exception, match="singular"):
        r.inverse(np.zeros((n, n)))
    np.testing.assert_allclose(r.inverse(np.eye(n) * 1e-100), np.eye(n) * 1e100)


@pytest.mark.parametrize("n", [0, 1, 3, 4, 7, 1001])
def test_simd_tails(n):
    a = np.linspace(-1, 1, n)
    b = np.linspace(2, 3, n)
    np.testing.assert_allclose(
        r.dot(a, b, simd=True), np.dot(a, b), rtol=1e-13, atol=1e-12
    )
    np.testing.assert_allclose(r.dot(a, b), np.dot(a, b), rtol=1e-13, atol=1e-12)


def test_invalid():
    with pytest.raises(Exception, match="zero vector"):
        r.normalize(np.zeros(3))
    with pytest.raises(ValueError):
        r.matmul(np.eye(4), np.eye(4))
    with pytest.raises(ValueError):
        r.dot(np.arange(8.0)[::2], np.arange(4.0), simd=True)
    with pytest.raises(TypeError):
        r.dot(np.arange(3, dtype="f4"), np.arange(3.0))


def test_simd_element_alignment_and_offset():
    base = np.arange(20.0, dtype=np.float64)
    a = base[1:10]
    b = base[2:11]
    assert a.flags.c_contiguous and a.ctypes.data == base.ctypes.data + 8
    np.testing.assert_allclose(r.dot(a, b, simd=True), np.dot(a, b))
