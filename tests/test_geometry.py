import numpy as np
import pytest
import rabbit_mojo as r


@pytest.mark.parametrize("simd", [False, True])
def test_transform(simd):
    points = np.arange(30.0).reshape(10, 3)[::-2]
    transform = np.eye(4)
    transform[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    transform[:3, 3] = [2, 3, 4]
    expected = points @ transform[:3, :3].T + transform[:3, 3]
    out = np.empty(points.shape)
    assert r.transform_points(points, transform, out, simd=simd) is out
    np.testing.assert_allclose(out, expected)
    np.testing.assert_allclose(
        r.transform_vectors(points, transform, simd=simd), expected - transform[:3, 3]
    )
    np.testing.assert_allclose(r.transform_points(points, transform, points), expected)
    assert r.transform_points(np.empty((0, 3)), transform).shape == (0, 3)


def test_alias_and_affine():
    points = np.arange(18.0).reshape(6, 3)
    with pytest.raises(ValueError):
        r.transform_points(points[:3], np.eye(4), points[1:4])
    t = np.eye(4)
    t[3, 0] = 1
    with pytest.raises(ValueError):
        r.transform_points(points, t)
    out = np.empty_like(points)
    out.flags.writeable = False
    with pytest.raises(ValueError):
        r.transform_points(points, np.eye(4), out)


def test_rotation():
    rotation = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    a = np.arange(18.0).reshape(2, 3, 3)
    a = a + a.swapaxes(-1, -2)
    rotated = r.rotate_tensors(a, rotation)
    np.testing.assert_allclose(rotated, rotation @ a @ rotation.T)
    np.testing.assert_allclose(r.rotate_tensors(rotated, rotation.T), a)
    for key in ("I1", "I2", "I3", "norm", "J2"):
        np.testing.assert_allclose(
            r.tensor_invariants(rotated)[key], r.tensor_invariants(a)[key]
        )
    np.testing.assert_allclose(
        r.rotate_vectors(np.ones((2, 3)), rotation), np.ones((2, 3)) @ rotation.T
    )


def test_parallel():
    points = np.arange(90.0).reshape(30, 3)
    t = np.eye(4)
    t[:3, 3] = [1, 2, 3]
    np.testing.assert_array_equal(
        r.parallel_transform(points, t), r.transform_points(points, t)
    )
    np.testing.assert_array_equal(
        r.parallel_transform(points, t), r.parallel_transform(points, t)
    )


@pytest.mark.parametrize("workers", [1, 2, 3, 8])
def test_parallel_strides_and_lifetimes(workers):
    rng = np.random.default_rng(55)
    for _ in range(10):
        points = np.asfortranarray(rng.normal(size=(31, 3)))[::-2]
        t = np.eye(4)
        t[:3, :3] = rng.normal(size=(3, 3))
        t[:3, 3] = rng.normal(size=3)
        actual = r.parallel_transform(points, t, workers=workers)
        np.testing.assert_allclose(
            actual, points @ t[:3, :3].T + t[:3, 3], rtol=1e-12, atol=1e-12
        )
    assert r.parallel_transform(np.empty((0, 3)), np.eye(4), workers=workers).shape == (
        0,
        3,
    )


def test_readonly_inputs_and_strided_output():
    a = np.arange(9.0).reshape(3, 3)
    a.flags.writeable = False
    t = np.eye(4)
    t.flags.writeable = False
    storage = np.zeros((6, 6))
    out = storage[::2, ::2]
    assert r.transform_points(a, t, out=out) is out
    np.testing.assert_array_equal(out, a)
    np.testing.assert_array_equal(r.parallel_transform(a, t), a)
