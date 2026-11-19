import numpy as np
import pytest
import rabbit_mojo as r


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_linear_grid(rank):
    shape = (4,) * rank
    indices = np.indices(shape, dtype=float)
    field = sum((i + 1) * indices[i] for i in range(rank))
    q = np.array([[0.25] * rank, [1.75] * rank, [3.0] * rank])
    np.testing.assert_allclose(r.interpolate(field, q), q @ np.arange(1, rank + 1))
    # Same physical function through negative strides and a nonzero origin/spacing.
    np.testing.assert_allclose(
        r.interpolate(
            field, q, origin=(1.0,) * rank, spacing=(2.0,) * rank, policy="clamp"
        ),
        np.maximum((q - 1) / 2, 0) @ np.arange(1, rank + 1),
    )
    qbad = np.full((1, rank), -1.0)
    with pytest.raises(Exception, match="outside"):
        r.interpolate(field, qbad)
    np.testing.assert_allclose(r.interpolate(field, qbad, policy="clamp"), 0)


def test_image():
    y, x = np.indices((4, 5), dtype=float)
    image = 2 * x + 3 * y
    xy = np.array([[0.25, 0.5], [1.75, 2.25], [4.0, 3.0]])
    np.testing.assert_allclose(r.sample_image(image, xy), 2 * xy[:, 0] + 3 * xy[:, 1])
    np.testing.assert_allclose(
        r.sample_image(image, xy, method="nearest"),
        image[
            np.floor(xy[:, 1] + 0.5).astype(int), np.floor(xy[:, 0] + 0.5).astype(int)
        ],
    )
    np.testing.assert_allclose(
        r.sample_image(image[::-1, ::-1], xy),
        r.sample_image(image[::-1, ::-1].copy(), xy),
    )
    assert r.sample_image(image, np.empty((0, 2))).size == 0
    with pytest.raises(ValueError):
        r.sample_image(image, np.array([[np.nan, 0.0]]))


def test_q4():
    vertices = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 3.0], [0.0, 3.0]])
    n, g = r.q4_shape(0.0, 0.0)
    np.testing.assert_allclose(n, 0.25)
    np.testing.assert_allclose(g.sum(axis=0), 0)
    for px, py in [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2)]:
        actual, details = r.q4_integrate(vertices, powers=(px, py))
        assert actual == pytest.approx(
            2 ** (px + 1) / (px + 1) * 3 ** (py + 1) / (py + 1)
        )
        np.testing.assert_allclose(details["determinants"], 1.5)
        np.testing.assert_allclose(
            np.linalg.det(details["jacobians"]), details["determinants"]
        )
    with pytest.raises(ValueError):
        r.q4_integrate(vertices[::-1])
    with pytest.raises(ValueError):
        r.q4_integrate(np.zeros((4, 2)))


def test_interpolation_nonunit_strides():
    base = np.arange(80.0).reshape(8, 10)
    field = base[1::2, ::2]
    coordinates = np.array([[0.4, 1.2], [1.8, 2.7]])
    np.testing.assert_allclose(
        r.interpolate(field, coordinates), r.interpolate(field.copy(), coordinates)
    )
    with pytest.raises(ValueError):
        r.interpolate(np.ones((1, 3)), coordinates)
    with pytest.raises(ValueError):
        r.interpolate(np.ones((3, 3)), coordinates, spacing=(0, 1))
