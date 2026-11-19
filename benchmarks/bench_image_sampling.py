"""Native image sampling versus a vectorised NumPy bilinear implementation."""

import numpy as np
import rabbit_mojo as rm
from _common import measure, report


def numpy_sample(image, xy):
    x, y = xy.T
    ix = np.floor(x).astype(int)
    iy = np.floor(y).astype(int)
    dx = x - ix
    dy = y - iy
    return (
        (1 - dx) * (1 - dy) * image[iy, ix]
        + dx * (1 - dy) * image[iy, ix + 1]
        + (1 - dx) * dy * image[iy + 1, ix]
        + dx * dy * image[iy + 1, ix + 1]
    )


def run():
    data = []
    rng = np.random.default_rng(7)
    image = rng.normal(size=(256, 256))
    for n in (4, 50_000):
        xy = rng.uniform(0, 254.99, size=(n, 2))
        expected = numpy_sample(image, xy)
        np.testing.assert_allclose(
            rm.sample_image(image, xy), expected, rtol=1e-11, atol=1e-13
        )
        for name, function in {
            "numpy_vectorised_bilinear": lambda: numpy_sample(image, xy),
            "mojo_bilinear": lambda: rm.sample_image(image, xy),
        }.items():
            data.append(
                {
                    "n": n,
                    "image_shape": list(image.shape),
                    "dtype": "float64",
                    "case": name,
                    **measure(function),
                }
            )
    return data


if __name__ == "__main__":
    report("image_sampling", run)
