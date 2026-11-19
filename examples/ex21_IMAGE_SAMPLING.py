"""Grey images borrow storage; xy coordinates use integer pixel centres and nearest ties round up.

Implementation: src/rabbit_mojo/_mojo/interpolation.mojo; tests/test_interpolation.py.
Inputs below are deterministic; assertions specify the expected numerical output.
"""

import numpy as np
import rabbit_mojo as rm


def main():
    y, x = np.indices((4, 5), dtype=float)
    image = 2 * x + 3 * y
    xy = np.array([[0.25, 0.5], [1.75, 2.25], [4.0, 3.0]])
    values = rm.sample_image(image, xy)
    np.testing.assert_allclose(values, 2 * xy[:, 0] + 3 * xy[:, 1])
    nearest = rm.sample_image(image, xy, method="nearest")
    np.testing.assert_allclose(
        nearest,
        image[
            np.floor(xy[:, 1] + 0.5).astype(int), np.floor(xy[:, 0] + 0.5).astype(int)
        ],
    )
    np.testing.assert_allclose(
        rm.sample_image(image[::-1], xy), rm.sample_image(image[::-1].copy(), xy)
    )
    print("image", image, "xy", xy, "bilinear", values, "nearest", nearest)
    print(
        "clamped outside sample",
        rm.sample_image(image, np.array([[-1.0, -1.0]]), policy="clamp"),
    )


if __name__ == "__main__":
    main()
