"""NumPyView borrows C storage at ranks 1/2/3; writes are visible without replacing allocation.

Implementation: src/rabbit_mojo/_mojo/views.mojo; tests/test_numpy_views.py.
Inputs below are deterministic; assertions specify the expected numerical output.
"""

import numpy as np
import rabbit_mojo as rm


def main():
    for shape in ((8,), (2, 4), (2, 2, 2)):
        array = np.arange(8.0, dtype=np.float64).reshape(shape)
        before = array.copy()
        address = array.ctypes.data
        assert rm.scale_inplace(array, 2.0) is array
        np.testing.assert_array_equal(array, before * 2)
        assert array.ctypes.data == address
        print("shape", shape, "borrowed address", hex(address), "result", array)
    points = np.arange(6.0).reshape(2, 3)
    out = np.empty_like(points)
    assert rm.transform_points(points, np.eye(4), out=out) is out
    np.testing.assert_array_equal(out, points)
    print("Caller-provided output:", out)


if __name__ == "__main__":
    main()
