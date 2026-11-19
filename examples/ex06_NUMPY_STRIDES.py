"""Signed element strides handle offset/reverse/Fortran/non-contiguous views without copies.

Implementation: src/rabbit_mojo/_mojo/views.mojo; tests/test_numpy_views.py.
Inputs below are deterministic; assertions specify the expected numerical output.
"""

import numpy as np
import rabbit_mojo as rm


def main():
    base = np.arange(24.0).reshape(4, 6)
    for view in (base[1:], base[:, ::2], base[::-1], np.asfortranarray(base), base.T):
        expected = view.copy() * 2
        address = view.ctypes.data
        rm.scale_inplace(view, 2)
        np.testing.assert_array_equal(view, expected)
        assert address == view.ctypes.data
        print("shape", view.shape, "byte strides", view.strides, "result", view)
    print(
        "Raw pointers never escape the synchronous call; misaligned or overlapping writes fail."
    )


if __name__ == "__main__":
    main()
