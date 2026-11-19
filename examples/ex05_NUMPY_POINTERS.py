"""Typed pointers borrow synchronous storage; offset slices retain their base allocation.

Implementation: src/rabbit_mojo/_mojo/views.mojo; tests/test_numpy_views.py.
Inputs below are deterministic; assertions specify the expected numerical output.
"""

import numpy as np
import rabbit_mojo as rm


def main():
    for dtype in (np.float32, np.float64, np.int64):
        base = np.arange(7, dtype=dtype)
        view = base[2:]
        assert view.ctypes.data == base.ctypes.data + 2 * base.itemsize
        rm.scale_inplace(view, 3)
        print(
            "dtype",
            dtype,
            "offset address",
            hex(view.ctypes.data),
            "shape",
            view.shape,
            "base",
            base,
        )
        np.testing.assert_array_equal(base, [0, 1, 6, 9, 12, 15, 18])
    readonly = np.arange(3.0)
    readonly.flags.writeable = False
    try:
        rm.scale_inplace(readonly)
    except ValueError as error:
        print("Read-only write rejected:", error)
    else:
        raise AssertionError("read-only accepted")


if __name__ == "__main__":
    main()
