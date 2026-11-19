"""PythonModuleBuilder registers functions/types; wrappers validate keywords, native Error becomes Exception.

Implementation: src/rabbit_mojo/_mojo/bindings.mojo; tests/test_bindings.py.
Inputs below are deterministic; assertions specify the expected numerical output.
"""

import numpy as np
import rabbit_mojo as rm


def main():
    result = rm.scale_inplace(a=np.arange(3.0), factor=3.0)
    np.testing.assert_array_equal(result, [0, 3, 6])
    print("Public keyword call:", result)
    assert rm.native_affine(3.0, factor=2.0, offset=1.0) == 7.0
    print("Native **kwargs call:", rm.native_affine(3.0, factor=2.0, offset=1.0))
    assert rm.Counter.zero().value() == 0
    c = rm.Counter(7)
    assert c.increment() == 8
    print("Native constructor/method:", c.value())
    for operation in (
        lambda: rm.native_error("deliberate native error"),
        lambda: rm.transform_points(np.zeros((2, 2)), np.eye(4)),
        lambda: rm._native.scale(array=np.arange(3.0), factor=2.0),
    ):
        try:
            operation()
        except Exception as error:
            print(type(error).__name__ + ":", error)
        else:
            raise AssertionError("invalid call succeeded")
    print(
        "scale is positional-only; native_affine explicitly binds **kwargs; public wrappers accept keywords."
    )


if __name__ == "__main__":
    main()
