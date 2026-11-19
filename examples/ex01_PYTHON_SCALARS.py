"""Native scalar conversions copy values; float32 deliberately loses precision.

Implementation: src/rabbit_mojo/_mojo/scalar.mojo; tests/test_bindings.py.
Inputs below are deterministic; assertions specify the expected numerical output.
"""

import rabbit_mojo as rm


def main():
    result = rm.scalar_roundtrip(42, 1 + 2**-30, True, "strain ε", None)
    print("Python inputs: int=42, float=1+2^-30, bool=True, Unicode, None")
    print("Native round-trip (Int, Int32, f64, f32, bool, text, optional):", result)
    assert result[0:2] == (42, 42) and result[2] != result[3]
    assert result[4] is False and result[-1] is None
    assert rm.scalar_roundtrip(0, 0.0, False, "", 3.0)[-1] == 6.0
    try:
        rm.scalar_roundtrip(2**31, 1.0, True, "overflow")
    except OverflowError as error:
        print("Explicit fixed-width boundary:", error)
    else:
        raise AssertionError("overflow was accepted")


if __name__ == "__main__":
    main()
