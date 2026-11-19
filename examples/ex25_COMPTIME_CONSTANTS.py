"""Compile-time polynomial coefficients/degree select native instantiations; x remains runtime data.

Implementation: src/rabbit_mojo/_mojo/comptime.mojo; tests/test_comptime.py.
Inputs below are deterministic; assertions specify the expected numerical output.
"""

import rabbit_mojo as rm


def main():
    for degree in (2, 3):
        result = rm.polynomial(2.0, degree=degree)
        expected = sum((k + 1) * 2.0**k for k in range(degree + 1))
        assert result == expected
        print(
            "runtime x=2; compiled degree",
            degree,
            "coefficients [1,2,3,4]; result",
            result,
        )
    print("Zig pair: examples/zig/ex25_COMPTIME_CONSTANTS.zig; results 17 and 49.")


if __name__ == "__main__":
    main()
