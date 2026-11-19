"""List/dict values are copied to native collections; an object wrapper preserves identity.

Implementation: src/rabbit_mojo/_mojo/scalar.mojo; tests/test_bindings.py.
Inputs below are deterministic; assertions specify the expected numerical output.
"""

import rabbit_mojo as rm


def main():
    settings = {"scale": 2.0}
    result = rm.collection_roundtrip(
        [1.0, 2.0, 3.0], (4.0, 5.0), settings, rm.Mode.SCALE
    )
    print("list [1,2,3], coordinate (4,5), settings", settings, "enum", rm.Mode.SCALE)
    print(
        "native reduction, native tuple sum, enum integer, retained Python dict:",
        result,
    )
    assert result[:3] == (12.0, 9.0, 1) and result[3] is settings
    try:
        rm.collection_roundtrip(["bad"], (0.0, 1.0), settings)
    except TypeError as error:
        print("Rejected invalid typed list:", error)
    else:
        raise AssertionError("invalid list accepted")


if __name__ == "__main__":
    main()
