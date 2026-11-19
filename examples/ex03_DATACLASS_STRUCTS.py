"""Slotted dataclass fields are copied into Material; Counter owns reusable native state.

Implementation: src/rabbit_mojo/_mojo/scalar.mojo; tests/test_bindings.py.
Inputs below are deterministic; assertions specify the expected numerical output.
"""

import numpy as np
import rabbit_mojo as rm


def main():
    material = rm.IsotropicElastic(200.0, 0.25)
    print(
        "Python dataclass:", material, "native lambda/mu:", rm.lame_parameters(material)
    )
    assert rm.lame_parameters(material) == (80.0, 80.0)
    converted_back = rm.material_roundtrip(material)
    assert converted_back == material and converted_back is not material
    print("Native struct fields -> new Python dataclass:", converted_back)
    assert rm.Counter.zero().value() == 0
    counter = rm.Counter(3)
    assert counter.increment() == 4 and counter.increment() == 5
    print("Reused bound Mojo Counter:", counter, "value:", counter.value())
    # Transform3D holds the Python owner; View in views.mojo retains it for a scoped borrow.
    configuration = rm.Transform3D(np.eye(4))
    points = np.array([[1.0, 2.0, 3.0]])
    np.testing.assert_array_equal(rm.transform_points(points, configuration), points)
    print("Array-bearing configuration:", configuration.matrix)


if __name__ == "__main__":
    main()
