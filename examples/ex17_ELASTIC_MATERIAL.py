"""IsotropicElastic is copied into a native Material; strain uses tensorial shear.

Implementation: src/rabbit_mojo/_mojo/tensor.mojo; tests/test_tensor.py.
Inputs below are deterministic; assertions specify the expected numerical output.
"""

import numpy as np
import rabbit_mojo as rm


def main():
    material = rm.IsotropicElastic(200e9, 0.3)
    strain = np.array([[0.001, 0.0002, 0.0], [0.0002, 0.0, 0.0], [0.0, 0.0, 0.0]])
    stress, energy = rm.elastic_stress(strain, material)
    lam, mu = rm.lame_parameters(material)
    expected = lam * np.trace(strain) * np.eye(3) + 2 * mu * strain
    np.testing.assert_allclose(stress, expected)
    np.testing.assert_allclose(energy, 0.5 * np.sum(stress * strain))
    print(
        "material",
        material,
        "lambda/mu",
        (lam, mu),
        "strain",
        strain,
        "stress",
        stress,
        "energy density",
        energy,
    )
    print(
        "Voigt stiffness (engineering shear ordering xx yy zz yz xz xy):",
        rm.elastic_stiffness(material),
    )


if __name__ == "__main__":
    main()
