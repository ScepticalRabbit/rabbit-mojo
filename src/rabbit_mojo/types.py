"""Configuration is copied fieldwise; Python and Mojo structs do not share an ABI."""

from dataclasses import dataclass
from enum import IntEnum
import math
import numpy as np


class Mode(IntEnum):
    ADD = 0
    SCALE = 1


@dataclass(slots=True)
class IsotropicElastic:
    youngs_modulus: float
    poissons_ratio: float

    def __post_init__(self):
        if not math.isfinite(self.youngs_modulus) or self.youngs_modulus <= 0:
            raise ValueError("Young's modulus must be finite and positive")
        if not math.isfinite(self.poissons_ratio) or not -1 < self.poissons_ratio < 0.5:
            raise ValueError("Poisson ratio must satisfy -1 < nu < 0.5")


@dataclass(slots=True)
class NewtonOptions:
    atol: float = 1e-12
    rtol: float = 1e-12
    max_iterations: int = 50


@dataclass(slots=True, frozen=True)
class NewtonResult:
    root: float
    converged: bool
    iterations: int
    residual: float


@dataclass(slots=True)
class Transform3D:
    matrix: np.ndarray
