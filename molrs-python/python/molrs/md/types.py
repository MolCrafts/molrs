"""Typed MD containers.

``MDState`` / ``ForceOutput`` / ``MDObservables`` are numpy-facing named
tuples so hooks can mutate ``state.vel`` in place. Integrators accept any
object with these attributes (including the PyO3 ``MDState``).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


class ForceOutput(NamedTuple):
    """Energy + forces from a potential-like provider.

    Attributes:
        energy: Scalar total energy in amu·Å²/fs².
        forces: Per-atom forces ``(N, 3)`` ``= -∂E/∂pos``.
    """

    energy: float
    forces: NDArray[np.floating]


class MDState(NamedTuple):
    """Dynamical state advanced one step by an integrator."""

    pos: NDArray[np.floating]
    vel: NDArray[np.floating]
    forces: NDArray[np.floating]
    energy: float


class MDObservables(NamedTuple):
    """Per-observation thermodynamic snapshot handed to MD hooks."""

    pos: NDArray[np.floating]
    vel: NDArray[np.floating]
    forces: NDArray[np.floating]
    potential: float
    kinetic: float
    total: float
    temperature: float
