"""In-process MD: integrators, pair potentials, runner.

LJ (Rust)::

    nl = VerletSkin(NeighborList(rc + skin), rc, pos, box, skin=skin)
    lj = LJ(eps, sigma, rc)
    vv = VelocityVerlet(dt, potential=lj, neighbors=nl, mass=mass)

ForceField + Frame (Python)::

    MD().set_forcefield(ff).run(frame, n_steps, dt=dt)

Pre-compiled ``Potentials`` (optional)::

    MD().set_potential(ff.to_potentials(frame), energy_scale=ff.units).run(frame, n_steps, dt=dt)

``LJ.calc_energy`` / ``calc_force`` / ``eval → (energy, forces)``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .._lib import NeighborList, VerletSkin
from .._lib import md as _md

from .driver import MD, FrameVelocityVerlet
from .runner import CheckpointHook, MDHook, MDRunner, VelocityInitHook
from .types import ForceOutput, MDObservables, MDState

MaxwellBoltzmann = _md.MaxwellBoltzmann
kb_md = _md.kb_md
energy_to_md = _md.energy_to_md
preset_energy_to_md = _md.preset_energy_to_md
MD_ENERGY = _md.MD_ENERGY

PRECISIONS = ("double", "mixed", "single")


def resolve_prec(prec: str) -> tuple[np.dtype, np.dtype]:
    key = str(prec).strip().lower()
    if key == "double":
        return np.dtype(np.float64), np.dtype(np.float64)
    if key == "mixed":
        return np.dtype(np.float32), np.dtype(np.float64)
    if key == "single":
        return np.dtype(np.float32), np.dtype(np.float32)
    raise ValueError(f"prec must be one of {PRECISIONS}, got {prec!r}")


def _as_state(rs) -> MDState:
    return MDState(
        np.asarray(rs.pos, dtype=np.float64),
        np.asarray(rs.vel, dtype=np.float64),
        np.asarray(rs.forces, dtype=np.float64),
        float(rs.energy),
    )


def _reject_non_double(*, prec, dtype) -> None:
    if prec not in (None, "double") or (
        dtype is not None and np.dtype(dtype) != np.dtype(np.float64)
    ):
        raise ValueError(
            "molrs.md is float64 only; mixed/single belong in molrs, not a numpy twin"
        )


class VelocityVerlet(_md.VelocityVerlet):
    """NVE velocity-Verlet: ``VelocityVerlet(dt, potential=…, neighbors=…, mass=…)``."""

    def __new__(
        cls,
        dt: float,
        *,
        potential: LJ,
        neighbors: VerletSkin,
        mass: float | NDArray[np.floating],
        prec: str | None = None,
        dtype: np.dtype | type | None = None,
    ):
        _reject_non_double(prec=prec, dtype=dtype)
        return super().__new__(
            cls,
            float(dt),
            potential=potential,
            neighbors=neighbors,
            mass=np.atleast_1d(np.asarray(mass, dtype=np.float64)),
        )

    def __init__(
        self,
        dt: float,
        *,
        potential: LJ,
        neighbors: VerletSkin,
        mass: float | NDArray[np.floating],
        prec: str | None = None,
        dtype: np.dtype | type | None = None,
    ) -> None:
        self.prec = "double"
        self.compute_dtype = np.dtype(np.float64)
        self.acc_dtype = np.dtype(np.float64)
        self.dtype = self.compute_dtype
        self.mass = np.atleast_1d(np.asarray(mass, dtype=np.float64))
        self.potential = potential

    def initial(self, pos: NDArray[np.floating], vel: NDArray[np.floating]) -> MDState:
        return _as_state(
            super().initial(
                np.asarray(pos, dtype=np.float64),
                np.asarray(vel, dtype=np.float64),
            )
        )

    def step(self, state: MDState) -> MDState:
        return self.advance(state)

    def advance(self, state: MDState) -> MDState:
        return _as_state(super().advance(state))

    def advance_n(self, state: MDState, n_steps: int) -> MDState:
        return _as_state(super().advance_n(state, int(n_steps)))


class Langevin(_md.Langevin):
    """BAOAB Langevin — all required pieces in ``__init__``."""

    def __new__(
        cls,
        dt: float,
        *,
        gamma: float,
        kbt: float,
        potential: LJ,
        neighbors: VerletSkin,
        mass: float | NDArray[np.floating],
        seed: int = 0,
    ):
        return super().__new__(
            cls,
            float(dt),
            gamma=float(gamma),
            kbt=float(kbt),
            potential=potential,
            neighbors=neighbors,
            mass=np.atleast_1d(np.asarray(mass, dtype=np.float64)),
            seed=int(seed),
        )

    def __init__(
        self,
        dt: float,
        *,
        gamma: float,
        kbt: float,
        potential: LJ,
        neighbors: VerletSkin,
        mass: float | NDArray[np.floating],
        seed: int = 0,
    ) -> None:
        self.mass = np.atleast_1d(np.asarray(mass, dtype=np.float64))
        self.potential = potential

    def initial(self, pos: NDArray[np.floating], vel: NDArray[np.floating]) -> MDState:
        return _as_state(
            super().initial(
                np.asarray(pos, dtype=np.float64),
                np.asarray(vel, dtype=np.float64),
            )
        )

    def step(self, state: MDState, noise: NDArray[np.floating]) -> MDState:
        return _as_state(super().step(state, np.asarray(noise, dtype=np.float64)))

    def advance(self, state: MDState) -> MDState:
        return _as_state(super().advance(state))

    def advance_n(self, state: MDState, n_steps: int) -> MDState:
        return _as_state(super().advance_n(state, int(n_steps)))


class LJ(_md.LJ):
    """Pair potential: ``calc_energy`` / ``calc_force`` / ``eval → (E, F)``."""

    def __new__(
        cls,
        epsilon: float,
        sigma: float,
        cutoff: float,
        *,
        n: int = 12,
        m: int = 6,
        shifted: bool = True,
        smeared: bool = False,
    ):
        return super().__new__(
            cls,
            float(epsilon),
            float(sigma),
            float(cutoff),
            n=int(n),
            m=int(m),
            shifted=bool(shifted),
            smeared=bool(smeared),
        )

    def __init__(
        self,
        epsilon: float,
        sigma: float,
        cutoff: float,
        *,
        n: int = 12,
        m: int = 6,
        shifted: bool = True,
        smeared: bool = False,
    ) -> None:
        return

    def calc_energy(self, neighbors: VerletSkin, pos: NDArray[np.floating]) -> float:
        return float(super().calc_energy(neighbors, np.asarray(pos, dtype=np.float64)))

    def calc_force(self, neighbors: VerletSkin, pos: NDArray[np.floating]) -> NDArray[np.float64]:
        return np.asarray(
            super().calc_force(neighbors, np.asarray(pos, dtype=np.float64)),
            dtype=np.float64,
        )

    def eval(
        self, neighbors: VerletSkin, pos: NDArray[np.floating]
    ) -> tuple[float, NDArray[np.float64]]:
        energy, forces = super().eval(neighbors, np.asarray(pos, dtype=np.float64))
        return float(energy), np.asarray(forces, dtype=np.float64)


__all__ = [
    "CheckpointHook",
    "ForceOutput",
    "FrameVelocityVerlet",
    "Langevin",
    "MD",
    "MD_ENERGY",
    "MDHook",
    "MDObservables",
    "MDRunner",
    "MDState",
    "MaxwellBoltzmann",
    "NeighborList",
    "PRECISIONS",
    "LJ",
    "VelocityInitHook",
    "VelocityVerlet",
    "VerletSkin",
    "energy_to_md",
    "kb_md",
    "preset_energy_to_md",
    "resolve_prec",
]
