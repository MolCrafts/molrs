"""ForceField + Frame MD — Python velocity-Verlet over compiled ``Potentials``.

Distinct from the LJ path (``VelocityVerlet(dt, potential=LJ(...), neighbors=…)``
in Rust). Frame topology is compiled at :meth:`MD.run` time::

    MD().set_forcefield(ff).run(frame, n_steps, dt=dt)
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from .._lib import NeighborList
from .._lib import md as _md
from .runner import MDHook, MDRunner
from .types import ForceOutput, MDState


def _energy_scale(units_or_factor: float | str) -> float:
    if isinstance(units_or_factor, str):
        return float(_md.preset_energy_to_md(units_or_factor))
    return float(units_or_factor)


class _PotentialsForce:
    """``pos (N,3) → ForceOutput`` via ``Potentials.calc_energy_forces``."""

    def __init__(self, pots: object, scale: float) -> None:
        self._pots = pots
        self._scale = float(scale)

    def __call__(self, pos: NDArray[np.floating]) -> ForceOutput:
        pos = np.asarray(pos, dtype=np.float64)
        coords = np.ascontiguousarray(pos.reshape(-1), dtype=np.float64)
        energy, forces = self._pots.calc_energy_forces(coords)
        forces = np.asarray(forces, dtype=np.float64)
        if forces.ndim == 1:
            forces = forces.reshape(-1, 3)
        scale = self._scale
        return ForceOutput(float(energy) * scale, forces * scale)


class FrameVelocityVerlet:
    """NVE velocity-Verlet over compiled ``molrs.ff.Potentials``.

    Units: ``dt`` in fs, ``mass`` in amu; ``energy_scale`` converts pot output
    (kcal/mol for ``real``) to amu·Å²/fs².
    """

    removed_dof: int = 3
    acc_dtype = np.dtype(np.float64)

    def __init__(
        self,
        dt: float,
        *,
        potential: object,
        mass: float | NDArray[np.floating],
        energy_scale: float | str = "real",
    ) -> None:
        if not hasattr(potential, "calc_energy_forces"):
            raise TypeError(
                f"potential must be molrs.ff.Potentials, got {type(potential).__name__}"
            )
        if len(potential) == 0:
            raise ValueError(
                "Potentials is still deferred (len==0); compile with ff.to_potentials(frame) first"
            )
        self.dt = float(dt)
        self.potential = potential
        self._force = _PotentialsForce(potential, _energy_scale(energy_scale))
        mass_arr = np.atleast_1d(np.asarray(mass, dtype=np.float64))
        if mass_arr.size == 0 or np.any(~np.isfinite(mass_arr)) or np.any(mass_arr <= 0.0):
            raise ValueError("mass must be strictly positive")
        self.mass = mass_arr
        self.inv_mass = (1.0 / mass_arr).reshape(-1, 1)

    def initial(self, pos: NDArray[np.floating], vel: NDArray[np.floating]) -> MDState:
        pos = np.asarray(pos, dtype=np.float64).copy()
        vel = np.asarray(vel, dtype=np.float64).copy()
        if pos.shape != vel.shape or pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(f"pos/vel must share shape (N, 3), got {pos.shape} / {vel.shape}")
        if pos.shape[0] != self.mass.shape[0] and self.mass.shape[0] != 1:
            raise ValueError(
                f"n_atoms {pos.shape[0]} disagrees with mass length {self.mass.shape[0]}"
            )
        if self.mass.shape[0] == 1 and pos.shape[0] != 1:
            self.mass = np.full(pos.shape[0], float(self.mass[0]))
            self.inv_mass = (1.0 / self.mass).reshape(-1, 1)
        out = self._force(pos)
        return MDState(pos, vel, np.asarray(out.forces, dtype=np.float64).copy(), float(out.energy))

    def advance(self, state: MDState) -> MDState:
        half = 0.5 * self.dt
        vel = np.asarray(state.vel, dtype=np.float64) + half * np.asarray(
            state.forces, dtype=np.float64
        ) * self.inv_mass
        pos = np.asarray(state.pos, dtype=np.float64) + half * vel
        pos = pos + half * vel
        out = self._force(pos)
        vel = vel + half * np.asarray(out.forces, dtype=np.float64) * self.inv_mass
        return MDState(pos, vel, np.asarray(out.forces, dtype=np.float64), float(out.energy))

    def advance_n(self, state: MDState, n_steps: int) -> MDState:
        for _ in range(int(n_steps)):
            state = self.advance(state)
        return state


def _stack_xyz(atoms: object) -> NDArray[np.float64]:
    return np.stack(
        [np.asarray(atoms["x"], dtype=np.float64),
         np.asarray(atoms["y"], dtype=np.float64),
         np.asarray(atoms["z"], dtype=np.float64)],
        axis=1,
    )


def _stack_vel(atoms: object, shape: tuple[int, int]) -> NDArray[np.float64]:
    if all(name in atoms for name in ("vx", "vy", "vz")):
        return np.stack(
            [np.asarray(atoms["vx"], dtype=np.float64),
             np.asarray(atoms["vy"], dtype=np.float64),
             np.asarray(atoms["vz"], dtype=np.float64)],
            axis=1,
        )
    return np.zeros(shape, dtype=np.float64)


class MD:
    """Compose ``ForceField`` + Frame over :class:`FrameVelocityVerlet`.

    Attach the force field once; topology compilation happens in :meth:`run`::

        MD().set_forcefield(ff).run(frame, n_steps, dt=dt)

    :meth:`set_potential` is for callers who already hold compiled
    ``molrs.ff.Potentials`` (no frame at attach time).
    """

    def __init__(self) -> None:
        self.force: object | None = None
        self.potential: object | None = None
        self._energy_scale: float = 1.0
        self.neighbors: NeighborList | None = None
        self.hooks: list[MDHook | tuple[MDHook, int]] = []
        self.runner: MDRunner | None = None
        self._adapter: _PotentialsForce | None = None

    def set_forcefield(self, force: object) -> MD:
        """Attach a molrs ``ForceField``. Compile with ``to_potentials(frame)`` in :meth:`run`."""
        if not hasattr(force, "to_potentials"):
            raise TypeError(
                f"set_forcefield expects a ForceField, got {type(force).__name__}"
            )
        self.force = force
        self.potential = None
        self._adapter = None
        self.runner = None
        return self

    def set_potential(
        self,
        potential: object,
        *,
        energy_scale: float | str = "real",
    ) -> MD:
        """Attach pre-compiled ``molrs.ff.Potentials`` (advanced; skips :meth:`set_forcefield`)."""
        if not hasattr(potential, "calc_energy_forces"):
            raise TypeError(
                f"set_potential expects molrs.ff.Potentials, got {type(potential).__name__}"
            )
        if len(potential) == 0:
            raise ValueError(
                "Potentials is still deferred (len==0); compile with ff.to_potentials(frame) first"
            )
        self.force = None
        self.potential = potential
        self._energy_scale = _energy_scale(energy_scale)
        self._adapter = _PotentialsForce(potential, self._energy_scale)
        return self

    def set_neighbors(self, neighbors: NeighborList) -> MD:
        """Optional. Ignored for compile-once bonded ``Potentials`` (v1)."""
        self.neighbors = neighbors
        return self

    def set_hooks(self, hooks: Sequence[MDHook | tuple[MDHook, int]] | None) -> MD:
        self.hooks = list(hooks or [])
        return self

    def _wire(self, frame: object) -> tuple[object, float]:
        if self.force is not None:
            pots = self.force.to_potentials(frame)
            if len(pots) == 0:
                raise ValueError("force.to_potentials(frame) produced empty Potentials")
            scale = _energy_scale(getattr(self.force, "units", "real"))
            self.potential = pots
            self._energy_scale = scale
        elif self.potential is not None:
            pots = self.potential
            scale = self._energy_scale
        else:
            raise RuntimeError("set_forcefield or set_potential before run")
        self._adapter = _PotentialsForce(pots, scale)
        return pots, scale

    def compute_force(self, pos: NDArray[np.floating]) -> ForceOutput:
        if self._adapter is None:
            raise RuntimeError("set_forcefield or set_potential before compute_force")
        return self._adapter(pos)

    def run(
        self,
        frame: object,
        n_steps: int,
        *,
        dt: float,
        chunk: int | None = None,
        mass: NDArray[np.floating] | float | None = None,
    ) -> MDState:
        """Integrate ``n_steps`` of NVE; write coordinates (and velocities) back."""
        pots, scale = self._wire(frame)
        if chunk is None:
            chunk = 1
        atoms = frame["atoms"]
        pos = _stack_xyz(atoms)
        vel = _stack_vel(atoms, pos.shape)
        if mass is None:
            if "mass" not in atoms:
                raise ValueError("frame['atoms'] must carry a mass column, or pass mass=")
            mass_arr = np.asarray(atoms["mass"], dtype=np.float64)
        else:
            mass_arr = np.atleast_1d(np.asarray(mass, dtype=np.float64))
            if mass_arr.size == 1:
                mass_arr = np.full(pos.shape[0], float(mass_arr[0]))

        ig = FrameVelocityVerlet(
            float(dt),
            potential=pots,
            mass=mass_arr,
            energy_scale=scale,
        )
        self.runner = MDRunner(ig, hooks=self.hooks, kb=float(_md.kb_md()))
        md = self.runner.run(pos, vel, mass_arr, int(n_steps), chunk=int(chunk))

        atoms["x"] = md.pos[:, 0]
        atoms["y"] = md.pos[:, 1]
        atoms["z"] = md.pos[:, 2]
        atoms["vx"] = md.vel[:, 0]
        atoms["vy"] = md.vel[:, 1]
        atoms["vz"] = md.vel[:, 2]
        return md
