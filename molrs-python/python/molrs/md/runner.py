"""Hook-driven MD runner.

:class:`MDRunner` owns the observation loop. Hooks receive the step count
and typed physics. Trajectory *format* is not this module's: attach a
:class:`CheckpointHook` that wraps any writer.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from .types import MDObservables, MDState


class IntegratorLike(Protocol):
    """Narrow protocol the runner drives."""

    removed_dof: int

    def initial(self, pos: object, vel: object) -> MDState: ...

    def advance_n(self, state: MDState, n_steps: int) -> MDState: ...


class MDHook:
    """Lifecycle observer for an MD run.

    Subclass and override only what you need; every method is a no-op by
    default. Hooks that act on a step cadence must declare it in
    :attr:`cadence` so the runner can validate ``chunk``.
    """

    cadence: int | None = None

    def on_init(self, runner: MDRunner, state: MDState) -> None:
        """Called once after the entry force is cached, before any step."""

    def on_run_start(self, runner: MDRunner) -> None:
        """Called once before the first step."""

    def on_step_start(self, runner: MDRunner, step: int, state: MDState) -> None:
        """Called before each hook-visible advance."""

    def on_step_end(self, runner: MDRunner, step: int, obs: MDObservables) -> None:
        """Called after each hook-visible advance."""

    def on_run_end(self, runner: MDRunner) -> None:
        """Called once after the last step."""


class VelocityInitHook(MDHook):
    """Write initial velocities from a distribution (LAMMPS ``velocity create``)."""

    def __init__(self, distribution: object) -> None:
        self.distribution = distribution

    def on_init(self, runner: MDRunner, state: MDState) -> None:
        vel = self.distribution.velocities(state.pos, runner.integrator.mass)
        state.vel[:] = np.asarray(vel, dtype=state.vel.dtype)


class FrameWriter(Protocol):
    """Anything that can persist one frame. Users pass ``XYZWriter`` etc."""

    def write_frame(self, frame: object) -> None: ...


class CheckpointHook(MDHook):
    """Call ``writer.write_frame(frame)`` every ``cadence`` steps.

    The engine does not own a file format. Compose with an existing writer::

        CheckpointHook(XYZWriter("traj.xyz"), cadence=10)
    """

    def __init__(self, writer: FrameWriter, *, cadence: int = 1, frame: object | None = None) -> None:
        if cadence < 1:
            raise ValueError(f"cadence must be >= 1, got {cadence}")
        self.writer = writer
        self.cadence = int(cadence)
        self.frame = frame

    def on_step_end(self, runner: MDRunner, step: int, obs: MDObservables) -> None:
        if step % self.cadence:
            return
        frame = self.frame
        if frame is None:
            raise RuntimeError(
                "CheckpointHook has no frame to write; pass frame= at construction "
                "or set hook.frame before run"
            )
        atoms = frame["atoms"]
        atoms["x"] = np.asarray(obs.pos[:, 0])
        atoms["y"] = np.asarray(obs.pos[:, 1])
        atoms["z"] = np.asarray(obs.pos[:, 2])
        if "vx" in atoms:
            atoms["vx"] = np.asarray(obs.vel[:, 0])
            atoms["vy"] = np.asarray(obs.vel[:, 1])
            atoms["vz"] = np.asarray(obs.vel[:, 2])
        self.writer.write_frame(frame)


def _normalize_hooks(
    hooks: Sequence[MDHook | tuple[MDHook, int]] | None,
) -> list[MDHook]:
    if not hooks:
        return []
    normalized = []
    for idx, item in enumerate(hooks):
        if isinstance(item, tuple):
            hook, priority = item
            normalized.append((hook, priority, idx))
        else:
            normalized.append((item, 100, idx))
    normalized.sort(key=lambda x: (x[1], x[2]))
    return [hook for hook, _, _ in normalized]


class MDRunner:
    """Drive an integrator through the MD hook lifecycle."""

    def __init__(
        self,
        integrator: IntegratorLike,
        *,
        hooks: Sequence[MDHook | tuple[MDHook, int]] | None = None,
        kb: float,
        dof: int | None = None,
    ) -> None:
        self.integrator = integrator
        self.hooks: list[MDHook] = _normalize_hooks(hooks)
        self._kb = float(kb)
        self._dof = dof

    def run(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        mass: np.ndarray,
        n_steps: int,
        *,
        chunk: int = 1,
    ) -> MDState:
        """Integrate ``n_steps``, firing hooks per chunk.

        ``mass`` is used for kinetic / temperature reporting. The integrator
        already owns mass for dynamics (passed at construction).
        """
        acc = np.dtype(getattr(self.integrator, "acc_dtype", np.float64))
        mass_col = np.asarray(mass, dtype=acc).reshape(-1, 1)
        if self._dof is not None:
            dof = self._dof
        else:
            removed = int(getattr(self.integrator, "removed_dof", 3))
            dof = max(1, 3 * int(pos.shape[0]) - removed)

        chunk = max(1, int(chunk))
        for hook in self.hooks:
            if hook.cadence is not None and hook.cadence % chunk:
                raise ValueError(
                    f"{type(hook).__name__} fires every {hook.cadence} steps, which chunk="
                    f"{chunk} would silently skip; make it a multiple of chunk"
                )
        md = self.integrator.initial(pos, vel)
        for hook in self.hooks:
            hook.on_init(self, md)
        for hook in self.hooks:
            hook.on_run_start(self)
        done = 0
        while done < n_steps:
            n = min(chunk, n_steps - done)
            for hook in self.hooks:
                hook.on_step_start(self, done, md)
            md = self.integrator.advance_n(md, n)
            done += n
            vel_acc = np.asarray(md.vel, dtype=acc)
            kinetic = float(0.5 * (mass_col * vel_acc * vel_acc).sum())
            potential = float(np.asarray(md.energy).reshape(()))
            temperature = 2.0 * kinetic / (dof * self._kb)
            obs = MDObservables(
                pos=md.pos,
                vel=md.vel,
                forces=md.forces,
                potential=potential,
                kinetic=kinetic,
                total=potential + kinetic,
                temperature=temperature,
            )
            for hook in self.hooks:
                hook.on_step_end(self, done, obs)
        for hook in self.hooks:
            hook.on_run_end(self)
        return md
