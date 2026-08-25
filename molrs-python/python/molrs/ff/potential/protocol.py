"""The one Potential contract: ``calc_energy_forces(pos) -> (energy, forces)``.

This is a :func:`typing.runtime_checkable` :class:`~typing.Protocol`. Any
object that defines ``calc_energy_forces`` satisfies it structurally —
explicit subclassing is optional. ``isinstance`` only checks that the
method **exists**, not its signature (PEP 544), and is not suitable for
hot-path dispatch. Integrators dispatch on concrete Rust types first and
fall back to duck typing last.

The Python contract is coordinates-only. Neighbour / pair data never
crosses the FFI into a duck-typed implementation; custom forces (NN /
Torch) own their own neighbour correctness.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Potential(Protocol):
    """Structured force provider: one method, coordinates in, energy+forces out."""

    def calc_energy_forces(self, pos):
        raise NotImplementedError(
            "Potential.calc_energy_forces must return "
            "(energy: float, forces: float64 (N, 3) ndarray)"
        )
