"""Evaluable force terms — ``molrs::ff::potential``.

* :class:`PotentialCompiler` — compiles a
  :class:`~molrs.ff.forcefield.ForceField` against a typed frame into
  :class:`Potentials` (``compile``), or into :class:`TypedPotentials` — each
  kernel with its special-bonds weights — for a neighbour-driven integrator
  (``compile_typed``).
* :class:`Potentials` — kernels evaluated together
  (``calc_energy_forces``); ``push`` moves one more member in.
* :func:`kernel` — the kernel of **any** style the force-field IR prices (a
  built-in, a style registered through :mod:`molrs.ff.ir` by expression or
  Python kernel, a style of a custom category) over explicit instances: atom
  indices and one parameter row per term, as stored (angle values in
  degrees). It returns a :class:`Potentials`::

      pots = Potentials()
      pots.push(kernel("bond", "harmonic", [[0, 1], [1, 2]], k=300.0, r0=1.4))
      pots.push(kernel("angle", "harmonic", [[0, 1, 2]], k=50.0, theta0=109.5))
      energy, forces = pots.calc_energy_forces(pos)

* :class:`LJCut` — the one-type ``lj/cut`` kernel a neighbour loop feeds
  (the MD integrators' nonbond kernel).
* :func:`intramolecular_pairs` — the special-bonds pair list of a typed
  frame, what ``PotentialCompiler.compile`` prices nonbonded terms over.
* :class:`Potential` — the protocol every Python-defined force provider
  satisfies (one method, ``calc_energy_forces(pos) -> (energy, forces)``).
"""

from ..._lib import (
    LJCut,
    PotentialCompiler,
    Potentials,
    TypedPotentials,
    intramolecular_pairs,
    kernel,
)
from ._protocol import Potential

Potential.__module__ = __name__

__all__ = [
    "LJCut",
    "Potential",
    "PotentialCompiler",
    "Potentials",
    "TypedPotentials",
    "intramolecular_pairs",
    "kernel",
]
