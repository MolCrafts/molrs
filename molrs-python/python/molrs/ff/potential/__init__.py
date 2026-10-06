"""Potential forms (``molrs::ff::potential``).

* :class:`Potential` — the protocol every Python-defined force provider
  satisfies (one method, ``calc_energy_forces(pos) -> (energy, forces)``).
* The kernels, one class per style, constructed from explicit instances (atom
  indices and one parameter row each, in the force field's convention —
  LAMMPS's, angles in degrees) and moved into a :class:`~molrs.ff.Potentials`
  by ``Potentials.push``::

      pots = Potentials()
      pots.push(BondHarmonic(atomi, atomj, k, r0))
      pots.push(AngleHarmonic(atomi, atomj, atomk, k, theta0_deg))
      energy, forces = pots.calc_energy_forces(pos)

  ``LJCut(epsilon, sigma, cutoff)`` is the one-type kernel a neighbour loop
  feeds; ``LJCut.compiled(...)`` is the same style over a fixed pair list.
"""

from ..._lib import potential as _potential
from .protocol import Potential

AngleHarmonic = _potential.AngleHarmonic
BondHarmonic = _potential.BondHarmonic
DihedralPeriodic = _potential.DihedralPeriodic
ImproperCvff = _potential.ImproperCvff
ImproperPeriodic = _potential.ImproperPeriodic
LJCut = _potential.LJCut
PairCoulCut = _potential.PairCoulCut

__all__ = [
    "AngleHarmonic",
    "BondHarmonic",
    "DihedralPeriodic",
    "ImproperCvff",
    "ImproperPeriodic",
    "LJCut",
    "PairCoulCut",
    "Potential",
]
