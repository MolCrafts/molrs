"""Potential forms (``molrs::ff::potential``).

* :class:`Potential` — the protocol every Python-defined force provider
  satisfies (one method, ``calc_energy_forces(pos) -> (energy, forces)``).
* :func:`kernel` — the kernel of **any** style the force-field IR prices (a
  built-in, a style registered through :mod:`molrs.ff.ir` by expression or
  Python kernel, a style of a custom category) over explicit instances: atom
  indices and one parameter row per term, as stored (angle values in
  degrees). It returns a :class:`~molrs.ff.Potentials`, which
  ``Potentials.push`` moves into a larger collection::

      pots = Potentials()
      pots.push(kernel("bond", "harmonic", [[0, 1], [1, 2]], k=300.0, r0=1.4))
      pots.push(kernel("angle", "harmonic", [[0, 1, 2]], k=50.0, theta0=109.5))
      energy, forces = pots.calc_energy_forces(pos)

* :class:`LJCut` — the one-type ``lj/cut`` kernel a neighbour loop feeds
  (the MD integrators' nonbond kernel).
"""

from ..._lib import potential as _potential
from .protocol import Potential

LJCut = _potential.LJCut
kernel = _potential.kernel

__all__ = ["LJCut", "Potential", "kernel"]
