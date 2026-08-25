"""In-process MD: one ``Potential`` concept, Rust integrators, the ``MD`` driver.

The user-facing spelling of everything here is ``molpy.md`` (a verbatim
re-export); ``molrs.md`` is the engine-side home of the same objects.

End to end (Ar-like LJ dimer)::

    import numpy as np
    from molpy import Box, md

    pos = np.array([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]])
    rc, skin = 7.5, 1.0
    # search cutoff = rc + skin (what the engine indexes);
    # force cutoff = rc (what the potential sees); skin is the rebuild buffer.
    nl = md.VerletSkin(md.NeighborList(rc + skin), rc, pos, Box.cubic(20.0), skin=skin)
    eps = 0.238  # caller units; MD does not convert
    vv = md.VelocityVerlet(1.0, potential=md.LJCut(eps, 3.405, rc),
                           neighbors=nl, mass=np.full(2, 39.948))
    state = vv.initial(pos, np.zeros_like(pos))
    state = vv.advance_n(state, 100)

(Engine-side, ``NeighborList`` / ``VerletSkin`` live at the molrs top level —
one spelling per namespace.) The integrator owns the neighbour loop: it runs
the skin's rebuild policy and feeds fresh pairs to the nonbond potential;
Python never does pair bookkeeping.

Units contract — the engine is **unit-agnostic**. Take constants from
``molpy.UnitPreset`` (engine path ``molrs.UnitPreset``)::

    kb = molpy.UnitPreset("real").boltzmann()
    md.MaxwellBoltzmann(kb * 300.0, seed=0)
    md.MD().run(frame, n, dt=dt, kb=kb, thermo=100)

External forces (the NN/Torch seam) subclass :class:`Potential`::

    class Spring(md.Potential):
        def calc_energy_forces(self, pos):
            return 0.05 * float((pos * pos).sum()), -0.1 * pos

ForceField + Frame runs go through the :class:`MD` driver::

    md.MD().set_forcefield(ff).set_neighbors(cutoff=rc, skin=2.0).run(
        frame, 1000, dt=1.0, kb=molpy.UnitPreset("real").boltzmann()
    )

Precision: ``MD(dtype=np.float64)`` is the only entry. ``np.float32`` / mixed
raise; those loops belong in the Rust integrators.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "molrs.md is experimental in 0.14: APIs may change or be removed in a "
    "future minor release.",
    FutureWarning,
    stacklevel=2,
)

from .._lib import md as _md
from ..ff import Potentials
from ..ff.potential import Potential

LJCut = _md.LJCut
Langevin = _md.Langevin
MDState = _md.MDState
MaxwellBoltzmann = _md.MaxwellBoltzmann
VelocityVerlet = _md.VelocityVerlet

from .driver import MD

__all__ = [
    "LJCut",
    "Langevin",
    "MD",
    "MDState",
    "MaxwellBoltzmann",
    "Potential",
    "Potentials",
    "VelocityVerlet",
]
