"""Force fields — ``molrs::ff``.

One submodule per Rust submodule, so the Python path and the Rust path are the
same words; this package itself holds nothing but them:

* :mod:`~molrs.ff.forcefield` — the :class:`~molrs.ff.forcefield.ForceField`
  container with its ``Style`` / ``Type`` handles, and the force-field file
  readers and writers (LAMMPS, GROMACS, AMBER, OpenMM XML, CMAP)
* :mod:`~molrs.ff.potential` — evaluable force terms: the
  ``PotentialCompiler``, the ``Potentials`` it builds, ``kernel`` for any
  style over explicit instances, ``PairLjCut``, and the ``Potential`` protocol
* :mod:`~molrs.ff.typifier` — the subclassable ``Typifier`` base and its
  ``Match``, the built-in atom typers, and ``assign_cmaps``
* :mod:`~molrs.ff.charge` — partial-charge models (AM1-BCC / ABCG2,
  Mulliken, Gasteiger)
* :mod:`~molrs.ff.ir` — the force-field IR as a protocol: register a new
  category or style (expression or Python kernel) with nothing rebuilt
* :mod:`~molrs.ff.params` — the parameter tables molrs ships (AMBER 1-4
  scales, the CL&Pol polarizabilities)
* :mod:`~molrs.ff.scale_lj` — CL&Pol fragment scaling of Lennard-Jones
  parameters
"""

from . import charge, forcefield, ir, params, potential, scale_lj, typifier

__all__ = [
    "charge",
    "forcefield",
    "ir",
    "params",
    "potential",
    "scale_lj",
    "typifier",
]
