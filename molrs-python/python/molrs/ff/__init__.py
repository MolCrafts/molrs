"""Force fields — ``molrs::ff``.

One submodule per Rust submodule, so the Python path and the Rust path are the
same words; this package itself holds nothing but them:

* :mod:`~molrs.ff.forcefield` — the :class:`~molrs.ff.forcefield.ForceField`
  data model with its ``Style`` / ``ForceFieldType`` handles (its files are
  :mod:`molrs.io`'s)
* :mod:`~molrs.ff.potential` — evaluable force terms: the
  ``PotentialCompiler``, the ``Potentials`` it builds,
  ``compile_explicit_terms`` for any style over explicit terms,
  ``PairLjCut``, and the ``Potential`` protocol
* :mod:`~molrs.ff.typifier` — the subclassable ``Typifier`` base and its
  ``TypeAssignment``, the built-in typifiers, and ``assign_cmaps``
* :mod:`~molrs.ff.charge` — partial-charge models (AM1-BCC / ABCG2,
  Mulliken, Gasteiger)
* :mod:`~molrs.ff.ir` — the force-field IR as a protocol: register a new
  category or style (expression or Python kernel) with nothing rebuilt
* :mod:`~molrs.ff.params` — the parameter tables molrs ships (the CL&Pol
  polarizabilities and fragment scaling table)
* :mod:`~molrs.ff.clpol_scaling` — CL&Pol fragment scaling of Lennard-Jones
  parameters
"""

from . import charge, clpol_scaling, forcefield, ir, params, potential, typifier

__all__ = [
    "charge",
    "clpol_scaling",
    "forcefield",
    "ir",
    "params",
    "potential",
    "typifier",
]
