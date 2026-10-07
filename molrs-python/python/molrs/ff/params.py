"""Parameter tables molrs ships — ``molrs::ff::params``.

* :func:`clpol_polarizability` — the CL&Pol ``alpha.ff`` Drude table, the
  one molrs ships or one read from a file.
* :func:`clpol_fragment_scaling` — the CL&Pol ``fragment.ff`` scaling table
  :func:`molrs.ff.clpol_scaling.scale_lj` reads.

AMBER's 1-4 divisors are engine constants:
:data:`molrs.core.constants.AMBER_SCEE` / :data:`~molrs.core.constants.AMBER_SCNB`.
"""

from .._native import clpol_fragment_scaling, clpol_polarizability

__all__ = ["clpol_fragment_scaling", "clpol_polarizability"]
