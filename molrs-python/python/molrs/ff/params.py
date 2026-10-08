"""Parameter tables molrs ships — ``molrs::ff::params``.

* :func:`clpol_polarizability` — the CL&Pol ``alpha.ff`` Drude table molrs
  ships. A caller's own ``alpha.ff`` is read by
  :func:`molrs.io.read_clpol_alpha`.

AMBER's 1-4 divisors are engine constants:
:data:`molrs.core.constants.AMBER_SCEE` / :data:`~molrs.core.constants.AMBER_SCNB`.
"""

from .._native import clpol_polarizability

__all__ = ["clpol_polarizability"]
