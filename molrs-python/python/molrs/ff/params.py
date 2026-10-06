"""Parameter tables molrs ships — ``molrs::ff::params``.

* :data:`AMBER_SCEE` / :data:`AMBER_SCNB` — AMBER's default 1-4
  electrostatic and van der Waals scale divisors (1.2 and 2.0).
* :func:`clpol_polarizability` — the CL&Pol ``alpha.ff`` Drude table, the
  one molrs ships or one read from a file.
"""

from __future__ import annotations

from .._lib import AMBER_SCEE, AMBER_SCNB, clpol_polarizability

__all__ = ["AMBER_SCEE", "AMBER_SCNB", "clpol_polarizability"]
