"""Physical-consistency checks on dielectric / transport spectra.

These judge a *result* against physics — Kramers-Kronig causality, the
conductivity sum rule, and agreement between the Einstein-Helfand and
Green-Kubo routes. They are deliberately separate from the raw kernels in
:mod:`molrs.compute.dielectric` and :mod:`molrs.compute.transport`: producing
a quantity and judging one are different responsibilities, the same split the
raw-compute / explicit-:mod:`~molrs.compute.fit` boundary already draws.

Was ``molrs.validate`` (and ``compute::validate`` on the Rust side). "Validate"
now names one thing only — Frame / Block schema conformance, which is about
data structure and applies to every Frame regardless of physics. Judging a
computed spectrum against physics is a different question and gets a different
word.

All computation is in Rust; these are thin Python re-exports.
"""

from molrs._lib import (
    check_conductivity_sum_rule as conductivity_sum_rule,
    check_kramers_kronig as kramers_kronig,
    check_route_agreement as route_agreement,
)

__all__ = [
    "conductivity_sum_rule",
    "kramers_kronig",
    "route_agreement",
]
