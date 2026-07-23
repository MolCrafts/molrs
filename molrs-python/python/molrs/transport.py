"""Ion-transport trajectory-analysis kernels.

All computation is in Rust. The kernels are grouped under three thin namespace
classes (static methods over the compiled ``molrs._lib`` PyO3 functions) so
callers reach them as ``molrs.transport.Onsager.correlation(...)`` rather than
through a flat list of free functions.

These are the molrs ports of the *tame* recipes
(<https://github.com/Roy-Kid/tame>): ``onsager`` (Onsager transport
coefficients), ``jacf`` (current-ACF Green–Kubo conductivity), and ``persist``
(pair-survival / residence-time correlations).
"""

from ._lib import (
    transport_onsager_correlation,
    transport_pair_survival_tcf,
)


class Onsager:
    """Onsager collective mean-displacement cross-correlation (static)."""

    correlation = staticmethod(transport_onsager_correlation)


# The bundled ``Jacf.green_kubo_conductivity`` (raw JACF + fitted sigma) was
# removed in compute-fit-03-cleanup: compose :class:`molrs.GreenKuboConductivity`
# (raw current ACF) with :class:`molrs.RunningIntegral` and a
# ``1/(3·V·k_B·T)`` MD→SI prefactor instead.


class Persist:
    """Pair-survival (persistence) time-correlation functions (static)."""

    pair_survival_tcf = staticmethod(transport_pair_survival_tcf)


__all__ = ["Onsager", "Persist"]
