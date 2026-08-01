"""Ion-transport trajectory-analysis kernels.

All computation is in Rust. Two shapes live here, matching ``compute::transport``:

* **Raw computes** — :class:`VACF`, :class:`GreenKuboDiffusion`,
  :class:`EinsteinDiffusion`, :class:`EinsteinConductivity`,
  :class:`GreenKuboConductivity`, :class:`DebyeRelaxation`. Each returns ONLY a
  raw curve; the derived coefficient comes from an explicit
  :mod:`molrs.compute.fit` step.
* **Namespace classes** — :class:`Onsager` and :class:`Persist`, thin static
  wrappers over the compiled free functions so callers reach them as
  ``Onsager.correlation(...)`` rather than through a flat function list.

The namespace classes are the molrs ports of the *tame* recipes
(<https://github.com/Roy-Kid/tame>): ``onsager`` (Onsager transport
coefficients) and ``persist`` (pair-survival / residence-time correlations).
"""

from molrs._lib import (
    VACF as VACF,
    DebyeFit as DebyeFit,
    DebyeRelaxation as DebyeRelaxation,
    EinsteinConductivity as EinsteinConductivity,
    EinsteinDiffusion as EinsteinDiffusion,
    GreenKuboConductivity as GreenKuboConductivity,
    GreenKuboDiffusion as GreenKuboDiffusion,
    transport_onsager_correlation,
    transport_pair_survival_tcf,
)


class Onsager:
    """Onsager collective mean-displacement cross-correlation (static)."""

    correlation = staticmethod(transport_onsager_correlation)


# The bundled ``Jacf.green_kubo_conductivity`` (raw JACF + fitted sigma) was
# removed in compute-fit-03-cleanup: compose :class:`GreenKuboConductivity`
# (raw current ACF) with :class:`molrs.compute.fitting.CumulativeTrapezoid` and a
# ``1/(3·V·k_B·T)`` MD→SI prefactor instead.


class Persist:
    """Pair-survival (persistence) time-correlation functions (static)."""

    pair_survival_tcf = staticmethod(transport_pair_survival_tcf)


__all__ = [
    "VACF",
    "GreenKuboDiffusion",
    "EinsteinDiffusion",
    "EinsteinConductivity",
    "GreenKuboConductivity",
    "DebyeRelaxation",
    "DebyeFit",
    "Onsager",
    "Persist",
]
