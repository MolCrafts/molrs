"""Ion-transport raw Computes — identity re-exports of the Rust surface.

Compose yourself (same as Rust):

* Green–Kubo σ: ``GreenKuboConductivity`` → ``CumulativeTrapezoid`` → scale
* Einstein–Helfand σ: ``EinsteinConductivity`` → ``LinearFit`` → scale
* Self-diffusion D: ``VACF`` / ``EinsteinDiffusion`` → fit


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
