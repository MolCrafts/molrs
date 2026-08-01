"""Generic curve fits — the cross-quantity tools, in scipy's vocabulary.

``CumulativeTrapezoid`` is ``scipy.integrate.cumulative_trapezoid``;
``LinearFit`` is an OLS line over a window (``scipy.stats.linregress``).

Model fits tied to one physical quantity do NOT live here — ``DebyeFit`` sits
with ``DebyeRelaxation`` in :mod:`molrs.compute.transport`, because compute,
fit and check for one quantity belong in one place."""

from molrs._lib import (
    LinearFit as LinearFit,
    CumulativeTrapezoid as CumulativeTrapezoid,
    Plateau as Plateau,
)

__all__ = [
    "LinearFit",
    "CumulativeTrapezoid",
    "Plateau",
]
