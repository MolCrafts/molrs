"""Spectra derived from raw ACFs and polarizability, plus their checks.

The consistency checks live here rather than in a module of their own:
Kramers-Kronig judges ε(ω) and the sum rule judges σ(ω), both produced by the
spectra in this same module. compute, fit and check for one physical quantity
belong together."""

from molrs._lib import (
    DipoleAutocorrelationSpectrum,
    DipoleRateCrossSpectrum,
    EinsteinHelfandSpectrum,
    GreenKuboSpectrum,
    IRSpectrum,
    PowerSpectrum,
    RamanSpectrum,
    ResonanceRamanSpectrum,
    RoaSpectrum,
    VcdSpectrum,
    polarizability_finite_field,
)
from molrs._lib import (
    check_conductivity_sum_rule as conductivity_sum_rule,
)
from molrs._lib import (
    check_kramers_kronig as kramers_kronig,
)
from molrs._lib import (
    check_route_agreement as route_agreement,
)

__all__ = [
    "DipoleAutocorrelationSpectrum",
    "DipoleRateCrossSpectrum",
    "EinsteinHelfandSpectrum",
    "GreenKuboSpectrum",
    "IRSpectrum",
    "PowerSpectrum",
    "RamanSpectrum",
    "ResonanceRamanSpectrum",
    "RoaSpectrum",
    "VcdSpectrum",
    "conductivity_sum_rule",
    "kramers_kronig",
    "polarizability_finite_field",
    "route_agreement",
]
