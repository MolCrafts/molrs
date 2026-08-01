"""Spectra derived from raw ACFs and polarizability, plus their checks.

The consistency checks live here rather than in a module of their own:
Kramers-Kronig judges ε(ω) and the sum rule judges σ(ω), both produced by the
spectra in this same module. compute, fit and check for one physical quantity
belong together."""

from molrs._lib import (
    check_conductivity_sum_rule as conductivity_sum_rule,
    check_kramers_kronig as kramers_kronig,
    check_route_agreement as route_agreement,
    PowerSpectrum as PowerSpectrum,
    IRSpectrum as IRSpectrum,
    RamanSpectrum as RamanSpectrum,
    EinsteinHelfandSpectrum as EinsteinHelfandSpectrum,
    GreenKuboSpectrum as GreenKuboSpectrum,
    VcdSpectrum as VcdSpectrum,
    RoaSpectrum as RoaSpectrum,
    ResonanceRamanSpectrum as ResonanceRamanSpectrum,
    polarizability_finite_field as polarizability_finite_field,
)

__all__ = [
    "conductivity_sum_rule",
    "kramers_kronig",
    "route_agreement",
    "PowerSpectrum",
    "IRSpectrum",
    "RamanSpectrum",
    "EinsteinHelfandSpectrum",
    "GreenKuboSpectrum",
    "VcdSpectrum",
    "RoaSpectrum",
    "ResonanceRamanSpectrum",
    "polarizability_finite_field",
]
