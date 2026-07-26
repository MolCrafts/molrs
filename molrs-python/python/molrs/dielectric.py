"""Dielectric raw-observable kernels.

All computation is in Rust. The kernels are grouped under a single
:class:`Dielectric` namespace (a thin set of static methods over the compiled
``molrs._lib`` PyO3 functions) so callers reach them as
``molrs.dielectric.Dielectric.static_dielectric_constant(...)`` rather than
through a flat list of free functions.

Only the **raw / defined** dielectric quantities live here: dipole moment,
current density, current partition, and the Neumann static dielectric constant.

The frequency-dependent ε(ω) spectrum is no longer a bundled free function. It
is the explicit raw-compute + Fit composition (compute-fit-04-dielectric):

* Einstein–Helfand route — :class:`molrs.DebyeRelaxation` (raw fluctuation
  dipole ACF + ⟨M²⟩ + V/T/Ewald-BC) → :class:`molrs.EinsteinHelfandSpectrum`.
* Green–Kubo route — :class:`molrs.GreenKuboConductivity` (raw current ACF) →
  :class:`molrs.GreenKuboSpectrum`.

The bundled ``einstein_helfand_conductivity`` was likewise removed in
compute-fit-03-cleanup: compose :class:`molrs.EinsteinConductivity` (raw
collective-dipole MSD) with :class:`molrs.LinearFit` and a
``slope/(6·V·k_B·T)`` MD→SI prefactor instead.
"""

from ._lib import (
    dielectric_compute_current_density,
    dielectric_compute_dipole_moment,
    dielectric_decompose_current,
    dielectric_static_dielectric_constant,
)


class Dielectric:
    """Namespace of raw dielectric kernels (all static).

    Grouping the kernels under one class keeps the public ``molrs.dielectric``
    surface to a single name instead of a flat list of free functions. Every
    method is a ``staticmethod`` forwarding straight to the Rust implementation.
    """

    compute_dipole_moment = staticmethod(dielectric_compute_dipole_moment)
    compute_current_density = staticmethod(dielectric_compute_current_density)
    decompose_current = staticmethod(dielectric_decompose_current)
    static_dielectric_constant = staticmethod(dielectric_static_dielectric_constant)


__all__ = ["Dielectric"]
