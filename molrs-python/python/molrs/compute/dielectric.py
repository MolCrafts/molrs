"""Dielectric raw-observable kernels.

All computation is in Rust. The kernels are the static methods of one native
:class:`Dielectric` namespace, so callers reach them as
``molrs.compute.dielectric.Dielectric.static_dielectric_constant(...)``.

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

from molrs._lib import Dielectric

__all__ = ["Dielectric"]
