"""CL&Pol fragment scaling of Lennard-Jones parameters — ``molrs::ff::scale_lj``.

:func:`scale_lj` returns a copy of a force field whose LJ well depths (and,
optionally, diameters) are scaled per fragment pair by the SAPT factor
:func:`compute_k_ij`; :class:`FragmentScaling` is one fragment's charge,
dipole and polarizability, and :func:`fragment_scaling_data` the table molrs
ships.
"""

from __future__ import annotations

from .._lib import FragmentScaling, compute_k_ij, fragment_scaling_data, scale_lj

__all__ = ["FragmentScaling", "compute_k_ij", "fragment_scaling_data", "scale_lj"]
