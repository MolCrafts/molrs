"""Physical and engine constants — ``molrs::core::constants``, in full.

CODATA 2018 / SI-2019 values (``AVOGADRO``, ``BOLTZMANN``,
``SPEED_OF_LIGHT``, …), unit-conversion factors (``KJ_PER_KCAL``,
``ANGSTROM_PER_NM``, ``ANGSTROM3_PER_CM3``, …), and the constants engines
and force fields define (``COULOMB_REAL``, ``AMBER_COULOMB``,
``AMBER_SCEE`` / ``AMBER_SCNB``, ``UFF_COULOMB``, …). Every name and value
is the Rust constant of the same name; a constant added in Rust appears here
with no edit.
"""

from .._lib import constants as _constants

__all__ = sorted(name for name in dir(_constants) if name.isupper())
globals().update({name: getattr(_constants, name) for name in __all__})
