"""Units — ``molrs::units``.

The native unit engine: :class:`Unit`, :class:`Quantity`, a
:class:`UnitRegistry` for custom definitions, and the engine presets
(:class:`UnitPreset`: ``"real"``, ``"metal"``, …, with their Boltzmann
constants and conversions). A failed parse, definition, arithmetic step or
conversion raises :exc:`UnitsError` (a ``ValueError``).

:data:`AMBER_COULOMB` is the Coulomb constant AMBER uses (kcal·Å/(mol·e²);
``18.2223²``, the factor behind prmtop charges), from
``molrs::units::constants``.
"""

from ._lib import (
    AMBER_COULOMB,
    Quantity,
    Unit,
    UnitPreset,
    UnitRegistry,
    UnitsError,
)

__all__ = [
    "AMBER_COULOMB",
    "Quantity",
    "Unit",
    "UnitPreset",
    "UnitRegistry",
    "UnitsError",
]
