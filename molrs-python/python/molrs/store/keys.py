"""Canonical Frame column names, projected from the compiled Rust tables.

Every scalar is a :class:`Key`; ordered groups (``COORDS``, …) are lists of
:class:`Key`. A column added to the Rust tables appears here with no edit.
"""

from .._lib import keys as _keys

__all__ = sorted(name for name in dir(_keys) if not name.startswith("_"))
globals().update({name: getattr(_keys, name) for name in __all__})
