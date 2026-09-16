"""Canonical Frame column names, projected from the compiled Rust tables."""

from ._lib import keys as _keys


def __getattr__(name: str):
    return getattr(_keys, name)


def __dir__() -> list[str]:
    return sorted(n for n in dir(_keys) if not n.startswith("_"))
