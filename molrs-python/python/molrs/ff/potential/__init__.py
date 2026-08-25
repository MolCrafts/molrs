"""Potential forms (``molrs::ff::potential``)."""
from . import soft  # noqa: F401
from .protocol import Potential

__all__ = ["Potential", "soft"]
