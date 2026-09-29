"""Potential forms (``molrs::ff::potential``): the :class:`Potential` protocol
every Python-defined force provider satisfies."""

from .protocol import Potential

__all__ = ["Potential"]
