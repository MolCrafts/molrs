"""The one Compute contract: ``compute(...)``.

A :func:`typing.runtime_checkable` :class:`~typing.Protocol`. Existing
pyclass kernels already satisfy it by defining ``compute`` — they are not
modified. ``isinstance`` only checks method presence (PEP 544) and is not
a hot-path dispatch tool.

``__call__`` and ``dump()`` are not part of this contract.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Compute(Protocol):
    """Structured analysis kernel: one method, ``compute``."""

    def compute(self, *args, **kwargs):
        raise NotImplementedError("Compute.compute must be implemented")
