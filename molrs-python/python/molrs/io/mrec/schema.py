"""Runtime validation of the mrec record schema.

The checks live in ``molrs::io::mrec::schema``. This module is the Python
binding, not a second implementation. ``meta["molrec_version"]`` is the sole
version key of a record; identity is that key plus the ``*.mrec/`` path
suffix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ..._lib import (
    MREC_MOLREC_VERSION as MOLREC_VERSION,
    MREC_RESERVED_META_KEYS as RESERVED_META_KEYS,
    mrec_validate_frame as validate_frame,
    mrec_validate_meta as _validate_meta,
    mrec_validate_path as _validate_path,
)

__all__ = [
    "MOLREC_VERSION",
    "RESERVED_META_KEYS",
    "validate_frame",
    "validate_meta",
    "validate_path",
]


def validate_path(path: str | Path) -> None:
    """Refuse the retired ``.zarr`` / ``.zarr.zip`` scientific suffixes.

    Args:
        path: Filesystem path of a record store.

    Raises:
        ValueError: If the path uses a retired suffix.
    """
    _validate_path(str(path))


def validate_meta(meta: Mapping[str, Any]) -> None:
    """Validate the mandatory ``meta`` version key against the mrec contract.

    Args:
        meta: Record-level metadata mapping.

    Raises:
        ValueError: If ``molrec_version`` is missing or not a version this
            reader supports (``1..=MOLREC_VERSION``).
    """
    _validate_meta(dict(meta))
