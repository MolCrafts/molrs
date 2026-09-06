"""Runtime validation of the mrec record schema.

The checks live in ``molrs::io::mrec::schema``. This module is the Python
binding, not a second implementation. While the record contract is in
development ``meta["molrec_version"]`` is optional: an absent key means no
version validation, a present one must be an integer in ``1..=MOLREC_VERSION``.
Identity of a record is the ``*.mrec/`` path suffix plus a Zarr root.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..._lib import (
    MREC_MOLREC_VERSION as MOLREC_VERSION,
)
from ..._lib import (
    MREC_RESERVED_META_KEYS as RESERVED_META_KEYS,
)
from ..._lib import (
    mrec_validate_frame as validate_frame,
)
from ..._lib import (
    mrec_validate_meta as _validate_meta,
)
from ..._lib import (
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
    """Validate the optional ``meta`` version key against the mrec contract.

    Args:
        meta: Record-level metadata mapping.

    Raises:
        ValueError: If ``molrec_version`` is present and not an integer in
            ``1..=MOLREC_VERSION``. An absent key passes.
    """
    _validate_meta(dict(meta))
