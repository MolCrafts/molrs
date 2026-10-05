"""Runtime validation of the mrec record schema.

The checks live in ``molrs::io::mrec::schema``; this module is their Python
name, not a second implementation. Writers always stamp
``meta["molrec_version"]``; readers validate it only when present — an integer
in ``1..=MOLREC_VERSION`` — and an absent key is no version check. Identity of
a record is the ``*.mrec/`` path suffix plus a Zarr root.
"""

from ..._lib import MREC_MOLREC_VERSION as MOLREC_VERSION
from ..._lib import MREC_RESERVED_META_KEYS as RESERVED_META_KEYS
from ..._lib import mrec_validate_frame as validate_frame
from ..._lib import mrec_validate_meta as validate_meta
from ..._lib import mrec_validate_path as validate_path

__all__ = [
    "MOLREC_VERSION",
    "RESERVED_META_KEYS",
    "validate_frame",
    "validate_meta",
    "validate_path",
]
