"""Runtime validation of the mrec record schema.

The checks live in ``molrs::io::mrec::schema``; this module is their Python
name, not a second implementation. Writers always stamp
``meta["molrec_version"]``; readers validate it only when present — an integer
in ``1..=``:data:`molrs.io.mrec.MOLREC_VERSION` — and an absent key is no version check. Identity of
a record is the ``*.mrec/`` path suffix plus a Zarr root.
"""

from ..._lib import mrec as _mrec

validate_frame = _mrec.schema.validate_frame
validate_meta = _mrec.schema.validate_meta
validate_path = _mrec.schema.validate_path

__all__ = [
    "validate_frame",
    "validate_meta",
    "validate_path",
]
