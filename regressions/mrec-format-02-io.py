r"""Write a Record to a ``*.mrec`` directory and read the contract brand back.

Scientific-record I/O belongs on ``molrs.io.mrec`` (spec mrec-format-03-python
moved ``Record.write`` / ``Record.read`` off the memory carrier). This script
writes a one-frame record whose directory name ends in ``.mrec``, reads
``format_name`` and ``record_schema_version`` as literals, and asserts that
``write_record`` did not also emit a sibling ``*.zarr.zip``.

Python has no public ``pack`` door in this cut. The archive suffix
``*.mrec.zip`` is owned by Rust ``molrs::io::mrec::pack`` (unit-tested in
``molrs/src/io/zarr/pack.rs``); this script does not invent a zip writer
and does not spawn cargo.

Provenance of the goldens: hand-written literals, no external oracle and no
third-party scientific package at run time (``molrs`` only). Runner:

    uv --directory molrs-python run python ../regressions/mrec-format-02-io.py

(any environment carrying a molrs wheel built against RECORD_FORMAT_NAME="mrec"
will do; 2026-08-30).
"""
from __future__ import annotations

import os
import tempfile
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import molrs

with tempfile.TemporaryDirectory() as tmp:
    store = os.path.join(tmp, "record.mrec")
    assert store.endswith(".mrec"), store

    record = molrs.Record()
    record.set_frame(molrs.Frame())
    molrs.io.mrec.write_record(store, record)

    loaded = molrs.io.mrec.read_record(store)
    meta = loaded.meta
    assert meta["format_name"] == "mrec", meta.get("format_name")
    assert meta["record_schema_version"] == 1, meta.get("record_schema_version")
    assert loaded.count_frames() == 1, loaded.count_frames()

    zarr_zips = [name for name in os.listdir(tmp) if name.endswith(".zarr.zip")]
    assert zarr_zips == [], f"unexpected .zarr.zip siblings: {zarr_zips}"
    assert os.path.isdir(store), store

print(
    "mrec-format-02-io ok: wrote record.mrec format_name=mrec "
    "record_schema_version=1 no sibling .zarr.zip"
)
