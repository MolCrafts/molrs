r"""Write a frame to a ``*.mrec`` directory and read the version key back.

Scientific-record I/O belongs on ``molrs.io.mrec``. This script writes a
one-frame store whose directory name ends in ``.mrec``, reads it back with
``read_frame``, and asserts ``molrec_version`` as the literal ``1`` from
``molrs.io.mrec.schema``. It never constructs a Record.

Provenance of the goldens: hand-written literals, no external oracle and no
third-party scientific package at run time (``molrs`` only). Runner:

    uv --directory molrs-python run python ../regressions/mrec-format-02-io.py

(any environment carrying a molrs wheel that exposes ``molrs.io.mrec`` will do;
2026-08-31).
"""
from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

import molrs

with tempfile.TemporaryDirectory() as tmp:
    store = Path(tmp) / "record.mrec"
    assert store.suffix == ".mrec", store

    molrs.io.mrec.write_frame(store, molrs.Frame())
    loaded = molrs.io.mrec.read_frame(store)
    assert loaded is not None
    assert molrs.io.mrec.sections(store) == frozenset({"meta", "frame"})

    meta = molrs.io.mrec.read_meta(store)
    molrs.io.mrec.schema.validate_meta(meta)
    assert meta["molrec_version"] == molrs.io.mrec.schema.MOLREC_VERSION
    assert "format_name" not in meta, meta

    zarr_zips = [p.name for p in Path(tmp).iterdir() if p.name.endswith(".zarr.zip")]
    assert zarr_zips == [], f"unexpected .zarr.zip siblings: {zarr_zips}"
    assert store.is_dir(), store

print(
    "mrec-format-02-io ok: wrote record.mrec molrec_version=1 "
    "no sibling .zarr.zip"
)
