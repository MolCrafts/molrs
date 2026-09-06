r"""Write a system through ``molrs.io.mrec`` and read its meta back.

Scientific-record I/O belongs on ``molrs.io.mrec``. This script writes a
one-system store to a directory whose name ends in ``.mrec``, reads it back
with ``read_system``, and pins the dev-phase versioning rule: writers stamp no
``molrec_version`` while the contract is in development
(`.claude/notes/notes.md`, 2026-09-02). It never constructs a Record.

Provenance of the goldens: hand-written literals, no external oracle and no
third-party scientific package at run time (``molrs`` + ``numpy`` only, numpy
being how molrs hands out columns). Runner:

    uv --directory molrs-python run python ../regressions/mrec-format-03-python.py

(any environment carrying a molrs wheel that exposes ``molrs.io.mrec`` will do;
2026-08-31).
"""
from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

import molrs

ATOM_X = (0.0, 1.0, 0.5)
ATOM_Y = (0.25, 0.0, 2.0)
ATOM_Z = (0.0, 4.0, 0.125)
N_ATOMS = 3

with tempfile.TemporaryDirectory() as tmp:
    store = Path(tmp) / "record.mrec"
    assert store.suffix == ".mrec", store

    atoms = molrs.Block()
    atoms["x"] = np.array(ATOM_X, dtype=np.float64)
    atoms["y"] = np.array(ATOM_Y, dtype=np.float64)
    atoms["z"] = np.array(ATOM_Z, dtype=np.float64)
    system = molrs.Frame()
    system["atoms"] = atoms

    molrs.io.mrec.write_system(store, system)
    loaded = molrs.io.mrec.read_system(store)
    got = loaded["atoms"]
    assert got.nrows == N_ATOMS, got.nrows
    np.testing.assert_array_equal(np.asarray(got["x"]), np.array(ATOM_X, dtype=np.float64))
    np.testing.assert_array_equal(np.asarray(got["y"]), np.array(ATOM_Y, dtype=np.float64))
    np.testing.assert_array_equal(np.asarray(got["z"]), np.array(ATOM_Z, dtype=np.float64))

    meta = molrs.io.mrec.read_meta(store)
    molrs.io.mrec.schema.validate_meta(meta)
    # Dev-phase versioning: nothing is stamped. This is a pin on the writer,
    # not a shrug — it fails the moment a writer starts stamping again.
    assert "molrec_version" not in meta, meta
    assert "format_name" not in meta, meta
    assert store.is_dir(), store

print(
    "mrec-format-03-python ok: write_system/read_system record.mrec "
    "with unstamped meta"
)
