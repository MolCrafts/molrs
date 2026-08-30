r"""Write a Record through ``molrs.io.mrec`` and read the contract brand back.

Scientific-record I/O belongs on ``molrs.io.mrec`` (spec mrec-format-03-python),
not on the memory carriers. This script writes a one-system record to a
directory whose name ends in ``.mrec``, reads it back with ``read_record``, and
asserts ``format_name`` as the literal ``"mrec"``. It never calls
``Record.read``, ``Record.write``, ``Trajectory.read``, or ``Trajectory.write``.

Provenance of the goldens: hand-written literals, no external oracle and no
third-party scientific package at run time (``molrs`` + ``numpy`` only, numpy
being how molrs hands out columns). Runner:

    uv --directory molrs-python run python ../regressions/mrec-format-03-python.py

(any environment carrying a molrs wheel that exposes ``molrs.io.mrec`` will do;
2026-08-30).
"""
from __future__ import annotations

import os
import tempfile
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

import molrs

# Å; dyadic so a bit-exact f64 round-trip is the golden, not a tolerance.
ATOM_X = (0.0, 1.0, 0.5)
ATOM_Y = (0.25, 0.0, 2.0)
ATOM_Z = (0.0, 4.0, 0.125)
N_ATOMS = 3

with tempfile.TemporaryDirectory() as tmp:
    store = os.path.join(tmp, "record.mrec")
    assert store.endswith(".mrec"), store

    atoms = molrs.Block()
    atoms["x"] = np.array(ATOM_X, dtype=np.float64)
    atoms["y"] = np.array(ATOM_Y, dtype=np.float64)
    atoms["z"] = np.array(ATOM_Z, dtype=np.float64)
    system = molrs.Frame()
    system["atoms"] = atoms

    record = molrs.Record()
    record.set_system(system)
    record.meta = {"creator": {"name": "mrec-format-03-python"}}
    molrs.io.mrec.write_record(store, record)

    loaded = molrs.io.mrec.read_record(store)
    meta = loaded.meta
    assert meta["format_name"] == "mrec", meta.get("format_name")
    assert meta["record_schema_version"] == 1, meta.get("record_schema_version")
    assert loaded.system is not None, "system section missing after read_record"
    got = loaded.system["atoms"]
    assert got.nrows == N_ATOMS, got.nrows
    np.testing.assert_array_equal(np.asarray(got["x"]), np.array(ATOM_X, dtype=np.float64))
    np.testing.assert_array_equal(np.asarray(got["y"]), np.array(ATOM_Y, dtype=np.float64))
    np.testing.assert_array_equal(np.asarray(got["z"]), np.array(ATOM_Z, dtype=np.float64))
    assert loaded.meta["creator"]["name"] == "mrec-format-03-python"
    assert os.path.isdir(store), store

print(
    "mrec-format-03-python ok: write_record/read_record record.mrec "
    "format_name=mrec record_schema_version=1"
)
