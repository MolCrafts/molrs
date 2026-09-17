r"""The ``molrs.io.mrec`` doors: write a frame and a system, read them back.

Scientific-record I/O belongs on ``molrs.io.mrec``. Both stores live in a
directory whose name ends in ``.mrec``, and both pin the dev-phase versioning
rule: writers stamp no ``molrec_version`` while the contract is in development,
and ``validate_meta`` accepts its absence. Neither constructs a Record.

Goldens are hand-written literals — no external oracle and no third-party
scientific package at run time (``molrs`` and ``numpy``, the latter only because
it is how molrs hands out columns). Runner:

    uv --directory molrs-python run python ../regressions/io_mrec.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

import molrs

ATOM_X = (0.0, 1.0, 0.5)
ATOM_Y = (0.25, 0.0, 2.0)
ATOM_Z = (0.0, 4.0, 0.125)


def _assert_unstamped(store: Path) -> None:
    """Dev-phase versioning: nothing is stamped.

    A pin on the writer, not a shrug — it fails the moment one starts stamping.
    """
    meta = molrs.io.mrec.read_meta(store)
    molrs.io.mrec.schema.validate_meta(meta)
    assert "molrec_version" not in meta, meta
    assert "format_name" not in meta, meta


with tempfile.TemporaryDirectory() as tmp:
    store = Path(tmp) / "frame.mrec"
    assert store.suffix == ".mrec", store

    molrs.io.mrec.write_frame(store, molrs.Frame())
    assert molrs.io.mrec.read_frame(store) is not None
    assert molrs.io.mrec.sections(store) == frozenset({"meta", "frame"})
    _assert_unstamped(store)

    siblings = [p.name for p in Path(tmp).iterdir() if p.name.endswith(".zarr.zip")]
    assert siblings == [], f"unexpected .zarr.zip siblings: {siblings}"
    assert store.is_dir(), store

with tempfile.TemporaryDirectory() as tmp:
    store = Path(tmp) / "system.mrec"

    atoms = molrs.Block()
    atoms["x"] = np.array(ATOM_X, dtype=np.float64)
    atoms["y"] = np.array(ATOM_Y, dtype=np.float64)
    atoms["z"] = np.array(ATOM_Z, dtype=np.float64)
    system = molrs.Frame()
    system["atoms"] = atoms

    molrs.io.mrec.write_system(store, system)
    got = molrs.io.mrec.read_system(store)["atoms"]
    assert got.nrows == len(ATOM_X), got.nrows
    np.testing.assert_array_equal(np.asarray(got["x"]), np.array(ATOM_X))
    np.testing.assert_array_equal(np.asarray(got["y"]), np.array(ATOM_Y))
    np.testing.assert_array_equal(np.asarray(got["z"]), np.array(ATOM_Z))
    _assert_unstamped(store)
    assert store.is_dir(), store

print("io_mrec ok: frame + system round-trip through *.mrec, meta unstamped")
