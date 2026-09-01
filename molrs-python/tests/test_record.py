"""FFI smoke tests for the mrec primitive doors and public-surface naming.

The Record aggregate is not a Python type. Depth (layout conformance, version
rejection) lives in the Rust unit tests.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import molrs

_REPO = Path(__file__).resolve().parents[2]
_PUBLIC_TREES = (
    _REPO / "molrs-python" / "python" / "molrs",
    _REPO / "molrs-python" / "src",
    _REPO / "molrs-cxxapi" / "src",
    _REPO / "molrs-wasm" / "src",
    _REPO / "molrs-capi" / "src",
    _REPO / "molrs-ffi" / "src",
)
_EXEMPT_SUFFIXES = (
    "/core/store/record.rs",
    "/io/zarr/",
)
_EXEMPT_URL = "https://github.com/MolCrafts/molrec"


class TestRecordIsGone:
    def test_record_is_not_on_the_package(self) -> None:
        assert not hasattr(molrs, "Record")
        assert not hasattr(molrs, "MolRec")
        assert not hasattr(molrs, "Observables")

    def test_stub_does_not_declare_record(self) -> None:
        stub = (Path(inspect.getfile(molrs)).parent / "_lib.pyi").read_text()
        tree = ast.parse(stub)
        names = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
        assert "Record" not in names
        assert "Observables" not in names
        assert "MolRec" not in names

    def test_old_public_names_are_gone(self) -> None:
        assert not hasattr(molrs.Trajectory, "read_zarr")
        assert not hasattr(molrs.Trajectory, "write_zarr")
        assert not hasattr(molrs.Trajectory, "read")
        assert not hasattr(molrs.Trajectory, "write")


class TestPublicSurfaceNaming:
    """Public identifiers name the object, not the storage technology.

    Exempt: the engine type in core/store/record.rs, the io::zarr
    adapter path, and the molrec contract URL.
    """

    def test_public_trees_do_not_spell_molrec_or_zarr_as_api_names(self) -> None:
        import re

        public_ident = re.compile(
            r'name\s*=\s*"MolRec"'
            r"|class\s+MolRec\b"
            r"|def\s+(read_zarr|write_zarr)\b"
            r"|fn\s+(read_zarr|write_zarr|write_frame_zarr|read_frame_zarr_first)\b"
            r"|MolRecReader\b"
        )
        hits: list[str] = []
        for root in _PUBLIC_TREES:
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                if path.suffix not in {".rs", ".py", ".pyi"}:
                    continue
                posix = path.as_posix()
                if any(s in posix for s in _EXEMPT_SUFFIXES):
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
                for i, line in enumerate(text.splitlines(), 1):
                    if _EXEMPT_URL in line:
                        continue
                    if public_ident.search(line):
                        hits.append(f"{path.relative_to(_REPO)}:{i}:{line.strip()}")
        assert not hits, "public surface still spells MolRec/zarr:\n" + "\n".join(hits)


class TestDumpConcatenatorUnchanged:
    """``molrs.io.TrajectoryReader`` stays the LAMMPS/XYZ/DCD concatenator."""

    def test_io_trajectory_reader_constructs_from_native_readers(self) -> None:
        from molrs.io import TrajectoryReader

        params = inspect.signature(TrajectoryReader.__init__).parameters
        assert "readers" in params
        assert "path" not in params
        assert hasattr(TrajectoryReader, "read_frame")
        assert hasattr(TrajectoryReader, "n_frames")
