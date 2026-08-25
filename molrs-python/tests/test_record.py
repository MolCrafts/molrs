"""FFI smoke tests for the Record aggregate.

Depth (layout conformance, version rejection, preserve-the-unknown) lives in the
Rust unit tests; this file only proves the Python seam constructs, round-trips,
and exposes exactly the surface `_lib.pyi` declares.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
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
    "/io/store/zarr/",
)
_EXEMPT_URL = "https://github.com/MolCrafts/molrec"


@pytest.fixture
def record_path(tmp_path: Path) -> Path:
    return tmp_path / "record.zarr"


class TestRecordRoundtrip:
    def test_frame_only_record_round_trips(self, record_path: Path) -> None:
        record = molrs.Record()
        record.set_frame(molrs.Frame())
        record.write(str(record_path))

        loaded = molrs.Record.read(str(record_path))
        assert loaded.count_frames() == 1

    def test_meta_stamps_the_contract_keys(self, record_path: Path) -> None:
        record = molrs.Record()
        record.set_frame(molrs.Frame())
        record.write(str(record_path))

        meta = molrs.Record.read(str(record_path)).meta
        assert meta["record_schema_version"] == 1
        assert meta["format_name"] == "molrec"

    def test_nested_meta_and_method_round_trip(self, record_path: Path) -> None:
        record = molrs.Record()
        record.set_frame(molrs.Frame())
        record.meta = {"version": [0, 2], "creator": {"name": "pytest"}}
        record.method = {"type": "static_structure", "description": "smoke"}
        record.write(str(record_path))

        loaded = molrs.Record.read(str(record_path))
        assert loaded.meta["creator"]["name"] == "pytest"
        assert loaded.meta["version"] == [0, 2]
        assert loaded.method["type"] == "static_structure"

    def test_trajectory_section_round_trips(self, record_path: Path) -> None:
        record = molrs.Record()
        record.set_frame(molrs.Frame())
        record.add_frame(molrs.Frame())
        record.add_frame(molrs.Frame())
        record.write(str(record_path))

        loaded = molrs.Record.read(str(record_path))
        assert loaded.count_frames() == 2
        assert len(loaded.trajectory) == 2

    def test_record_without_a_state_section_is_refused(self, record_path: Path) -> None:
        with pytest.raises(Exception):
            molrs.Record().write(str(record_path))


class TestObservablesView:
    def test_add_mutates_the_owning_record(self, record_path: Path) -> None:
        record = molrs.Record()
        record.set_frame(molrs.Frame())
        record.observables.add(
            molrs.ScalarObservable(
                "total_energy",
                np.array([1.0, 1.5, 2.0]),
                unit="eV",
                axes=["timestep"],
                time_dependent=True,
            )
        )
        # The getter returns a live view, not a detached copy.
        assert "total_energy" in record.observables

        record.write(str(record_path))
        loaded = molrs.Record.read(str(record_path))
        assert loaded.observables.get("total_energy").kind == "scalar"

    def test_add_vector_returns_and_stores(self, record_path: Path) -> None:
        record = molrs.Record()
        record.set_frame(molrs.Frame())
        dipole = record.observables.add_vector(
            "dipole", np.array([[0.1, 0.2, 0.3]]), unit="D"
        )
        record.write(str(record_path))

        loaded = molrs.Record.read(str(record_path))
        assert loaded.observables.get("dipole").kind == dipole.kind == "vector"


class TestStubMatchesRuntime:
    """`_lib.pyi` once declared a class no Rust pyclass implemented."""

    @staticmethod
    def _stub_path() -> Path:
        return Path(inspect.getfile(molrs)).parent / "_lib.pyi"

    def test_stub_is_syntactically_valid(self) -> None:
        ast.parse(self._stub_path().read_text())

    @classmethod
    def _stub_members(cls, class_name: str) -> set[str]:
        tree = ast.parse(cls._stub_path().read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                return {
                    child.name
                    for child in node.body
                    if isinstance(child, ast.FunctionDef)
                    and not child.name.startswith("_")
                }
        raise AssertionError(f"_lib.pyi declares no class {class_name}")

    @pytest.mark.parametrize("class_name", ["Record", "Observables"])
    def test_every_declared_member_exists_at_runtime(self, class_name: str) -> None:
        runtime = getattr(molrs, class_name)
        missing = sorted(
            name for name in self._stub_members(class_name) if not hasattr(runtime, name)
        )
        assert not missing, f"{class_name} stub declares absent members: {missing}"

    def test_the_contract_forbids_a_record_root_parameters_section(self) -> None:
        assert "parameters" not in self._stub_members("Record")
        assert not hasattr(molrs.Record, "parameters")

    def test_old_public_names_are_gone(self) -> None:
        assert not hasattr(molrs, "MolRec")
        assert not hasattr(molrs.Record, "read_zarr")
        assert not hasattr(molrs.Record, "write_zarr")
        assert not hasattr(molrs.Trajectory, "read_zarr")
        assert not hasattr(molrs.Trajectory, "write_zarr")


class TestPublicSurfaceNaming:
    """Public identifiers name the object, not the storage technology.

    Exempt: the engine type in core/store/record.rs, the io::store::zarr
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
