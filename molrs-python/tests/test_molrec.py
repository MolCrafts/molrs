"""FFI smoke tests for the MolRec record aggregate.

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


@pytest.fixture
def record_path(tmp_path: Path) -> Path:
    return tmp_path / "record.zarr"


class TestMolRecRoundtrip:
    def test_frame_only_record_round_trips(self, record_path: Path) -> None:
        record = molrs.MolRec()
        record.set_frame(molrs.Frame())
        record.write_zarr(str(record_path))

        loaded = molrs.MolRec.read_zarr(str(record_path))
        assert loaded.count_frames() == 1

    def test_meta_stamps_the_contract_keys(self, record_path: Path) -> None:
        record = molrs.MolRec()
        record.set_frame(molrs.Frame())
        record.write_zarr(str(record_path))

        meta = molrs.MolRec.read_zarr(str(record_path)).meta
        assert meta["record_schema_version"] == 1
        assert meta["format_name"] == "molrec"

    def test_nested_meta_and_method_round_trip(self, record_path: Path) -> None:
        record = molrs.MolRec()
        record.set_frame(molrs.Frame())
        record.meta = {"version": [0, 2], "creator": {"name": "pytest"}}
        record.method = {"type": "static_structure", "description": "smoke"}
        record.write_zarr(str(record_path))

        loaded = molrs.MolRec.read_zarr(str(record_path))
        assert loaded.meta["creator"]["name"] == "pytest"
        assert loaded.meta["version"] == [0, 2]
        assert loaded.method["type"] == "static_structure"

    def test_trajectory_section_round_trips(self, record_path: Path) -> None:
        record = molrs.MolRec()
        record.set_frame(molrs.Frame())
        record.add_frame(molrs.Frame())
        record.add_frame(molrs.Frame())
        record.write_zarr(str(record_path))

        loaded = molrs.MolRec.read_zarr(str(record_path))
        assert loaded.count_frames() == 2
        assert len(loaded.trajectory) == 2

    def test_record_without_a_state_section_is_refused(self, record_path: Path) -> None:
        with pytest.raises(Exception):
            molrs.MolRec().write_zarr(str(record_path))


class TestObservablesView:
    def test_add_mutates_the_owning_record(self, record_path: Path) -> None:
        record = molrs.MolRec()
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

        record.write_zarr(str(record_path))
        loaded = molrs.MolRec.read_zarr(str(record_path))
        assert loaded.observables.get("total_energy").kind == "scalar"

    def test_add_vector_returns_and_stores(self, record_path: Path) -> None:
        record = molrs.MolRec()
        record.set_frame(molrs.Frame())
        dipole = record.observables.add_vector(
            "dipole", np.array([[0.1, 0.2, 0.3]]), unit="D"
        )
        record.write_zarr(str(record_path))

        loaded = molrs.MolRec.read_zarr(str(record_path))
        assert loaded.observables.get("dipole").kind == dipole.kind == "vector"


class TestStubMatchesRuntime:
    """`_lib.pyi` once declared a `MolRec` no Rust pyclass implemented.

    A type stub that describes a class the runtime does not have passes every
    type check and fails at import time, so the stub itself needs a gate.
    """

    @staticmethod
    def _stub_path() -> Path:
        return Path(inspect.getfile(molrs)).parent / "_lib.pyi"

    def test_stub_is_syntactically_valid(self) -> None:
        """The stub was invalid Python for long enough to hide an orphan class.

        A `.pyi` no parser accepts silently type-checks nothing, so this is the
        gate the rest of the stub's guarantees rest on.
        """
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

    @pytest.mark.parametrize("class_name", ["MolRec", "Observables"])
    def test_every_declared_member_exists_at_runtime(self, class_name: str) -> None:
        runtime = getattr(molrs, class_name)
        missing = sorted(
            name for name in self._stub_members(class_name) if not hasattr(runtime, name)
        )
        assert not missing, f"{class_name} stub declares absent members: {missing}"

    def test_the_contract_forbids_a_record_root_parameters_section(self) -> None:
        assert "parameters" not in self._stub_members("MolRec")
        assert not hasattr(molrs.MolRec, "parameters")
