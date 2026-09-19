"""The streaming ``*.mrec`` surface: declared schemas, landing cadence,
carry-forward semantics, column projection, reopen, pack."""

from __future__ import annotations

import json
from pathlib import Path

import molrs
import numpy as np
import pytest
from molrs.io import mrec


def _frame(n: int, seed: int, bonds: bool = True) -> molrs.Frame:
    rng = np.random.default_rng(seed)
    pos = rng.random((n, 3)) * 10.0
    blocks = {"atoms": {"x": pos[:, 0], "y": pos[:, 1], "z": pos[:, 2]}}
    if bonds:
        i = np.arange(max(n - 1, 0), dtype=np.uint64)
        blocks["bonds"] = {"atomi": i, "atomj": i + 1}
    frame = molrs.Frame(blocks)
    frame.box = molrs.Box.cube(10.0)
    return frame


class TestStore:
    def test_the_store_is_a_record_with_a_root_and_meta(self, tmp_path: Path) -> None:
        path = tmp_path / "run.mrec"
        with mrec.TrajectoryWriter(path, mrec.SequenceSchema.from_frame(_frame(4, 0))) as w:
            w.append(_frame(4, 0))
        assert (path / "zarr.json").is_file()
        assert mrec.sections(path) == frozenset({"meta", "trajectory"})
        assert mrec.read_meta(path) == {"molrec_version": mrec.schema.MOLREC_VERSION}

    def test_meta_is_written_when_handed_in(self, tmp_path: Path) -> None:
        path = tmp_path / "run.mrec"
        schema = mrec.SequenceSchema.from_frame(_frame(2, 0))
        with mrec.TrajectoryWriter(path, schema, meta={"creator": {"name": "test"}}) as w:
            w.append(_frame(2, 0))
        assert mrec.read_meta(path) == {
            "creator": {"name": "test"},
            "molrec_version": mrec.schema.MOLREC_VERSION,
        }

    def test_every_array_is_sharded_with_the_index_at_the_start(self, tmp_path: Path) -> None:
        path = tmp_path / "run.mrec"
        frames = [_frame(4, 0), _frame(5, 1), _frame(6, 2)]  # ragged: the CSR index exists
        with mrec.TrajectoryWriter(path, mrec.SequenceSchema.from_frames(frames)) as w:
            w.append(frames[0], step=0)
            w.append(frames[1], step=1)
            w.append(frames[2], step=7)  # 0, 1, 7 is no progression: `step` is an array
        for name in ("step", "atoms/x", "atoms/offset", "atoms/step_index", "bonds/atomi"):
            meta = json.loads((path / "trajectory" / name / "zarr.json").read_text())
            (sharding,) = [c for c in meta["codecs"] if c["name"] == "sharding_indexed"]
            assert sharding["configuration"]["index_location"] == "start", name
            inner = [c["name"] for c in sharding["configuration"]["codecs"]]
            assert inner[-1] == "crc32c", name
        x = json.loads((path / "trajectory/atoms/x/zarr.json").read_text())
        assert [c["name"] for c in x["codecs"][0]["configuration"]["codecs"]] == [
            "bytes",
            "crc32c",
        ]

    def test_a_regular_run_costs_one_array_per_column_and_nothing_else(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "run.mrec"
        with mrec.TrajectoryWriter(path, mrec.SequenceSchema.from_frame(_frame(4, 0))) as w:
            for k in range(6):
                w.append(_frame(4, k), step=10 * k, time=0.5 * k)
        files = sorted(str(p.relative_to(path)) for p in path.rglob("*") if p.is_file())
        # Six group documents (root, meta, trajectory, atoms, bonds, box) +
        # five columns x 2 files: no step / time arrays, no index arrays, and
        # the fixed cell is the box group's attributes, not arrays.
        assert len(files) == 16, files
        attrs = json.loads((path / "trajectory/zarr.json").read_text())["attributes"]
        assert attrs["nstep"] == 6
        assert attrs["step_progression"] == {"start": 0, "stride": 10}
        assert attrs["time_progression"] == {"start": 0.0, "stride": 0.5}
        box = json.loads((path / "trajectory/box/zarr.json").read_text())["attributes"]
        assert box["vectors"][0][0] == 10.0
        atoms = json.loads((path / "trajectory/atoms/zarr.json").read_text())["attributes"]
        assert atoms == {"uniform_rows": 4, "dense_updates": True}
        r = mrec.TrajectoryReader(path)
        assert r.step == [0, 10, 20, 30, 40, 50]
        assert r.time == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
        assert r.read_frame(5)["atoms"].nrows == 4
        assert r.box_at(5) is not None


class TestSemantics:
    def test_an_omitted_block_carries_forward_and_an_empty_one_is_present(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "run.mrec"
        first = _frame(3, 0)
        schema = mrec.SequenceSchema.from_frame(first)
        with mrec.TrajectoryWriter(path, schema, flush_every=1) as w:
            w.append(first)
            w.append(_frame(3, 1, bonds=False))  # omitted: carry forward
            empty = _frame(3, 2, bonds=False)
            empty["bonds"] = {
                "atomi": np.array([], dtype=np.uint64),
                "atomj": np.array([], dtype=np.uint64),
            }
            w.append(empty)  # zero rows: present and empty
        r = mrec.TrajectoryReader(path)
        assert r.read_frame(1)["bonds"].nrows == 2
        assert r.read_frame(2)["bonds"].nrows == 0
        assert set(r.read_frame(2)["bonds"].keys()) == {"atomi", "atomj"}
        assert r.block_update_at("bonds", 0) == 0
        assert r.block_update_at("bonds", 1) == 0
        assert r.block_update_at("bonds", 2) == 1

    def test_a_block_before_its_first_update_is_absent(self, tmp_path: Path) -> None:
        path = tmp_path / "run.mrec"
        frames = [_frame(3, 0, bonds=False), _frame(3, 1)]
        with mrec.TrajectoryWriter(path, mrec.SequenceSchema.from_frames(frames)) as w:
            for f in frames:
                w.append(f)
        r = mrec.TrajectoryReader(path)
        assert "bonds" not in list(r.read_frame(0).keys())
        assert r.block_update_at("bonds", 0) is None
        assert r.block_update_at("bonds", 1) == 0


class TestWriter:
    def test_the_cadence_is_derived_and_overridable(self, tmp_path: Path) -> None:
        schema = mrec.SequenceSchema.from_frame(_frame(1000, 0, bonds=False))
        w = mrec.TrajectoryWriter(tmp_path / "a.mrec", schema)
        w.append(_frame(1000, 0, bonds=False))
        # 4 MiB / 24 000 B = 174, rounded up to whole 3-frame chunks.
        assert w.flush_every == 174
        assert w.committed == 0
        w.close()

        w = mrec.TrajectoryWriter(tmp_path / "b.mrec", schema, flush_every=2)
        for k in range(5):
            w.append(_frame(1000, k, bonds=False))
        assert w.committed == 4
        w.close()
        assert len(mrec.TrajectoryReader(tmp_path / "b.mrec")) == 5

    def test_time_without_step_keeps_automatic_numbering(self, tmp_path: Path) -> None:
        path = tmp_path / "run.mrec"
        schema = mrec.SequenceSchema.from_frame(_frame(2, 0))
        with mrec.TrajectoryWriter(path, schema) as w:
            w.append(_frame(2, 0), time=0.5)
            w.append(_frame(2, 1), time=1.5)
        r = mrec.TrajectoryReader(path)
        assert r.step == [0, 1]
        assert r.time == [0.5, 1.5]

    def test_reopen_continues_after_the_last_committed_frame(self, tmp_path: Path) -> None:
        path = tmp_path / "run.mrec"
        schema = mrec.SequenceSchema.from_frame(_frame(2, 0))
        with mrec.TrajectoryWriter(path, schema) as w:
            w.append(_frame(2, 0), step=10)
        with mrec.TrajectoryWriter.open(path) as w:
            assert w.committed == 1
            w.append(_frame(2, 1), step=20)
        r = mrec.TrajectoryReader(path)
        assert r.step == [10, 20]
        assert r.read_frame(1)["atoms"].nrows == 2

    def test_compression_is_a_named_choice(self, tmp_path: Path) -> None:
        schema = mrec.SequenceSchema.from_frame(_frame(2, 0, bonds=False))
        path = tmp_path / "gz.mrec"
        with mrec.TrajectoryWriter(path, schema, compression="gzip:3") as w:
            w.append(_frame(2, 0, bonds=False))
        x = json.loads((path / "trajectory/atoms/x/zarr.json").read_text())
        inner = x["codecs"][0]["configuration"]["codecs"]
        assert [c["name"] for c in inner] == ["bytes", "gzip", "crc32c"]
        assert inner[1]["configuration"]["level"] == 3
        with pytest.raises(ValueError, match="compression"):
            mrec.TrajectoryWriter(tmp_path / "bad.mrec", schema, compression="lzma")


class TestSchema:
    def test_a_declared_schema_writes_the_same_store_as_a_derived_one(self, tmp_path: Path) -> None:
        frame = _frame(3, 0)
        declared = (
            mrec.SequenceSchema()
            .declare_block("atoms", rows=3)
            .declare_column("atoms", "x", "f64")
            .declare_column("atoms", "y", "f64")
            .declare_column("atoms", "z", "f64")
            .declare_column("bonds", "atomi", "u64")
            .declare_column("bonds", "atomj", "u64")
        )
        assert declared.block_names() == ["atoms", "bonds"]
        assert declared.column_names("atoms") == ["x", "y", "z"]
        path = tmp_path / "declared.mrec"
        with mrec.TrajectoryWriter(path, declared) as w:
            w.append(frame)
        pinned = json.loads((path / "trajectory/zarr.json").read_text())["attributes"]
        assert set(pinned["sequence_schema"]["blocks"]) == {"atoms", "bonds"}
        assert pinned["sequence_schema"]["blocks"]["atoms"]["columns"]["x"]["dtype"] == "f64"
        with pytest.raises(ValueError, match="reserved"):
            mrec.SequenceSchema().declare_column("step", "a", "f64")
        with pytest.raises(ValueError, match="f128"):
            mrec.SequenceSchema().declare_column("atoms", "x", "f128")

    def test_declared_meta_with_fill_lands_the_fill(self, tmp_path: Path) -> None:
        schema = mrec.SequenceSchema.from_frame(_frame(2, 0, bonds=False))
        schema.declare_meta_with_fill("temp", 300.0)
        path = tmp_path / "run.mrec"
        with mrec.TrajectoryWriter(path, schema) as w:
            w.append(_frame(2, 0, bonds=False))
        r = mrec.TrajectoryReader(path)
        assert r.read_frame(0).meta["temp"] == 300.0
        assert schema.meta_keys() == [("temp", "f64")]


class TestReader:
    def test_read_columns_decodes_only_what_is_named(self, tmp_path: Path) -> None:
        path = tmp_path / "run.mrec"
        with mrec.TrajectoryWriter(path, mrec.SequenceSchema.from_frame(_frame(4, 0))) as w:
            w.append(_frame(4, 0))
        r = mrec.TrajectoryReader(path)
        frame = r.read_columns(0, [("atoms", "x")])
        assert list(frame["atoms"].keys()) == ["x"]
        assert "bonds" not in list(frame.keys())
        assert frame.box is not None
        with pytest.raises(ValueError, match="nope"):
            r.read_columns(0, [("atoms", "nope")])

    def test_box_at_resolves_the_cell(self, tmp_path: Path) -> None:
        path = tmp_path / "run.mrec"
        with mrec.TrajectoryWriter(path, mrec.SequenceSchema.from_frame(_frame(2, 0))) as w:
            w.append(_frame(2, 0))
            w.append(_frame(2, 1))
        r = mrec.TrajectoryReader(path)
        box = r.box_at(1)
        assert box is not None
        assert np.allclose(np.diag(box.matrix), 10.0)

    def test_a_packed_store_reads_back_identically(self, tmp_path: Path) -> None:
        path = tmp_path / "run.mrec"
        with mrec.TrajectoryWriter(path, mrec.SequenceSchema.from_frame(_frame(5, 0))) as w:
            for k in range(3):
                w.append(_frame(5, k))
        before = mrec.TrajectoryReader(path).read_frame(2)["atoms"]["x"].copy()
        archive = mrec.pack(path)
        assert archive.endswith(".mrec.zip") and not path.exists()
        r = mrec.TrajectoryReader(archive)
        assert len(r) == 3
        np.testing.assert_array_equal(r.read_frame(2)["atoms"]["x"], before)
