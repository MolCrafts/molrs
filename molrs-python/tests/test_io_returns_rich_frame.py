"""molrs.io readers return the canonical rich Frame (spec frame-block-sink-03).

Every ``molrs.io.read_*`` entry point yields the spec-01 rich Frame
(``molrs.frame.Frame``), never the bare PyO3 frame, with canonical field names
applied first and Block buffers shared (the wrap is a view, not a copy).
"""

import numpy as np

import molrs
import molrs.io as mio
from molrs.frame import Block as RichBlock, Frame as RichFrame


# --- ac-001: single-frame readers return rich Frame -------------------------


class TestSingleFrameReturnsRich:
    def test_read_pdb(self, pdb_dir):
        assert isinstance(mio.read_pdb(str(pdb_dir / "water.pdb")), RichFrame)

    def test_read_xyz(self, xyz_dir):
        assert isinstance(mio.read_xyz(str(xyz_dir / "methane.xyz")), RichFrame)

    def test_read_gro(self, gro_dir):
        frames = mio.read_gro(str(gro_dir / "ubiquitin.gro"))
        assert isinstance(frames, list)
        assert all(isinstance(f, RichFrame) for f in frames)

    def test_read_lammps_data(self, tests_data_dir):
        f = mio.read_lammps_data(str(tests_data_dir / "lammps-ff" / "peptide.data"))
        assert isinstance(f, RichFrame)


# --- ac-002: returned frame exposes the rich Frame API ----------------------


class TestRichApiPresent:
    def test_rich_surface(self, pdb_dir):
        f = mio.read_pdb(str(pdb_dir / "water.pdb"))
        # Rich surface: exact typed meta, blocks iterator, to_dict,
        # and __getitem__ returning a rich Block.
        assert isinstance(f.meta, dict)
        assert all(isinstance(b, RichBlock) for b in f.blocks)
        assert "blocks" in f.to_dict()
        assert isinstance(f["atoms"], RichBlock)


# --- ac-003: canonical field names preserved after wrap ---------------------


class TestCanonicalFields:
    def test_lammps_canonical_not_raw(self, tests_data_dir):
        f = mio.read_lammps_data(str(tests_data_dir / "lammps-data" / "molid.lmp"))
        cols = set(f["atoms"].keys())
        # canonical names present (charge, molecule_id, x/y/z) ...
        assert {"charge", "molecule_id", "x", "y", "z"} <= cols
        # ... and the raw LAMMPS keys are gone (canonicalize ran before wrap)
        assert "q" not in cols
        assert "mol" not in cols


# --- ac-004: trajectory readers yield rich Frames ---------------------------


class TestTrajectoryReturnsRich:
    def test_read_pdb_trajectory_multi_model(self, pdb_dir):
        frames = mio.read_pdb_trajectory(str(pdb_dir / "model.pdb"))
        assert isinstance(frames, list)
        assert len(frames) == 2  # model.pdb has two MODEL records
        assert all(isinstance(f, RichFrame) for f in frames)

    def test_read_pdb_trajectory_single_model(self, pdb_dir):
        # short-cryst1.pdb has no MODEL records -> one-element trajectory.
        frames = mio.read_pdb_trajectory(str(pdb_dir / "short-cryst1.pdb"))
        assert isinstance(frames, list)
        assert len(frames) == 1
        assert isinstance(frames[0], RichFrame)

    def test_read_xyz_trajectory(self, xyz_dir):
        reader = mio.read_xyz_trajectory(str(xyz_dir / "trajectory.xyz"))
        frames = list(reader)
        assert len(frames) >= 1
        assert all(isinstance(f, RichFrame) for f in frames)

    def test_trajectory_reader_indexing(self, xyz_dir):
        reader = mio.read_xyz_trajectory(str(xyz_dir / "trajectory.xyz"))
        assert isinstance(reader[0], RichFrame)
        assert all(isinstance(f, RichFrame) for f in reader.read_all())


# --- ac-005: wrapping preserves zero-copy of Block buffers ------------------


class TestZeroCopyWrap:
    def test_wrap_shares_block_memory(self, pdb_dir):
        # Mirror exactly what the facade does internally: read bare, canonicalize,
        # then wrap — and assert the wrap shares the pre-wrap buffer.
        bare = molrs.read_pdb(str(pdb_dir / "water.pdb"))
        mio._pdb_fmt.canonicalize_frame(bare)
        before = np.asarray(bare["atoms"].view("x"))
        rich = RichFrame.from_dict(bare)
        after = np.asarray(rich["atoms"].view("x"))
        assert np.shares_memory(before, after)

    def test_view_is_arc_backed(self, pdb_dir):
        f = mio.read_pdb(str(pdb_dir / "water.pdb"))
        v = np.asarray(f["atoms"].view("x"))
        assert v.base is not None  # window into Rust storage, not a fresh copy


# --- ac-006: base→subclass upgrade path is identity-on-rich -----------------


class TestUpgradeIdentity:
    def test_from_dict_on_rich_returns_equivalent_rich(self, pdb_dir):
        rich = mio.read_pdb(str(pdb_dir / "water.pdb"))
        again = RichFrame.from_dict(rich)
        assert isinstance(again, RichFrame)
        assert set(again.keys()) == set(rich.keys())
        np.testing.assert_array_equal(
            np.asarray(again["atoms"].view("x")),
            np.asarray(rich["atoms"].view("x")),
        )
