"""molrs.io readers return the canonical rich Frame (spec frame-block-sink-03).

Self-contained fixtures (molrs write → molrs read). No external corpus.
"""

from __future__ import annotations

import numpy as np

import molrs
import molrs.io as mio
from molrs.frame import Block as RichBlock, Frame as RichFrame


class TestSingleFrameReturnsRich:
    def test_read_pdb(self, water_pdb):
        assert isinstance(mio.read_pdb(str(water_pdb)), RichFrame)

    def test_read_xyz(self, water_xyz):
        assert isinstance(mio.read_xyz(str(water_xyz)), RichFrame)

    def test_read_gro(self, water_gro):
        frames = mio.read_gro(str(water_gro))
        assert isinstance(frames, list)
        assert all(isinstance(f, RichFrame) for f in frames)

    def test_read_lammps_data(self, water_lammps_data):
        f = mio.read_lammps_data(str(water_lammps_data))
        assert isinstance(f, RichFrame)


class TestRichApiPresent:
    def test_rich_surface(self, water_pdb):
        f = mio.read_pdb(str(water_pdb))
        assert isinstance(f.meta, dict)
        assert all(isinstance(b, RichBlock) for b in f.blocks)
        assert "blocks" in f.to_dict()
        assert isinstance(f["atoms"], RichBlock)


class TestCanonicalFields:
    def test_lammps_canonical_not_raw(self, water_lammps_data):
        f = mio.read_lammps_data(str(water_lammps_data))
        cols = set(f["atoms"].keys())
        assert {"x", "y", "z"} <= cols
        # raw LAMMPS keys must not leak through the facade
        assert "q" not in cols
        assert "mol" not in cols


class TestTrajectoryReturnsRich:
    def test_read_pdb_trajectory(self, water_pdb):
        frames = mio.read_pdb_trajectory(str(water_pdb))
        assert isinstance(frames, list)
        assert len(frames) >= 1
        assert all(isinstance(f, RichFrame) for f in frames)

    def test_read_xyz_trajectory(self, water_xyz):
        reader = mio.read_xyz_trajectory(str(water_xyz))
        frames = list(reader)
        assert len(frames) >= 1
        assert all(isinstance(f, RichFrame) for f in frames)

    def test_trajectory_reader_indexing(self, water_xyz):
        reader = mio.read_xyz_trajectory(str(water_xyz))
        assert isinstance(reader[0], RichFrame)
        assert all(isinstance(f, RichFrame) for f in reader.read_all())


class TestZeroCopyWrap:
    def test_wrap_shares_block_memory(self, water_pdb):
        bare = molrs.io.raw.read_pdb(str(water_pdb))
        mio._pdb_fmt.canonicalize_frame(bare)
        before = np.asarray(bare["atoms"].view("x"))
        rich = RichFrame.from_dict(bare)
        after = np.asarray(rich["atoms"].view("x"))
        assert np.shares_memory(before, after)

    def test_view_is_arc_backed(self, water_pdb):
        f = mio.read_pdb(str(water_pdb))
        v = np.asarray(f["atoms"].view("x"))
        assert v.base is not None


class TestUpgradeIdentity:
    def test_from_dict_on_rich_returns_equivalent_rich(self, water_pdb):
        rich = mio.read_pdb(str(water_pdb))
        again = RichFrame.from_dict(rich)
        assert isinstance(again, RichFrame)
        assert set(again.keys()) == set(rich.keys())
        np.testing.assert_array_equal(
            np.asarray(again["atoms"].view("x")),
            np.asarray(rich["atoms"].view("x")),
        )
