"""GROMACS TRR/XTC binding smoke tests.

Self-contained: trajectories are written with molrs, then read back.
No external corpus, no third-party MD packages.
"""

from __future__ import annotations

import numpy as np
import pytest

import molrs
import molrs.io as mio


class TestTopLevelEagerReaders:
    def test_read_trr_returns_list_of_frames(self, water_trr):
        frames = molrs.io.raw.read_trr(str(water_trr))
        assert isinstance(frames, list) and len(frames) >= 1
        atoms = frames[0]["atoms"]
        assert atoms.nrows > 0
        for axis in ("x", "y", "z"):
            assert np.all(np.isfinite(atoms.view(axis)))

    def test_read_xtc_returns_list_of_frames(self, water_xtc):
        frames = molrs.io.raw.read_xtc(str(water_xtc))
        assert isinstance(frames, list) and len(frames) >= 1
        assert frames[0]["atoms"].nrows > 0


class TestLazyFacadeReaders:
    def test_trr_returns_reader(self, water_trr):
        reader = mio.read_trr_trajectory(str(water_trr))
        assert isinstance(reader, mio.TrajectoryReader)
        assert reader.n_frames == len(reader) > 0

    def test_xtc_returns_reader(self, water_xtc):
        reader = mio.read_xtc_trajectory(str(water_xtc))
        assert isinstance(reader, mio.TrajectoryReader)
        assert reader.n_frames > 0

    def test_random_access_matches_sequential(self, water_trr):
        path = str(water_trr)
        reader = mio.read_trr_trajectory(path)
        eager = molrs.io.raw.read_trr(path)
        assert reader.n_frames == len(eager)
        last = reader.read_frame(-1)["atoms"].view("x")
        assert np.allclose(last, eager[-1]["atoms"].view("x"))

    def test_out_of_range_raises(self, water_xtc):
        reader = mio.read_xtc_trajectory(str(water_xtc))
        with pytest.raises(IndexError):
            reader.read_frame(10_000_000)

    def test_multi_file_concatenates(self, water_trr):
        path = str(water_trr)
        single = mio.read_trr_trajectory(path).n_frames
        doubled = mio.read_trr_trajectory([path, path])
        assert doubled.n_frames == 2 * single


class TestWriteRoundTrip:
    def test_trr_roundtrip_exact(self, water_trr, tmp_path):
        frames = molrs.io.raw.read_trr(str(water_trr))
        out = tmp_path / "out.trr"
        mio.write_trr(str(out), frames)
        back = molrs.io.raw.read_trr(str(out))
        assert len(back) == len(frames)
        for a, b in zip(frames, back):
            assert np.allclose(a["atoms"].view("x"), b["atoms"].view("x"), atol=1e-5)

    def test_xtc_roundtrip_within_precision(self, water_xtc, tmp_path):
        frames = molrs.io.raw.read_xtc(str(water_xtc))
        out = tmp_path / "out.xtc"
        mio.write_xtc(str(out), frames)
        back = molrs.io.raw.read_xtc(str(out))
        assert len(back) == len(frames)
        for a, b in zip(frames, back):
            # XTC is lossy at 1/precision (default 1000 → 1e-3 nm).
            assert np.allclose(a["atoms"].view("x"), b["atoms"].view("x"), atol=2e-3)
