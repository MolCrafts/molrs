"""FFI smoke tests for DCD read/write bindings.

Self-contained: fixtures are written with ``molrs.io.raw.write_dcd``. No chemfiles
corpus and no third-party trajectory software.
"""

from __future__ import annotations

import os
import tempfile

import pytest

import molrs


class TestReadDcd:
    def test_parses_written_trajectory(self, water_dcd):
        frames = molrs.io.raw.read_dcd(str(water_dcd))
        assert len(frames) == 2
        for i, frame in enumerate(frames):
            assert "atoms" in frame, f"frame {i} missing atoms block"
            assert frame["atoms"].nrows == 3

    def test_missing_file_raises_os_error(self):
        with pytest.raises(OSError):
            molrs.io.raw.read_dcd("/nonexistent/path.dcd")


class TestDcdTrajReader:
    def test_random_access_matches_sequential(self, water_dcd):
        path = str(water_dcd)
        sequential = molrs.io.raw.read_dcd(path)
        reader = molrs.io.raw.DCDTrajReader(path)
        assert len(reader) == len(sequential)
        for n in reversed(range(len(sequential))):
            frame = reader[n]
            assert frame["atoms"].nrows == sequential[n]["atoms"].nrows

    def test_iteration(self, water_dcd):
        reader = molrs.io.raw.DCDTrajReader(str(water_dcd))
        n = len(reader)
        assert sum(1 for _ in reader) == n
        assert sum(1 for _ in reader) == n

    def test_negative_index(self, water_dcd):
        path = str(water_dcd)
        reader = molrs.io.raw.DCDTrajReader(path)
        eager = molrs.io.raw.read_dcd(path)
        assert reader[-1]["atoms"].nrows == eager[-1]["atoms"].nrows

    def test_index_error(self, water_dcd):
        reader = molrs.io.raw.DCDTrajReader(str(water_dcd))
        with pytest.raises(IndexError):
            _ = reader[10_000_000]


class TestDcdTrajReaderMolpyAligned:
    def test_n_frames_matches_len(self, water_dcd):
        reader = molrs.io.raw.DCDTrajReader(str(water_dcd))
        assert reader.n_frames == len(reader)

    def test_read_frame_matches_eager(self, water_dcd):
        path = str(water_dcd)
        eager = molrs.io.raw.read_dcd(path)
        reader = molrs.io.raw.DCDTrajReader(path)
        assert reader.read_frame(0)["atoms"].nrows == eager[0]["atoms"].nrows
        assert reader.read_frame(-1)["atoms"].nrows == eager[-1]["atoms"].nrows

    def test_read_frame_out_of_range_raises(self, water_dcd):
        reader = molrs.io.raw.DCDTrajReader(str(water_dcd))
        with pytest.raises(IndexError):
            reader.read_frame(10_000_000)

    def test_read_frames(self, water_dcd):
        reader = molrs.io.raw.DCDTrajReader(str(water_dcd))
        n = len(reader)
        frames = reader.read_frames([0, n - 1, -1])
        assert len(frames) == 3

    def test_read_all_matches_eager(self, water_dcd):
        path = str(water_dcd)
        eager = molrs.io.raw.read_dcd(path)
        reader = molrs.io.raw.DCDTrajReader(path)
        assert len(reader.read_all()) == len(eager)


class TestWriteDcd:
    def test_round_trip_atom_count(self, water_frame):
        with tempfile.NamedTemporaryFile(suffix=".dcd", delete=False) as tmp:
            tmpname = tmp.name
        try:
            molrs.io.raw.write_dcd(tmpname, [water_frame, water_frame])
            frames = molrs.io.raw.read_dcd(tmpname)
            assert len(frames) == 2
            assert frames[0]["atoms"].nrows == 3
        finally:
            os.unlink(tmpname)
