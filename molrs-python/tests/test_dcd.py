import glob
import os
import tempfile

import pytest

import molrs

from conftest import _resolve_tests_data_dir

# Parametrize decorators are evaluated at collection time, before fixtures are
# available, so the DCD directory is resolved here using the same shared-data
# resolver that backs the ``tests_data_dir`` fixture.
DCD_DATA_DIR = str(_resolve_tests_data_dir() / "dcd")
LAMMPS_DATA_DIR = str(_resolve_tests_data_dir() / "lammps")

# Files excluded from the byte-faithful round-trip: a 4D-dynamics file (extra
# `w` column) and a fixed-atoms file (the reader expands to full NATOMS per
# frame, so the rewritten file won't mirror the original layout). Mirrors
# `writer_unsupported` in molrs-io/tests/test_io/test_dcd.rs.
ROUND_TRIP_EXCLUDED = {"4d-dynamic.dcd", "fixed-atoms.dcd"}

# Only the 4D-dynamics file is actually rejected by the writer (the `w`
# column is unsupported). `fixed-atoms.dcd`, once read, is ordinary and
# writes fine.
WRITER_REJECTS = {"4d-dynamic.dcd"}


def _all_dcd_files():
    files = sorted(glob.glob(os.path.join(DCD_DATA_DIR, "*.dcd")))
    if not files:
        # Called during collection (parametrize); skip the whole module cleanly
        # when the DCD test data is absent instead of erroring.
        pytest.skip(f"no DCD test data found in {DCD_DATA_DIR}", allow_module_level=True)
    return files


def _ids(paths):
    return [os.path.basename(p) for p in paths]


class TestReadDcd:
    @pytest.mark.parametrize("path", _all_dcd_files(), ids=_ids(_all_dcd_files()))
    def test_every_file_parses(self, path):
        frames = molrs.read_dcd(path)
        assert len(frames) > 0, f"{os.path.basename(path)}: no frames"
        for i, frame in enumerate(frames):
            assert "atoms" in frame, f"frame {i} missing atoms block"
            assert frame["atoms"].nrows > 0

    def test_missing_file_raises_os_error(self):
        with pytest.raises(OSError):
            molrs.read_dcd("/nonexistent/path.dcd")


class TestDcdTrajReader:
    @pytest.mark.parametrize("path", _all_dcd_files(), ids=_ids(_all_dcd_files()))
    def test_random_access_matches_sequential(self, path):
        sequential = molrs.read_dcd(path)
        reader = molrs.DCDTrajReader(path)
        assert len(reader) == len(sequential)
        # Random access (reverse order) must match the eager read.
        for n in reversed(range(len(sequential))):
            frame = reader[n]
            assert frame["atoms"].nrows == sequential[n]["atoms"].nrows

    def test_iteration(self):
        path = _all_dcd_files()[0]
        reader = molrs.DCDTrajReader(path)
        n = len(reader)
        count = sum(1 for _ in reader)
        assert count == n
        # Re-iteration works (cursor resets).
        assert sum(1 for _ in reader) == n

    def test_negative_index(self):
        path = _all_dcd_files()[0]
        reader = molrs.DCDTrajReader(path)
        eager = molrs.read_dcd(path)
        assert reader[-1]["atoms"].nrows == eager[-1]["atoms"].nrows

    def test_index_error(self):
        reader = molrs.DCDTrajReader(_all_dcd_files()[0])
        with pytest.raises(IndexError):
            _ = reader[10_000_000]


class TestDcdTrajReaderMolpyAligned:
    """The molpy `BaseTrajectoryReader`-style surface."""

    def test_n_frames_matches_len(self):
        reader = molrs.DCDTrajReader(_all_dcd_files()[0])
        assert reader.n_frames == len(reader)

    def test_read_frame_matches_eager(self):
        path = _all_dcd_files()[0]
        eager = molrs.read_dcd(path)
        reader = molrs.DCDTrajReader(path)
        assert reader.read_frame(0)["atoms"].nrows == eager[0]["atoms"].nrows
        assert reader.read_frame(-1)["atoms"].nrows == eager[-1]["atoms"].nrows

    def test_read_frame_out_of_range_raises(self):
        reader = molrs.DCDTrajReader(_all_dcd_files()[0])
        with pytest.raises(IndexError):
            reader.read_frame(10_000_000)

    def test_read_frames(self):
        path = _all_dcd_files()[0]
        reader = molrs.DCDTrajReader(path)
        n = len(reader)
        idx = [0, n - 1, -1]
        frames = reader.read_frames(idx)
        assert len(frames) == 3

    def test_read_all_matches_eager(self):
        path = _all_dcd_files()[0]
        eager = molrs.read_dcd(path)
        reader = molrs.DCDTrajReader(path)
        assert len(reader.read_all()) == len(eager)

    def test_read_range(self):
        path = _all_dcd_files()[0]
        reader = molrs.DCDTrajReader(path)
        n = len(reader)
        assert len(reader.read_range(0, n)) == n
        assert len(reader.read_range()) == n  # defaults span all frames

    def test_read_range_step_zero_raises(self):
        reader = molrs.DCDTrajReader(_all_dcd_files()[0])
        with pytest.raises(ValueError):
            reader.read_range(0, 5, 0)

    def test_slice_getitem(self):
        # Pick a multi-frame file so the slice is meaningful.
        path = next(
            (p for p in _all_dcd_files() if len(molrs.read_dcd(p)) > 2),
            _all_dcd_files()[0],
        )
        eager = molrs.read_dcd(path)
        reader = molrs.DCDTrajReader(path)
        sub = reader[0:2]
        assert isinstance(sub, list)
        assert len(sub) == 2
        assert sub[0]["atoms"].nrows == eager[0]["atoms"].nrows
        # Full reverse slice.
        rev = reader[::-1]
        assert len(rev) == len(eager)

    def test_context_manager_and_close(self):
        path = _all_dcd_files()[0]
        with molrs.DCDTrajReader(path) as reader:
            assert len(reader) > 0
        # After __exit__ the reader is closed.
        with pytest.raises(ValueError):
            _ = len(reader)

    def test_close_then_read_raises(self):
        reader = molrs.DCDTrajReader(_all_dcd_files()[0])
        reader.close()
        with pytest.raises(ValueError):
            reader.read_frame(0)


class TestLammpsTrajReaderAlignment:
    """The aligned surface is shared, so smoke-test it on LAMMPS too."""

    def _a_dump(self):
        files = sorted(glob.glob(os.path.join(LAMMPS_DATA_DIR, "*.lammpstrj")))
        if not files:
            pytest.skip(f"no LAMMPS dump test data in {LAMMPS_DATA_DIR}")
        return files[0]

    def test_aligned_surface(self):
        reader = molrs.LAMMPSTrajReader(self._a_dump())
        assert reader.n_frames == len(reader)
        assert reader.read_frame(0)["atoms"].nrows > 0
        assert len(reader.read_all()) == len(reader)
        with molrs.LAMMPSTrajReader(self._a_dump()) as r:
            assert r.read_frame(-1)["atoms"].nrows > 0


class TestWriteDcd:
    @pytest.mark.parametrize(
        "path",
        [p for p in _all_dcd_files() if os.path.basename(p) not in ROUND_TRIP_EXCLUDED],
        ids=[
            os.path.basename(p)
            for p in _all_dcd_files()
            if os.path.basename(p) not in ROUND_TRIP_EXCLUDED
        ],
    )
    def test_round_trip(self, path):
        frames = molrs.read_dcd(path)
        with tempfile.NamedTemporaryFile(suffix=".dcd", delete=False) as tmp:
            tmpname = tmp.name
        try:
            molrs.write_dcd(tmpname, frames)
            frames2 = molrs.read_dcd(tmpname)
            assert len(frames2) == len(frames)
            assert frames2[0]["atoms"].nrows == frames[0]["atoms"].nrows
        finally:
            os.unlink(tmpname)

    @pytest.mark.parametrize("name", sorted(WRITER_REJECTS))
    def test_writer_rejects_unsupported(self, name):
        path = os.path.join(DCD_DATA_DIR, name)
        if not os.path.exists(path):
            pytest.skip(f"{name} not present")
        frames = molrs.read_dcd(path)
        with tempfile.NamedTemporaryFile(suffix=".dcd", delete=False) as tmp:
            tmpname = tmp.name
        try:
            with pytest.raises(OSError):
                molrs.write_dcd(tmpname, frames)
        finally:
            os.unlink(tmpname)
