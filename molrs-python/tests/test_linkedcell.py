import numpy as np
import pytest
import molrs


class TestLinkedCellConstruction:
    def test_basic(self, cubic_box):
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
        lc = molrs.LinkedCell(pts, 2.0, cubic_box)
        assert "LinkedCell" in repr(lc)

    def test_bad_shape(self, cubic_box):
        with pytest.raises(ValueError, match="N,3"):
            molrs.LinkedCell(np.ones((3, 2), dtype=np.float32), 2.0, cubic_box)


class TestLinkedCellPairs:
    def test_close_pair_found(self, cubic_box):
        pts = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 5.0, 5.0]],
            dtype=np.float32,
        )
        lc = molrs.LinkedCell(pts, 2.0, cubic_box)
        pairs = lc.pairs()
        assert pairs.shape[1] == 2
        pair_set = {(int(p[0]), int(p[1])) for p in pairs}
        assert (0, 1) in pair_set

    def test_distant_pair_not_found(self, cubic_box):
        pts = np.array(
            [[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]],
            dtype=np.float32,
        )
        lc = molrs.LinkedCell(pts, 1.0, cubic_box)
        pairs = lc.pairs()
        assert len(pairs) == 0

    def test_pbc_wrapping(self):
        box_ = molrs.Box.cube(10.0)
        pts = np.array(
            [[0.1, 0.0, 0.0], [9.9, 0.0, 0.0]],
            dtype=np.float32,
        )
        lc = molrs.LinkedCell(pts, 1.0, box_)
        pairs = lc.pairs()
        assert len(pairs) == 1

    def test_single_point_no_pairs(self, cubic_box):
        pts = np.array([[5.0, 5.0, 5.0]], dtype=np.float32)
        lc = molrs.LinkedCell(pts, 2.0, cubic_box)
        assert len(lc.pairs()) == 0


class TestLinkedCellUpdate:
    def test_update_removes_pairs(self, cubic_box):
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
        lc = molrs.LinkedCell(pts, 2.0, cubic_box)
        assert len(lc.pairs()) == 1

        pts2 = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float32)
        lc.update(pts2, cubic_box)
        assert len(lc.pairs()) == 0

    def test_update_bad_shape(self, cubic_box):
        pts = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        lc = molrs.LinkedCell(pts, 2.0, cubic_box)
        with pytest.raises(ValueError, match="N,3"):
            lc.update(np.ones((3, 2), dtype=np.float32), cubic_box)
