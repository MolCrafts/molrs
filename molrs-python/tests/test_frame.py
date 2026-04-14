import numpy as np
import pytest
import molrs


class TestFrameConstruction:
    def test_empty(self):
        f = molrs.Frame()
        assert len(f) == 0
        assert f.keys() == []
        assert f.simbox is None

    def test_repr_empty(self):
        r = repr(molrs.Frame())
        assert "Frame" in r
        assert "no" in r  # simbox=no


class TestFrameBlockAccess:
    def test_setitem_getitem(self):
        f = molrs.Frame()
        b = molrs.Block()
        b.insert("x", np.array([1.0, 2.0], dtype=np.float64))
        f["atoms"] = b
        assert "atoms" in f
        assert len(f) == 1

        atoms = f["atoms"]
        assert atoms.nrows == 2

    def test_getitem_returns_live_block_handle(self):
        f = molrs.Frame()
        b = molrs.Block()
        b.insert("x", np.array([1.0, 2.0], dtype=np.float64))
        f["atoms"] = b

        atoms = f["atoms"]
        atoms.insert("y", np.array([3.0, 4.0], dtype=np.float64))

        np.testing.assert_allclose(f["atoms"].view("y"), [3.0, 4.0])

    def test_getitem_missing_raises_key_error(self):
        f = molrs.Frame()
        with pytest.raises(KeyError):
            _ = f["missing"]

    def test_delitem(self):
        f = molrs.Frame()
        f["atoms"] = molrs.Block()
        del f["atoms"]
        assert "atoms" not in f

    def test_delitem_missing_raises_key_error(self):
        f = molrs.Frame()
        with pytest.raises(KeyError):
            del f["missing"]

    def test_contains(self):
        f = molrs.Frame()
        f["atoms"] = molrs.Block()
        assert "atoms" in f
        assert "bonds" not in f

    def test_keys(self):
        f = molrs.Frame()
        f["atoms"] = molrs.Block()
        f["bonds"] = molrs.Block()
        assert sorted(f.keys()) == ["atoms", "bonds"]

    def test_overwrite_block(self):
        f = molrs.Frame()
        b1 = molrs.Block()
        b1.insert("x", np.array([1.0], dtype=np.float64))
        f["atoms"] = b1
        assert f["atoms"].nrows == 1

        b2 = molrs.Block()
        b2.insert("x", np.array([1.0, 2.0], dtype=np.float64))
        f["atoms"] = b2
        assert f["atoms"].nrows == 2


class TestFrameSimbox:
    def test_default_none(self):
        assert molrs.Frame().simbox is None

    def test_set_simbox(self):
        f = molrs.Frame()
        box_ = molrs.Box.cube(10.0)
        f.simbox = box_
        assert f.simbox is not None
        assert pytest.approx(f.simbox.volume(), abs=1) == 1000.0

    def test_clear_simbox(self):
        f = molrs.Frame()
        f.simbox = molrs.Box.cube(10.0)
        f.simbox = None
        assert f.simbox is None

    def test_repr_with_simbox(self):
        f = molrs.Frame()
        f.simbox = molrs.Box.cube(10.0)
        assert "yes" in repr(f)


class TestFrameMeta:
    def test_set_and_get(self):
        f = molrs.Frame()
        f.meta = {"title": "test", "source": "pytest"}
        meta = f.meta
        assert meta["title"] == "test"
        assert meta["source"] == "pytest"

    def test_empty_meta(self):
        f = molrs.Frame()
        assert len(f.meta) == 0

    def test_overwrite_meta(self):
        f = molrs.Frame()
        f.meta = {"a": "1"}
        f.meta = {"b": "2"}
        assert "b" in f.meta
        assert "a" not in f.meta


class TestFrameValidation:
    def test_validate_empty(self):
        molrs.Frame().validate()

    def test_validate_consistent(self):
        f = molrs.Frame()
        b = molrs.Block()
        b.insert("x", np.array([1.0, 2.0, 3.0], dtype=np.float64))
        b.insert("y", np.array([0.0, 1.0, 2.0], dtype=np.float64))
        f["atoms"] = b
        f.validate()
