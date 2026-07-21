import numpy as np
import pytest
import molrs
from molrs.molrs import Block, Frame, MetaValue  # bare PyO3 cores


class TestFrameConstruction:
    def test_empty(self):
        assert molrs.FRAME_SCHEMA_VERSION == 2
        f = Frame()
        assert len(f) == 0
        assert f.keys() == []
        assert f.box is None

    def test_from_dict_blocks_envelope(self):
        f = Frame.from_dict(
            {
                "blocks": {
                    "atoms": {
                        "symbol": ["C", "H"],
                        "x": np.array([0.0, 1.0], dtype=np.float64),
                    }
                },
                "meta": {"source": MetaValue("string", "pytest")},
            }
        )

        assert sorted(f.keys()) == ["atoms"]
        assert f["atoms"].nrows == 2
        assert list(f["atoms"].view("symbol")) == ["C", "H"]
        np.testing.assert_allclose(f["atoms"].view("x"), [0.0, 1.0])
        assert f.meta["source"].dtype == "string"
        assert f.meta["source"].value == "pytest"

    @pytest.mark.parametrize(
        "data",
        [
            {"blocks": {}},
            {"blocks": {}, "metadata": {}},
            {"blocks": {}, "meta": {}, "metadata": {}},
            {"atoms": {}},
        ],
    )
    def test_from_dict_rejects_noncanonical_envelopes(self, data):
        with pytest.raises(TypeError, match="exactly 'blocks' and 'meta'"):
            Frame.from_dict(data)

    def test_repr_empty(self):
        r = repr(Frame())
        assert "Frame" in r
        assert "no" in r  # box=no


class TestFrameBlockAccess:
    def test_setitem_getitem(self):
        f = Frame()
        b = Block()
        b.insert("x", np.array([1.0, 2.0], dtype=np.float64))
        f["atoms"] = b
        assert "atoms" in f
        assert len(f) == 1

        atoms = f["atoms"]
        assert atoms.nrows == 2

    def test_getitem_returns_live_block_handle(self):
        f = Frame()
        b = Block()
        b.insert("x", np.array([1.0, 2.0], dtype=np.float64))
        f["atoms"] = b

        atoms = f["atoms"]
        atoms.insert("y", np.array([3.0, 4.0], dtype=np.float64))

        np.testing.assert_allclose(f["atoms"].view("y"), [3.0, 4.0])

    def test_getitem_missing_raises_key_error(self):
        f = Frame()
        with pytest.raises(KeyError):
            _ = f["missing"]

    def test_delitem(self):
        f = Frame()
        f["atoms"] = Block()
        del f["atoms"]
        assert "atoms" not in f

    def test_delitem_missing_raises_key_error(self):
        f = Frame()
        with pytest.raises(KeyError):
            del f["missing"]

    def test_contains(self):
        f = Frame()
        f["atoms"] = Block()
        assert "atoms" in f
        assert "bonds" not in f

    def test_keys(self):
        f = Frame()
        f["atoms"] = Block()
        f["bonds"] = Block()
        assert sorted(f.keys()) == ["atoms", "bonds"]

    def test_overwrite_block(self):
        f = Frame()
        b1 = Block()
        b1.insert("x", np.array([1.0], dtype=np.float64))
        f["atoms"] = b1
        assert f["atoms"].nrows == 1

        b2 = Block()
        b2.insert("x", np.array([1.0, 2.0], dtype=np.float64))
        f["atoms"] = b2
        assert f["atoms"].nrows == 2


class TestFrameBox:
    def test_default_none(self):
        assert Frame().box is None

    def test_set_box(self):
        f = Frame()
        box_ = molrs.Box.cube(10.0)
        f.box = box_
        assert f.box is not None
        assert pytest.approx(f.box.volume(), abs=1) == 1000.0

    def test_clear_box(self):
        f = Frame()
        f.box = molrs.Box.cube(10.0)
        f.box = None
        assert f.box is None

    def test_repr_with_box(self):
        f = Frame()
        f.box = molrs.Box.cube(10.0)
        assert "yes" in repr(f)


class TestFrameMeta:
    def test_exact_dtype_roundtrip(self):
        f = Frame()
        f.meta = {
            "tag": MetaValue("i64", 9_007_199_254_740_993),
            "temperature": MetaValue("f32", 300.0),
            "stress": MetaValue("f64x6", [1, 2, 3, 4, 5, 6]),
        }
        assert f.meta["tag"].dtype == "i64"
        assert f.meta["tag"].value == 9_007_199_254_740_993
        assert f.meta["temperature"].dtype == "f32"
        assert f.meta["stress"].dtype == "f64x6"
        assert f.meta["stress"].value == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    def test_untyped_value_is_rejected(self):
        f = Frame()
        with pytest.raises(TypeError, match="must be a MetaValue"):
            f.meta = {"legacy": "string-only"}

    def test_set_and_get(self):
        f = Frame()
        f.meta = {
            "title": MetaValue("string", "test"),
            "source": MetaValue("string", "pytest"),
        }
        meta = f.meta
        assert meta["title"].value == "test"
        assert meta["source"].value == "pytest"

    def test_empty_meta(self):
        f = Frame()
        assert len(f.meta) == 0

    def test_overwrite_meta(self):
        f = Frame()
        f.meta = {"a": MetaValue("i64", 1)}
        f.meta = {"b": MetaValue("i64", 2)}
        assert "b" in f.meta
        assert "a" not in f.meta


class TestFrameValidation:
    def test_validate_empty(self):
        Frame().validate()

    def test_validate_consistent(self):
        f = Frame()
        b = Block()
        b.insert("x", np.array([1.0, 2.0, 3.0], dtype=np.float64))
        b.insert("y", np.array([0.0, 1.0, 2.0], dtype=np.float64))
        f["atoms"] = b
        f.validate()
