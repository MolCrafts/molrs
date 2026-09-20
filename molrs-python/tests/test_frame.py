from collections.abc import MutableMapping

import numpy as np
import pytest
import molrs
from molrs._lib import Block, Frame, MetaValue  # bare PyO3 cores


class TestFrameConstruction:
    def test_empty(self):
        assert molrs.FRAME_SCHEMA_VERSION == 2
        f = Frame()
        assert len(f) == 0
        assert f.keys() == []
        assert f.box is None

    def test_setitem_populates_blocks_and_meta(self):
        f = Frame()
        atoms = Block()
        atoms.insert("symbol", ["C", "H"])
        atoms.insert("x", np.array([0.0, 1.0], dtype=np.float64))
        f["atoms"] = atoms
        f.meta = {"source": MetaValue("string", "pytest")}

        assert sorted(f.keys()) == ["atoms"]
        assert f["atoms"].nrows == 2
        assert list(f["atoms"].view("symbol")) == ["C", "H"]
        np.testing.assert_allclose(f["atoms"].view("x"), [0.0, 1.0])
        assert f.meta["source"] == "pytest"

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
        assert f.meta["tag"] == 9_007_199_254_740_993
        assert f.meta["temperature"] == pytest.approx(300.0)
        assert f.meta["stress"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    def test_json_document_values_are_accepted(self):
        f = Frame()
        f.meta = {"legacy": "string-only", "nested": {"tool": "molrec", "run": 3}}
        assert f.meta["legacy"] == "string-only"
        assert f.meta["nested"] == {"tool": "molrec", "run": 3}
        assert f.meta.dtype("nested") == "json"

    def test_set_and_get(self):
        f = Frame()
        f.meta = {"title": "test", "source": "pytest"}
        meta = f.meta
        assert meta["title"] == "test"
        assert meta["source"] == "pytest"
        assert meta == {"title": "test", "source": "pytest"}

    def test_empty_meta(self):
        f = Frame()
        assert len(f.meta) == 0
        assert dict(f.meta) == {}

    def test_overwrite_meta(self):
        f = Frame()
        f.meta = {"a": 1}
        f.meta = {"b": 2}
        assert "b" in f.meta
        assert "a" not in f.meta

    def test_update_and_get(self):
        f = Frame()
        f.meta.update({"a": 1})
        f.meta.update(b="x")
        assert f.meta.get("a") == 1
        assert f.meta.get("missing") is None
        assert f.meta.get("missing", 9) == 9

    def test_unsupported_value_is_rejected(self):
        f = Frame()
        with pytest.raises(TypeError, match="not JSON-serializable"):
            f.meta["bad"] = object()

    def test_none_is_a_json_null_not_a_rejection(self):
        # `meta` is a JSON document, and JSON has null. Rejecting a bare None
        # while accepting {"a": None} would be an arbitrary split.
        f = Frame()
        f.meta["absent"] = None
        assert f.meta["absent"] is None
        assert f.meta.dtype("absent") == "json"

    def test_write_through(self):
        f = Frame()
        f.meta["title"] = "water"
        assert f.meta["title"] == "water"
        f.meta["title"] = "ice"
        assert f.meta["title"] == "ice"
        del f.meta["title"]
        assert "title" not in f.meta

    def test_existing_key_keeps_its_dtype(self):
        f = Frame()
        f.meta["temperature"] = MetaValue("f32", 300.0)
        assert f.meta.dtype("temperature") == "f32"
        f.meta["temperature"] = 310.0
        assert f.meta.dtype("temperature") == "f32"
        assert f.meta["temperature"] == pytest.approx(310.0)

    def test_reassigning_a_read_value_is_an_identity(self):
        f = Frame()
        f.meta = {
            "tag": MetaValue("i64", 9_007_199_254_740_993),
            "stress": MetaValue("f64x6", [1, 2, 3, 4, 5, 6]),
        }
        before = {k: f.meta.dtype(k) for k in f.meta}
        for key in list(f.meta):
            f.meta[key] = f.meta[key]
        assert {k: f.meta.dtype(k) for k in f.meta} == before
        assert f.meta["tag"] == 9_007_199_254_740_993

    def test_value_that_does_not_fit_the_slot_is_refused(self):
        f = Frame()
        f.meta["count"] = MetaValue("i64", 3)
        with pytest.raises(TypeError, match="is i64"):
            f.meta["count"] = 1.5

    def test_mapping_protocol(self):
        f = Frame()
        f.meta = {"a": 1, "b": "two"}
        assert isinstance(f.meta, MutableMapping)
        assert sorted(f.meta) == ["a", "b"]
        assert dict(f.meta) == {"a": 1, "b": "two"}
        assert f.meta == {"a": 1, "b": "two"}
        assert f.meta.pop("a") == 1
        f.meta.setdefault("c", 3)
        assert f.meta["c"] == 3
        f.meta |= {"d": 4}
        assert f.meta["d"] == 4
        f.meta.clear()
        assert len(f.meta) == 0

    def test_copying_a_frame_keeps_exact_dtypes(self):
        # dict(meta) drops the tags, so the copy path must not go through it.
        f = Frame()
        f.meta = {"temperature": MetaValue("f32", 300.0)}
        rich = molrs.Frame(f)
        assert rich.meta.dtype("temperature") == "f32"
        assert rich.meta.copy() == {"temperature": pytest.approx(300.0)}
        assert set(rich.meta.typed()) == {"temperature"}
        assert rich.meta.typed()["temperature"].dtype == "f32"

    def test_nested_document_is_a_snapshot(self):
        f = Frame()
        f.meta["run"] = {"step": 1}
        f.meta["run"]["step"] = 2
        assert f.meta["run"] == {"step": 1}, "in-place nested edits do not persist"
        document = f.meta["run"]
        document["step"] = 2
        f.meta["run"] = document
        assert f.meta["run"] == {"step": 2}


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
