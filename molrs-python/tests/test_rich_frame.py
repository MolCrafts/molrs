"""Rich Frame/Block Python layer (molrs.frame) — pandas-style API on the PyO3 core.

These cover the spec ``frame-block-sink-molrs-01-richcore`` acceptance criteria
(ac-001 .. ac-010). The rich types live at ``molrs.frame.Block`` /
``molrs.frame.Frame`` and are NOT yet shadowed onto the top-level
``molrs.Block`` / ``molrs.Frame`` — that cutover lands with the molpy adoption
(chain spec 04), so the bare-core tests in test_block.py / test_frame.py stay
valid alongside these.
"""

from io import StringIO

import numpy as np
import pytest

import molrs
from molrs.frame import Block, Frame


# --- ac-001: full dict-like mapping protocol --------------------------------


class TestBlockMappingProtocol:
    def _blk(self) -> Block:
        return Block({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})

    def test_len_iter_contains_keys(self):
        b = self._blk()
        assert len(b) == 2
        assert sorted(b.keys()) == ["x", "y"]
        assert sorted(iter(b)) == ["x", "y"]
        assert "x" in b and "z" not in b

    def test_get(self):
        b = self._blk()
        np.testing.assert_allclose(b.get("x"), [1.0, 2.0, 3.0])
        assert b.get("missing") is None
        assert b.get("missing", 42) == 42

    def test_items_values(self):
        b = self._blk()
        items = dict(b.items())
        assert set(items) == {"x", "y"}
        assert len(list(b.values())) == 2

    def test_update_pop_setdefault_clear_popitem(self):
        b = self._blk()
        b.update({"z": np.array([7.0, 8.0, 9.0])})
        assert "z" in b
        popped = b.pop("z")
        np.testing.assert_allclose(popped, [7.0, 8.0, 9.0])
        assert "z" not in b
        # setdefault inserts when absent, returns existing otherwise
        b.setdefault("w", np.array([0.0, 0.0, 0.0]))
        assert "w" in b
        b.clear()
        assert len(b) == 0


# --- ac-002: selector dispatch matches molpy semantics ----------------------


class TestBlockSelector:
    def _blk(self) -> Block:
        return Block(
            {
                "x": [1.0, 2.0, 3.0],
                "y": [4.0, 5.0, 6.0],
                "id": np.array([10, 20, 30], dtype=np.int64),
            }
        )

    def test_str_returns_column_ndarray(self):
        b = self._blk()
        col = b["x"]
        assert isinstance(col, np.ndarray)
        np.testing.assert_allclose(col, [1.0, 2.0, 3.0])

    def test_int_returns_single_row_dict(self):
        b = self._blk()
        row = b[0]
        assert isinstance(row, dict)
        assert row["id"] == 10
        np.testing.assert_allclose(row["x"], 1.0)

    def test_slice_returns_sub_block(self):
        b = self._blk()
        sub = b[0:2]
        assert isinstance(sub, Block)
        assert sub.nrows == 2

    def test_bool_mask_returns_filtered_sub_block(self):
        b = self._blk()
        sub = b[np.array([True, False, True])]
        assert isinstance(sub, Block)
        assert sub.nrows == 2
        np.testing.assert_allclose(sub["x"], [1.0, 3.0])

    def test_list_str_returns_2d_array(self):
        b = self._blk()
        arr = b[["x", "y"]]
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3, 2)

    def test_callable_selector(self):
        b = self._blk()
        out = b[lambda blk: blk["x"].sum()]
        assert out == pytest.approx(6.0)


# --- ac-003: list[str] error contracts --------------------------------------


class TestBlockListIndexErrors:
    def test_dtype_mismatch_raises_value_error(self):
        b = Block({"a": np.array([1.0, 2.0]), "b": np.array([1, 2], dtype=np.int64)})
        with pytest.raises(ValueError):
            _ = b[["a", "b"]]

    def test_shape_mismatch_raises_value_error(self):
        b = Block(
            {"a": np.zeros((2, 3)), "b": np.zeros((2, 2))}
        )
        with pytest.raises(ValueError):
            _ = b[["a", "b"]]

    def test_construction_length_mismatch_surfaces_real_error(self):
        # A per-column length mismatch must report the real, actionable cause
        # (differing row counts), NOT the misleading "must be array-like"
        # blanket — the column value IS array-like. Regression: the ValueError
        # message previously masked the length/shape error with
        # "Value must be array-like ... got numpy.ndarray".
        with pytest.raises(ValueError) as exc:
            Block({"x": np.zeros(6), "y": np.zeros(4)})
        assert "array-like" not in str(exc.value)

    def test_construction_non_array_like_still_reports_array_like(self):
        # A genuinely non-array-like scalar keeps the friendly guidance.
        with pytest.raises(ValueError) as exc:
            Block({"x": object()})
        assert "array-like" in str(exc.value)

    def test_missing_key_raises_key_error(self):
        b = Block({"a": [1.0, 2.0]})
        with pytest.raises(KeyError):
            _ = b[["a", "missing"]]

    def test_empty_list_raises_key_error(self):
        b = Block({"a": [1.0, 2.0]})
        with pytest.raises(KeyError):
            _ = b[[]]


# --- ac-004: sort / sort_ ---------------------------------------------------


class TestBlockSort:
    def test_sort_returns_new_block_original_unchanged(self):
        b = Block({"x": [3.0, 1.0, 2.0], "y": [30.0, 10.0, 20.0]})
        s = b.sort("x")
        assert s is not b
        np.testing.assert_allclose(s["x"], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(s["y"], [10.0, 20.0, 30.0])
        np.testing.assert_allclose(b["x"], [3.0, 1.0, 2.0])  # original intact

    def test_sort_reverse(self):
        b = Block({"x": [1.0, 2.0, 3.0]})
        s = b.sort("x", reverse=True)
        np.testing.assert_allclose(s["x"], [3.0, 2.0, 1.0])

    def test_sort_inplace_returns_self(self):
        b = Block({"x": [3.0, 1.0, 2.0]})
        out = b.sort_("x")
        assert out is b
        np.testing.assert_allclose(b["x"], [1.0, 2.0, 3.0])


# --- ac-005: iterrows / itertuples ------------------------------------------


class TestBlockIteration:
    def test_iterrows(self):
        b = Block({"x": [1.0, 2.0], "id": np.array([10, 20], dtype=np.int64)})
        rows = list(b.iterrows())
        assert rows[0][0] == 0
        assert rows[0][1]["id"] == 10
        assert rows[1][1]["x"] == pytest.approx(2.0)

    def test_itertuples(self):
        b = Block({"x": [1.0, 2.0], "id": np.array([10, 20], dtype=np.int64)})
        tup = list(b.itertuples())
        assert tup[0].id == 10
        assert tup[1].x == pytest.approx(2.0)

    def test_empty_block_yields_nothing(self):
        assert list(Block().iterrows()) == []
        assert list(Block().itertuples()) == []


# --- ac-006: copy/shape/nrows/to_dict/from_dict/rename/view -----------------


class TestBlockShapeAndSerialization:
    def test_copy_is_independent(self):
        b = Block({"x": [1.0, 2.0, 3.0]})
        c = b.copy()
        c["x"] = np.array([9.0, 9.0, 9.0])
        np.testing.assert_allclose(b["x"], [1.0, 2.0, 3.0])

    def test_shape_and_nrows(self):
        b = Block({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
        assert b.nrows == 3
        assert b.shape == (3, 2)

    def test_empty_shape(self):
        assert Block().shape == ()
        assert Block().nrows == 0

    def test_to_dict_from_dict_round_trip(self):
        b = Block({"x": [1.0, 2.0], "id": np.array([10, 20], dtype=np.int64)})
        d = b.to_dict()
        b2 = Block.from_dict(d)
        np.testing.assert_allclose(b2["x"], [1.0, 2.0])
        np.testing.assert_array_equal(b2["id"], [10, 20])

    def test_rename(self):
        b = Block({"old": [1.0, 2.0]})
        b.rename("old", "new")
        assert "new" in b and "old" not in b

    def test_rename_missing_raises(self):
        b = Block({"x": [1.0]})
        with pytest.raises(KeyError):
            b.rename("missing", "new")

    def test_view_returns_column(self):
        b = Block({"x": [1.0, 2.0, 3.0]})
        v = b.view("x")
        assert isinstance(v, np.ndarray)
        np.testing.assert_allclose(v, [1.0, 2.0, 3.0])


# --- ac-007: to_csv / from_csv round-trip (Rust core) -----------------------


class TestBlockCsv:
    def test_headered_round_trip(self):
        src = Block(
            {
                "x": [1.0, 2.0],
                "id": np.array([10, 20], dtype=np.int64),
                "name": ["p", "q"],
            }
        )
        text = src.to_csv()
        rt = Block.from_csv(StringIO(text))
        np.testing.assert_allclose(rt["x"], [1.0, 2.0])
        np.testing.assert_array_equal(rt["id"], [10, 20])
        assert list(rt["name"]) == ["p", "q"]

    def test_dtype_inference(self):
        rt = Block.from_csv(StringIO("a,b,c\n1,1.5,x\n2,2.5,y\n"))
        assert str(rt["a"].dtype).startswith("int")
        assert str(rt["b"].dtype).startswith("float")
        assert list(rt["c"]) == ["x", "y"]

    def test_headerless_with_names(self):
        rt = Block.from_csv(StringIO("1,2\n3,4\n"), header=["a", "b"])
        np.testing.assert_array_equal(rt["a"], [1, 3])
        np.testing.assert_array_equal(rt["b"], [2, 4])

    def test_empty_csv_raises_value_error(self):
        with pytest.raises(ValueError):
            Block.from_csv(StringIO(""))

    def test_to_csv_no_header(self):
        b = Block({"x": np.array([1, 2], dtype=np.int64)})
        text = b.to_csv(header=False)
        assert "x" not in text.splitlines()[0]

    def test_to_csv_writes_file(self, tmp_path):
        b = Block({"x": [1.0, 2.0]})
        path = tmp_path / "out.csv"
        assert b.to_csv(path) is None
        rt = Block.from_csv(path)
        np.testing.assert_allclose(rt["x"], [1.0, 2.0])


# --- ac-008: rich Frame surface ---------------------------------------------


class TestRichFrame:
    def test_getitem_returns_rich_block(self):
        f = Frame({"atoms": {"x": [1.0, 2.0]}})
        atoms = f["atoms"]
        assert isinstance(atoms, Block)
        np.testing.assert_allclose(atoms["x"], [1.0, 2.0])

    def test_setitem_accepts_dict_and_block(self):
        f = Frame()
        f["atoms"] = {"x": [1.0, 2.0]}
        f["bonds"] = Block({"i": np.array([0], dtype=np.int64)})
        assert "atoms" in f and "bonds" in f
        assert len(f) == 2

    def test_box_round_trip(self):
        f = Frame()
        f.box = molrs.Box.cube(10.0)
        assert f.box is not None
        assert f.box.volume() == pytest.approx(1000.0, abs=1.0)

    def test_blocks_iterates_rich_blocks(self):
        f = Frame({"atoms": {"x": [1.0]}, "bonds": {"i": np.array([0], dtype=np.int64)}})
        blks = list(f.blocks)
        assert len(blks) == 2
        assert all(isinstance(b, Block) for b in blks)

    def test_metadata(self):
        f = Frame(
            {"atoms": {"x": [1.0]}},
            meta={"title": molrs.MetaValue("string", "t")},
        )
        assert f.meta["title"].value == "t"
        f.meta = {**f.meta, "step": molrs.MetaValue("i64", 5)}
        assert f.meta["step"].value == 5

    def test_to_dict_from_dict(self):
        f = Frame(
            {"atoms": {"x": [1.0, 2.0]}},
            meta={"title": molrs.MetaValue("string", "t")},
        )
        d = f.to_dict()
        assert "blocks" in d and "meta" in d
        f2 = Frame.from_dict(d)
        np.testing.assert_allclose(f2["atoms"]["x"], [1.0, 2.0])

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
        with pytest.raises(ValueError, match="exactly 'blocks' and 'meta'"):
            Frame.from_dict(data)

    def test_copy_is_independent(self):
        f = Frame({"atoms": {"x": [1.0, 2.0]}})
        f.box = molrs.Box.cube(5.0)
        f2 = f.copy()
        f2["atoms"]["x"] = np.array([9.0, 9.0])
        np.testing.assert_allclose(f["atoms"]["x"], [1.0, 2.0])
        assert f2.box is not None

    def test_keys_contains_len_delitem(self):
        f = Frame({"atoms": {"x": [1.0]}, "bonds": {"i": np.array([0], dtype=np.int64)}})
        assert sorted(f.keys()) == ["atoms", "bonds"]
        del f["bonds"]
        assert "bonds" not in f and len(f) == 1


# --- ac-009: zero-copy Arc-backed views -------------------------------------


class TestZeroCopy:
    def test_numeric_view_shares_core_memory(self):
        b = Block({"pos": np.arange(9.0).reshape(3, 3)})
        v = b.view("pos")
        # Arc-backed view: numpy array is a window into Rust storage, never a
        # per-access copy (it carries a non-None base owner).
        assert v.base is not None

    def test_frame_block_write_through(self):
        # frame[key] returns a live alias; mutating it is visible on re-read.
        f = Frame()
        f["atoms"] = {"x": [1.0, 2.0]}
        f["atoms"]["y"] = np.array([3.0, 4.0])
        np.testing.assert_allclose(f["atoms"]["y"], [3.0, 4.0])


# --- ac-010: parity with molpy core/frame.py reference ----------------------


class TestMolpyParity:
    """The rich types reproduce the documented molpy core/frame.py behavior."""

    def test_selector_parity(self):
        b = Block({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
        # str -> 1D column, list[str] -> 2D stack, int -> row dict, slice -> Block
        assert b["x"].ndim == 1
        assert b[["x", "y"]].shape == (3, 2)
        assert isinstance(b[1], dict) and b[1]["x"] == pytest.approx(2.0)
        assert isinstance(b[1:3], Block)

    def test_sort_parity(self):
        b = Block({"x": [3.0, 1.0, 2.0], "y": [3.0, 1.0, 2.0]})
        s = b.sort("x")
        np.testing.assert_allclose(s["x"], s["y"])  # rows move together

    def test_iterrows_parity(self):
        b = Block({"x": [1.0, 2.0]})
        assert [i for i, _ in b.iterrows()] == [0, 1]

    def test_to_dict_parity(self):
        b = Block({"x": [1.0, 2.0]})
        assert set(b.to_dict()) == {"x"}


def test_setitem_scalar_column_raises():
    """A scalar (0-D) column is rejected fail-fast, not silently dropped.

    The Rust Store only holds >=1-D columns. Previously Block.__setitem__
    silently ``return``ed on a 0-D array, so ``block["q"] = 1.0`` quietly did
    nothing; now it raises so the mistake surfaces.
    """
    b = Block({"x": np.array([1.0, 2.0])})
    with pytest.raises(ValueError, match="at least 1-D"):
        b["q"] = 1.0
    # the column was not created
    assert "q" not in b
