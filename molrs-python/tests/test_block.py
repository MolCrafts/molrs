import numpy as np
import pytest
import molrs
from molrs._lib import Block, Frame  # bare PyO3 cores (Block/Frame are the rich shadow)


class TestBlockConstruction:
    def test_empty(self):
        b = Block()
        assert b.nrows is None
        assert len(b) == 0
        assert b.keys() == []

    def test_repr_empty(self):
        b = Block()
        assert "Block" in repr(b)
        assert "None" in repr(b)


class TestBlockInsert:
    def test_f32(self):
        b = Block()
        b.insert("x", np.array([1.0, 2.0, 3.0], dtype=np.float64))
        assert b.nrows == 3
        assert len(b) == 1
        assert "x" in b

    def test_f64(self):
        b = Block()
        b.insert("x", np.array([1.0, 2.0], dtype=np.float64))
        assert b.nrows == 2

    def test_i64(self):
        b = Block()
        b.insert("id", np.array([10, 20, 30], dtype=np.uint32))
        assert b.nrows == 3

    def test_bool(self):
        b = Block()
        b.insert("mask", np.array([True, False, True]))
        assert b.nrows == 3

    def test_u32(self):
        b = Block()
        b.insert("idx", np.array([0, 1, 2], dtype=np.uint32))
        assert b.nrows == 3

    def test_2d_array(self):
        b = Block()
        b.insert("pos", np.zeros((5, 3), dtype=np.float64))
        assert b.nrows == 5

    def test_nrows_enforcement(self):
        b = Block()
        b.insert("x", np.array([1.0, 2.0], dtype=np.float64))
        with pytest.raises(ValueError):
            b.insert("y", np.array([1.0, 2.0, 3.0], dtype=np.float64))

    def test_int32_accepted(self):
        b = Block()
        b.insert("count", np.array([1, 2], dtype=np.int32))
        assert b.dtype("count") == "int"

    def test_overwrite_key(self):
        b = Block()
        b.insert("x", np.array([1.0, 2.0], dtype=np.float64))
        b.insert("x", np.array([3.0, 4.0], dtype=np.float64))
        result = b.view("x").flatten()
        np.testing.assert_allclose(result, [3.0, 4.0])


class TestBlockGet:
    def test_roundtrip_f32(self):
        b = Block()
        original = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        b.insert("x", original)
        result = b.view("x")
        np.testing.assert_allclose(result.flatten(), original, atol=1e-6)

    def test_roundtrip_f64(self):
        b = Block()
        original = np.array([1.1, 2.2], dtype=np.float64)
        b.insert("x", original)
        result = b.view("x")
        np.testing.assert_allclose(result.flatten(), original, atol=1e-12)

    def test_roundtrip_i64(self):
        b = Block()
        b.insert("id", np.array([10, 20], dtype=np.uint32))
        result = b.view("id")
        np.testing.assert_array_equal(result.flatten(), [10, 20])

    def test_roundtrip_bool(self):
        b = Block()
        b.insert("m", np.array([True, False]))
        result = b.view("m")
        np.testing.assert_array_equal(result.flatten(), [True, False])

    def test_roundtrip_u32(self):
        b = Block()
        b.insert("idx", np.array([0, 42], dtype=np.uint32))
        result = b.view("idx")
        np.testing.assert_array_equal(result.flatten(), [0, 42])

    def test_missing_key_raises_key_error(self):
        b = Block()
        with pytest.raises(KeyError):
            b.view("nonexistent")

    def test_roundtrip_2d(self):
        b = Block()
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        b.insert("pos", data)
        result = b.view("pos")
        np.testing.assert_allclose(result, data)

    def test_view_returns_numpy_array_with_native_owner(self):
        b = Block()
        b.insert("x", np.array([1.0, 2.0, 3.0], dtype=np.float64))
        view = b.view("x")
        assert isinstance(view, np.ndarray)
        assert view.base is not None
        np.testing.assert_allclose(view, [1.0, 2.0, 3.0])

    def test_get_uses_view_backing_for_numeric_columns(self):
        b = Block()
        b.insert("x", np.array([4.0, 5.0], dtype=np.float64))
        result = b.view("x")
        assert isinstance(result, np.ndarray)
        assert result.base is not None
        np.testing.assert_allclose(result, [4.0, 5.0])


class TestBlockOperations:
    def test_keys(self):
        b = Block()
        b.insert("x", np.array([1.0], dtype=np.float64))
        b.insert("y", np.array([2.0], dtype=np.float64))
        keys = sorted(b.keys())
        assert keys == ["x", "y"]

    def test_contains(self):
        b = Block()
        b.insert("x", np.array([1.0], dtype=np.float64))
        assert "x" in b
        assert "y" not in b

    def test_remove(self):
        b = Block()
        b.insert("x", np.array([1.0], dtype=np.float64))
        b.remove("x")
        assert "x" not in b
        assert len(b) == 0

    def test_remove_missing_raises_key_error(self):
        b = Block()
        with pytest.raises(KeyError):
            b.remove("nonexistent")

    def test_dtype(self):
        b = Block()
        b.insert("x", np.array([1.0], dtype=np.float64))
        b.insert("id", np.array([1], dtype=np.uint32))
        assert b.dtype("x") == "float"
        assert b.dtype("id") == "uint"

    def test_dtype_missing_raises_key_error(self):
        b = Block()
        with pytest.raises(KeyError):
            b.dtype("missing")

    def test_has_f64_not_f32(self):
        b = molrs.Block()
        b.insert("x", np.array([1.0], dtype=np.float64))
        b.insert("id", np.array([1], dtype=np.uint32))
        assert b.has_f64("x")
        assert not b.has_f32("x")
        assert not b.has_f64("id")
        assert not b.has_f64("missing")
        np.testing.assert_array_equal(b.get_f64("x"), np.array([1.0]))
        with pytest.raises(TypeError, match="must be f64"):
            b.get_f64("id")
        with pytest.raises(KeyError, match="f64"):
            b.get_f64("missing")
        np.testing.assert_array_equal(
            b.get_f32("missing", np.array([9.0], dtype=np.float32)),
            np.array([9.0], dtype=np.float32),
        )

    def test_frame_has_f64(self):
        f = molrs.Frame()
        f["atoms"] = molrs.Block()
        f["atoms"]["x"] = np.array([1.0, 2.0], dtype=np.float64)
        assert f.has_f64("atoms", "x")
        assert not f.has_f32("atoms", "x")
        np.testing.assert_array_equal(f.get_f64("atoms", "x"), [1.0, 2.0])

    def test_repr(self):
        b = Block()
        b.insert("x", np.array([1.0, 2.0], dtype=np.float64))
        r = repr(b)
        assert "Block" in r
        assert "2" in r


class TestBlockSubscriptAssignment:
    """``block[key] = array`` is the subscript form of ``insert``.

    A store you can read with ``[]`` but must write with a named method is half
    a mapping. These pin that the two spellings are one operation, not two
    code paths that can drift.
    """

    def test_setitem_stores_a_column(self):
        b = Block()
        b["x"] = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        assert np.array_equal(b["x"], np.array([1.0, 2.0, 3.0]))
        assert "x" in b

    def test_setitem_matches_insert(self):
        by_setitem, by_insert = Block(), Block()
        # `id` is declared uint by the Frame schema; honour it in both paths.
        column = np.array([4, 5, 6], dtype=np.uint32)
        by_setitem["id"] = column
        by_insert.insert("id", column)
        assert by_setitem.keys() == by_insert.keys()
        assert np.array_equal(by_setitem["id"], by_insert["id"])
        assert by_setitem.dtype("id") == by_insert.dtype("id")

    def test_setitem_is_checked_against_the_frame_schema(self):
        """The subscript form does not bypass the declared dtype."""
        b = Block()
        with pytest.raises(ValueError, match="schema"):
            b["id"] = np.array([1, 2], dtype=np.int64)  # schema says uint

    def test_setitem_stores_a_string_column(self):
        b = Block()
        b["name"] = ["C", "H", "O"]
        assert list(b["name"]) == ["C", "H", "O"]

    def test_setitem_replaces_an_existing_column(self):
        b = Block()
        b["x"] = np.zeros(3, dtype=np.float64)
        b["x"] = np.ones(3, dtype=np.float64)
        assert np.array_equal(b["x"], np.ones(3))
        assert len(b) == 1

    def test_setitem_enforces_the_row_count(self):
        b = Block()
        b["x"] = np.zeros(3, dtype=np.float64)
        with pytest.raises(ValueError):
            b["y"] = np.zeros(2, dtype=np.float64)

    def test_setitem_rejects_an_object_column(self):
        # The numpy-only store contract holds through the subscript form too.
        b = Block()
        with pytest.raises((TypeError, ValueError)):
            b["bad"] = np.array([object(), object()], dtype=object)

    def test_roundtrip_through_all_three_dunders(self):
        b = Block()
        b["x"] = np.array([1.0], dtype=np.float64)
        assert "x" in b
        del b["x"]
        assert "x" not in b


class TestBlockMultiColumnIndexing:
    """``block["x", "y", "z"]`` — several equal-shaped columns, side by side."""

    @staticmethod
    def _xyz() -> molrs.Block:
        return molrs.Block(
            {
                "x": np.array([0.0, 1.0], dtype=np.float64),
                "y": np.array([2.0, 3.0], dtype=np.float64),
                "z": np.array([4.0, 5.0], dtype=np.float64),
                "id": np.array([1, 2], dtype=np.uint32),
            }
        )

    def test_tuple_and_list_keys_agree(self):
        # Both spellings are the same request, so both give (nrows, len(key)).
        # A caller reading coordinates must not have to remember which bracket
        # form transposes the result.
        block = self._xyz()
        np.testing.assert_array_equal(block["x", "y", "z"], block[["x", "y", "z"]])

    def test_columns_are_stacked_one_per_output_column(self):
        stacked = self._xyz()["x", "y", "z"]
        assert stacked.shape == (2, 3)
        np.testing.assert_allclose(stacked, [[0.0, 2.0, 4.0], [1.0, 3.0, 5.0]])

    def test_the_canonical_coordinate_keys_work_as_a_tuple(self):
        assert self._xyz()[tuple(molrs.keys.COORDS)].shape == (2, 3)

    def test_a_missing_column_names_itself(self):
        with pytest.raises(KeyError, match="absent"):
            self._xyz()["x", "absent"]

    def test_mixed_dtypes_are_refused(self):
        # Stacking a float column onto a uint one would silently upcast.
        with pytest.raises(ValueError, match="dtype"):
            self._xyz()["x", "id"]
