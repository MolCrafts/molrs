import numpy as np
import pytest
import molrs


class TestBlockConstruction:
    def test_empty(self):
        b = molrs.Block()
        assert b.nrows is None
        assert len(b) == 0
        assert b.keys() == []

    def test_repr_empty(self):
        b = molrs.Block()
        assert "Block" in repr(b)
        assert "None" in repr(b)


class TestBlockInsert:
    def test_f32(self):
        b = molrs.Block()
        b.insert("x", np.array([1.0, 2.0, 3.0], dtype=np.float32))
        assert b.nrows == 3
        assert len(b) == 1
        assert "x" in b

    def test_f64(self):
        b = molrs.Block()
        b.insert("x", np.array([1.0, 2.0], dtype=np.float64))
        assert b.nrows == 2

    def test_i64(self):
        b = molrs.Block()
        b.insert("id", np.array([10, 20, 30], dtype=np.int64))
        assert b.nrows == 3

    def test_bool(self):
        b = molrs.Block()
        b.insert("mask", np.array([True, False, True]))
        assert b.nrows == 3

    def test_u32(self):
        b = molrs.Block()
        b.insert("idx", np.array([0, 1, 2], dtype=np.uint32))
        assert b.nrows == 3

    def test_2d_array(self):
        b = molrs.Block()
        b.insert("pos", np.zeros((5, 3), dtype=np.float32))
        assert b.nrows == 5

    def test_nrows_enforcement(self):
        b = molrs.Block()
        b.insert("x", np.array([1.0, 2.0], dtype=np.float32))
        with pytest.raises(ValueError):
            b.insert("y", np.array([1.0, 2.0, 3.0], dtype=np.float32))

    def test_int32_accepted(self):
        b = molrs.Block()
        b.insert("x", np.array([1, 2], dtype=np.int32))
        assert b.dtype("x") == "int"

    def test_overwrite_key(self):
        b = molrs.Block()
        b.insert("x", np.array([1.0, 2.0], dtype=np.float32))
        b.insert("x", np.array([3.0, 4.0], dtype=np.float32))
        result = b.view("x").flatten()
        np.testing.assert_allclose(result, [3.0, 4.0])


class TestBlockGet:
    def test_roundtrip_f32(self):
        b = molrs.Block()
        original = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b.insert("x", original)
        result = b.view("x")
        np.testing.assert_allclose(result.flatten(), original, atol=1e-6)

    def test_roundtrip_f64(self):
        b = molrs.Block()
        original = np.array([1.1, 2.2], dtype=np.float64)
        b.insert("x", original)
        result = b.view("x")
        np.testing.assert_allclose(result.flatten(), original, atol=1e-12)

    def test_roundtrip_i64(self):
        b = molrs.Block()
        b.insert("id", np.array([10, 20], dtype=np.int64))
        result = b.view("id")
        np.testing.assert_array_equal(result.flatten(), [10, 20])

    def test_roundtrip_bool(self):
        b = molrs.Block()
        b.insert("m", np.array([True, False]))
        result = b.view("m")
        np.testing.assert_array_equal(result.flatten(), [True, False])

    def test_roundtrip_u32(self):
        b = molrs.Block()
        b.insert("idx", np.array([0, 42], dtype=np.uint32))
        result = b.view("idx")
        np.testing.assert_array_equal(result.flatten(), [0, 42])

    def test_missing_key_raises_key_error(self):
        b = molrs.Block()
        with pytest.raises(KeyError):
            b.view("nonexistent")

    def test_roundtrip_2d(self):
        b = molrs.Block()
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        b.insert("pos", data)
        result = b.view("pos")
        np.testing.assert_allclose(result, data)

    def test_view_returns_numpy_array_with_native_owner(self):
        b = molrs.Block()
        b.insert("x", np.array([1.0, 2.0, 3.0], dtype=np.float32))
        view = b.view("x")
        assert isinstance(view, np.ndarray)
        assert view.base is not None
        np.testing.assert_allclose(view, [1.0, 2.0, 3.0])

    def test_get_uses_view_backing_for_numeric_columns(self):
        b = molrs.Block()
        b.insert("x", np.array([4.0, 5.0], dtype=np.float32))
        result = b.view("x")
        assert isinstance(result, np.ndarray)
        assert result.base is not None
        np.testing.assert_allclose(result, [4.0, 5.0])


class TestBlockOperations:
    def test_keys(self):
        b = molrs.Block()
        b.insert("x", np.array([1.0], dtype=np.float32))
        b.insert("y", np.array([2.0], dtype=np.float32))
        keys = sorted(b.keys())
        assert keys == ["x", "y"]

    def test_contains(self):
        b = molrs.Block()
        b.insert("x", np.array([1.0], dtype=np.float32))
        assert "x" in b
        assert "y" not in b

    def test_remove(self):
        b = molrs.Block()
        b.insert("x", np.array([1.0], dtype=np.float32))
        b.remove("x")
        assert "x" not in b
        assert len(b) == 0

    def test_remove_missing_raises_key_error(self):
        b = molrs.Block()
        with pytest.raises(KeyError):
            b.remove("nonexistent")

    def test_dtype(self):
        b = molrs.Block()
        b.insert("x", np.array([1.0], dtype=np.float32))
        b.insert("id", np.array([1], dtype=np.int64))
        assert b.dtype("x") == "float"
        assert b.dtype("id") == "int"

    def test_dtype_missing_raises_key_error(self):
        b = molrs.Block()
        with pytest.raises(KeyError):
            b.dtype("missing")

    def test_repr(self):
        b = molrs.Block()
        b.insert("x", np.array([1.0, 2.0], dtype=np.float32))
        r = repr(b)
        assert "Block" in r
        assert "2" in r
