import numpy as np
import pytest
import molrs


class TestBoxConstruction:
    def test_cube(self):
        b = molrs.Box.cube(10.0)
        assert pytest.approx(b.volume(), abs=1e-3) == 1000.0

    def test_ortho(self):
        b = molrs.Box.ortho(np.array([2.0, 3.0, 4.0], dtype=np.float32))
        assert pytest.approx(b.volume(), abs=1e-3) == 24.0

    def test_triclinic(self):
        h = np.array(
            [[2.0, 1.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]], dtype=np.float32
        )
        b = molrs.Box(h)
        assert b.volume() > 0

    def test_singular_matrix_raises_value_error(self):
        h = np.zeros((3, 3), dtype=np.float32)
        with pytest.raises(ValueError):
            molrs.Box(h)

    def test_cube_zero_raises_value_error(self):
        with pytest.raises(ValueError):
            molrs.Box.cube(0.0)

    def test_cube_negative_raises_value_error(self):
        with pytest.raises(ValueError):
            molrs.Box.cube(-5.0)

    def test_custom_origin(self):
        b = molrs.Box.cube(
            10.0, origin=np.array([1.0, 2.0, 3.0], dtype=np.float32)
        )
        np.testing.assert_allclose(b.origin, [1.0, 2.0, 3.0], atol=1e-6)

    def test_custom_pbc(self):
        b = molrs.Box.cube(
            10.0, pbc=np.array([True, False, True])
        )
        assert b.pbc[0] == True
        assert b.pbc[1] == False
        assert b.pbc[2] == True


class TestBoxProperties:
    def test_h(self, cubic_box):
        h = cubic_box.h
        assert h.shape == (3, 3)
        assert pytest.approx(h[0, 0]) == 10.0
        assert pytest.approx(h[1, 1]) == 10.0
        assert pytest.approx(h[2, 2]) == 10.0

    def test_origin(self, cubic_box):
        o = cubic_box.origin
        assert o.shape == (3,)
        np.testing.assert_allclose(o, [0.0, 0.0, 0.0], atol=1e-6)

    def test_pbc(self, cubic_box):
        pbc = cubic_box.pbc
        assert pbc.shape == (3,)
        assert all(pbc)

    def test_volume(self, ortho_box):
        assert pytest.approx(ortho_box.volume(), abs=1e-2) == 750.0

    def test_lengths(self, ortho_box):
        lengths = ortho_box.lengths()
        np.testing.assert_allclose(lengths, [5.0, 10.0, 15.0], atol=1e-4)

    def test_lattice_vectors(self, cubic_box):
        for i in range(3):
            v = cubic_box.lattice(i)
            assert v.shape == (3,)


class TestBoxCoordinateTransforms:
    def test_roundtrip_frac_cart(self, cubic_box, sample_points):
        frac = cubic_box.to_frac(sample_points)
        cart = cubic_box.to_cart(frac)
        np.testing.assert_allclose(cart, sample_points, atol=1e-4)

    def test_frac_values(self, cubic_box):
        pts = np.array([[5.0, 5.0, 5.0]], dtype=np.float32)
        frac = cubic_box.to_frac(pts)
        np.testing.assert_allclose(frac, [[0.5, 0.5, 0.5]], atol=1e-4)

    def test_wrap(self, cubic_box):
        pts = np.array([[11.0, -1.0, 25.0]], dtype=np.float32)
        wrapped = cubic_box.wrap(pts)
        frac = cubic_box.to_frac(wrapped)
        assert np.all(frac >= 0.0)
        assert np.all(frac < 1.0)

    def test_wrap_identity(self, cubic_box):
        pts = np.array([[5.0, 5.0, 5.0]], dtype=np.float32)
        wrapped = cubic_box.wrap(pts)
        np.testing.assert_allclose(wrapped, pts, atol=1e-4)


class TestBoxDisplacement:
    def test_delta_no_mic(self, cubic_box):
        p1 = np.array([[0.1, 0.0, 0.0]], dtype=np.float32)
        p2 = np.array([[9.9, 0.0, 0.0]], dtype=np.float32)
        d = cubic_box.delta(p1, p2, minimum_image=False)
        assert pytest.approx(d[0, 0], abs=1e-3) == 9.8

    def test_delta_with_mic(self, cubic_box):
        p1 = np.array([[0.1, 0.0, 0.0]], dtype=np.float32)
        p2 = np.array([[9.9, 0.0, 0.0]], dtype=np.float32)
        d = cubic_box.delta(p1, p2, minimum_image=True)
        assert abs(d[0, 0]) < 1.0

    def test_delta_shape_mismatch(self, cubic_box):
        p1 = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        p2 = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
        with pytest.raises(ValueError):
            cubic_box.delta(p1, p2)


class TestBoxContainment:
    def test_isin_inside(self, cubic_box, sample_points):
        inside = cubic_box.isin(sample_points)
        assert inside.shape == (5,)
        assert all(inside)

    def test_isin_outside(self, cubic_box):
        pts = np.array([[-1.0, 0.0, 0.0], [11.0, 0.0, 0.0]], dtype=np.float32)
        inside = cubic_box.isin(pts)
        assert not inside[0]
        assert not inside[1]


class TestBoxErrors:
    def test_bad_h_shape(self):
        with pytest.raises(ValueError, match="3x3"):
            molrs.Box(np.ones((2, 3), dtype=np.float32))

    def test_bad_origin_length(self):
        h = np.eye(3, dtype=np.float32) * 10.0
        with pytest.raises(ValueError, match="length 3"):
            molrs.Box(h, origin=np.array([1.0, 2.0], dtype=np.float32))

    def test_bad_pbc_length(self):
        h = np.eye(3, dtype=np.float32) * 10.0
        with pytest.raises(ValueError, match="3 elements"):
            molrs.Box(h, pbc=np.array([True, False]))

    def test_bad_to_frac_shape(self, cubic_box):
        with pytest.raises(ValueError, match="N,3"):
            cubic_box.to_frac(np.ones((3, 2), dtype=np.float32))

    def test_bad_lattice_index(self, cubic_box):
        with pytest.raises(ValueError, match="0, 1, or 2"):
            cubic_box.lattice(3)

    def test_repr(self, cubic_box):
        r = repr(cubic_box)
        assert "Box" in r
        assert "1000" in r
