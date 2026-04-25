import numpy as np
import pytest
import molrs


class TestSphere:
    def test_contains(self):
        s = molrs.Sphere(np.array([0.0, 0.0, 0.0], dtype=np.float64), 2.0)
        pts = np.array(
            [
                [0.0, 0.0, 0.0],  # center
                [1.0, 0.0, 0.0],  # inside
                [2.0, 0.0, 0.0],  # on surface
                [2.1, 0.0, 0.0],  # outside
            ],
            dtype=np.float64,
        )
        mask = s.contains(pts)
        assert mask[0] and mask[1] and mask[2]
        assert not mask[3]

    def test_bounds(self):
        s = molrs.Sphere(np.array([1.0, 2.0, 3.0], dtype=np.float64), 2.0)
        b = s.bounds()
        assert b.shape == (3, 2)
        np.testing.assert_allclose(b[:, 0], [-1.0, 0.0, 1.0], atol=1e-5)
        np.testing.assert_allclose(b[:, 1], [3.0, 4.0, 5.0], atol=1e-5)

    def test_bad_center_shape(self):
        with pytest.raises(ValueError, match="length 3"):
            molrs.Sphere(np.array([0.0, 0.0], dtype=np.float64), 1.0)

    def test_bad_points_shape(self):
        s = molrs.Sphere(np.array([0.0, 0.0, 0.0], dtype=np.float64), 1.0)
        with pytest.raises(ValueError, match="N, 3"):
            s.contains(np.ones((3, 2), dtype=np.float64))

    def test_repr(self):
        s = molrs.Sphere(np.array([1.0, 2.0, 3.0], dtype=np.float64), 5.0)
        r = repr(s)
        assert "Sphere" in r
        assert "5.00" in r


class TestHollowSphere:
    def test_contains(self):
        hs = molrs.HollowSphere(
            np.array([0.0, 0.0, 0.0], dtype=np.float64), 2.0, 5.0
        )
        pts = np.array(
            [
                [0.0, 0.0, 0.0],  # center - inside inner
                [1.0, 0.0, 0.0],  # inside inner
                [3.0, 0.0, 0.0],  # in shell
                [5.0, 0.0, 0.0],  # on outer surface
                [5.1, 0.0, 0.0],  # outside
            ],
            dtype=np.float64,
        )
        mask = hs.contains(pts)
        assert not mask[0] and not mask[1]
        assert mask[2] and mask[3]
        assert not mask[4]

    def test_bounds(self):
        hs = molrs.HollowSphere(
            np.array([0.0, 0.0, 0.0], dtype=np.float64), 2.0, 5.0
        )
        b = hs.bounds()
        assert b.shape == (3, 2)
        np.testing.assert_allclose(b[:, 0], [-5.0, -5.0, -5.0], atol=1e-5)

    def test_repr(self):
        hs = molrs.HollowSphere(
            np.array([0.0, 0.0, 0.0], dtype=np.float64), 2.0, 5.0
        )
        assert "HollowSphere" in repr(hs)
        assert "2.00" in repr(hs)


class TestRegionAnd:
    def test_sphere_and_sphere(self):
        s1 = molrs.Sphere(np.array([0.0, 0.0, 0.0], dtype=np.float64), 3.0)
        s2 = molrs.Sphere(np.array([2.0, 0.0, 0.0], dtype=np.float64), 3.0)
        intersection = s1 & s2

        pts = np.array(
            [
                [1.0, 0.0, 0.0],  # inside both
                [-2.5, 0.0, 0.0],  # inside s1 only
                [4.5, 0.0, 0.0],  # inside s2 only
            ],
            dtype=np.float64,
        )
        mask = intersection.contains(pts)
        assert mask[0] and not mask[1] and not mask[2]

    def test_hollow_and_sphere(self):
        hs = molrs.HollowSphere(
            np.array([0.0, 0.0, 0.0], dtype=np.float64), 1.0, 5.0
        )
        s = molrs.Sphere(np.array([3.0, 0.0, 0.0], dtype=np.float64), 3.0)
        result = hs & s
        assert isinstance(result, molrs.Region)


class TestRegionOr:
    def test_sphere_or_sphere(self):
        s1 = molrs.Sphere(np.array([0.0, 0.0, 0.0], dtype=np.float64), 1.0)
        s2 = molrs.Sphere(np.array([5.0, 0.0, 0.0], dtype=np.float64), 1.0)
        union = s1 | s2

        pts = np.array(
            [
                [0.0, 0.0, 0.0],  # inside s1
                [5.0, 0.0, 0.0],  # inside s2
                [2.5, 0.0, 0.0],  # outside both
            ],
            dtype=np.float64,
        )
        mask = union.contains(pts)
        assert mask[0] and mask[1] and not mask[2]


class TestRegionNot:
    def test_not_sphere(self):
        s = molrs.Sphere(np.array([0.0, 0.0, 0.0], dtype=np.float64), 2.0)
        complement = ~s

        pts = np.array(
            [
                [0.0, 0.0, 0.0],  # inside sphere
                [3.0, 0.0, 0.0],  # outside sphere
            ],
            dtype=np.float64,
        )
        mask = complement.contains(pts)
        assert not mask[0] and mask[1]

    def test_not_hollow_sphere(self):
        hs = molrs.HollowSphere(
            np.array([0.0, 0.0, 0.0], dtype=np.float64), 2.0, 5.0
        )
        result = ~hs
        assert isinstance(result, molrs.Region)


class TestRegionChaining:
    def test_shell_via_and_not(self):
        outer = molrs.Sphere(np.array([0.0, 0.0, 0.0], dtype=np.float64), 5.0)
        inner = molrs.Sphere(np.array([0.0, 0.0, 0.0], dtype=np.float64), 2.0)
        shell = outer & (~inner)

        pts = np.array(
            [
                [0.0, 0.0, 0.0],  # inside inner
                [3.0, 0.0, 0.0],  # in shell
                [6.0, 0.0, 0.0],  # outside
            ],
            dtype=np.float64,
        )
        mask = shell.contains(pts)
        assert not mask[0] and mask[1] and not mask[2]

    def test_composed_bounds(self):
        s1 = molrs.Sphere(np.array([0.0, 0.0, 0.0], dtype=np.float64), 3.0)
        s2 = molrs.Sphere(np.array([2.0, 0.0, 0.0], dtype=np.float64), 3.0)
        result = s1 & s2
        b = result.bounds()
        assert b.shape == (3, 2)

    def test_composed_repr(self):
        s1 = molrs.Sphere(np.array([0.0, 0.0, 0.0], dtype=np.float64), 3.0)
        s2 = molrs.Sphere(np.array([2.0, 0.0, 0.0], dtype=np.float64), 3.0)
        assert "composed" in repr(s1 & s2)

    def test_region_and_region(self):
        s1 = molrs.Sphere(np.array([0.0, 0.0, 0.0], dtype=np.float64), 5.0)
        s2 = molrs.Sphere(np.array([0.0, 0.0, 0.0], dtype=np.float64), 3.0)
        r1 = s1 & s2
        r2 = ~s2
        r3 = r1 | r2  # composed & composed
        assert isinstance(r3, molrs.Region)

    def test_type_error_on_bad_operand(self):
        s = molrs.Sphere(np.array([0.0, 0.0, 0.0], dtype=np.float64), 1.0)
        with pytest.raises(TypeError):
            s & "not_a_region"
