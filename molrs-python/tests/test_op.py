"""FFI seam of ``molrs.op`` (assembly-07, mirrors ``molrs/src/op/``).

These prove that the numeric primitives cross: arrays arrive as float64 of the
documented shape, ``Fit`` is a frozen record whose ``freedom`` crosses as its
lowercase name, and ``SuperposeError`` maps to ``ValueError``. They re-derive
no numerics: superposition and the eigen-gap are proven by the unit tests in
``molrs/src/op/superpose.rs``. The one geometric value asserted (a pure
translation) is hand-checkable.
"""

from __future__ import annotations

import molrs
import numpy as np
import pytest
from molrs import _lib

# Position, numerical (tester contract: 1e-8).
POS_TOL = 1e-8

# Three non-collinear points: the rotation is unique.
TRIANGLE = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


class TestModule:
    def test_op_is_a_registered_submodule(self):
        assert hasattr(_lib, "op")
        assert molrs.op.superpose is not None

    def test_default_gap_tol(self):
        # `op::superpose::DEFAULT_GAP_TOL`.
        assert molrs.op.DEFAULT_GAP_TOL == 1e-4


class TestSuperpose:
    def test_returns_a_fit_with_float64_arrays_of_the_documented_shape(self):
        fit = molrs.op.superpose(TRIANGLE, TRIANGLE + [1.0, 2.0, 3.0])

        assert isinstance(fit, molrs.op.Fit)
        assert fit.rotation.dtype == np.float64
        assert fit.rotation.shape == (3, 3)
        assert fit.translation.dtype == np.float64
        assert fit.translation.shape == (3,)
        assert fit.center.shape == (3,)
        assert isinstance(fit.rmsd, float)
        assert isinstance(fit.rho, float)

    def test_a_pure_translation_is_recovered(self):
        # Hand-derived: target = reference + t ⇒ R = I, translation = t,
        # rmsd = 0, center = target centroid = (1/3, 1/3, 0) + t.
        t = np.array([1.0, 2.0, 3.0])
        fit = molrs.op.superpose(TRIANGLE, TRIANGLE + t)

        np.testing.assert_allclose(fit.rotation, np.eye(3), rtol=0, atol=POS_TOL)
        np.testing.assert_allclose(fit.translation, t, rtol=0, atol=POS_TOL)
        np.testing.assert_allclose(
            fit.center, [1.0 / 3.0 + 1.0, 1.0 / 3.0 + 2.0, 3.0], rtol=0, atol=POS_TOL
        )
        assert fit.rmsd == pytest.approx(0.0, abs=POS_TOL)

    def test_explicit_uniform_weights_equal_the_default(self):
        target = TRIANGLE + [1.0, 2.0, 3.0]
        default = molrs.op.superpose(TRIANGLE, target)
        weighted = molrs.op.superpose(TRIANGLE, target, weights=np.ones(3))

        np.testing.assert_array_equal(weighted.rotation, default.rotation)
        np.testing.assert_array_equal(weighted.translation, default.translation)

    @pytest.mark.parametrize(
        ("reference", "freedom", "has_axis"),
        [
            (TRIANGLE, "unique", False),
            # Two points fix every direction but the line through them.
            (TRIANGLE[:2], "spin", True),
            # One point fixes no rotation at all.
            (TRIANGLE[:1], "free", False),
        ],
        ids=["unique", "spin", "free"],
    )
    def test_freedom_crosses_as_its_lowercase_name(self, reference, freedom, has_axis):
        fit = molrs.op.superpose(reference, reference + [0.5, 0.0, 0.0])

        assert fit.freedom == freedom
        if has_axis:
            assert fit.axis.dtype == np.float64
            assert fit.axis.shape == (3,)
        else:
            assert fit.axis is None

    def test_fit_is_frozen(self):
        fit = molrs.op.superpose(TRIANGLE, TRIANGLE)
        with pytest.raises(AttributeError):
            fit.rmsd = 1.0

    def test_a_length_mismatch_is_a_value_error(self):
        with pytest.raises(ValueError, match="mismatch"):
            molrs.op.superpose(TRIANGLE, TRIANGLE[:2])

    def test_a_weight_count_mismatch_is_a_value_error(self):
        with pytest.raises(ValueError):
            molrs.op.superpose(TRIANGLE, TRIANGLE, weights=np.ones(2))

    def test_a_non_xyz_array_is_a_value_error(self):
        with pytest.raises(ValueError):
            molrs.op.superpose(TRIANGLE[:, :2], TRIANGLE[:, :2])


class TestCentroid:
    def test_weighted_centroid(self):
        # Hand-derived: (1·0 + 3·2) / 4 = 1.5.
        c = molrs.op.centroid(
            np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]), np.array([1.0, 3.0])
        )
        assert c.dtype == np.float64
        np.testing.assert_allclose(c, [1.5, 0.0, 0.0], rtol=0, atol=1e-12)

    def test_zero_total_weight_has_no_centroid(self):
        assert molrs.op.centroid(TRIANGLE, np.zeros(3)) is None
