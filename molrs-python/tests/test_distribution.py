"""Angular distributions bin radians, because that is what the kernel emits.

The bounds a caller omits come from the observable itself, never from a second
copy of the range in the binding — a copy in degrees is what made an exact 90°
angle land in the bin centred at 2.5.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import molrs
from molrs.compute.distribution import (
    AngleDistribution,
    DihedralDistribution,
    DistanceDistribution,
)

# Three samples at 30°, 90°, 90° over three bins of [0, π]. Centres are
# π/6, π/2, 5π/6 with sines ½, 1, ½, so density is 1:2:0 while the corrected
# density is flat — a shape no uncorrected array can have.
DENSITY = [0.3183098861837907, 0.6366197723675814, 0.0]
SIN_CORRECTED = [0.47746482927568606, 0.477464829275686, 0.0]
EDGES = [0.0, 1.0471975511965976, 2.0943951023931953, 3.141592653589793]


def _three_angles() -> molrs.Frame:
    """Three angle triples: 30°, 90°, 90°, each on its own vertex."""
    c30, s30 = math.cos(math.radians(30.0)), math.sin(math.radians(30.0))
    x = [1.0, 0.0, c30, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    y = [0.0, 0.0, s30, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
    frame = molrs.Frame()
    frame["atoms"] = molrs.Block(
        {
            "x": np.array(x, dtype=np.float64),
            "y": np.array(y, dtype=np.float64),
            "z": np.zeros(9, dtype=np.float64),
        }
    )
    frame["angles"] = molrs.Block(
        {
            "atomi": np.array([0, 3, 6], dtype=np.uint64),
            "atomj": np.array([1, 4, 7], dtype=np.uint64),
            "atomk": np.array([2, 5, 8], dtype=np.uint64),
        }
    )
    return frame


class TestAngleDistributionDefaultRange:
    def test_default_range_is_the_observables_own(self):
        result = AngleDistribution(3).compute(_three_angles())
        np.testing.assert_allclose(np.asarray(result.bin_edges), EDGES, atol=1e-12)
        np.testing.assert_array_equal(np.asarray(result.counts), [1, 2, 0])

    def test_a_right_angle_bins_at_pi_over_two(self):
        # On the old degree axis this landed in the bin centred at 2.5.
        result = AngleDistribution(180).compute(_three_angles())
        centres = np.asarray(result.bin_centers)
        counts = np.asarray(result.counts)
        occupied = centres[counts > 0]
        assert occupied.max() == pytest.approx(math.pi / 2, abs=math.pi / 180)

    def test_density_and_sin_correction_match_the_closed_form(self):
        result = AngleDistribution(3).compute(_three_angles())
        np.testing.assert_allclose(np.asarray(result.density), DENSITY, rtol=1e-12)
        np.testing.assert_allclose(
            np.asarray(result.density_sin_corrected), SIN_CORRECTED, rtol=1e-12
        )

    def test_explicit_radian_bounds_agree_with_the_default(self):
        default = AngleDistribution(3).compute(_three_angles())
        explicit = AngleDistribution(3, 0.0, math.pi).compute(_three_angles())
        np.testing.assert_array_equal(
            np.asarray(default.counts), np.asarray(explicit.counts)
        )

    def test_half_supplied_bounds_are_refused(self):
        with pytest.raises(ValueError, match="both `min` and `max`"):
            AngleDistribution(3, min=0.0)
        with pytest.raises(ValueError, match="both `min` and `max`"):
            AngleDistribution(3, max=math.pi)


class TestDihedralDistributionDefaultRange:
    def test_default_range_is_signed_and_spans_two_pi(self):
        d = DihedralDistribution(4)
        result = d.compute(_dihedral_frame())
        edges = np.asarray(result.bin_edges)
        assert edges[0] == pytest.approx(-math.pi)
        assert edges[-1] == pytest.approx(math.pi)

    def test_half_supplied_bounds_are_refused(self):
        with pytest.raises(ValueError, match="both `min` and `max`"):
            DihedralDistribution(4, min=-math.pi)


class TestDistanceDistributionKeepsMandatoryBounds:
    def test_bounds_stay_positional(self):
        # A distance has no natural range, so there is nothing to default to.
        with pytest.raises(TypeError):
            DistanceDistribution(4)  # type: ignore[call-arg]


def _dihedral_frame() -> molrs.Frame:
    """One planar trans quadruple; the value does not matter, the axis does."""
    frame = molrs.Frame()
    frame["atoms"] = molrs.Block(
        {
            "x": np.array([0.0, 1.0, 1.0, 2.0], dtype=np.float64),
            "y": np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float64),
            "z": np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64),
        }
    )
    frame["dihedrals"] = molrs.Block(
        {
            "atomi": np.array([0], dtype=np.uint64),
            "atomj": np.array([1], dtype=np.uint64),
            "atomk": np.array([2], dtype=np.uint64),
            "atoml": np.array([3], dtype=np.uint64),
        }
    )
    return frame
