r"""The angular distribution bins radians, and the sin correction is flat.

Three angle samples at 30°, 90°, 90° over three bins of `[0, π]`. The centres
are π/6, π/2, 5π/6 with sines ½, 1, ½, so the raw density is 1:2:0 while the
solid-angle-corrected density is **flat** — a shape no uncorrected array can
have, which is what makes this a golden rather than a smoke test.

The goldens are the closed form, not a recorded run:

    density                = [1/π, 2/π, 0]
    w_i                    = density_i / sin(c_i) = [2/π, 2/π, 0]
    ∫ w dθ                 = (4/π)(π/3) = 4/3
    density_sin_corrected  = w / (4/3) = [3/(2π), 3/(2π), 0]

Solid-angle Jacobian: `dΩ = sin θ dθ dφ`, so `P(θ) = H(θ)/sin θ` — VOTCA CSG
theory, <https://www.votca.org/csg/theory.html>. No third-party scientific
package. Runner:

    uv --directory molrs-python run python ../regressions/distribution_angular.py
"""

from __future__ import annotations

import math

import numpy as np

import molrs
from molrs.compute.distribution import AngleDistribution

DENSITY = (1.0 / math.pi, 2.0 / math.pi, 0.0)
SIN_CORRECTED = (3.0 / (2.0 * math.pi), 3.0 / (2.0 * math.pi), 0.0)


def three_angles() -> molrs.Frame:
    """30°, 90°, 90°, each on its own vertex so no atom is shared."""
    c30, s30 = math.cos(math.radians(30.0)), math.sin(math.radians(30.0))
    frame = molrs.Frame()
    frame["atoms"] = molrs.Block(
        {
            "x": np.array([1.0, 0.0, c30, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            "y": np.array([0.0, 0.0, s30, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]),
            "z": np.zeros(9),
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


frame = three_angles()
result = AngleDistribution(3).compute(frame)

# The axis is the observable's own [0, π] — not a degree range re-declared in
# the binding, which is what put an exact 90° angle in the bin centred at 2.5.
np.testing.assert_allclose(
    np.asarray(result.bin_edges), [0.0, math.pi / 3, 2 * math.pi / 3, math.pi], atol=1e-12
)
np.testing.assert_array_equal(np.asarray(result.counts), [1, 2, 0])
np.testing.assert_allclose(np.asarray(result.density), DENSITY, rtol=1e-12)
np.testing.assert_allclose(
    np.asarray(result.density_sin_corrected), SIN_CORRECTED, rtol=1e-12
)

# Flat where the raw density is 1:2 — the property that proves the correction
# was applied on a radian axis.
corrected = np.asarray(result.density_sin_corrected)
assert abs(corrected[0] - corrected[1]) < 1e-12, corrected

# A right angle lands at π/2 at any resolution.
fine = AngleDistribution(180).compute(frame)
centres = np.asarray(fine.bin_centers)
occupied = centres[np.asarray(fine.counts) > 0]
assert abs(occupied.max() - math.pi / 2) < math.pi / 180, occupied

# Half a range is an error, not a silent default.
for kwargs in ({"min": 0.0}, {"max": math.pi}):
    try:
        AngleDistribution(3, **kwargs)
    except ValueError as exc:
        assert "both `min` and `max`" in str(exc), exc
    else:
        raise SystemExit(f"AngleDistribution(3, **{kwargs}) should have raised")

print(
    "distribution_angular ok: edges [0, pi], counts [1, 2, 0], "
    "density 1:2:0, sin-corrected flat"
)
