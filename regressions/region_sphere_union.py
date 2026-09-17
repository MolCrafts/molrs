r"""The solvent-accessible void of a bead cloud as a molrs region.

Smoke, not a gate: 40 000 random beads in a periodic 83.89 σ cube at
4.18 Å/σ (the number density of the PE-in-solvent scene molpack fills), one
sphere per bead of radius ``r_bead + r_probe``, and the void is
``~SphereUnion``. Prints the void
fraction on 10⁶ probes and the per-probe cost of ``distance``. No third-party
scientific package is imported.

Runner:

    uv --directory molrs-python run python ../regressions/region_sphere_union.py
"""

from __future__ import annotations

import time

import numpy as np

import molrs

SIGMA_A = 4.18
L = 83.89 * SIGMA_A
N_BEADS = 40_000
R_BEAD = 0.5 * SIGMA_A
R_PROBE = 1.0
N_PROBES = 1_000_000


def main() -> None:
    rng = np.random.default_rng(0)
    centers = rng.uniform(0.0, L, size=(N_BEADS, 3))
    box = molrs.Box.cube(L, np.zeros(3), np.array([True, True, True]))

    t0 = time.perf_counter()
    polymer = molrs.SphereUnion(centers, R_BEAD + R_PROBE, box=box)
    void = ~polymer
    t_build = time.perf_counter() - t0

    probes = rng.uniform(0.0, L, size=(N_PROBES, 3))
    t0 = time.perf_counter()
    inside = void.contains(probes)
    t_contains = time.perf_counter() - t0
    t0 = time.perf_counter()
    d = void.distance(probes[:100_000])
    t_distance = time.perf_counter() - t0

    # A random cloud at this density: void fraction ≈ exp(−n·4/3·π·R³).
    n = N_BEADS / L**3
    expect = np.exp(-n * 4.0 / 3.0 * np.pi * (R_BEAD + R_PROBE) ** 3)
    assert d.shape == (100_000,)
    assert np.all((d <= 0.0) == inside[:100_000])
    print(f"build            : {t_build * 1e3:.0f} ms for {N_BEADS} spheres")
    print(f"void fraction    : {inside.mean():.4f} (Poisson estimate {expect:.4f})")
    print(f"contains         : {t_contains / N_PROBES * 1e6:.2f} us/probe")
    print(f"distance         : {t_distance / 100_000 * 1e6:.2f} us/probe")
    print("region_sphere_union ok")


if __name__ == "__main__":
    main()
