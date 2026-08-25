"""One LJ pair kernel, two construction paths, one number.

(a) ``molrs.md.LJCut`` over a single hand-fed pair via ``eval_pairs``.
(b) ``pair:lj/cut`` ForceField compiled through ``to_potentials``.
Both must match the 12-6 closed form ``4ε[(σ/r)¹² − (σ/r)⁶]`` and differ
from each other by at most 1 ulp.
"""
from __future__ import annotations

import math
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

import molrs
from molrs.md import LJCut

EPS = 1.0
SIGMA = 1.0
R = 1.5
CUTOFF = 3.0
GOLD = 4.0 * EPS * ((SIGMA / R) ** 12 - (SIGMA / R) ** 6)


def _ulp_diff(a: float, b: float) -> int:
    if a == b:
        return 0
    return abs(np.float64(a).view(np.int64) - np.float64(b).view(np.int64))


lj = LJCut(EPS, SIGMA, CUTOFF, shifted=False)
disp = np.array([[R, 0.0, 0.0]], dtype=np.float64)
e_loop, _f_loop = lj.eval_pairs(
    2,
    np.array([0], dtype=np.uint32),
    np.array([1], dtype=np.uint32),
    disp,
    np.array([R * R], dtype=np.float64),
)

ff = molrs.ff.ForceField("lj")
ff.def_pairstyle("lj/cut", {"cutoff": CUTOFF})
ff.def_pairtype("lj/cut", "A", "A", {"epsilon": EPS, "sigma": SIGMA})
frame = molrs.Frame()
atoms = molrs.Block()
atoms.insert("type", np.array(["A", "A"], dtype=str))
atoms.insert("x", np.array([0.0, R]))
atoms.insert("y", np.array([0.0, 0.0]))
atoms.insert("z", np.array([0.0, 0.0]))
frame["atoms"] = atoms
pairs = molrs.Block()
pairs.insert("i", np.array([0], dtype=np.int64))
pairs.insert("j", np.array([1], dtype=np.int64))
frame["pairs"] = pairs
coords = np.array([0.0, 0.0, 0.0, R, 0.0, 0.0], dtype=np.float64)
e_ff, _f_ff = ff.to_potentials(frame).calc_energy_forces(coords)

rel_loop = abs(e_loop - GOLD) / abs(GOLD)
rel_ff = abs(e_ff - GOLD) / abs(GOLD)
assert rel_loop < 1e-15, f"loop path relative error {rel_loop}"
assert rel_ff < 1e-15, f"compiled path relative error {rel_ff}"
assert _ulp_diff(float(e_loop), float(e_ff)) <= 1, (
    f"loop vs compiled differ by more than 1 ulp: {e_loop!r} vs {e_ff!r}"
)
print(f"pair-kernel-merge ok: E={e_loop:.16e} gold={GOLD:.16e}")
assert math.isfinite(e_loop)
