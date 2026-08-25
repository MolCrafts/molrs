"""MD driver NVE conservation over neighbor-driven LJCut (0.14).

Driver-level twin of the Rust integrator gold in
``molrs/src/md/integrators.rs``: 64 Ar-like atoms, 1200 steps, relative
total-energy drift < 5e-5, and ``rebuild_count > 0``.

The engine is unit-agnostic: this script converts kcal/mol → MD energy
via ``UnitRegistry`` and takes ``k_B`` from ``UnitPreset("real")``.
Engine-side spelling ``molrs.md`` is used here; users spell ``molpy.md``.
"""
from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

import molrs
import molrs.md as md

EPSILON_KCAL = 0.238
SIGMA = 3.405
MASS_AMU = 39.948
N_SIDE = 4
CUTOFF = 6.0
SKIN = 0.8
N_STEPS = 1200
SAMPLE_EVERY = 50
DT_FS = 1.0
TOLERANCE = 5e-5

reg = molrs.UnitRegistry()
epsilon = (
    reg.quantity(EPSILON_KCAL, "kilocalorie_per_mole")
    .to("amu * angstrom ** 2 / femtosecond ** 2")
    .value
)
kb = molrs.UnitPreset("real").boltzmann()
spacing = 2.0 ** (1.0 / 6.0) * SIGMA
n = N_SIDE ** 3

pos = np.zeros((n, 3), dtype=np.float64)
for i in range(n):
    pos[i, 0] = (i % N_SIDE) * spacing
    pos[i, 1] = ((i // N_SIDE) % N_SIDE) * spacing
    pos[i, 2] = (i // (N_SIDE * N_SIDE)) * spacing
mass = np.full(n, MASS_AMU)
vel = md.MaxwellBoltzmann(kb * 100.0, seed=42).velocities(pos, mass)

frame = molrs.Frame()
atoms = molrs.Block()
atoms.insert("type", np.array(["Ar"] * n, dtype=str))
atoms.insert("x", pos[:, 0])
atoms.insert("y", pos[:, 1])
atoms.insert("z", pos[:, 2])
atoms.insert("mass", mass)
atoms.insert("vx", vel[:, 0])
atoms.insert("vy", vel[:, 1])
atoms.insert("vz", vel[:, 2])
frame["atoms"] = atoms
frame.box = molrs.Box.cube(N_SIDE * spacing)

driver = (
    md.MD(dtype=np.float64)
    .set_potential(md.LJCut(epsilon, SIGMA, CUTOFF))
    .set_neighbors(cutoff=CUTOFF, skin=SKIN)
)
state = driver.run(frame, N_STEPS, dt=DT_FS, kb=kb, thermo=SAMPLE_EVERY)

totals = [row["etotal"] for row in driver.thermo]
e0 = totals[0]
e_mean = float(np.mean(totals))
max_dev = max(abs(e - e0) for e in totals)
drift = max_dev / abs(e_mean)

assert e_mean != 0.0, "degenerate NVE setup: mean energy is zero"
assert drift < TOLERANCE, f"NVE relative energy drift {drift:.3e} exceeds {TOLERANCE}"
assert driver.rebuild_count is not None and driver.rebuild_count > 0, (
    "neighbour list never rebuilt; the run was not using the skin"
)
print(
    f"nve ok: drift={drift:.3e} rebuilds={driver.rebuild_count} E0={e0:.6e}"
)
