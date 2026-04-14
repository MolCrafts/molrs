---
name: molrs-science
description: Scientific correctness standards for molrs — force-field equations, integration algorithms, units, physical invariants, reference citations. Reference document only; no procedural workflow.
---

Reference standard for molrs scientific correctness. The `molrs-scientist` agent applies these rules; this file defines them.

## Domain Coverage

- Classical MD force fields: LJ, Coulomb, harmonic bonds/angles, MMFF94, OPLS-AA, AMBER
- Integration algorithms: velocity-Verlet, RESPA
- Neighbor list algorithms: cell list, Verlet list
- Periodic boundary conditions: minimum image, triclinic cells
- Constraint algorithms: SHAKE, RATTLE, LINCS
- Statistical mechanics: thermostats, barostats, ensemble averages
- Coarse-grained models, packing algorithms
- 3D coordinate generation: distance geometry, ETKDG

## Unit System (molrs "real" units)

| Quantity | Unit | Note |
|---|---|---|
| Length | Å | |
| Energy | kcal/mol | |
| Force | kcal/(mol·Å) | |
| Time | fs | |
| Mass | g/mol (amu) | |
| Charge | e | elementary |
| Temperature | K | |

Precision: `F = f64` throughout (always). The `f64` feature flag is deprecated and ignored.

### Key constants & conversions

- `1 kcal/mol = 4.184 kJ/mol = 0.04336 eV`
- `kB = 1.987204e-3 kcal/(mol·K)`
- Coulomb constant in molrs units: `C ≈ 332.0637133 kcal·Å·mol⁻¹·e⁻²`

## Common Potential Forms

```text
Harmonic bond:   V = K (r - r₀)²              [LAMMPS convention: K = k/2]
LJ 12-6:         V = 4ε [(σ/r)¹² - (σ/r)⁶]
Coulomb:         V = C · qᵢqⱼ / (εᵣ · r)
Morse:           V = D [1 - exp(-α (r - r₀))]²
MMFF bond:       V = 143.9325 · (kb/2) · Δr² · (1 + cs·Δr + (7/12)·cs²·Δr²)
MMFF angle:      V = 0.043844 · (ka/2) · Δθ² · (1 + cb·Δθ)         [rad]
Harmonic angle:  V = (kθ/2) (θ - θ₀)²                              [rad]
Periodic torsion: V = (1/2) Σₙ Vₙ [1 + cos(n·φ - δₙ)]
```

Always cite the reference equation + paper / book / source code line in the rustdoc.

## Physical Invariants (MUST hold)

| Invariant | Where to check |
|---|---|
| `V(r → ∞) → 0` | non-bonded potentials with cutoff |
| `V(r_eq) = minimum`, `F(r_eq) = 0` | bonded potentials |
| `F = -dV/dr` | every potential (numerical gradient test) |
| `f_ij = -f_ji` (Newton's 3rd) | every pair potential |
| Total energy conserved | NVE MD (within numerical precision) |
| Temperature equilibrates to target | NVT MD |
| Minimum image consistent with cutoff | PBC-using code |
| Constraint gradient: TRUE `∂V/∂x` accumulated with `+=` | every constraint in `molrs-pack` |

## Numerical Stability Hazards

- **Division by zero**: pair distance `r → 0`, denominator in MMFF bond stretch.
- **Overflow**: `(σ/r)¹²` when `r → 0`; clamp or use cutoff guard.
- **Catastrophic cancellation**: subtracting near-equal energies in conservation tests — use double precision and compare `drift / mean`.
- **Loss of precision in long simulations**: Verlet integrator drift (use Kahan summation or careful ordering only when required).

## Reference Implementations to Cross-Check

- **MMFF94**: RDKit `Code/ForceField/MMFF`, OpenBabel `src/forcefields/forcefieldmmff94.cpp`
- **LJ / Coulomb**: LAMMPS `src/MOLECULE/`, GROMACS `src/gromacs/mdlib/`
- **Packing**: Packmol Fortran source (see `learn-packmol` skill for the discipline)
- **3D coordinate generation**: RDKit ETKDG (`Code/DistGeom/`, `Code/GraphMol/DistGeomHelpers/`)
- **Stereochemistry**: RDKit `Code/GraphMol/Chirality.cpp`

## Severity Bands for Findings

- **ERROR** — wrong physics, wrong equation, wrong sign, wrong units. Block merge.
- **WARNING** — ambiguous reference, missing citation, untested edge case (e.g., cutoff discontinuity).
- **PASS** — equation matches reference, units consistent, invariants verified.

## Compliance Checklist

- [ ] Equation cited with paper / book / source `file:line`
- [ ] Units consistent end-to-end (kcal/mol, Å, fs, e)
- [ ] Limiting cases verified (`r → 0`, `r → ∞`, `r = r_eq`)
- [ ] Numerical gradient matches analytical
- [ ] Newton's 3rd law for pair potentials
- [ ] Energy conservation for NVE integrators
- [ ] Cutoff handling (energy/force continuity at `r_c`)
- [ ] Numerical stability hazards mitigated (no `1/r` without guard)
- [ ] PBC: minimum-image consistent with cutoff
