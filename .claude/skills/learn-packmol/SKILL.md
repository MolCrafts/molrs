---
name: learn-packmol
description: Read the Packmol source code and explain how it works in detail, mapping it to the molrs-pack implementation. Identify any discrepancies and their implications.
---

When learn code, always include:

- Designing, modifying, or reviewing packing algorithms
- Changing optimizer parameters, line search, convergence criteria
- Adding/modifying constraints or penalty functions
- Changing the movebad heuristic, tolerance strategy, or initialization
- Tuning hyperparameters or defaults
- Refactoring the packing loop structure

## Strict Rule: Packmol Is the Reference Implementation

The Packmol Fortran source at `/Users/roykid/work/packmol` is the **authoritative reference** for all algorithmic decisions in `molrs-pack`. You MUST:

1. **Read the relevant Packmol source file(s) before proposing any change** to algorithms, hyperparameters, or optimization strategies.
2. **Match Packmol's behavior** unless there is an explicit, documented reason to diverge (e.g. the exact Euler chain-rule gradient, which is a known intentional improvement).
3. **Never silently fall back** to an alternative algorithm, different default value, or simplified formulation when the Packmol approach seems complex or hard to port. If you believe a deviation is warranted, you MUST stop and ask the developer for approval, explaining exactly what Packmol does and what you propose instead.

## Packmol Source Map

Read these files as needed — do NOT guess what they contain:

| Packmol file | What it defines |
|---|---|
| `src/gencan.f` | GENCAN optimizer: SPG + Truncated Newton, Armijo line search, trust region, CG solver, all convergence criteria |
| `src/pgencan.f90` | GENCAN wrapper: sets rotation bounds, delmin=2.0, maxfc=10*maxit, epsgpsn=1e-6, calls easygencan |
| `src/computef.f90` | Objective function: fdist (distance penalty) + frest (constraint penalty), cell-list traversal |
| `src/computeg.f90` | Analytical gradient: atom-level Cartesian gradient → chain rule to DOF (position + Euler angles) |
| `src/fparc.f90` | Atom-pair distance penalty: quartic `(d_tol² - d²)²` with optional short-range enhancement |
| `src/gparc.f90` | Gradient of atom-pair distance penalty |
| `src/comprest.f90` | Geometric constraint penalties: 15 constraint types (box, sphere, ellipsoid, cylinder, plane, gaussian) |
| `src/gwalls.f90` | Gradient of geometric constraints |
| `src/heuristics.f90` | movebad heuristic: identify worst molecules, relocate near good ones or random, flashsort ranking |
| `src/initial.f90` | Initial placement: random COM + angles, constraint pre-fitting loop, overlap avoidance |
| `src/restmol.f90` | Single-molecule constraint optimization |
| `src/polartocart.f90` | Euler angle → rotation matrix (beta=Y, gamma=Z, theta=X), derivative matrices |
| `src/swaptype.f90` | Per-type sequential packing: freeze other types, pack one type at a time |
| `src/cell_indexing.f90` | Cell linked-list spatial indexing |
| `src/random.f90` | RNG (seed-controlled deterministic) |
| `app/packmol.f90` | Main loop: outer iteration, radscale tightening, movebad activation, convergence checking |

## Canonical Hyperparameters

These values come directly from Packmol. When in doubt, read the source file listed in the "Source" column.

### GENCAN Optimizer (gencan.f + pgencan.f90)

| Parameter | Packmol value | Source location |
|---|---|---|
| gamma (Armijo) | 1e-4 | gencan.f:431 |
| beta (Wolfe) | 0.5 | gencan.f:430 |
| theta (angle condition) | 1e-6 | gencan.f:432 |
| sigma1 (safeguard lower) | 0.1 | gencan.f:433 |
| sigma2 (safeguard upper) | 0.9 | gencan.f:434 |
| lspgmi (SPG step min) | 1e-10 | gencan.f:478 |
| lspgma (SPG step max) | 1e10 | gencan.f:477 |
| maxextrap | 100 | gencan.f:469 |
| mininterp | 4 | gencan.f:470 |
| nint (interpolation divisor) | 2.0 | gencan.f |
| next (extrapolation multiplier) | 2.0 | gencan.f |
| epsgpsn (projected gradient tol) | 1e-6 | pgencan.f90:63 |
| fmin (objective threshold) | 1e-5 | pgencan.f90 |
| delmin (trust radius min) | 2.0 | pgencan.f90:68 |
| maxfc | 10 * maxit | pgencan.f90:64 |
| steabs (FD step absolute) | 1e-10 | gencan.f:405 |
| sterel (FD step relative) | 1e-7 | gencan.f:406 |
| cgepsi (CG tol initial) | 0.1 | gencan.f:698 |
| cgepsf (CG tol final) | 1e-5 | gencan.f:699 |
| epsnqmp (CG progress tol) | 1e-4 | gencan.f:710 |
| maxitnqmp (CG stall limit) | 5 | gencan.f:711 |
| cgmaxit | max(1, 10*log(nind)) | gencan.f |

### Penalty Functions

| Parameter | Packmol value | Source location |
|---|---|---|
| scale (box constraint) | 1.0 | initial.f90:50 |
| scale2 (sphere/ellipsoid/cylinder) | 0.01 | initial.f90:51 |
| Distance penalty | quartic: `(d_tol² - d²)²` | fparc.f90 |
| Box penalty | quadratic per axis: `(violation)²` | comprest.f90 type 2/3 |
| Sphere inside | `scale2 * min(d² - R², 0)²` | comprest.f90 type 8 |
| Sphere outside | `scale2 * max(d² - R², 0)²` | comprest.f90 type 4 |

### Tolerance Strategy (packmol.f90)

| Parameter | Packmol value | Source location |
|---|---|---|
| discale (initial inflation) | 1.1 | input, default |
| Tightening factor | 0.9 per step | packmol.f90:~811-948 |
| Tightening trigger | `fdist<precision OR fimp<2%` while `radscale>1.0` | packmol.f90 |
| Convergence uses | `radius_ini` (original, not inflated) | packmol.f90 |

### movebad Heuristic (heuristics.f90)

| Parameter | Packmol value | Source location |
|---|---|---|
| Activation condition | `radscale==1.0 AND fimp<=10%` | packmol.f90 |
| movefrac | 0.5 (max fraction to relocate) | input, default |
| movebadrandom | false (perturb good molecule) | input, default |
| Perturbation radius | `0.6 * rnd() * dmax` around good molecule | heuristics.f90:~120-127 |
| Angle copy | Copy Euler angles from good molecule | heuristics.f90 |
| Post-relocation | Call restmol to fit constraints | heuristics.f90 |

### Main Loop (packmol.f90)

| Aspect | Packmol behavior |
|---|---|
| Type ordering | Pack types 1..N individually, then all together |
| Skip individual packing | `packall=true` |
| Init phase (init1) | Geometric constraints only, no distance penalty |
| Convergence | `fdist < precision AND frest < precision` using original tolerance |

## Known Intentional Divergences in molrs-pack

These are the ONLY places where molrs-pack deliberately differs from Packmol. Any other discrepancy is a bug unless explicitly approved by the developer:

1. **Euler angle gradients**: molrs-pack uses exact analytical Jacobian `dR/d(angle) * x_ref` instead of Packmol's torque approximation. This is strictly more correct.
2. **Random rotation**: molrs-pack uses Shoemake's method for uniform SO(3) sampling instead of uniform Euler angles. This produces unbiased rotations.
3. **movebad details**: molrs-pack uses a two-pass strategy (center relocation + rotation fix) with `movefrac=0.05` default and `min_distance * 1.5` relocation radius, which differs from Packmol's single-pass with `movefrac=0.5` and `0.6*rnd()*dmax`. Any further changes to movebad MUST be checked against Packmol first.

## Mandatory Workflow

When you receive a task touching molrs-pack:

1. **Identify** which Packmol subsystem is relevant (optimizer, constraints, movebad, tolerance, initialization, main loop).
2. **Read** the corresponding Packmol source files listed above. Do not rely on memory or summaries.
3. **Compare** the proposed change against Packmol's implementation.
4. **If compatible**: proceed, citing the Packmol source location that validates the approach.
5. **If incompatible**: STOP. Report to the developer:
   - What Packmol does (file, line, exact behavior)
   - What the proposed change would do differently
   - Why the divergence exists or is needed
   - Ask for explicit approval before proceeding.

**Never** introduce a silent fallback, simplified approximation, or alternative default that deviates from Packmol without developer sign-off.
