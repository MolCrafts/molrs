# Force-field conventions

molrs has one force-field convention, and it is LAMMPS's. For every style
molrs registers, the energy expression, the factors (no hidden ½), the
parameter meanings and the parameter units are those of the LAMMPS style of
the same name, or of the LAMMPS style it corresponds to. Reading a LAMMPS
force field and writing it back is therefore the identity on coefficients,
and every other engine — GROMACS, OpenMM, AMBER, the GAFF / OPLS-AA / MMFF /
UFF tables — converts to and from this convention at its reader, writer or
typifier, never in a kernel.

This page is the reference for that convention: what each style computes,
what its parameters mean, which engine form maps onto it and how, and how the
terms the next releases add (Urey–Bradley, explicit 1-4 pairs, CMAP) are
represented.

## Units

A `ForceField` declares a LAMMPS unit preset (`ForceField.units`: `real`,
`metal`, `lj`; `real` when undeclared). Every parameter is a number in that
preset. Two rules are LAMMPS's own and hold in every preset:

!!! important "Angles are degrees"
    Every **angle-valued parameter** — an equilibrium angle (`theta0`,
    `chi0`), a dihedral or improper phase (`phase`, `phase<m>`,
    `phi1..phi3`) — is stored in **degrees**, exactly as a LAMMPS
    `*_coeff` line writes it. A **force constant** is per **radian**ⁿ
    (`angle harmonic` `k` is energy/rad²), as in LAMMPS. The kernels convert
    the angle to radians once, when they are built.

    This makes "molrs convention = LAMMPS convention" literally true for the
    stored values: `angle_coeff c3-c3-oh 76.79 109.66` is the type
    `{k: 76.79, theta0: 109.66}`, with no conversion in either direction.

The `forcefield` section of a record states `"angle": "degree"` beside its
preset. A section written by molrs ≤ 0.15 states `"angle": "radian"`, so
0.16 refuses it (its preset and its stated angle unit disagree) rather than
reading it in the wrong convention.

The LAMMPS reader keeps the file's `units` (a `metal` file is a `metal` force
field, `coul/cut` taking LAMMPS's `metal` Coulomb constant); the LAMMPS writer
converts energies and lengths only when the target `units` differs from the
force field's. Every other reader produces `real`.

## Style reference

`k`, `r0`, `theta0`, … are the parameter names. They are LAMMPS's symbols,
lower-cased, with one exception: LAMMPS spells two different things `d` — a
phase (`dihedral charmm`, `fourier`) and a sign (`dihedral harmonic`,
`improper cvff`) — and its multiplicity `n`, so molrs names those slots
`phase`, `sign` and `periodicity`. The values and their order are LAMMPS's.

"0.16" says what changed in 0.16 (see the [migration guide](../migration.md));
the energy of a physical system did not change with it.

### Bonds

| Style | Energy | Parameters (units) | LAMMPS | 0.16 |
|---|---|---|---|---|
| `harmonic` | k (r − r0)² | `k` (E/L²), `r0` (L) | `bond_style harmonic` `K r0` | `k` is LAMMPS's `K` (was ½k form: `k_new = k_old / 2`) |
| `morse` | d0 [1 − e^(−alpha (r − r0))]² | `d0` (E), `alpha` (1/L), `r0` (L) | `bond_style morse` `D0 alpha r0` | `D` renamed `d0` |
| `class2` | k2 Δ² + k3 Δ³ + k4 Δ⁴, Δ = r − r0 | `r0`, `k2`, `k3`, `k4` | `bond_style class2` `r0 K2 K3 K4` | unchanged |
| `mmff_bond` | MMFF94 cubic-quartic stretch | per-instance `kb`, `r0` | none | unchanged (molrs's definition) |
| `uff_bond` | ½ kb (r − r0)² (RDKit) | per-instance `kb`, `r0` | none | unchanged (molrs's definition) |

### Angles

| Style | Energy | Parameters (units) | LAMMPS | 0.16 |
|---|---|---|---|---|
| `harmonic` | k (θ − theta0)² | `k` (E/rad²), `theta0` (deg) | `angle_style harmonic` `K theta0` | `k_new = k_old / 2`; `theta0` degrees (was radians) |
| `class2` | k2 Δ² + k3 Δ³ + k4 Δ⁴, Δ = θ − theta0 | `theta0` (deg), `k2`, `k3`, `k4` (E/radⁿ) | `angle_style class2` `theta0 K2 K3 K4` (its `bb` / `ba` cross terms are not implemented) | `theta0` degrees |
| `mmff_angle`, `mmff_stbn` | MMFF94 bend / stretch-bend | per-instance `ka`, `theta0` (deg), `kba_*` | none | the `theta0` column is degrees |
| `uff_angle` | UFF Fourier / order-n bend (RDKit) | per-instance `ka`, `order`, `c0..c2` (`theta0` kept as metadata, deg) | none | `theta0` metadata degrees |

### Dihedrals

| Style | Energy | Parameters (units) | LAMMPS | 0.16 |
|---|---|---|---|---|
| `periodic` | Σₘ kₘ [1 + cos(nₘ φ − γₘ)] | `k<m>` (E), `periodicity<m>`, `phase<m>` (deg); or one term `k`, `periodicity`, `phase` | `dihedral_style fourier` `m K1 n1 d1 …` | phases degrees; the `dihedral fourier` alias is gone (LAMMPS `fourier` reads as `periodic`) |
| `charmm` | k [1 + cos(n φ − d)] | `k` (E), `periodicity`, `phase` (deg), `w` | `dihedral_style charmm` `K n d w` | `phase` degrees; `w ≠ 0` refused at compile time (see [1-4](#1-4-interactions)) |
| `opls` | ½[k1(1 + cos φ) + k2(1 − cos 2φ) + k3(1 + cos 3φ) + k4(1 − cos 4φ)] | `k1..k4` (E) | `dihedral_style opls` | unchanged |
| `multi/harmonic` | Σₙ₌₁⁵ aₙ cosⁿ⁻¹ φ | `a1..a5` (E) | `dihedral_style multi/harmonic` | unchanged |
| `harmonic` | k [1 + sign cos(n φ)] | `k` (E), `sign` (±1), `periodicity` | `dihedral_style harmonic` `K d n` | new kernel (the LAMMPS reader read it, nothing priced it) |
| `class2` | Σₙ₌₁³ kₙ [1 − cos(n φ − phiₙ)] | `k1, phi1, k2, phi2, k3, phi3` (E, deg) | `dihedral_style class2` (core term; `mbt`/`ebt`/`at`/`aat`/`bb13` not implemented) | phases degrees |
| `mmff_torsion` | ½[V1(1 + cos φ) + V2(1 − cos 2φ) + V3(1 + cos 3φ)] | per-instance `v1..v3` | none | unchanged |
| `uff_torsion` | V/2 [1 − cosTerm cos(n φ)] (RDKit) | per-instance `V`, `order`, `cosTerm` | none | unchanged |

### Impropers

| Style | Energy | Parameters (units) | Atom order | LAMMPS | 0.16 |
|---|---|---|---|---|---|
| `harmonic` | k (χ − chi0)², χ = \|φ(I,J,K,L)\| | `k` (E/rad²), `chi0` (deg) | I is the centre | `improper_style harmonic` `K chi0` | `chi0` degrees |
| `cvff` | k [1 + sign cos(n φ(I,J,K,L))] | `k` (E), `sign` (±1), `periodicity` | I is the centre | `improper_style cvff` `K d n` | unchanged |
| `periodic` | k [1 + cos(n φ(I,J,K,L) − γ)] | `k` (E), `periodicity`, `phase` (deg) | AMBER's: K is the centre | `improper_style cvff` (one term, γ ∈ {0°, 180°}: `d = cos γ`) | `phase` degrees; OpenMM rows re-ordered |
| `mmff_oop` | ½·143.9325 koop χ², χ the Wilson angle of bond I→L to plane (I,J,K) | per-instance `koop` | I is the centre | none (geometry of `improper_style umbrella` / `fourier`) | centre first (was second) |
| `uff_inversion` | K [c0 + c1 cos ω + c2 cos 2ω], ω of bond I→L to plane (I,J,K) | per-instance `K`, `c0..c2` | I is the centre | `improper_style fourier` | centre first (was second) |

### Pair styles

| Style | Energy | Parameters (units) | LAMMPS | 0.16 |
|---|---|---|---|---|
| `lj/cut` | C ε [(σ/r)ⁿ − (σ/r)ᵐ], C = n/(n−m)·(n/m)^(m/(n−m)); 4ε[(σ/r)¹² − (σ/r)⁶] at n = 12, m = 6 | `epsilon` (E), `sigma` (L); style `cutoff`, `mixing`, `n`, `m`, `shift` | `pair_style lj/cut` (n ≠ 12 or m ≠ 6: `mie/cut`; `shift`: `pair_modify shift yes`; `mixing`: `pair_modify mix`) | unchanged |
| `lj/class2` | ε [2(σ/r)⁹ − 3(σ/r)⁶] | `epsilon`, `sigma` | `pair_style lj/class2` | unchanged |
| `buck` | a e^(−r/rho) − c/r⁶ | `a` (E), `rho` (L), `c` (E·L⁶) | `pair_style buck` `A rho C` | unchanged |
| `morse` | d0 [(1 − e^(−alpha (r − r0)))² − 1] | `d0` (E), `alpha` (1/L), `r0` (L) | `pair_style morse` `D0 alpha r0` | the compiled kernel read `D0`, the neighbour-driven one `d0`; both read `d0` |
| `coul/cut` | coulomb qᵢqⱼ / (dielectric (r + delta)) | style `coulomb` (E·L/e²), `dielectric`, `delta` (L), `cutoff` | `pair_style coul/cut` with `delta = 0` (the buffer is molrs's, for MMFF; the LAMMPS writer refuses `delta ≠ 0` and `dielectric ≠ 1`). LAMMPS fixes the constant (`qqr2e`) per `units` | unchanged |
| `coul/long/pme` | Ewald-summed coulomb qᵢqⱼ/r | style `coulomb`, `cutoff`, `alpha`, `order`, `grid_*` | `pair_style coul/long` + `kspace_style pppm` | unchanged |
| `thole` | T(r) qᵢqⱼ/r, T = 1 − (1 + s r/2) e^(−s r), s = ½(aᵢ + aⱼ)/(αᵢαⱼ)^(1/6) | per type `charge`, `alpha` (L³), `damp` | `pair_style thole` `alpha damp` (LAMMPS damps the Drude charges of the atoms; molrs's per-type `charge` is its own) | `a_thole` renamed `damp` |
| `coul/tt` | fₙ(r) qᵢqⱼ/r (Tang–Toennies) | style `b`, `c`, `order` | `pair_style coul/tt` (`n` = `order`) | unchanged |
| `uff_lj`, `mmff_vdw` | UFF x/D LJ; MMFF buffered 14-7 | per-instance / per-type | none | unchanged |

`special_bonds` (the force field's `[1-2, 1-3, 1-4]` weights for van der
Waals and Coulomb) is LAMMPS's `special_bonds lj … coul …`.

## Improper atom order

Every LAMMPS improper style that is a dihedral (`harmonic`, `cvff`) prices
the dihedral **I-J-K-L of the atoms in the order the data file lists them**,
and so does every molrs kernel of such a style. LAMMPS names the first atom
the centre ("atom of symmetry") of `harmonic` and `cvff`; CHARMM writes its
impropers that way. The out-of-plane styles (`fourier`, `umbrella`; molrs's
`uff_inversion`, `mmff_oop`) take the centre first too.

AMBER prices its improper over the dihedral with the centre **third**. No
order with the centre first has that dihedral — an AMBER improper's axis runs
through its centre — so an AMBER improper cannot be re-ordered centre-first
and keep its energy. molrs therefore stores `improper periodic` (the AMBER /
GAFF / OpenMM improper) in **AMBER's order**: the order whose dihedral is the
improper angle, which is also the order a LAMMPS data file lists it in for
`cvff` to reproduce AMBER (verified with LAMMPS: the AMBER improper of the
hand molecule below gives `0.00338479168892827` kcal/mol as `cvff` in AMBER
order; molrs gives `0.003384791688934619`).

| Source | File order | Stored order |
|---|---|---|
| LAMMPS data / `improper_coeff` | I, J, K, L | as written |
| GROMACS funct 4 (periodic), funct 2 (harmonic) | i, j, k, l (GROMACS prices φ(i,j,k,l)) | as written |
| AMBER prmtop, frcmod, GAFF typifier | i, j, K, l (centre third) | as written |
| OpenMM `<Improper class1 … class4>`, `ordering` default / `amber` | c1 = centre; OpenMM prices φ(c2, c3, c1, c4) | (c2, c3, c1, c4); the writer writes the inverse |
| OpenMM, `ordering="charmm"`, no wildcard | OpenMM prices φ(c1, c2, c3, c4) | as written |
| OpenMM, `ordering="smirnoff"` | three permutations averaged | refused |
| molrs topology perception (`generate_topology`, `trivalent_impropers`) | centre first | a force field that wants AMBER's order re-orders (GAFF does) |
| UFF, MMFF typifiers | — | centre first |

A frame's `impropers` row lists its atoms in the order of its type's
endpoints. Where OpenMM picks the order of two peripherals of equal type by
element and index, a caller that builds frame rows from an OpenMM-read field
decides the same way.

Up to 0.15 the OpenMM reader stored the file order, and the kernel priced
φ(c1, c2, c3, c4) where OpenMM prices φ(c2, c3, c1, c4): every improper of an
OpenMM-read field was priced over the wrong dihedral (0.40× OpenMM's energy
on the regression molecule). The OpenMM writer had the mirror error, and wrote
`cvff` rows OpenMM prices over a different dihedral; it now refuses `cvff`
(OpenMM cannot price a dihedral that starts at the centre).

## 1-4 interactions

LAMMPS has three mechanisms, and molrs represents each with LAMMPS's
parameters:

1. **Global weights** — `special_bonds lj w12 w13 w14 coul w12 w13 w14`:
   `ForceField.special_bonds`. AMBER is `lj 0 0 ½ coul 0 0 5/6`, OPLS-AA
   `0 0 ½`, CHARMM `0 0 0` (its 1-4 pairs are priced by the dihedral, below).
2. **Per-type 1-4 Lennard-Jones** — `pair_style lj/charmm/coul/charmm` (and
   `lj/charmm/coul/long`), `pair_coeff I J epsilon sigma epsilon14 sigma14`.
   In molrs, a pair style `lj/charmm` (van der Waals half, as `lj/cut` is of
   `lj/cut/coul/cut`) with columns `epsilon`, `sigma`, `epsilon14`,
   `sigma14`, mixed by the style's `mixing` (LAMMPS's default for these styles
   is `arithmetic`); self rows per type, explicit cross rows per type pair.
   GROMACS `[ pairtypes ]` lands in the cross rows' `epsilon14`/`sigma14`.
3. **Per-dihedral weight** — `dihedral_style charmm` `w`: the dihedral prices
   the 1-4 pair of its own end atoms, `w·[LJ(epsilon14, sigma14) + C qᵢqⱼ/r]`,
   beside `special_bonds` 1-4 weights of 0 (`w` = 1, ½ in six-membered rings,
   0 in four- and five-membered rings).

molrs 0.16 implements (1) and stores (3)'s `w`; `lj/charmm` and the
dihedral's 1-4 pair are the next release's kernels. Until then a
`dihedral charmm` type with `w ≠ 0` is **refused at compile time**, naming the
type and the weight: under `special_bonds charmm` that pair would otherwise be
silently zero. `w = 0` (AMBER's use of the style) compiles and prices
LAMMPS's `K[1 + cos(nφ − d)]`.

**Per-pair exceptions LAMMPS cannot express** — a GROMACS `[ pairs ]` row
with explicit parameters, an OpenMM `NonbondedForce` exception, an AMBER
dihedral whose `SCEE`/`SCNB` differ from the field's (the prmtop reader
refuses a non-uniform pair today) — are per-instance columns on the Frame's
`pairs` block, the rows the pair kernels already price (`atomi`, `atomj`,
`is_14`). A column overrides, for its row, what the style would give it:

| Column | Meaning |
|---|---|
| `epsilon`, `sigma` | the LJ parameters of this pair, in place of the pair style's row or mixing |
| `lj_scale` | the van-der-Waals weight of this pair, in place of `special_bonds.lj` for its class |
| `charge_product` | qᵢqⱼ (e²) of this pair, in place of the atoms' product |
| `coul_scale` | the Coulomb weight of this pair, in place of `special_bonds.coul` |

A null (absent) cell takes the style's value. OpenMM's exception
`(chargeProd, sigma, epsilon)` is `charge_product`, `sigma`, `epsilon` with
both scales 1; a GROMACS funct-1 `[ pairs ]` row with parameters is `sigma`,
`epsilon` (`lj_scale` 1); AMBER's per-dihedral divisors are
`lj_scale = 1/SCNB`, `coul_scale = 1/SCEE`. LAMMPS can express none of these
per pair, so the LAMMPS writers refuse a frame or field that carries them
(not yet reachable: no reader produces them in 0.16).

## Urey–Bradley

LAMMPS carries Urey–Bradley in one angle style, and so does molrs:
**`angle charmm`**, E = k(θ − theta0)² + k_ub(r₁₃ − r_ub)², parameters `k`
(E/rad²), `theta0` (deg), `k_ub` (E/L²), `r_ub` (L) — `angle_coeff t K theta0
K_ub r_ub`. It adds no exclusion (the 1-3 pair is excluded by
`special_bonds`). There is no separate Urey–Bradley category. Engine maps:

| Source | `angle charmm` |
|---|---|
| CHARMM `.prm` `ANGLES` `Ktheta Theta0 Kub S0` | as written (CHARMM has no ½) |
| GROMACS `[ angletypes ]` funct 5 `θ₀ k_θ r13 k_UB` (½k forms, nm, kJ/mol) | `k = k_θ/(2·4.184)`, `theta0 = θ₀`, `k_ub = k_UB/(2·418.4)`, `r_ub = 10·r13` |
| OpenMM `<AmoebaUreyBradleyForce><UreyBradley … k d>` (OpenMM adds a `HarmonicBondForce` term with `2k`, so `k` is un-halved) | `k_ub = k/418.4`, `r_ub = 10·d`, joined with the `HarmonicAngleForce` row of the same classes |

## CMAP

The convention is LAMMPS `fix cmap` (CHARMM's correction map):

- a crossterm names five atoms `(atomi, atomj, atomk, atoml, atomm)`;
  φ = dihedral(atomi, atomj, atomk, atoml), ψ = dihedral(atomj, atomk, atoml,
  atomm), both in (−180°, 180°];
- a map is an N×N grid (CHARMM: N = 24, 15° spacing) of energies in the
  force field's energy unit, stored **φ-major**: element `[i][j]` (flat index
  `i·N + j`) is the energy at φ = −180° + i·360°/N, ψ = −180° + j·360°/N —
  the order of a CHARMM / LAMMPS `.cmap` file (each `# phi` block is one φ
  row of N ψ values);
- the energy between grid points is LAMMPS's bicubic interpolation, with the
  derivatives LAMMPS precomputes from periodic cubic splines.

A `cmap` style's row holds one map as an array parameter (molrs 0.16 adds
array params and the five-endpoint `cmaps` block). OpenMM's
`CMAPTorsionForce` stores `energy[i + N·j]` at φ = 2πi/N, ψ = 2πj/N (origin
0, φ fastest): its reader maps element `(i, j)` to molrs `[(i + N/2) mod N]
[(j + N/2) mod N]`; OpenMM interpolates with a natural periodic bicubic
spline, so energies off the grid points differ from LAMMPS's at the
interpolation's accuracy. GROMACS `[ cmaptypes ]` lists CHARMM's grid; its
reader must be checked against a GROMACS energy before it is trusted.

## Engine maps at a glance

| Engine | bond `k` | angle `k`, `theta0` | phases | impropers |
|---|---|---|---|---|
| LAMMPS | `K` | `K`, deg | deg | as written |
| GROMACS (`.top`/`.itp`) | `k_b/2`, kJ→kcal, nm→Å | `k_θ/2`, deg | deg | as written |
| OpenMM XML | `k/2`, kJ→kcal, nm→Å | `k/2`, rad→deg | rad→deg | (c1..c4) ↔ (c2, c3, c1, c4) |
| AMBER prmtop | `RK` | `TK`, rad→deg | rad→deg | AMBER order |
| AMBER frcmod (writer) | `RK = k` | `TK = k`, deg | deg | AMBER order |
| GAFF / GAFF2 tables | `K` | `K`, deg | deg | AMBER order |
| OPLS-AA table (GROMACS `oplsaa.ff`) | `k_b/2` | `k_θ/2`, deg | — | — |

## How this is checked

- Each style has a hand-value test against the LAMMPS manual's formula.
- `ff::convention_invariance` holds the 0.16 energies of GAFF-, OPLS-AA-,
  MMFF94- and UFF-typed acetanilide and of a GROMACS-, OpenMM- and
  LAMMPS-read hand molecule to the values molrs 0.15.1 computed on the same
  inputs, term by term, at 1e-12 relative; every one matches bit for bit,
  except the OpenMM improper (the fix above), which now equals the
  GROMACS-read value of the same improper and the hand value of OpenMM's
  formula.
- The LAMMPS-read hand molecule run through LAMMPS (`run 0`) gives the
  same per-term energies as molrs to ≤ 2e-13 relative: bond
  0.162750104621288, angle 1.35959339751695, dihedral 0.692979891423841,
  improper 0.431717012867386, van der Waals 1.22012795938037, Coulomb
  −10.7066619897381 kcal/mol.
