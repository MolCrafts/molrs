# Force-field IR

molrs holds every force field in one intermediate representation, the
**force-field IR**: one set of styles, each with a fixed energy expression,
parameter meanings and parameter units. The IR adopts LAMMPS's definitions as
its standard — for every style molrs registers, the expression, the factors
(no hidden ½), the parameter meanings and the units are those of the LAMMPS
style of the same name, or of the LAMMPS style it corresponds to. Reading a
LAMMPS force field and writing it back is therefore the identity on
coefficients, and every other engine — GROMACS, OpenMM, AMBER, the GAFF /
OPLS-AA / MMFF / UFF tables — converts to and from the IR at its reader,
writer or typifier, never in a kernel.

This page is the reference for the IR: what each style computes, what its
parameters mean, which engine form maps onto it and how, and how Urey–Bradley,
explicit 1-4 pairs and CMAP are represented.

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

    This makes "the force-field IR stores exactly what LAMMPS writes" literally true for the
    stored values: `angle_coeff c3-c3-oh 76.79 109.66` is the type
    `{k: 76.79, theta0: 109.66}`, with no conversion in either direction.

The `forcefield` section of a record states `"angle": "degree"` beside its
preset. A record molrs ≤ 0.15 wrote is `molrec_version` 1, whose sections
state `"angle": "radian"` (or, under `lj`, nothing) and hold the ½k harmonic
forms; 0.16 converts its numbers to these definitions on read, exactly, or
refuses it ([Records: molrec_version 2](../migration.md#records-molrec_version-2)).
A section built in memory with `"angle": "radian"` beside a preset is refused
(its preset and its stated angle unit disagree) rather than read in the wrong
units.

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
| `charmm` | k (θ − theta0)² + k_ub (r₁₃ − r_ub)² | `k` (E/rad²), `theta0` (deg), `k_ub` (E/L²), `r_ub` (L) | `angle_style charmm` `K theta0 K_ub r_ub` | new kernel ([Urey–Bradley](#ureybradley)) |
| `class2` | k2 Δ² + k3 Δ³ + k4 Δ⁴, Δ = θ − theta0 | `theta0` (deg), `k2`, `k3`, `k4` (E/radⁿ) | `angle_style class2` `theta0 K2 K3 K4` (its `bb` / `ba` cross terms are not implemented) | `theta0` degrees |
| `mmff_angle`, `mmff_stbn` | MMFF94 bend / stretch-bend | per-instance `ka`, `theta0` (deg), `kba_*` | none | the `theta0` column is degrees |
| `uff_angle` | UFF Fourier / order-n bend (RDKit) | per-instance `ka`, `order`, `c0..c2` (`theta0` kept as metadata, deg) | none | `theta0` metadata degrees |

### Dihedrals

| Style | Energy | Parameters (units) | LAMMPS | 0.16 |
|---|---|---|---|---|
| `periodic` | Σₘ kₘ [1 + cos(nₘ φ − γₘ)] | `k<m>` (E), `periodicity<m>`, `phase<m>` (deg); or one term `k`, `periodicity`, `phase` | `dihedral_style fourier` `m K1 n1 d1 …` | phases degrees; the `dihedral fourier` alias is gone (LAMMPS `fourier` reads as `periodic`) |
| `charmm` | k [1 + cos(n φ − d)], plus `w`·(its end atoms' 1-4 pair) | `k` (E), `periodicity`, `phase` (deg), `w` | `dihedral_style charmm` `K n d w` | `phase` degrees; `w` prices the 1-4 pair (see [1-4](#1-4-interactions)) |
| `opls` | ½[k1(1 + cos φ) + k2(1 − cos 2φ) + k3(1 + cos 3φ) + k4(1 − cos 4φ)] | `k1..k4` (E) | `dihedral_style opls` | unchanged |
| `multi/harmonic` | Σₙ₌₁⁵ aₙ cosⁿ⁻¹ φ | `a1..a5` (E) | `dihedral_style multi/harmonic` | unchanged |
| `nharmonic` | Σᵢ₌₁ᴺ aᵢ cosⁱ⁻¹ φ | `a1..aN` (E), contiguous, N ≥ 1 | `dihedral_style nharmonic` `N A1 … AN` | new style |
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
| `lj/cut` | C ε [(σ/r)ⁿ − (σ/r)ᵐ], C = n/(n−m)·(n/m)^(m/(n−m)); 4ε[(σ/r)¹² − (σ/r)⁶] at n = 12, m = 6 | `epsilon` (E), `sigma` (L); style `cutoff`, `mixing`, `n`, `m`, `shift` | `pair_style lj/cut` (`shift`: `pair_modify shift yes`; `mixing`: `pair_modify mix`; n ≠ 12 or m ≠ 6, LAMMPS's `mie/cut`, is neither read nor written) | `pair_style lj/cut` alone reads with no Coulomb style (was `coul/cut` beside it); `pair_modify shift yes` read and written (was dropped) |
| `lj/class2` | ε [2(σ/r)⁹ − 3(σ/r)⁶] | `epsilon`, `sigma` | `pair_style lj/class2` | unchanged |
| `buck` | a e^(−r/rho) − c/r⁶ | `a` (E), `rho` (L), `c` (E·L⁶) | `pair_style buck` `A rho C` | unchanged |
| `morse` | d0 [(1 − e^(−alpha (r − r0)))² − 1] | `d0` (E), `alpha` (1/L), `r0` (L) | `pair_style morse` `D0 alpha r0` | the compiled kernel read `D0`, the neighbour-driven one `d0`; both read `d0` |
| `coul/cut` | coulomb qᵢqⱼ / (dielectric (r + delta)) | style `coulomb` (E·L/e²), `dielectric`, `delta` (L), `cutoff` | `pair_style coul/cut` with `delta = 0` (the buffer is molrs's, for MMFF; the LAMMPS writer refuses `delta ≠ 0` and `dielectric ≠ 1`). LAMMPS fixes the constant (`qqr2e`) per `units` | unchanged |
| `lj/charmm` | 4ε[(σ/r)¹² − (σ/r)⁶]·S(r), S CHARMM's switch from `inner` to `cutoff` | `epsilon`, `sigma`, `epsilon14`, `sigma14` (absent → `epsilon`, `sigma`); style `inner`, `cutoff`, `mixing` (default `arithmetic`), `one_four` (`"regular"`, the default, or `"epsilon14"`: what a `special_bonds` 1-4 pair is priced at, see [1-4](#1-4-interactions)) | `pair_style lj/charmm/coul/charmm`, van-der-Waals half; `pair_coeff i j ε σ ε₁₄ σ₁₄` (`one_four = "epsilon14"` has no LAMMPS form) | new |
| `coul/charmm` | coulomb qᵢqⱼ/(dielectric r)·S(r); force (C qᵢqⱼ/r²)·S(r), LAMMPS's switched force, not the gradient | style `coulomb`, `dielectric`, `inner`, `cutoff` | `pair_style lj/charmm/coul/charmm`, Coulomb half (`inner2 outer2` when its cutoffs differ) | new |
| `coul/long/pme` | Ewald-summed coulomb qᵢqⱼ/r | style `coulomb`, `cutoff`, `alpha`, `order`, `grid_*` | `pair_style lj/cut/coul/long` (the real-space half; `kspace_style` states an accuracy, not `alpha`, so the Ewald parameters are neither read nor written, and a LAMMPS-read style prices nothing until they are stated) | `lj/cut/coul/long` reads as this (was a plain `coul/cut`) |
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
| GROMACS funct 4 (periodic), funct 2 (harmonic) | i, j, k, l (GROMACS prices φ(i,j,k,l); pdb2gmx lists a CHARMM improper centre first, an AMBER one centre third) | as written |
| AMBER prmtop, frcmod, GAFF typifier | i, j, K, l (centre third) | as written |
| chamber prmtop `CHARMM_IMPROPERS` (`improper harmonic`) | I (centre), J, K, L | as written |
| OpenMM `<Improper class1 … class4>`, `ordering` default / `amber` | c1 = centre; OpenMM prices φ(c2, c3, c1, c4) | (c2, c3, c1, c4); the writer writes the inverse |
| OpenMM, `ordering="charmm"`, no wildcard | OpenMM prices φ(c1, c2, c3, c4) | as written |
| OpenMM, `ordering="smirnoff"` | three permutations averaged | refused |
| OpenMM `<CustomTorsionForce>` harmonic improper (default ordering `charmm`) | OpenMM prices φ(c1, c2, c3, c4) without a wildcard, φ(c2, c3, c1, c4) with one | as OpenMM prices; the writer refuses a wildcard |
| OpenMM `<RBTorsionForce><Improper>` | — | refused (no improper style is a cosine polynomial) |
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
   The pair styles price a 1-4 pair at their own parameters (and switch),
   times the weight.
2. **Per-type 1-4 Lennard-Jones** — `pair_style lj/charmm/coul/charmm inner
   outer [inner2 outer2]`, `pair_coeff I J epsilon sigma [epsilon14
   sigma14]`. In molrs, a pair style `lj/charmm` (van der Waals half, as
   `lj/cut` is of `lj/cut/coul/cut`) with columns `epsilon`, `sigma`,
   `epsilon14`, `sigma14` and style params `inner`, `cutoff`, `mixing`
   (LAMMPS's default for these styles is `arithmetic`; `epsilon14` /
   `sigma14` mix like `epsilon` / `sigma`, as LAMMPS's `init_one` mixes them);
   self rows per type, explicit cross rows per type pair. Its Coulomb half is
   `coul/charmm` (`coulomb`, `dielectric`, `inner`, `cutoff`). Both switch
   with CHARMM's
   S(r) = (r_c² − r²)²(r_c² + 2r² − 3r_in²)/(r_c² − r_in²)³ between `inner`
   and `cutoff`, at both compile doors; the Lennard-Jones force is the
   gradient, the Coulomb force is LAMMPS's switched force C qᵢqⱼ S/r², which
   is not (`pair_lj_charmm_coul_charmm.cpp`). `lj/charmm/coul/long` is not
   read. GROMACS `[ pairtypes ]` land in `epsilon14`/`sigma14` (self rows
   and cross rows, see [GROMACS topologies](#gromacs-topologies)).

   LAMMPS prices a `special_bonds` 1-4 pair under this style at the
   **regular** `epsilon` / `sigma`; `epsilon14` / `sigma14` reach only the
   `w` pairs of mechanism 3. Engines that price every bond-graph 1-4 pair
   once with its own 1-4 parameters (OpenMM's `<LennardJonesForce>`
   `sigma14`/`epsilon14`, GROMACS `[ pairtypes ]`, a chamber prmtop's 1-4
   table) mean it at `epsilon14` / `sigma14`. The style param `one_four`
   says which:

   | `one_four` | A `special_bonds` 1-4 pair (no `w`, no override) is priced at |
   |---|---|
   | absent or `"regular"` | the regular `epsilon` / `sigma` — LAMMPS |
   | `"epsilon14"` | `epsilon14` / `sigma14` (the cross row's for an explicit pair, else the two types' mixed by `mixing`) |

   Any other value is refused (by the compile doors, `to_section` /
   `from_section` and every reader of a record). LAMMPS's pair style has no
   `"epsilon14"` form, so the IR holds those pairs as per-pair override
   rows (below): **`ForceField.materialize_one_four(frame)`** (Rust
   `ForceField::materialize_one_four`) writes, on every `is_14` row of the
   frame's `pairs` (built when absent) that no `w > 0` dihedral covers,
   `epsilon`, `sigma` = the pair's 1-4 parameters and `lj_scale`,
   `coul_scale` = the `special_bonds` 1-4 weights, keeping a cell already
   set; under `"regular"` (or `lj/cut`) it writes the regular parameters, so
   the energy does not change. A field declaring `"epsilon14"` needs it on
   its frames before compiling: both compile doors refuse a 1-4 pair whose
   1-4 parameters differ from its regular ones and that neither an override
   nor `w` covers, naming `materialize_one_four`, and the LAMMPS writer
   refuses the param (its deck would price the regular parameters).
3. **Per-dihedral weight** — `dihedral_style charmm` `w`: each dihedral
   prices the pair of its own end atoms,

   E₁₄ = w·[4ε₁₄((σ₁₄/r)¹² − (σ₁₄/r)⁶) + C qᵢqⱼ/r],

   with the `lj/charmm` 1-4 parameters of the two types, the Coulomb style's
   C = `coulomb`/`dielectric`, no cutoff and no switch
   (`dihedral_charmm.cpp`; LAMMPS tallies it into `evdwl` and `ecoul`). `w`
   is 1, ½ in six-membered rings, 0 in four- and five-membered rings; a pair
   at the ends of several dihedrals takes the sum of their `w`. As in LAMMPS,
   a field with any `w > 0` is refused unless its `special_bonds` 1-4 weights
   are both 0 (else the pair would be priced twice) and it has a `lj/charmm`
   style and a Coulomb style; `w` outside [0, 1] is refused. `w = 0` (AMBER's
   use of the style) prices LAMMPS's `K[1 + cos(nφ − d)]` alone.

**Per-pair exceptions LAMMPS cannot express** — a GROMACS `[ pairs ]` row
with explicit parameters, an OpenMM `NonbondedForce` exception, an AMBER
dihedral whose `SCEE`/`SCNB` differ from the field's (the prmtop frame
reader writes them, see [AMBER prmtop](#amber-prmtop)) — are per-instance
float columns on the
Frame's `pairs` block, the rows the pair kernels already price (`atomi`,
`atomj`, `is_14`):

| Column | Meaning |
|---|---|
| `epsilon`, `sigma` | the LJ parameters of this pair, in place of the pair style's row or mixing (or the dihedral's `epsilon14` / `sigma14`) |
| `lj_scale` | the van-der-Waals weight of this pair, in place of `special_bonds.lj` for its class (or of `w`) |
| `charge_product` | qᵢqⱼ (e²) of this pair, in place of the atoms' product |
| `coul_scale` | the Coulomb weight of this pair, in place of `special_bonds.coul` (or of `w`) |

A null cell (the column's validity mask) takes the value the pair has
without the row. OpenMM's exception `(chargeProd, sigma, epsilon)` is
`charge_product`, `sigma`, `epsilon` with both scales 1; a GROMACS funct-1
`[ pairs ]` row with parameters is `sigma`, `epsilon` (`lj_scale` 1,
`coul_scale` fudgeQQ), a funct-2 row adds `charge_product` and its own
fudgeQQ; AMBER's
per-dihedral divisors are `lj_scale = 1/SCNB`, `coul_scale = 1/SCEE`. A pair
style other than a 12-6 `lj/cut` or `lj/charmm` and a plain Coulomb
(`coul/cut` with `delta = 0`, `coul/charmm`) has no exception form, and a
field with one beside an exception is refused. LAMMPS can express none of
these per pair, so the LAMMPS writers refuse a frame that carries them,
naming the columns. (molrec retired its `pair14` category for these columns
and `epsilon14` / `sigma14`; a `pair14` table in a record is kept as an
unknown category.)

**Precedence**, per pair and per quantity: **per-pair override > dihedral
charmm `w` > `special_bonds`**. Every pair that carries an override cell or
sits at the ends of a `w > 0` dihedral is priced once, by one exceptions
kernel at both compile doors,

E = lj_w·4ε[(σ/r)¹² − (σ/r)⁶] + coul_w·C qᵢqⱼ/r   (no cutoff, no switch),

where each of ε, σ, qᵢqⱼ, lj_w, coul_w is the override cell if there is one,
else the dihedral's (ε₁₄, σ₁₄, qᵢqⱼ, Σw, Σw) when `w > 0`, else the pair
style's (ε, σ, qᵢqⱼ, `special_bonds` weight of the pair's bond-distance
class). The regular pair kernels price an override pair at weight 0: the
compiled door leaves its `pairs` row out, the neighbour-driven door zeroes
its weight (`PairWeights`). A `w` pair needs no such step — its
`special_bonds` 1-4 weight is 0. A cell is priced only by the style it
belongs to: `epsilon`, `sigma`, `lj_scale` under a Lennard-Jones style,
`charge_product`, `coul_scale` under a Coulomb style. A field without that
style ignores them, so a bonded-only (or Coulomb-only) field compiled on a
frame with materialized 1-4 cells prices no Lennard-Jones. Converting an exception table to a global
weight is not exact (r₁₄ depends on φ and on the other coordinates), so no
reader or writer does it.

## Urey–Bradley

LAMMPS carries Urey–Bradley in one angle style, and so does molrs:
**`angle charmm`**, E = k(θ − theta0)² + k_ub(r₁₃ − r_ub)², parameters `k`
(E/rad²), `theta0` (deg), `k_ub` (E/L²), `r_ub` (L) — `angle_coeff t K theta0
K_ub r_ub`, r₁₃ the distance between the angle's end atoms. It adds no
exclusion (the 1-3 pair is excluded by `special_bonds`). There is no separate
Urey–Bradley category. A field that mixes it with other angle styles is
LAMMPS's `angle_style hybrid harmonic charmm` (`angle_coeff t charmm K theta0
K_ub r_ub`), which the LAMMPS reader and writer read and write; molrs prices
each angle row under the style that defines its type. 0.16 has the kernel,
the LAMMPS reader and writer, the OpenMM reader and writer and the GROMACS
reader and writer; the other
engines' maps below are how their readers map onto the IR:

| Source | `angle charmm` |
|---|---|
| CHARMM `.prm` `ANGLES` `Ktheta Theta0 Kub S0` | as written (CHARMM has no ½) |
| GROMACS `[ angletypes ]` funct 5 `θ₀ k_θ r13 k_UB` (½k forms, nm, kJ/mol) | `k = k_θ/(2·4.184)`, `theta0 = θ₀`, `k_ub = k_UB/(2·418.4)`, `r_ub = 10·r13` |
| OpenMM `<AmoebaUreyBradleyForce><UreyBradley … k d>` (`AmoebaUreyBradleyForceBuilder.addUreyBradleys` adds `HarmonicBondForce.addBond(a1, a3, d, 2*k)`, energy ½·2k(r − d)², so `k` is un-halved) | `k_ub = k/418.4`, `r_ub = 10·d`, joined with the `HarmonicAngleForce` row of the same three labels in either direction; without one, `k = 0`. Checked against OpenMM's energy ([OpenMM XML](#openmm-xml)) |
| chamber prmtop `CHARMM_UREY_BRADLEY` (i, k, type), `…_FORCE_CONSTANT` `K_ub`, `…_EQUIL_VALUE` | every angle of the file `angle charmm`: `k = TK`, `theta0`, and `k_ub = K_ub`, `r_ub` of the term on its end atoms (0, 0 without one) |

## CMAP

The IR follows LAMMPS `fix cmap` (CHARMM's correction map), and the
kernel is `cmap charmm`, a step-for-step port of LAMMPS's
`src/MOLECULE/fix_cmap.cpp`:

- a crossterm names five atoms `(atomi, atomj, atomk, atoml, atomm)`;
  φ = dihedral(atomi, atomj, atomk, atoml), ψ = dihedral(atomj, atomk, atoml,
  atomm) — LAMMPS's `atan2` dihedrals, in degrees, the IUPAC sign (molrs's
  `compute_dihedral`), in [−180°, 180°) with 180° read as −180°;
- a map is an N×N grid (CHARMM and LAMMPS: N = 24, 15° spacing) of energies
  in the force field's energy unit, stored **φ-major**: element `[i][j]` (flat
  index `i·N + j`) is the energy at φ = −180° + i·Δ, ψ = −180° + j·Δ,
  Δ = 360°/N — the order of a CHARMM / LAMMPS `.cmap` file (each `# phi`
  block is one φ row of N ψ values);
- the energy between grid points is LAMMPS's bicubic interpolation, with the
  derivatives LAMMPS precomputes from cubic splines.

A `cmap` style's row holds one map as its `grid` array parameter (`f64`
N×N, the layout above; the `forcefield` section's `grid` column is
`f64[T, N, N]`), and a frame's `cmaps` block lists the crossterms
(`atomi` … `atomm`, `type`).

### The interpolation, exactly

**Node derivatives** (`set_map_derivatives`). With x_m = ⌊N/2⌋, the map is
extended periodically to 2N × 2N: `T[p][q] = E[(p − x_m) mod N][(q − x_m) mod N]`,
so index `p` is the angle −180° + (p − x_m)·Δ and the extension covers
[−360°, 360°). `spline(y)` is LAMMPS's *natural* cubic spline on spacing Δ:
second derivatives `y''` with `y''₀ = y''_{n−1} = 0` and, for i = 1 … n−2,

```text
p = 1/(y''_{i−1} + 4),  y''_i = −p,
u_i = ((6y_{i+1} − 12y_i + 6y_{i−1})/Δ² − u_{i−1})·p,   u₀ = 0
y''_j = y''_j·y''_{j+1} + u_j   for j = n−2 … 0.
```

Every row `T[k]` is splined along ψ. For a node (φ index i, ψ index j), the
row splines give, down all 2N rows k, the value `Y_k = T[k][j']` and the slope
`D_k = (T[k][j'+1] − T[k][j'])/Δ − (Δ/3)·T''[k][j'] − (Δ/6)·T''[k][j'+1]`
(j' = j + x_m); `Y` and `D` are splined along φ, and with i' = i + x_m

```text
∂E/∂φ   = (Y_{i'+1} − Y_{i'})/Δ − (Δ/3)·Y''_{i'} − (Δ/6)·Y''_{i'+1}
∂E/∂ψ   = D_{i'}
∂²E/∂φ∂ψ = (D_{i'+1} − D_{i'})/Δ − (Δ/3)·D''_{i'} − (Δ/6)·D''_{i'+1}
```

per degree (per degree² for the cross term). The doubled map keeps the
natural end conditions half a period from every node they are read at, so
the slopes are those of a periodic spline to within its decay.

**Patch** (`bc_coeff`, `bc_interpol`). The cell is
`c_φ = ⌊(φ + 180°)/Δ⌋`, `c_ψ = ⌊(ψ + 180°)/Δ⌋`; its corners, counter-clockwise
from (c_φ, c_ψ), wrap mod N. The 16 coefficients `c_ab` are Numerical Recipes'
`bcucof` weight matrix applied to the corner values, Δ·∂E/∂φ, Δ·∂E/∂ψ and
Δ²·∂²E/∂φ∂ψ, and with t = (φ − φ_{c_φ})/Δ, u = (ψ − ψ_{c_ψ})/Δ

```text
E = Σ_{a,b=0..3} c_ab tᵃ uᵇ,
∂E/∂φ = (180/π)/Δ · Σ a·c_ab tᵃ⁻¹ uᵇ,   ∂E/∂ψ = (180/π)/Δ · Σ b·c_ab tᵃ uᵇ⁻¹
```

(per radian). At a node E is the grid value; E and its slopes are continuous
across every cell edge and across ±180°.

**Forces** (`post_force`). F = −(∂E/∂φ)·∇φ − (∂E/∂ψ)·∇ψ with LAMMPS's own
∇φ, ∇ψ expressions, so the crossterm is distributed onto the five atoms
exactly as LAMMPS distributes it: atomi feels φ only, atomm ψ only, and the
three shared atoms both; the forces sum to zero and exert no torque.

Two LAMMPS behaviours are kept because they change energies: a crossterm
with a degenerate dihedral plane (any of the four cross products
|r_ij × r_jk|² below 10⁻⁴ Å⁴) contributes **nothing**; and the map is a true
2-D function — no sum of 1-D torsions reproduces it. One is generalised: LAMMPS
fixes N = 24 and at most six maps, the kernel takes any N ≥ 2 and any number
of maps (LAMMPS stores the derivative grids rotated by N/2 and finds their cell
from φ wrapped to [0°, 360°); molrs stores them unrotated and reads them at the
value cell — the same numbers but for float ties on a cell edge).

### Crossterms and LAMMPS files

- `assign_cmaps(frame, ff)` (Rust `molrs::ff::assign_cmaps`) builds the
  `cmaps` block: five atoms whose dihedrals `(a, b, c, d)` and `(b, c, d, e)`
  are both rows of `dihedrals` (either stored direction) and whose atom types
  equal a cmap row's `itom … mtom` **forward** — never reversed, since
  reading the five atoms backwards swaps φ and ψ.
- A LAMMPS `fix cmap` file reads (`read_lammps_cmap`,
  `LammpsFfReader::read_cmap_str`) into rows named `"1"` … `"K"` — map `t`
  is crossterm type `t` — and writes (`write_lammps_cmap`,
  `LammpsFfWriter::write_cmap_str`) the `cmaps` labels' grids in label id
  order, in CHARMM's layout: CHARMM's own file comes back line for line.
- The data file's `N crossterms` header line and `CMAP` section
  (`index type a1 … a5`) are the frame's `cmaps` block (`type_id` = map
  index), both ways.
- The include writer emits `fix cmap all cmap <cmap_file>` and
  `fix_modify cmap energy yes`; that fix must reach LAMMPS before
  `read_data <data> fix cmap crossterm CMAP`. The include reader reads a
  `fix cmap` line's file relative to the include.

OpenMM's `CMAPTorsionForce` stores `energy[i + N·j]` at φ = 2πi/N, ψ = 2πj/N
(origin 0, φ fastest — `CMAPTorsionForce::addMap`): its reader maps element
`(i, j)` to molrs `[(i + N/2) mod N][(j + N/2) mod N]` (each index shifted by
N/2, the axes swapped into φ-major), and refuses an odd N, which puts no
OpenMM node on −180°; the writer is the inverse. OpenMM takes its node slopes
from periodic splines where LAMMPS splines the doubled map with natural ends,
so energies off the grid points differ at the spline's end effect: 3·10⁻¹²
relative on CHARMM36's alanine map at an ACE-ALA-NME conformer
([OpenMM XML](#openmm-xml)). OpenMM's generator matches a crossterm's five
types forward or backward; `assign_cmaps` matches forward only. GROMACS `[ cmaptypes ]` lists CHARMM's grid in
the IR's layout (φ-major from −180°, GROMACS's `cmap_setup_grid_index`), so
its reader keeps element `i·N + j` at `[i][j]`; GROMACS's own interpolation
is CHARMM's too, and the CHARMM dipeptide's two crossterms price as GROMACS
2025.3 prices them to 6 × 10⁻¹⁶ (see [How this is checked](#how-this-is-checked)).

A prmtop stores a map as ParmEd's `CmapType.grid` — CHARMM's parameter-file
order, φ-major from −180° — in `CHARMM_CMAP_PARAMETER_nn` (a chamber file)
or `CMAP_PARAMETER_nn` (ff19SB), and the prmtop reader takes it as written.
sander's CMAP energy on a CHARMM36 alanine dipeptide and on an ff19SB one
off the grid points equals molrs's and LAMMPS's to 1e-14 relative (see
[AMBER prmtop](#amber-prmtop)): sander interpolates as LAMMPS does.

## Torsion forms and their exact conversions

Every Class-I torsion and dihedral-angle improper style above is a finite
Fourier series in φ — the signed dihedral of the atoms as stored, the angle
every molrs kernel and LAMMPS compute:

    E(φ) = Σₙ₌₀ aₙ cos nφ + bₙ sin nφ

That series is the intermediate of every conversion between forms
(`molrs::ff::forcefield::torsion`). Each form **embeds** exactly (the series
is the same function of φ, constant included) and **projects** back exactly
or refuses, naming the term that prevents it. Rows of several styles on one
quadruple are one torsion: their series add.

| Form | aₙ, bₙ of the form (n ≥ 1) | constant a₀ | image condition (n ≥ 1) |
|---|---|---|---|
| `dihedral periodic` (LAMMPS `fourier`), Σₘ kₘ[1 + cos(nₘφ − γₘ)] | aₙ += kₘ cos γₘ, bₙ += kₘ sin γₘ | Σₘ kₘ (fixed) | none — every series; back: kₙ = √(aₙ² + bₙ²), γₙ = atan2(bₙ, aₙ) |
| `dihedral charmm`, k[1 + cos(nφ − d)] | as one `periodic` term; `w` is not a torsion parameter and is carried beside the series | k (fixed) | one order |
| `improper periodic`, k[1 + cos(nφ − γ)] | as one `periodic` term | k (fixed) | one order |
| `dihedral harmonic`, `improper cvff`, k[1 + d cos nφ] | aₙ = k d | k (fixed) | one order, bₙ = 0; back: k = \|aₙ\|, d = sign aₙ |
| `dihedral opls`, ½Σ kₙ[1 ± cos nφ] | a₁ = k₁/2, a₂ = −k₂/2, a₃ = k₃/2, a₄ = −k₄/2 | ½Σkₙ (fixed) | bₙ = 0, n ≤ 4 |
| `dihedral class2` (torsion part), Σₙ₌₁³ kₙ[1 − cos(nφ − φₙ)] | aₙ = −kₙ cos φₙ, bₙ = −kₙ sin φₙ | Σkₙ (fixed) | n ≤ 3 |
| `dihedral multi/harmonic`, Σₙ₌₁⁵ Aₙ cosⁿ⁻¹φ | cosᵏφ = 2⁻ᵏ Σⱼ C(k, j) cos((k − 2j)φ) | free (exact) | bₙ = 0, n ≤ 4; back: cos nφ = Tₙ(cos φ) |
| `dihedral nharmonic`, Σᵢ₌₁ᴺ Aᵢ cosⁱ⁻¹φ | as `multi/harmonic` | free (exact) | bₙ = 0 |
| Ryckaert–Bellemans (GROMACS funct 3, OpenMM `RBTorsionForce`), Σₙ₌₀⁵ Cₙ cosⁿ(φ − 180°) | `nharmonic` with Aₙ₊₁ = (−1)ⁿ Cₙ (cos(φ − 180°) = −cos φ) | free (exact) | bₙ = 0, n ≤ 5 |

Every periodicity must be an integer, as LAMMPS requires of every style
here. A "sine term" (bₙ ≠ 0) is a phase other than 0° or 180°; phases on
multiples of 90° are evaluated exactly, so a 180° phase is bₙ = 0, not
1.2·10⁻¹⁶.

**The constant term.** A constant shifts no force, so two torsions are the
same physics iff their series agree for n ≥ 1; the canonical series drops
a₀. The polynomial forms (`multi/harmonic`, `nharmonic`, RB) carry a₀ back
exactly; every other form fixes its constant by its other parameters, and
reproduces the series up to that offset. In particular ΣCₙ = 0 is **not** an
image condition of RB → OPLS once the constant is dropped — only C₅ = 0 is.
The GROMACS reader (`[ dihedraltypes ]` funct 3) and the OpenMM XML reader
(`<RBTorsionForce>`) read RB as the polynomial it is (`multi/harmonic`, or
`nharmonic` when C₅ ≠ 0), which holds every RB row exactly, constant
included; neither refuses ΣCₙ ≠ 0 or C₅ ≠ 0.

The familiar chains are instances:

- **RB ↔ `multi/harmonic`**: A₁ = C₀, A₂ = −C₁, A₃ = C₂, A₄ = −C₃, A₅ = C₄
  (C₅ = 0); with C₅ ≠ 0 the exact target is `nharmonic` (N = 6).
- **RB ↔ OPLS** (GROMACS manual Eqs. 200–201): k₁ = −2C₁ − 3C₃/2,
  k₂ = −C₂ − C₄, k₃ = −C₃/2, k₄ = −C₄/4; back C₀ = k₂ + (k₁ + k₃)/2,
  C₁ = (−k₁ + 3k₃)/2, C₂ = −k₂ + 4k₄, C₃ = −2k₃, C₄ = −4k₄, C₅ = 0.
- **OPLS ↔ `periodic`**: one term per non-zero kₙ, `k = kₙ/2`, phase 0° at
  odd n and 180° at even n (for kₙ > 0).

**Outside the image.**

- `improper harmonic`, K(|φ| − chi0)², and LAMMPS `dihedral quadratic`,
  K(φ − φ0)², are not finite Fourier series and are refused. A periodic
  improper k[1 + cos(nφ − γ)] and a harmonic one agree only to second order
  about the minimum: **K = n²k/2** in LAMMPS's un-halved K (K = 2k for
  AMBER's n = 2, γ = 180°; the `k_h = n²k` sometimes quoted is the ½-form
  harmonic ½k_h χ²), with chi0 the minimum nearest 0 and, back,
  γ = n·chi0 + 180°. The quartic terms differ by −k n⁴ δ⁴/24.
- `dihedral class2`'s cross terms (`mbt`, `ebt`, `at`, `aat`, `bb13`) couple
  the torsion to bonds and angles: outside the Class-I IR.
- `improper fourier` (molrs `uff_inversion`) and `mmff_oop` price an
  out-of-plane (Wilson) angle, not a dihedral.

In Rust:

```rust
use molrs::ff::forcefield::torsion::{FourierSeries, TorsionForm, TorsionRefusal};

// A stored row → its series (exact, constant included).
let row = TorsionForm::from_params("dihedral", "opls", &params)?;
let series = row.to_series()?;
// Several rows on one quadruple → one series; compare without the constant.
let total: FourierSeries = rows.iter().map(|r| r.to_series()).sum::<Result<_, _>>()?;
assert_eq!(total.canonical(), other.canonical());
// Series → a target style, or the reason it cannot be.
match TorsionForm::from_series("dihedral", "multi/harmonic", &series) {
    Ok(form) => form.to_params(),
    Err(TorsionRefusal::SineTerm { n, .. }) => todo!("phase off 0/180° at order {n}"),
    Err(other) => todo!("{other}"),
};
```

The per-form types (`Periodic`, `Charmm`, `CosineTerm`, `SignedCosine`,
`Opls`, `Class2`, `MultiHarmonic`, `NHarmonic`, `RyckaertBellemans`,
`ImproperHarmonic`) carry the same maps one form at a time.
`FourierSeries::chopped(tol)` zeroes coefficients below a tolerance first,
for input rounded on print. A test evaluates every registered kernel on
random geometries against its form's series, so the algebra and the kernels
cannot drift.

## OpenMM XML

`OplsXmlReader` (Python `read_opls_xml`, and `read_forcefield_xml` for a
file in OpenMM's schema) reads OpenMM's `<ForceField>` — its own CHARMM36,
AMBER and OPLS-AA ports and the foyer / molpy packs — and
`XmlForceFieldWriter` (`write_forcefield_xml`) writes the inverse, each
number in the shortest form that reads back to the same `f64` unless a
`precision` is given. The IR's definitions are LAMMPS's; OpenMM's are
converted at the boundary:

| OpenMM | IR | Conversion |
|---|---|---|
| `<HarmonicBondForce><Bond length k>` | `bond harmonic` | `r0 = 10·length`, `k = k/(2·418.4)` (OpenMM's ½k) |
| `<HarmonicAngleForce><Angle angle k>` | `angle harmonic` | `theta0` in degrees, `k = k/(2·4.184)` |
| `<AmoebaUreyBradleyForce><UreyBradley k d>` | `angle charmm`, joined with its angle row | `k_ub = k/418.4`, `r_ub = 10·d` ([Urey–Bradley](#ureybradley)) |
| `<PeriodicTorsionForce><Proper k_m periodicity_m phase_m>` | `dihedral periodic` | `k_m/4.184`, phases in degrees; the writer also writes `dihedral charmm` (`w = 0`), `harmonic` and `class2` here, term for term, constant included |
| `<PeriodicTorsionForce><Proper c0..c3>` (CL&P / foyer) | `dihedral opls` | `k_n = c_{n−1}/4.184` |
| `<PeriodicTorsionForce><Improper>` | `improper periodic` (one term) | stored in the order OpenMM prices ([Improper atom order](#improper-atom-order)) |
| `<RBTorsionForce><Proper c0..c5>` | `dihedral multi/harmonic` (C₅ = 0) or `nharmonic` (N = 6) | `Aₙ₊₁ = (−1)ⁿ Cₙ/4.184`, constant included; the writer writes `multi/harmonic`, `nharmonic` (N ≤ 6) and `opls` here |
| `<CustomTorsionForce energy="k*(theta-theta0)^2">` `<Improper>` | `improper harmonic` | `k/4.184`; OpenMM's θ is signed and LAMMPS's χ = \|φ\|, which agree at `theta0 = 0` only — another `theta0` is refused (CHARMM36's two `theta0 = π` rows among them) |
| `<CustomTorsionForce energy="k*(abs(theta)-theta0)^2">` | `improper harmonic` | `chi0` = theta0 in degrees (the writer's form when some `chi0 ≠ 0`) |
| `<CMAPTorsionForce><Map>`, `<Torsion map>` | `cmap charmm` | [CMAP](#cmap) |
| `<NonbondedForce coulomb14scale lj14scale><Atom charge sigma epsilon>` | `pair lj/cut` (`mixing` = the root's foyer `combining_rule`, else `arithmetic`) + `pair coul/cut`; `charge` on `atom full` | `sigma` × 10, `epsilon` ÷ 4.184; `special_bonds` `[0, 0, scale]` |
| `<LennardJonesForce lj14scale><Atom sigma epsilon [sigma14 epsilon14]>`, `<NBFixPair>` | `pair lj/charmm` (`arithmetic`, NBFIX as cross rows) + `pair coul/charmm`, the `<NonbondedForce>` beside it (its `epsilon` 0) giving the charges | 1-4: `special_bonds` `[0, 0, lj14scale]` / `[0, 0, coulomb14scale]`, and `one_four = "epsilon14"` when a type's 1-4 parameters differ ([1-4](#1-4-interactions)); an NBFIX row is OpenMM's 1-4 parameters for its pair too, LAMMPS's two-number cross row |

Rows key on `class{n}` or `type{n}` as written; OpenMM's wildcard (an empty
attribute) is `""`, and a row naming neither, which OpenMM ignores, is
refused. Every Coulomb style states OpenMM's constant (`ONE_4PI_EPS0` =
332.06371329919216 kcal·Å/(mol·e²), 9.9·10⁻⁹ above LAMMPS `real`'s
`qqr2e`); the writers do not carry it (each engine fixes its own). OpenMM's
cutoffs and switching are `createSystem` arguments, so no style read from a
file has a `cutoff` (or `inner`): the caller states them, for `NoCutoff` a
cutoff beyond every pair.

**Refused**, by name: `<Script>` / `<InitializationScript>`, every
`Custom*Force` other than the harmonic improper, a `<Proper>` under it,
an RB `<Improper>`, `ordering="smirnoff"`, a multi-term periodic improper,
an odd CMAP size, a wildcard Urey–Bradley row, an `<NBFixPair>` of a type
with itself, a `<NonbondedForce>` with non-zero `epsilon` beside a
`<LennardJonesForce>` (OpenMM prices both), and every other force (AMOEBA
multipoles, GBSA, Drude, …). The writer refuses what OpenMM's tags cannot
hold: a style outside the table (`improper cvff`, `bond morse`, …),
`dihedral charmm` `w ≠ 0`, `special_bonds` other than `[0, 0, s]`,
`sixthpower` mixing or `geometric` with cross rows, a cross row with its own
1-4 parameters, `lj/charmm` under `one_four = "regular"` with 1-4 parameters
of its own at a non-zero weight (OpenMM would use them), a Coulomb style
with `dielectric ≠ 1` or `delta ≠ 0`, charges on some atom types only, a
shifted or Mie `lj/cut`, a force field in units other than `real`, and two
types OpenMM's generator would match on the same labels with other
parameters (bonds, angles, propers and crossterms either way round,
impropers by their centre and the other three in any order; a proper's
periodic and RB rows on one quartet count, since OpenMM adds both) — the
same row twice is written once. A type without a `class` (a prmtop's, a
LAMMPS file's) is written as its own class, which OpenMM requires of every
`<Type>`. `<Residues>`, `<Patches>` and `<Info>` carry no parameters and are
skipped; placeholder atom types the reader makes for classes are not written.

A force-field XML is typed: OpenMM's generators find each term from the
atom types of a residue template. A system written for OpenMM therefore
needs residue templates whose charges are the atoms' (the field written
without type charges, so `<UseAttributeFromResidue name="charge"/>`), and
its rows must be what the generators find: a `dihedral` row on four atoms
that are no bonded chain (OPLS-AA's impropers as GROMACS funct 1) is the
same function as `improper periodic` over the same atoms and goes to OpenMM
as one; an `improper periodic` row's two outer atoms must be in the order
OpenMM's AMBER rule puts them (by element, then index). The equivalence
check ([Cross-engine equivalence](#cross-engine-equivalence)) builds the
templates from the frame and holds OpenMM's energies to molrs's.

## AMBER prmtop

AMBER is read, never written (no prmtop writer). The prmtop force-field
reader (`AmberPrmtopFfReader`) and frame reader (`read_amber_prmtop`) name
every type and row alike; a chamber prmtop (ParmEd's `chamber`, `%FLAG
CTITLE`) reads through the same pair.

| prmtop | IR |
|---|---|
| `BOND_*` `RK`, `ANGLE_*` `TK` (no ½), radians | `bond harmonic`, `angle harmonic` (`angle charmm` in a chamber file, see [Urey–Bradley](#ureybradley)) |
| `DIHEDRAL_*` `PK`, `PN`, phase; the rows of one quartet and each negative-`PN` chain are one torsion | `dihedral periodic` `k<m>`, `periodicity<m>`, `phase<m>` (degrees), terms sorted by periodicity; a second distinct set of terms on one type quartet (tleap reuses a quartet's first match, so two torsions of one quartet can differ) is the type `<quartet>@<n>` |
| an improper (negative 4th pointer) | `improper periodic` in AMBER's order; a multi-term one (several rows, or a chain) is one frame row and one type `<quartet>@<n>` per term, as LAMMPS `cvff` holds one term |
| a phase within 0.004 rad of ±π | ±180° exactly, as sander's `rdparm` (tleap writes π as `3.14159400`) |
| `CHARMM_IMPROPERS` `K_ψ (ψ − ψ₀)²` | `improper harmonic` `k = K_ψ`, `chi0 = ψ₀`, centre first; ψ₀ other than 0° / 180° refused (LAMMPS prices \|ψ\|) |
| `CHARMM_CMAP_*` / `CMAP_*` | `cmap charmm`, a type per map named by the five atom types (qualified `@<residue>` of the Cα when one name stands for two maps, as ff19SB's do); the frame's `cmaps` block |
| `LENNARD_JONES_ACOEF/BCOEF` via ICO | `lj/cut` (`lj/charmm` in a chamber file) self rows; a cross row where the entry is not Lorentz–Berthelot (NBFIX) |
| `LENNARD_JONES_14_ACOEF/BCOEF` (chamber) | `lj/charmm` `epsilon14` / `sigma14` (cross rows where not Lorentz–Berthelot) and `one_four = "epsilon14"` when the table differs from the regular one |
| `CHARGE` | ÷ 18.2223, `coul/cut` at 332.0522173; ÷ √332.0716 and `coul/charmm` at 332.0716 in a chamber file |
| `SCEE_SCALE_FACTOR` / `SCNB_SCALE_FACTOR` per torsion type | `special_bonds` 1-4 = 1/divisor most 1-4 rows carry; the frame's `pairs` give every 1-4 pair weighted otherwise its `coul_scale` / `lj_scale` |
| `AMBER_ATOM_TYPE` | the type name; `<name>~<class>` where one name stands for two LJ classes or masses (a chamber file cuts CHARMM's types to four characters) |

sander prices a 1-4 pair once per proper row whose 3rd pointer is not
negative, at that row's `1/SCEE`, `1/SCNB` (never an improper's, whatever its
pointer). The field's weights are the divisors most such rows carry; the
frame reader's `pairs` block lists the pairs whose summed weight differs —
another divisor (GLYCAM's 1.0 beside ff14SB's 1.2 / 2.0), a pair two rows
list, a 1-4 pair of the topology no row lists (weight 0, or 1 if the
exclusion list leaves it out) — with only the differing cells set.
`intramolecular_pairs` keeps those cells when it builds the full pair list.
A chamber file's 1-4 Lennard-Jones (`one_four = "epsilon14"`) reaches a
frame through `ForceField.materialize_one_four`. Build the full list first
(`intramolecular_pairs`, which keeps the frame reader's cells), then
materialize: `materialize_one_four` builds a list only when the frame has
no `pairs`, and the frame reader's `pairs` hold only the odd 1-4 pairs.

sander clamps the cosine of a harmonic angle to ±0.999 before taking its
arccosine, so it prices an angle beyond 177.44° as 177.44°; LAMMPS and molrs
price the IR's `k (θ − theta0)²` at the angle itself. A prmtop with a
near-linear angle (an `sp` carbon, θ₀ ≈ 180°) therefore prices its angle term
differently in sander, and only there.

Refused by name: polarizable (`IPOL > 0`), 12-6-4 (`LENNARD_JONES_CCOEF`),
non-zero 10-12 (`HBOND_ACOEF/BCOEF`, a negative ICO), perturbed, solvent-cap
and `IFBOX = 3` files; a 1-4 row on a negative-`PN` chain (sander prices the
pair once per chained term, and its Coulomb at a factor unlike any other 1-4
pair's); a 1-4 row on a bonded or angle-end pair; two terms of one improper
with one periodicity; a Urey–Bradley term on no angle or on several.

## GROMACS topologies

`GromacsTopFfReader` reads a topology's directives into a force field
(`read`) or a whole `.top` into the force field and a typed frame
(`read_system`; Python `molrs.ff.read_gromacs_system`); the writer
(`GromacsTopFfWriter`) is the inverse of the directive map
(`write_str`) and of `read_system` (`write_system_str`). Every row is
exact; GROMACS's ½k forms are halved into LAMMPS's `K`, nm → Å, kJ → kcal,
degrees stay degrees. Every Coulomb style states GROMACS's own constant
(`GROMACS_COULOMB`, its `ONE_4PI_EPS0` from CODATA 2018: 332.06371329919205
kcal·Å/(mol·e²), 9.9·10⁻⁹ above LAMMPS `real`'s; 0.15 stated LAMMPS's).

| GROMACS | IR |
|---|---|
| `[ defaults ]` comb-rule 2 / 3 | `mixing` `arithmetic` / `geometric` on the Lennard-Jones style (comb-rule 1, C6/C12, refused) |
| `[ defaults ]` gen-pairs, fudgeLJ, fudgeQQ | `special_bonds` `lj [0, 0, fudgeLJ]`, `coul [0, 0, fudgeQQ]` (gen-pairs `no`: `lj` 1-4 = 1, no pair is generated) |
| `[ atomtypes ]` V W | `atom full` type + Lennard-Jones self row `sigma = 10·V`, `epsilon = W/4.184` |
| `[ nonbond_params ]` funct 1 | explicit cross row |
| `[ pairtypes ]` funct 1 | `lj/charmm` + `coul/charmm`, `one_four = "epsilon14"`: `epsilon14 = ε/fudgeLJ`, `sigma14 = σ` on the self rows, and cross rows where the mix of the self rows would not give the pair GROMACS's 1-4 parameters (to 10⁻¹²) — so `special_bonds` × LJ(ε₁₄, σ₁₄) is GROMACS's 1-4 energy for every type pair. A pairtype equal to the generated pair changes nothing (`lj/cut` stays) |
| `[ bondtypes ]` funct 1 / 3 | `bond harmonic` (`k = k_b/2`) / `bond morse` |
| `[ angletypes ]` funct 1 / 5 | `angle harmonic` / `angle charmm` ([Urey–Bradley](#ureybradley)) |
| `[ dihedraltypes ]` funct 1 | `dihedral periodic`, one term |
| `[ dihedraltypes ]` funct 9 | `dihedral periodic`: the consecutive rows on equal labels are the terms `k<m>`, `periodicity<m>`, `phase<m>` of one type, in file order |
| `[ dihedraltypes ]` funct 3 (RB) | `dihedral multi/harmonic` `aₙ₊₁ = (−1)ⁿ Cₙ`, constant included; `dihedral nharmonic` (N = 6) when C₅ ≠ 0 |
| `[ dihedraltypes ]` funct 5 (Fourier) | `dihedral opls`, `kₙ = Cₙ` |
| `[ dihedraltypes ]` funct 2 | `improper harmonic`, `K = k_ξ/2`, `chi0 = ξ₀` ∈ {0°, 180°} (the signed GROMACS form equals `K(|φ| − chi0)²` there and nowhere else, so other ξ₀ are refused); atoms as written |
| `[ dihedraltypes ]` funct 4 | `improper periodic`; atoms as written (AMBER order) |
| `[ cmaptypes ]` funct 1 | `cmap charmm`, the grid unchanged in layout ([CMAP](#cmap)) |
| `X` | the empty wildcard |

The writer writes `dihedral periodic` with several terms as consecutive
funct-9 rows, `multi/harmonic` and `nharmonic` (N ≤ 6) as funct 3, `opls` as
funct 5, `dihedral harmonic` / `improper cvff` and `dihedral charmm` with
`w = 0` as periodic rows at phase 0° / 180°, `cmap charmm` as `[ cmaptypes ]`,
and the 1-4 parameters of an `lj/charmm` declared `one_four = "epsilon14"`
as `[ pairtypes ]` exactly where GROMACS would generate other ones. It
refuses `dihedral charmm` with `w > 0` (GROMACS prices a 1-4 pair by
`[ pairs ]`, never by a dihedral), `epsilon14` / `sigma14` on an `lj/charmm`
that does not price 1-4 pairs with them, two types GROMACS would read as
one (same labels, same function-code table), and a force field in units
other than `real`. `dihedral class2`'s torsion goes out as funct-9 rows at
phase φₙ + 180° (k[1 − cos(nφ − φₙ)] = k[1 + cos(nφ − φₙ − 180°)]). No
Coulomb constant is written: GROMACS prices at its own, as LAMMPS and OpenMM
do at theirs.

**Systems.** `read_system` types each molecule row as GROMACS's own lookup
does — bonds, angles, Fourier dihedrals and cmaps by exact bond types
(cmaps forward only), the other dihedrals by the first row with the most
non-wildcard matches, either way, with all of a funct-9 row's terms; a row
with parameters of its own (an OPLS-AA `improper_*` macro, expanded) gets a
type of its own, `<labels>@gmx_<n>`. Its frame holds `atoms`, `bonds`,
`angles`, `dihedrals` (funct 1, 9, 3, 5), `impropers` (funct 2, 4), `cmaps`,
`constraints` (`[ constraints ]`, `[ settles ]`; `r0` in Å), `exclusions`
and `pairs`: every pair GROMACS prices — per molecule, and every pair of
two molecules (up to `MAX_ATOMS_FOR_A_FULL_PAIR_LIST` atoms; above it a
neighbour list, `compile_typed`, finds those) — the `[ pairs ]` rows
flagged `is_14`. A `[ pairs ]` row with parameters carries them as per-pair
overrides — funct 1 `sigma`, `epsilon`, `lj_scale` 1, `coul_scale` fudgeQQ;
funct 2 also `charge_product` and its own fudgeQQ — which LAMMPS cannot hold.
Virtual sites, restraints, polarization and free-energy B states have no IR
form and are refused by name. A system whose field reads with
`one_four = "epsilon14"` (CHARMM's pairtypes) compiles once its 1-4 pairs
are written out as per-pair rows, `ForceField.materialize_one_four(frame)`.

`write_system_str(ff, frame)` writes the directives (no bonded
`[ *types ]` tables: each row carries its type's parameters on its line,
one funct-9 line per periodic term, so no lookup can pick another type;
`[ cmaptypes ]` stays, and a crossterm must be one GROMACS's lookup finds)
and one `[ moleculetype ]` (`nrexcl` 3) per molecule — a bond-graph
component, a run of consecutive atoms — with `[ atoms ]` (each atom's
charge and mass), `[ bonds ]`, `[ pairs ]`, `[ angles ]`, `[ dihedrals ]`
(propers and impropers), `[ cmap ]`, `[ exclusions ]` and `[ constraints ]`,
then `[ molecules ]`. `[ pairs ]` lists the frame's 1-4 pairs: funct 1, with
`σ ε` when only the override cells `epsilon` / `sigma` differ from the
generated pair, else funct 2 `fudgeQQ qᵢqⱼ 1 σ lj_scale·ε` — so every
override cell holds. `[ exclusions ]` holds each pair of a molecule beyond
three bonds the frame does not price, so GROMACS prices exactly the
frame's `pairs` (built by `intramolecular_pairs` when absent). Refused by
name: a priced pair within three bonds that is no 1-4 pair (or a 1-4 pair
beyond them), override cells without `epsilon` and `sigma`, a crossterm
GROMACS would give another grid, a molecule whose atoms are not
consecutive or a row across two molecules, and every force-field refusal
above. A topology `read_system` reads from it is the system: written again,
it is the same file.

## GAFF and GAFF2

`GaffTypifier` (Python `molrs.ff.typifier.GaffTypifier(parameter_set=
"gaff" | "gaff2")`) matches a molecule whose atoms carry GAFF types —
`AtdTypifier` with the same `parameter_set` stamps them — against the
`gaff.dat` / `gaff2.dat` tables compiled into molrs (GAFF 1.81, GAFF2 2.2.30,
AmberTools 26.1), and writes the IR: `atom full` (mass), `pair lj/cut`
(σ = 2·R\*/2^(1/6), ε), `pair coul/cut` at AMBER's 332.0522173,
`bond harmonic` and `angle harmonic` (`k = K`, θ₀ in degrees),
`dihedral periodic` (`k<m> = PK/IDIVF`), `improper periodic` (AMBER order,
centre third) and `special_bonds` ½ / ⅚ (SCNB 2, SCEE 1.2). Charges are not
a GAFF parameter: a charge model writes `atoms.charge`.

What the table lacks is estimated as AmberTools estimates it, and each
estimate carries `estimated`, `estimate_penalty`, `estimate_method` and
`estimate_analog`:

- **Torsions** follow parmchk2's `chk_torsion`: equivalent-type rows, the
  wildcard row `X-j-k-X` (a parameter, not an estimate), equivalent-type
  wildcard rows, then the cheapest corresponding-type row and wildcard row,
  scored as parmchk2 scores them.
- **Impropers** follow parmchk2's `chk_improper` (estimates at the centres
  `PARMCHK.DAT` flags as planar) and then tleap: an improper wherever tleap
  finds a row for a triple of an atom's neighbours, its atoms in the order
  tleap gives them.
- **Bonds and angles** follow parmchk2's `chk_bond` / `chk_angle`:
  equivalent-type rows, the cheapest corresponding-type row (an angle within
  `THRESHOLD_BA`), then, for an angle, Wang's empirical `K_θ` and the mean
  θ₀ of the `A-B-A` and `C-B-C` rows, for its own, equivalent and
  corresponding types. A corresponding type scores parmchk2's columns:
  `bl` + `blf` at a bond end, `ba` + `baf` at an angle end, (`cba` +
  `cbaf`)·`WEIGHT_BA_CTR` at its vertex (caffeine's `c-cc-na` is
  `c2-cc-na` at 2.6).

parmchk2 searches the molecule in atom and bond order and reuses a name's
first estimate (an improper estimate is itself a row later impropers can
copy), so a name can be estimated differently in two molecules (aspirin's
ester carbon takes `c3-o -c -oh`, ethyl acetate's the `X -X -c -o` amide
term). An estimated term is therefore named with its analog and penalty,
`<types>@<analog>_<penalty>` (`c3-o-c-os@c3.o.c.oh_8.5`; the improper
default `@default_0.0`), so one output force field holds both.

Checked against AmberTools 26.1:

- antechamber + parmchk2 + tleap + sander on 127 molecules (neutral and
  charged; aromatic, heteroaromatic, conjugated, strained, S / P / B /
  halogen chemistry) under both sets, with antechamber's atom types: every
  bond, angle, torsion and improper row (atoms, atom order, every parameter),
  every estimate's analog and penalty (all 634 frcmod rows), every atom's
  σ / ε, and every energy term at perturbed coordinates — bond, angle,
  dihedral with impropers, 1-4 and other van der Waals and Coulomb — agree.
  The angle energy agrees to the 4·10⁻⁷ by which tleap's π (3.141594) moves
  θ₀, except at a near-linear angle: sander clamps cos θ to ±0.999 (θ never
  above 177.44°), LAMMPS and molrs do not (phenylacetylene's `c1` / `cg`
  angles, 0.055 kcal/mol).
- parmchk2 on 800 type graphs built to need bond and angle estimates: every
  bond and angle it writes — parameters, analog, penalty, and every `ATTN`
  (a missing term here) — agrees.
- `AtdTypifier` against antechamber's atom types: identical on 125 of the
  127, given the same Kekulé structure. antechamber re-derives bond orders
  from connectivity alone; for azulene and cyclooctatetraene it settles on
  the other Kekulé structure, and the `cc` / `cd` colouring of the
  conjugated system flips with it.

A term neither the table nor parmchk2's search reaches is an error where
parmchk2 writes a zero marked `ATTN, need revision`.

## Parameters as frame columns

`ForceField.materialize_params(frame, prefix=…)` (Rust
`ForceField::materialize_params`) writes the parameters this force field
gives each row of a typed frame next to the row, as columns
`<prefix><parameter>`:

| Block | Rows priced by | Columns (`lj/cut` + `harmonic` + `periodic` field) |
|---|---|---|
| `bonds`, `angles`, `dihedrals`, `impropers`, `cmaps` | the type its `type` names, under the one style of the category that defines it | `k`, `r0`; `k`, `theta0`; `k<m>`, `periodicity<m>`, `phase<m>`; `k`, `periodicity`, `phase` |
| `atoms` | `atoms.type`: every `atom` style's type, every `pair` style's self row | `mass`; `epsilon`, `sigma` |

Nothing in it is per style: the columns are the numeric parameters the
field stores for each type, in the IR's units (degrees, `K` without a ½, the
field's `units`), so a style added to the IR is written by the same call. A
row whose type lacks a parameter another row's type has — a two-term torsion
beside a three-term one, `estimate_penalty` on an estimated term only — is a
null cell (`Block.validity(column)`), not a zero. String and array
parameters, a pair style without per-type rows (`coul/cut`) and cross rows
(NBFIX, which have no per-atom form; the style's `mixing` combines the self
rows) are not written; charges are frame data unless the field stores them
per type. A column already present under a written name is replaced. It
returns block → columns written, and refuses a row whose type no style (or
two styles) of its category defines.

The use it is for is a **reference field**: type a molecule with one field
and read another's parameters beside the same rows — e.g. GAFF2's bonded
terms and Lennard-Jones as the reference of a learned Class-I field,

```python
labelled = molrs.ff.typifier.AtdTypifier(parameter_set="gaff2").typify(mol)
gaff2 = molrs.ff.typifier.GaffTypifier(parameter_set="gaff2")
frame = gaff2.typify(labelled).to_frame()
gaff2.forcefield().materialize_params(frame, prefix="gaff2_")
frame["bonds"]["gaff2_k"], frame["atoms"]["gaff2_sigma"]
```

## Engine maps at a glance

| Engine | bond `k` | angle `k`, `theta0` | phases | impropers |
|---|---|---|---|---|
| LAMMPS | `K` | `K`, deg | deg | as written |
| GROMACS (`.top`/`.itp`) | `k_b/2`, kJ→kcal, nm→Å | `k_θ/2`, deg | deg | as written |
| OpenMM XML | `k/2`, kJ→kcal, nm→Å | `k/2`, rad→deg | rad→deg | (c1..c4) ↔ (c2, c3, c1, c4); see [OpenMM XML](#openmm-xml) |
| AMBER prmtop | `RK` | `TK`, rad→deg | rad→deg (±π snapped) | AMBER order |
| chamber prmtop | `RK` | `TK`, rad→deg (`angle charmm`) | rad→deg | AMBER order; `CHARMM_IMPROPERS` centre first |
| AMBER frcmod (writer) | `RK = k` | `TK = k`, deg | deg | AMBER order |
| GAFF / GAFF2 tables | `K` | `K`, deg | deg | AMBER order |
| OPLS-AA table (GROMACS `oplsaa.ff`) | `k_b/2` | `k_θ/2`, deg | — | — |

Each engine fixes its Coulomb constant, and each reader states its engine's
on the Coulomb style (`coulomb`, kcal·Å/(mol·e²)); no writer writes one, so
a field is priced by an engine at that engine's:

| Engine | Coulomb constant | Relative to LAMMPS `real` |
|---|---|---|
| LAMMPS `real` (`qqr2e`) | 332.06371 | — |
| OpenMM (`ONE_4PI_EPS0`, CODATA 2018) | 332.06371329919216 | + 9.9·10⁻⁹ |
| GROMACS 2025 (`ONE_4PI_EPS0`, CODATA 2018, its own expression) | 332.06371329919205 | + 9.9·10⁻⁹ (one ulp below OpenMM's) |
| AMBER (charges × 18.2223) | 332.05221729 | − 3.5·10⁻⁵ |
| CHARMM (chamber prmtop) | 332.0716 | + 2.4·10⁻⁵ |

## Cross-engine equivalence

One molecule per force-field family, read from its native format into the
IR, written by molrs to every engine format that can hold it, and priced by
every engine — the source's own included — at three configurations
(`ff::equivalence_check`, `scripts/ff_equivalence_check.sh`):

| Source | Family, molecule | Native format (engine) |
|---|---|---|
| `ff14sb` | AMBER ff14SB, ACE-PHE-NME | prmtop (sander) |
| `gaff2` | GAFF2, the modXNA DMA fragment | prmtop (sander) |
| `chamber` | CHARMM36 (Urey–Bradley, CHARMM impropers, CMAP, 1-4 table), alanine dipeptide + α-D-glucose | chamber prmtop (sander) |
| `charmm36` | CHARMM36 (Urey–Bradley, harmonic impropers, CMAP, `sigma14`/`epsilon14`, an NBFIX row), ACE-ALA-NME | OpenMM XML (OpenMM) |
| `oplsaa` | OPLS-AA (RB, geometric mixing, funct-1 impropers), ACE-ALA-ALA-NME | GROMACS `.top` (GROMACS) |

Each source's IR is written as a LAMMPS data file and include, an OpenMM
`<ForceField>` XML (with residue templates built from the frame) and a
GROMACS topology (`write_system_str`); each engine prices each file and the
source itself — LAMMPS `run 0`; OpenMM 8.6.1 `Reference`, `NoCutoff`, one
force group per term; GROMACS 2025.3 double precision, `mdrun -rerun`,
plain cut-off past every pair; pysander (AmberTools 26.1, `cut = 999`) —
at the source's coordinates plus a seeded 0.05 Å Gaussian, on the 0.01 Å
grid a `.gro` holds exactly. Terms: bond, angle (with Urey–Bradley),
dihedral, improper, CMAP, van der Waals and Coulomb (each with its 1-4
pairs), the total, and the forces as ΣF·v (v a seeded Gaussian) and Σ|F|².
Every number is held to molrs's energy of the IR form that engine read,
its Coulomb at that engine's own constant (the energy is linear in it; the
constants are in [Engine maps at a glance](#engine-maps-at-a-glance)).

Where an engine reads a field in another exact form, molrs writes that
form, and prices it as the source to 10⁻¹² (`each_engine_form_prices_as_the_source`):
CHARMM's 1-4 table goes to LAMMPS as `special_bonds` 0 and one zero-`K`
`dihedral charmm` row of `w` = 1 per 1-4 pair; to OpenMM, OPLS-AA's funct-1
impropers (dihedral rows on atoms that are no bonded chain) go as the
`improper periodic` they are, and charges per atom in the residue
templates.

The worst error over the three configurations and every term — relative to
the term, or to 1 kcal/mol for a term below it — and of the forces (ΣF·v to
|F||v|, Σ|F|² relative), each engine against molrs:

| Source | native | LAMMPS | OpenMM | GROMACS |
|---|---|---|---|---|
| `ff14sb` | E 2·10⁻⁹ ᵃ, F 5·10⁻⁹ | E 1·10⁻¹⁴, F 2·10⁻¹⁵ | E 6·10⁻¹⁵, F 9·10⁻¹⁵ | E 1·10⁻¹¹, F 2·10⁻¹¹ |
| `gaff2` | E 2·10⁻¹⁰ ᵃ, F 8·10⁻¹² | E 4·10⁻¹⁵, F 5·10⁻¹⁶ | E 5·10⁻¹⁵, F 6·10⁻¹⁶ | E 1·10⁻⁹ ᵇ, F 4·10⁻¹⁰ |
| `chamber` | E 6·10⁻¹⁴, F 4·10⁻¹⁵ | E 9·10⁻¹⁴, F 6·10⁻¹⁶ | E 8·10⁻⁸ ᶜ, F 5·10⁻¹¹ | E 1·10⁻⁹ ᵇ, F 5·10⁻⁹ |
| `charmm36` | E 2·10⁻¹², F 1·10⁻¹³ | E 9·10⁻¹⁴, F 2·10⁻⁵ ᵈ | E 2·10⁻¹², F 1·10⁻¹³ | E 5·10⁻⁹ ᵇ, F 6·10⁻¹⁰ |
| `oplsaa` | E 1·10⁻⁹ ᵇ, F 2·10⁻⁹ | E 1·10⁻¹⁵, F 2·10⁻¹⁷ | E 1·10⁻¹⁴, F 9·10⁻¹⁵ | E 1·10⁻⁹ ᵇ, F 2·10⁻⁹ |

Every exact path is within 10⁻¹¹ but where the engine computes something
else, each below the acceptance bar of 10⁻⁶:

- ᵃ sander reads the prmtop's eight-digit `LENNARD_JONES_ACOEF/BCOEF`
  entries; molrs and the other engines mix the self terms (van der Waals).
- ᵇ GROMACS prices 1-4 pairs from cubic-spline tables (≈10⁻⁹ kcal/mol on the
  1-4 van der Waals and Coulomb).
- ᶜ OpenMM takes its CMAP node slopes from periodic splines, the IR (LAMMPS
  `fix cmap`) from natural splines over the doubled map: 8·10⁻⁸ kcal/mol on
  a CMAP term of −0.026 kcal/mol.
- ᵈ LAMMPS's `improper_style harmonic` clamps sin χ at 0.001 in its force
  (`improper_harmonic.cpp`, `SMALL`), so its forces are not the gradient of
  its energy within 0.057° of planar; one configuration holds a CHARMM
  improper at χ = 4.7·10⁻⁴ rad. Its energies, and every other
  configuration's forces, are molrs's to 10⁻¹³.

The Coulomb constants differ by up to 3.5·10⁻⁵ (AMBER's against the others'),
which is not an error of any engine: each prices at its own, and the table
holds each to molrs at that constant. Every file written also reads back
into the IR it was written from — written again it is the same file, and
molrs prices it the same (`every_written_file_reads_back_as_written`) — and
every source's IR, and each engine's form of it, persists through a record.
The engines' numbers are pinned in `molrs/src/ff/testdata/equivalence/
engines.tsv`, so the check runs without the engines;
`scripts/ff_equivalence_check.sh --pin` reruns them on a compute node.

## Completeness

Every style molrs registers a kernel for, and every field-level setting,
against every engine format molrs reads or writes and the record (molrec
v2): **✓** exact — priced, read or written as the IR means it, held by the
tests `ff::completeness` names (a numbered note bounds a ✓ or states the
convention it holds under); **refused** — an error naming the style or
setting, never a silent drop or an approximation; **—** the format has no such
thing. The table is generated: `ff::completeness` holds it with the tests
behind every cell, and fails when a registered kernel has no row, a row names
no kernel, a cited test does not exist, or this table differs from its
rendering (`MOLRS_WRITE_COMPLETENESS=1 cargo mrs-test --
ff::completeness::the_guide_holds_the_generated_table` rewrites it).

<!-- completeness:begin (generated by ff::completeness; do not edit) -->
| Style or setting | kernel | LAMMPS read | LAMMPS write | OpenMM read | OpenMM write | GROMACS read | GROMACS write | prmtop read | molrec v2 |
|---|---|---|---|---|---|---|---|---|---|
| `bond harmonic` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `bond morse` | ✓ | ✓ | ✓ | refused [1] | refused [2] | ✓ | ✓ | — | ✓ |
| `bond class2` | ✓ | refused [3] | refused [3] | — | refused [3] | — | refused [3] | — | ✓ |
| `bond mmff_bond` | ✓ | — | refused [4] | — | refused [4] | — | refused [4] | — | ✓ |
| `bond uff_bond` | ✓ | — | refused [4] | — | refused [4] | — | refused [4] | — | ✓ |
| `angle harmonic` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `angle charmm` | ✓ | ✓ | ✓ | ✓ [5] | ✓ | ✓ | ✓ | ✓ | ✓ |
| `angle class2` | ✓ | refused [3] | refused [3] | — | refused [3] | — | refused [3] | — | ✓ |
| `angle mmff_angle` | ✓ | — | refused [4] | — | refused [4] | — | refused [4] | — | ✓ |
| `angle mmff_stbn` | ✓ | — | refused [4] | — | refused [4] | — | refused [4] | — | ✓ |
| `angle uff_angle` | ✓ | — | refused [4] | — | refused [4] | — | refused [4] | — | ✓ |
| `dihedral periodic` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `dihedral charmm` | ✓ | ✓ | ✓ | — | ✓ [6] | — | ✓ [7] | — | ✓ |
| `dihedral opls` | ✓ | ✓ | ✓ | ✓ | ✓ [8] | ✓ | ✓ | — | ✓ |
| `dihedral multi/harmonic` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — | ✓ |
| `dihedral nharmonic` | ✓ | ✓ | ✓ | ✓ | ✓ [9] | ✓ | ✓ [10] | — | ✓ |
| `dihedral harmonic` | ✓ | ✓ | ✓ | — | ✓ | — | ✓ | — | ✓ |
| `dihedral class2` | ✓ | refused [11] | refused [12] | — | ✓ | — | ✓ | — | ✓ |
| `dihedral mmff_torsion` | ✓ | — | refused [4] | — | refused [4] | — | refused [4] | — | ✓ |
| `dihedral uff_torsion` | ✓ | — | refused [4] | — | refused [4] | — | refused [4] | — | ✓ |
| `improper harmonic` | ✓ | ✓ | ✓ [13] | ✓ [14] | ✓ [15] | ✓ [16] | ✓ [17] | ✓ | ✓ |
| `improper cvff` | ✓ | ✓ | ✓ | — | refused [18] | — | ✓ | — | ✓ |
| `improper periodic` | ✓ | — | ✓ [19] | ✓ | ✓ [20] | ✓ | ✓ | ✓ | ✓ |
| `improper mmff_oop` | ✓ | — | refused [4] | — | refused [4] | — | refused [4] | — | ✓ |
| `improper uff_inversion` | ✓ | — | refused [4] | — | refused [4] | — | refused [4] | — | ✓ |
| `cmap charmm` | ✓ | ✓ | ✓ | ✓ [21] | ✓ [22] | ✓ | ✓ | ✓ | ✓ |
| `pair lj/cut` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `pair lj/charmm` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `pair coul/cut` | ✓ | ✓ | ✓ [23] | ✓ | ✓ [24] | ✓ | ✓ [25] | ✓ | ✓ |
| `pair coul/charmm` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `pair coul/long/pme` | ✓ | ✓ [26] | ✓ [27] | — | refused [28] | — | refused [29] | — | ✓ |
| `pair lj/class2` | ✓ | refused [3] | refused [3] | — | refused [3] | — | refused [3] | — | ✓ |
| `pair buck` | ✓ | refused [30] | refused [30] | — | refused [30] | — | refused [30] | — | ✓ |
| `pair morse` | ✓ | refused [30] | refused [30] | — | refused [30] | — | refused [30] | — | ✓ |
| `pair thole` | ✓ | refused [31] | refused [31] | — | refused [31] | — | refused [31] | — | ✓ |
| `pair coul/tt` | ✓ | refused [31] | refused [31] | — | refused [31] | — | refused [31] | — | ✓ |
| `pair mmff_vdw` | ✓ | — | refused [4] | — | refused [4] | — | refused [4] | — | ✓ |
| `pair uff_lj` | ✓ | — | refused [4] | — | refused [4] | — | refused [4] | — | ✓ |
| `special_bonds` | ✓ | ✓ | ✓ | ✓ [32] | ✓ [33] | ✓ | ✓ [33] | ✓ | ✓ |
| `per-pair overrides (pairs epsilon, sigma, lj_scale, charge_product, coul_scale)` | ✓ | — | refused [34] | — | — | ✓ | ✓ | ✓ | ✓ |
| `lj/charmm one_four = "epsilon14"` | ✓ | — | ✓ [35] | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `mixing arithmetic` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `mixing geometric` | ✓ | ✓ | ✓ | ✓ [36] | ✓ [37] | ✓ | ✓ | — | ✓ |
| `mixing sixthpower` | ✓ | ✓ | ✓ | — | refused [38] | — | refused [39] | — | ✓ |
| `cross rows (NBFIX)` | ✓ | ✓ | ✓ | ✓ | ✓ [40] | ✓ | ✓ | ✓ | ✓ |
| `lj/cut shift` | ✓ | ✓ | ✓ | — | refused [41] | — | refused [42] | — | ✓ |
| `lj/cut n, m (Mie)` | ✓ | refused [43] | refused [44] | — | refused [45] | — | refused [46] | — | ✓ |
| `Coulomb constant` | ✓ | ✓ | — | ✓ | — | ✓ | — | ✓ | ✓ |
| `cutoff, inner (switch)` | ✓ | ✓ | ✓ | — | — | — | — | — | ✓ |
| `units presets (real, metal, lj)` | ✓ | ✓ | ✓ | — | refused [47] | — | refused [47] | — | ✓ |

1. OpenMM has a Morse bond only as a CustomBondForce, which the reader refuses
2. no HarmonicBondForce form
3. Class II, outside the Class-I IR (LAMMPS's class2 styles carry cross terms the IR has no form for)
4. molrs's own definition (a typifier's), no engine style
5. a wildcard Urey–Bradley row is refused
6. w = 0 (a periodic term); w ≠ 0 refused
7. w = 0 (funct 9); w > 0 refused
8. as RB, which reads back as multi/harmonic: the same series, constant included
9. N ≤ 6 (RB's C0 … C5); above refused
10. N ≤ 6; above refused
11. LAMMPS's dihedral class2 carries its mbt/ebt/at/aat/bb13 cross terms, outside the Class-I IR
12. LAMMPS's dihedral class2 needs its cross-term lines; the torsion alone is not written
13. LAMMPS clamps sin χ at 0.001 in its force: within 0.057° of planar its forces are not its energy's gradient
14. CustomTorsionForce k(θ−θ0)² at θ0 = 0, or k(|θ|−θ0)²; another signed θ0 refused
15. a wildcard endpoint refused (OpenMM would re-order the atoms)
16. funct 2 at ξ0 ∈ {0°, 180°}; another ξ0 refused
17. chi0 ∈ {0°, 180°}; another refused
18. OpenMM prices an improper over the dihedral with the centre third; cvff's starts at the centre
19. as cvff, at a phase of 0° or 180°; another refused
20. OpenMM orders the two outer atoms it finds first by element and index (its AMBER rule); a system whose stored order differs prices another dihedral
21. odd N refused; OpenMM takes its node slopes from periodic splines (≤ 10⁻⁷ kcal/mol off the IR's)
22. as OpenMM read: its interpolation ≤ 10⁻⁷ kcal/mol off the IR's
23. delta = 0 and dielectric = 1; the Coulomb constant is LAMMPS's own
24. the Coulomb constant is OpenMM's own
25. the Coulomb constant is GROMACS's own
26. lj/cut/coul/long: cutoff and constant; its kspace_style states an accuracy, not an alpha, so pricing is refused until the Ewald parameters are stated
27. the real-space lj/cut/coul/long; stated Ewald parameters refused (molrs's smooth PME is not LAMMPS's PPPM)
28. the long-range method is a createSystem argument
29. the long-range method is an .mdp setting
30. not a Class-I nonbonded form; LAMMPS's pair style of the name would hold it, which no reader or writer maps
31. polarizable (Drude) screening, outside the Class-I IR
32. [0, 0, s] (OpenMM's 14scale)
33. [0, 0, s]; another refused
34. LAMMPS has no per-pair 1-4 parameters
35. refused as such; its exact LAMMPS form is special_bonds 0 and one zero-K dihedral charmm row of w = 1 per 1-4 pair
36. foyer's combining_rule, which OpenMM's own app.ForceField ignores
37. foyer's combining_rule (OpenMM's own app.ForceField ignores it); with cross rows refused
38. OpenMM mixes arithmetically
39. no GROMACS comb-rule
40. a cross row with 1-4 parameters of its own refused (OpenMM prices an NBFIX 1-4 pair with the NBFIX row)
41. OpenMM's NonbondedForce has no shifted Lennard-Jones
42. a modifier is an .mdp setting
43. pair_style mie/cut is not read
44. pair_style mie/cut is not written
45. OpenMM's Lennard-Jones is 12-6
46. GROMACS's Lennard-Jones is 12-6
47. the conversions are from real
<!-- completeness:end -->

The styles marked outside the Class-I IR (Class II, Buckingham, Morse pair,
Thole, Tang–Toennies) and molrs's own typifier styles (MMFF94, UFF) are
priced and persisted, and no engine reader or writer maps them. Every
Class-I style of molrec's registry is a molrs kernel of the same name, but
two, which `compile` refuses by name ("no kernel for style …"):
`dihedral rb` (every molrs reader reads Ryckaert–Bellemans as the
`multi/harmonic` / `nharmonic` polynomial it is, constant included) and
`improper trefoil` (the SMIRNOFF average over three orderings; the OpenMM
reader refuses `ordering="smirnoff"`).

## How this is checked

- Every engine against every source, and every reader / writer against the
  IR: [Cross-engine equivalence](#cross-engine-equivalence) and
  [Completeness](#completeness).
- Each style has a hand-value test against the LAMMPS manual's formula.
- `ff::convention_invariance` holds the 0.16 energies of GAFF-, OPLS-AA-,
  MMFF94- and UFF-typed acetanilide and of a GROMACS-, OpenMM- and
  LAMMPS-read hand molecule to the values molrs 0.15.1 computed on the same
  inputs, term by term, at 1e-12 relative; every one matches bit for bit,
  except the OpenMM improper (the fix above), which now equals the
  GROMACS-read value of the same improper and the hand value of OpenMM's
  formula, and the OpenMM- and GROMACS-read Coulomb terms, which are
  0.15.1's times the ratio of the engine's own constant to LAMMPS's.
- `cmap charmm` against LAMMPS `fix cmap` (`run 0`, CHARMM36's alanine map
  and its transpose on three crossterms of an eight-atom backbone, files
  written by molrs; `scripts/lammps_cmap_check.sh`): E = −1.25779219530854869
  kcal/mol, molrs 1.1 × 10⁻¹⁵ relative off; 22 of the 24 force components
  bit for bit, the other two 2 × 10⁻¹⁶ off.
- The 1-4 mechanisms run through LAMMPS (`run 0`, `lj/charmm/coul/charmm
  3.5 4.2 3.0 5.0` on a charged seven-atom alcohol whose pairs fall inside,
  across and beyond both switches): `special_bonds charmm` with every `w` =
  1, every `w` = ½ with one dihedral listed twice, and `special_bonds` ½ /
  ⅚ with `w` = 0. Every `evdwl`, `ecoul`, `ebond`, `eangle`, `edihed`, `pe`
  matches molrs to ≤ 2.3e-15 relative (`evdwl` and `ecoul` bit for bit), and
  every force component to 1e-10 (`ff::one_four`). `special_bonds` ½ equals
  per-pair scales ½ equals per-pair parameters ε/2, qᵢqⱼ/2; `w` = 1 equals
  per-pair rows of ε₁₄, σ₁₄; `compile` equals `compile_typed`.
- GROMACS-read systems against GROMACS 2025.3 (double precision, `mdrun
  -rerun`, energies from the .edr) and LAMMPS (`run 0` on molrs's data file
  and include), `ff::forcefield::readers::gromacs::engine_check`,
  `scripts/gromacs_engine_check.sh`: ACE-ALA-ALA-NME under charmm27
  (Urey–Bradley, two CMAP crossterms, `[ pairtypes ]`, a
  `[ nonbond_params ]` row), amber99sb-ildn (funct 9, funct 4, fudge ½ / ⅚),
  oplsaa (funct 3, funct-1 impropers) and AMBER with `[ pairs ]` rows of
  their own, plain cut-off at 2.5 nm. molrs equals LAMMPS to ≤ 1e-14 on
  every term; it equals GROMACS to ≤ 2e-14 on bond, angle (with UB),
  dihedral, improper and CMAP, ≤ 1.3e-13 on LJ, ≤ 1.6e-9 on LJ-14 (GROMACS
  prices 1-4 pairs from cubic-spline tables) and ≤ 1e-11 on Coulomb (the
  reader states GROMACS's own constant; LAMMPS's Coulomb is held at the
  constants' ratio). CHARMM's
  1-4 pairs run in LAMMPS as zero-`K` `dihedral charmm` rows with `w` = 1
  beside `special_bonds` 0.
- The LAMMPS-read hand molecule run through LAMMPS (`run 0`) gives the
  same per-term energies as molrs to ≤ 2e-13 relative: bond
  0.162750104621288, angle 1.35959339751695, dihedral 0.692979891423841,
  improper 0.431717012867386, van der Waals 1.22012795938037, Coulomb
  −10.7066619897381 kcal/mol.
- OpenMM XML end to end (`ff::openmm_check`; `scripts/openmm_xml_check.py`
  prices with OpenMM 8.6.1's own `app.ForceField`, `Reference` platform,
  `NoCutoff`, each force in its own group; `scripts/openmm_xml_check.sh`
  runs LAMMPS `run 0` on the data file and include molrs writes): ACE-ALA-NME
  with CHARMM36 (Urey–Bradley, harmonic impropers, CMAP, `sigma14` /
  `epsilon14`, an NBFIX row — LAMMPS deck in its `dihedral charmm` `w` form),
  with AMBER ff14SB, and 1-propanol with OPLS-AA (RB, geometric mixing).
  kcal/mol:

  | Case | Term | OpenMM | LAMMPS | molrs |
  |---|---|---|---|---|
  | charmm | bond | 36.945047858708044 | 36.94504785870815 | 36.94504785870815 |
  | charmm | angle (incl. UB) | 17.449423871456695 | 17.449423871456627 | 17.44942387145663 |
  | charmm | dihedral | 6.19239592482837 | 6.1923959248283635 | 6.192395924828361 |
  | charmm | improper | 2.715834463096705 | 2.7158344630968245 | 2.715834463096736 |
  | charmm | cmap | 0.43270566338002386 | 0.43270566337874106 | 0.43270566337874106 |
  | charmm | vdW (incl. 1-4) | 0.4406266917233237 | 0.4406266917233339 | 0.440626691723334 |
  | charmm | Coulomb | −24.645577292730245 | −24.64557704786598 | −24.645577292730195 |
  | amber | bond | 32.27593176355788 | 32.275931763558 | 32.275931763558 |
  | amber | angle | 15.661664106340247 | 15.661664106340188 | 15.66166410634019 |
  | amber | dihedral | 13.096829168247133 | 13.096829168247146 | 13.096829168247142 |
  | amber | improper | 1.030690929439921 | 1.03069092943996 | 1.0306909294399393 |
  | amber | vdW | 2.995530281438277 | 2.995530281438281 | 2.9955302814382825 |
  | amber | Coulomb | −36.45693850206638 | −36.4569381398514 | −36.45693850206634 |
  | opls | bond | 25.068019234759834 | 25.06801923475996 | 25.06801923475996 |
  | opls | angle | 3.651654362069225 | 3.6516543620692508 | 3.6516543620692508 |
  | opls | dihedral (RB) | −0.26085298030739323 | −0.26085298030739323 | −0.26085298030739335 |
  | opls | vdW | 0.10191738393823163 | 0.10191738393823166 | 0.1019173839382316 |
  | opls | Coulomb | 0.9415611810268669 | 0.9415611716720687 | 0.9415611810268705 |

  molrs is OpenMM to ≤ 2.3·10⁻¹⁴ relative, the CMAP to 3·10⁻¹² (its
  interpolation, [CMAP](#cmap)); molrs is LAMMPS to ≤ 3.3·10⁻¹⁴ except the
  Coulomb, which differs from LAMMPS by exactly the two Coulomb constants'
  ratio (9.9·10⁻⁹), as OpenMM does. The XML molrs writes back from each
  force field, priced by OpenMM, gives the source's energies to ≤ 6·10⁻¹⁵,
  and read → write → read is the identity.

- The prmtop readers against sander and LAMMPS (`ff::forcefield::readers::
  prmtop_check`, `scripts/prmtop_check.sh`): six prmtops AmberTools 26.1
  builds — ff14SB ACE-PHE-NME, a GAFF2 molecule, the same with two
  multi-term impropers, GLYCAM glucose beside an ff14SB dipeptide
  (non-uniform SCEE/SCNB), a CHARMM36 chamber file (Urey–Bradley, CHARMM
  impropers, CMAP, 1-4 table) and ff19SB ACE-ALA-NME (CMAP) — at perturbed
  coordinates. Every term (bond, angle, Urey–Bradley, dihedral with AMBER
  impropers, CHARMM improper, CMAP, 1-4 vdW, 1-4 Coulomb, vdW, Coulomb)
  matches pysander to ≤ 6.2e-9 relative and LAMMPS `run 0` on the files
  molrs writes to ≤ 3.2e-14 (the chamber 1-4 vdW through
  `materialize_one_four`, and in LAMMPS with `epsilon14`/`sigma14` in the
  regular slots). The van-der-Waals 1e-9 against sander is the
  file's eight-digit `LENNARD_JONES_ACOEF/BCOEF`: sander uses each printed
  off-diagonal entry, molrs and LAMMPS mix the self terms; every other
  term agrees to 1e-14.
- `angle charmm` through LAMMPS (`run 0`), as `angle_style charmm` on three
  atoms and as `angle_style hybrid harmonic charmm` on five: `pe` 0.024871552479721934
  and 0.35736516873047092 kcal/mol, which molrs reproduces bit for bit, and
  the per-atom forces to ≤ 2e-15 kcal/(mol·Å).
