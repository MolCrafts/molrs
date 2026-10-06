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
| `lj/cut` | C ε [(σ/r)ⁿ − (σ/r)ᵐ], C = n/(n−m)·(n/m)^(m/(n−m)); 4ε[(σ/r)¹² − (σ/r)⁶] at n = 12, m = 6 | `epsilon` (E), `sigma` (L); style `cutoff`, `mixing`, `n`, `m`, `shift` | `pair_style lj/cut` (n ≠ 12 or m ≠ 6: `mie/cut`; `shift`: `pair_modify shift yes`; `mixing`: `pair_modify mix`) | unchanged |
| `lj/class2` | ε [2(σ/r)⁹ − 3(σ/r)⁶] | `epsilon`, `sigma` | `pair_style lj/class2` | unchanged |
| `buck` | a e^(−r/rho) − c/r⁶ | `a` (E), `rho` (L), `c` (E·L⁶) | `pair_style buck` `A rho C` | unchanged |
| `morse` | d0 [(1 − e^(−alpha (r − r0)))² − 1] | `d0` (E), `alpha` (1/L), `r0` (L) | `pair_style morse` `D0 alpha r0` | the compiled kernel read `D0`, the neighbour-driven one `d0`; both read `d0` |
| `coul/cut` | coulomb qᵢqⱼ / (dielectric (r + delta)) | style `coulomb` (E·L/e²), `dielectric`, `delta` (L), `cutoff` | `pair_style coul/cut` with `delta = 0` (the buffer is molrs's, for MMFF; the LAMMPS writer refuses `delta ≠ 0` and `dielectric ≠ 1`). LAMMPS fixes the constant (`qqr2e`) per `units` | unchanged |
| `lj/charmm` | 4ε[(σ/r)¹² − (σ/r)⁶]·S(r), S CHARMM's switch from `inner` to `cutoff` | `epsilon`, `sigma`, `epsilon14`, `sigma14` (absent → `epsilon`, `sigma`); style `inner`, `cutoff`, `mixing` (default `arithmetic`) | `pair_style lj/charmm/coul/charmm`, van-der-Waals half; `pair_coeff i j ε σ ε₁₄ σ₁₄` | new |
| `coul/charmm` | coulomb qᵢqⱼ/(dielectric r)·S(r); force (C qᵢqⱼ/r²)·S(r), LAMMPS's switched force, not the gradient | style `coulomb`, `dielectric`, `inner`, `cutoff` | `pair_style lj/charmm/coul/charmm`, Coulomb half (`inner2 outer2` when its cutoffs differ) | new |
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
   read. GROMACS `[ pairtypes ]` will land in the cross rows'
   `epsilon14`/`sigma14`.
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
dihedral whose `SCEE`/`SCNB` differ from the field's (the prmtop reader
refuses a non-uniform pair today) — are per-instance float columns on the
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
`[ pairs ]` row with parameters is `sigma`, `epsilon` (`lj_scale` 1); AMBER's
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
`special_bonds` 1-4 weight is 0. Converting an exception table to a global
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
each angle row under the style that defines its type. 0.16 has the kernel and
the LAMMPS reader and writer; the other engines' maps below are the
convention their readers will follow:

| Source | `angle charmm` |
|---|---|
| CHARMM `.prm` `ANGLES` `Ktheta Theta0 Kub S0` | as written (CHARMM has no ½) |
| GROMACS `[ angletypes ]` funct 5 `θ₀ k_θ r13 k_UB` (½k forms, nm, kJ/mol) | `k = k_θ/(2·4.184)`, `theta0 = θ₀`, `k_ub = k_UB/(2·418.4)`, `r_ub = 10·r13` |
| OpenMM `<AmoebaUreyBradleyForce><UreyBradley … k d>` (OpenMM adds a `HarmonicBondForce` term with `2k`, so `k` is un-halved) | `k_ub = k/418.4`, `r_ub = 10·d`, joined with the `HarmonicAngleForce` row of the same classes |

## CMAP

The convention is LAMMPS `fix cmap` (CHARMM's correction map), and the
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
(origin 0, φ fastest): its reader maps element `(i, j)` to molrs `[(i + N/2) mod N]
[(j + N/2) mod N]`; OpenMM interpolates with a natural periodic bicubic
spline, so energies off the grid points differ from LAMMPS's at the
interpolation's accuracy. GROMACS `[ cmaptypes ]` lists CHARMM's grid; its
reader must be checked against a GROMACS energy before it is trusted.

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
The GROMACS and OPLS-XML readers, which keep the absolute energy, still
require ΣCₙ = 0 to within 10⁻⁴ kJ/mol beside C₅ = 0.

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
- The LAMMPS-read hand molecule run through LAMMPS (`run 0`) gives the
  same per-term energies as molrs to ≤ 2e-13 relative: bond
  0.162750104621288, angle 1.35959339751695, dihedral 0.692979891423841,
  improper 0.431717012867386, van der Waals 1.22012795938037, Coulomb
  −10.7066619897381 kcal/mol.
- `angle charmm` through LAMMPS (`run 0`), as `angle_style charmm` on three
  atoms and as `angle_style hybrid harmonic charmm` on five: `pe` 0.024871552479721934
  and 0.35736516873047092 kcal/mol, which molrs reproduces bit for bit, and
  the per-atom forces to ≤ 2e-15 kcal/(mol·Å).
