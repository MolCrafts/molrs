# Spec: RDKit ETKDGv3 Port to molrs-embed

**Status**: draft for review
**Owner**: molrs-embed maintainer
**Prerequisites**: SMARTS matcher (see §2.1), UFF parameter table (see §2.2)
**Target version**: ETKDGv3 (Wang et al. 2020) — optional srETKDGv3 extension
**Delivery**: two milestones M1 (DG-only) → M2 (full ETKDGv3)

---

## 1. Summary

Replace the current `molrs-embed` pipeline (fragment assembly + custom force field + L-BFGS + rotor search) with a faithful Rust port of RDKit's ETKDGv3 conformer-generation algorithm. Output quality target: ≥ 84% of a CSD benchmark set within 1.0 Å heavy-atom RMSD of experimental crystal structures, matching the number reported by Riniker & Landrum (2015). The port is **algorithmically equivalent** to RDKit — same stages, same force-field terms, same constants, same CSD torsion library — but is explicitly **not bit-exact** (floating-point ordering, RNG, and eigensolver differ).

## 2. Motivation

Current output is chemically unrealistic: sp3 torsions uniformly distributed, stereo centers occasionally invert, macrocycles collapse. Root causes identified in algorithmic analysis:

1. 3D-only eigendecomposition embedding — 3D space does not allow continuous chirality correction during minimization; RDKit uses 4D.
2. No experimental torsion (ET) library — the defining component of ETKDG is absent.
3. Chirality volume enforced only in post-embedding force field, not during embedding.
4. Single-stage minimization without the three-pass structure that separates distance/chirality enforcement from torsion bias.
5. Rotor search with random sampling substitutes for a structured torsion prior — noise does not reproduce gauche preferences or macrocycle conformations.

References:
- Riniker S., Landrum G.A. *Better Informed Distance Geometry: Using What We Know To Improve Conformation Generation.* J. Chem. Inf. Model. 2015, 55(12), 2562–2574. DOI: 10.1021/acs.jcim.5b00654
- Wang S., Witek J., Landrum G.A., Riniker S. *Improving Conformer Generation for Small Rings and Macrocycles Based on Distance Geometry and Experimental Torsional-Angle Preferences.* J. Chem. Inf. Model. 2020, 60(4), 2044–2058. DOI: 10.1021/acs.jcim.0c00025
- Havel T.F. *Distance Geometry: Theory, Algorithms, and Chemical Applications.* Encyclopedia of Computational Chemistry, Wiley, 1998.
- Crippen G.M., Havel T.F. *Distance Geometry and Molecular Conformation.* Research Studies Press, 1988.

## 3. Goals and Non-goals

### Goals (in scope)

- Port ETKDGv3 embed pipeline: bounds matrix → triangle smoothing → random metrization → 4D metric-matrix embedding → three-stage minimization → chirality/E-Z verification → retry loop.
- Port the CSD-derived torsion preference library (254+ SMARTS patterns with Fourier coefficients).
- Port UFF bond parameters needed for 1-2 distance bounds.
- Port the "basic knowledge" (K) force-field terms: sp2 planarity, sp linearity, bond/angle re-assertion.
- Multi-conformer generation with RMSD pruning (equivalent to `EmbedMultipleConfs`).
- Optional srETKDGv3 small-ring puckering extension, gated by an `EtkdgOptions` flag.
- Validate against 1290-molecule CSD benchmark from Riniker 2015 with the same success metric.
- SMARTS subgraph matcher built on `petgraph::algo::isomorphism::subgraph_isomorphisms_iter`.

### Non-goals (explicitly excluded)

- **Bit-exact match with RDKit.** Different RNG, floating-point ordering, and eigensolver tolerances make this infeasible and are not required for the quality target.
- **Custom VF2/Ullmann implementation.** Use `petgraph` subgraph isomorphism exclusively.
- **ETKDG v1 or v2.** Only v3 (and optional srETKDGv3). v1/v2 differ in torsion library content only; if needed later, swap in an older data file.
- **MMFF94-backed minimization after embedding.** RDKit's ETKDG intentionally stops at the ET(K)DG force field; downstream MMFF polishing is a separate user choice (already available in `molrs-ff`).
- **Keep the current `FragmentRules` embed algorithm.** It is removed. `builder.rs`, `fragment_data.rs`, and `rotor_search.rs` are deleted in M1.
- **Keep `EmbedAlgorithm` enum.** Only one algorithm remains; the enum becomes a single-variant relic and is removed.

## 4. Scope

- **Crates affected**:
  - `molrs-smiles` — internal reorganization: split into sibling `smiles/` and `smarts/` modules sharing a `chem/` preset module. Add SMARTS **matcher** (parser already exists).
  - `molrs-ff` — new `uff/` module hosting the UFF parameter table (used by embed for 1-2 bounds).
  - `molrs-embed` — near-complete rewrite. New `etkdg/` module tree with `bounds/`, `embed/`, `minimize/`, `torsions/`, `pipeline.rs`. Old modules (`builder.rs`, `fragment_data.rs`, `rotor_search.rs`, `geom.rs` minus still-used helpers) removed.
  - `molrs-core` — minor additions: ensure aromaticity perception, ring perception, and hybridization assignment match RDKit's defaults (see §8 risk). No new public API expected.
- **Traits created**:
  - `SubstructureMatcher` (in `molrs-smiles::smarts`) — returns matches of a compiled SMARTS pattern against a MolGraph.
  - `EtkdgStage` (in `molrs-embed::etkdg`) — internal trait for the three minimization stages; not public.
- **Traits extended**: none.
- **Data structures** (public):
  - `EtkdgOptions` — knobs (see §5.1).
  - `Conformer` — owns 3D coordinates + energy + per-stage diagnostics.
  - `EmbedFailureCause` — enum matching RDKit's `EmbedFailureCauses`.
  - `EmbedReport` — retained name; fields updated.
- **Feature flags**:
  - `etkdg-v3` (default on in molrs-embed) — enables the v3 torsion library.
  - `etkdg-smallring` — additionally enables srETKDGv3 small-ring torsions (opt-in; larger binary).
  - No separate M1/M2 flag; M1 = dev branch milestone, not a shipped feature gate.

## 5. Technical Design

### 5.1 Public API

The top-level entry point keeps its name for source compatibility:

```rust
/// Generate a single 3D conformer for `mol` using ETKDGv3.
///
/// Returns a new MolGraph with 3D coordinates attached, plus a EmbedReport
/// carrying per-stage diagnostics.
///
/// # Failure modes
/// Returns `Err(EmbedError::EmbedFailed { causes })` after exhausting
/// `opts.max_attempts` (default `10 * mol.num_atoms()`).
pub fn generate_3d(
    mol: &MolGraph,
    opts: &EtkdgOptions,
) -> Result<(MolGraph, EmbedReport), EmbedError>;

/// Generate up to `n` diverse conformers, pruned by heavy-atom RMSD.
/// Equivalent to RDKit `EmbedMultipleConfs`.
pub fn generate_conformers(
    mol: &MolGraph,
    opts: &EtkdgOptions,
    n: usize,
) -> Result<Vec<Conformer>, EmbedError>;

pub struct EtkdgOptions {
    pub random_seed: u64,               // RDKit: randomSeed
    pub max_attempts: Option<usize>,    // None => 10 * N_atoms
    pub prune_rms_thresh: Option<f64>,  // None = no pruning; RDKit default = -1 off, recommended 0.5–1.0
    pub use_small_ring_torsions: bool,  // srETKDGv3
    pub use_macrocycle_14_config: bool, // RDKit: useMacrocycle14config, default true for ≥9-membered rings
    pub embed_rms_thresh: f64,          // filter attempts too close to previous
    pub clear_confs: bool,              // drop existing conformers first
    pub num_threads: usize,             // parallel attempts; 0 => rayon default
    pub force_field: EtkdgForceField,   // Etkdg | EtkdgPlusUff (post-polish, opt-in)
    pub first_min_iters: usize,         // default 400
    pub collapse_iters: usize,          // default 200
    pub etk_min_iters: usize,           // default 300
}

pub enum EmbedFailureCause {
    InitialCoords,       // eigenvalue embedding degenerate
    FirstMinimization,   // DG-FF minimization did not converge
    CheckTetrahedral,    // tetrahedral center volume below threshold
    CheckChiralCenters,  // signed-volume wrong sign
    MinimizeFourthDim,   // 4D→3D collapse failed
    EtkMinimization,     // stage-3 minimization did not converge
    FinalChiralCenters,  // stereo regressed after stage 3
    LinearDoubleBond,    // sp2 geometry invalid
    BadDoubleBond,       // cis/trans inverted
    SmoothingFailed,     // triangle-inequality contradiction
}

pub struct EmbedReport {
    pub attempts: usize,
    pub stage_energies: [f64; 3],    // [first_min, collapse, etk_min]
    pub max_atom_energy: f64,        // compared against MAX_MINIMIZED_E_PER_ATOM
    pub failure_counts: HashMap<EmbedFailureCause, usize>,
    pub warnings: Vec<String>,
}
```

Old `EmbedOptions`, `EmbedAlgorithm`, `EmbedSpeed` are removed. A thin compatibility shim `impl From<OldEmbedOptions> for EtkdgOptions` may be provided for one release cycle, then deleted.

### 5.2 Crate structure changes

#### molrs-smiles (internal reorganization)

```
molrs-smiles/src/
├── lib.rs                    // public re-exports
├── chem/                     // SHARED presets — atom/bond chemistry primitives
│   ├── mod.rs
│   ├── elements.rs           // element symbols, valence, default masses
│   ├── bonds.rs              // bond-order enums, aromaticity flags
│   ├── hybridization.rs      // SP/SP2/SP3/SP3D2 rules
│   ├── aromaticity.rs        // Daylight/RDKit-compatible aromaticity perception
│   └── rings.rs              // (may proxy to molrs-core::rings)
├── smiles/                   // SMILES system (parser + writer)
│   ├── mod.rs
│   ├── ast.rs
│   ├── scanner.rs
│   ├── parser.rs
│   ├── to_atomistic.rs
│   └── validate.rs
└── smarts/                   // SMARTS system (parser + matcher) — NEW sibling
    ├── mod.rs
    ├── ast.rs                // SMARTS IR: atom/bond predicate trees
    ├── parser.rs             // SMARTS-specific parser (separate from SMILES)
    ├── predicate.rs          // evaluate SMARTS atom/bond predicate on a target atom/bond
    ├── compile.rs            // SMARTS AST → compiled pattern with precomputed indexes
    ├── matcher.rs            // substructure matcher on MolGraph (petgraph-backed)
    └── recursive.rs          // support for $(…) recursive SMARTS
```

**Rationale for sibling split**: SMARTS is a *query language* with predicate logic (`&`, `,`, `;`, `!`, `$(...)`) and ring-membership tests; SMILES is a *structure serialization format*. Their ASTs share atom and bond vocabulary but diverge sharply at the top level. Treating SMARTS as "SMILES with extensions" historically led to bugs in many toolkits (OpenBabel, early RDKit). The `chem/` module holds the shared primitives (atomic number, default valence, bond-order categories, Daylight aromaticity rules) that both systems agree on; anything language-specific lives in its own module.

**Crate name**: kept as `molrs-smiles` for workspace stability; a future rename to `molrs-chemlang` or similar is deferred and not part of this spec.

**Public re-exports** in `molrs-smiles::lib`:
```rust
pub use smiles::{parse_smiles, write_smiles, SmilesIr};
pub use smarts::{parse_smarts, SmartsPattern, SubstructureMatcher};
pub mod chem;  // exposed for downstream callers that need the primitives
```

#### molrs-embed (rewrite)

```
molrs-embed/src/
├── lib.rs                    // public API
├── error.rs                  // EmbedError, EmbedFailureCause
├── options.rs                // EtkdgOptions
├── report.rs                 // EmbedReport, Conformer
├── pipeline.rs               // top-level state machine: embed attempts + retry loop
└── etkdg/
    ├── mod.rs
    ├── bounds/
    │   ├── mod.rs
    │   ├── pair12.rs         // UFF 1-2 bounds
    │   ├── pair13.rs         // law-of-cosines 1-3 bounds
    │   ├── pair14.rs         // cis/trans extrema, cis-only rules, macrocycle
    │   ├── pair15plus.rs     // VDW_SCALE_15, H-bond relaxation
    │   └── smoothing.rs      // triangle smoothing (keep current converged impl)
    ├── embed/
    │   ├── mod.rs
    │   ├── metrize.rs        // partial random metrization with re-smoothing
    │   ├── metric_matrix.rs  // Crippen-Havel Gram matrix, 4D eigendecomposition
    │   └── random_coords.rs  // fallback when eigenvalue embedding degenerates
    ├── minimize/
    │   ├── mod.rs
    │   ├── dg_ff.rs          // stage 1 & 2 force field (distance + chiral + 4D)
    │   ├── etk_ff.rs         // stage 3 force field (torsion + K + bounds)
    │   ├── contribs.rs       // DistViolation, ChiralViolation, FourthDim, Torsion, Improper
    │   ├── bfgs.rs           // dense BFGS matching RDKit
    │   └── stages.rs         // first-min, collapse, etk-min wiring
    ├── torsions/
    │   ├── mod.rs
    │   ├── library.rs        // parse CSD torsion XML → static table
    │   ├── patterns.rs       // compiled SMARTS + Fourier coefficients
    │   └── match_bonds.rs    // per rotatable bond, select most-specific match
    ├── chirality.rs          // signed-volume evaluation, check functions
    └── stereo.rs             // double-bond E/Z verification
```

#### molrs-ff (new uff/ submodule)

```
molrs-ff/src/
└── uff/
    ├── mod.rs
    ├── params.rs             // generated static table
    └── bonds.rs              // r_ij = r_i + r_j + r_BO - r_EN helper
```

### 5.3 Data tables: sourcing, location, license

| Data | Source | Location in molrs | Format |
|---|---|---|---|
| UFF atom-type table (~126 types × 11 fields) | RDKit `Code/ForceField/UFF/Params.cpp` | `molrs-ff/data/uff.txt` (human-readable) → Rust static via `build.rs` | Compile-time `&'static [UffRow]` |
| ETKDGv3 torsion preferences (~254 SMARTS + coefficients) | RDKit `Code/GraphMol/ForceField/UFF/AtomTyper.cpp` and `Data/Params/torsion_preferences.xml` upstream | `molrs-embed/data/etkdg_v3.txt` → parsed by `build.rs` | Compile-time `&'static [TorsionRule]`; SMARTS strings compiled lazily at first use via `LazyLock` |
| srETKDGv3 small-ring torsions | RDKit `Code/GraphMol/DistGeomHelpers/smallring_torsion_preferences.xml` | `molrs-embed/data/etkdg_v3_smallring.txt` | Same as above; behind `etkdg-smallring` feature |

**Licensing**: RDKit is distributed under BSD-3-Clause, identical to molrs's own license. Derived data files carry the same license; add the following at `LICENSES/RDKIT.BSD` at the workspace root and reference it from each derived file's header:

```
This parameter table is derived from RDKit
(https://github.com/rdkit/rdkit), Copyright (c) 2006–present Greg Landrum
and the RDKit contributors, licensed under BSD-3-Clause. A copy of the
upstream license is in LICENSES/RDKIT.BSD.
```

A `NOTICE.md` at the workspace root lists derived files with their RDKit source path and commit hash.

**Rule for future parameter tables** (binding convention):

1. Parameter tables owned by a single consuming crate live as that crate's submodule: `<crate>/src/<domain>/params.rs`.
2. Parameter tables consumed by multiple crates live in the most-upstream consumer (following the dependency DAG). Do **not** create a dedicated `molrs-data` crate; versioning and discoverability are worse and nothing currently requires cross-crate sharing. Revisit only if three or more crates need the same table.
3. Raw data stays in a sibling `data/` directory as a human-readable text file and is converted to Rust via `build.rs` at compile time. This keeps diffs reviewable and avoids committing large `.rs` files whose content is mechanical.
4. Every derived data file's first line (after license header) records upstream source path + git commit hash + local validation checksum.

Under this rule: UFF lives in `molrs-ff/` (consumed by embed and future MD fixes). Torsion library lives in `molrs-embed/` (ETKDG-specific). Shared chemistry presets live in `molrs-smiles/src/chem/` (consumed by smiles, smarts, and eventually embed for pattern-based features).

### 5.4 Algorithm — embed pipeline state machine

Canonical source: RDKit `Code/GraphMol/DistGeomHelpers/Embedder.cpp::embedPoints()`.

```
generate_3d(mol, opts):
  rng = seed(opts.random_seed)
  mol' = add_hydrogens_if_needed(mol)
  bounds = build_bounds_matrix(mol', opts)          // §5.5
  bounds = smooth_bounds(bounds)                    // §5.6 — may fail → SmoothingFailed
  torsion_matches = match_etkdg_patterns(mol')      // §5.9
  k_terms = collect_knowledge_terms(mol')           // §5.10

  for attempt in 0..max_attempts:
    D = metrize(bounds, rng)                        // §5.7
    X4 = metric_matrix_embed_4d(D)                  // §5.7; may return Err(InitialCoords)
    X4 = first_minimization(X4, bounds, mol')       // §5.8 stage 1; 400 iters
    if !pass_tetrahedral(X4, mol'): continue
    if !pass_chiral_centers(X4, mol'): continue
    X3 = collapse_fourth_dim(X4, bounds, mol')      // §5.8 stage 2; 200 iters
    X3 = etk_minimize(X3, torsion_matches, k_terms) // §5.8 stage 3; 300 iters
    if !pass_tetrahedral(X3, mol'): continue
    if !pass_chiral_centers(X3, mol'): continue
    if !pass_double_bonds(X3, mol'): continue
    if max_per_atom_energy(X3) > 0.05: continue
    return (attach_coords(mol', X3), report)
  return Err(EmbedFailed { causes: collected failure counts })
```

### 5.5 Bounds matrix construction

Input: MolGraph with hybridization, bond orders, formal charges, ring perception, stereo annotations.
Output: `(lower: Array2<f64>, upper: Array2<f64>)` with `lower[i,j] ≤ upper[i,j]`, `lower[i,i] = upper[i,i] = 0`.

Constants (from RDKit `BoundsMatrixBuilder.cpp`):

| Name | Value |
|---|---|
| `DIST12_DELTA` | 0.01 Å |
| `DIST13_TOL` | 0.04 Å |
| `GEN_DIST_TOL` (1-4) | 0.06 Å |
| `DIST15_TOL` | 0.08 Å |
| `VDW_SCALE_15` | 0.7 |
| `H_BOND_LENGTH` | 1.8 Å |
| `MIN_MACROCYCLE_RING_SIZE` | 9 |

Rules:

- **1-2 (bonded)**: `r_ij = r_i + r_j + r_BO − r_EN` from UFF. `r_BO = −λ · (r_i + r_j) · ln(BO)` with λ = 0.1332. `r_EN = r_i · r_j · (√χ_i − √χ_j)² / (χ_i · r_i + χ_j · r_j)`. Tolerance ± `DIST12_DELTA`. Missing UFF params fall back to `(vdW_i + vdW_j) × [0.5, 1.5]`.
- **1-3 (angles)**: reference angle from hybridization table:
  - sp: 180°
  - sp2: 120°
  - sp3: 109.47°
  - 3-membered ring: 60°
  - 4-membered ring: 90°
  - Octahedral/trigonal-bipyramidal: per RDKit table
  Lower = `law_of_cosines(r_12_lo, r_23_lo, θ − tol)`, upper = `law_of_cosines(r_12_hi, r_23_hi, θ + tol)` with `tol = DIST13_TOL` (doubled for large sp2 atoms as in RDKit).
- **1-4 (torsions)**: from atoms `i-j-k-l`, compute `d_14(θ)` for dihedral θ ∈ [0, π]. The cis minimum and trans maximum give default bounds. Then specialize:
  - If `j-k` is a double bond: E/Z annotation forces trans-only (180 ± tol) or cis-only (0 ± tol).
  - If atoms j and k are both sp2 and in the same ring of size ≤ 8 (and not flagged as macrocycle): cis-only.
  - If bond `j-k` is aromatic or in a conjugated system: cis-only with tightened bounds.
  - If macrocycle mode active and ring size ≥ 9: keep default cis/trans range.
  Tolerance: `GEN_DIST_TOL`.
- **1-5 and beyond**: lower = `VDW_SCALE_15 × (vdW_i + vdW_j)`, upper = `1000 Å` (later tightened by smoothing). H-bond donor-acceptor pairs: lower relaxed to `H_BOND_LENGTH = 1.8`.

### 5.6 Triangle smoothing

Keep `molrs-embed`'s existing converged `smooth_bounds_converged` implementation (Floyd-Warshall iterated to fixed point). This is strictly better than RDKit's single sweep — tighter bounds, same correctness, no regression risk. On contradiction (any `lower > upper`) fail with `SmoothingFailed`.

### 5.7 Metrization and 4D metric-matrix embedding

**Partial random metrization** (`embed/metrize.rs`):

```
D = zero matrix
for (i, j) in random permutation of atom pairs:
    d_ij = uniform(lower[i,j], upper[i,j])
    D[i,j] = D[j,i] = d_ij
    re-smooth bounds with d_ij fixed (local update)
return D
```

This matches RDKit's `pickRandomDistMat`. Re-smoothing keeps geometric consistency and avoids independence bias.

**Metric-matrix embedding** (`embed/metric_matrix.rs`):

```
d0i² = (1/N) Σ_j D[i,j]² − (1/N²) Σ_{j<k} D[j,k]²
T[i,j] = ½ (d0i² + d0j² − D[i,j]²)          # Gram matrix
(λ, v) = top_k_eigenpairs(T, k=4)
if any λ_a ≤ 0 for a in 1..4:
    return Err(InitialCoords)               # caller falls back or retries
X[i, a] = √λ_a · v_a[i]                      # N × 4 coordinates
```

Eigendecomposition via `ndarray-linalg` `eigh` (LAPACK `dsyevd`). No Jacobi; LAPACK is faster and equally accurate for this size. Top-4 selection by sorting eigenvalues descending.

### 5.8 Three-stage minimization

**Optimizer**: dense BFGS (not L-BFGS), matching RDKit. Maintain full Hessian approximation; N is always small enough (≤ ~1000 atoms) that O(N²) memory is fine. Line search: backtracking Armijo with `c₁ = 10⁻⁴`.

Convergence: gradient norm `< ERROR_TOL = 1e-5` or max iterations reached.

**Stage 1 — first minimization** (4D, DG force field):

Terms (weights shown):
- `DistViolationContribs` (w = 1.0): `E = ((d² − U²)/U²)²` if `d > U`; `E = ((L² − d²)/L²)²` if `d < L` and `d < basinThresh = 5.0`; else 0.
- `ChiralViolationContribs` (w = 1.0): `E = (V_signed − V_target)²` for each tetrahedral stereocenter, `V_signed = det(r_n1 − r_n0, r_n2 − r_n0, r_n3 − r_n0)`, `V_target ∈ {+1, −1}` (sign only, magnitude learned by minimization).
- `FourthDimContribs` (w = 0.1): `E = x_4²` per atom.

Max iterations: `opts.first_min_iters = 400`.

**Stage 2 — 4D collapse**:

Same force field, new weights: chiral w = 0.2, fourth-dim w = 1.0. Pulls atoms onto the 3D hyperplane. Max iterations: `opts.collapse_iters = 200`. After convergence, drop the 4th coordinate: `X3 = X4[:, 0..3]`.

**Stage 3 — ET(K)DG minimization** (3D, torsion + knowledge):

Terms:
- Distance restraints re-applied: bond (1-2) with `U_lo = r_12_lo − padding`, `U_hi = r_12_hi + padding`; angle (1-3) similar with stiffer weight. Padding `0.02 Å`.
- `TorsionContribs`: `V(θ) = Σ_{n=1..6} V_n (1 + s_n cos(nθ))` per matched rotatable bond (§5.9).
- `ImproperContribs`: sp2 planarity, sp linearity, amide N planarity; quadratic penalty on out-of-plane angle with `planarityTolerance = 0.7`.

Max iterations: `opts.etk_min_iters = 300`. Post-check: `max_per_atom_energy ≤ MAX_MINIMIZED_E_PER_ATOM = 0.05`.

### 5.9 Torsion preferences matching

Pipeline:

1. At crate load time, parse `data/etkdg_v3.txt` into `Vec<TorsionRule { smarts: &str, fourier: [FourierTerm; 6] }>`. Patterns are kept in library order (RDKit convention: most-specific patterns appear first).
2. For a given molecule, iterate rotatable bonds (bond order 1 or aromatic, both atoms have degree ≥ 2, not in rings smaller than `MIN_MACROCYCLE_RING_SIZE` unless srETKDG enabled).
3. For each rotatable bond `j-k`, iterate torsion quartets `i-j-k-l` where `i ∈ neighbors(j) \ {k}` and `l ∈ neighbors(k) \ {j}`. Walk the torsion rules in order; the first matching rule wins (most-specific-first). Cache `(bond_id, rule_id, atom_quartet)`.
4. Unmatched rotatable bonds fall back to generic templates (sp3-sp3, sp3-sp2, sp2-sp2) built into the library at the end of the file.

SMARTS matching uses `molrs-smiles::smarts::SubstructureMatcher`, which wraps `petgraph::algo::isomorphism::subgraph_isomorphisms_iter` with `node_match` and `edge_match` closures evaluating the SMARTS predicate AST.

### 5.10 Knowledge (K) terms

Collected from the molecule once:

- Every sp2 atom with three heavy neighbors → improper planarity term on the four atoms.
- Every sp atom with two neighbors → linearity term (bond angle 180 ± ε).
- Every aromatic ring → planarity terms on the ring atoms (handled via per-atom sp2 planarity, no separate ring term).
- Every amide nitrogen (N-C(=O)-* sp2) → planarity via generic sp2 rule (no amide-specific term needed in v3; it was in v1 and subsumed in v3).

### 5.11 Chirality and E/Z verification

After stage 1 and stage 3, re-check:

- **Tetrahedral**: for each stereocenter atom with 4 neighbors, compute signed volume. Reject if `|V| < MIN_TETRAHEDRAL_CHIRAL_VOL = 0.50 Å³` (atom is near-planar) or if `sign(V) ≠ V_target` (inverted). Tolerance on magnitude: `TETRAHEDRAL_CENTERINVOLUME_TOL = 0.30`.
- **Double bonds**: for each E/Z-annotated double bond, compute the dihedral and confirm cis vs trans within 20°. Additionally verify linearity: the two atoms bonded to each double-bond terminus should be nearly coplanar with the double bond.

Failure routes the attempt to the retry loop with cause recorded.

### 5.12 Multi-conformer generation

`generate_conformers(mol, opts, n)`:

1. Precompute `bounds`, `torsion_matches`, `k_terms` ONCE (they depend only on topology).
2. Run `n` attempts (or `opts.num_threads`-way parallel via rayon; each thread gets its own RNG seeded with `opts.random_seed + attempt_index`).
3. For each successful conformer, compute heavy-atom RMSD against every accepted conformer (with optimal alignment). Reject if any RMSD < `opts.prune_rms_thresh`.
4. Stop when `n` conformers accepted or `max_attempts` exhausted.

## 6. Constraints and Invariants

- **F = f64 everywhere.** No raw `f32` or `f64` in algorithm code — use the `F` alias per workspace convention.
- **Coordinate format**: `Array2<f64>` shape `[N, 4]` during embedding (stage 1 + 2), shape `[N, 3]` during stage 3 and final output. Not the flat `[x0,y0,z0,...]` form used by `molrs-ff` potentials — ETKDG does not interoperate with `molrs-ff::Potential`.
- **Gradient sign convention** (from CLAUDE.md): force-field terms accumulate the TRUE gradient `∂E/∂x` INTO `g` with `+=`; BFGS negates for descent direction.
- **No panics** on malformed input. Every boundary returns `Result`. Assertions (`debug_assert!`) OK in inner loops.
- **SMARTS matcher is Send + Sync.** Compiled patterns must be safely shared across rayon threads for multi-conformer generation.
- **RNG discipline**: one `SmallRng` (`rand_xoshiro::Xoshiro256PlusPlus`) per attempt, seeded deterministically from `opts.random_seed + attempt`. Never share RNG across threads.
- **Aromaticity and ring perception must be pre-computed on the input MolGraph and agree with RDKit's defaults** (Daylight-style). If disagreement found during benchmarking, log and align (see §8).

## 7. Test Criteria

### 7.1 Unit tests

**Bounds matrix**:
- 1-2 bounds for H-H, C-C single, C-C aromatic, C=O, C≡N match RDKit values within 1e-5.
- 1-3 bounds for water (H-O-H, sp3 O), methane (H-C-H, sp3 C), ethylene (H-C=C) match within 1e-5.
- 1-4 bounds for butane (anti / gauche), ethylene derivatives (E/Z-enforced), benzene (cis-only) match within 1e-4.
- Smoothing fails cleanly on contradicting inputs (return `SmoothingFailed`, no panic).

**Metric matrix embedding**:
- Random distance matrix from a known 4-atom tetrahedron (edge length 1) reconstructs positions within 1e-6 after centering and aligning.
- Degenerate case (all distances zero) returns `InitialCoords` error, no panic.

**DG force-field terms**:
- `DistViolationContribs` numerical gradient matches analytical within 1e-6 at `h = 1e-7` on random configurations.
- `ChiralViolationContribs` gradient vs numerical on a tetrahedral test case: match within 1e-6.
- Signed volume for a left-handed tetrahedron is negative; for right-handed, positive.

**SMARTS matcher**:
- `[C;X4]` matches only sp3 carbons in methane, ethane, benzene.
- `[c;r6]` matches only aromatic ring-6 atoms.
- `$(C=O)` recursive SMARTS matches carbonyl carbons in acetone.
- Most-specific match: given a molecule and two patterns (generic sp3-sp3, specific amide N-C), the amide pattern wins for matching atoms.

**Torsion library**:
- Loading `etkdg_v3.txt` produces non-empty rule set; every rule compiles to a valid SMARTS pattern.
- `butane` gets the sp3-sp3 generic torsion rule; `formamide` gets an amide-specific rule.

### 7.2 Integration tests

**Small molecules** (expected pass rate 100%, RMSD vs RDKit reference < 0.3 Å):
- water, methane, ethane, butane, ethylene, benzene, cyclohexane, glucose (α-D), serine.

**Stereocenters**:
- (R)- and (S)-alanine preserve configuration: run 100 attempts each, zero flips.
- (E)- and (Z)-2-butene double-bond geometry preserved.
- 2,3-dihydroxybutanedioic acid (tartaric acid, RR/SS/meso): three stereoisomers give three distinct conformers.

**Multi-conformer**:
- `generate_conformers(n=100)` on n-hexane with `prune_rms_thresh = 0.5` returns > 10 distinct conformers.
- Parallel and serial produce the same result set (order-independent).

**Failure modes**:
- Disconnected molecule returns error, not panic.
- Molecule with no valid 3D embedding (impossible ring) returns `EmbedFailed` with `SmoothingFailed` dominant.

### 7.3 Numerical validation — CSD benchmark

**Benchmark set**: the 1290-molecule subset used by Riniker & Landrum 2015 (CSD subset, heavy atoms only, single conformer per molecule). Source: published as supplementary with the 2015 paper. To be downloaded into `molrs-embed/target/benchmark-data/csd-1290/` via a helper script analogous to `fetch-test-data.sh`.

**Metric**: for each molecule, generate one conformer with the default seed; compute heavy-atom RMSD against the crystal structure after optimal rigid alignment. Pass if RMSD < 1.0 Å.

**Target**: ≥ 84% pass rate. Sub-target: no regressions from Riniker's reported numbers on the per-class breakdown (macrocycles ≥ 50%, small rings ≥ 85%, others ≥ 85%).

Benchmark runs as `cargo test --features slow-tests -p molrs-embed csd_benchmark` (requires the downloaded dataset). CI runs on demand, not per-PR.

### 7.4 Regression tests

A frozen set of 20 molecules, their expected conformers stored as XYZ files under `tests-data/etkdg_regression/`. Each CI run regenerates with fixed seed and compares coordinates within 1e-3 Å (allowing minor floating-point drift from dependency updates). Larger drift requires a recorded update to the frozen files with justification.

## 8. Performance Requirements

- Single conformer for a drug-like molecule (30–60 heavy atoms): ≤ 200 ms on a single core of an M2 Pro. RDKit baseline: ~100 ms; 2× is acceptable for a first port.
- 100 conformers for the same molecule with `num_threads=8`: ≤ 1.5 s wall time.
- Memory: O(N²) for bounds matrices, acceptable up to N ≈ 2000.

Benchmarks via criterion in `molrs-embed/benches/etkdg.rs`:

- `embed_water`, `embed_butane`, `embed_benzene`, `embed_glucose`, `embed_drugs_30`, `embed_drugs_60`.
- `multi_conf_serial_100`, `multi_conf_parallel_100`.

## 9. Migration and Compatibility

- **Breaking**: `EmbedOptions` → `EtkdgOptions`, `EmbedAlgorithm` removed, `EmbedSpeed` removed. Single-release deprecation shim provided, deleted in version +2.
- **Removed modules**: `builder.rs`, `fragment_data.rs`, `rotor_search.rs`, `stereo_guard.rs` (replaced by `etkdg::chirality` and `etkdg::stereo`). Fragment data files under `molrs-embed/data/fragments/` deleted.
- **Python bindings** (`molrs-python/`) updated to expose `generate_3d` and new `generate_conformers`; old options struct gets a compatibility constructor.
- **Public API re-exports** in `molrs-embed::lib` and `molrs::prelude` updated; downstream users must switch to `EtkdgOptions`.

## 10. Delivery Milestones

### M1 — DG + three-stage minimization without torsion library

**Definition of Done**:
- `molrs-smiles` reorganized into `chem/` + `smiles/` + `smarts/` siblings. SMARTS parser + matcher working, tested.
- `molrs-ff::uff` parameter table loaded, bond distance calculation unit-tested.
- `molrs-embed::etkdg::bounds` complete: 1-2, 1-3, 1-4, 1-5+ rules implemented and unit-tested against RDKit values.
- `molrs-embed::etkdg::embed` complete: 4D metric matrix embedding, metrization.
- `molrs-embed::etkdg::minimize` stages 1, 2, 3 implemented — BUT stage 3 runs with K-terms + distance restraints only; torsion library is stubbed (all rotatable bonds get the generic sp3-sp3 fallback).
- `generate_3d` and `generate_conformers` public API complete.
- Unit tests (§7.1) pass.
- Integration tests (§7.2) pass for small molecules and stereocenters.
- `builder.rs`, `fragment_data.rs`, `rotor_search.rs`, `stereo_guard.rs` deleted.

**Expected quality**: ≥ 70% CSD benchmark pass rate (lower than M2's 84% due to missing torsion preferences). This is still a substantial improvement over current output.

### M2 — Full ETKDGv3 with torsion library

**Definition of Done**:
- `molrs-embed/data/etkdg_v3.txt` ported from RDKit upstream with recorded commit hash.
- `etkdg::torsions` module complete: library loading, per-bond pattern matching, Fourier term construction.
- `TorsionContribs` wired into stage 3 force field.
- Srinter smallring torsion data ported behind `etkdg-smallring` feature.
- CSD benchmark ≥ 84% pass rate (see §7.3).
- Regression tests (§7.4) passing.
- Performance benchmarks (§8) meet targets.
- NOTICE.md updated, RDKIT.BSD license file present, data file provenance headers complete.

## 11. Risks and Open Questions

### 11.1 Aromaticity perception divergence (high risk)

`molrs-core`'s aromaticity perception must agree with RDKit's on the benchmark set. RDKit uses Daylight-style perception with modifications for heteroatoms and fused systems. If divergence is found, options are (a) add a `RdkitCompatibleAromaticity` variant to `molrs-core`, or (b) align the default. Preference: (b), to keep the codebase converged. **Action item for M1**: write a test that compares aromatic atom sets on 100 molecules; fix any divergence before starting bounds tests.

### 11.2 CIP stereochemistry coverage (medium risk)

RDKit implements full CIP rules including sphere-by-sphere tie-breaking and auxiliary descriptors. `molrs-core::stereochemistry` has basic R/S assignment but coverage beyond simple stereocenters is unverified. For ETKDG, what matters is that the target chiral volume sign is correct. If CIP coverage is insufficient, stereo-guard test failures will surface. **Action item**: audit `molrs-core::stereochemistry` against a known CIP test set (e.g., the Fontana et al. 2020 benchmark) before M1 integration tests.

### 11.3 Torsion library license confirmation (low risk)

The RDKit torsion preferences XML is distributed under BSD-3 as part of the RDKit source tree. Double-confirm by reading the file header before M2 checkpoint; if any entry derives from a paper with incompatible terms (unlikely — Guba 2013 and follow-ups are openly published), exclude it.

### 11.4 SMARTS recursive matching performance (low risk)

`$(...)` recursive SMARTS triggers a nested subgraph search. For large molecules with many rotatable bonds this could blow up. Mitigation: memoize recursive-pattern matches per target atom; typical recursive patterns in the ETKDG library match <5 atom types per molecule, so cost is bounded.

### 11.5 Eigendecomposition vs random-coords fallback frequency

RDKit falls back to random coordinates when the top-4 eigenvalues include any non-positive value. For large, highly-constrained molecules this happens often. **Action item during M1**: instrument the metric-matrix embedder to count fallbacks on the benchmark set; if >20%, investigate whether smoothing is under-converging or whether metrization pairs are chosen in a bad order.

### 11.6 Deferred: MD coupling

ETKDG conformers are sometimes polished with MMFF94 via RDKit. We are not bundling this in the ETKDG pipeline; users who want it call `molrs-ff::minimize(frame, MMFF94)` downstream. This is documented but not tested as part of this spec.

---

## 12. Implementation Order (advisory — handed to `/molrs-impl`)

1. SMARTS matcher foundation: reorganize `molrs-smiles`, add `chem/` module, build predicate evaluator, wire to petgraph, test.
2. UFF parameter table port and `uff/bonds` helper, test against RDKit reference values.
3. Bounds matrix module with all four distance classes, test against RDKit values on ≥ 20 small molecules.
4. Metric-matrix 4D embedder + metrization, test tetrahedron reconstruction.
5. DG-FF terms (distance, chiral, fourth-dim) with analytical-vs-numerical gradient tests.
6. Dense BFGS optimizer, test on Rosenbrock and on a small DG-FF problem.
7. Stage 1 + Stage 2 wiring, chirality/E-Z checks, first integration tests on small molecules.
8. Stage 3 force field (K-terms only, no torsion library yet), complete pipeline, retry loop → **M1 ships**.
9. Torsion library parser + compile-time table, per-bond matcher.
10. `TorsionContribs` wired into Stage 3, library smoke tests.
11. srETKDGv3 small-ring extension.
12. CSD benchmark harness, performance tuning → **M2 ships**.
