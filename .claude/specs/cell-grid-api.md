---
title: cell-grid-api — public, PBC-aware lattice cell grid shared by LinkCell and external packers
status: draft
created: 2026-07-25
---

# cell-grid-api — public, PBC-aware lattice cell grid

## Summary

Extract the cell-assignment and stencil logic currently private to `LinkCell`
into a public `CellGrid` type layered on `SimBox`, with **explicit per-axis PBC
semantics** (wrap on periodic axes, clamp on non-periodic axes) and a stencil
rule that stays correct when a lattice direction holds fewer than three cells.
`LinkCell` is refactored to call it, so there is exactly one implementation of
"which cell does this point fall in, and which cells are its neighbours" in the
workspace.

This is the molrs-side enabler for molpack deleting its private orthorhombic
port (`molpack/src/cell.rs` — a direct translation of Packmol's
`cell_indexing.f90` / `pbc.f90`) and packing into arbitrary triclinic cells.

This is a **zero-performance-loss refactor**, guarded by per-function and
per-caller microbenches under `.claude/notes/performance.md`'s gates. It is not
a compatibility-preserving one: the behaviour being replaced is wrong on
non-periodic axes, so the target is correctness against `BruteForce`, not
agreement with the code it replaces.

## Domain basis

- A link-cell search is valid only if each cell's width along every lattice
  direction is ≥ the cutoff measured as a **plane distance**, not as a
  lattice-vector length. For a tilted cell `|a_k|` overestimates the width, so
  cells become thinner than the cutoff and pairs are silently missed.
  `SimBox::nearest_plane_distance()` gives the correct measure;
  `counting_sort_impl` already uses it and `CellGrid` inherits that rule.
- Once cells are indexed in **fractional** space, the 27-cell (or 13-forward)
  stencil in index space is identical for orthorhombic and triclinic cells; the
  geometry re-enters only through the minimum-image displacement, which
  `SimBox::shortest_vector_impl` already provides. Triclinic support therefore
  costs no new pair-loop code — only correct cell sizing and correct stencils.
- Packmol's `setcell` **clamps** out-of-box coordinates on non-periodic axes so
  that atoms driven outside the confinement during optimisation still land in a
  valid cell. A packer needs this: unlike an MD trajectory, intermediate
  iterates are routinely outside the box. Today `get_cell3` applies `% celldim`
  on all three axes regardless of `SimBox::pbc()` — positive overflow wraps
  (wrong on a non-periodic axis) while negative fractions saturate to cell 0
  through the float→int `as` cast (accidentally right). The semantics must be
  explicit and symmetric.

## Design

### `CellGrid`

New module `molrs/src/core/spatial/neighbors/grid.rs`, re-exported from
`spatial::neighbors`:

```rust
pub struct CellGrid {
    celldim: [u32; 3],
    pbc: [bool; 3],
}

impl CellGrid {
    /// Cells sized so every lattice direction has width >= cutoff measured as
    /// a plane distance: `celldim[k] = max(1, floor(npd[k] / cutoff))`.
    pub fn for_cutoff(bx: &SimBox, cutoff: F) -> Self;
    pub fn with_dims(celldim: [u32; 3], pbc: [bool; 3]) -> Self;

    pub fn celldim(&self) -> [u32; 3];
    pub fn n_cells(&self) -> usize;

    /// Cell triple for a point. Wraps on periodic axes; clamps into
    /// `[0, celldim[k] - 1]` on non-periodic axes.
    pub fn cell3(&self, bx: &SimBox, r: [F; 3]) -> [u32; 3];
    pub fn flat(&self, c: [u32; 3]) -> usize;
    pub fn unflat(&self, icell: usize) -> [u32; 3];

    /// Up to 26 distinct neighbour cells, self excluded.
    pub fn stencil_all(&self, icell: usize, out: &mut [usize; 27]) -> usize;

    /// Half stencil: forward neighbours only, such that every unordered cell
    /// pair is produced exactly once across a full sweep of all cells.
    pub fn stencil_forward(&self, icell: usize, out: &mut [usize; 13]) -> usize;
}
```

### Semantics pinned by this spec

**Wrap vs clamp.** Per axis, from `pbc`. Periodic axis: fractional coordinate is
wrapped into `[0, 1)` before binning. Non-periodic axis: the cell index is
clamped into `[0, celldim[k] - 1]`, so a point far outside the box lands in the
nearest edge cell and is still seen by the pair loop.

**Small dimensions.** The offset set per axis is:

| `celldim[k]` | offsets | wrap? |
|---|---|---|
| 1 | `{0}` | n/a |
| 2 | `{0, +1}` | **no** — `+1` is dropped at `c = 1` |
| ≥ 3 | `{-1, 0, +1}` | yes on periodic axes |

The `celldim == 2` rule is the load-bearing one. With wrap enabled, cell 0's
`+1` neighbour is cell 1 *and* cell 1's `+1` neighbour wraps back to cell 0, so
the unordered cell pair `{0, 1}` is emitted twice and every cross-cell pair is
double-counted. Because minimum image already selects the nearest periodic
image, visiting the pair once is both necessary and sufficient. The same
argument makes `celldim == 1` correct with a self-cell-only stencil.

**Distinctness.** `stencil_all` and `stencil_forward` return distinct cell
indices; callers may assume no duplicates.

### `SimBox::make_fractional_raw_arr3`

`make_fractional_fast_arr3` folds every axis into `[0, 1)` unconditionally, so
by the time a caller sees the fractional coordinate the information needed to
clamp a non-periodic axis is already gone. It gains a raw sibling that skips the
fold; the wrapping version is re-expressed as `raw` followed by `f - f.floor()`
per component, so there is one implementation and the two cannot drift.

### `LinkCell` refactor

`counting_sort_impl` constructs `CellGrid::for_cutoff(bx, self.cutoff)` and
calls `grid.cell3` / `grid.flat`; `visit_pairs`, `compute_pairs_serial`,
`compute_pairs_parallel` and `visit_neighbors_of_pt` call the grid's stencils.
The private `get_cell3`, `stencil_range`, `wrap`, `collect_stencil_into`,
`stencil_fwd_into` and `stencil_all_into` are deleted in the same commit.

`CellGrid::flat` keeps the current flattening order
(`cz * dy * dx + cy * dx + cx`). Not for compatibility — a different order would
be equally correct — but because keeping it makes the refactor's diff readable
and leaves `sorted_idx` / `cell_start` untouched for inputs the old code already
handled correctly, which shortens the review.

### Performance protocol (binding)

No sentinel copies of the replaced bodies are kept. The guard is a set of
**permanent** microbenches in the `neighbors/cellgrid` and `neighbors/traversal`
criterion groups, which `bench.yml` tracks over time — a later change to the
cell arithmetic shows up as a step on the dashboard instead of hiding inside a
build timing. Per `.claude/notes/performance.md` an end-to-end bench alone is
not admissible evidence: its ±3% noise floor is the same magnitude as the gate
and it carries no per-function signal.

Groups and their role:

| bench | role |
|---|---|
| `neighbors/cellgrid/cell_of/{ortho,triclinic}` | function-level: cell assignment |
| `neighbors/cellgrid/stencil_fwd/{ortho,triclinic}` | function-level: stencil walk |
| `neighbors/traversal/build/{ortho,triclinic}` | caller-level: build path |
| `neighbors/traversal/visit_pairs/{ortho,triclinic}` | caller-level: pair traversal |
| `neighbors/{build,update,build_soa,query_columns}` | pre-existing end-to-end alarm |

Gates: function-level ≤ **+1%** and caller-level ≤ **+2%** against the recorded
baseline series on the CI bench runner; the pre-existing groups ≤ **+10%** (catastrophic alarm only).

Reference figures from this branch (criterion median, N = 1000 points,
cutoff 4, box edge 30, `--sample-size 100 --measurement-time 4`), two runs:

| bench | ortho | triclinic |
|---|---|---|
| `cellgrid/cell_of` | 14.1 / 15.2 µs | 28.6 / 29.7 µs |
| `cellgrid/stencil_fwd` | 56.1 / 54.1 µs | 46.3 / 47.5 µs |
| `traversal/build` | 7.8 / 5.7 ms | 3.0 / 7.7 ms |
| `traversal/visit_pairs` | 513 / 693 µs | 2.30 / 4.27 ms |

**These are not gate baselines.** They were taken on a shared login node and the
run-to-run spread is ±8% at function level and worse than 2× on the
rayon-parallel `build`, so a ±1% / ±2% gate is not measurable here. The gates are
evaluated on the CI bench runner, where `bench.yml` records the series; the
numbers above serve only to fix the order of magnitude and to document what was
measured when the design decision below was taken. Anyone re-checking these
locally needs a quiet, pinned core.

Two things the figures do show, both robust to the noise: the triclinic path is
~2× the orthorhombic one at cell assignment (an `inv · dr` matrix product versus
three multiplies) and ~4× at traversal (larger cells at equal edge length, plus a
more expensive minimum image). Neither is a regression — the triclinic path did
not exist as a supported configuration before.

The one measurement that shaped the design: with the wrap/clamp dispatch behind
a call boundary, `cell_of` costs ~10% more than an unconditional wrap on an
orthorhombic box, but inlined it is *faster* than the unconditional wrap on both
box kinds — the dispatch is loop-invariant and gets hoisted, while the triclinic
path saves the three wraps it used to perform and immediately discard. Hence
`cell_of` and `cell3` are `#[inline(always)]`, and that is a load-bearing
attribute, not decoration.

## Files

- `molrs/src/core/spatial/neighbors/grid.rs` — new, `CellGrid`
- `molrs/src/core/spatial/neighbors/mod.rs` — `pub use grid::CellGrid;`
- `molrs/src/core/spatial/neighbors/linkcell.rs` — route through `CellGrid`;
  delete the private cell/stencil helpers in the same commit
- `molrs/src/core/spatial/region/simbox.rs` — `make_fractional_raw_arr3`
- `molrs/benches/core/neighbors/linkcell.rs` — `neighbors/cellgrid` and
  `neighbors/traversal` groups, ortho + triclinic
- `molrs/src/core/spatial/neighbors/grid.rs` inline `#[cfg(test)]` +
  `molrs/tests/` — brute-force equivalence matrix

## Tasks

1. `CellGrid` core: `for_cutoff` / `with_dims` / `cell3` / `flat` / `unflat`,
   wrap-vs-clamp semantics, unit tests.
2. `stencil_all` / `stencil_forward` with the small-dimension table above, plus
   distinctness and exactly-once tests.
3. Permanent function + caller microbenches, ortho and triclinic; record the
   baseline.
4. Refactor `LinkCell` onto `CellGrid`; delete the replaced helpers.
5. Brute-force equivalence matrix (below) in **pair and query mode**, including
   points seeded outside the box on non-periodic axes.

## Testing

`BruteForce` is the oracle. The matrix is
{orthorhombic, hexagonal (a=b, γ=120°), strongly tilted triclinic}
× pbc ∈ {(t,t,t), (t,t,f), (f,f,f)}
× cutoffs chosen so that `celldim` components take the values 1, 2, 3 and 5.

For each combination the pair set from `LinkCell::visit_pairs` must equal
`BruteForce`'s pair set exactly — same unordered pairs, same multiplicity (no
duplicates, no omissions) — and every `dist_sq` must agree to 1e-12.

Points are deliberately seeded outside the box on non-periodic axes to exercise
clamping, and at the exact cell boundaries (fractional 0.0 and 1.0) to pin the
half-open convention.

## Out of scope

- molpack's adoption of this API — separate spec `triclinic-cell-downshift` in
  the molpack repo.
- Replacing `LinkCell`'s sorted-particle layout, adding a Verlet skin, or any
  GPU path.
- Preserving the replaced behaviour. There is no compatibility shim, no
  sentinel copy of the old bodies, and no bit-identity requirement against the
  previous cell assignment: the old code wrapped non-periodic axes, which was
  wrong, so agreeing with it is not a goal. Correctness is defined against
  `BruteForce`, not against the code being replaced.
