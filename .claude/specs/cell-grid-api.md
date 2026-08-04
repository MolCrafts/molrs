---
title: cell-grid-api — public, PBC-aware lattice cell grid shared by LinkCell and external packers
status: done
created: 2026-07-25
closed: 2026-08-04
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
    /// Buffer is 27 wide, not 13: on a small grid the pre-dedup candidate set
    /// can exceed the eventual count.
    pub fn stencil_forward(&self, icell: usize, out: &mut [usize; 27]) -> usize;
}
```

### Semantics pinned by this spec

**Wrap vs clamp.** Per axis, from `pbc`. Periodic axis: fractional coordinate is
wrapped into `[0, 1)` before binning. Non-periodic axis: the cell index is
clamped into `[0, celldim[k] - 1]`, so a point far outside the box lands in the
nearest edge cell and is still seen by the pair loop.

**Small dimensions.** The offset block is `{-1, 0, +1}` on **every** axis
regardless of how few cells it holds. Out-of-range offsets are dropped on
non-periodic axes, wrapped on periodic ones, and the result is sorted and
deduplicated, so an axis with one or two cells needs no special case.

This corrects the rule this spec was drafted with. The concern was that at
`celldim == 2` with wrap, cell 0's `+1` neighbour is cell 1 while cell 1's `+1`
wraps back to cell 0, double-counting the pair. That does not happen: the
`nc > icell` filter on the forward stencil already admits the pair from one side
only, and the dedup collapses aliased offsets. The real defect is the opposite
one, and it is in the **full** stencil: the replaced code used a `{0, +1}`
offset set when an axis held two cells, so on a *non-periodic* axis the upper
cell dropped its only neighbour and reported an empty stencil. That is invisible
on the pair path — the forward filter hides it — and shows up only on the query
path, which is why the equivalence matrix has to run in both modes.

**Distinctness.** `stencil_all` and `stencil_forward` return distinct cell
indices; callers may assume no duplicates.

### `SimBox::make_fractional_raw_arr3`

`make_fractional_fast_arr3` folds every axis into `[0, 1)` unconditionally, so
by the time a caller sees the fractional coordinate the information needed to
clamp a non-periodic axis is already gone. It gains a raw sibling that skips the
fold; the wrapping version is re-expressed as `raw` followed by `f - f.floor()`
per component, so there is one implementation and the two cannot drift.

### Allocation-free triclinic minimum image

`SimBox::mic_kernel`'s triclinic branch built an `Array1` per call, twice
(`inv.dot(dr)` and `h.dot(frac)`). That is the innermost pair kernel of every
consumer — a packer evaluates it millions of times per objective evaluation —
so two heap allocations per pair dominated the arithmetic outright. Rewritten as
stack 3×3 products with the same summation order (ndarray sums k ascending), so
the result is bit-identical and pinned by a test against the allocating form.

Measured on `neighbors/traversal/visit_pairs/triclinic`: 2.07 ms → 786 µs, a
2.6× speedup. It brings triclinic traversal to ~1.7× the orthorhombic path
rather than ~4×, which is what makes a triclinic pair loop viable for molpack at
all.

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
- `molrs/src/core/spatial/neighbors/linkcell.rs` — `#[cfg(test)] mod
  equivalence`, the matrix (the crate keeps coverage in `#[cfg(test)]` modules
  next to the code; `autotests = false`, no `tests/` tree)

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

The oracle is a direct O(N²) double loop over `SimBox::shortest_vector_impl`,
sharing no cell-assignment or stencil code with the algorithm under test, so a
defect in either cannot cancel out. `BruteForce` is checked against the same
oracle in the same loop, which keeps the oracle honest. The matrix is
{orthorhombic, hexagonal (a=b, γ=120°), strongly tilted triclinic}
× pbc ∈ {(t,t,t), (t,t,f), (f,f,f)}
× cutoffs chosen so that `celldim` components take the values 1, 2, 3 and 5.

For each combination the pair set from `LinkCell::visit_pairs` must equal
`BruteForce`'s pair set exactly — same unordered pairs, same multiplicity (no
duplicates, no omissions) — and every `dist_sq` must agree to 1e-12.

Points are deliberately seeded outside the box on non-periodic axes to exercise
clamping, and at the exact cell boundaries (fractional 0.0 and 1.0) to pin the
half-open convention. Point generation uses a small in-test LCG rather than the
`rand` crate: same stream on every platform, no dependency-version drift in a
correctness oracle.

The matrix is mutation-checked, not merely green. Reverting either fix — the
per-axis clamp, or the symmetric offset block — makes both the pair and the
query test fail, naming the configuration that broke. A matrix that passes
without ever having been shown to fail proves only that it ran.

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
