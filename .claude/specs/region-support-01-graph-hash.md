---
title: Region-support 1/2 — structural graph hash + canonical order + isomorphism on MolGraph
status: draft
created: 2026-07-05
---

# Structural graph hash + canonical order + isomorphism (MolGraph, Python-exposed)

> Add an **isomorphism-invariant structural hash** (Weisfeiler–Lehman / Morgan-style) on molrs
> `MolGraph`, plus a **canonical node ordering** and a **graph-equality / isomorphism** check.
> Native (serves both `Atomistic` and `CoarseGrain`), sits next to the existing
> `topo_distances` / `generate_topology` / SMARTS-matcher kernels, exposed to Python. This is
> the **dedup key** molpy's incremental-typification `AffectedRegion` hashes by, so identical
> polymer junctions retype once. Consumed by molpy `incremental-typify-*`.
> Architecture: `molpy/.claude/notes/incremental-typification-design.md`.

## Summary

Three related graph primitives on `MolGraph` (backing `Atomistic`/`CoarseGrain`), exposed via
PyO3:

1. **`structural_hash() -> u64`** — WL/Morgan hash invariant under graph isomorphism. Node label
   = element (atoms) / bead type (CG) + degree + formal charge + aromatic flag; edge label =
   bond order. Iterate WL refinement to convergence → 64-bit digest.
2. **`canonical_order() -> Vec<NodeId>`** — the WL-refinement stable ordering, so a caller can
   line up two isomorphic graphs' nodes deterministically (needed to map cached types onto a new
   region's atoms).
3. **`is_isomorphic(&self, other) -> bool`** — full-graph isomorphism, to resolve the rare hash
   collision before a cache hit is trusted. Reuse the existing backtracking subgraph-isomorphism
   machinery (`chem/smarts/matcher.rs`) or a WL-guided VF2.

## Domain basis

### Verified molrs pieces this builds on

- `MolGraph` (`molrs/src/core/system/molgraph.rs`) is the generic typed relation graph both
  `Atomistic` (`atomistic.rs`) and `CoarseGrain` (`coarsegrain.rs`) wrap
  (`as_molgraph`/`into_inner`/`try_from_molgraph`, `atomistic.rs:500-533`) — a primitive on
  `MolGraph` serves AA + CG.
- Existing kernels to reuse: `topo_distances` (`atomistic.rs:407`), `Topology::neighbors`
  (`topology.rs:164`), `neighbor_bonds`/`incident_bond_ids` (`:196,226`), `canonical_path`
  (`atomistic.rs:541`, endpoint sorting — a precedent, not a graph hash), the backtracking
  subgraph-iso matcher (`chem/smarts/matcher.rs:252`).
- `Element` derives `Hash` (`element.rs:5`); aromatic flag convention = bond order 1.5 /
  `is_aromatic` prop (`chem/smarts/ast.rs:70`); `perceive_aromaticity` (`chem/aromaticity.rs:660`).
- **No structural/canonical graph hash exists today** (verified across both repos) — greenfield.

### Why WL

Weisfeiler–Lehman gives an isomorphism-invariant multiset-refinement hash in near-linear time;
identical local environments (polymer junctions) converge to identical colors → identical hash.
A canonical order falls out of the final color refinement (ties broken by a stable secondary key).

## Design

### 1. Hash — `molrs/src/core/system/graph_hash.rs` (new)

```rust
pub fn structural_hash(g: &MolGraph) -> u64;
pub fn canonical_order(g: &MolGraph) -> Vec<NodeId>;
pub fn is_isomorphic(a: &MolGraph, b: &MolGraph) -> bool;
```

- WL refinement: initial color = hash(node label); each round color = hash(sorted multiset of
  neighbor (edge-label, color)); repeat until the color partition stops refining (or N rounds).
  Final graph hash = hash(sorted multiset of node colors).
- Node label reads element via the atoms' `element`/atomic-number component (CG: `bead_type`);
  degree from incident bonds; charge/aromatic from components when present (absent → neutral /
  non-aromatic). Edge label = bond `order` (default 1.0).
- `canonical_order` = nodes sorted by (final color, initial color, stable tiebreak).
- `is_isomorphic`: quick reject on `structural_hash` mismatch, else VF2/backtracking match with the
  WL colors as node-compat pruning.

### 2. Convenience on `Atomistic`/`CoarseGrain`

Thin methods delegating to the `MolGraph` fns (both wrap `MolGraph`): `structural_hash()`,
`canonical_order()`, `is_isomorphic(&other)`.

### 3. Python (`molrs-python`)

On `PyAtomistic` (and `PyCoarseGrain`): `structural_hash() -> u64`, `canonical_order() -> Vec<u64>`
(handles via `node_to_u64`), `is_isomorphic(&self, other: &PyAtomistic) -> bool`. Mirror the
`PySmartsPattern` binding idioms (`molgraph.rs`).

## Files to create or modify

- `molrs/src/core/system/graph_hash.rs` (new) — `structural_hash`/`canonical_order`/`is_isomorphic`
- `molrs/src/core/system/mod.rs` — module + re-export; `atomistic.rs`/`coarsegrain.rs` thin methods
- `molrs-python/src/core/system/molgraph.rs` — `PyAtomistic`/`PyCoarseGrain` methods
- `molrs-python/python/molrs/molrs.pyi` — stubs
- `molrs/tests/` + `molrs-python/tests/` — Rust unit + Python tests

## Tasks

- [ ] **T1**: WL `structural_hash(&MolGraph)` — label init + neighbor-multiset refinement to convergence
- [ ] **T2**: `canonical_order(&MolGraph)` — stable ordering from final colors
- [ ] **T3**: `is_isomorphic(a,b)` — hash pre-reject + WL-pruned VF2/backtracking
- [ ] **T4**: thin `Atomistic`/`CoarseGrain` methods delegating to MolGraph
- [ ] **T5**: PyO3 `structural_hash`/`canonical_order`/`is_isomorphic` on `PyAtomistic`/`PyCoarseGrain` + register + stubs
- [ ] **T6**: tests — invariance under node-order permutation; two identical junctions hash equal; non-isomorphic differ; charge/aromatic/bond-order sensitivity; CG bead-type graphs
- [ ] **T7**: quality gate — `cargo fmt --all --check` / `clippy -D warnings` / `check` / `test --all-features`; Python smoke

## Testing strategy

- **isomorphism invariance** — build a graph, build a node-permuted copy → equal `structural_hash`,
  `is_isomorphic == True`.
- **junction dedup** — two copies of the same local environment (e.g. an ethylene-glycol repeat
  junction) hash equal; a chemically different junction differs.
- **label sensitivity** — changing one atom's element / charge / aromatic flag, or a bond order,
  changes the hash.
- **canonical order** — two isomorphic graphs' `canonical_order` induce a consistent node bijection.
- **CG** — a coarse-grained bead graph hashes by bead type + topology.
- No regression: existing SMARTS/typifier tests unchanged.

## Out of scope

- **Region extraction / typify cache** — molpy `incremental-typify-*`
- **`Reaction.apply` touched-atom return** — `region-support-02`
- **Sub-hash of a rooted radius-N ball as a separate API** — the region subgraph is extracted in
  molpy then whole-hashed here; a rooted-WL variant is a follow-up if needed
