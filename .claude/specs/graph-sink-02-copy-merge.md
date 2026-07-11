---
title: "graph-sink 2/4 — lock copy (handle-preserve) + merge (remap + handle map)"
status: done
created: 2026-07-11
chain: graph-sink
depends_on: ""
blocks: "graph-sink-03-python-bind; molpy:graph-sink-01-wire-algorithms"
---

# copy / merge contracts (engine)

> Chain **graph-sink** 2/4 (molrs core).  
> Locked: **A** copy = handle-preserving clone; **B** merge = structural remap + returned map.  
> Decision note: `molpy/.claude/notes/graph-sink-decisions.md`.

## Summary

Make the two composition primitives **explicit, tested, and map-returning** where needed:

1. **`copy` / `Clone`** — deep clone; **node and relation handles preserved** (generational keys
   unchanged in the clone). Document this as the only supported copy semantics.
2. **`merge`** — structural merge already exists (`MolGraph::merge`) but **drops the node map**.
   Change it to return `HashMap<NodeId, NodeId>` (old-in-other → new-in-self). Optionally also
   return relation maps if cheap; **node map is mandatory**.

No identity-preserving "rebind the same object" API. No dual merge.

## Domain basis

### Current code

- `MolGraph::merge` (`molgraph.rs:~817`) already builds `node_map` internally and discards it.
- `Clone` on `MolGraph` / leaves uses slotmap clone → **handles preserved** (verify with test).
- Python `PyAtomistic.copy` returns `self.inner.clone()` — already handle-preserving.
- molpy's Python `copy` re-spawns via `def_atom` → **new handles** — will be deleted in molpy
  wire-up; this spec freezes the engine contract first.

### Why return the merge map

Cross-graph identity is handle-based (decision B). Callers that held handles into `other` need
`old → new` after `self.merge(other)`. Without the map they must re-match by coordinates/labels
(guessing identity — forbidden).

## Design

### 1. `Clone` contract (document + test)

```rust
// Behaviour (not a new type):
// let b = a.clone();
// for id in a.node_ids() { assert!(b.get_node(id).is_ok()); }
// Components and relations bitwise-equal under the same handles.
```

- Relation handles also preserved under clone.
- Public leaf method `copy(&self) -> Self` may alias `clone()` for API symmetry with Python.

### 2. `merge` signature change

```rust
impl MolGraph {
    /// Merge `other` into `self`, consuming `other`.
    /// Returns map: NodeId in `other` → NodeId in `self`.
    pub fn merge(&mut self, other: MolGraph) -> HashMap<NodeId, NodeId>;
}
```

- Kinds matched by name; register missing kinds on `self` (existing behaviour).
- All relation kinds transferred with remapped endpoints (existing).
- Empty `other` → empty map; `self` unchanged in structure.
- **Breaking** for in-crate Rust callers of `merge` (return type was `()`). Grep and fix
  all workspace call sites (molrs, molpack consumers if any in-tree, cxxapi).

Leaf wrappers:

```rust
impl Atomistic {
    pub fn merge(&mut self, other: Atomistic) -> HashMap<AtomId, AtomId>;
}
impl CoarseGrain {
    pub fn merge(&mut self, other: CoarseGrain) -> HashMap<BeadId, BeadId>;
    // also transfer `members` map: remap bead keys; foreign atom u64s unchanged
}
```

CoarseGrain membership: for each bead in `other`, after node remap, re-insert
`set_bead_members(new_bead, old_members.clone())`.

### 3. Optional relation map

Not required for v1. If easy, return `(node_map, rel_maps_by_kind)`; otherwise node map only.
molpy only needs node map for view rebinding after merge.

### 4. Explicit non-goals in API surface

- No `merge_identity`.
- No `copy_reindex`.
- Doc comments on `merge` / `clone` must state handle semantics in one sentence each.

## Files to create or modify

- `molrs/src/core/system/molgraph.rs` — `merge` return type
- `molrs/src/core/system/atomistic.rs` / `coarsegrain.rs` — leaf `merge` + CG members
- All in-workspace `merge(` call sites
- `molrs/tests/core/` — contract tests
- rustdoc on merge/clone

## Tasks

- [x] **T1**: Change `MolGraph::merge` to return `HashMap<NodeId, NodeId>`; fix call sites
- [x] **T2**: `Atomistic::merge` / `CoarseGrain::merge` + membership transfer
- [x] **T3**: Clone/copy handle-preservation tests (nodes + bonds + angles)
- [x] **T4**: Merge map correctness tests (two molecules, query remapped bond endpoints)
- [x] **T5**: rustdoc contracts
- [x] **T6**: quality gate

## Testing strategy

- **copy**: build ethane; clone; assert same atom/bond handle sets; mutate clone component →
  original unchanged.
- **merge map**: mol A (C-C), mol B (O-H); `map = A.merge(B)`; every B handle appears as key;
  `A.n_atoms() == 4`; bond O–H endpoints equal `map[old_o], map[old_h]`.
- **merge empty**: map empty; n_atoms unchanged.
- **CG merge**: membership on B's bead survives under remapped bead id.

## Out of scope

- Python binding of the map → `graph-sink-03-python-bind`
- molpy deleting identity-merge → molpy wire-up
- extract_* → `graph-sink-01-extract`
