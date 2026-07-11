---
title: "graph-sink 1/4 — induced subgraph + radius ball extraction on MolGraph"
status: done
created: 2026-07-11
chain: graph-sink
depends_on: ""
blocks: "graph-sink-03-python-bind; molpy:graph-sink-01-wire-algorithms"
---

# Induced subgraph + extract_ball (engine primitive)

> Chain **graph-sink** 1/4 (molrs core).  
> Locked decisions: `molpy/.claude/notes/graph-sink-decisions.md` (A copy / B merge / C def_*).  
> This spec only adds **structure-fact** extraction kernels. No force-field / reach policy.

## Summary

Expose two native graph-extraction operations on `MolGraph`, with thin `Atomistic` /
`CoarseGrain` wrappers:

1. **`induced_subgraph(nodes)`** — induced subgraph on an explicit node set; remaps handles;
   returns `(subgraph, old→new node map)`.
2. **`extract_ball(centers, radius, kind, *, copy_higher_order)`** — multi-source BFS ball over a
   2-ary relation kind (default bonds), then materialise the induced subgraph; return
   subgraph + boundary + parent map + hops.

These replace molpy's Python `_extract_mapped` / `extract_subgraph` (O(ball) when
`copy_higher_order=false` regenerates higher-order terms locally).

## Domain basis

### Existing kernels to reuse

- `Atomistic::topo_distances(source, max_hops)` — single-source BFS over bonds
  (`atomistic.rs`); multi-source ball is the union of bounded BFS with min-hops-to-center.
- Arity-2 adjacency: `neighbor_relations` / `incident_bond_ids` on `MolGraph` / `Atomistic`.
- Higher-order kinds (`angles`/`dihedrals`/`impropers`) are **not** indexed per atom —
  scanning them is O(graph). Region path must **not** do that.
- molpy reference behaviour: `Atomistic._extract_mapped` in
  `molpy/src/molpy/core/atomistic.py` (to be deleted in molpy `graph-sink-01-wire-algorithms`).

### Iron law 2

Extraction answers **syntax/structure facts** ("which nodes lie within R hops of centers",
"what is the induced edge set"). Whether the result is a "retype-safe region" is molpy policy
and stays out of this crate.

## Design

### 1. Return types

```rust
// molrs/src/core/system/extract.rs (new) or inline in molgraph.rs

pub struct InducedSubgraph {
    pub graph: MolGraph,
    /// old parent NodeId → new NodeId in `graph`
    pub node_map: HashMap<NodeId, NodeId>,
}

pub struct ExtractedBall {
    pub graph: MolGraph,
    /// nodes in the ball that have a neighbor outside the ball (parent handles)
    pub boundary: Vec<NodeId>,
    /// new NodeId → parent NodeId
    pub parent_of: HashMap<NodeId, NodeId>,
    /// parent NodeId → hops from nearest center
    pub hops: HashMap<NodeId, i64>,
    /// old parent NodeId → new NodeId (inverse convenience)
    pub node_map: HashMap<NodeId, NodeId>,
}
```

Leaf wrappers return typed leaves (`Atomistic` / `CoarseGrain`) by
`try_from_molgraph` / domain constructors, not bare `MolGraph`, when called on a leaf.

### 2. `MolGraph::induced_subgraph`

```rust
pub fn induced_subgraph(&self, nodes: &[NodeId]) -> InducedSubgraph
```

- Only live nodes; stale handles ignored or error (fail-fast: **error on stale** — iron law 5).
- Copy all node components for selected nodes.
- For every registered kind: copy a relation iff **all** endpoints are in the node set.
- Remap relation endpoints via `node_map`.
- Empty `nodes` → empty graph + empty map.

### 3. `MolGraph::extract_ball`

```rust
pub fn extract_ball(
    &self,
    centers: &[NodeId],
    radius: i64,
    kind: KindId,           // usually bond kind
    copy_higher_order: bool,
) -> Result<ExtractedBall, MolRsError>
```

- `radius < 0` → error.
- Multi-source BFS on `kind` only (arity must be 2; else error).
- `hops[v] = min distance to any center`; include all `v` with `hops[v] <= radius`.
- Boundary: selected node with a `kind`-neighbor outside the selected set (reported as
  **parent** handles before remap; also available via `node_map` for new handles).
- Materialise subgraph:
  - Always copy nodes + all arity-2 relations of `kind` induced on the ball.
  - If `copy_higher_order`: copy every other kind induced on the ball (scan relations —
    OK for small graphs / `extract_subgraph` verbatim path).
  - If `!copy_higher_order`: only nodes + the ball's `kind` edges; caller regenerates
    angles/dihedrals (Atomistic wrapper may call `generate_topology`).

### 4. Leaf convenience

```rust
impl Atomistic {
    pub fn induced_subgraph(&self, atoms: &[AtomId]) -> Result<(Atomistic, HashMap<AtomId, AtomId>), MolRsError>;
    pub fn extract_subgraph(
        &self,
        centers: &[AtomId],
        radius: i64,
        *,
        regenerate_topology: bool, // true => copy_higher_order=false + generate_topology
    ) -> Result<ExtractedAtomistic, MolRsError>;
}

impl CoarseGrain {
    pub fn induced_subgraph(...);
    pub fn extract_subgraph(...); // no generate_topology; only bonds
}
```

`ExtractedAtomistic` mirrors `ExtractedBall` with `Atomistic` instead of `MolGraph`.

Membership on `CoarseGrain` (bead→atom handles): **copy membership entries** for selected beads
when present; foreign atom handles stay as opaque `u64` (no remapping of foreign world).

### 5. Complexity contract

| Mode | Cost |
|---|---|
| `extract_ball(..., copy_higher_order=false)` | O(\|ball\| × degree) via adjacency |
| `copy_higher_order=true` | O(\|ball\| × degree + \|all higher-order relations\|) scan |

Document the second mode as "small-graph / verbatim only".

## Files to create or modify

- `molrs/src/core/system/extract.rs` (new) **or** methods on `molgraph.rs`
- `molrs/src/core/system/mod.rs` — module + re-exports
- `molrs/src/core/system/atomistic.rs` — leaf wrappers + optional `generate_topology`
- `molrs/src/core/system/coarsegrain.rs` — leaf wrappers + membership copy
- `molrs/tests/core/` — unit/integration tests (synthetic graphs OK for pure graph logic)

## Tasks

- [x] **T1**: `MolGraph::induced_subgraph` + stale-handle fail-fast
- [x] **T2**: multi-source bounded BFS + boundary + hops
- [x] **T3**: `extract_ball` with `copy_higher_order` true/false
- [x] **T4**: `Atomistic::extract_subgraph` + regenerate path
- [x] **T5**: `CoarseGrain::extract_subgraph` + membership copy
- [x] **T6**: tests — ethane/PEO-like chain balls, empty centers, radius 0, higher-order count
- [x] **T7**: quality gate — fmt / clippy -D warnings / test --all-features

## Testing strategy

- Radius-0 around one atom = singleton (or atom + no neighbors); no exterior bonds.
- Linear chain of 10 atoms, centers={5}, radius=2 → 5 atoms; boundary = ends of the ball.
- Multi-center: hops = min over centers.
- Ethane with generated angles: `regenerate_topology=true` yields angles without scanning parent.
- Stale center handle → `Err`.
- CG two-bead bond: extract one bead radius 0 → one bead, membership preserved if set.

## Out of scope

- Python bindings → `graph-sink-03-python-bind`
- copy/merge semantics → `graph-sink-02-copy-merge`
- molpy view wrapping → molpy `graph-sink-01-wire-algorithms`
- AffectedRegion / TypeScope / reach policy
