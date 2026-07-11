---
title: "graph-sink 3/4 — Python bindings for extract / merge map / copy contract"
status: done
created: 2026-07-11
chain: graph-sink
depends_on: "graph-sink-01-extract; graph-sink-02-copy-merge"
blocks: "molpy:graph-sink-01-wire-algorithms"
---

# PyO3 surface for graph-sink engine APIs

> Chain **graph-sink** 3/4 (`molrs-python`).  
> Depends on 01 (extract) + 02 (copy/merge). No new algorithms — bind only.

## Summary

Expose to Python (`import molrs`):

| Method | Type | Returns |
|---|---|---|
| `Atomistic.copy()` | already exists | independent clone, **same handles** |
| `CoarseGrain.copy()` | add if missing | same |
| `Atomistic.merge(other) -> dict[int,int]` | **new / change** | old→new **node** handles |
| `CoarseGrain.merge(other) -> dict[int,int]` | new | same |
| `Atomistic.induced_subgraph(nodes) -> (Atomistic, dict[int,int])` | new | subgraph + map |
| `Atomistic.extract_subgraph(centers, radius, *, regenerate_topology=False)` | new | structured result |
| `CoarseGrain.extract_subgraph(...)` | new | structured result |

Handle encoding: existing `node_to_u64` / `node_from_u64`.

## Design

### 1. Extract result object (prefer typed over huge tuple)

```python
# molrs.ExtractedSubgraph (or ExtractedBall)
class ExtractedSubgraph:
    graph: Atomistic | CoarseGrain
    boundary: list[int]          # parent handles
    parent_of: dict[int, int]    # new_handle -> parent_handle
    hops: dict[int, int]         # parent_handle -> hops
    node_map: dict[int, int]     # parent_handle -> new_handle
```

PyO3: small `#[pyclass]` wrapping the Rust struct; getters only.

Alternatively a 5-tuple if the project prefers fewer types — **prefer named class** for molpy
readability (iron law 4: real data-carrying type).

### 2. `merge` Python signature

```python
def merge(self, other: Atomistic) -> dict[int, int]:
    """Consume topology of other into self. other is left empty (or invalid).
    Returns map: handle_in_other -> handle_in_self.
    """
```

- After merge, `other` should be empty: either move out of `other.inner` or document that
  `other` is cleared via `std::mem::take` / replace with `Atomistic::new()`.
  **Fail-fast:** prefer leaving `other` empty so reuse is obvious error (0 nodes).
- Does **not** accept molpy subclass specially — duck-type via same pyclass / extract inner.

### 3. `copy` documentation

Update docstring: "Deep copy; **handles preserved**."

### 4. Stubs

`molrs-python/python/molrs/molrs.pyi` updated for all new methods and `ExtractedSubgraph`.

### 5. Tests

`molrs-python/tests/test_graph_sink.py` (new):

- copy handle preservation
- merge map round-trip
- extract_subgraph radius ball counts
- regenerate_topology flag

## Files to create or modify

- `molrs-python/src/core/system/molgraph.rs` — methods on `PyAtomistic` / `PyCoarseGrain`
- `molrs-python/python/molrs/molrs.pyi`
- `molrs-python/src/lib.rs` — register `ExtractedSubgraph` if new class
- `molrs-python/tests/test_graph_sink.py`

## Tasks

- [x] **T1**: Bind `merge` → `dict[int,int]`; clear `other`
- [x] **T2**: Bind `induced_subgraph` / `extract_subgraph` + result type
- [x] **T3**: Ensure `CoarseGrain.copy` exists and matches Atomistic
- [x] **T4**: `.pyi` stubs
- [x] **T5**: Python tests
- [x] **T6**: quality gate (maturin develop + pytest molrs-python)

## Testing strategy

- Pure molrs Python (no molpy import).
- Ethane extract radius 1 from one C → both C + 3H or whatever connectivity implies; map sizes match.
- merge two ethanes → 4 C? wait ethane is 2C+6H - just use bare C-C chains for simplicity.

## Out of scope

- molpy `def_*` / Entity interning
- Version publish process (but API must be ready to release)
- Geometric hydrogens → `graph-sink-04-hydrogens-coords`
