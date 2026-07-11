---
title: "graph-sink 4/4 — geometric coordinates for add_hydrogens"
status: done
created: 2026-07-11
chain: graph-sink
depends_on: ""
blocks: "molpy:graph-sink-03-valence"
---

# add_hydrogens places X–H geometry when parent has xyz

> Chain **graph-sink** 4/4 (molrs core + optional python bind).  
> Closes the gap between molrs `chem::add_hydrogens` (counts only) and molpy
> `complete_valence` (counts + tetrahedral placement).

## Summary

Extend `molrs::chem::hydrogens::add_hydrogens` so that when a heavy atom has
`x`/`y`/`z` components, each newly added H receives coordinates at a standard
X–H length along valence-completing directions (same geometry intent as
`molpy/core/capping.py`). When the heavy atom lacks coordinates, H is still
added **without** xyz (current behaviour) — fail-open for topology-only graphs,
not a silent wrong position.

Also ensure the returned molecule **retains all relation kinds** of the clone
(already true if implemented via `mol.clone()` + add atoms/bonds only).

## Domain basis

### Current `add_hydrogens` (`chem/hydrogens.rs`)

- Clones mol; for each heavy under-valent by `implicit_h_count`, adds H with
  `element`/`mass` and bond order 1.0.
- **No coordinates.**
- Valence rule uses formal charge → effective atomic number (RDKit-like) — **keep**.

### Current molpy `complete_valence` (`capping.py`)

- Custom `_VALENCE` table + simple O/N charge tweaks.
- Places H with tetrahedral directions (`_cap_directions`) and `_CAP_LEN`.
- Copies **all** link kinds via Python views (angles/dihedrals preserved).
- Uses **new handles** (def_atom re-spawn) — will become clone-based under decision A.

### Unification rule

- **Counts / chemistry**: molrs `implicit_h_count` wins (already more careful on charge).
- **Geometry**: port molpy tetrahedral placement into Rust when xyz present.
- molpy `complete_valence` becomes a thin forwarder (molpy `graph-sink-03-valence`).

## Design

### 1. Placement algorithm (port, not invent)

For each heavy with `missing = n_implicit > 0` and finite x,y,z:

1. Collect unit vectors to existing bonded neighbors that also have xyz.
2. Compute `k = missing` directions via tetrahedral completion (same cases as
   molpy `_cap_directions`: n≥3, n=2, n=1, n=0).
3. Bond length from table (C 1.09, N 1.01, O 0.96, S 1.34, default 1.0 Å).
4. Set H `x,y,z`.

Constants named in code with rustdoc ("initial geometry; force field may refine").

### 2. API

```rust
pub fn add_hydrogens(mol: &Atomistic) -> Atomistic; // enhanced in place
```

No new flag required for v1: auto-place when heavy has xyz. Optional later:
`AddHydrogenOptions { place_coords: bool }` only if a caller needs topology-only
on a graph that has xyz — **not** in this spec.

### 3. Preserve non-bond relations

Implementation must start from `mol.clone()` (handle-preserving) and only
`add_atom` + `add_bond` for caps. Angles/dihedrals of the parent remain.

### 4. Python

`molrs.add_hydrogens(mol)` already exists — behaviour upgrade only; docstring
mentions coordinate placement.

## Files to create or modify

- `molrs/src/core/chem/hydrogens.rs` — placement
- `molrs/tests` or inline `#[cfg(test)]` for geometry
- `molrs-python` docstring only (if binding already exists)
- Optional: tiny scientific test vs known methanol geometry distances

## Tasks

- [x] **T1**: Port direction / length tables into hydrogens.rs
- [x] **T2**: Place H when heavy has xyz; skip coords when missing
- [x] **T3**: Tests — count parity with old implicit_h; C with 3H gets ~1.09 Å bonds
- [x] **T4**: Test under-valent radical fragment gains H count matching valence rule
- [x] **T5**: quality gate

## Testing strategy

- Methane from bare C (no bonds): 4 H; if C at origin, all H at ~1.09 Å.
- Ethene C=C (order 2): 4 H total (2 per C), not 6.
- Heavy without xyz: H present, `has(h, "x") == false`.
- Parent angles still present after add_hydrogens on a typed ethane with angles.

## Out of scope

- molpy deleting `capping.py` body → molpy `graph-sink-03-valence`
- Force-field minimization of cap positions
- Changing formal-charge valence rules
