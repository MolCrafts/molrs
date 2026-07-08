---
title: Reaction-SMARTS 1/2 — expose atom-map SMARTS matcher + graph-edit conveniences to Python
status: done
created: 2026-07-05
---

# Expose the SMARTS matcher (atom-map aware) + graph-edit ops to Python

> molrs already has a production, atom-map-aware SMARTS engine in Rust
> (`molrs::SmartsPattern`, `core/chem/smarts/`), used by the OPLS typifier. It is **not**
> exposed to Python — only the typifiers are. This spec adds a thin PyO3 wrapper
> (`SmartsPattern`) returning **map-keyed** matches, plus the few missing graph-edit
> conveniences (`remove_atom`/`remove_bond`/`set_bond_order`/`copy`) on `PyAtomistic`.
> Consumed by molpy's crosslinking layer (`molcrafts-molpy` chain `crosslink-*`) and by the
> reaction/SMIRKS applier (this chain `02`).

## Summary

Two thin binding tasks — **no new matching algorithm, no new core graph ops**:

1. **`PySmartsPattern`** (`#[pyclass(name = "SmartsPattern")]`) wrapping the existing core
   `molrs::SmartsPattern` (`molrs/src/core/chem/smarts/`). Exposes `parse` / `find_matches`
   / `has_match`, and crucially a **map-keyed** result `find_matches_mapped(mol) ->
   list[dict[int, int]]` (Daylight atom-map number → atom handle), since the crosslinking /
   reaction layer keys everything on atom maps.
2. **Graph-edit conveniences on `PyAtomistic`**: `remove_atom(handle)`,
   `remove_bond(handle)`, `set_bond_order(bond_handle, order)`, `copy()`. The core fns
   already exist (`atomistic.rs:121/165/175`, `#[derive(Clone)]` `:55`); only Python
   wrappers are missing (today only generic `despawn`/`remove_relation` are exposed).

## Domain basis

### What already exists (verified, molrs source)

Canonical engine — `molrs/src/core/chem/smarts/` (re-exported `molrs::SmartsPattern`,
`core/mod.rs:57`):

- `SmartsPattern::parse(smarts: &str) -> Result<SmartsPattern, MolRsError>` (`mod.rs:75`)
- `find_matches(&self, mol: &Atomistic) -> Vec<Vec<AtomId>>` (`mod.rs:83`) — indexed by
  query-atom order; DFS backtracking VF2, non-uniquified (RDKit `uniquify=False`)
- `has_match(&self, mol: &Atomistic) -> bool` (`mod.rs:88`)
- `map_label(&self, query_atom: usize) -> Option<u32>` (`mod.rs:117`) — **Daylight atom
  maps `[C:1]` already parsed** (`QueryAtom.map_label`, `parser.rs:515-525`), emitting
  `AtomPrimitive::Any` so the map **adds no match constraint** (correct "ignored in
  molecule SMARTS" semantics). Recursive `$(...)` supported.
- Production-grade: the OPLS typifier drives it (`ff/typifier/opls/layered.rs:161`,
  `deps.rs:60`), parity-tested.

Graph edit + topology (core `Atomistic`, `molrs/src/core/system/atomistic.rs`):

- `add_atom`/`add_bond`/`remove_atom` (cascades incident higher terms, `:121`)/
  `remove_bond` (`:165`)/`set_bond_prop` (`:175`); `add_angle`/`add_dihedral`/`add_improper`
- `generate_topology(gen_angle, gen_dihedral, clear_existing) -> Result<(usize,usize)>`
  (`:326`, idempotent, canonical-path dedup)
- `topo_distances(source) -> Vec<(AtomId, i64)>` (`:407`, BFS)
- `#[derive(Clone)]` (`:55`); `perceive_aromaticity(&mut Atomistic)`
  (`core/chem/aromaticity.rs:660`)

Python today (`molrs-python/src/core/system/molgraph.rs`): `PyAtomistic` exposes
`add_atom`/`add_bond`/`add_angle`/`add_dihedral`/`add_improper`/`generate_topology`/
`topo_distances`/`spawn`/`despawn`/`add_relation`/`remove_relation`. **No SMARTS pyclass;
no domain `remove_atom`/`remove_bond`/`set_bond_order`/`copy`.**

`AtomId ↔ u64` handle helpers exist (`node_to_u64`/`node_from_u64`, `molgraph.rs`).

## Design

### 1. `PySmartsPattern` (`molrs-python/src/…/smarts.rs`, new)

```rust
#[pyclass(name = "SmartsPattern")]
pub struct PySmartsPattern { inner: molrs::SmartsPattern }

#[pymethods]
impl PySmartsPattern {
    #[new] fn new(smarts: &str) -> PyResult<Self>            // molrs::SmartsPattern::parse
    fn has_match(&self, mol: &PyAtomistic) -> bool
    fn find_matches(&self, mol: &PyAtomistic) -> Vec<Vec<u64>>          // 位置序，atom handle
    fn find_matches_mapped(&self, mol: &PyAtomistic) -> Vec<HashMap<u32, u64>>  // 映射号→handle
    #[getter] fn num_query_atoms(&self) -> usize
    fn map_label(&self, query_atom: usize) -> Option<u32>
}
```

- `find_matches_mapped` = zip `find_matches` (per-match `Vec<AtomId>`) with `map_label(i)`
  for each query atom `i` that carries a map, → `{map: handle}`. Query atoms without a map
  are omitted from the dict (they still constrain the match). `AtomId → u64` via
  `node_to_u64`.
- Registered in `lib.rs` module init alongside the typifiers.
- No changes to the core matcher; pure binding.

### 2. `PyAtomistic` graph-edit conveniences (`molgraph.rs`)

```rust
fn remove_atom(&mut self, handle: u64) -> PyResult<()>       // core remove_atom (cascades)
fn remove_bond(&mut self, handle: u64) -> PyResult<()>       // core remove_bond
fn set_bond_order(&mut self, handle: u64, order: f64) -> PyResult<()>   // core set_bond_prop("order")
fn copy(&self) -> PyAtomistic                                 // core Clone
```

Each forwards to the existing core fn; `remove_atom` keeps the cascade semantics. `copy`
returns an independent `Atomistic` (clone), matching molpy's immutable-transform need.

## Files to create or modify

- `molrs-python/src/core/chem/smarts.rs` (new) — `PySmartsPattern`
- `molrs-python/src/core/chem/mod.rs` — expose the new module
- `molrs-python/src/lib.rs` — register `SmartsPattern`
- `molrs-python/src/core/system/molgraph.rs` — add `remove_atom`/`remove_bond`/
  `set_bond_order`/`copy` to `PyAtomistic`
- `molrs-python/tests/` (or `molrs/tests/`) — Python tests for matcher + edit conveniences
- `molrs.pyi` (if maintained) — type stubs for the new symbols

## Tasks

- [x] **T1**: `PySmartsPattern` — `new(smarts)`, `has_match`, `find_matches`, `map_label`,
  `num_query_atoms`; wrap `molrs::SmartsPattern`; register in `lib.rs`
- [x] **T2**: `find_matches_mapped -> Vec<HashMap<u32,u64>>` — zip match × `map_label`, `AtomId→u64`
- [x] **T3**: `PyAtomistic.remove_atom`/`remove_bond`/`set_bond_order`/`copy` — forward to core fns
- [x] **T4**: Python tests — atom-map read-out, `[C:1]` vs `[C]` same match set, edit conveniences
- [x] **T5**: quality gate — `cargo fmt --check && cargo clippy -D warnings && cargo test --all-features`;
  Python import + smoke

## Testing strategy

- **matcher parity** — `SmartsPattern("[C:1][O:2][H:3]").find_matches_mapped(mol)` on a known
  molecule returns the expected count; each dict `{1:c,2:o,3:h}` has handles of element C/O/H.
- **Daylight map semantics** — `find_matches("[C:1]")` and `find_matches("[C]")` return the
  **same** atom set (map adds no constraint).
- **edit conveniences** — `remove_atom` drops atom + incident bonds; `set_bond_order` changes
  order; `copy()` yields an independent graph (mutating the copy leaves the original intact).
- No regression: existing OPLS/MMFF typifier tests unchanged (core matcher untouched).

## Out of scope

- **Reaction SMARTS / SMIRKS `>>`** — chain `02`
- **CoarseGrain SMARTS matching** — Engine A matches `Atomistic`; CG matching is a follow-up
- **Engine B** (`io/smiles/smarts/`, positional/`atom_class` semantics) — not touched; Engine A is canonical
- **New matching algorithm / aromaticity model changes** — pure binding of the existing engine
