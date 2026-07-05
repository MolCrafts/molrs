---
title: Region-support 2/2 — Reaction.apply returns touched atom handles
status: draft
created: 2026-07-05
depends_on: reaction-smarts-02-smirks-applier
---

# Reaction.apply returns the touched atom handles

> `molrs.Reaction.apply` currently returns `()`. To let molpy build an `AffectedRegion` (the
> radius-N ball to retype), `apply` must report **which atoms it touched** — the atoms whose
> local environment changed and therefore need re-typification. Small, additive change on top of
> `reaction-smarts-02`. Consumed by molpy `incremental-typify-01`.
> Architecture: `molpy/.claude/notes/incremental-typification-design.md`.

## Summary

Change `Reaction::apply` (Rust) and `PyReaction.apply` (Python) to return the **touched atom
handles**: the atoms directly involved in the edit whose environment changed —

- endpoints of every formed / broken / order-changed bond,
- newly added atoms,
- surviving neighbors of deleted atoms (their environment changed),
- atoms whose properties (charge/element/H) were set.

molpy then expands these to a retype-safe radius via `extract_subgraph`. `apply` only reports the
**seed** touched atoms; it does not compute the ball.

## Domain basis

- `Reaction::apply(&self, mol: &mut Atomistic, binding) -> Result<(), MolRsError>`
  (`chem/smarts/reaction.rs:639`) already knows, from its compiled `Transform`, exactly which
  atoms it forms/breaks bonds on, adds, deletes, and sets props on — the touched set is a
  byproduct, not new analysis.
- `AtomId ↔ u64` via `node_to_u64` (bindings). Deleted atoms' own handles are gone; report their
  surviving neighbors instead (captured before deletion).

## Design

```rust
// Rust
pub fn apply(&self, mol: &mut Atomistic, binding: &HashMap<u32, AtomId>)
    -> Result<Vec<AtomId>, MolRsError>;   // returns touched (surviving) atom ids, deduped
```
```rust
// Python (PyReaction)
fn apply(&self, mol: &mut PyAtomistic, binding: HashMap<u32,u64>) -> PyResult<Vec<u64>>;
```

- Collect touched ids during `apply`'s existing edit steps (bond endpoints preserved+formed,
  added-atom ids, deleted atoms' surviving neighbors, prop-set atoms). Dedup. Return after
  `generate_topology`/`perceive_aromaticity`.
- Backward-compat: this changes the return type from `None` to `list[int]` — update the one
  existing caller path + tests (molpy crosslink currently ignores the return, so `list[int]`
  is harmless there until `incremental-typify-01` uses it).

## Files to create or modify

- `molrs/src/core/chem/smarts/reaction.rs` — `apply` returns `Vec<AtomId>`
- `molrs-python/src/core/system/molgraph.rs` — `PyReaction.apply -> Vec<u64>`
- `molrs-python/python/molrs/molrs.pyi` — stub return type
- `molrs/tests/` + `molrs-python/tests/` — assert touched set

## Tasks

- [ ] **T1**: collect touched (surviving) atom ids in `Reaction::apply`'s edit steps; dedup; return `Vec<AtomId>`
- [ ] **T2**: `PyReaction.apply -> Vec<u64>`; update stub
- [ ] **T3**: tests — touched set = bond-forming endpoints + added + deleted-neighbors + prop-set atoms; deleted atoms' own ids absent
- [ ] **T4**: quality gate — fmt/clippy/check/test all green; Python smoke

## Testing strategy

- **touched set** — amine+ester→amide: touched includes N and C (bond formed) and the ester
  oxygen's surviving carbon (leaving-group neighbor); the deleted atoms' own handles are NOT in the
  set. thiol-ene: touched = the two carbons + sulfur.
- **added atom** — a reaction adding an atom includes the new atom's handle.
- No regression: `reaction-smarts-02` reaction tests + molpy crosslink (ignores return) unaffected.

## Out of scope

- **Radius-N ball / region extraction** — molpy (this returns only seed touched atoms)
- **structural hash** — `region-support-01`
