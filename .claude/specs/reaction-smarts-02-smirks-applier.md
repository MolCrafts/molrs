---
title: Reaction-SMARTS 2/2 — Daylight reaction-SMARTS (SMIRKS) parser + applier
status: done
created: 2026-07-05
depends_on: reaction-smarts-01-python-matcher
---

# Daylight reaction-SMARTS (SMIRKS) transform: parse + apply

> Add a reaction-transform engine to molrs: parse a Daylight reaction SMARTS
> `reactants >> products`, derive the graph edit from the **atom-map diff** (Daylight SMIRKS
> transform semantics), and apply it to a matched occurrence by editing an `Atomistic` in
> place. Entirely greenfield (molrs has zero reaction support today) but sits directly on
> chain-`01`'s map-labelled matches and the existing core graph-edit + `generate_topology`
> primitives. Exposed to Python as `Reaction`; consumed by molpy's `crosslink-*` chain.

## Summary

New Rust `Reaction` type (`molrs/src/core/chem/smarts/reaction.rs`) + `PyReaction` binding:

- **Parse** `LHS >> RHS` (and tolerate `LHS > agent > RHS`, agent ignored) into reactant
  (multi-component, split on top-level `.`) + product `SmartsPattern`s, keyed by shared
  `map_label`s (chain-`01` already parses `[C:n]`).
- **Compile** a `Transform` from the LHS↔RHS atom-map diff (Daylight SMIRKS semantics):
  mapped-both-sides = preserved; unmapped LHS = deleted; unmapped RHS = added; bond present
  in product-not-reactant = formed; reactant-not-product = broken; order/charge/element
  changes applied.
- **Apply** to one occurrence: given a `{map → AtomId}` binding, edit the `Atomistic` in
  place via existing core fns (`add_bond`/`remove_bond`/`add_atom`/`remove_atom`/
  `set_bond_prop`), then `generate_topology(true,true,false)` + `perceive_aromaticity`.

**Reaction SMARTS, not strict SMIRKS**: SMARTS queries are allowed on reacting atoms
(RDKit-style), so functional groups can be queried (`[N;H2:1]`); we honor the SMIRKS
*transform* mechanics but not the "reacting atoms must be SMILES" restriction. No strict mode.

## Domain basis

### Daylight SMIRKS transform rules (verified, Daylight Theory Manual)

- *"reactant and product … same numbers and types of mapped atoms and the atom maps must be
  pairwise"* → each map class appears ≤ once per side; mapped-both = preserved.
- *"non-mapped atoms may be added or deleted"* → unmapped LHS deleted (leaving group),
  unmapped RHS added.
- bond diff between mapped atoms → form / break / change-order; *"stoichiometry is 1-1"*.
- hydrogens: explicit H must appear mapped on both sides; `[H]` = a hydrogen atom.

### molrs pieces this builds on (verified)

- chain-`01`: `SmartsPattern::parse` + `map_label` (LHS/RHS parse with atom maps), and the
  Python-exposed matcher.
- core edits (`core/system/atomistic.rs`): `add_bond`:158 / `remove_bond`:165 /
  `add_atom`:104 / `remove_atom`:121 / `set_bond_prop`:175; `generate_topology`:326;
  `perceive_aromaticity` (`core/chem/aromaticity.rs:660`).
- aromaticity convention: bond order 1.5 / `is_aromatic` prop (`ast.rs:70`) — a rewrite that
  changes aromatic bonds sets order/flag then re-perceives.

## Design

### 1. Parse — `Reaction::parse`

```rust
pub struct Reaction {
    reactants: Vec<SmartsPattern>,   // LHS components (top-level '.')
    product:   SmartsPattern,        // RHS
    transform: Transform,            // compiled map diff
}
impl Reaction {
    pub fn parse(reaction_smarts: &str) -> Result<Reaction, MolRsError>;
    pub fn reactant_smarts(&self) -> Vec<String>;          // for the caller to match per-component
    pub fn forming_bonds(&self) -> Vec<(u32, u32)>;        // map pairs of new bonds (distance criterion)
}
```

`>>` split is a top-level string scan (`>` is unambiguous in SMARTS); each side parsed by the
chain-`01` parser.

### 2. Compile — `Transform` (map diff, Daylight semantics)

```rust
pub struct Transform {
    form_bonds:  Vec<(u32, u32, f64)>,   // (map_a, map_b, order)
    break_bonds: Vec<(u32, u32)>,
    set_order:   Vec<(u32, u32, f64)>,
    delete_maps: Vec<usize>,             // unmapped LHS query-atom indices
    add_atoms:   Vec<AddAtomSpec>,       // unmapped RHS atoms (element/charge + connections)
    set_props:   Vec<(u32, AtomPropDelta)>,  // charge/element/H changes on preserved atoms
}
```

Validation: a map class appearing on only one side is an error (Daylight pairwise rule) —
except that a fully-unmapped atom is a delete (LHS) / add (RHS). Enforce "map class ≤ once per
side."

### 3. Apply — `Reaction::apply`

```rust
pub fn apply(&self, mol: &mut Atomistic, binding: &HashMap<u32, AtomId>) -> Result<(), MolRsError>;
```

Order: delete unmapped-LHS atoms → add unmapped-RHS atoms (element/charge from RHS template,
connect per `add_atoms`) → break/form bonds + set order → set preserved-atom props →
`generate_topology(true, true, false)` → `perceive_aromaticity`. Added atoms get **no
coordinates** (geometry is the caller's later concern).

### 4. Python — `PyReaction`

```rust
#[pyclass(name = "Reaction")]
struct PyReaction { inner: Reaction }
#[pymethods] impl PyReaction {
    #[new] fn new(reaction_smarts: &str) -> PyResult<Self>
    #[getter] fn reactant_patterns(&self) -> Vec<PySmartsPattern>   // LHS 组分，供 molpy 分别匹配
    #[getter] fn forming_bonds(&self) -> Vec<(u32, u32)>            // 成键映射号对，供距离判据
    fn apply(&self, mol: &mut PyAtomistic, binding: HashMap<u32, u64>) -> PyResult<()>  // 就地改
}
```

## Files to create or modify

- `molrs/src/core/chem/smarts/reaction.rs` (new) — `Reaction` / `Transform` / `compile` / `apply`
- `molrs/src/core/chem/smarts/mod.rs` — export `Reaction`; re-export at crate root (`core/mod.rs`)
- `molrs-python/src/core/chem/smarts.rs` — add `PyReaction`
- `molrs-python/src/lib.rs` — register `Reaction`
- `molrs/tests/` + `molrs-python/tests/` — Rust unit + Python integration tests
- `molrs.pyi` — stubs for `Reaction`

## Tasks

- [x] **T1**: `Reaction::parse` — top-level `>>` split (+ `>agent>` tolerate), LHS `.`-split components, each side via chain-`01` parser
- [x] **T2**: `compile` → `Transform` — LHS↔RHS map diff (form/break/set_order/delete/add/set_props); pairwise-map validation
- [x] **T3**: `Reaction::apply` — order delete→add→break/form→set→`generate_topology`→`perceive_aromaticity`; reuse core edit fns
- [x] **T4**: `AddAtomSpec` handling — unmapped RHS atoms element/charge from template, no coords
- [x] **T5**: `PyReaction` — `new`/`reactant_patterns`/`forming_bonds`/`apply`; register
- [x] **T6**: tests — parse, transform compile (Daylight semantics), single-occurrence apply + topology refresh
- [x] **T7**: quality gate — fmt/clippy/check/test all green; Python smoke

## Testing strategy

- **parse** — `"[N;H2:1].[C:2](=O)OC >> [N:1][C:2]=O"`: 2 reactant components, product parsed, maps 1/2 both sides.
- **compile (Daylight)** — above → `form_bonds` has (1,2); unmapped ester O + alkyl → delete; no false add.
  `"[C:1]=[C:2].[S;H1:3] >> [C:1][C:2][S:3]"` → set_order (1,2)→1.0 + form (2,3). One-sided map class → error.
- **apply** — bind an amine+ester occurrence, `apply(mol, {1:n,2:c,…})`: new N–C bond present; leaving atoms gone;
  atom count = orig − leaving; angles/dihedrals around the new bond regenerated via `generate_topology`.
- **add atoms** — a reaction with an unmapped RHS atom: new atom built with correct element, connected, in topology.
- No regression: chain-`01` matcher + typifier tests unchanged.

## Out of scope

- **Pair selection / conversion / distance loop** — molpy `crosslink-*` (this only applies one occurrence)
- **CoarseGrain reactions** — Engine A/apply target `Atomistic`; CG follow-up
- **Strict SMIRKS validation** (SMILES-only reacting atoms) — not done; permissive reaction SMARTS
- **Added-atom coordinate generation** — caller's later minimize step
- **`>agent>` agent semantics** — parsed-tolerated, agent ignored
