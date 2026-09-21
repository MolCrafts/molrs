---
spec: cgsmiles-02a-fragment-core
created: 2026-09-21
criteria:
  - id: ac-001
    summary: "Fragment is a MolGraph newtype peer with exactly the justified surface"
    type: code
    pass_when: "molrs/src/core/system/fragment.rs defines `pub struct Fragment { graph: MolGraph, bond: KindId, port: KindId }` with `new` (registers `bonds` and `ports`, both arity 2, ids distinct — tested; no registration order asserted), `Default`, `Deref`/`DerefMut` to `MolGraph`, `try_from_molgraph`, `into_inner`, `as_molgraph`, `add_atom_xyz`, `add_atom_bare`, `add_bond(a, b) -> Result<BondId, MolRsError>` (stamping `BondType::Single` / `BondNumber::Single` through `bond::write_bond_class`, which `Atomistic::set_bond_class` also calls), `n_atoms`, `n_bonds`, `#[derive(Debug, Clone)]`, the port surface, the `frag_id` surface and `to_frame`/`from_frame`; no `remove_port`, `as_molgraph_mut`, `induced_subgraph`, `extract_subgraph`, `merge`, `copy`, graph-hash delegate or `ExtractedFragment` exists on or beside `Fragment`; `Fragment`, `Port`, `PortId`, `PortKind` are re-exported from molrs/src/core/system/mod.rs and molrs/src/core/mod.rs."
    status: pending
  - id: ac-002
    summary: "PortKind is a closed enum validated only at the string boundary"
    type: code
    pass_when: "`pub enum PortKind { Symmetric, Left, Right, Shared }` exists with `as_str` returning `$`, `<`, `>`, `!` and `FromStr` returning `MolRsError::validation` for `\"$1.5\"` and `\"X\"` (both tested); `pub type PortId = RelationId`; `pub struct Port { anchor: AtomId, handle: AtomId, kind: PortKind, label: String, order: BondNumber }`; the rustdoc states that `label == \"\"` means unnamed, that the glyph is the storage form by decision, and that `PortKind` and 01a's `DescriptorKind` are two enums by design (AST vs core, the `BondKind`/`BondType` precedent) with the single conversion site in 02b."
    status: pending
  - id: ac-003
    summary: "Fragment::try_from_molgraph enforces the element invariant and never panics on kind conflicts"
    type: code
    pass_when: "Tests in fragment.rs show `try_from_molgraph` returning `MolRsError::validation` naming the node for a graph with a node lacking `keys::ELEMENT`, keeping both kind ids with no extra kind for a graph that already registered `bonds` and `ports` at arity 2, and returning `MolRsError::validation` (no panic) for a graph carrying a 3-ary kind named `ports`."
    status: pending
  - id: ac-004
    summary: "Atomistic and CoarseGrain resolve standard kinds by name with an arity check, not by dense id"
    type: code
    pass_when: "molrs/src/core/system/atomistic.rs and coarsegrain.rs contain no `unwrap_or(KindId(` in `try_from_molgraph`; each resolves its kinds by `kind_id(name)` with an arity check returning `MolRsError::validation` on conflict and otherwise idempotent `register_kind`; tests `try_from_molgraph_resolves_bonds_by_name_not_kind_zero` (a `MolGraph` whose first registered kind is `ports` with one populated port relation promotes with `n_bonds() == 0`, and a subsequent `add_bond` leaves the ports relation count unchanged) and `try_from_molgraph_rejects_conflicting_arity` (a 3-ary `bonds` kind → `MolRsError::validation`, no panic) pass in both files; the signature type `fn(MolGraph) -> Result<Self, MolRsError>` is unchanged (only the `mut mol` binding is added, which is not part of the public signature); the `atomistic.rs:604-605` rustdoc sentence about re-registration is now true and both rustdocs carry an `# Errors` entry for the arity conflict."
    status: pending
  - id: ac-005
    summary: "add_port checks endpoint roles and stores the prefixed triple; port() is the validating reader"
    type: code
    pass_when: "Tests in fragment.rs show `add_port` returning `MolRsError::not_found` for a stale atom, `MolRsError::validation` for a handle whose element is not `H`, for a handle not bonded to the anchor, and for `order == BondNumber::Unknown`; `add_port` writes `port_kind` (Str glyph), `port_label` (Str) and `port_order` (Int, through `From<BondNumber> for PropValue`) via `set_relation_prop` and `port()` reads the triple back through `BondNumber::from_prop`; `port()` returns `MolRsError::validation` for a relation missing a prop and for a `port_order` that does not read back as a definite `BondNumber` (`Unknown`, including out-of-range and negative codes), written through `DerefMut`; `n_ports()` counts two ports on one anchor; `add_bond` round-trips with `bond_type` / `bond_number` reading back `Single`; `try_from_molgraph` does not scan ports."
    status: pending
  - id: ac-006
    summary: "frag_id is a per-atom Int node property"
    type: code
    pass_when: "`set_frag_id(atom, id: u32) -> Result<(), MolRsError>` and `frag_id(atom) -> Option<u32>` round-trip per atom under the node key `frag_id` stored as `PropValue::Int`, two atoms hold two distinct values, `set_frag_id` rejects a value above `i32::MAX` with `MolRsError::validation`, and no whole-fragment broadcast setter exists."
    status: pending
  - id: ac-007
    summary: "inherit_frag_ids labels degree-1 neighbours of labelled atoms and nothing else"
    type: code
    pass_when: "Tests show: a degree-1 unlabelled H whose single neighbour carries a `frag_id` receives that value; an unlabelled atom with no labelled neighbour stays `None`; an already-labelled atom is unchanged; the return value equals the number of atoms labelled; a second call on the result labels nothing and returns 0 (one pass over a snapshot, not a fixpoint); the rustdoc states the precondition that a degree-1 atom deliberately left unassigned will be relabelled."
    status: pending
  - id: ac-008
    summary: "Frame round trip preserves ports, bonds and frag_id with no block relabeling and an all-or-nothing frag_id column"
    type: code
    pass_when: "`Fragment::to_frame` delegates to `MolGraph::to_frame` unrenamed and emits `atoms`, `bonds` and `ports` blocks (the `ports` block carrying `atomi`/`atomj` and the three `port_*` columns); a zero-port fragment's frame has no `ports` block; the `frag_id` column is present only when every atom carries one (a partially labelled fragment round-trips as unlabelled); `Fragment::from_frame(&f.to_frame())` restores port count, bond count and every `frag_id`; `from_frame` returns `Err` for a frame without an `atoms` block and `MolRsError::validation` for a frame whose `atoms` block carries no `element` column (test `from_frame_rejects_atoms_without_element`)."
    status: pending
  - id: ac-009
    summary: "mapping.rs is deleted and the docs it leaves stale are corrected"
    type: code
    pass_when: "molrs/src/core/system/mapping.rs does not exist; `grep -rn \"CGMapping\\|WeightScheme\" molrs/src molrs-ffi/src molrs-python/src molrs-wasm/src molrs-capi/src molrs-cxxapi/src docs` returns nothing; `pub mod mapping;` and the `CGMapping`/`WeightScheme` re-export are gone from system/mod.rs and core/mod.rs; the system/mod.rs module doc no longer says \"CG mapping\"; the `core` row of CLAUDE.md's crate-structure table no longer says \"atom-type mapping\"; `CoarseGrain::set_bead_members`, `bead_members` and `beads_of_atom` still exist."
    status: pending
  - id: ac-010
    summary: "No schema edit: the five names stay open props reserved in rustdoc"
    type: code
    pass_when: "`git diff --stat -- molrs/src/core/store/schema/` is empty; no `FRAG_ID` (or `PORT_*`) constant is added anywhere; the `Fragment` rustdoc names `frag_id`, `port_kind`, `port_label`, `port_order` and the `ports` block as open props reserved by this spec and points at the pending schema-vocabulary spec."
    status: pending
  - id: ac-011
    summary: "Public-API doctest on the fragment module runs green"
    type: runtime
    pass_when: "`cargo test --doc -p molcrafts-molrs --features full,filesystem,stream` passes and the module-level `//!` example in fragment.rs builds `C–C` with `add_atom_xyz` plus an `H` handle with `add_atom_bare`, two bonds (C–C, C–H), one `PortKind::Symmetric` port `(C0, H)` with label `\"A\"` and order `BondNumber::Single`, sets `frag_id` on every atom, and asserts the hard-coded values `n_atoms() == 3`, `n_bonds() == 2`, `n_ports() == 1`, the port's label and order read back, and `frag_id(atom) == Some(1)`, using only public API."
    status: pending
  - id: ac-012
    summary: "Rustdoc, doc-rot fix and deferred items are in place without touching notes.md"
    type: docs
    pass_when: "`add_port`'s rustdoc states the `(anchor, handle)` order and its `# Errors`; the `Fragment` rustdoc states the all-or-nothing `frag_id` column and the `\"\"` = unnamed label rule; `inherit_frag_ids`'s rustdoc states its precondition; the `Fragment` rustdoc states that the anchor–handle bond is checked at `add_port` only; the `coerce_canonical` doc at molgraph.rs:124 no longer cites bond `order`; the implementation summary names the `frag_id`-vs-`res_id`/`mol_id` decision, 01d's direct `frag_id` writer on `Atomistic`, the `read_frame` unregistered-kind skip (molgraph.rs:1017-1021) and the `emit_column` validity-mask drop (molgraph.rs:150-185) as deferred items for `/mol:note` / `/mol:fix`; `git diff` shows no change to .claude/notes/notes.md."
    status: pending
  - id: ac-013
    summary: "Full gate green"
    type: runtime
    pass_when: "`cargo fmt --check`, `cargo clippy -p molcrafts-molrs --all-targets --features full,filesystem,stream -- -D warnings`, `cargo test -p molcrafts-molrs --lib --features full,filesystem,stream` and `cargo test --doc -p molcrafts-molrs --features full,filesystem,stream` all exit 0."
    status: pending
out_of_scope:
  - "induced_subgraph / extract_subgraph / merge / copy / ExtractedFragment on Fragment (added by the link that first calls them)"
  - "remove_port / as_molgraph_mut (no caller yet)"
  - "Maintaining the anchor–handle bond invariant across DerefMut (checked at add_port only)"
  - "01d's direct frag_id stamp on Atomistic (second writer; reconciled by the schema-vocabulary spec)"
  - "Schema declaration of frag_id / port_* / ports (pending schema-vocabulary spec)"
  - "read_frame silently skipping unregistered kinds (routed via implementation summary)"
  - "emit_column dropping the validity mask (routed to /mol:fix)"
  - "Parsing fragment strings into a Fragment (02b)"
  - "Conformer generation for fragments (02c)"
  - "Python / WASM bindings (02d)"
  - "Multi-parent membership (frag_id list)"
  - "ports block support in any io format; perceive treatment of ports as valences"
---

# Acceptance — cgsmiles-02a-fragment-core

Done means: `Fragment` exists as a third `MolGraph` newtype with a closed port vocabulary, a per-atom `frag_id` and a Frame round trip that never invents a label; the kind-aliasing defect in the sibling promotions is fixed by name-and-arity resolution that cannot panic; `mapping.rs` and its stale docs are gone; and no schema file changes.

- **ac-001 / ac-002** pin the shape: the surface is exactly what this spec's tests and the named successor links justify, and the vocabulary is an enum validated only where strings enter.
- **ac-003 / ac-004** are the found-and-fixed criteria: the promotion constructors resolve kinds by name with an arity check, proven by the ports-first graph with a populated port relation — the case `KindId(0)` aliasing gets wrong.
- **ac-005 – ac-007** are the port and membership contracts: endpoint roles are checked at `add_port`, `port()` validates on read, `frag_id` is per atom, and `inherit_frag_ids` is a single documented pass.
- **ac-008** is the serialization contract, including the all-or-nothing `frag_id` column that keeps the `emit_column` defect from turning "unlabelled" into "labelled 0".
- **ac-009 / ac-010** are the deletion sweep and the no-schema-edit guard.
- **ac-011** is this repo's executable public-API example (no `regressions/` tree); **ac-012 / ac-013** are documentation and the gate.
