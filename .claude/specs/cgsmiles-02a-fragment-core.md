---
title: CGsmiles 02a — Fragment core type and port surface
slug: cgsmiles-02a-fragment-core
status: approved
created: 2026-09-21
revised: 2026-09-21
chain: cgsmiles (01a → 01b → 01c → 01d → 01e → 02a → 02b → 02c → 02d → 03)
depends_on: none (pure core; may land in parallel with the 01* links)
---

# CGsmiles 02a — Fragment core type and port surface

## Summary

`molrs` can represent an all-atom molecule (`Atomistic`) and a coarse-grained one (`CoarseGrain`), but it has no representation of a *fragment*: a molecular graph that is deliberately incomplete, carrying named attachment points at which it will later be bonded to other fragments. This spec adds `Fragment`, a third newtype peer of `Atomistic` and `CoarseGrain` over `MolGraph`, plus the closed port vocabulary (`PortKind`) and the port surface (`add_port` / `port` / `ports` / `n_ports`) that the rest of the `cgsmiles-02` chain consumes: 02b turns a parsed fragment string into a `Fragment` and bonds handle atoms to anchors, 02c promotes a `Fragment` to `Atomistic` for conformer generation and back and relabels the hydrogens the pipeline adds, 02d binds it to Python. Only the surface this spec's own tests exercise — and the surface a named successor link calls — is shipped. While writing the newtype the existing kind-resolution in `Atomistic::try_from_molgraph` and `CoarseGrain::try_from_molgraph` was **found** to alias an unregistered kind onto `KindId(0)` (`mol.kind_id("bonds").unwrap_or(KindId(0))`), so a graph whose first registered kind is something else silently reports that foreign kind's relations as its bonds; this spec **fixes** it by resolving the standard kinds by name with an arity check, never by dense id.

## Domain basis

Requirement identifiers are the chain's CGsmiles catalogue. DOC = cgsmiles.readthedocs.io; REF = the CGsmiles reference implementation, github.com/gruenewald-lab/CGsmiles at commit 910c9ee (2026-09-14); BIG = Lin, T.-S. et al., *BigSMILES: A Structurally-Based Line Notation for Describing Macromolecules*, ACS Cent. Sci. **5**, 1523–1531 (2019), DOI 10.1021/acscentsci.9b00476.

- **R4.1 — A fragment is a molecular graph plus attachment points.** In the BigSMILES grammar a repeat unit is a SMILES fragment decorated with *bonding descriptors*; the fragment graph alone is not a molecule, because each descriptor marks a valence that is satisfied only when the fragment is joined to a neighbour [BIG; DOC fragments].
- **R4.3 — The descriptor vocabulary is closed.** BigSMILES spells complementary descriptors `<` and `>` (a `<` may bond only a `>`) and the self-complementary descriptor `$` (a `$` bonds any `$` of the same label); CGsmiles adds the shared descriptor `!`. Four glyphs, fixed by the grammar — not an open string. They map 1:1 onto `PortKind::{Left, Right, Symmetric, Shared}`, the same role names 01a gives the notation-side `io::smiles::DescriptorKind`; the two enums are deliberately distinct (Design, *Ports*) and 02b owns the one conversion between them [BIG; DOC].
- **R4.4 — A descriptor carries a label.** `$A` and `$B` are distinct classes that do not bond each other; the label is free-form text from the input and stays a `String` on the port [DOC; BIG].
- **R4.5 — A descriptor carries a bond order.** The bond formed when two descriptors are consumed has an integer order (single / double / triple / quadruple); the port stores it as a `BondNumber` (`core/system/bond.rs:44-54`), the multiplicity vocabulary `core` already owns; `BondNumber::Unknown` is not a legal port order. Nothing in this link forms the bond — 02b does — so the order is recorded and read back, never interpreted here.
- **R4.17 — A port names two atoms, not one.** A descriptor sits between the fragment's *anchor* atom (which keeps its place in the product molecule) and a *handle* atom — a real capping hydrogen, which is what the notation itself makes of an unconsumed descriptor ("interpreted as additional hydrogen atoms where applicable" [DOC chirality; REF resolve.py:431-433]) and what a paired descriptor's bond replaces. Storing only the anchor loses which valence was meant on a multivalent anchor, so the port is an arity-2 relation `(anchor, handle)`, which is exactly what the `MolGraph` kind registry expresses.
- **R5.1 — Every atom records the fragment instance it came from.** The reference annotates each atom of a resolved molecule with `fragid`, the identifier of the fragment instance that contributed it [REF resolve.py:290-300], so a resolved molecule can be partitioned back into fragments. `Fragment::set_frag_id(atom, id)` / `frag_id(atom)` write and read that annotation under the node property `frag_id`, per atom.
- **R5.2 — Fragment identity survives serialization.** A fragment that round-trips through the tabular `Frame` must come back with its ports and its `frag_id` values intact, otherwise the R5.1 partition is lost the first time a fragment is written to disk or crosses the FFI boundary. `[!]` (shared atoms) is rejected at parse (01c), so a `Fragment` never holds a shared atom and `frag_id` is a scalar, not a list.

Units: none of the port data is dimensioned. Atom coordinates carried by `add_atom_xyz` are Å, as everywhere in `core`.

## Design

### The newtype

`Fragment` is the third leaf over `MolGraph`, built on the `CoarseGrain` pattern (`molrs/src/core/system/coarsegrain.rs:57`): a private `graph: MolGraph` plus the `KindId`s the leaf owns, `Deref`/`DerefMut` to `MolGraph` for the generic graph surface, `Default` delegating to `new()`, and `#[derive(Debug, Clone)]` as both siblings carry (`atomistic.rs:72`, `coarsegrain.rs:57`) — 02d's `PyFragment::copy` clones the core value.

```rust
pub struct Fragment { graph: MolGraph, bond: KindId, port: KindId }
```

`new()` registers `bonds` (arity 2) and `ports` (arity 2) and keeps both ids. Registration order carries no meaning — every consumer resolves kinds by name (`read_frame` at `molgraph.rs:1017-1021`, `materialize_induced` at `extract.rs:269-272`) once the dense-id fallbacks below are gone — and no test asserts an order. Both fields are read: `bond` by `add_bond`, `port` by the port surface.

Shipped surface, and the caller that justifies each:

| Symbol | Justified by |
|---|---|
| `new` / `Default` / `Deref` / `DerefMut` | this spec's tests; peer parity with `Atomistic` / `CoarseGrain` |
| `try_from_molgraph(mol) -> Result<Self>` | 02b, 02c; tested here (element invariant, kind idempotence, arity conflict) |
| `into_inner` / `as_molgraph` | 02c (`ElementGraph::as_molgraph`; promotion to `Atomistic` and back) |
| `add_atom_xyz` / `add_atom_bare` | 02d's typed `PyFragment.add_atom` wrapper — the same door `PyAtomistic` (`molrs-python/src/core/system/molgraph.rs:656`) exposes over its leaf, and how molpy's hand-built fragments get atoms; this spec's doctest (02b builds its handles on the `Atomistic` before promotion and calls neither) |
| `add_bond(a, b) -> Result<BondId>` | 02d's typed `PyFragment.add_bond` wrapper (peer of `PyAtomistic::add_bond` `:665` and `PyCoarseGrain::add_bond` `:1439`), the one door through which a hand-built template gets classed bonds; this spec's doctest and round-trip test |
| *(if 02d ships `PyFragment` without the typed `add_atom` / `add_bond` pair, `add_atom_xyz`, `add_atom_bare`, `add_bond` and `bond::write_bond_class` come out with it — the doctest then builds through `Atomistic` and promotes, as 02b does)* | |
| `n_atoms` / `n_bonds` | this spec's doctest; 02d's `n_atoms` getter; the same two counts `Atomistic` exposes (`atomistic.rs:171`) over `n_nodes` / `n_relations(self.bond)` |
| `add_port` / `port` / `ports` / `n_ports` | 02b (writes ports), 02d (`views.Port`, `n_ports`); tested here |
| `set_frag_id(atom, id)` / `frag_id(atom)` | R5.1; this spec's tests and doctest; 02c's round-trip fixture writes and reads per atom (01d never holds a `Fragment` — see *Per-atom membership*) |
| `inherit_frag_ids() -> usize` | 02c (relabels pipeline-added hydrogens); tested here |
| `to_frame` / `from_frame` | R5.2; 02d (frame at the binding seam); tested here |
| `PortKind::as_str` / `FromStr` | the string boundary; 02d needs `as_str` |

Deliberately **not** shipped (see Out of scope): `remove_port` and `as_molgraph_mut` (no link names a caller, and sibling parity is not a caller — Shape check 3), `induced_subgraph`, `extract_subgraph`, `merge`, `copy`, the `graph_hash` delegates (`Deref` already reaches them) and an `ExtractedFragment` type.

### `try_from_molgraph` and the kind-aliasing fix — by name, with an arity check, never a panic

`Atomistic::try_from_molgraph` (`atomistic.rs:606-627`) and `CoarseGrain::try_from_molgraph` (`coarsegrain.rs:243-258`) resolve their kinds with `mol.kind_id("bonds").unwrap_or(KindId(0))` (and `1`/`2`/`3` for angles/dihedrals/impropers). On a graph that registered some *other* kind first — precisely what a `Fragment` produces, and what this spec therefore makes reachable — the promoted type reports a foreign kind's relations as its bonds, and the next `add_bond` writes into that foreign kind. This is a live defect in a surface this spec depends on, so per the iron law it is fixed here (found + fixed), not worked around. The rustdoc above it (`atomistic.rs:604-605`) already promises "the graph's relation kinds are re-registered to the standard set", which the `unwrap_or` body does not do; the fix makes that sentence true, and both rustdocs gain an `# Errors` entry for the arity conflict.

The replacement is **not** a bare `register_kind` call: `register_kind` asserts on an arity conflict (`molgraph.rs:470-473`), and a `Result`-returning constructor must not abort on caller-supplied input. All three `try_from_molgraph` take `mut mol` and, for each kind name they own: if `mol.kind_id(name)` is `Some(kid)` and `mol.arity(kid) != expected`, return `MolRsError::validation` naming the kind and both arities; otherwise `register_kind(name, arity)`, which is documented idempotent for a matching name+arity (`molgraph.rs:464-482`) and creates a fresh, distinct kind when absent. Verified consequences: by-value `mut` is not part of the public signature, so no caller changes (all eleven `try_from_molgraph` occurrences — `atomistic.rs:606,649,657,681,1140,1144`, `coarsegrain.rs:243,283,307,403,407` — are in `core/system/`, none in a binder); `to_frame` skips kinds with no relations (`molgraph.rs:986`), so a newly registered empty kind adds no frame block. The four in-crate re-wrap paths (`atomistic.rs:657,681`, `coarsegrain.rs:283,307`, fed by `materialize_induced`, which mirrors every parent kind by name) are exactly where a `Fragment`-derived graph carrying `ports` will meet this code, which is why the check is by name and arity.

`Fragment::try_from_molgraph` additionally enforces the element invariant, reusing the `Atomistic` pattern verbatim (`atomistic.rs:606-615`): every node must carry `keys::ELEMENT`, else `MolRsError::validation` naming the node. Tests cover the element rejection, idempotence on a graph that already registered both kinds, and the conflicting-arity rejection (a graph with a 3-ary kind named `ports`).

### Ports

`PortKind` is a closed enum, following the `BondType` / `BondNumber` precedent (`bond.rs:23-54`): a domain vocabulary is an enum, never a validated `String` inside the type system. It departs from that precedent in one respect, stated so it is a decision and not drift: its storage form is the notation glyph (`$ < > !`) as a `Str` column rather than a numeric `code()`. Reason: `port_kind` is an undeclared open column (no schema dtype to pin a code table to), the glyph is the only spelling a user ever writes or reads (`[$]` in the notation, molpy's `def_port(..., "$")`, 02d's Python seam), and a one-character string in a Frame is self-describing where a bare integer is not. The coupling this creates is named: `core` adopts the BigSMILES/CGsmiles glyphs as the canonical short names of the four port roles; it does not parse the notation. **`PortKind` and 01a's `io::smiles::DescriptorKind` are two enums over the same four role names by design**, mirroring the existing split between the notation AST's `BondKind` and the core model's `BondType`/`BondNumber` (`io/smiles/smiles/to_atomistic.rs:353-377`; 01d moves that pair onto `BondKind` in `chem/ast.rs` — the precedent is the split, not the file): the AST names what was written, `core` names what is stored, and the one conversion site between them belongs to 02b (`cgsmiles/to_fragment.rs`) — never to `core`, which does not name `io`, and never duplicated elsewhere.

```rust
pub enum PortKind { Symmetric, Left, Right, Shared }   // $  <  >  !
pub type PortId = RelationId;                             // mirrors `BondId` (atomistic.rs:52)
pub struct Port { pub anchor: AtomId, pub handle: AtomId, pub kind: PortKind, pub label: String, pub order: BondNumber }
pub fn add_port(&mut self, anchor: AtomId, handle: AtomId, kind: PortKind, label: &str, order: BondNumber) -> Result<PortId, MolRsError>;
```

`as_str()` returns the glyph; `FromStr` returns `MolRsError::validation` for anything else. The conversion happens only at the `set_relation_prop` / frame boundary, as `BondType::code()` does — inside the type system a bad descriptor is unrepresentable, so the only place a bad glyph can arrive is a string boundary (`from_str`, `from_frame`), and that is the only place validation lives. `label` is a `String` where `""` means **unnamed** (a bare `$`/`<`/`>`), stated in the rustdoc. `order` is a `BondNumber` — the multiplicity vocabulary `core` already owns and the type 01d hands `set_bond_class` — written as `port_order` (Int) through the existing `impl From<BondNumber> for PropValue` (`bond.rs:155-159`) and read back with `BondNumber::from_prop` (`bond.rs:131-138`) after the prop's presence is checked; `add_port` rejects `BondNumber::Unknown` with `MolRsError::validation` (a port always has a definite multiplicity, R4.5), so no hand-rolled integer conversion exists anywhere on the port surface — `from_prop` is total, an out-of-range or negative code reads as `Unknown`, and `port()` rejects that.

**What `anchor` and `handle` are.** `anchor` is the fragment atom that keeps its place in the product molecule; `handle` is the real capping hydrogen bonded to it (R4.17) — a node with `element = "H"`, counted by `n_atoms()`, never a dummy `*` (the conformer needs `element`, and the notation's own meaning of an unpaired descriptor is a hydrogen). Endpoint order is `(anchor, handle)` and is load-bearing: `MolGraph::add_relation(self.port, &[anchor, handle])` (`molgraph.rs:683`) already rejects a stale or unknown atom handle with `not_found`, so `add_port` does not re-check existence; it does check that `handle` carries `element == "H"` and is bonded to `anchor` (both cheap, both stated in `# Errors`), which is what makes a swapped call site fail instead of recording a backwards port. The descriptor is then written with three `set_relation_prop` calls (`molgraph.rs:750`).

Descriptor keys are **prefixed**: `port_kind` (Str), `port_label` (Str), `port_order` (Int). Reason: `order` is already used as an open relation-prop key at an incompatible dtype (`F64`, written by the UFF typifier at `ff/typifier/uff/mod.rs:259,294,316`, documented at `molgraph.rs:372`) and `label` at `molgraph.rs:1409`; `check_schema` binds a key across *every* block by design (`schema/column.rs:33-41`), so a future declaration of `order` would have to reconcile UFF's `F64` with a port's integer. The prefix pre-empts that dtype conflict at the Frame boundary and makes the port props self-describing in a flat `ports` block.

`port()` is the validating reader: a relation of the `ports` kind missing any of the three props, whose `port_kind` fails `FromStr`, or whose `port_order` (stored as `Int`, i.e. `i32` — `PropValue` has no unsigned variant, `molgraph.rs:71-76`) does not read back as a definite `BondNumber` (`Unknown`, including out-of-range and negative codes), yields `MolRsError::validation`; tested. It validates the three props only: the anchor–handle **bond** is a construction-time check at `add_port`, not an invariant maintained across `DerefMut` — a caller can `remove_relation` the bond through `Deref` and leave a port whose handle is unbonded; the `Fragment` rustdoc states this and Out of scope lists it. Promotion does **not** scan ports, so `try_from_molgraph` stays O(nodes) like its siblings.

`add_bond(a, b) -> Result<BondId, MolRsError>` mirrors `Atomistic::add_bond` (`atomistic.rs:178-182`) **exactly**: `add_relation(self.bond, &[a, b])`, then the default class `(BondType::Single, BondNumber::Single)` — `Atomistic::add_bond` stamps both facts through `set_bond_type` → `set_bond_class` (`:206-224`), and a fragment bond must not be the one classless bond in the crate (`valence_demand` in `perceive/hydrogens.rs:381` reads the number; a template built by hand — this spec's doctest, molpy's fragments through 02d — would otherwise carry `Unknown` where a 02b template carries `Single`). This is the second real use of the two-key write in `Atomistic::set_bond_class` (`keys::BOND_TYPE`, `keys::BOND_NUMBER`), so it is extracted, not copied: a `pub(crate) fn write_bond_class(graph: &mut MolGraph, kind: KindId, id: RelationId, bond_type: BondType, bond_number: BondNumber) -> Result<(), MolRsError>` in `core/system/bond.rs` (the vocabulary's home) is called by `Atomistic::set_bond_class` and by `Fragment::add_bond`; `Fragment` gains no `set_bond_class` of its own (no caller). The round-trip test reads `bond_type` / `bond_number` back as `Single`.

### Per-atom membership: a node property, not a side map

`CoarseGrain` keeps bead → atoms in a `HashMap<BeadId, Vec<u64>>` (`coarsegrain.rs:49-61`) because those atom handles name a *foreign world*. `Fragment` is the inverse situation — the atom and the fragment ordinal live in one graph, and the relation is a scalar per atom — so it is a node property:

```rust
pub fn set_frag_id(&mut self, atom: AtomId, id: u32) -> Result<(), MolRsError>;
pub fn frag_id(&self, atom: AtomId) -> Option<u32>;
pub fn inherit_frag_ids(&mut self) -> usize;
```

Per atom, mirroring `Atomistic::set_atom` (`atomistic.rs:151-158`): membership is a per-atom fact (02c reads and re-labels per atom), so a whole-fragment broadcast would be a second door callers bypass through `Deref`. **Second writer, named:** 01d's expansion stamps `frag_id = PropValue::Int(node index)` directly on an `Atomistic` (`cgsmiles-01d-resolve`, *Expansion*) — it never holds a `Fragment`, so it cannot go through `set_frag_id`, and the `i32::MAX` check below does not guard that door (a node index above `i32::MAX` is unreachable in practice, but the door is unvalidated). This spec reserves the key and names the second door; the pending schema-vocabulary spec, which declares `frag_id`, is where the two writers get one validated owner. The implementation summary lists it with the other routed items. Storage width: node columns are `I = i32` (`molgraph.rs:590`); `set_frag_id` returns `MolRsError::validation` above `i32::MAX`. `inherit_frag_ids` is the `Fragment` invariant 02c relies on ("a `frag_id` propagates to atoms attached to a labelled atom"): one pass over a snapshot of the input labels, never a fixpoint; only degree-1 nodes without a `frag_id` whose single neighbour carries one are labelled; returns the number of atoms labelled; documented precondition "intended for atoms newly attached by a pipeline (e.g. hydrogens added by the conformer) — a degree-1 atom deliberately left unassigned will be relabelled".

### Why the key is `frag_id`, not `res_id` or `mol_id` — a recorded decision

`mol_id` groups atoms into molecules (`schema/mod.rs:216-222`); a fragment instance is sub-molecular. `res_id` (`schema/mod.rs:288-295`) is the biopolymer residue key that the 2026-09-21 operator decision earmarks for the pending **schema-vocabulary spec**, which gives the residue/fragment concept one abstract name across molrs and its I/O formats; writing fragment instances into `res_id` now would couple this chain to exactly the vocabulary that spec replaces. `frag_id` is the provisional open-prop spelling of that one concept (mirroring the reference's `fragid`), and the schema-vocabulary spec reconciles the two names (rename or alias) when it declares the fragment/port vocabulary. This decision is named in the implementation summary and recorded by the orchestrator through `/mol:note`.

### Schema position — no schema edit

`frag_id`, `port_kind`, `port_label`, `port_order` and the `ports` block are **open props**, declared in no schema. Verified: `check_schema` (`core/store/block/mod.rs:869-872`) returns `Ok` for any key absent from the schema vocabulary ("a key with no spec is unconstrained — that is the extension point", `schema/column.rs:38-41`), and `coerce_canonical` (`molgraph.rs:129-138`) falls through when `canonical_dtype` is `None`, so these round-trip through `to_frame` / `read_frame` today as plain columns. Declaring `frag_id` canonical while leaving the port triple and the `ports` block undeclared would be half a decision; the whole fragment/port vocabulary is declared together by the pending schema-vocabulary spec, and when it declares `port_kind` the glyph-vs-code storage form is re-decided with it. This link touches no file under `core/store/schema/` and defines no `FRAG_ID` constant. `Fragment`'s rustdoc states that the five names are open props **reserved by this spec**.

### `to_frame` / `from_frame`

`to_frame` delegates to `MolGraph::to_frame` unrenamed: a fragment's nodes *are* atoms, so unlike `CoarseGrain` there is no `atoms`→`beads` relabeling. The emitted frame carries an `atoms` block and, when at least one port exists, a `ports` block (`atomi`/`atomj` endpoints plus the three `port_*` columns); a `bonds` block when bonds exist (`to_frame` skips empty kinds, `molgraph.rs:986`, so a zero-port fragment's frame is byte-identical to an `Atomistic` frame). **`frag_id` is emitted all-or-nothing:** `emit_column` (`molgraph.rs:150-185, 975`) drops the validity mask and writes the default `0` for a null cell (a pre-existing defect on every open `Int` column, routed — see Out of scope), so an unassigned `frag_id` is not representable in a `Frame`; `Fragment::to_frame` therefore emits the `frag_id` column only when every atom carries one, and a partially labelled fragment round-trips as "unlabelled" rather than as "labelled 0" (a caller that wants labels in the Frame calls `inherit_frag_ids` first; 02d respects the same rule at the Python seam). `from_frame` builds `Self::new()` — registering `bonds` and `ports`, the registration `read_frame` requires to restore a kind's relations (`molgraph.rs:1017-1021`) — delegates to `read_frame`, and re-checks the element invariant. Found, not changed here: `read_frame` silently skips blocks for kinds not registered on the receiver, so `Atomistic::from_frame` on a `Fragment` frame drops the `ports` block without a word (routed via the implementation summary).

### Doc rot fixed in place

`coerce_canonical`'s doc (`molgraph.rs:124`) cites a canonical bond `order` Float field beside `charge` as its widening examples; `order` is not in `SCHEMA_COLUMNS`, so that half of the example never happens. The fix is striking the `order` clause from that sentence (`charge` already stands as the live example), local and stage-allowed, so it is fixed here (found + fixed) rather than routed. `.claude/notes/notes.md` is `/mol:note`'s file and is not edited.

### Deletion sweep — `mapping.rs`, and the docs it leaves stale

`core/system/mapping.rs` (`CGMapping`, `WeightScheme`) has zero consumers: the symbols appear only in the file itself and its re-export at `core/mod.rs:82`; the 20 sibling repos hold only vendored copies of molrs. The module, its `pub mod` line (`system/mod.rs:13`) and its re-export go; `CoarseGrain::set_bead_members` / `bead_members` / `beads_of_atom` are independent and stay. Two docs go stale with it and are corrected in the same change: `system/mod.rs:1-3` ("…and CG mapping") and the `core` row of CLAUDE.md's crate-structure table (`CLAUDE.md:186`, "atom-type mapping" — `mapping.rs` is the only thing in `core/` that phrase can refer to; line 186 sits below the `mol:bootstrap:managed end` marker, where free-form rows are edited directly by the implementation, as 01b does for line 189, while `.claude/notes/notes.md` stays `/mol:note`'s); `architecture-rules.md:57` does not list it and needs no edit.

### Reuse decision

- `reuse` — `CoarseGrain` newtype pattern (`coarsegrain.rs:57-94`, `260-273`): struct shape, `Deref`/`DerefMut`/`Default`, `new`, `into_inner`, `as_molgraph`.
- `reuse` — `MolGraph::register_kind` / `kind_id` / `arity` / `add_relation` / `set_relation_prop` / `remove_relation` / `relations` / `n_relations`: the port and bond surfaces are thin domain naming of these.
- `reuse` — element-invariant pattern (`atomistic.rs:606-615`) and `set_atom` shape (`atomistic.rs:151-158`) for the per-atom `frag_id` accessors.
- `reuse` — `BondType` closed-vocabulary-with-`code` pattern (`bond.rs:23-54`) for `PortKind`; `BondNumber` as the port order; `pub type BondId = RelationId` (`atomistic.rs:52`) for `PortId`.
- `generalize` — the two-key bond-class write in `Atomistic::set_bond_class` (`atomistic.rs:206-215`) → `bond::write_bond_class`, shared with `Fragment::add_bond` (second real use).
- `generalize` — `CGMapping.bead_mask` → the per-atom `frag_id` node prop; the deletion of `mapping.rs` completes that promotion.
- `new` — `Fragment`, `Port`, `PortId`, `PortKind`, `inherit_frag_ids`: no existing type expresses a graph with unsatisfied named valences, and nothing propagates a node property along bonds.

## Files to create or modify

- `molrs/src/core/system/fragment.rs` (new) — `Fragment`, `Port`, `PortId`, `PortKind`, module `//!` doc with the doctest, inline `#[cfg(test)]` tests.
- `molrs/src/core/system/mod.rs` — `pub mod fragment;` + re-export; remove `pub mod mapping;`; rewrite the module doc.
- `molrs/src/core/mod.rs` — re-export `Fragment`, `Port`, `PortId`, `PortKind`; remove the `CGMapping`/`WeightScheme` re-export.
- `molrs/src/core/system/atomistic.rs` — `try_from_molgraph`: by-name resolution with arity check, then idempotent `register_kind`; the `:604-605` rustdoc made true plus an `# Errors` entry; `set_bond_class` delegates to `bond::write_bond_class`; tests (ports-first graph with a populated port relation; conflicting arity).
- `molrs/src/core/system/bond.rs` — `pub(crate) fn write_bond_class` (the two-key write moved out of `Atomistic::set_bond_class`), with its unit test.
- `molrs/src/core/system/coarsegrain.rs` — same fix and tests for `bonds`.
- `molrs/src/core/system/molgraph.rs` — the `coerce_canonical` doc sentence (`:124`): strike the bond `order` clause.
- `molrs/src/core/system/mapping.rs` — delete.
- `CLAUDE.md` — `core` row of the crate-structure table (line 186): strike "atom-type mapping".

## Tasks

- [ ] Write failing unit tests in `molrs/src/core/system/fragment.rs` for `PortKind` (`as_str` glyphs `$ < > !`, `FromStr` rejects `"$1.5"` and `"X"`), `Fragment::new` (registers `bonds` and `ports` at arity 2, ids distinct), the element invariant, `try_from_molgraph` idempotence (graph already registering both kinds keeps both ids, no extra kind) and the conflicting-arity rejection (a 3-ary kind named `ports` → `MolRsError::validation`, no panic)
- [ ] Implement `PortKind`, `Port`, `PortId` and the `Fragment` newtype in `molrs/src/core/system/fragment.rs` (`new`, `Default`, `Deref`, `DerefMut`, `try_from_molgraph` with the arity check, `into_inner`, `as_molgraph`, `add_atom_xyz`, `add_atom_bare`, `add_bond`, `n_atoms`, `n_bonds`; `#[derive(Debug, Clone)]`)
- [ ] Write failing unit tests for the port surface: `add_port` rejects a stale atom, a non-H handle, a handle not bonded to the anchor, and `BondNumber::Unknown`; `add_port` records kind/label/order and `port()` reads the triple back; `port()` returns `Err` for a missing prop and for a `port_order` that does not read back as a definite `BondNumber` (`Unknown`, an out-of-range or a negative code), written through `DerefMut`; `n_ports` over two ports on one anchor; `add_bond` round trip reading back `BondType::Single` and `BondNumber::Single`
- [ ] Implement `add_port`, `port`, `ports`, `n_ports` on `Fragment` with the prefixed props `port_kind` / `port_label` / `port_order`, and extract `write_bond_class` into `molrs/src/core/system/bond.rs` (called by `Atomistic::set_bond_class` and `Fragment::add_bond`; `Atomistic`'s existing bond tests stay green)
- [ ] Write failing unit tests for `set_frag_id(atom, id)` / `frag_id(atom)` (per-atom round trip, distinct values on two atoms, `> i32::MAX` rejected) and for `inherit_frag_ids` (a degree-1 unlabelled H with one labelled neighbour inherits; an unlabelled atom with no labelled neighbour stays `None`; an already-labelled atom is unchanged; the return value counts the labelled atoms; one pass, not a fixpoint); then implement all three
- [ ] Write failing unit tests for `Fragment::to_frame` (emits `atoms`, `bonds` and `ports` blocks; `frag_id` column present only when every atom is labelled; a zero-port fragment's frame has no `ports` block) and `Fragment::from_frame` (`from_frame(&f.to_frame())` preserves ports, bonds and `frag_id`; a frame with no `atoms` block is rejected; a frame whose `atoms` block lacks `element` is rejected); then implement both delegating to `MolGraph::to_frame` / `read_frame` with no block relabeling
- [ ] Write failing kind-resolution tests in `molrs/src/core/system/atomistic.rs` and `molrs/src/core/system/coarsegrain.rs` — a `MolGraph` whose first registered kind is `ports` **with one populated port relation** promotes with `n_bonds() == 0`, a subsequent `add_bond` leaves the ports count unchanged, and a graph registering `bonds` with arity 3 yields `MolRsError::validation`, not a panic; then fix both `try_from_molgraph` (by-name lookup + arity check, then idempotent `register_kind`)
- [ ] Wire the public surface and complete the deletion sweep: declare and re-export `fragment` in `molrs/src/core/system/mod.rs` and `molrs/src/core/mod.rs`, delete `molrs/src/core/system/mapping.rs` with its `pub mod` line and `CGMapping` / `WeightScheme` re-export, rewrite the `system/mod.rs` module doc, strike "atom-type mapping" from the `core` row at `CLAUDE.md:186`, strike the bond `order` clause of the `coerce_canonical` doc at `molgraph.rs:124`, and add the `//!` doctest (`C–C` plus an `H` handle on the first carbon: 3 atoms, 2 bonds, one `Symmetric` port, per-atom `frag_id`) and rustdoc per `.claude/notes/docs.md` including the reserved open-prop names, the `""` = unnamed label, the all-or-nothing `frag_id` column, the two-enum decision (`PortKind` vs `DescriptorKind`) and the construction-time-only anchor–handle bond check; name the `frag_id`-vs-`res_id`/`mol_id` decision, 01d's direct `frag_id` writer on `Atomistic`, the `read_frame` unregistered-kind skip and the `emit_column` validity-mask drop as deferred items in the implementation summary for `/mol:note` / `/mol:fix`
- [ ] Run full check + test suite: `cargo fmt --check`, `cargo clippy -p molcrafts-molrs --all-targets --features full,filesystem,stream -- -D warnings`, `cargo test -p molcrafts-molrs --lib --features full,filesystem,stream`, `cargo test --doc -p molcrafts-molrs --features full,filesystem,stream`

## Testing strategy

Per `.claude/notes/testing.md`: inline `#[cfg(test)]` modules next to the code, one behaviour per test, hand-written inputs, no `tests/` tree and no `regressions/` tree in this repo. Each test targets a single function or method.

**`molrs/src/core/system/fragment.rs`** — `port_kind_glyphs_round_trip`; `port_kind_from_str_rejects_unknown_glyph`; `new_registers_bonds_and_ports`; `try_from_molgraph_rejects_node_without_element`; `try_from_molgraph_keeps_registered_kind_ids`; `try_from_molgraph_rejects_conflicting_arity` (no panic); `add_bond_round_trips` (relation exists; `bond_type` / `bond_number` read back `Single`); `add_port_rejects_stale_atom`; `add_port_rejects_non_hydrogen_handle`; `add_port_rejects_unbonded_handle`; `add_port_rejects_unknown_order`; `add_port_records_kind_label_and_order`; `port_rejects_missing_or_unmapped_props`; `n_ports_counts_two_ports_on_one_anchor`; `set_frag_id_per_atom_round_trips` (two atoms, two values); `set_frag_id_rejects_above_i32_max`; `inherit_frag_ids_labels_degree_one_neighbours` / `_leaves_orphans_none` / `_does_not_overwrite` / `_returns_count` / `_is_one_pass`; `to_frame_emits_atoms_bonds_and_ports_blocks`; `to_frame_omits_frag_id_when_partially_labelled`; `to_frame_omits_ports_block_when_no_ports`; `from_frame_restores_ports_bonds_and_frag_id`; `from_frame_rejects_frame_without_atoms`; `from_frame_rejects_atoms_without_element`.

**`molrs/src/core/system/bond.rs`** — `write_bond_class_stamps_both_keys`.

**`molrs/src/core/system/atomistic.rs` and `coarsegrain.rs`** — `try_from_molgraph_resolves_bonds_by_name_not_kind_zero` (ports-first graph with a populated port relation: `n_bonds() == 0`, later `add_bond` leaves the ports count unchanged) and `try_from_molgraph_rejects_conflicting_arity`.

**Public-API example.** This repo has no `regressions/` tree; the equivalent is the rustdoc example on public API, run by `cargo test --doc` (`CLAUDE.md` § Testing Rules: `--lib` does not run doctests). The module-level `//!` doc of `fragment.rs` builds `C–C` with `add_atom_xyz` plus an `H` handle with `add_atom_bare`, bonds C–C and C–H (`add_bond` twice), adds one `PortKind::Symmetric` port `(C0, H)` with label `"A"` and order `BondNumber::Single`, sets `frag_id` on every atom, and asserts hard-coded values: `n_atoms() == 3`, `n_bonds() == 2`, `n_ports() == 1`, the port's label/order read back, and `frag_id(atom) == Some(1)`. No third-party software at test time.

**Gate.** `cargo fmt --check`; `cargo clippy -p molcrafts-molrs --all-targets --features full,filesystem,stream -- -D warnings`; `cargo test -p molcrafts-molrs --lib --features full,filesystem,stream`; `cargo test --doc -p molcrafts-molrs --features full,filesystem,stream`.

## Out of scope

- **`remove_port` / `as_molgraph_mut`** — no link names a caller (02b writes ports, 02c reads through `as_molgraph`); each is added by the link that first calls it.
- **Maintaining the anchor–handle bond across `DerefMut`** — `add_port` checks it once; `port()` validates props only; a bond removed through `Deref` leaves a dangling port, stated in the rustdoc.
- **01d's direct `frag_id` stamp on `Atomistic`** — the second, unvalidated writer of the key; named here, reconciled by the schema-vocabulary spec.
- **`induced_subgraph` / `extract_subgraph` / `merge` / `copy` / `ExtractedFragment`** — each is added by the link that first calls it, with the test that pins it. Graph-hash delegates are reachable through `Deref`; a delegating wrapper would be a second public name.
- **Schema declaration of the fragment/port vocabulary** — `frag_id`, `port_kind`, `port_label`, `port_order` and the `ports` block stay out of `SCHEMA_COLUMNS` / `BlockSpec`; the pending schema-vocabulary spec declares them together and reconciles `frag_id` with `res_id`. No file under `molrs/src/core/store/schema/` is touched here.
- **`read_frame` silently skipping unregistered kinds** (`molgraph.rs:1017-1021`) — pre-existing, widened by this spec (an `Atomistic::from_frame` on a `Fragment` frame drops `ports`); found, named, routed via the implementation summary; not changed here.
- **`emit_column` dropping the validity mask** (`molgraph.rs:150-185`): `column_i32` returns `(data, mask)` and the mask is discarded as `_`, so every null cell of any open `Int` column is written as `0`. Pre-existing and not specific to `frag_id`; found, named, routed via the implementation summary to `/mol:fix`. The all-or-nothing `frag_id` column above is this spec's local guard, not the fix.
- **Parsing CGsmiles / fragment strings into a `Fragment`** — 02b (`to_fragment`), which consumes `try_from_molgraph` and `add_port` (it builds handles and their bonds on the `Atomistic` before promotion).
- **Conformer generation for fragments** — 02c, which composes `generate` and `inherit_frag_ids`.
- **Python / WASM bindings (`PyFragment`, `views.Port`)** — 02d.
- **Multi-parent membership** (a `frag_id` list) — excluded by R5.2 for v1.
- **Reader / writer support for a `ports` block in any file format (`io`)**, and any `perceive` treatment of a port as a valence.
