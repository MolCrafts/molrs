---
title: CGsmiles 02b — `CGSmilesIR::to_fragment` (atomistic fragment templates)
slug: cgsmiles-02b-to-fragment
status: approved
created: 2026-09-21
chain: cgsmiles (01a → 01b → 01c → 01d → 01e → 02a → 02b → 02c → 02d → 03)
depends_on: cgsmiles-01a-descriptors, cgsmiles-01c-fragments, cgsmiles-01d-resolve, cgsmiles-02a-fragment-core
---

# CGsmiles 02b — `CGSmilesIR::to_fragment` (atomistic fragment templates)

## Summary

After 01c a `CGSmilesIR` holds the atomistic fragment table — the last `{…}` block's bodies, parsed as descriptor-bearing `SmilesIR` values — and after 01d it can expand the whole string into one molecule. Neither gives a caller the *pieces*: the instance-free graph of a single named fragment with its unsatisfied valences made explicit, which is what a builder needs to place `#PEO` a thousand times without re-parsing, and what 02c gives coordinates and 02d hands to Python. This link adds `CGSmilesIR::to_fragment`, returning one `Fragment` (02a) per definition in `self.fragments.last()`: each body is converted once by 01a's `fragment_to_atomistic`, every bonding descriptor becomes a real capping hydrogen bonded to its anchor, the graph is promoted to `Fragment`, and each `(anchor, handle)` pair is recorded as a port carrying the descriptor's kind, label and bond order. The result is a template and nothing more — no coordinates, no `frag_id`, no hydrogen repletion, no perception, no inter-fragment bonds.

## Domain basis

DOC = cgsmiles.readthedocs.io; REF = github.com/gruenewald-lab/CGsmiles @ 910c9ee; BIG = Lin, T.-S. et al., *BigSMILES: A Structurally-Based Line Notation for Describing Macromolecules*, ACS Cent. Sci. **5**, 1523–1531 (2019), DOI 10.1021/acscentsci.9b00476. Rules are quoted verbatim from the chain's catalogue.

- **R4.1** — A fragment is a molecular graph plus attachment points. In the BigSMILES grammar a repeat unit is a SMILES fragment decorated with bonding descriptors; the fragment graph alone is not a molecule, because each descriptor marks a valence that is satisfied only when the fragment is joined to a neighbour [BIG; DOC fragments].
- **R4.3** — The descriptor vocabulary is closed: `<`/`>` complementary, `$` self-complementary, CGsmiles adds `!` (shared). They map 1:1 onto `PortKind::{Left, Right, Symmetric, Shared}` [BIG; DOC].
- **R4.4** — A descriptor carries a label; `$A` and `$B` are distinct classes [DOC; BIG].
- **R4.5** — A descriptor carries a bond order (single/double/triple/quadruple); the port stores it; nothing in this link forms the bond [DOC; BIG].
- **R4.17** — Unconsumed descriptors are discarded; at the atomistic level the freed valence is filled with hydrogen ("interpreted as additional hydrogen atoms where applicable") [DOC chirality; REF resolve.py:326-327, :431-433]. A port names two atoms: the anchor (keeps its place in the product) and a handle — a real capping hydrogen, which is what the notation makes of an unconsumed descriptor and what a paired descriptor's bond replaces; the port is the arity-2 relation `(anchor, handle)`.
- **R4.20** — Each fragment block's descriptors belong to their own level; never inherited or matched across levels [REF resolve.py:402-426].
- **R5.1** — Atom–instance membership is per atom (`fragid` in the reference) [REF resolve.py:290-300] — not stamped on templates; 01d stamps instances.

Units: none (topology only; a hydrogen handle has no coordinates here; 02c adds Å positions).

## Design

### The method and its one walker

```rust
impl CGSmilesIR {
    pub fn to_fragment(&self) -> Result<BTreeMap<String, Fragment>, SmilesError>;
}
```

A method on the type that owns the data, exactly as 01d's `CGSmilesIR::to_atomistic` is, in a new private module `molrs/src/io/smiles/cgsmiles/to_fragment.rs` declared `mod to_fragment;` from `cgsmiles/mod.rs` beside 01c's `instantiate` / `validate` and 01d's `resolve` / `to_atomistic`. **`io/smiles/mod.rs` is not edited** — verified: an inherent method needs no re-export, `CGSmilesIR` is already re-exported there by 01b, and `Fragment` / `PortKind` reach the caller from `core` via 02a. This link adds **no new free function, no re-export and no error variant** — `to_fragment` is the one new public method, on a type already exported. Its rustdoc says why the unit is a table rather than one value: a CGsmiles string defines its atomistic fragments as one table, and a builder wants the whole library at once.

For every `(name, def)` of `self.fragments.last()`, in `BTreeMap` order:

1. `let FragmentBody::Smiles(ir) = &def.body else { return Err(CgNotExpandable(name)) }` — 01c's positional dispatch guarantees the last table's bodies are `Smiles`, so the `Graph` arm is unreachable through `parse_cgsmiles` and exists for totality, mirroring 01d's use of the same kind on the same shape.
2. `fragment_to_atomistic(ir)?` → `(Atomistic, Vec<(AtomId, BondingDescriptor)>)` in visit order. 01a's `Builder` is the **only** walker over a fragment body and it runs **once per definition**, not per instance: a template is definition-level, so 01d's per-call `FragmentCache` (which amortizes cloning a body across instances) has nothing to amortize here and is not used, not copied and not made shared. A `SmilesError` from this call propagates unchanged — re-wrapping it would hide the inner kind, and 01c's `CgLastBlockNotAtomistic` box is a parse-stage rule, not a conversion-stage one.
3. Handles and their bonds are written **on the `Atomistic`, before promotion** (see below): per `(anchor, desc)` in map order, `let order = port_order(desc, def.span)?`, `let handle = atomistic.add_atom_bare("H")` (`atomistic.rs:134`, infallible — it returns `AtomId`, not `Result`, so there is nothing to map), `atomistic.add_bond(anchor, handle)` with its `Result` mapped to `CgBuild`; the tuple `(anchor, handle, port_kind(desc.kind), &desc.label, order)` is pushed to a local list.
4. `Fragment::try_from_molgraph(atomistic.into_inner())` — the crate's re-wrap-then-side-car conversion shape (`coarsegrain.rs:277-300`). `into_inner` is zero-cost (`atomistic.rs:630`) and neither it nor `try_from_molgraph` renumbers nodes, so every `AtomId` collected in step 3 stays valid through the promotion. A `Fragment::new()` plus atom-by-atom copy is forbidden here: it would drop `is_aromatic`, `h_count`, `formal_charge`, `isotope`, `stereo` and `atom_class`, which the SMILES builder wrote (`smiles/to_atomistic.rs:169-233`) and which 02c and every downstream consumer need. The incoming graph also carries `Atomistic::new`'s `angles` / `dihedrals` / `impropers` kinds; they ride along empty and `to_frame` skips empty kinds (02a), so they cost nothing and are not stripped.
5. `fragment.add_port(anchor, handle, kind, label, order)` per recorded tuple, `Result` mapped to `CgBuild`. 02a's `add_port` re-checks that the handle carries `element == "H"` and is bonded to the anchor; both hold by construction, which is why step 3 must precede step 4.

Ports are added in `fragment_to_atomistic` map order, so for a given definition the *n*-th port corresponds to the *n*-th entry of that map — the same index 01d's `PairEnd::Body.port` uses. This link states that correspondence but does **not** assert `Fragment::ports()` iteration order, which 02a does not promise: the tests assert the port *set* (anchor, kind, label, order) and `n_ports()`.

### Where the handle bond is written — decided, with the 02a consequence

The handle atom and the handle–anchor bond are written on the **`Atomistic`, before promotion**; only ports are written on the `Fragment`. Three reasons, the second decisive:

1. `Atomistic` ships all of `add_atom_bare` (`:134`), `add_bond` (`:178`) and `set_bond_class` (`:206`) today. `Fragment` reaches no `set_bond_class`: through `Deref` to `MolGraph` it has only `set_relation_prop`, so writing a bond class on a `Fragment` means writing raw `bond_type` / `bond_number` props — crossing a newtype's own abstraction to say something `Atomistic` already names.
2. **Verified in tree, and it contradicts the librarian report**: `Atomistic::add_bond` is not a bare `add_relation` — it calls `set_bond_type(bid, BondType::Single)` (`atomistic.rs:178-182`), which calls `set_bond_class(bid, Single, BondType::Single.implied_number() == Some(BondNumber::Single))` (`:221-224`, `bond.rs:99-106`). One `Atomistic::add_bond` therefore already writes **both** facts, `(BondType::Single, BondNumber::Single)`, which is exactly the bond class this spec requires on every handle bond. 02a's `Fragment::add_bond` is specified as delegating to `add_relation(self.bond, &[a, b])` only, so the same call on a `Fragment` would leave **both** props unset — precisely the silent-`Unknown` failure the requirement exists to prevent.
3. The split falls on the seam of the crate's conversion pattern: handles and bonds are *graph*, ports are *side-car*.

**Deviation from the request, stated so it can be rejected.** The request asks for an explicit `set_bond_class(bid, BondType::Single, BondNumber::Single)` after each handle bond, on the premise that "the sibling `add_hydrogens` sets only the type". That premise does not hold: `perceive/hydrogens.rs:108-110` calls `Atomistic::add_bond` (which writes both) and then re-writes `set_bond_type(bid, Single)` (which writes both again) behind a `let _ =`. `bond_number` is **not** left `Unknown` there, and `valence_demand`'s `.count().max(1)` (`hydrogens.rs:381`) is reading a real `Single`, not surviving on its fallback. Copying an explicit `set_bond_class` into `to_fragment` would reproduce that redundant second write, not avoid it. This spec therefore makes **no** class call — `add_bond` is the single writer — and pins the outcome by test instead: every handle bond has `BondType::Single` **and** `BondNumber::Single`. The real rot in `hydrogens.rs` (redundant re-write, discarded `Result`, `mass` literal) is named in Out of scope and routed; that file is not touched here.

**Consequence for 02a, applied by the orchestrator at staging.** 02a's shipped-surface table used to justify `Fragment::add_bond` and `add_atom_bare` by 02b; after this decision 02b calls neither. Their production callers are 02d's typed `PyFragment.add_atom` / `add_bond` wrappers — the same pair `PyAtomistic` (`molrs-python/src/core/system/molgraph.rs:656,665`) and `PyCoarseGrain` (`:1439`) already expose over their Rust leaves, and the door through which molpy's hand-built fragments get atoms and bonds — plus 02a's own doctest; 02a's table and its Out-of-scope cross-reference now say so. The finding that matters: 02a's `Fragment::add_bond` stamps the default class `(BondType::Single, BondNumber::Single)` through a writer shared with `Atomistic::set_bond_class`, so a hand-built template carries the same two bond facts a 02b template gets from the `Atomistic` side. 02b edits no other link's spec and no file under `core/`.

### Mappings, and where they live

`fn port_kind(kind: DescriptorKind) -> PortKind` and `fn port_order(desc: &BondingDescriptor, span: Span) -> Result<BondNumber, SmilesError>` are **private module-level functions in `to_fragment.rs`**, not an `impl From<DescriptorKind> for PortKind`. Both types are crate-local so the orphan rule permits the `From` impl in `io` (it cannot live in `core`: `PortKind` is always-on, `DescriptorKind` is behind `smiles`), but a trait impl is permanently public surface, and this conversion has exactly one call site with no second user named by any link — 02d binds `PortKind` to Python and never sees `DescriptorKind`. CLAUDE.md's Shape check #3 says do not extract for one call site; the same page's "Inline until the second real use" grants the exception these two take: *a unit test must target that unit*. `port_kind`'s four arms (including the `Shared` totality arm) and `port_order`'s guard are the two places this link can be wrong in a way the end-to-end counts would hide, so each is a private function with its own test. Promote to `From` when a second caller exists.

`port_kind` is total and 1:1 per R4.3: `Symmetric→Symmetric`, `Left→Left`, `Right→Right`, `Shared→Shared`. `Shared` cannot arrive through `parse_cgsmiles` (01c raises `CgSquashUnsupported` in `validate_ir` at any level); the arm exists so the match is exhaustive without a `_` catch-all.

`port_order` implements R4.5: `None → BondNumber::Single`; `Some(k) → k.bond_number()` (01d moves the mapping onto `BondKind` as `pub(crate) fn bond_number`), guarded — a result of `BondNumber::Unknown` (what `Aromatic`, `Up`, `Down`, `Any` and `Ring` map to) returns 01a's `InvalidDescriptorOrder(kind)`. **`Unknown` is never handed to `add_port`** — 02a's `add_port` rejects it as well, but the guard here names the written `BondKind` that caused it (01a's `InvalidDescriptorOrder(BondKind)` payload) instead of a bare validation error; 02a's port stores the `BondNumber` itself, so no integer conversion exists on this path. The guard is not redundant with 01a's parse-time `validate_descriptor`: `BondingDescriptor` has public fields and 01a states that `fragment_to_atomistic` does not re-validate, so a hand-built IR can carry `Some(Aromatic)` — whose `bond_number()` is `BondNumber::Unknown`. This is the same reason 01a's `write_fragment_smiles` re-checks. `label` passes through verbatim, `""` meaning unnamed per 02a and R4.4.

### Errors: reuse only, and the input the IR does not have

Every `MolRsError` from `add_bond`, `try_from_molgraph` or `add_port` becomes 01c's `CgBuild(String)` — the reader's internal-invariant variant — spanned at `def.span`, with a message naming the fragment and the underlying error. Every error this method builds uses 01b's four-argument constructor, `SmilesError::new(kind, span, input, Notation::CGsmiles)`: `notation` is a required argument with no default precisely so a CGsmiles error can never render as "SMILES parse error" (01b), and the `smiles/to_atomistic.rs` sites this file otherwise imitates stamp `Notation::Smiles`, which must not be copied. Never `InvalidElement`, which is the lossy mislabel 01d routes to `/mol:fix` (`smiles/to_atomistic.rs:241-247`). **No fallible call is discarded**: there is no `let _ =` anywhere in the new file, and a private `fn cg_build(def: &CGFragmentDef, e: MolRsError) -> SmilesError` is the one construction site (three callers, so the extraction is past its second use).

A base-only IR (`self.fragments.is_empty()`) is an **error**, not an empty map: `CgNotExpandable` spanned at `self.span`. Its `String` payload is a fragment name at 01d's call site (a `Graph` body at the last level) and the phrase `base-only string (no fragment table)` here, so the one `Display` arm both links share must read correctly for either: **`{0}: no atomistic body to expand`** — "PEO: no atomistic body to expand" / "base-only string (no fragment table): no atomistic body to expand". That wording is pinned in 01d's error section and asserted on the rendered text here. Silence is wrong here by the operator rule *raise, never silently drop* — an empty `BTreeMap` is indistinguishable from "this string defines no fragments" and from "a table was there and I dropped it", and a builder that asks for templates and gets zero has no way to tell a base-only string from a bug. Reusing 01d's kind (rather than adding a variant) keeps the two "this IR has no atomistic level" answers on one name.

**Found, named, routed: `SmilesError` needs an `input: &str` that `CGSmilesIR` does not carry.** 01b froze the IR as `{ levels, span }` (plus 01c's `fragments`, 01d's `pairs`); there is no source text on it, so `to_fragment(&self)`, like 01d's `to_atomistic(&self)`, must pass `""` as the `input` of `SmilesError::new(kind, span, input, Notation::CGsmiles)`. Verified this cannot panic: `Display` guards the caret with `pos.min(self.input.len())` (`error.rs:89-95`), so the message degrades to "position N" with an empty context line. Cost is bounded because two of the three kinds this method can raise are unreachable from `parse_cgsmiles` (`CgBuild` is a reader bug by 01c's definition; `InvalidDescriptorOrder` is refused at parse by 01a's `validate_descriptor`); only the base-only `CgNotExpandable` is user-reachable, and its message carries the whole diagnosis. The fix — carrying the source on `CGSmilesIR`, or a `Display` that says so when it has no input — belongs to whichever link owns the IR shape, affects 01d identically, and is routed via the implementation summary to `/mol:note` / `/mol:fix`. It is not worked around here.

This link also relies on — and must not work around — 02a's fix of `try_from_molgraph`'s `unwrap_or(KindId(0))` kind aliasing. The graph handed over registers `bonds` first, so even the unfixed code would accidentally resolve correctly; nothing in `to_fragment` reorders registrations or re-registers a kind to make that so.

### Hydrogen-valence semantics: why two opposite paths are both right

Verified against `perceive/hydrogens.rs:49-115, 276-341, 375-384` and `smiles/to_atomistic.rs:169-233`:

- `implicit_h_count` short-circuits on a declared `h_count` (`:281-283`); otherwise `valence_demand` (`:375-384`) sums the atom's **real incident bonds**.
- For an **organic-subset anchor** (`[$]COC[$]`) the SMILES builder writes no `h_count` (`:171-177`), so the handle is counted as a bond: C0 has O + handle = 2, and a later `add_hydrogens` adds exactly the remaining 2 → coordination 4. The handle is *subtracted* from the repletion budget, correctly.
- For a **bracket anchor** (`[$][CH3]`, `{#OHter=[$][O-]}`) the builder always writes `h_count` (`:211`) and `implicit_h_count` is then bond-blind, so the handle is *additive* — which is also the notation's meaning: the descriptor is an extra valence beyond the declared hydrogens.

Both are correct for opposite reasons, and the single rule that keeps them so is: **`to_fragment` writes no `h_count`, no `formal_charge` and no `is_aromatic` on anchor or handle** — it adds an atom and a bond and touches nothing the builder wrote. Any "adjust the anchor's `h_count` by one" is therefore wrong on both paths and is not done. `add_atom_bare` writes `element` only — no `mass` literal (the `1.008` at `hydrogens.rs:101` is rot where `Element::atomic_mass()` exists), no coordinates, no `h_count`.

**A handle is distinguishable from a repletion hydrogen only through the `ports` relation.** `remove_hydrogens` (`hydrogens.rs:246-266`) removes *every* degree-1 H, handles included; `add_hydrogens` skips existing H but adds none to them. That is the reason 02a's port is arity-2 `(anchor, handle)` rather than anchor-only, and 02c relies on it when it relabels pipeline hydrogens. Stated in the `to_fragment` rustdoc so a caller does not round-trip a template through `remove_hydrogens` and wonder where its ports went.

### What a template is not

No coordinates on any atom. No `frag_id` on any atom (R5.1: membership is per atom of an *instance*; 01d's `to_atomistic` stamps it, a template has no instance). No hydrogen repletion (R4.17 is the caller's `add_hydrogens` step, demonstrated in the doctest). No perception, no kekulization, no inter-fragment bonds, no consultation of `self.pairs` or `self.levels`. Keys of the returned map are exactly the key set of `fragments.last()` — including definitions the level above never references, which 01c deliberately keeps: by R4.1 a fragment is a graph plus attachment points, a fact independent of use.

### Reuse decision

- `Atomistic::into_inner` (`atomistic.rs:630`) — **reuse**: the only sanctioned way out of the newtype; pairs with `Fragment::try_from_molgraph`.
- `CoarseGrain::induced_subgraph` (`coarsegrain.rs:277-300`) — **pattern**: re-wrap the graph, then write side-car data in a second pass. Steps 3–5 are exactly that; no `X::new()` + atom-by-atom copy. Note carried: `try_from_molgraph` registers the `ports` *kind*, it registers no ports — every port is written by `add_port`.
- `add_hydrogens` (`hydrogens.rs:49-115`) — **pattern, not reuse**: it is the in-tree precedent for creating an explicit H bonded to a heavy atom (element-only on the coordinate-free path, `:93-95`, never touching the anchor), and `to_fragment` follows that shape by hand. It is **not called** by `to_fragment`; it appears only as the downstream consumer in the doctest.
- X–H bond write (`hydrogens.rs:108-110`) — **neither reuse nor generalize; the librarian's reading is corrected**. `Atomistic::add_bond` already writes `(Single, Single)`; the `set_bond_type` on `:109` is a redundant second write behind `let _ =`. `to_fragment` calls `add_bond` and stops. The rot is named in Out of scope and routed to `/mol:fix`; `hydrogens.rs` is not edited by this link.
- `h.set("mass", 1.008_f64)` (`hydrogens.rs:101`) — **rot, do not copy**: `add_atom_bare` writes element only.
- `BondKind::bond_number` (01d's home for the former `bond_kind_to_number`) — **reuse** as the `Option<BondKind>` → `BondNumber` mapping, wrapped by `port_order`'s guard; no second order table. `BondNumber::count()` (`bond.rs:142`) is **not** needed: 02a's port stores the `BondNumber` itself. The `Aromatic → Unknown` trap is the guard's reason for existing.
- `BTreeMap<String, _>` name tables (`io/data/frcmod.rs:18`, `io/zarr/sequence.rs:700`) — **pattern**: settled container and deterministic iteration; the key-set invariant is a one-line test.
- `h_count` seam (`to_atomistic.rs:169-233`; `hydrogens.rs:276-341`) — **reuse as a constraint**: no decrement is needed and none must be written; see the hydrogen-valence section.
- Placement — **accepted as advised**: new `io/smiles/cgsmiles/to_fragment.rs`, method on `CGSmilesIR`; `core/system/fragment.rs` rejected (core importing a SMILES IR inverts the spine), `cgsmiles/ast.rs` rejected (01c keeps it a data module). Scope is `io::smiles::cgsmiles`; `io` depends on `core` + `perceive` only, which this respects.
- Closest pattern — **followed**: `smiles/to_atomistic.rs` error/builder style (`SmilesError::new(kind, span, input, notation)` with `Notation::CGsmiles` here, fallible calls propagated), `to_<target>` naming, `# Errors` rustdoc on every `Result`-returning item. No `Builder` struct is introduced: the only walker is 01a's, entered once per definition.
- `DescriptorKind → PortKind` as a `From` impl — **new, rejected in that form**: private fn, for the reason above.

## Files to create or modify

- `molrs/src/io/smiles/cgsmiles/to_fragment.rs` (new) — `impl CGSmilesIR { pub fn to_fragment }`, private `port_kind` / `port_order` / `cg_build`, the rustdoc doctest, inline `#[cfg(test)]` tests.
- `molrs/src/io/smiles/cgsmiles/mod.rs` (created by 01b, extended by 01c/01d; unmerged in the working tree) — add `mod to_fragment;` beside `mod resolve; mod to_atomistic;`, and one rustdoc sentence placing templates (02b) next to expansion (01d).

## Tasks

- [ ] Write failing unit tests in `molrs/src/io/smiles/cgsmiles/to_fragment.rs` for the private mappings — `port_kind` over all four `DescriptorKind` variants including `Shared`; `port_order` for `None → Single`, `Some(Double) → Double`, `Some(Triple) → Triple`, `Some(Quadruple) → Quadruple`, `Some(Aromatic) → InvalidDescriptorOrder` and never `Unknown` — then implement `port_kind` and `port_order` in that new file and declare `mod to_fragment;` in `molrs/src/io/smiles/cgsmiles/mod.rs`
- [ ] Write failing unit tests in `molrs/src/io/smiles/cgsmiles/to_fragment.rs` for template shape — F2 `{[#OH][#PEO]|3[#OH]}.{#OH=[$]O,#PEO=[$]COC[$]}` returning keys exactly `{OH, PEO}` with PEO 5 atoms / 4 bonds / 2 `Symmetric` ports (anchors C0 and C2, label `""`, order `BondNumber::Single`) and OH 2 atoms / 1 bond / 1 port; F5 `#BB=[>]CC[<][$]` → 5 atoms, `n_ports() == 3`, two ports on C1 (`Left`, `Symmetric`); F6 `#GLY=[>]NCC(=O)[<]` → 6 atoms, `Right` on N and `Left` on the carbonyl C; labels `a`/`b` preserved for `Clc[$a]c[$b]`; an unreferenced definition still yielding a template — then implement `CGSmilesIR::to_fragment` in that file
- [ ] Write failing unit tests in `molrs/src/io/smiles/cgsmiles/to_fragment.rs` for the property discipline — a handle carries `element` only (no `mass`, no `x`/`y`/`z`, no `h_count`, no `is_aromatic`, no `frag_id`); an organic anchor gains no `h_count` and no `formal_charge`; a bracket body `[$][13CH3-]` keeps `isotope == 13`, `h_count == 3`, `formal_charge == -1` and an aromatic body keeps `is_aromatic` (proving promotion re-wraps rather than copies); every handle bond reads back `BondType::Single` **and** `BondNumber::Single`; no atom carries a position — then make them pass without adding a `set_bond_class` call
- [ ] Write failing unit tests in `molrs/src/io/smiles/cgsmiles/to_fragment.rs` for the error paths — a base-only IR (`{[#EO]|5}`) → `CgNotExpandable` carrying the payload `base-only string (no fragment table)`; a hand-built IR whose last table holds a `FragmentBody::Graph` → `CgNotExpandable`; a hand-built descriptor with `order: Some(BondKind::Aromatic)` → `InvalidDescriptorOrder` spanned at the definition; every kind asserted by `matches!` on `err.kind`, never by message substring — then implement the `cg_build` mapping (`CgBuild` at `CGFragmentDef.span`, never `InvalidElement`) with no `let _ =` on any fallible call
- [ ] Add rustdoc per `.claude/notes/docs.md` to `molrs/src/io/smiles/cgsmiles/to_fragment.rs` and `molrs/src/io/smiles/cgsmiles/mod.rs` — `# Errors` on `to_fragment`, the handle-is-a-capping-hydrogen meaning (R4.17), the no-`h_count`/`formal_charge`/`is_aromatic` rule with both valence paths, the `remove_hydrogens` hazard, the port-order correspondence with 01d's `PairEnd::Body.port`, and the template-is-instance-free rule (no `frag_id`, no coordinates) — plus the doctest on `CGSmilesIR::to_fragment` (parse F2, `to_fragment()`, assert PEO 5 atoms / 4 bonds / 2 ports, then `Atomistic::try_from_molgraph(into_inner)` + `add_hydrogens` → 9 atoms with `n_ports() == 2`), and name in the implementation summary the deferred items: the `hydrogens.rs:101,108-110` rot and the missing source text on `CGSmilesIR` (shared with 01d)
- [ ] Run the gate: `cargo fmt --check`, `cargo clippy -p molcrafts-molrs --all-targets --features full,filesystem,stream -- -D warnings`, `cargo test -p molcrafts-molrs --lib --features full,filesystem,stream`, `cargo test --doc -p molcrafts-molrs --features full,filesystem,stream`

## Testing strategy

Per CLAUDE.md § Testing Rules and `.claude/notes/testing.md`: unit tests are inline `#[cfg(test)]` modules next to the code — all of this link's tests live in `molrs/src/io/smiles/cgsmiles/to_fragment.rs`, the file whose functions they target. There is no `molrs/tests/` tree and no `regressions/` tree in this repo; the public-API example is the rustdoc doctest, which `--lib` does not run. Every expected value below is hand-derived from the rules in Domain basis and written as a literal; no third-party chemistry software is involved at any point, at test time or as a value source. Each test targets one function: `port_kind`, `port_order`, or `CGSmilesIR::to_fragment`.

**`port_kind`** — `map_each_descriptor_kind_to_its_port_kind`: all four arms, `Shared` included (R4.3 is 1:1).

**`port_order`** — `defaults_to_single_when_no_order_written` (`None → BondNumber::Single`, R4.5); `reads_the_written_bond_number` (`Double`, `Triple`, `Quadruple`); `rejects_an_aromatic_descriptor_order` (`Some(Aromatic)` → `InvalidDescriptorOrder`, and the returned value is an `Err`, never `Ok(BondNumber::Unknown)`).

**`CGSmilesIR::to_fragment` — happy path, hand-derived counts (heavy atoms + handles).**
- `returns_one_template_per_definition`: F2 → keys exactly `{OH, PEO}`, equal to `fragments.last()`'s key set.
- `builds_the_peo_template`: PEO `[$]COC[$]` → 5 atoms, 4 bonds, 2 ports; anchors C0 and C2, both `PortKind::Symmetric`, label `""`, order `BondNumber::Single`.
- `builds_the_hydroxyl_template`: OH `[$]O` → 2 atoms, 1 bond, 1 port.
- `puts_two_ports_on_one_anchor`: F5 `#BB=[>]CC[<][$]` → 2 heavy + 3 handles = 5 atoms, 4 bonds, `n_ports() == 3`; C0 `Right`; C1 carries both `Left` and `Symmetric`.
- `anchors_glycine_ports`: F6 `#GLY=[>]NCC(=O)[<]` → 4 heavy + 2 handles = 6 atoms, 5 bonds; `Right` on N, `Left` on the carbonyl C (the branch does not advance the anchor, 01a's rule).
- `keeps_descriptor_labels`: `Clc[$a]c[$b]` → port labels `"a"` and `"b"` (R4.4).
- `keeps_a_definition_the_level_above_never_references`.

**`CGSmilesIR::to_fragment` — property discipline (the scientific commitment).**
- `writes_only_the_element_on_a_handle`: no `mass`, no `x`/`y`/`z`, no `h_count`, no `is_aromatic`, no `formal_charge`, no `frag_id`.
- `leaves_an_organic_anchor_untouched`: C0 of `[$]COC[$]` has no `h_count` and no `formal_charge` — the invariant that makes a later `add_hydrogens` add exactly 2 H to it (`hydrogens.rs:375-384` sums the handle bond).
- `preserves_bracket_atom_properties`: `[$][13CH3-]` → anchor keeps `isotope == 13`, `h_count == 3`, `formal_charge == -1`; an aromatic body keeps `is_aromatic` — the test that fails if promotion ever becomes `Fragment::new()` + copy.
- `sets_single_class_on_every_handle_bond`: `BondType::Single` and `BondNumber::Single` on every handle bond of the PEO template.
- `adds_no_coordinates_and_no_frag_id`.

**`CGSmilesIR::to_fragment` — errors**, each asserted by `matches!` on `err.kind` (including the payload string where one is pinned; rendered wording belongs to 01d's `Display` test in `error.rs`): `rejects_a_base_only_ir` (`{[#EO]|5}` → `CgNotExpandable` whose payload is `base-only string (no fragment table)`; the rendered wording `{0}: no atomistic body to expand` is asserted by 01d's `Display` test in `error.rs`, where the arm lives); `rejects_a_graph_body_in_the_last_table` (hand-built IR → `CgNotExpandable`); `rejects_an_invalid_descriptor_order` (hand-built IR carrying `Some(Aromatic)` → `InvalidDescriptorOrder`, span = the definition's).

**Public-API example (this repo's regression-example equivalent; CLAUDE.md makes the rustdoc doctest the public-API gate).** On `CGSmilesIR::to_fragment`: parse F2 with `parse_cgsmiles`, call `to_fragment()`, assert PEO has `n_atoms() == 5`, `n_bonds() == 4`, `n_ports() == 2`; then `Atomistic::try_from_molgraph(peo.into_inner())` + `add_hydrogens` and assert `n_atoms() == 9` (3 heavy + 2 handles + 2 + 2 repletion H, hand-derived: C0 = O + handle = 2 bonds → 2 H, C2 likewise, O already at valence 2) with the ports still 2 and every port handle still degree 1 after re-wrapping to `Fragment`. Public API only, hard-coded values, nothing imported or shelled out at test time. Runs under `cargo test --doc -p molcrafts-molrs --features full,filesystem,stream`.

**Gate.** `cargo fmt --check`; `cargo clippy -p molcrafts-molrs --all-targets --features full,filesystem,stream -- -D warnings`; `cargo test -p molcrafts-molrs --lib --features full,filesystem,stream`; `cargo test --doc -p molcrafts-molrs --features full,filesystem,stream`.

## Out of scope

- **Coordinates for a `Fragment`, and relabelling pipeline-added hydrogens** — 02c, which composes `generate` and `inherit_frag_ids` and which relies on this link's `(anchor, handle)` ports to tell a handle from a repletion hydrogen.
- **Python / WASM exposure of `to_fragment`** (`dict[str, Fragment]`, `views.Port`) — 02d; the molpy builder consumes templates through it.
- **Any change under `molrs/src/core/`** — 02a owns `Fragment`, `PortKind`, `add_port`, `try_from_molgraph` and the `KindId(0)` aliasing fix; this link is a caller and adds no workaround. 02a's surface table already cites 02d's typed `PyFragment.add_atom` / `add_bond` wrappers for `add_atom_bare` / `add_bond` (amendment applied at staging).
- **Any change under `molrs/src/perceive/`.** Named rot in `perceive/hydrogens.rs`, found while verifying the valence semantics, **not** fixed here because this link does not touch the file: `:108-110` re-writes the bond class `Atomistic::add_bond` already wrote and discards the `Result` with `let _ =`; `:101` writes the literal `1.008` where `Element::atomic_mass()` exists (`element.rs:988`); `:263` discards `remove_atom`'s `Result`. Named in the implementation summary and routed to `/mol:fix`.
- **`CGSmilesIR` carrying its source text** — `SmilesError::new` needs an `input: &str` the IR does not have, so `to_fragment` passes `""` exactly as 01d's `to_atomistic` does (verified non-panicking, `error.rs:89-95`). Found, named, routed; a predecessor-shape change affecting 01d equally, not a 02b workaround.
- **Hydrogen repletion, perception, kekulization inside `to_fragment`** — R4.17 says the freed valence is filled with hydrogen; the caller composes `add_hydrogens`, as the doctest shows.
- **Instance membership (`frag_id`) and inter-fragment bonds on templates** — R5.1 and 01d's `to_atomistic`; a template is instance-free by definition.
- **A query for unpaired ports, or reading `self.pairs` / `self.levels`** — no in-tree caller; 01d already left that door closed.
- **Writing a `Fragment` back to CGsmiles text, and any reader/writer support for a `ports` block in a file format** — no link in this chain covers it.
- **`[!]` / squash support** — 01c raises `CgSquashUnsupported`; `PortKind::Shared` is only representable here.
