
---
spec: cgsmiles-02b-to-fragment
created: 2026-09-21
criteria:
  - id: ac-001
    summary: to_fragment returns exactly one template per atomistic definition
    type: code
    pass_when: |
      For F2 `{[#OH][#PEO]|3[#OH]}.{#OH=[$]O,#PEO=[$]COC[$]}`, the keys of
      `CGSmilesIR::to_fragment()` equal the key set of `ir.fragments.last()`
      (`{OH, PEO}`), including a definition the level above never references,
      asserted by a unit test in
      molrs/src/io/smiles/cgsmiles/to_fragment.rs.
    status: pending
  - id: ac-002
    summary: Template shape matches the hand-derived fixtures
    type: code
    pass_when: |
      Unit tests assert PEO = 5 atoms / 4 bonds / 2 ports (anchors C0 and C2),
      OH = 2 atoms / 1 bond / 1 port, `#BB=[>]CC[<][$]` = 5 atoms / 4 bonds /
      `n_ports() == 3` with two ports on C1, and `#GLY=[>]NCC(=O)[<]` = 6 atoms
      / 5 bonds with ports on N and the carbonyl C.
    status: pending
  - id: ac-003
    summary: Descriptor kind and label map 1:1 onto the port (R4.3, R4.4)
    type: scientific
    pass_when: |
      `port_kind` maps Symmetric/Left/Right/Shared onto
      PortKind::Symmetric/Left/Right/Shared with no catch-all arm, and
      `Clc[$a]c[$b]` yields ports labelled "a" and "b" while a bare `[$]`
      yields label "".
    status: pending
  - id: ac-004
    summary: Descriptor bond order becomes the port order, never zero (R4.5)
    type: scientific
    pass_when: |
      `port_order` returns `BondNumber::Single` for `None`, `Double`/`Triple`/
      `Quadruple` for the written orders, and `Err(InvalidDescriptorOrder)` for
      Some(Aromatic) or any kind whose `bond_number()` is `Unknown` — no input
      produces `Ok(BondNumber::Unknown)`.
    status: pending
  - id: ac-005
    summary: to_fragment writes no valence-bearing property on anchor or handle
    type: scientific
    pass_when: |
      Unit tests show a handle carries `element` only (no mass, x/y/z, h_count,
      is_aromatic, formal_charge, frag_id), an organic anchor gains no h_count
      and no formal_charge, a bracket body `[$][13CH3-]` keeps
      isotope == 13, h_count == 3, formal_charge == -1 after promotion, and no
      atom of any returned template carries `x`/`y`/`z` or `frag_id`.
    status: pending
  - id: ac-006
    summary: Every handle-anchor bond carries both bond facts
    type: code
    pass_when: |
      For the PEO template, every handle bond reads back
      `BondType::Single` and `BondNumber::Single` (the test); the absence of a
      `set_bond_class` call in molrs/src/io/smiles/cgsmiles/to_fragment.rs is
      verified by reading the diff, not by a test.
    status: pending
  - id: ac-007
    summary: Error discipline — reused kinds, spanned, nothing discarded
    type: code
    pass_when: |
      A base-only IR and a `FragmentBody::Graph` in the last table both yield
      `CgNotExpandable`, and the base-only case carries the payload
      `base-only string (no fragment table)` (which 01d's `Display` arm renders
      as `{0}: no atomistic body to expand`, asserted in 01d's error.rs test); a
      hand-built `Some(Aromatic)` descriptor yields `InvalidDescriptorOrder`;
      every MolRsError becomes `CgBuild` spanned at `CGFragmentDef.span`; every
      error the method returns carries `notation == Notation::CGsmiles`
      (asserted on the base-only case); the absence of `let _ =` and of
      `InvalidElement` in `molrs/src/io/smiles/cgsmiles/to_fragment.rs` and the
      absence of a new `SmilesErrorKind` variant are verified by reading the
      diff, not by a test.
    status: pending
  - id: ac-008
    summary: Public-API doctest reproduces the template and its repletion
    type: runtime
    pass_when: |
      `cargo test --doc -p molcrafts-molrs --features full,filesystem,stream`
      passes, and the doctest on `CGSmilesIR::to_fragment` asserts the
      hard-coded values: PEO template `n_atoms() == 5`, `n_bonds() == 4`,
      `n_ports() == 2`, and after `Atomistic::try_from_molgraph(into_inner)` +
      `add_hydrogens`, `n_atoms() == 9` with `n_ports() == 2` and every port
      handle still degree 1. No third-party software at test time.
    status: pending
  - id: ac-009
    summary: Rustdoc states the handle contract and the port-order rule
    type: docs
    pass_when: |
      `CGSmilesIR::to_fragment` carries an `# Errors` section and documents:
      the handle as a capping hydrogen (R4.17), the no-h_count/formal_charge/
      is_aromatic rule with both valence paths, that `remove_hydrogens` strips
      handles, the port-order correspondence with 01d's `PairEnd::Body.port`,
      and that templates carry no `frag_id` and no coordinates.
    status: pending
  - id: ac-010
    summary: Full gate green with no scope leakage
    type: code
    pass_when: |
      `cargo fmt --check`, `cargo clippy -p molcrafts-molrs --all-targets
      --features full,filesystem,stream -- -D warnings` and `cargo test -p
      molcrafts-molrs --lib --features full,filesystem,stream` all pass, and
      the diff touches only
      molrs/src/io/smiles/cgsmiles/to_fragment.rs (new) and
      molrs/src/io/smiles/cgsmiles/mod.rs, and to_fragment.rs names neither
      `self.pairs`, `self.levels` nor `FragmentCache` (diff review).
    status: pending
out_of_scope:
  - Coordinates for a Fragment and hydrogen relabelling (02c)
  - Python / WASM binding of to_fragment (02d)
  - Any edit under molrs/src/core/ (02a) or molrs/src/perceive/
  - Hydrogen repletion, perception or kekulization inside to_fragment
  - frag_id stamping and inter-fragment bonds (01d's to_atomistic)
  - Writing a Fragment back to CGsmiles text
---

# Acceptance criteria

"Done" for this link means a caller can take a parsed CGsmiles string and get
back, per atomistic fragment definition, a `Fragment` that is a faithful
template: the body's own atoms with every property the SMILES builder wrote
(ac-005), one capping hydrogen per bonding descriptor bonded with a fully
specified single bond (ac-002, ac-006), and one port per descriptor carrying
the kind, label and order the notation wrote (ac-003, ac-004). Nothing is
invented and nothing is dropped: no coordinates, no `frag_id`, no repletion
hydrogens, and no silent empty map when the string has no atomistic table
(ac-007).

The binding demonstration is ac-008 — the rustdoc doctest is this repo's
regression example, and it proves the one claim that unit counts alone cannot:
that a template's handles are counted as real bonds, so a downstream
`add_hydrogens` completes the anchors to exactly the right coordination.

ac-009 and ac-010 keep the link honest about its own boundaries: the semantics
that make ac-005 load-bearing are written down where the next reader will find
them, and the diff stays inside `io::smiles::cgsmiles` — the rot found in
`perceive/hydrogens.rs` and the missing source text on `CGSmilesIR` are
reported and routed, not patched around.
