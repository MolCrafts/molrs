---
spec: cgsmiles-01d-resolve
created: 2026-09-21
criteria:
  - id: ac-001
    summary: "One written-aromatic predicate on the shared AST vocabulary"
    type: code
    pass_when: "`AtomSpec::written_aromatic` exists in molrs/src/io/smiles/chem/ast.rs; `Builder::add_atom_node` in smiles/to_atomistic.rs reaches its aromatic decision only through that call (no remaining `aromatic: true` literal test in that function); a unit test covers Organic, Bracket-Element, BracketSymbol::Aromatic, Wildcard and Query inputs; resolve.rs reads the `is_aromatic` stamp on the cached fragment Atomistic and contains no second aromaticity rule."
    status: verified
    verified_by: impl-gate
    last_checked: 2026-09-21
    note: "chem/ast.rs:168 pub(crate) written_aromatic; add_atom_node has no `aromatic: true` literal; tests cover Organic, Bracket-Element, BracketSymbol::Aromatic, Wildcard, Query; resolve.rs:292 reads the is_aromatic stamp only"
  - id: ac-002
    summary: "BondKind owns its bond_type / bond_number mapping"
    type: code
    pass_when: "`BondKind::bond_type` and `BondKind::bond_number` exist in chem/ast.rs, the free `bond_kind_to_type` / `bond_kind_to_number` are deleted from smiles/to_atomistic.rs, the former call site uses the methods, a test pins the whole table incl. Aromatic -> (Aromatic, Unknown), and the `Quadruple -> BondType::Double` approximation is stated in the method rustdoc."
    status: verified
    verified_by: impl-gate
    last_checked: 2026-09-21
    note: "chem/ast.rs:261/282 bond_type/bond_number with the Quadruple → Double rustdoc; free fns deleted; caller smiles/to_atomistic.rs:356; whole-table tests incl. Aromatic → (Aromatic, Unknown)"
  - id: ac-003
    summary: "Edge provenance is a tag on one edge list"
    type: code
    pass_when: "`EdgeOrigin { Written, Derived { level, pair } }` and `CGEdge.origin` exist in cgsmiles/ast.rs; a test on a parsed multi-level IR asserts every edge produced by parser.rs and instantiate.rs is `Written` before resolution; fixture F8 asserts `levels[1].edges.len() == 5` with the last two edges `Derived { level: 0, pair: 0 }` and `Derived { level: 0, pair: 1 }` in that order, each `CGBondOrder::Single` with `span` equal to the inducing level-0 edge's span, and every `ResolvedPair.edge` in `pairs[1]` indexing that final list."
    status: verified
    verified_by: impl-gate
    last_checked: 2026-09-21
    note: "ast.rs:181 EdgeOrigin, :341 CGEdge.origin; resolve.rs:788 all Written before resolution; :836/:858/:870/:881 F8 five edges, Derived{0,0}/{0,1} last, Single, inducing span, pairs[1] indexes the final list"
  - id: ac-004
    summary: "PairEnd is two variants, not a field-dependent struct"
    type: code
    pass_when: "`pub enum PairEnd { Sub { node, port }, Body { instance, port } }` is declared in cgsmiles/ast.rs with no Option field and no `instance` on `Sub`; fixture F8 asserts `Sub` ends at level 0 whose `levels[1].nodes[node].parent` equals the pair's edge endpoint, and `Body` ends at level 1."
    status: verified
    verified_by: impl-gate
    last_checked: 2026-09-21
    note: "ast.rs:112 PairEnd { Sub{node,port}, Body{instance,port} }, no Option; resolve.rs:817 Sub ends' parent equals the edge endpoint, :881 Body ends at level 1"
  - id: ac-005
    summary: "Pairing follows R4.6, R4.12 and R4.13 (whole-descriptor flip, written order, consumption)"
    type: scientific
    pass_when: "Unit tests in cgsmiles/resolve.rs pass: flip is an involution (Left<->Right, Symmetric<->Symmetric, Shared<->Shared); descriptors differing only in label, or only in effective order, are incompatible while `[$]` pairs `-[$]`; F2 yields four head-to-tail pairs (edge (OH0,PEO1) consumes PEO1's first port, edge (PEO1,PEO2) its second) and no free port; F3 yields three pairs in edge order (0,1),(1,2),(0,2) with ends B0.a0<->B1.a0, B1.a1<->B2.a0, B0.a1<->B2.a1; F5 yields three pairs matching the Domain-basis step list with BB0.C1 `<` and BB2.C0 `>` in no pair; F6 yields two Left/Right pairs N(i)-C(i+1); F7 yields three label-forced pairs; a base-only IR `{[#PEO][#PEO][#PEO]}` yields `pairs == [[]]` with its two written edges intact and no error."
    status: verified
    verified_by: scientist
    last_checked: 2026-09-21
    note: "scientist evaluator 2026-09-21 against resolve.py @ 910c9ee: whole-descriptor flip, exact label, effective order, written-order scan with consumption; plus direct flip/compatible table tests (involution incl. Shared↔Shared, label-only and order-only differences refused, [$] pairs -[$]) and F2/F3/F5/F6/F7/base-only pins; per-atom entity grouping at the last level fixed and pinned"
  - id: ac-006
    summary: "A Double coarse edge makes two single bonds, not one double"
    type: scientific
    pass_when: "F4 `{[#SC3]=[#SC3]}.{#SC3=[$]CCC[$]}` yields two pairs on the one edge (`CGEdge.order == CGBondOrder::Double`, multiplicity 2) with bond indices 0 and 1 and both `ResolvedPair.kind == BondKind::Single`."
    status: verified
    verified_by: scientist
    last_checked: 2026-09-21
    note: "scientist evaluator 2026-09-21: multiplicity() is the only loop bound, no CGBondOrder reaches a bond class; F4 two pairs bond 0/1 both Single, expansion 6/6 with both inter bonds BondType::Single (CGsmiles docs multiple_resolutions)"
  - id: ac-007
    summary: "Bond-kind precedence reproduces the Daylight rule"
    type: scientific
    pass_when: "Fixture F9 passes all four cases: no symbol between two written-aromatic ports -> BondKind::Aromatic; explicit `-` on one side paired with a bare `[$]` -> BondKind::Single (biphenyl); `=` on both sides -> BondKind::Double; `=` against no symbol -> CgUnmatchableEdge; and the Kekulé spelling `{[#PH][#PH]}.{#PH=[$]C1=CC=CC=C1}` -> BondKind::Single."
    status: verified
    verified_by: scientist
    last_checked: 2026-09-21
    note: "scientist evaluator 2026-09-21: written symbol wins, absence between written-aromatic ports → Aromatic (BondType::Aromatic + BondNumber::Unknown, no 1.5), Kekulé → Single, -[$] vs [$] → Single, =/= → Double, = vs none → unmatchable; stamp read off a never-perceived cached body (Daylight SMILES §3.2.2)"
  - id: ac-008
    summary: "An unmatchable edge errors rather than being dropped"
    type: code
    pass_when: "`{[#A][#B]}.{#A=[$a]C,#B=[$b]C}`, `{[#A][#B]}.{#A=CC[>],#B=[>]O}` and F7 with one label flipped each return Err whose kind is `CgUnmatchableEdge { level, edge }` naming the offending edge, with `notation == Notation::CGsmiles`, a non-empty input string and a span; no test asserts a silently reduced edge or pair count."
    status: verified
    verified_by: impl-gate
    last_checked: 2026-09-21
    note: "resolve.rs:755 (F7 label flip), :950 (= vs none), :971 ([$a]/[$b]), :990 (two Right) each CgUnmatchableEdge{level,edge} with notation, input, span; no reduced-count assertion anywhere"
  - id: ac-009
    summary: "Expansion reproduces the hand-derived atom and bond counts with real bond classes"
    type: scientific
    pass_when: "CGSmilesIR::to_atomistic gives F2 11 atoms / 10 bonds (inter-fragment bonds BondType::Single), F3 6/6 with BondType::Aromatic + BondNumber::Unknown on the three inter-bead bonds, F4 6/6 with both inter-bead bonds BondType::Single, F5 (hand-built IR) 16/17 with BB2.C1 of degree 3, F6 13/12 and F7 (hand-built IR) 9/9; no atom is created for any descriptor."
    status: verified
    verified_by: scientist
    last_checked: 2026-09-21
    note: "scientist evaluator 2026-09-21 re-derived every count: F2 11/10, F3 6/6 aromatic inter, F4 6/6, F5 16/17 with BB2.C1 degree 3, F6 13/12, F7 9/9; set_bond_class called for every pair; no atom per descriptor"
  - id: ac-010
    summary: "Every expanded atom carries frag_id for its instance"
    type: runtime
    pass_when: "For F2, every atom has property \"frag_id\" == PropValue::Int(i) where i is the node's index in the lowest CG level, and the set of values is exactly 0..5."
    status: verified
    verified_by: impl-gate
    last_checked: 2026-09-21
    note: "to_atomistic.rs:491 frag_ids == [0,1,1,1,2,2,2,3,3,3,4] as PropValue::Int (set 0..5)"
  - id: ac-011
    summary: "Expansion is topology only, refuses a graph body, and discards no Result"
    type: code
    pass_when: "A test asserts the expanded molecule has no atom positions and no added hydrogens (atom count equals the bodies' written heavy-atom count); a lowest-level `FragmentBody::Graph` returns `CgNotExpandable`, and a base-only IR returns `CgNotExpandable` with the payload `base-only string (no fragment table)`; and no `let _ =` appears on any fallible call in cgsmiles/resolve.rs or cgsmiles/to_atomistic.rs (MolRsError maps to CgBuild)."
    status: verified
    verified_by: impl-gate
    last_checked: 2026-09-21
    note: "to_atomistic.rs:570 no positions, :585 no hydrogens, :602 graph body → CgNotExpandable, :658 base-only payload; zero `let _ =`/`.ok()` in resolve.rs and to_atomistic.rs, MolRsError → CgBuild"
  - id: ac-012
    summary: "One body walk per fragment definition, never per instance"
    type: code
    pass_when: "`fragment_to_atomistic` is called only through the private `FragmentCache` in cgsmiles/resolve.rs and cgsmiles/to_atomistic.rs, which is constructed as a local inside `resolve` and inside `to_atomistic` (no field of `CGSmilesIR` holds an `Atomistic`; no `static`, `thread_local` or `OnceCell` in cgsmiles/) and whose rustdoc states the no-perception invariant; no other walker over a fragment body exists in cgsmiles/; a test with one definition used by three instances asserts the cache holds one entry and the conversion ran once; a test on `fragment_to_atomistic(\"C([$]O)[>]\")` pins the two-entry visit-order map."
    status: verified
    verified_by: impl-gate
    last_checked: 2026-09-21
    note: "fragment_to_atomistic called once under cgsmiles/, at resolve.rs:513 inside FragmentCache::get_or_build; cache is a local in resolve and in to_atomistic, no static/thread_local/OnceCell, no Atomistic on CGSmilesIR; rustdoc states the no-perception invariant; tests resolve.rs:1031 (one entry for three lookups) and :1014 (walker-order map)"
  - id: ac-013
    summary: "Public rustdoc example for CGSmilesIR::to_atomistic runs green"
    type: runtime
    pass_when: "`cargo test --doc -p molcrafts-molrs --features full,filesystem,stream` passes, and the example on `CGSmilesIR::to_atomistic` parses F2 `{[#OH][#PEO]|3[#OH]}.{#OH=[$]O,#PEO=[$]COC[$]}` and asserts the hard-coded values `n_atoms() == 11` and `n_bonds() == 10` using only public API."
    status: verified
    verified_by: impl-gate
    last_checked: 2026-09-21
    note: "chem/ast.rs:168 pub(crate) written_aromatic; add_atom_node has no `aromatic: true` literal; tests cover Organic, Bracket-Element, BracketSymbol::Aromatic, Wildcard, Query; resolve.rs:292 reads the is_aromatic stamp only"
  - id: ac-014
    summary: "Invariant, placement rationale and deferred decisions are written down"
    type: docs
    pass_when: "`CGSmilesIR`'s rustdoc states that an IR that exists is fully instantiated, validated and resolved; cgsmiles/mod.rs carries the sentence on why pairing is parse-stage while ring closure stays build-stage; `CgNotExpandable`'s `Display` arm renders `{0}: no atomistic body to expand` and a test in error.rs asserts it for a fragment-name payload and for the payload `base-only string (no fragment table)`; `CgUnmatchableEdge`'s rustdoc says a derived edge carries a copy of the inducing parent edge's span; the `to_atomistic` rustdoc names the per-atom `frag_id` key; the rustdoc documents parse-order pairing and its difference from the reference's adjacency order and the isomorphism contract with the molpy builder; `git diff` shows no change to .claude/notes/notes.md and the implementation summary names the `frag_id`-vs-`mol_id`/`res_id` decision, the `Quadruple -> Double` approximation, the six discarded Results in smiles/to_atomistic.rs (:258, :263, :268, :283, and the `set_atom` calls at :336, :340) and the `InvalidElement` mislabel at :241-247 as deferred items for `/mol:note` / `/mol:fix`."
    status: verified
    verified_by: impl-gate
    last_checked: 2026-09-21
    note: "chem/ast.rs:261/282 bond_type/bond_number with the Quadruple → Double rustdoc; free fns deleted; caller smiles/to_atomistic.rs:356; whole-table tests incl. Aromatic → (Aromatic, Unknown)"
  - id: ac-015
    summary: "Full gate is green with no new lints"
    type: runtime
    pass_when: "`cargo fmt --check`, `cargo clippy -p molcrafts-molrs --all-targets --features full,filesystem,stream -- -D warnings`, `cargo test -p molcrafts-molrs --lib --features full,filesystem,stream` and `cargo test --doc -p molcrafts-molrs --features full,filesystem,stream` all exit 0."
    status: verified
    verified_by: impl-gate
    last_checked: 2026-09-21
    note: "ast.rs:181 EdgeOrigin, :341 CGEdge.origin; resolve.rs:788 all Written before resolution; :836/:858/:870/:881 F8 five edges, Derived{0,0}/{0,1} last, Single, inducing span, pairs[1] indexes the final list"
out_of_scope:
  - "Six discarded Results in smiles/to_atomistic.rs (routed to /mol:fix)"
  - "add_bond_with wrapping a bond failure as InvalidElement at smiles/to_atomistic.rs:241-247 (routed to /mol:fix)"
  - "BondType has no Quadruple (approximation documented; separate spec)"
  - "CGMapping deletion (02a)"
  - "A query for unpaired ports (no caller yet)"
  - "The first-character-only compatibility mode (legacy=False)"
  - "Python bindings (01e); to_fragment (02b)"
  - "Coordinates, hydrogen repletion, aromaticity perception, kekulization"
  - "Writing CGsmiles"
---

# Acceptance — cgsmiles-01d-resolve

Done means: descriptor pairing is computed once by the reader and stored on the IR with provenance-tagged derived edges; the whole-descriptor flip rule, written-order greedy matching and the Daylight bond-kind precedence are proven by hand-derived fixtures; expansion of the lowest CG level reproduces the hand-derived atom and bond counts with real bond classes and per-atom `frag_id`; one walker per fragment definition; and the doc gate runs the public example.

- **ac-001 / ac-002** close the two duplicate-implementation findings: the notation's aromatic declaration and the `BondKind → (BondType, BondNumber)` mapping each get one owner on the shared vocabulary.
- **ac-003 / ac-004** are shape criteria: provenance is a tag on one list, and a pair end is an enum whose variants are independently readable.
- **ac-005 – ac-007** are the science of the notation — the pairing rules R4.6/R4.12/R4.13, the multiplicity rule R6.2, and the precedence rule R4.14 — proven by literal expectations. `ac-006` guards the single most likely implementation error (`CGBondOrder::Double` read as one `BondKind::Double`).
- **ac-008 / ac-011** are the no-silent-debt criteria: a written edge that cannot be matched is an error, and no fallible call on the new path is discarded.
- **ac-009 / ac-010** prove the expansion contract, including the `set_bond_class` step that `add_bond`'s `Single` default would otherwise hide, and the `frag_id` stamp.
- **ac-012** is the anti-duplication criterion for the single walker, and pins the cache as a per-call local so the IR never grows a second representation of its bodies.
- **ac-013** is this repo's regression example: a doctest, since molrs has no `regressions/` tree.
- **ac-014** makes the invariants and deferred decisions readable by someone who never sees this spec, without the implementation writing to `/mol:note`'s file.
