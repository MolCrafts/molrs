---
spec: cgsmiles-01b-graph
created: 2026-09-21
revised: 2026-09-21
criteria:
  - id: ac-001
    summary: "The notation is a required field on SmilesError, stamped at every construction site"
    type: code
    pass_when: "molrs/src/io/smiles/error.rs defines `pub enum Notation { Smiles, Smarts, CGsmiles }` and `SmilesError` has a public `notation: Notation` field that is a required argument of `SmilesError::new` (no `Default`, no defaulted `Smiles`); `Scanner::new(input, notation)` stamps it and `error`/`error_at` use it; every non-scanner `SmilesError::new` call site (chem/validation.rs, smiles/to_atomistic.rs, smiles/write.rs, smiles/from_atomistic.rs, smiles/validate.rs, smiles/local_smarts.rs) passes its notation explicitly, with `Smarts` at write.rs's SMARTS emit path and local_smarts.rs; `Parser` derives the notation from its `Dialect` by a total mapping (`Smiles | FragmentSmiles → Smiles`, `Smarts → Smarts`) and holds no second notation field."
    status: verified
    last_checked: 2026-09-21
  - id: ac-002
    summary: "Display renders the notation prefix for all three notations, guarded by tests"
    type: code
    pass_when: "Tests in error.rs assert that a CGsmiles error of a reused kind (`TrailingCharacters`) renders `\"CGsmiles parse error at position N\"` with the caret under column N of the full input, that a `parse_smiles` error renders `\"SMILES parse error …\"`, and that a `parse_smarts` error renders `\"SMARTS parse error …\"` (previously mislabelled as SMILES); one test per new `Cg*` kind asserts its message text; `test_display_with_caret` is unchanged."
    status: verified
    last_checked: 2026-09-21
  - id: ac-003
    summary: "The CG IR lives in cgsmiles/ast.rs with the frozen field names and no public constructor"
    type: code
    pass_when: "molrs/src/io/smiles/cgsmiles/ast.rs defines `CGSmilesIR { levels: Vec<CGGraph>, span }`, `CGGraph { nodes, edges }`, `CGNode { name, charge: Option<F>, annotations: Vec<(String, String)>, descriptors: Vec<BondingDescriptor>, span }` with no `instance` field, `CGEdge { i, j, order: CGBondOrder, span }` and `#[derive(Debug, Clone, Copy, PartialEq, Eq)] enum CGBondOrder { Single, Double, Triple, Quadruple }` with `multiplicity(&self) -> u8` returning 1, 2, 3, 4 (tested in that file); `CGSmilesIR` has no public constructor; nothing CG-related is added under chem/; the `CGNode.charge` rustdoc says \"partial charge in e (CGsmiles `q`, positional slot 2), not a formal charge\"."
    status: verified
    last_checked: 2026-09-21
  - id: ac-004
    summary: "One private module, one public path, no generalized Parser"
    type: code
    pass_when: "molrs/src/io/smiles/mod.rs declares `mod cgsmiles;` without `pub` and re-exports `parse_cgsmiles`, `CGSmilesIR`, `CGGraph`, `CGNode`, `CGEdge`, `CGBondOrder` and `Notation`; cgsmiles/mod.rs holds `pub fn parse_cgsmiles(text: &str) -> Result<CGSmilesIR, SmilesError>` and `pub use` re-exports of its types; `CgParser` in cgsmiles/parser.rs is private; molrs/src/io/smiles/parser.rs gains no CG dialect or mode and changes only to pass the notation to `Scanner::new`."
    status: verified
    last_checked: 2026-09-21
  - id: ac-005
    summary: "The core grammar reproduces the F1 fixtures exactly"
    type: scientific
    pass_when: "Unit tests in cgsmiles/parser.rs assert, with literal expectations: F1.1 one level / one node `PEO` / no edges and `levels.len() == 1`; F1.2 edges exactly `[(0,1,Single),(1,2,Single)]`; F1.3 `[(0,1,Double),(1,2,Triple),(2,3,Quadruple)]`; F1.5 `[(0,1),(0,2)]`; F1.6 three nodes with the ring-closure edge `(0,2)` last; F1.7 `charge == Some(-0.5)` and `annotations == [(\"kind\",\"ether\")]`; F1.8 one `Symmetric` descriptor on each of two nodes and one edge; F1.14 `{[#A]%12[#B][#C]%12}` 3/3 and `{[#A]1[#B][#A]1[#C]}` 4/4; F1.15 the A–C ring edge `Double`; nodes and edges are in parse order."
    status: pending
  - id: ac-006
    summary: "Positional dialect binding follows the normative slot table (slot 2 = q, slot 3 = w)"
    type: scientific
    pass_when: "Tests assert `{[#A;0.5]}` → `charge == Some(0.5)`; `{[#A;0;0.5]}` → `Err` with kind `CgUnsupportedAnnotation { key: \"w\", value: \"0.5\" }`; `{[#A;0;1]}` → `charge == Some(0.0)` with no error; `{[#A;w=2]}` and `{[#A;x=S]}` → `CgUnsupportedAnnotation`; a fourth positional slot → `CgMalformedAnnotation`; after binding, `annotations` contains neither `q` nor `w`."
    status: pending
  - id: ac-007
    summary: "Every refused input names its exact kind"
    type: code
    pass_when: "One test per rejection asserts the exact `SmilesErrorKind` by `matches!`, never by message substring: `{}` → `CgEmptyBlock`; `{[#A;]}`, `{[#A;=1]}`, `{[#A;q=x]}` → `CgMalformedAnnotation`; `{[#*;q=1]}` → `CgAnnotationOnWildcard`; `{[#A].[#B]}` → `CgInvalidBondOrder`; `{[#A]=}` → `CgDanglingBond`; `{[#A]%1}` → `CgInvalidRingMarker`; `{[#A]1[#B]1[#A]}` → `CgDuplicateEdge { i: 0, j: 1 }`; `{[#A]1[#B]}` → `UnmatchedRingClosure`; `{[#A]|0}` and `{[#A]|x}` → `CgInvalidRepeatCount`; `{[#A]1|2[#B]1}` and `{[#A]1[#B]1|2}` → `CgRepeatOnRingMarker`; `{[#A](|2[#B])}` and `{[#A]([#B])([#C])|3}` → `CgRepeatOnBranchedNode`; `{[#PEO][#PEO]}[#X]` → `TrailingCharacters` in a test named `second_block_is_trailing_characters`; every error has `notation == Notation::CGsmiles`."
    status: verified
    last_checked: 2026-09-21
  - id: ac-008
    summary: "|n replay rewinds the same scanner over the full input"
    type: code
    pass_when: "Tests assert F1.4 four nodes / three `Single` edges; F1.9 `{[#A]([#B][#C])|2}` six nodes / five edges in parse order `[(0,1),(1,2),(3,4),(4,5),(0,3)]` (the unit is copied anchor-included per R2.20 and the chaining edge is pushed after the copy's internal edges; fixture ratified by the orchestrator on 2026-09-21 after the original five-node expectation was found to contradict R2.20); F1.13 `{[#A]([#B][#B])|3}` 9 nodes / 8 edges and `{[#A][#B]([#C])|3}` 7 nodes / 6 edges; `{[#A][$1]|3}` yields three nodes each with one `Symmetric` descriptor labelled `1`; a bond symbol before `|` is promoted to the chaining edge order (R2.22); an error raised while a repeated unit is parsed has `err.span.start` within the original string and `err.input` equal to the full text (there is no replay-only error path: every construct in a unit is validated on the first pass, so the test is named for what it proves); cgsmiles/parser.rs never constructs a `Scanner` over a substring."
    status: verified
    last_checked: 2026-09-21
  - id: ac-009
    summary: "Scanner::seek clamps and keeps spans well-formed"
    type: code
    pass_when: "`pub fn seek(&mut self, pos: usize)` exists in chem/scanner.rs with tests showing that seeking back to an earlier offset re-reads the same bytes and `error()` at the re-read position reports that position in the full input, that `seek(len + 1)` lands at `len` (asserted by value, no `debug_assert!`), and that a span minted after a backward seek satisfies `start <= end`."
    status: verified
    last_checked: 2026-09-21
  - id: ac-010
    summary: "Public-API doctest parses F1.2 through the public path"
    type: runtime
    pass_when: "`cargo test --doc -p molcrafts-molrs --features full,filesystem,stream` passes and the cgsmiles/mod.rs doctest uses only `molrs::io::smiles::parse_cgsmiles` to parse `{[#PEO][#PEO][#PEO]}` and asserts the hard-coded values three nodes, `levels[0].edges.len() == 2` and `edges[0].order.multiplicity() == 1`."
    status: verified
    last_checked: 2026-09-21
  - id: ac-011
    summary: "Module docs, table rows and deferred items are recorded; notes.md untouched"
    type: docs
    pass_when: "The cgsmiles/mod.rs rustdoc cites DOC *Basic graph description* and the reference implementation as `CGsmiles @ 910c9ee, read_cgsmiles.py:134`, states the order-0 and non-default-`w` refusals as molrs v1 choices naming the follow-up `cgsmiles-*` link and the schema-vocabulary spec, states the `levels.len() == 1` invariant, the replayed-span caveat on `CGNode`/`CGEdge`, `charge` in e, and the descriptor-vs-ring-marker asymmetry beside `CgRepeatOnRingMarker`; the io/smiles/mod.rs header names CGsmiles with a pipeline line (01a's header rewrite verified, no doc-text test added); the `io` rows of CLAUDE.md (module table) and .claude/notes/architecture-rules.md § Module ownership name CGsmiles; the implementation summary names the four deferred items (two SMARTS parsers/ASTs, `[#6]` vs `Element::by_number` at element.rs:1128, the `%n` → `UnexpectedEnd` mislabel at parser.rs:227, the `io::smiles` module-name inception) for `/mol:note`; `git diff` shows no change to .claude/notes/notes.md."
    status: pending
  - id: ac-012
    summary: "Full gate green"
    type: runtime
    pass_when: "`cargo fmt --check`, `cargo clippy -p molcrafts-molrs --all-targets --features full,filesystem,stream -- -D warnings`, `cargo test -p molcrafts-molrs --lib --features full,filesystem,stream` and `cargo test --doc -p molcrafts-molrs --features full,filesystem,stream` all exit 0."
    status: verified
    last_checked: 2026-09-21
out_of_scope:
  - "Multi-block strings, fragment bodies, levels.len() > 1 (01c)"
  - "Descriptor pairing, pairs, atom expansion (01d)"
  - "Reader/writer seam and CGsmiles emission (01e)"
  - "Bond order 0 and non-default w (refused by v1 choice; later cgsmiles-* link)"
  - "The four deferred items (via /mol:note → /mol:refactor / /mol:fix)"
  - "Python / WASM / C bindings for parse_cgsmiles"
---

# Acceptance — cgsmiles-01b-graph

Done means: `parse_cgsmiles` reads one coarse block into a spanned IR whose field names are frozen for the chain, every malformed input names its exact kind under a message that says which notation was being parsed, `|n` replay rewinds the one scanner over the full input, and the SMARTS mislabel is corrected along the way.

- **ac-001 / ac-002** close the notation-labelling finding: the notation is a required field at construction, not a classifier over variant names, so the five reused kinds and the SMARTS paths render correctly — and the prefix has a guard before anything relies on it.
- **ac-003 / ac-004** pin the shape: the IR beside its single consumer, one public path, `CGBondOrder` as a multiplicity distinct from `BondKind`, and no third mode inside the OpenSMILES parser.
- **ac-005 – ac-007** are the grammar as literal expectations: the F1 table, the normative positional slot table (the two fixtures whose expectations flip under the reversed reading), and one exact kind per refusal.
- **ac-008 / ac-009** are the replay contract: same scanner, full-input spans, clamped `seek`, no substring scanner.
- **ac-010** is this repo's executable public-API example (no `regressions/` tree); **ac-011 / ac-012** are documentation and the gate.
