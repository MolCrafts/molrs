---
spec: cgsmiles-01c-fragments
created: 2026-09-21
criteria:
  - id: ac-001
    summary: "AST gains fragments, CGFragmentDef, FragmentBody and CGNode.parent with the invariants in rustdoc"
    type: code
    pass_when: "molrs/src/io/smiles/cgsmiles/ast.rs defines `CGFragmentDef { name, body, span }` and `enum FragmentBody { Graph(CGGraph), Smiles(SmilesIR) }`; `CGSmilesIR` has a public `fragments: Vec<BTreeMap<String, CGFragmentDef>>` field and `CGNode` a `parent: Option<usize>` field; the `CGSmilesIR` rustdoc states the authority rule (fragments[k] authoritative, levels[k+1] derived), the alignment invariant and \"a CGSmilesIR that exists is fully instantiated and validated\"; the `CGNode.parent` rustdoc states \"None in levels[0] and in every FragmentBody::Graph body; set by instantiation, never inherited\"; no file under chem/ changes."
    status: verified
    verified_by: impl-gate
    last_checked: 2026-09-21
    note: "ast.rs:77/84/104/181; rustdoc §Authority, §Alignment invariant, 'fully instantiated and validated' scoped to parse_cgsmiles values (fields are pub); parent rule at ast.rs:173-180; git diff shows no change under chem/"
  - id: ac-002
    summary: "instantiate and validate_ir are private steps; one public re-export path"
    type: code
    pass_when: "`fn instantiate(parent: &CGGraph, defs: &BTreeMap<&str, &CGGraph>, input: &str) -> Result<CGGraph, SmilesError>` exists in molrs/src/io/smiles/cgsmiles/instantiate.rs with no `pub` and is not a method on `CGGraph`; `validate_ir(ir, input)` in cgsmiles/validate.rs is `pub(crate)`; `parse_cgsmiles` remains the only public function of the format and calls syntax → instantiate → validate in that order; molrs/src/io/smiles/mod.rs declares `mod cgsmiles;` without `pub` and is the only file re-exporting `CGFragmentDef` and `FragmentBody` publicly, with cgsmiles/mod.rs holding the inner `pub use`."
    status: verified
    verified_by: impl-gate
    last_checked: 2026-09-21
    note: "instantiate.rs:41 pub(super) free fn (not pub, not a method); validate.rs:35 pub(crate); parser.rs:209/233 call instantiate then validate_ir; io/smiles/mod.rs:111 `mod cgsmiles;` private, :129 the only public re-export; cgsmiles/mod.rs:208 inner pub use"
  - id: ac-003
    summary: "Block splitting and the fragment-table grammar raise the five structural errors with spans into the full input"
    type: code
    pass_when: "Unit tests in cgsmiles/parser.rs show `parse_cgsmiles` returning `CgExpectedBlock` for `{[#A]}.#A=[$]C[$]` and `{[#A]}.{#A=CC}}`, `CgMalformedFragmentEntry` for `{[#A]}.{#A}`, `CgEmptyFragmentBody` for `{[#A]}.{#A=}`, `CgEmptyBlock` for `{[#A]}.{}` and `CgDuplicateFragment(\"A\")` for `{[#A]}.{#A=CC,#A=CCC}`, each with `notation == Notation::CGsmiles`, `input` equal to the whole string and a span inside it; `{[#OHter]}.{#OHter=[$][O-].[Na+]}` parses with one body, `[*;s=C,0]` is one entry, and a body with a second `=` splits on the first; no `CgEmptyFragmentBlock` or `CgAtomAnnotationUnsupported` variant exists in the crate; cgsmiles/ contains no regex and no `str::split` over the input."
    status: verified
    verified_by: impl-gate
    last_checked: 2026-09-21
    note: "parser.rs tests: separator/second-block/text-after → CgExpectedBlock, entry-without-equals → CgMalformedFragmentEntry, empty body, empty block, duplicate name; OHter/[Na+] one body (:2124), [*;s=C,0] one entry (:2268), first-equals split (:2135); zero CgEmptyFragmentBlock/CgAtomAnnotationUnsupported in crate; zero regex/str::split under cgsmiles/"
  - id: ac-004
    summary: "01b's second-block rejection is removed and replaced by positive multi-block tests"
    type: code
    pass_when: "The test `second_block_is_trailing_characters` and 01b's single-block invariant assertion in cgsmiles/parser.rs are gone (the per-shape assertions of ac-010 replace them), and the `levels.len() == 1` invariant sentence is gone from the cgsmiles/mod.rs and `CGSmilesIR` rustdoc; `{[#PEO][#PEO]}[#X]` returns `CgExpectedBlock`; passing tests in cgsmiles/parser.rs parse F2 `{[#OH][#PEO]|3[#OH]}.{#OH=[$]O,#PEO=[$]COC[$]}` to `levels[0]` with 5 nodes / 4 edges, every `parent == None`, `fragments[0]` keyed `{OH, PEO}` with both bodies `FragmentBody::Smiles` (PEO 3 atoms / two `Symmetric` descriptors, OH 1 atom / one), and `{[#EO]|5}` to `fragments.is_empty()`."
    status: verified
    verified_by: impl-gate
    last_checked: 2026-09-21
    note: "second_block_is_trailing_characters and TrailingCharacters absent from cgsmiles/; the remaining `levels.len() == 1` text is the alignment-invariant base-only clause, not 01b's sentence; F2 tests :1879-1929; `{[#EO]|5}` :2077; `{[#PEO][#PEO]}[#X]` :2338"
  - id: ac-005
    summary: "Positional dispatch parses CG bodies at absolute offsets and atomistic last-table bodies through parse_fragment_smiles"
    type: scientific
    pass_when: "Parsing F8 `{[#B1][#B2][#B1]}.{#B1=[>][#PEO][#PEO][<],#B2=[>][#PE][#PE][<]}.{#PEO=[>]COC[<],#PE=[>]CC[<]}` yields `fragments[0][\"B1\"].body` as `Graph` with 2 nodes / 1 edge and `DescriptorKind::Right` on node 0, `Left` on node 1, and `fragments[1][\"PEO\"].body` as `Smiles` with 3 atoms carrying `Right`/`Left`; `|n` as a body's last token parses; a body-level `|n` over a ring marker returns `CgRepeatOnRingMarker` whose span start equals the token's offset in the full string; a descriptor-only branch attaches `Right` to the `[#B]` node in `{[#A]}.{#A=[#B]([>])[#C]}.{#B=[>]CC[<],#C=[>]CC[<]}` and to the nitrogen in `{[#A]}.{#A=N([>])C}`; F8's leading `[>]` binds to B1's first node; a definition never referenced parses cleanly; cgsmiles/parser.rs never constructs a `Scanner` over a substring."
    status: verified
    verified_by: scientist
    last_checked: 2026-09-21
    note: "scientist evaluator 2026-09-21: nine clauses each pinned by a named parser.rs test; semantics match CGsmiles reference read_fragments.py (leading descriptor → first node, descriptor-only branch → enclosing anchor, first-`=` split) @ 910c9ee"
  - id: ac-006
    summary: "CgLastBlockNotAtomistic carries the boxed inner kind, the inner span and the full input"
    type: code
    pass_when: "`{[#A]}.{#A=CC(}` returns `CgLastBlockNotAtomistic(inner)` with `*inner == SmilesErrorKind::UnclosedBranch` and `{[#A]}.{#A=[#X][#X]}` returns `CgLastBlockNotAtomistic(inner)` with `*inner == SmilesErrorKind::UnexpectedChar('#')`; in both the outer error's `input` is the full CGsmiles string, its `notation` is `Notation::CGsmiles`, its span equals the inner span shifted by the body's start offset in the full string, and its rendered message opens with the CGsmiles prefix and states the last-block-is-atomistic rule."
    status: verified
    verified_by: impl-gate
    last_checked: 2026-09-21
    note: "parser.rs:2434 (UnclosedBranch boxed+rebased) and :2394 (UnexpectedChar('#')); message rule pinned at :2415 and error.rs:844/854"
  - id: ac-007
    summary: "AtomAnnotationUnsupported propagates unchanged; cgsmiles/ has no annotation pre-scan"
    type: code
    pass_when: "`{[#A][#B]}.{#A=[C;0.5]C,#B=CC}` and `{[#T]}.{#T=C1=CCCC[*;s=C,0][*;s=C,0]}` both return `SmilesErrorKind::AtomAnnotationUnsupported(_)` (not wrapped in `CgLastBlockNotAtomistic`) with the bracket atom's span re-based into the full input, `input` equal to the full string and `notation == Notation::CGsmiles`, and no function under molrs/src/io/smiles/cgsmiles/ inspects `;` inside `[` … `]` — the kind is produced only by 01a's `parse_bracket_atom`, and cgsmiles/parser.rs may name the variant only in the re-basing match arm that propagates it un-wrapped."
    status: verified
    verified_by: impl-gate
    last_checked: 2026-09-21
    note: "parser.rs:2250/2271 un-wrapped AtomAnnotationUnsupported; only construction site is 01a's parse_bracket_atom; parser.rs:458 is the propagating arm; cgsmiles/ reads ';' only as the annotation-field separator of [#NAME;...] nodes, never inside atomistic bodies"
  - id: ac-008
    summary: "check_coverage is the single owner of CgUndefinedFragment and runs at every level"
    type: code
    pass_when: "`{[#A][#B]}.{#A=CC}` returns `CgUndefinedFragment(\"B\")`, `{[#A]}.{#A=[#X]}.{#Y=CC}` returns `CgUndefinedFragment(\"X\")` and `{[#A][#Q]}.{#A=[#X]}.{#X=CC}` returns `CgUndefinedFragment(\"Q\")` (coverage runs before instantiation), each with the referencing node's span; `CgUndefinedFragment` is constructed only inside `CgParser::check_coverage`; cgsmiles/instantiate.rs contains no `CgUndefinedFragment`, and `instantiate` with a name absent from `defs` returns `CgBuild(_)` naming the fragment, with no `panic!`, `expect`, `unwrap` or `debug_assert!` on that path."
    status: verified
    verified_by: impl-gate
    last_checked: 2026-09-21
    note: "three CgUndefinedFragment tests + wildcard; single construction site parser.rs:254 inside check_coverage; instantiate.rs has none, absent name → CgBuild (:51, test :281), no panic/expect/unwrap on that path"
  - id: ac-009
    summary: "Instantiation copies fragments disjointly with parent back-pointers and creates no inter-copy edges"
    type: scientific
    pass_when: "For F8, `levels.len() == 2`, `levels[1].nodes` has 6 entries named PEO,PEO,PE,PE,PEO,PEO with `parent` values `[Some(0), Some(0), Some(1), Some(1), Some(2), Some(2)]`, `levels[1].edges` has exactly three entries, all `CGBondOrder::Single`, joining (0,1), (2,3), (4,5), no edge joins nodes with different parents, and the descriptors on each copy equal those of the body it came from (`Right` on the first node of each copy, `Left` on the second); a body reused twice yields copies whose `parent` values differ."
    status: verified
    verified_by: scientist
    last_checked: 2026-09-21
    note: "scientist evaluator 2026-09-21: F8 level-1 layout pinned by six parser.rs tests + five instantiate.rs tests; matches resolve.py::resolve_disconnected_molecule / graph_utils.merge_graphs (disjoint copies, fragid → parent, intra edges only)"
  - id: ac-010
    summary: "Alignment invariant holds for the base-only, two-block and three-block shapes"
    type: code
    pass_when: "Three tests assert `{[#A][#B]}` gives `fragments.is_empty() && levels.len() == 1`, F2 gives `fragments.len() == 1 && levels.len() == 1`, and F8 gives `fragments.len() == 2 && levels.len() == 2`."
    status: verified
    verified_by: impl-gate
    last_checked: 2026-09-21
    note: "parser.rs:2052 base-only, :2061 F2, F8 two-tables/two-levels test"
  - id: ac-011
    summary: "validate_ir rejects squash descriptors at every level"
    type: code
    pass_when: "`validate_ir` returns `CgSquashUnsupported` — spanned at the carrying node for the base graph and graph bodies, at the definition (`CGFragmentDef.span`) for an atomistic body — for `{[#SC4]1[#TC5][#TC5]1}.{#SC4=Cc(c[!])c[!],#TC5=[!]ccc[!]}` (atomistic body), for `{[#A]}.{#A=[!][#X]}.{#X=[$]C[$]}` (graph body) and for `{[#A][!]}` (base graph); `parse_cgsmiles` surfaces the same error for each; `DescriptorKind::Shared` is matched in cgsmiles/validate.rs and nowhere else under cgsmiles/."
    status: verified
    verified_by: impl-gate
    last_checked: 2026-09-21
    note: "validate.rs:175/201/225 on validate_ir (graph body, smiles body, base graph); parse-level pins at parser.rs:2225 (F9 atomistic body) and :2234 (base graph); the graph-body fixture reaches validate_ir through the same parse_cgsmiles path (parser.rs:209) and is pinned on validate_ir directly; DescriptorKind::Shared is matched only at validate.rs:78 (parser.rs:909 constructs it from the `!` glyph, which is not a match)"
  - id: ac-012
    summary: "Every must-raise fixture is a separately named test asserting the exact kind"
    type: code
    pass_when: "The fifteen must-raise strings of the Testing strategy (squash, two `AtomAnnotationUnsupported`, `CgDuplicateFragment`, three `CgUndefinedFragment`, three `CgExpectedBlock`, `CgEmptyBlock`, `CgEmptyFragmentBody`, `CgMalformedFragmentEntry`, two `CgLastBlockNotAtomistic`) each have their own `#[test]` function named for the variant (the two squash fixtures tested directly on `validate_ir` belong to ac-011 and are not part of the fifteen), and every one asserts the variant with `matches!` on `err.kind`, never `is_err()` and never a message substring; `Display` tests in error.rs cover the eight new `Cg*` arms."
    status: verified
    verified_by: impl-gate
    last_checked: 2026-09-21
    note: "all fifteen present as named #[test] fns asserting `matches!(err.kind, …)` (3 CgExpectedBlock, 1 CgMalformedFragmentEntry, 1 CgEmptyFragmentBody, 1 CgEmptyBlock, 1 CgDuplicateFragment, 3 CgUndefinedFragment, 2 CgLastBlockNotAtomistic, 2 AtomAnnotationUnsupported, 1 squash via parse_cgsmiles); zero is_err() in cgsmiles/ tests; error.rs:771-866 Display tests for every new arm"
  - id: ac-013
    summary: "F8 doctest on the public API passes under the doc gate"
    type: runtime
    pass_when: "`cargo test --doc -p molcrafts-molrs --features full,filesystem,stream` passes with a non-ignored rustdoc example in cgsmiles/mod.rs that parses F8 through `molrs::io::smiles::parse_cgsmiles` and asserts the hard-coded values `levels.len() == 2`, `levels[0].nodes.len() == 3`, `levels[1].nodes.len() == 6`, `levels[1].nodes[3].parent == Some(1)` and `matches!(fragments[1][\"PEO\"].body, FragmentBody::Smiles(_))`."
    status: verified
    verified_by: impl-gate
    last_checked: 2026-09-21
    note: "cgsmiles/mod.rs F8 doctest; cargo test --doc green in the 2026-09-21 gate"
  - id: ac-014
    summary: "Rustdoc records the pipeline decision, the last-block limitation and the non-mirrored reference behaviours"
    type: docs
    pass_when: "`parse_cgsmiles`'s rustdoc states that the last block must be an atomistic (OpenSMILES) body, positionally, with no opt-out, and the `Display` text of `CgLastBlockNotAtomistic` states the same rule; the cgsmiles/mod.rs module doc names the three non-mirrored reference behaviours (duplicate names, `{}`, `|n` as last body token) and the level/atomistic asymmetry; cgsmiles/instantiate.rs and cgsmiles/validate.rs each carry a `//!` module doc naming their single step and citing the rule tags they implement; 01b's `levels.len() == 1` invariant sentence and `TrailingCharacters` no longer appear in the cgsmiles/mod.rs rustdoc; the implementation summary names the recorded bend of Shape check #2 as a deferred item for `/mol:note`; `git diff` shows no change to .claude/notes/notes.md."
    status: pending
  - id: ac-015
    summary: "Full gate green"
    type: runtime
    pass_when: "`cargo fmt --check`, `cargo clippy -p molcrafts-molrs --all-targets --features full,filesystem,stream -- -D warnings`, `cargo test -p molcrafts-molrs --lib --features full,filesystem,stream` and `cargo test --doc -p molcrafts-molrs --features full,filesystem,stream` all exit 0."
    status: verified
    verified_by: impl-gate
    last_checked: 2026-09-21
    note: "cargo fmt --check, clippy (lib + cxxapi), test --lib, test --doc, rustdoc link lint all exit 0 on 2026-09-21"
out_of_scope:
  - "Descriptor pairing, pairs, EdgeOrigin, inter-copy edges, CGSmilesIR::to_atomistic (01d)"
  - "fragment_to_atomistic (01a); to_fragment (02b)"
  - "Changes to parse_smiles / parse_fragment_smiles / Dialect beyond 01a's AtomAnnotationUnsupported amendment"
  - "Changes to CGBondOrder, CGEdge, CgInvalidBondOrder, CgRepeatOnRingMarker, Scanner::seek (01b)"
  - "Support for [!] squash, weights, chirality, positional atom annotations, wildcard overloading (v1 RAISE)"
  - "A flag to keep the deepest level coarse-grained"
  - "The R3.2 order-0 exemption (unreachable)"
  - "A Span on CGGraph"
  - "Writing multi-block CGsmiles"
---

# Acceptance — cgsmiles-01c-fragments

Done means: the reader accepts the full multi-block notation through the one public door, every intermediate table is expanded once with `parent` back-pointers and no inter-copy edges, the last table's bodies are kept as descriptor-bearing `SmilesIR` values, every refused input names its variant at a span into the original string, and the invariants a successor relies on are written where a reader of the type will see them.

- **ac-001 / ac-002** are the shape contract: the additive AST surface with its documented invariants, and the privacy that makes the recorded bend of Shape check #2 safe (no public `instantiate`, no `CGGraph` method, one public re-export path).
- **ac-003 – ac-008** are the reader's user-visible behaviour: the structural grammar errors, the removal of 01b's second-block rejection, positional dispatch at absolute offsets, the boxed inner kind on `CgLastBlockNotAtomistic`, unchanged propagation of 01a's annotation diagnostic (no second lexer), and the single owner of name coverage with `CgBuild` as the internal-error path.
- **ac-009 – ac-011** are the notation semantics: R4.10 expansion, R4.20 absence of inter-copy edges, R3.3 descriptor placement, the alignment invariant, and the squash rejection by one structural rule.
- **ac-012 / ac-013** are the test-shape obligations: fifteen named must-raise tests asserting exact kinds, and the F8 doctest that serves as this repo's executable example in place of a `regressions/` script.
- **ac-014 / ac-015** are documentation and the gate.
