---
spec: cgsmiles-01a-descriptors
created: 2026-09-21
criteria:
  - id: ac-001
    summary: "AST carries descriptors as a typed struct with role-named kinds and optional order"
    type: code
    pass_when: "molrs/src/io/smiles/chem/ast.rs defines `pub struct BondingDescriptor { kind: DescriptorKind, label: String, order: Option<BondKind> }` and `pub enum DescriptorKind { Symmetric, Left, Right, Shared }` (each variant's /// naming its glyph and pairing rule), `AtomNode` has `pub descriptors: Vec<BondingDescriptor>`, `chem/ast.rs` still has zero `use` lines, and `cargo clippy -p molcrafts-molrs --all-targets --features full,filesystem,stream -- -D warnings` is clean."
    status: verified
    last_checked: 2026-09-21
  - id: ac-002
    summary: "One Dialect enum replaces ParserMode and the writer Mode"
    type: code
    pass_when: "`pub(crate) enum Dialect { Smiles, Smarts, FragmentSmiles }` exists in molrs/src/io/smiles/chem/mod.rs; `ParserMode` (parser.rs) and the writer's `Mode` (smiles/write.rs) no longer exist; the three SMARTS-only guards in smiles/write.rs (formerly `mode == Mode::Smiles` at :144, :333, :340) test `dialect != Dialect::Smarts` or an exhaustive match."
    status: verified
    last_checked: 2026-09-21
  - id: ac-003
    summary: "Seven new error variants render routing messages"
    type: code
    pass_when: "Unit tests in molrs/src/io/smiles/error.rs assert a distinct non-empty Display message for DescriptorInPlainSmiles, DescriptorsUnconvertible, BondInsideDescriptor, InvalidDescriptorLabel, InvalidDescriptorOrder, DanglingDescriptor and AtomAnnotationUnsupported, that the first two name `parse_fragment_smiles` and `fragment_to_atomistic` respectively, and that the last says the annotation is unsupported rather than that the bracket is unclosed."
    status: verified
    last_checked: 2026-09-21
  - id: ac-004
    summary: "Descriptor anchoring follows R4.2/R4.3"
    type: scientific
    pass_when: "Unit tests in molrs/src/io/smiles/parser.rs assert parse_fragment_smiles gives: \"[$]COC[$]\" -> descriptor on atom 0 and atom 2 only, order None; \"[>]NCC(=O)[<]\" -> `>` on atom 0 and `<` on the last chain atom; \"Clc[$a]c[$b]\" -> labels \"a\" and \"b\" on the two aromatic carbons; \"[>][$1]COC[<]\" -> atom 0 carries [Right, Symmetric(\"1\")] in that written order."
    status: pending
  - id: ac-005
    summary: "Bond order is read outside the bracket on both junction forms"
    type: scientific
    pass_when: "Unit tests assert order Some(Double) for the descriptor in both \"CC=[$]\" and \"[$]=CCC\" (with C-C-C left single in the latter), order None with a double C0=C1 bond for \"C[$]=CC\", and errors BondInsideDescriptor for \"[<=1]\" and \"[$-]\" and InvalidDescriptorOrder for \"c:[$]\"."
    status: pending
  - id: ac-006
    summary: "Branch-only and mixed-branch descriptors bind to the parent anchor"
    type: code
    pass_when: "Unit tests assert parse_fragment_smiles(\"N([>])C\") yields 2 atoms, 1 bond and the `>` descriptor on N; \"C([$])O\" yields `$` on C; \"C([>]N)C\" puts `>` on the first C, not on N; \"C(=[>])C\" gives the first C a `>` with order Some(Double); \"C()\" still returns a parse error."
    status: verified
    last_checked: 2026-09-21
  - id: ac-007
    summary: "The plain and SMARTS dialects are unchanged and strict"
    type: code
    pass_when: "Unit tests assert parse_smiles(\"[$]COC[$]\") and parse_smiles(\"[$(C)]\") return DescriptorInPlainSmiles, parse_fragment_smiles(\"[$]\") and parse_fragment_smiles(\"C.[$]\") return DanglingDescriptor, parse_fragment_smiles(\"[C;0.5]\") and parse_fragment_smiles(\"[*;s=C,0]\") return AtomAnnotationUnsupported while parse_smiles(\"[C;0.5]\") returns UnclosedBracket, and that parse_smiles(\"[NH4+]\"), parse_smarts(\"[!C]\"), parse_smarts(\"[$(C)]\") and parse_smiles(\"C$C\") (2 atoms, Quadruple bond) produce exactly the IRs they produce today."
    status: verified
    last_checked: 2026-09-21
  - id: ac-008
    summary: "to_atomistic refuses descriptor-bearing IR through the single choke point"
    type: code
    pass_when: "A unit test in molrs/src/io/smiles/smiles/to_atomistic.rs asserts to_atomistic on the IR from parse_fragment_smiles(\"[$]COC[$]\") returns SmilesErrorKind::DescriptorsUnconvertible, and `Builder` carries exactly one owned collection `descriptors: Vec<(AtomId, BondingDescriptor)>` gated by one `collect_descriptors: bool`, with the guard in `add_atom_node` and no second walker."
    status: verified
    last_checked: 2026-09-21
  - id: ac-009
    summary: "fragment_to_atomistic returns the graph plus the visit-ordered descriptor map"
    type: code
    pass_when: "Unit tests assert fragment_to_atomistic on \"[$]COC[$]\" returns 3 atoms, 2 bonds and the map [(atom 0, Symmetric), (atom 2, Symmetric)]; on \"[>][$1]COC[<]\" atom 0's entries are [Right, Symmetric(\"1\")]; on \"C([$]O)[>]\" both entries are on atom 0 with `$` before `>`; on \"C(N[<])[>]\" the `>` entry (atom 0, the head) precedes the `<` entry (atom 1, inside the branch) — visit order, the reverse of text order; and to_atomistic / fragment_to_atomistic give identical graphs for \"CC(=O)O\"."
    status: verified
    last_checked: 2026-09-21
  - id: ac-010
    summary: "The SMILES FrameReader surfaces the rejection, never a silent drop"
    type: code
    pass_when: "A unit test in molrs/src/io/smiles/frame_reader.rs asserts reading the line \"[>]COC[<]\" returns Err with ErrorKind::InvalidData whose message names the fragment entry point, while \"CCO\" still reads to 3 atoms and 2 bonds."
    status: verified
    last_checked: 2026-09-21
  - id: ac-011
    summary: "Descriptor well-formedness has one home in chem/validation.rs and one call site"
    type: code
    pass_when: "`pub(crate) fn validate_descriptor` exists in molrs/src/io/smiles/chem/validation.rs (its module header amended to cover grammar-level validation shared across dialects), is called from the parser at descriptor construction and nowhere else, and unit tests assert it accepts (\"a1\", Some(Double)) and returns InvalidDescriptorLabel for (\"a-\", None) and InvalidDescriptorOrder for (\"\", Some(Aromatic)); validate_smiles on a hand-built descriptor IR returns DescriptorInPlainSmiles."
    status: verified
    last_checked: 2026-09-21
  - id: ac-012
    summary: "Graph-to-IR paths stay descriptor-free"
    type: code
    pass_when: "A unit test asserts every AtomNode produced by from_atomistic for ethanol has an empty `descriptors` vector and that write_smiles of that IR equals the string it produces before this change."
    status: verified
    last_checked: 2026-09-21
  - id: ac-013
    summary: "Fragment writer round-trips; plain writer and SMARTS constructs refuse"
    type: code
    pass_when: "Unit tests in molrs/src/io/smiles/smiles/write.rs assert write_fragment_smiles is idempotent over a second parse+write pass for \"[$]COC[$]\" and \"[$]=CCC\" (written as \"C=[$]CC\"), \"C[$]=CC\" is stable, \"CC-[$]\" writes as \"CC-[$]\" and re-parses with order Some(Single), \"[!]C\" round-trips with DescriptorKind::Shared, the re-parsed IR keeps kind/label/order, write_smiles on the same IR returns DescriptorInPlainSmiles, write_fragment_smiles on a Query-atom IR and on a BondQuery::Not IR returns an error, and write_fragment_smiles on a hand-built descriptor with Some(Aromatic) returns InvalidDescriptorOrder."
    status: verified
    last_checked: 2026-09-21
  - id: ac-014
    summary: "Public-API doctest runs the fragment path end to end"
    type: runtime
    pass_when: "`cargo test --doc -p molcrafts-molrs --features full,filesystem,stream` passes, and the doctest in molrs/src/io/smiles/mod.rs is runnable (no `ignore`), calls parse_fragment_smiles(\"[$]COC[$]\") then fragment_to_atomistic, and asserts the hard-coded values n_atoms() == 3, ports.len() == 2, and both DescriptorKind::Symmetric."
    status: verified
    last_checked: 2026-09-21
  - id: ac-015
    summary: "Module docs stop claiming parse_smarts feeds perceive::smarts"
    type: docs
    pass_when: "molrs/src/io/smiles/mod.rs no longer states at :3-6 or in the SMARTS pipeline block (:24) that this module's parse_smarts is the frontend of crate::perceive::smarts; every new pub item carries a /// doc comment (# Errors on each Result-returning function); `BondKind::Quadruple`'s doc notes the `$` glyph overload with descriptors; `write_fragment_smiles`'s doc states the canonical-trailing-form contract."
    status: pending
  - id: ac-016
    summary: "Deferred decisions are named, not written into notes.md by the implementation"
    type: docs
    pass_when: "The implementation summary names (a) the cross-repo consumer audit of `io::smiles::parse_smarts` (zero in-repo non-test consumers as of 2026-09-21) and (b) the `C[$]=CC` mid-chain disambiguation inference as deferred decisions for `/mol:note`, and `git diff` shows no change to `.claude/notes/notes.md`."
    status: pending
  - id: ac-017
    summary: "Full gate green"
    type: runtime
    pass_when: "`cargo fmt --check`, `cargo clippy -p molcrafts-molrs --all-targets --features full,filesystem,stream -- -D warnings`, `cargo test -p molcrafts-molrs --lib --features full,filesystem,stream` and `cargo test --doc -p molcrafts-molrs --features full,filesystem,stream` all exit 0."
    status: verified
    last_checked: 2026-09-21
out_of_scope:
  - "Descriptor pairing and compatibility (R4.6–R4.16) — 01d"
  - "[!] squash semantics and its RAISE — 01c"
  - "Fragment bodies, to_fragment, the multi-level CGsmiles string — 01c, 02a–02d"
  - "Hydrogen repletion of unconsumed descriptors"
  - "Python / WASM / C binder exposure of the fragment dialect"
  - "Deleting or relocating io::smiles::parse_smarts (deferred decision for /mol:note)"
  - "CoarseGrain construction from a descriptor-bearing IR"
---

# Acceptance — cgsmiles-01a-descriptors

Done means: bonding descriptors are typed IR data behind their own fragment dialect (`parse_fragment_smiles` / `fragment_to_atomistic` / `write_fragment_smiles`), one `Dialect` enum serves parser and writer, the plain SMILES surface and the registered `SmilesReader` refuse descriptors with routing errors, every `node.spec` match site has an explicit tested decision, the module header tells the truth, and both the `--lib` and `--doc` gates are green.

`ac-001`–`ac-003` are the representation, the dialect enum and the error surface; `ac-002` also pins the writer-guard polarity fix that the `Dialect` merge would otherwise break silently. `ac-004`–`ac-007` bind the parser to the Domain basis: `ac-004`/`ac-005` are `scientific` because they assert CGsmiles semantics (R4.2/R4.3 anchoring, R4.4 out-of-bracket order) against the cited reference fixtures; `ac-007` is the one-name-per-dialect fix, with the plain and SMARTS entries provably unchanged.

`ac-008`–`ac-010` are the silent-loss fix stated three ways — one walker with one flag refuses on the plain path, hands the descriptors back as data on the fragment path, and the registered `FrameReader` surfaces the refusal. `ac-011`–`ac-013` are the per-site decisions at `chem/validation.rs`, `from_atomistic.rs` and `write.rs`, each with its own test, which is what buys the `AtomNode` field its keep.

`ac-014` is this repo's public-API example: molrs has no `regressions/` tree, so the rustdoc doctest under `cargo test --doc` is the equivalent gate, with hand-derived literal expectations and no third-party software at test time. `ac-015`–`ac-016` clear the documentation findings and keep `notes.md` under `/mol:note`'s ownership; `ac-017` is the delivery gate.

Breaking-change note for the release: `SmilesErrorKind` gains seven variants (public, not `#[non_exhaustive]`) and `AtomNode` gains a public field; verified that no in-repo binder and no sibling repo (molpy, molpack, molrec, molvis, Atomiverse) constructs either or matches exhaustively on the enum, so this is visible only at the minor bump.
