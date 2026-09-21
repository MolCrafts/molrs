---
spec: cgsmiles-02c-conformer-fragment
created: 2026-09-21
criteria:
  - id: ac-001
    summary: "ElementGraph trait lives in conformer with exactly two methods"
    type: code
    pass_when: "molrs/src/conformer/element_graph.rs defines `pub trait ElementGraph: Sized` with exactly `as_molgraph(&self) -> &MolGraph` and `try_from_molgraph(mol: MolGraph) -> Result<Self, MolRsError>` (no `finish_embed` or other hook), implements it for Atomistic and Fragment in that same file by forwarding to the inherent methods, and molrs/src/conformer/mod.rs re-exports it so `molrs::conformer::ElementGraph` resolves; `grep -rn \"crate::conformer\" molrs/src/core` returns nothing."
    status: pending
  - id: ac-002
    summary: "Conformer::generate is generic and binder call sites still compile"
    type: code
    pass_when: "`Conformer::generate<M: ElementGraph>(&self, mol: &M) -> Result<(M, ConformerReport), MolRsError>` is the only generate method on Conformer, its body calls the unchanged `etkdg::generate_3d_impl` and nothing else between the two conversions, and `cargo check --manifest-path molrs-wasm/Cargo.toml` and `cargo check --manifest-path molrs-python/Cargo.toml` both succeed while `git diff --stat molrs-wasm/src molrs-python/src` is empty."
    status: pending
  - id: ac-003
    summary: "Fragment survives the Atomistic round trip with ports and frag_id, no ETKDG"
    type: code
    pass_when: "Round-trip tests in molrs/src/conformer/element_graph.rs pass under `cargo test -p molcrafts-molrs --lib --features full,filesystem,stream`: Fragment -> Atomistic::try_from_molgraph -> into_inner -> Fragment::try_from_molgraph preserves port count, per-atom frag_id and bond count, with bonds and ports remaining distinct kinds, and no ETKDG run in the test; a compile-time witness instantiates the bound for both types."
    status: pending
  - id: ac-004
    summary: "add_hydrogens contract holds on a ported graph"
    type: scientific
    pass_when: "A unit test in molrs/src/perceive/hydrogens.rs builds a C–C–O graph whose two handle hydrogens are real bonded atoms additionally recorded in a `ports` kind (no h_count / formal_charge set), runs add_hydrogens, and asserts n_ports() is unchanged, every added H has exactly one bond, and the atom count is nine (3 heavy + 2 handles + 4 added, hand-derived from heavy valence 4+4+2 minus 4 internal minus 2 handles)."
    status: pending
  - id: ac-005
    summary: "The 02a prerequisite is named and not re-asserted here"
    type: code
    pass_when: "Summary, Design and Out of scope name idempotent `register_kind` in Atomistic::try_from_molgraph and CoarseGrain::try_from_molgraph (with populated-port guard tests in atomistic.rs / coarsegrain.rs), `Fragment::inherit_frag_ids` and the all-or-nothing frag_id column as cgsmiles-02a's; molrs/src/conformer/element_graph.rs contains no test that asserts `Atomistic::try_from_molgraph`'s kind resolution."
    status: pending
  - id: ac-006
    summary: "Module doctest shows both call paths and the visible relabel step"
    type: runtime
    pass_when: "`cargo test --doc -p molcrafts-molrs --features full,filesystem,stream` passes and the molrs/src/conformer/mod.rs module `//!` example shows `Conformer::new(...).generate(mol)` for an `Atomistic` and for a `Fragment`, with the Fragment arm binding the result as a `Fragment` and then calling `inherit_frag_ids()`."
    status: pending
  - id: ac-007
    summary: "CLAUDE.md records the generic signature; rules amendment routed to /mol:note"
    type: docs
    pass_when: "CLAUDE.md § Conformer Pipeline states `Conformer::new(opts).generate(mol) -> Result<(M, ConformerReport)>` with `M: ElementGraph` and names Atomistic and Fragment as the implementors; `git diff` shows no change to .claude/notes/architecture-rules.md or .claude/notes/notes.md; the implementation summary names the trait-principle-1 amendment (witnesses io/reader.rs:69, core/store/frame_access.rs:46, ElementGraph) as a deferred decision for `/mol:note`."
    status: pending
  - id: ac-008
    summary: "generate rustdoc names the bound, the invariant, the unread ports block and units"
    type: docs
    pass_when: "The `///` block on `Conformer::generate` names `ElementGraph`, states that the returned type equals the input type, states that `ports` and `frag_id` survive the round trip and that the `ports` kind passes through the MMFF staging Frame as an unread block, states the 2-clones-where-1-existed cost, and gives coordinate units as Å."
    status: pending
  - id: ac-009
    summary: "Format and lint gates clean including cxxapi"
    type: code
    pass_when: "`cargo fmt --check`, `cargo clippy -p molcrafts-molrs --all-targets --features full,filesystem,stream -- -D warnings` and `cargo clippy --manifest-path molrs-cxxapi/Cargo.toml --all-targets -- -D warnings` all exit 0."
    status: pending
  - id: ac-010
    summary: "Full lib and doc test gate green"
    type: runtime
    pass_when: "`cargo test -p molcrafts-molrs --lib --features full,filesystem,stream` and `cargo test --doc -p molcrafts-molrs --features full,filesystem,stream` exit 0, and the lib suite still completes in the seconds range (no ETKDG run added to it by this spec)."
    status: pending
out_of_scope:
  - "The 02a prerequisite: idempotent register_kind in Atomistic/CoarseGrain, their guard tests, Fragment::inherit_frag_ids, the all-or-nothing frag_id column"
  - "Python and WASM binder edits (02d)"
  - "architecture-rules.md trait principle 1 amendment (via /mol:note)"
  - "CoarseGrain: ElementGraph"
  - "Port-aware embedding geometry"
  - "Removing the extra MolGraph clone"
  - "A strict Frame block vocabulary declaring the ports block"
---

# Acceptance — cgsmiles-02c-conformer-fragment

Done means: one generic `generate` under its existing name, over a two-method trait that lives beside its only consumer; `Fragment` round-trips through the promotion at each end with ports and `frag_id` intact; the pipeline's added hydrogens are relabelled by a visible caller step, not a hidden hook; no test in this spec runs ETKDG; and the binders compile untouched.

**ac-001 / ac-002 — shape.** One trait, two methods, in its consumer's module, and one generic entry point. The grep in ac-001 is the standing check that no `core → conformer` edge was introduced; the empty binder diff in ac-002 is what makes "source-compatible" a fact rather than a claim.

**ac-003 / ac-004 — the two units that own the properties.** The promotion round trip belongs to the trait impls; the nine-atom count and the one-bond-per-H property belong to `add_hydrogens`, so its file is where a regression is caught. Together they cover what a seeded pipeline run would have proved, without a multi-stage test in the default gate.

**ac-005 — the prerequisite, by name.** The kind-aliasing fix and its guard tests are 02a's; this spec depends on them and does not re-assert a foreign module's contract.

**ac-006 — the repo's executable example.** molrs has no `regressions/` tree; the doctest is the public-API example that a rename would otherwise break invisibly, and it is where the visible `inherit_frag_ids` step is shown.

**ac-007 / ac-008 — documentation.** The stale CLAUDE.md line is corrected by the implementation; the rules amendment goes through `/mol:note`.
