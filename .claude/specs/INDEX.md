# Specs

Live specs only. A spec is deleted when its work has landed — the record of what
shipped is the git history and `.claude/notes/release.md`, not a stack of closed
spec files.

| Spec | State |
|------|-------|
| [frame-meta-dict-parity-01-ordered](frame-meta-dict-parity-01-ordered.md) — MetaMap HashMap→IndexMap, shift_remove, opaque iterators, order-preserving serde wire, Zarr round-trip contract | approved |
| [frame-meta-dict-parity-02-binder-order](frame-meta-dict-parity-02-binder-order.md) — delete the three binder alphabetical sorts, popitem LIFO, four enumeration surfaces agree | approved |
| [frame-meta-dict-parity-03-untyped-write](frame-meta-dict-parity-03-untyped-write.md) — delete the typed-slot rule; a plain write takes the value's own dtype; dtype is durable on two named paths | approved |
| [frame-meta-dict-parity-04-dict-views](frame-meta-dict-parity-04-dict-views.md) — live collections.abc views, one key-acceptance rule, read/write borrow split, delete the clone-per-read | approved |
| [frame-meta-dict-parity-05-document](frame-meta-dict-parity-05-document.md) — MetaDocument + tuples: every frame.meta door hands back a frozen value | approved |
| [frame-meta-dict-parity-07-sequence](frame-meta-dict-parity-07-sequence.md) — sequence meta-key maps follow declaration order; other BTreeMaps stay sorted | approved |
| [cgsmiles-03-release](cgsmiles-03-release.md) — release molrs 0.15.0: version literals, release notes, full gate, tag, Publish | approved |
| [opls-gromacs-01-gromacs-io](opls-gromacs-01-gromacs-io.md) — GROMACS force-field reader/writer are directive-only and correct (funct map, comb-rule 3, RB↔Fourier once, refusals by name); free doors deleted | code-complete (chain gate owed) |
| [opls-gromacs-02-table](opls-gromacs-02-table.md) — OPLS-AA table regenerated from pinned GROMACS v2026.3 oplsaa.ff (classes = bond_type, geometric mixing, provenance); rules split into OplsRuleRow | code-complete (chain gate owed) |
| [opls-gromacs-03-rules](opls-gromacs-03-rules.md) — molrs-owned Daylight OPLS typing rules, aromaticity on a private copy, pairwise override dominance, opls_150/178; Python skip_directives; chain gate | in-progress |

The `frame-meta-dict-parity-*` links are one chain (`frame.meta` becomes a
Python dict in behaviour, revised 2026-09-22). They land on the unreleased
0.15 tree. No version bump, no pin window, no migration guide, and no
release-notes link. Chain order: 01 → 02 → 04 (02 and 04 require 01; 04 also
requires 02 and 03); 03 and 05 do not change enumeration order. `07-sequence`
is the meta-key maps in `io/zarr/sequence.rs` (01 routed them). There is no
`06-release-notes`.

The nine `system-forcefield-*` links (closed 2026-09-25; record: git history and `.claude/notes/release.md` § v0.15.0) were one chain (2026-09-25, operator-ruled): after typing, the system force field is the union of the parameters of every typified molecule, built only through `def_style().def_type()`; ForceField depends on no Frame/Atomistic; the typifier base owns the output and typifiers only implement `match`; coefficient writing is one label-driven capability. Chain order 01 → 09, linear. Shared rules for every link of this chain: The chain lands before `cgsmiles-03-release` tags 0.15.0; molpy is broken on this tree from 01 on, by design, and follows in its own chain after the tag.

- Rust first, FFI last (operator ruling 2026-09-25; replaces the per-link binder seam rule): links 07–08 change only `molrs/`; the binder crates may not compile against the tree until 09 brings all five (molrs-python, molrs-wasm, molrs-capi, molrs-ffi, molrs-cxxapi) to the settled API.
- Verification (operator ruling 2026-09-25): every task of links 07–09 runs only `cargo mrs-test [-- <module filter>]` (one build configuration; an edit rebuilds in ~6 s, the suite runs in ~5 s). Never a second `CARGO_TARGET_DIR`, worktree, copied crate or hand-typed feature string; no clippy, fmt check, doctest, rustdoc or binder build inside the chain. All of that happens once, at the chain end in 09: update the binders, commit (the pre-commit hook runs rustfmt + clippy), then `prek run --all-files --hook-stage pre-push`, which discharges the full-gate criterion of every link; the links then close together.
- Definitions vs edits: `def_style` / `def_type` / `def_type_at` define. `set_type_param`, `set_type_str_param`, `remove_type` and `remove_style`, reached through `get_style_mut`, edit an existing definition and sit outside the conflict rule; `rename_type` is the one edit that can land on an existing name and carries the collision rule (02).
- Constitution: no `.claude/notes/law.md` exists; governing rules are CLAUDE.md § Design preferences / § Testing Rules, `.claude/notes/architecture-rules.md` (io and ff never name each other), notes.md § Binding-surface symmetry.
- English only.
- No `regressions/` tree in molrs; no committed public-API script.
- No A/B harness (operator ruling 2026-09-25): invariance is guarded by the existing unit tests of the touched modules, which run unedited. A/B tasks and criteria in the remaining links are dropped.
- Domain basis: links 01–06 declare none; 07 cites MMFF (Halgren 1996) and UFF (Rappé 1992); 08 cites GAFF (Wang 2004).
- Release order: public API changes on the unreleased 0.15 tree; the chain lands before cgsmiles-03-release tags 0.15.0 (operator decided backmap primitives fold into 0.15.0 too). The molpy half starts after the tag; molpy is broken on this tree from 01 on, by design.
- Stage experimental: moved APIs are deleted outright (no deprecation shims), C symbols included.

The ten `cgsmiles-*` links are one chain (molrs half of the CG→all-atom backmapping plan, 2026-09-21); they land in chain order and 03 ships 0.15.0, after which molpy's `backmap-*` chain may start.

## release-0-14 chain — closed 2026-09-20

The five release specs (08 ship molrs, 09 molpy rebase, 10 molpy mirror,
11 molpy docs, 12 joint smoke) are closed. Everything in them that is code or
docs has landed on molrs `chore/test-orthogonalization` and molpy
`ci/precommit-uv-parity`; what remains is release mechanics, kept as the
manual checklist in `.claude/notes/release.md` § v0.14.0.

- **08** merge to `master`, tag `v0.14.0`, publish (crates.io / npm / PyPI),
  swap the molnex `.dev1` wheel — operator-run, see release.md.
- **09** molpy branch and pin bump — operator-run; the pin
  `molcrafts-molrs>=0.14.0,<0.15` is already in molpy's pyproject.
- **10** the shared formats (pdb, top, amber, lammps data / molecule / log,
  force-field xml) and Box geometry are molrs-backed and `molpy.md` re-exports
  `molrs.md` by identity. The "bit-parity on a committed corpus" acceptance
  was dropped with the corpus: the test suites are unit-only
  (`.claude/notes/testing.md`). Still open, as its own public-API decision:
  molpy's callable `compute.base.Compute` shells versus the molrs `Compute`
  Protocol (`compute(...)`) — the verb-unification question
  (assemble / build / apply / run / typify / compute).
- **11** typifier spellings fixed, user-facing "molrs" wording replaced,
  `docs/getting-started/migration-0-14.md` written and in the nav. molpy has
  no `docs/zh/` tree, so the bilingual criterion is void. The spelling and
  parity *gates* it asked for are not written: source-text gates are not unit
  tests.
- **12** the full-import and warning-scope gates are not unit tests and are
  not written; the molnex smoke and the molpy tag are release mechanics.

The three `opls-gromacs-*` links are one chain (2026-09-25, operator-ruled): OPLS-AA follows GROMACS `share/top/oplsaa.ff` (v2026.3, commit 42105e46…) for types, charges, classes, LJ and bonded parameters; the typing SMARTS rules are molrs-owned Daylight SMARTS with explicit bonds and aromaticity perceived before matching; CL&P does not land in molrs (molpy deletes its typifiers in its own chain). Chain order 01 → 03, linear. Shared rules: each task verifies with `cargo mrs-test [-- filter]` only; no A/B harness, no second build cache; Rust first — the one binder change (molrs-python) is in 03; links 01–02 are never committed on their own; one full gate at the end of 03 discharges every link's full-gate criterion. English only; stage experimental.
