# Specs

Live specs only. A spec is deleted when its work has landed — the record of what
shipped is the git history and `.claude/notes/release.md`, not a stack of closed
spec files.

| Spec | State |
|------|-------|
| [cgsmiles-01e-python-ir](cgsmiles-01e-python-ir.md) — Python `molrs.io.CGSmilesIR` and its read-only nested records; class-level stub-parity guard | approved |
| [cgsmiles-02a-fragment-core](cgsmiles-02a-fragment-core.md) — `core::Fragment` newtype with `PortKind` ports and `frag_id`; by-name kind resolution fix; `mapping.rs` deleted | approved |
| [cgsmiles-02b-to-fragment](cgsmiles-02b-to-fragment.md) — `CGSmilesIR::to_fragment`: instance-free templates with hydrogen handles and ports | approved |
| [cgsmiles-02c-conformer-fragment](cgsmiles-02c-conformer-fragment.md) — `ElementGraph` trait; generic `Conformer::generate` over `Atomistic` and `Fragment` | approved |
| [cgsmiles-02d-python-fragment](cgsmiles-02d-python-fragment.md) — Python `molrs.Fragment`, `views.Port`, `Conformer.generate(Fragment)`, `CGSmilesIR.to_fragment()`; shared `from_core` helper | approved |
| [cgsmiles-03-release](cgsmiles-03-release.md) — release molrs 0.15.0: version literals, release notes, full gate, tag, Publish | approved |

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
