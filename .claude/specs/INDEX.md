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

The `frame-meta-dict-parity-*` links are one chain (`frame.meta` becomes a
Python dict in behaviour, revised 2026-09-22). They land on the unreleased
0.15 tree. No version bump, no pin window, no migration guide, and no
release-notes link. Chain order: 01 → 02 → 04 (02 and 04 require 01; 04 also
requires 02 and 03); 03 and 05 do not change enumeration order. `07-sequence`
is the meta-key maps in `io/zarr/sequence.rs` (01 routed them). There is no
`06-release-notes`.

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
