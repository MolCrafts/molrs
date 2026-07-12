# molrs — Evolving Decisions

Short-lived working notes captured by `/mol:note`. Stable entries are
promoted into `CLAUDE.md` and removed from here.

Format per entry:

```
## YYYY-MM-DD — <topic>
**Decision:** <one-liner>
**Why:** <motivation — constraint, incident, experiment result>
**Status:** provisional | hardening | promoted (→ CLAUDE.md §section)
```

Run `/mol:note sweep` monthly to surface stale entries (> 90 days without
status change) and conflicts with `CLAUDE.md`.

---

## 2026-06-19 — GAFF: AmberTools-only, no native molrs GAFF
**Decision:** GAFF support is **AmberTools (antechamber) delegation only** — no
native molrs GAFF typifier, `gaff.dat` parser, or clean-room typing. The
`gaff-typifier-redesign` spec (and the `gaff-typifier-01..05` chain it superseded)
is dropped; do not re-propose a native GAFF path. The `ff/typifier/estimate` GAFF
*empirical formulas* (Badger / Wang2004 Eq.5 / parmchk2 substitution) are
**unaffected** — that is a general missing-parameter estimator, not GAFF typing.
**Why:** Trustworthy native GAFF typing would need a clean-room reimplementation of
antechamber's GPL `atomtype.c` ruleset + a vendored `gaff.dat` + an
antechamber-parity corpus. AmberTools already produces both the atom types and
AM1-BCC charges authoritatively; reimplementing it in molrs is not worth the
maintenance + correctness burden.
**Status:** provisional

## 2026-07-12 — petgraph is GONE: molrs has zero petgraph dependency
**Decision:** petgraph is removed entirely (`molrs/Cargo.toml`: dropped the
`petgraph` optional dep and `dep:petgraph` from the `smiles` feature). molrs now
has **no petgraph dependency at all**.
**Why:** It had become a **dead dependency**. The 2026-06-12 note below kept it on
the premise that "petgraph does real work: `subgraph_isomorphisms_iter` (VF2) powers
SMARTS substructure matching". That premise is **false in the current code**: the
SMARTS matcher (`core/chem/smarts/matcher.rs`, moving to `perceive/smarts/`) is a
**hand-rolled** backtracking VF2-style matcher, a semantics-only port of RDKit's
`SubstructMatch.cpp`. At some point after that note the VF2 was reimplemented and
the dep was simply left behind. Verified before removal: `use petgraph` / `petgraph::`
appear **zero** times in every `.rs` across molrs, molrs-python, molrs-cxxapi,
molrs-ffi, molrs-wasm, molrs-capi, tests, benches and examples — the only hits are
comments *asserting petgraph is not used*, plus a grep-gate test
(`tests/compute/hbond.rs::hbond_source_has_no_petgraph`). `cargo check --features smiles`
passes without it; `cargo tree -i petgraph` no longer resolves.
Also corrected the two CLAUDE.md claims this invalidated (the `smiles` feature no
longer "pulls in petgraph"; SMARTS lives in `core/chem/smarts/`, not `io/smiles/`).
**Status:** stable
**Supersedes:** the 2026-06-12 decision below, whose "do NOT remove it" rests on the
now-false VF2 premise. Kept for history.

## 2026-06-12 — petgraph: removed from molrs-core, KEPT in molrs-io (VF2) — SUPERSEDED 2026-07-12
**Decision:** `molrs-core` no longer depends on petgraph (spec `core-drop-petgraph`,
commit eedc1e6): `Topology`/`topo_distances`/`chem::rings` run on native MolGraph
adjacency. `molrs-io` **keeps** petgraph (feature-gated behind `smiles`) — do NOT
remove it; do NOT move VF2 into molrs-core/molgraph.
**Why:** The two uses are opposite. In core, petgraph was a redundant adjacency
wrapper over a graph MolGraph already holds; the algorithms (path enumeration,
BFS, flood-fill, Horton SSSR) were already hand-rolled — removing it was ~free and
slimmed the foundation crate (topo_distances also got faster, 100k 7.61ms→4.36ms).
In molrs-io, petgraph does real work: `subgraph_isomorphisms_iter` (VF2) powers
SMARTS substructure matching (validated vs RDKit), plus the weighted `UnGraph<N,E>`
containers for the target + query pattern. Re-implementing VF2 = ~500–800 lines of
correctness-critical backtracking we'd self-maintain, for a dep that's mature,
MIT/Apache, and only pulled when `smiles` is enabled. Moving VF2 into molgraph is
wrong layer (core uses no VF2; query semantics don't fit the domain-agnostic graph).
Revisit ONLY under a hard constraint (WASM size budget, zero-dep policy, petgraph
deprecation).
**Status:** SUPERSEDED — the VF2 was later hand-rolled (RDKit `SubstructMatch.cpp`
port) and the dep was left behind dead. See the 2026-07-12 entry above.

## 2026-05-28 — BLAS/LAPACK backend selection is the binary's job, not molrs's

**Decision:** `molrs-core/Cargo.toml` keeps `ndarray-linalg = "0.18"` with
no backend feature pre-selected (`openblas-system` / `netlib-static` /
`intel-mkl-*` / etc.). Picking a backend is the responsibility of the
top-level binary that consumes molrs (test runner, downstream app), not
of molrs-core itself.

**Why:**
- ndarray-linalg README, verbatim: "If you are creating a library
  depending on this crate, we encourage you not to link any backend."
  Cargo features are additive — if molrs-core picks `openblas-system`,
  every downstream is forced onto OpenBLAS forever.
- ndarray-linalg 0.18 backends: `openblas-{system,static}`,
  `netlib-{system,static}`, `intel-mkl-*` variants. **No `accelerate`
  feature**; Apple Silicon + Accelerate.framework is not officially
  supported by ndarray-linalg in this version.
- Consequence: `cargo test --all-features` on a fresh checkout will
  fail to link with `Undefined symbols: _cblas_sgemv, _dgetrf_, ...`
  unless the developer provides a backend externally.

**How to actually run `--all-features` tests locally:**

Either (a) the canonical `blas-src` / `lapack-src` dev-dependency
pattern in the test crate, e.g. in `molrs-core/Cargo.toml`:

```toml
[dev-dependencies]
openblas-src = { version = "0.10", features = ["system"] }
```

plus `#[cfg(test)] extern crate openblas_src;` at the top of
`molrs-core/src/lib.rs`, plus `brew install openblas` (it provides
CBLAS, unlike `brew install lapack` which is Fortran-ABI only).

Or (b) opt-in via CLI on the developer's machine without touching
Cargo.toml — but cargo doesn't have a clean per-invocation override
for downstream features; (a) is the canonical path.

**Action item (not blocking):** the `cargo test --all-features` line in
`CLAUDE.md` is misleading because it won't run on a clean checkout. We
should either drop `--all-features` from CLAUDE.md's quick-start, or
adopt option (a) and document `brew install openblas` as prerequisite.

**Status:** provisional — captured during `frame-block-subclass` impl
when the user's local `cargo test --all-features` couldn't link;
unrelated to that spec; not blocking.

## 2026-05-13 — Frame is pure `HashMap<String, Block>`, no Grid special case

**Decision:** Remove `Grid` (`grid.rs`) and `UniformGridField` (`field.rs`)
from `molrs-core`. Frame stores only `HashMap<String, Block>` + `meta` +
`SimBox`. No grid-specific methods on Frame or Block. Grid semantics belong at
the I/O boundary (CHGCAR/Cube reader → Block columns + spatial metadata in
Frame meta), not as privileged types in the core data model.

**Why:**
- `Grid` is just `{named arrays} + {dim, origin, cell, pbc}` — named arrays
  are Block columns, spatial metadata is Frame meta. No new type needed.
- `UniformGridField` duplicates Grid's spatial definition and `FieldEncoding`
  had only one variant — premature abstraction with no callers.
- Frame having `HashMap<String, Grid>` alongside `HashMap<String, Block>` is
  a special case that complicates the API (7 extra methods, separate Zarr
  code path, separate wasm index path, separate Python class).

**Status:** provisional
