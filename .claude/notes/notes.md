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

## 2026-07-12 — BCC bond typing: two things that are NOT what we assumed
**1. Kekulé choice IS charge-relevant end-to-end — my earlier "spend no budget there" was WRONG.**
It is true that the *table* is Kekulé-invariant: BCCPARM holds identical deltas for bond
types 7, 8 and 10 (7-vs-10: 25/25 keys identical; 8-vs-10: 15/15). I concluded from that
the Kekulé phase could not affect charges. **That conclusion is false**, because the Kekulé
structure also determines the **ATOM** types: imidazolium has two degenerate (equal-penalty)
Kekulé structures that place the N=C double bond on *different* nitrogens, and those two N
then get different BCC atom types → different charges. So the tie-break is load-bearing and
had to be calibrated (`min endpoint asc, max endpoint desc`, doubles first) to reproduce
AmberTools on all 9 aromatic oracle molecules. Table-invariance ⇏ pipeline-invariance.

**2. Aromatic bond types are NOT resolved from the `ar` precursor — they are re-perceived.**
antechamber does not walk adjacency to turn the SYBYL `ar` token (type 10) into 7/8. It
**re-derives the Kekulé structure from scratch** via the APS.DAT valence-state penalty
search (verified: flipping benzene's Kekulé phase in the input SDF does not move the
output). The sequential precursor rule in `finalize()` gives benzene the *opposite* phase
from the oracle. molrs therefore implements penalty-minimising kekulization over the
aromatic subsystem using antechamber's own APS numbers — not a precursor resolution.
**Status:** stable

## 2026-07-12 — BCC type 6: nitrate divergence is deliberate (owner decision)
**Decision:** molrs fixes antechamber's input-order dependence in `bondtype.c` part3 and
accepts two divergences. Rule shipped: *N of degree 2–3, bonded to a terminal O/S, NOT a
conjugated centre, carrying another terminal O/S* → type 6. Exhaustive neighbour scan,
symmetric in endpoints.

| molecule | antechamber | molrs | |
|---|---|---|---|
| nitrite / nitromethane / nitrobenzene / TMAO | 6,6 / 9,9 / 9,9 / 9 | same | match |
| **nitrate** | 6, 9, 9 | **9, 9, 9** | divergence — symmetry fix |
| **pyridine-N-oxide** | 6 | **9** | divergence — branch-B artifact |

**Why:** antechamber's part3 has an unbraced `break` (branch A stops at the first neighbour
→ depends on input bond order) and an unconditional assignment (branch B skips the check
entirely). Measured: nitrate's three N–O get 6/9/9, so two *topologically identical* O⁻ get
different types → final charges −0.6997 / −0.4180 / −0.4180, a **0.28 e break across three
equivalent oxygens**. Charge equivalencing cannot save it (`-eq` averages AM1 charges
*before* BCC; the damage is in the bond types applied *after*). Both divergences were shown
to be pure coin-flips: writing pyridine-N-oxide's bond as `O-N` gives 6, as `N-O` gives 9 —
same molecule, same file format.
**The conjugated gate is what preserves nitromethane** (without it an exhaustive scan types
its N–O as 6, a +0.1317 vs −0.1500 swing); it is the source's own stated intent
(`bondtype.c:312` — *"index6 >= 2 makes NO2 bonds are delocalized bonds"*).
**Status:** stable — do not "restore parity" on nitrate; it is a deliberate correctness win.

## 2026-07-12 — antechamber reimplementation: licensing posture (owner decision)
**Decision (project owner, 2026-07-12):** Proceed with reimplementing antechamber's
perception algorithms (`bondtype.c` → spec 03, `equatom.c` → spec 04, `atomtype.c` →
specs 05/06) in molrs. Stated basis: **学习用途 (educational/research use), and the
AmberTools developers have given permission to read their code.**
**Context this decision was taken against (recorded so it is not lost, not re-litigated):**
- molrs is **BSD-3-Clause** (`Cargo.toml:26`); AmberTools/antechamber is **GPL-3**.
- The chem-perceive work is **source-derived, not clean-room**: the algorithm rules
  (bondtype.c `finalize()` part2/4/5 ordering, equatom.c Eq.(I) path-score, the
  `-eq` levels) were obtained by reading the antechamber C source.
- molrs also now ships ~27k lines of Rust generated from AmberTools' `.DAT`/`.DEF`
  parameter tables (spec 02).
- This was raised before spec 03 began and explicitly waived by the owner.
**Status:** stable — do NOT re-raise per-spec. If molrs's distribution posture ever
changes (e.g. a commercial redistribution question), revisit then, not before.

## 2026-07-12 — ONE SSSR in the tree: `core::system::topology::Topology::find_rings`
**Decision:** There is exactly **one** ring-perception algorithm in molrs, and it lives in
`core::system::topology`. `perceive::rings::find_rings(&Atomistic) -> RingInfo` is now a
thin **handle-keyed chemistry decoration** over it: it projects the graph into core's index
space, calls `Topology::find_rings`, and lifts `usize` back to `AtomId`/`BondId`. Do NOT
add a second SSSR; if you need rings, go through one of these two.
**Why:** The tree carried **two independent Horton-style SSSR implementations** — the
index-keyed one in core (used by `perceive::rotatable` and molrs-wasm) and a separate
handle-keyed one in `perceive::rings` (used by aromaticity, SMARTS, MMFF, AM1-BCC,
conformer, molrs-python). Before merging them I verified empirically that they agree on ring
**membership**, not merely ring sizes — identical canonicalised ring atom-sets and per-atom
ring counts across benzene, cyclopropane, naphthalene (fused), **cubane** (8v/12e — 5 SSSR
rings chosen from 6 faces, the classic ambiguous case), bicyclo[2.2.2], spiro, an acyclic
chain, and an adamantane-like cage. That mattered: SSSR is **not unique**, so if the two had
disagreed, delegating would have silently changed aromaticity / MMFF / SMARTS results.
Consolidation removed **193 lines** and the full suite stayed exactly on snapshot
(1746 passing), which independently corroborates the equivalence — every one of those
consumers reads `find_rings` and none moved.
**Status:** stable

## 2026-07-12 — rustdoc is a THIRD build gate, not a subset of fmt/clippy/check
**Decision:** `$META.build.check` and `.pre-commit-config.yaml` now both run
`RUSTDOCFLAGS='-D warnings' cargo doc --no-deps -p molcrafts-molrs --all-features`.
**Why:** A broken intra-doc link (`[`chem`]` in `core/mod.rs`, left behind when the module
moved to `perceive`) passed `cargo fmt`, `cargo clippy -D warnings` AND `cargo check`, and
was only caught by an out-of-band `cargo doc` run. Public-doc-links-to-private-item has the
same property. Turning the gate on required clearing **37 pre-existing rustdoc errors** across
20 files first (redundant explicit link targets, links to `Grid`/`Frame`/`FrameView` that had
moved or been deleted, 7 public→private links, and a malformed doctest fence in
`io/data/lammps_data.rs` whose ```no_run block had silently never compiled).
**Status:** stable

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

## 2026-07-13 — charge equivalencing: three corrections worth keeping
**1. It is PATH-SCORE, not automorphism orbits — and the witness is acetate.**
`equatom.c` scores every simple path (`Score = Σ (j+1)·0.11 + Z_j·0.08`), sorts, and
compares EXACTLY. Orbits are a strict subset of path-score classes. The concrete
divergence is NOT the theoretical order-blindness (a sweep of every valence-legal
fragment ≤5 heavy atoms found no ordinary molecule that merges two non-automorphic
atoms that way) — it is **bond-order / formal-charge blindness**, and acetate is in
the oracle: raw sqm −0.595/−0.597 → antechamber −0.596/−0.596, MERGED. But
`core/system/graph_hash.rs` folds bond order (line 143) and formal charge into its
colours, so an orbit engine would SPLIT the Kekulé C=O from the C–O⁻ and ship a
symmetry-broken carboxylate. Do not use graph_hash as the class engine.

**2. "Conserves total charge bitwise" is mathematically impossible.** A class mean is a
rounded f64, so `n·fl(Σq/n) ≠ Σq` unless n is a power of two. It also CONTRADICTS
"class members carry identical bits". Measured: 19/37 molecules happen to be bitwise
equal, 18/37 drift, worst 3.7e-16. antechamber carries the identical residual and does
not renormalize. The honest contract: total conserved to ULP scale, PLUS the bit-exact
half that does hold (singletons keep their bits; class members share bits).

**3. `scorepath()` scores every simple path, not just paths to terminal atoms.** It emits
once per DFS node (including the trivial one-atom path), and a 6-coordinate atom
terminates no scored path (the `con[6]` quirk). The "paths to terminal atoms" reading
gives a different path count and does not reproduce the oracle.

`-eq 2` is implemented (E/Z refinement, strictly FINER than `-eq 1`; verified on methyl
methacrylate: `-eq 2` keeps the cis/trans vinyl H apart at 0.139/0.125 where `-eq 1`
merges to 0.132). Note `ATOM_EQU.TYPE` gates E/Z on GAFF type names molrs never assigns,
so the gate is translated natively (C=C doubles + amide C–N).

Gotcha for oracle regeneration: `-pf y` DELETES `ANTECHAMBER_AM1BCC_PRE.AC`. Use `-pf n`.
**Status:** stable
