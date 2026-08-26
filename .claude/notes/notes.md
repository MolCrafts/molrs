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

## 2026-08-26 — development tools track latest; other-platform UB out of scope
**Decision:** rustc/clippy/rustfmt = `rust-toolchain.toml` `channel = "stable"`
and CI `dtolnay/rust-toolchain@stable`. wasm-opt = latest binaryen GitHub
release. uv action = latest major. pre-commit-hooks = latest tag. Never pin a
compiler/linter minor to hide CI drift. Other-platform UB (Windows/macOS-only)
is out of scope until 0.14 lands on `dev`.
**Why:** pinning rustc 1.96 made local prek green while GHA clippy on 1.98 was
red; the next rustc bump would just repeat it.
**Status:** active

## 2026-08-26 — identity scalar is Idx = u64; storage widths are preserved
**Decision:** `molrs::types::Idx = u64` replaces `U = u32`. Identity columns
(`id`, `mol_id`, `type_id`, `res_id`, `atomi`/`atomj`/`atomk`/`atoml`) and
schema `UInt` (`bond_type`, `bond_number`) store as `Column::UInt`. Every other
numpy/IO width (`f16`/`f32`/`i8`/`i16`/`i64`/`u8`/`u16`/`u32`/`c64`/`c128`)
keeps the width it arrived with — no narrowing-cast insert. WASM domain-uint
accessors take/return `BigUint64Array`; JS method names stay `*U32` for this
cut. `U` is also uranium: rename through the compiler, never a text replace.
**Why:** an identifier that wraps is not an identifier; a column that arrived
as `f32` must leave as `f32`. 0.14 is the last chance to change the identity
width before the minor line freezes.
**Status:** active (0.14)

## 2026-08-25 — public record API is Record, not MolRec
**Decision:** Public names are `molrs.Record` / `molrs::Record`, `Record.read` /
`Record.write`, `Trajectory.read` / `Trajectory.write`, cxxapi `write_frame` /
`read_first_frame`. No deprecated aliases. Internal `store::record::MolRec`,
`RECORD_FORMAT_NAME = "molrec"`, and the `io::store::zarr` adapter keep their
technical names.
**Why:** Public API names the object; the backend is not yet a caller-chosen
format (release-0-14-13). Cross-crate: Atomiverse checkpoint I/O must switch
to `write_frame` / `read_first_frame`.
**Status:** active

## 2026-08-25 — one Potential concept; md LJCut vs ff lj/cut merge deferred
**Decision:** The md force seam is the single `ff::potential::Potential` +
`Potentials` concept (owner ruling: no `ForceProvider`, no `PairForce`, no
`ForceStack`, no `PotentialsForce`). The nonbond case is served by the trait's
default-no-op `set_pairs(&[SkinPair], &SimBox)`: the loop (integrator) owns the
`VerletSkin`, runs its policy, and feeds fresh pairs after each rebuild —
neighbour machinery never lives in a potential. Unit conversion has exactly one
home: `Potentials::set_energy_scale` (factor from `md::units::
preset_energy_to_md`/`energy_to_md`); members always share one unit system.
**Closed 2026-08-25 by release-0-14-14:** one `LJCut` in `ff::potential::pair`
(Loop vs Compiled pair source), `VerletSkin::pairs_at` is the only MIC site,
`set_pairs` is gone, and `kspace` is not a ForceField category — PME registers
as pair style `coul/long/pme`. The `ff/potential/kspace` module stays as the
FFT compilation-unit boundary.
**Status:** closed (→ release-0-14-14)

## 2026-07-29 — pre-commit whole-tree gates need always_run
**Decision:** `cargo-fmt` and `cargo-clippy` in `.pre-commit-config.yaml` set `always_run: true` (same as the pre-push test hooks).
**Why:** CI rustfmt/clippy are whole-tree gates. Without `always_run`, prek skips them when the staged set has no "matching" files — rustfmt/clippy failures on `dev` (while_let_loop, too_many_arguments, region formatting) reached GitHub Actions despite the hooks being declared for pre-commit.
**Status:** active

## 2026-07-23 — private extension molrs._lib; docs CI fix
**Decision:** Rename compiled extension `molrs.molrs` → `molrs._lib` (public vs private). PyO3 classes use `module = "molrs"`. Docs: force_inspection + analysis attrs for griffe; Cloudflare Pages builds fixed.
**Status:** active (v0.9.2)

---

## 2026-07-22 — release order: molrs before molpy; master+tag; no scripts
**Decision:** Co-release is always molrs first, then molpy. Version landings on master require a `vX.Y.Z` tag. No pin-parity automation scripts — agents manually verify. Monorepo merge discarded.
**Status:** promoted (→ `.claude/notes/release.md` + CLAUDE.md § Release before molpy)

## 2026-07-22 — closed vacuous-green-tests + test-subset-assertions
**Decision:** both specs closed; gates already green, code already computed partitions / `#[ignore]` for molpy OPLS.
**Cleanup (tester audit):** trajectory soft `if !exists { return }` → `common::require_fixture`; gate shape-2 expanded; `testing.md` rewritten for single-crate + iron laws.
**Status:** promoted (→ testing.md Iron laws; architecture_gate)
## 2026-07-13 — GAFF decision REVERSED: molrs types GAFF/GAFF2 natively
**Decision (owner-authorised):** the 2026-06-19 "GAFF = AmberTools-only" decision below is
**REVERSED**. molrs now assigns GAFF, GAFF2, AMBER and SYBYL atom types natively, all
**37/37** against `antechamber -at {gaff,gaff2,amber,sybyl}`.
**Why the situation changed:** that decision rested on "a native GAFF typifier would need a
clean-room reimplementation of antechamber's GPL `atomtype.c` ruleset + a vendored
`gaff.dat` + an antechamber-parity corpus". All three now exist for other reasons: the ATD
rule engine was built for AM1-BCC (spec 05), the `.DEF` tables are generated Rust (spec 02),
and the 37-molecule antechamber oracle is the parity corpus. **Adding GAFF was a table, not
code** — `AtdParameterSet::{Gff,Gff2,Amber,Sybyl}` were already wired. The licensing concern
was separately waived by the owner (see the 2026-07-12 entry).
**Three defects the GAFF columns exposed** (all invisible to BCC/ABCG2/GAS):
1. The facts layer marked EVERY aromatic atom `AR1` and never wrote `AR2`/`AR3`/`AR4`.
   BCC/ABCG2/GAS only ever spell the property as `[AR1.AR2]` — an OR — so they are
   structurally blind to it. GFF discriminates `ca [AR1]` vs `cc [sb,db,AR2]`.
2. The terminal catch-all row (`ATD DU &` / `ATD ANY &`) of all seven `.DEF` files was
   dropped at generation. AMBER is the first column where an atom actually falls through,
   and antechamber's answer genuinely IS `DU` (nitromethane's nitro O, DMSO's S).
3. `cd`/`nd`/`cf`/`ch` are **not rows in ATOMTYPE_GFF.DEF at all** — antechamber renames half
   of every conjugated system in a second pass the `.DEF` never describes. The pairing is
   upstream data (`PARMCHK.DAT` `equivalent_flag`), so it is a table column, not an engine
   invention. The pass: 2-colour the CONJUGATED subgraph; a single bond keeps the letter, a
   double/triple bond flips it. NOT positional (pyridone is `cc cd cd cc`, not alternating),
   NOT gated on aromaticity (benzoquinone has no aromatic bond), and it propagates a COLOUR,
   not a name (vinylacetylene holds a `ce` and a `cg` in one component).
**Trap worth remembering:** `alternate` must key on PARMCHK's `equivalent_flag` COLUMN, never
on the type's spelling — `ATOMTYPE_AMBER.DEF` has real `CC`/`CD` rows (parm94 histidine
carbons, flag 0), and `ATOMTYPE_GAS.DEF`'s `cg` is a guanidinium carbon, not GAFF's sp
carbon. Both would be mis-paired by a name-keyed generator; the column is GAFF-namespaced.
**Status:** stable — supersedes the entry below.

## 2026-06-19 — GAFF: AmberTools-only, no native molrs GAFF — SUPERSEDED 2026-07-13
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

## 2026-07-13 — perception writes `bcc_bond_type`, NEVER `keys::TYPE`
**Decision:** `Perceive::find_bond_types` writes the perceived antechamber bond type to a
dedicated `BCC_BOND_TYPE` ("bcc_bond_type") prop. `keys::TYPE` on a BOND belongs to the
caller — it holds the force field's bond-type NAME (a String), which is what `to_frame` puts
in the bonds block and what every bonded kernel resolves by. Perception must neither read nor
write it. (`AtdTypifier::typify` writing the ATOM type to `keys::TYPE` as a String IS correct
— that is a typifier's job; `types_of()` exists for callers who want types without mutation.)
**Why:** it was writing an **i32** into the same column the force field needs as a **String**.
A component column is typed on first write and molrs correctly refuses to coerce — so on a
molecule that already carried FF bond-type names, **the write silently failed** (the error was
swallowed by a `let _ =`). `gaff_forcefield` (spec 09) had to copy the molecule into a fresh
graph stripped of that column just to build a force field; that workaround (`frame_ready_copy`)
is now deleted, and it had also been silently dropping angles/dihedrals and re-minting handles.
This is the same class of bug spec 07 closed for charge models (ac-004: charges come out
bitwise identical whatever `keys::TYPE` holds) — perception had simply never been brought
under that rule.
**Convention (already implicit, now explicit):** `keys.rs` holds core canonical, molpy-synced
fields. Every *perceived fact* is a const in the `perceive` layer — `is_aromatic`, `is_in_ring`,
`n_rings`, `is_rotatable`, `stereo`, `equiv_class`, and now `bcc_bond_type`.
**Status:** stable

## 2026-07-14 — chem-perceive whole-chain acceptance: the LESSONS (not the conclusions)

Sixteen specs, each green on its own slice, and the chain had a 150 kcal/mol hole in it.
These are the transferable lessons, written as rules rather than as war stories.

**1. A test that selects its input cannot be trusted to have covered it.**
`generic_path_total_energy_matches_rdkit` asserted on `["e_ethane"]` — one of exactly TWO
fixtures whose MMFF charges are all zero, i.e. the ONE input class that structurally cannot
expose a missing electrostatic term. Ten fixtures sat on disk, unread, for a month. The rule
that falls out is not "review your fixture lists"; it is: **where a list can be
directory-scanned, it MUST be, and where a subset is meant, the subset must be a PREDICATE
evaluated on the molecule, not a list of names.** A list you can write by hand is a list you
can shorten by hand. `tests/architecture_gate.rs::no_test_asserts_on_a_subset_of_its_fixtures`
now enforces it, and it caught four survivors — including a partition (`IDENTICAL_FIXTURES`,
`N_FIXTURES`) that omits `e_caffeine` and `e_big`, both of which *do* carry a delocalized
nitrogen. Nothing had ever asserted that MMFF94s changes caffeine's energy.

**2. A wrong reason is worse than no reason — it is an alibi.**
Next to `["e_ethane"]` sat a comment blaming "stretch-bend + torsion eq-fallback label
resolution". It was false (stbn and torsion agreed to five decimals on every fixture), and it
misdirected every reader for a month. **"Not yet implemented" is not a reason to exclude a
fixture; it is a reason to fail.** A gate now refuses an excuse that sits next to the thing it
excuses.

**3. A grep finds spellings; a gate finds semantics.**
The `ParamSource` criterion judged a ctor by the *spelling* of its binding (`_tp`, with Rust's
leading underscore) and therefore missed `pme_ctor` and `pair_coul_cut_ctor`, which spell it
`_type_params` and ignore it just as completely — 8 violators, not the 6 the grep found. And
the spelling criterion is blind by construction to the deliberate version: a ctor that binds
`type_params` and never reads it. **Proven, not asserted**: injecting exactly that shape leaves
all four tests of the existing spelling gate GREEN, while
`architecture_gate::param_source_is_bidirectional_on_semantics_not_spelling` names it exactly.
Ask what the body *does*, never what the name *says*.

**4. Stage tests cannot see a chain.** Each of the seven stages was green. The chain was not:
the GAFF force field the chain builds declares **no electrostatic style at all**, so
`to_potentials` returns an energy with no Coulomb term in it — silently, for every molecule,
ionic ones included. No stage test could see it, because no test ran the chain to an *energy*.
The tell had been sitting in the tree the whole time: `SpecialBonds.coul = [0, 0, 1/1.2]`, a
1-4 Coulomb scale factor declared for a term that does not exist. **A constant nothing consumes
is the same smell as 4,065 XML rows nothing reads.** Both mean: someone declared an intention
and nothing checked it.

**5. Forward assertions are fooled by "added it, in the wrong place". Assert ABSENCE.**
Zero-charge molecules must get EXACTLY 0.0 electrostatic energy — not "small", because every
term has a factor of zero in it and any tolerance would be hiding something. Molecules with no
delocalized N must be BIT-IDENTICAL between MMFF94 and MMFF94s — otherwise a "they differ" test
can pass on a difference that does not exist. Benzene must HAVE impropers (it had zero, and the
oop energy of 0.0 is very nearly right for a planar ring, which is how it hid).

**6. A gate that has never been red is indistinguishable from no gate.**
Every gate in this acceptance was proven to bite: the defect it guards was injected, the gate
went red, the injection was reverted. Two of the injections were themselves *wrong* the first
time (a conformer test injected a dependence on `x` while the rotation was *about* x; a
`needs_equivalencing` patch never matched its target string) — and only the bite-proof
discovered it. **The bite-proof is not paperwork. It is the test of the test.**

**7. An acceptance that fixes what it finds is where the last defect hides.**
This one fixed nothing. Three gates land RED, each naming a real defect, each getting its own
spec. The thing that would have reported them must not be the thing that swallows them.

**Status:** stable

## 2026-08-04 — chem-perceive-15 gates landed (architecture_gate binary)

**Decision:** Whole-chain acceptance gates live in
`molrs/tests/architecture_gate.rs` (wired via `[[test]] name = "architecture_gate"`;
`autotests = false`). Structural ac-001..ac-004 + reverse ac-007 are executable;
ac-005/ac-006 are `#[ignore]` stubs pointing at cxxapi oracle / molrs-python.
**Subset-fixture trap (restate):** a hand-picked fixture list is how ethane alone
hid a missing MMFF electrostatics term for a month. Prefer directory scan or a
predicate; if a subset is intentional, the reason must be in the test body and
"not yet implemented" is a fail, not an exclude. Reverse gates assert ABSENCE
(exactly-0 ele energy; bit-identical 94/94s; equal symmetric charges; benzene
HAS impropers). Acceptance must not quietly fix production.
**Status:** active

<!-- mol:note:topic:binder-surface-symmetry -->
## [2026-08-10] 绑定面对称原则(neighborlist 链后定调)

门面(公开 API)质量优先于内部实现;内部走渐进重构,不追求一步到位,不阻塞发布。

**Rule**: Rust / Python / WASM 三个表面的邻居 API 必须保持对称——同名
(`NeighborList` 引擎 / `Neighbors` 表)、同形(build/update/neighbors +
Option 列语义)、同默认(`FULL`)。新增或改动任一绑定面时,先对照另外两面。

已知不对称(内部重构优先序):

1. **wasm `NeighborQuery` 对称门改期到 0.15**（2026-08-25）。删除不在选项内：
   in-tree consumers are `compute/hbond/detect.rs` (`from_columns` /
   `free_columns`, `QueryMode::CrossQuery`), `compute/rdf/mod.rs`,
   `compute/dynamics/van_hove.rs`, `ff/potential/soft.rs`. wasm 尚无消费者
   （facade-first），0.14 不补对称门、也不删引擎类型。
2. `LinkedCell` / `BruteForce` 别名仅为 molvis 链接暂留(默认 FULL,安全);
   molvis 迁移到引擎 API 后**删除**,不长期维护双门。
3. 其余路由项按需慢做:core SoA `update_columns`、`neighbors/mod.rs` 拆
   `table.rs`(纯移动)。`Compute::Args` 借用化已完成(2026-08-10)。

**Status:** active

<!-- mol:note:topic:md-experimental-ship-0.14 -->
## [2026-08-24] md 以 experimental 身份进 0.14.0（最终架构：one Potential + loop-owned neighbors + merged MD）— REWRITTEN 2026-08-25

**Decision:** `molrs.md` 随 0.14.0 发布，标记 experimental：import 时发
`FutureWarning`（"molrs.md is experimental in 0.14: APIs may change or be
removed in a future minor release."），顶层 `molrs/__init__.py` 为 PEP 562
惰性加载（`__getattr__`/`__dir__`）——`import molrs` 保持无警告。

**用户命名空间裁决（maintainer, 2026-08-25）：** 用户可见的拼写一律
`molpy.*`，`molrs` 不出现在用户侧；molpy re-export molrs 接口
（`molpy.md.Potential`、`molpy.md.MD`……与 molrs.md **同一对象**）。md 切片
已在本轮完成 verbatim re-export + 完整性测试；全表面镜像是 0.14 的
tracked work item。用户可见 docstring 示例一律拼 `molpy.md`；engine 内部、
molrs 仓测试与 `regressions/` 保留 `import molrs`。

**ONE Potential concept（owner ruling）：** 唯一概念是
`ff::potential::Potential`（产出 energy+forces；类别
nonbond/bond/angle/dihedral/improper）。成员：`LJCut`（LAMMPS
`pair_style lj/cut`，md 的 nonbond kernel）、`Potentials`
（composite，自身也是 Potential，合并成员结果）、Python 侧抽象基类
`molrs.md.Potential`（**subclassable ABC**，maintainer 裁决：不再是
`Potential(f)` callable 包装——用户 `class MyPotential(molpy.md.Potential):`
override `calc_energy_forces(self, pos (N,3)) -> (energy, forces)`；基类方法
raise NotImplementedError；Rust 适配器 `SubclassPotential` 持实例引用、GIL
下调 override，异常经 ErrSlot 以原始 Python 异常上抛；实例是共享引用而非
move——`Potentials`/`VerletSkin` 仍是 move 语义，复用 raise ValueError）。
这是 NN/Torch 力的接缝。

**单位裁决（maintainer, 2026-08-25——MD 完全 unit-agnostic）：**
MD/积分器/`Potentials` 内零单位知识；用户在传入前自行换算一切，工具在
`md::units`：`energy_to_md(value, unit)`、`preset_energy_to_md(style)`
（LAMMPS-units 风格 preset："real"、"metal"……）、`kb_md()`。
`Potentials.set_energy_scale` 改为 **numeric-only**（PyO3 侧已删
`"real"` 字符串解析）——它是"用户算好的因子作用于合并后 energy+forces"的
机制，任何代码不得隐式调用。`MD.set_forcefield(ff, energy_scale=None)`：
None = 什么都不套；docstring 示范用户自己传
`energy_scale=preset_energy_to_md("real")`。仍带单位假设的 helper 已逐个
文档化：`MaxwellBoltzmann`（K+amu→Å/fs，内部 `kb_md`）、`Langevin` 的
`kbt`（能量，MD 单位下 = `kb_md()*T`）、`MD.run(temperature=…)` 的速度
初始化与 `thermo` 行的 `temp` 列（用 `kb_md()`）。

**Loop-owned neighbors：** 邻表是循环（积分器）的事。
`VelocityVerlet(dt, potential=…, neighbors=VerletSkin, mass=…)` /
`Langevin(…)`：积分器跑 skin 的 every/delay/check 重建策略，重建后经
`Potential::set_pairs(&[SkinPair], &SimBox)`（trait 默认 no-op；
`Potentials` 递归转发给全部成员）把新 pair 喂给 nonbond 成员。Python 不做
任何 pair bookkeeping。`NeighborList`→`VerletSkin`→积分器均为 move 语义；
积分器暴露只读 `num_edges`/`rebuild_count`/`ago`（无邻表时 None）。传
`LJCut` 必须带 `neighbors=`。

**Merged MD（一个 driver，一个装配步）：** 旧 `MD`+`MDRunner` 合并为
`md/driver.py` 的一个 `MD` 类；`FrameVelocityVerlet`、numpy 双胞胎
`MDState`/`ForceOutput`/`MDObservables`、`_as_state`、hook 层
（`MDHook`/`CheckpointHook`/`VelocityInitHook`）全部删除。保留的旧能力以
最简形式折入 `run()`：`temperature=`/`seed=`（LAMMPS `velocity create`，
经 `MaxwellBoltzmann`）与 `thermo=N` 间隔采样（LAMMPS thermo 词汇：
step/pe/ke/etotal/temp，存 `driver.thermo`）。表面：
`MD(prec="double")`（**prec 是精度 campaign 的预留接口**：
`PRECISIONS`/`resolve_prec`，只收 "double"，mixed/single raise 并声明将落
在 Rust 积分器）；`set_forcefield(ff, energy_scale=None)`（每次 run 用
`ff.to_potentials(frame)` 重新编译——存配置不存 moved 对象，第二次 run
天然可用）；`set_neighbors(prebuilt VerletSkin | cutoff/skin/every/delay/
check kwargs)`（prebuilt 是 single-shot，被消费后再 run 报指名
set_neighbors 的 ValueError；kwargs 每 run 现建）；`set_potential(pots)`
（advanced：调用者自己 set_energy_scale、自己管邻居正确性；`Potentials`
被 move 进积分器 = 一次 attach 一次 run）。pair 路径：`pair:lj/cut` 读
(epsilon, sigma, cutoff) 按 ff 原样构造 `LJCut` **push 进同一编译
collection**（无单位换算——用户的 energy_scale 统一作用）；单一
atom-type 参数集 only，多集或非 lj/cut style raise NotImplementedError；
cutoff 推导链 set_neighbors kwarg > style param > per-type max > prebuilt
skin cutoff，推不出 raise 指名 set_neighbors 的 ValueError；无 kspace
逻辑；pair+非空 bonds block 仍明拒（special_bonds 排除未实现，防
1-2/1-3/1-4 重复计入）。

**守恒测试归位（maintainer 裁决：tests/ 不放长物理跑）：** NVE 守恒
authority 是 Rust 单元测试（`molrs/src/md/integrators.rs`）；driver 级长跑
在 `regressions/release-0-14-01-md-driver-nve.py`（64 原子 Ar-like lj/cut
ForceField，dt=1fs×1200 步，断言相对漂移 <5e-5 且 `rebuild_count > 0`）。
`tests/test_md.py` 只留数步级 seam/驱动单测。

**Deferred to 0.15:** wasm/capi 绑定面对称性决策——md 0.14 仅 Python
（facade-first，尚无 wasm 消费者；参照 binder-surface-symmetry）；driver
pair 路径的 per-type mixing 与 special_bonds 排除；molpy 全表面镜像
（本轮只做了 md 切片）。`LJCut` vs `ff::pair::lj/cut` kernel 合并债见
2026-08-25 独立条目（保留）。

**Status:** provisional
