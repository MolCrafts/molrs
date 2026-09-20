# molrs — Evolving Decisions

Working notes captured by `/mol:note`: a decision, why it was taken, and what
it costs someone who does not know it. An entry leaves this file when it is
promoted into `CLAUDE.md`, lands in `.claude/notes/release.md` as shipped
behaviour, or is superseded — the record of what shipped is the git history,
not a pile of notes.

```
## YYYY-MM-DD — <topic>
**Decision:** <one-liner>
**Why:** <constraint, incident, or measurement>
**Status:** provisional | hardening | promoted (→ CLAUDE.md §section)
```

---

## 2026-09-14 — regions are solids with a signed distance; one type per shape

**Decision:** `spatial::region::Region` requires `bounds` + `distance`
(negative inside, positive outside, zero on the boundary) and derives
`contains_point` / `contains` / `distance_grad` from it. Every shape describes
its inside; outside, shells and voids are `NotRegion` / `AndRegion` /
`OrRegion`. `HollowSphere` is gone (`Sphere & ~Sphere`). New shapes:
`HalfSpace`, `Cylinder`, `Ellipsoid`, `Polyhedron` (a watertight `TriMesh`,
whatever produced it), `SphereUnion` (centres + radii, minimum image on the
box's periodic axes; with `r = r_vdW + r_probe` its complement is the
solvent-accessible void). molpack's own `Region` trait, `StlRegion` and BVH
move here; molpack keeps only its penalty policy.
**Why:** three parallel region models (molrs boolean, molpack SDF + mesh,
molvis TS ball field) for one concept; the PEO-in-void case needs a region
built from atoms in memory, no mesh file, and a lattice mask that asks the
region about ~10⁶ sites — a BVH per query, not a scan.
**Status:** provisional

## 2026-09-07 — cargo needs `target/` on local disk; a symlink, not `CARGO_TARGET_DIR`

**Decision:** make `<repo>/target` a symlink to node-local storage. Do **not**
solve this with `CARGO_TARGET_DIR` — several gates hard-code `target/...`
paths and will read a different tree than the one they just wrote.
**Why:** the checkout is on Lustre (`/nobackup`, and `/home` too), where cargo
blocks forever on its own artifact lock. Measured twice: `cargo doc` at 9 h 27 m
elapsed with 0 s of CPU, and a pre-push `cargo test --lib` at 33 min with 1 s of
CPU — no `rustc` child in either case, `wchan = ldlm_flock_completion_ast`, and
`lsof` showing the blocked process as the *only* holder of
`target/debug/.cargo-artifact-lock`. So it is a wedged flock, not a slow
fingerprint scan over a large artifact tree.

`CARGO_TARGET_DIR=/tmp/...` does unblock cargo, and that is what an earlier
version of this note recommended — but every committed path (`target/wheels`,
`target/release/libmolrs_capi.so`, the CMake `CARGO_TARGET_DIR` default) assumes
the one shared target dir; the symlink keeps them all pointing at one real
directory.

```bash
mv target target.lustre-cache        # keep or delete; it is only a cache
ln -s /tmp/molrs-target target
```

Node-local means node-local: on a different login node the symlink dangles and
the cache is cold. That is the cost, and it is smaller than losing a day to a
lock that never returns.
**Status:** hardening

## 2026-09-07 — `dump local` accepts every `dump_modify label`

**Decision:** the LAMMPS `dump local` reader accepts `ENTRIES` (the default)
plus `BONDS` / `ANGLES` / `DIHEDRALS` / `IMPROPERS` / `NEIGHBORS`, and records
which one in `frame.meta` as `dump_local_label`. Rows still land in `entries`
whatever the label says.
**Why:** the reader's own doc comment cited OVITO's LAMMPS-dump-local manual
while implementing one of the six labels it lists — and `dump_modify …
label BONDS` is precisely what that manual tells users to set, so the
recommended setup was a hard parse error. The label is also the *only*
meaning-bearing part of such a file: column names are whatever the dump
command was given, and the default is `c_bond[1] c_bond[2]`, which says
nothing. Consumers must read the label rather than re-guess from the header.
The rows keep the `entries` name because the label says what they mean, not
that they satisfy a contract-bearing block's schema — the argument recorded
at the `block_name` binding still holds.
**Status:** provisional

## 2026-09-07 — `Frame.getMeta` completes the wasm meta pair

**Decision:** `molrs-wasm` `Frame` exposes `getMeta(name) -> string | undefined`
alongside the existing `setMeta` / `getMetaScalar` / `setMetaScalar`. It
returns `undefined` for non-string values rather than stringifying them.
**Why:** `setMeta` had no reader, so a word-valued label written through it
was unreachable from JS — `getMetaScalar` returns `undefined` for anything
that does not parse as a number. Found while wiring `dump_local_label`
through to molvis. Not stringifying numeric meta keeps `"1"` and `1`
distinguishable at the boundary.
**Status:** provisional

## 2026-09-04 — amber-prmtop-complete-02 debts found at spec time

**Decision:** record, do not fix in this phase. (A third item — docs claiming
no `molrs/tests/` tree while `architecture_gate.rs` sits in it — was fixed on
2026-09-17.)
**Why:** iron-law naming of rot that is out of this spec's layer or below the extract-on-second-use bar.
**Status:** provisional

1. ~~σ/ε closed form duplicated between `ff/forcefield/readers/prmtop.rs` and `io/data/prmtop_tables.rs`.~~ Closed 2026-09-17: both call `core::math::pair_form::lj_ab_to_sigma_epsilon`. Two call sites *is* the extract bar — CLAUDE.md § Prefer says "extract only at a second call site", which this note had mis-stated as the third.
2. `forcefield/gaff.rs` cited `scripts/gen_gaff_energy_oracle.py` for the `AMBER_COULOMB` sander measurement; that generator is not in the tree. The value stands (`18.2223²`); the rustdoc no longer points at the missing script.


## 2026-08-26 — development tools track latest; other-platform UB out of scope
**Decision:** rustc/clippy/rustfmt = `rust-toolchain.toml` `channel = "stable"`
and CI `dtolnay/rust-toolchain@stable`. wasm-opt = latest binaryen GitHub
release. uv action = latest major. pre-commit-hooks = latest tag. Never pin a
compiler/linter minor to hide CI drift. Other-platform UB (Windows/macOS-only)
is out of scope until 0.14 lands on `dev`.
**Why:** pinning rustc 1.96 made local prek green while GHA clippy on 1.98 was
red; the next rustc bump would just repeat it.
**Status:** active

## [2026-08-10] Binding-surface symmetry (settled after the neighborlist chain)

The quality of the facade (the public API) outranks the internal implementation;
internals are refactored incrementally, without chasing a single sweep and without
blocking a release.

**Rule**: the Rust / Python / WASM surfaces must stay symmetric — same names
(`NeighborList` the engine / `Neighbors` the table), same shape (build / update /
neighbors + the Option column semantics), same defaults (`FULL`). Before adding to
or changing any one binding surface, check it against the other two.

Known asymmetries (in internal-refactor priority order):

1. **The wasm `NeighborQuery` symmetry gate is deferred to 0.15** (2026-08-25).
   Deletion is not among the options: in-tree consumers are
   `compute/hbond/detect.rs` (`from_columns` / `free_columns`,
   `QueryMode::CrossQuery`), `compute/rdf/mod.rs`, `compute/dynamics/van_hove.rs`
   and `ff/potential/soft.rs`. wasm has no consumer yet (facade-first), so 0.14
   neither adds the symmetry gate nor deletes the engine type.
2. The `LinkedCell` / `BruteForce` aliases survive only for the molvis link
   (default `FULL`, safe); once molvis moves to the engine API they are **deleted**
   — two doors are not maintained long-term.
3. The remaining routed items are done slowly, as needed: core SoA
   `update_columns`, splitting `neighbors/mod.rs` into `table.rs` (a pure move).
   Borrowing `Compute::Args` is done (2026-08-10).

**Status:** active

<!-- mol:note:topic:md-experimental-ship-0.14 -->
## 2026-09-20 — no BLAS feature; the 3x3 helpers are closed-form only

**Decision:** the `blas` feature and `ndarray-linalg` are gone. `core::math`
keeps the hand-written `det3` / `inv3` and a single-body `matmul` that `rayon`
parallelises by row.
**Why:** the LAPACK path served only a 3×3 determinant and inverse — slower
than the cofactor forms next to it — and it `.expect()`-panicked on a singular
matrix where the default build returned a value, so enabling an
"optimisation" feature changed panic semantics. No consumer in any sibling
repo enabled it. The 2026-05-28 entry about backend selection is superseded.
**Status:** locked
