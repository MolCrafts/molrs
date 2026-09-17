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
version of this note recommended — but it silently breaks the `link-static`
pre-push hook, which does `rm -rf target/wheels`, runs `maturin build`, and then
installs `target/wheels/*.whl`. With the variable set, maturin writes the wheel
under the override and the hook fails with `libmolrs_ffi.so could not be
located`. The symlink keeps every committed path (`target/wheels`,
`target/release/libmolrs_capi.so`, the CMake `CARGO_TARGET_DIR` default) pointing
at one real directory.

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

## [2026-08-10] 绑定面对称原则(neighborlist 链后定调)

门面(公开 API)质量优先于内部实现;内部走渐进重构,不追求一步到位,不阻塞发布。

**Rule**: Rust / Python / WASM 三个表面的 API 必须保持对称——同名
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

<!-- mol:note:topic:binder-surface-symmetry -->
