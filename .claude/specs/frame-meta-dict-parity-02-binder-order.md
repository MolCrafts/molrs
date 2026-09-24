---
title: Frame meta dict parity 02 — binder enumeration order
slug: frame-meta-dict-parity-02-binder-order
status: in-progress
created: 2026-09-22
chain: frame-meta-dict-parity (01-ordered → 02-binder-order → 03-untyped-write → 04-dict-views → 05-document → 07-sequence)
revised: 2026-09-22
depends_on: frame-meta-dict-parity-01-ordered
---

# Frame meta dict parity 02 — binder enumeration order

## Summary

Link 01 makes a frame's `meta` remember the order its keys were written. This link makes the four language bindings report that order instead of inventing one. Today three of them re-sort on the way out (`PyFrameMeta::keys`, `molrs_frame_meta_key`, `frame_meta_entries`) and the fourth leaks whatever order the backing map happened to have (`Frame.metaNames` in wasm), so the same frame enumerates its metadata four different ways and no binding matches `dict` semantics. After this link, a Python user, a JS user, a C caller and an Atomiverse C++ caller who write `a`, then `b`, then `c` all read back `a, b, c`, and each binding's own gate has a test that fails if a sort is ever reintroduced.

## Domain basis

None. `$META.science.required` is true, but this link changes no numerical result, no unit and no equation — it changes the order in which four bindings report string keys. No DOI/arXiv reference applies. Stated rather than silently omitted.

## Design

**Constitution.** There is no `.claude/notes/law.md` in this tree, and none is expected: `.claude/specs/cgsmiles-03-release.md:22` already records that, and names the substitute. The binding design constitution is the `mol:bootstrap` managed block of `CLAUDE.md` (Iron law — no silent debt; Prefer / Forbid; Shape check; Tests) together with `.claude/notes/architecture-rules.md` and `.claude/notes/testing.md`. This link adds **no public symbol** in any of the four roots — every code change is the deletion of a re-ordering step plus doc and test edits — so Shape check #1–#4 are satisfied vacuously, and no free helper is introduced.

**Chain position.** `frame-meta-dict-parity` is an ordered chain: **01** (core `MetaMap` → `IndexMap`, the serde wire, the Zarr frame-group round-trip contract) → **02** (this link, binder enumeration surfaces) → **04** (`PyFrameMeta::map` clone-per-read). This link has no effect without 01; 01's landing is `ac-001`, verified by inspection.

### Audit bound — how many enumeration surfaces exist, and who owns each

Five places in this repository turn a frame's metadata into an ordered sequence:

| # | Surface | Path:line | Today | Owner |
|---|---|---|---|---|
| 1 | `PyFrameMeta::keys` | `molrs-python/src/core/store/frame.rs:238-242` | `names.sort_unstable()` | **this link** |
| 2 | `Frame::meta_names` (wasm `metaNames`) | `molrs-wasm/src/core/frame.rs:489-497` | no sort; leaks map order | **this link** |
| 3 | `molrs_frame_meta_key` | `molrs-capi/src/frame.rs:860-888` (sort at `:872-873`) | `keys.sort()` | **this link** |
| 4 | `frame_meta_entries` | `molrs-cxxapi/src/lib.rs:1020-1026` (sort at `:1024`) | `values.sort_by_key` | **this link** |
| 5 | serde wire `impl Serialize for Frame` | `molrs/src/serialize.rs:511-513` (+ `FrameRepr` at `:527`) | `BTreeMap` both halves | **link 01** |

Surface 5 is a genuine fifth enumerating surface and it is **not** left open: link 01 changes both halves of the wire to `IndexMap`, and that change is carried in 01's Design, its Files-to-modify, its tasks and its `ac-006`. This link therefore neither edits `serialize.rs` nor spawns a task for it. The consequence for scoping: **`ac-011`'s cross-binder order claim is scoped to the four binder enumeration surfaces above, resting on 01's wire guarantee for anything that crosses serde.**

`molrs-ffi` has no metadata enumeration surface at all (searched: `molrs-ffi/src/**` contains no `meta_names` / `meta_keys` / `meta.keys` / `meta.iter`), so the handle crate needs no change and is not in Files.

### Surface 1 — Python: one class, two orders today

`PyFrameMeta` currently answers the same question two ways. `keys()` sorts (`:240`); everything that iterates `self.map()` directly does not — `as_dict` (`:163`), `__eq__` (`:225`), `__repr__` (`:229`), `copy` (`:337`), `typed` (`:345`), `__or__` (`:361`), `__ror__` (`:371`). The derived accessors `__iter__` (`:221`), `values` (`:244`), `items` (`:252`) and `popitem` (`:285`) go through `keys()` and inherit the sort. So `list(m)` and `list(m.copy())` can disagree on order today. Deleting the sort at `:240` collapses both paths onto one order (the frame's), which is what makes the class internally consistent — that is the point of the edit, not a side effect.

**Named behaviour change:** `popitem()` today pops the lexicographically last key. With the sort gone it pops the last-inserted key, which is `dict.popitem`'s documented LIFO contract. This is parity, not regression, and it is called out in the test and in the stub docstring.

### Surface 2 — wasm: no code change, a tightened guard

`meta_names` (`:489-497`) already hands back `frame.meta.keys()` untouched; its order is whatever the backing map has, which becomes insertion order once 01 lands. The code is correct as written. What is wrong is the test: `meta_names_contains_all_keys` (`:756-763`) asserts `len == 2` plus two `contains` calls, which passes under any permutation and under a reintroduced sort. It is tightened to a `Vec` equality and renamed. The existing fixture `frame_with_meta` (`:725-735`) inserts `energy` then `config` — already non-alphabetical — so no fixture change is needed.

**Gate reality for this surface:** the `wasm-pack` pre-push hook (`.pre-commit-config.yaml`) runs `wasm-pack build --release`, **not** `wasm-pack test`; only `.github/workflows/ci-wasm.yml:69` runs `wasm-pack test --node`. `CLAUDE.md` calls `prek run --all-files --hook-stage pre-push` the local CI mirror, so a green local gate will not have executed the tightened `meta_names` equality — it first runs on GHA.

### Surface 3 — capi: drop the sort, and put the test where a gate runs it

`molrs_frame_meta_key` drops `keys.sort()` (`:872-873`) and its rustdoc at `:858` stops saying "lexicographically sorted". This is a **breaking semantic change to a published C ABI**, so it must not ship without executing coverage.

The order assertion goes in **`molrs-capi/tests/cpp/test_molrs_capi.cpp`**, as a new `TEST_F(MolrsTest, FrameMetadataOrder)` beside the existing `TEST_F(MolrsTest, FrameMetadata)` at `:99-120`. That file already exercises `molrs_frame_put_meta` / `molrs_frame_read_meta` (`:106-117`) while covering **neither** `molrs_frame_meta_count` nor `molrs_frame_meta_key`, so the new case is additive and lands on an already-gated surface: pre-push `capi-tests` and `.github/workflows/ci-capi.yml:31-41` both configure, build and `ctest` it.

**No `#[cfg(test)] mod tests` is added to `molrs-capi/src/frame.rs`** — see the found-debt entry below for why that placement would be dead on arrival.

The committed cbindgen header `molrs-capi/include/molrs.h:223` carries the signature only, with no doc comment (`molrs-capi/cbindgen.toml:7` sets `documentation = false`), and the signature is unchanged, so the header is not in Files and a header diff is a failure signal.

**Cross-repo consumer check (all sibling repos, per the standing rule that a zero-consumer claim must span them):** no caller of `molrs_frame_meta_key` exists anywhere under `/home/jicli594/work/molcrafts` outside this crate — the only other hits are the vendored `Atomiverse/molrs` submodule copy of this same source. The C ABI order change has no known consumer to break.

### Surface 4 — cxxapi: drop the sort

`frame_meta_entries` drops `values.sort_by_key(|(a, _)| *a)` (`:1024`) and the now-pointless `Vec` materialisation collapses to a direct `frame.meta.iter()` map. Its test goes into the existing `#[cfg(test)] mod tests` at `molrs-cxxapi/src/lib.rs:1570`, next to `frame_metadata_roundtrip_preserves_exact_native_types` (`:1729-1754`), whose fixture already inserts `tag` then `stress` (non-alphabetical) but asserts with `.find()` and so is order-blind. That suite **is** gated: `cargo test --manifest-path molrs-cxxapi/Cargo.toml` runs in pre-push `cargo-test-binders` and in `.github/workflows/ci-rust.yml:54`.

**Cross-repo consumer check:** the bridge declaration is `molrs-cxxapi/build.rs:117`; no Atomiverse C++ call site depends on the entries being sorted (searched sibling repos; the only other hit is the vendored submodule copy).

### Fixture rule (applies to every test this link writes or tightens)

**Every fixture must be non-alphabetical in insertion order.** A test whose keys happen to be inserted in sorted order cannot distinguish "the binding preserved insertion order" from "the binding sorted". The counterexample is already in the tree: `test_mapping_protocol` (`molrs-python/tests/test_frame.py:219-229`) uses `{"a": 1, "b": "two"}` and asserts `sorted(f.meta) == ["a", "b"]` — both the fixture and the `sorted()` wrapper make it non-discriminating, so it is corrected here rather than left as a test that can never fail.

### Doc-scope rule (widened)

Exactly one rustdoc in the repository states an enumeration order for a metadata surface: `molrs-capi/src/frame.rs:858`. Widened rule for this link: **a doc that names an order for an enumeration surface is part of that surface and changes in the same commit as its code.** Under that rule three docs are in scope — `molrs-capi/src/frame.rs:858`, the `metaNames` rustdoc (`molrs-wasm/src/core/frame.rs:477-487`), and the `FrameMeta` stub docstring (`molrs-python/python/molrs/_lib.pyi:333-343`, which says nothing about order today while the class advertises itself as a `MutableMapping`).

**Trip-wire.** The scope statement in this document — that Zarr round-trip order is 01's contract and not this link's claim — is a statement about *this spec's boundary*, not a property of the code. It must **not** appear in any rustdoc on `MetaMap` (`molrs/src/core/store/meta.rs:304-306`) or `write_frame_group` (`molrs/src/io/zarr/frame_io.rs:522`). Wording such as "incidental, not guaranteed" landing there would leave two documents making opposite statements about one contract, since 01 lands that guarantee and its test.

### Found debt — named, not fixed (Iron law)

1. **The `molrs-capi` Rust unit-test suite is dead: no gate executes it.** Pre-push `capi-tests` runs `cargo build` + `cmake` + `ctest`; `.github/workflows/ci-capi.yml:25-41` runs clippy + build + cmake + ctest; the only `cargo test` binder lines anywhere name `molrs-ffi` and `molrs-cxxapi` only. Nothing runs `cargo test --manifest-path molrs-capi/Cargo.toml`. The standing evidence is **`molrs-capi/src/schema.rs:150-168`** — a `#[cfg(test)] mod tests` that compiles under `clippy --all-targets` and has never executed. **Route: `/mol:fix` — add `cargo test --manifest-path molrs-capi/Cargo.toml` to the pre-push `capi-tests` hook and to `ci-capi.yml`.** Not fixed inside this link: this link's obligation is to place its own assertion where a gate runs it, and gate topology is its own change with its own blast radius.

2. **capi deep-clones the whole frame to read one metadata key.** `molrs_frame_read_meta` (`molrs-capi/src/frame.rs:821`), `molrs_frame_meta_count` (`:847`) and `molrs_frame_meta_key` (`:868`) each call `store.inner.clone_frame(...)`. `clone_frame` (`molrs-ffi/src/store.rs:111`) copies every block; `Store::with_frame` (`molrs-ffi/src/store.rs:117`) is the borrow-only door and is the target. `molrs-wasm`'s `meta_names` (`molrs-wasm/src/core/frame.rs:490-496`) is the in-repo call site that already does it right. **Route: `/mol:refactor`.** Not fixed here: it is a performance change to three FFI entry points, orthogonal to ordering, and it lands on the same dead-suite problem as item 1.

3. **`PyFrameMeta::map()` deep-clones the entire `MetaMap` on every read** (`molrs-python/src/core/store/frame.rs:122-126`). **Route: link 04** of this chain, which owns it.

4. **Zarr frame-group round-trip order is owned by link 01 and verified by `ac-001`** — not an open follow-up. This link routes nothing there.

### Four-root concession (cited exception to the large-spec split rule)

`$META.arch.style` is `crate-graph`, and Files below touch four separate workspace roots, which normally trips the "crosses more than one package" split rule. **Operator-granted exception, recorded here as required.** No rule in this repo mandates one link per workspace root: `architecture-rules.md` § *Single crate (0.12+)* makes binders separate roots for build and addressing, which is a build rule, not a change-granularity rule; § *Module dependency rules (ENFORCED)* governs `molrs/src`, which this link does not touch. The positive case: three binders had each taken a **private copy** of an ordering policy that `MetaMap` owns, and each copy is a different policy (`sort_unstable`, `sort`, `sort_by_key`) applied at a different point. Deleting all three in one change returns the decision to one home; splitting them across three links would mean three intervals during which the four surfaces of one contract disagree with each other. The fourth root gains a documentation line and an assertion for the surface that already inherits correctly, so it carries no policy of its own either way.

The resulting audit property is narrow and checkable by inspection: **four enumerating surfaces, three of which lose exactly one sort call and one of which needs none.** No control flow is added anywhere, no symbol is created or removed, and no function signature changes.

### Reuse decision

No `librarian_report` was supplied for this link (blueprint refresh deferred by the caller), so the reuse resolution is made from a direct scan of the four roots:

- **`Store::with_frame` (`molrs-ffi/src/store.rs:117`) — `reuse`, but not here.** It is the correct door for capi's three metadata readers; adopting it is found-debt item 2, routed to `/mol:refactor`.
- **`frame_with_meta` (`molrs-wasm/src/core/frame.rs:725-735`) — `reuse`.** The wasm order test uses the existing fixture unchanged; its insertion order is already non-alphabetical.
- **`TEST_F(MolrsTest, FrameMetadata)` (`molrs-capi/tests/cpp/test_molrs_capi.cpp:99-120`) — `pattern`.** The new `FrameMetadataOrder` case follows its construction, `ASSERT_MOLRS_OK` style, `molrs_free_string` discipline and `molrs_frame_drop` teardown.
- **`frame_metadata_roundtrip_preserves_exact_native_types` (`molrs-cxxapi/src/lib.rs:1729-1754`) — `pattern`.** The new cxxapi test follows its `empty_meta_entry` / `frame_set_meta_entry` construction, replacing the order-blind `.find()` assertions with a key-sequence equality.
- **A shared cross-binder "meta key order" helper — `new — rejected`.** The four roots are separate workspaces with no shared crate below them except `molrs`/`molrs-ffi`, and each change is a *deletion*. Extracting anything would add a symbol to satisfy zero call sites, against Forbid (free helpers over object access) and against "inline until the second real use".

## Files to create or modify

- `molrs-python/src/core/store/frame.rs`
- `molrs-python/tests/test_frame.py`
- `molrs-python/python/molrs/_lib.pyi`
- `molrs-wasm/src/core/frame.rs`
- `molrs-capi/src/frame.rs`
- `molrs-capi/tests/cpp/test_molrs_capi.cpp`
- `molrs-cxxapi/src/lib.rs`
- `.claude/notes/notes.md`

No new files. The committed header `molrs-capi/include/molrs.h` is cbindgen output with unchanged signatures and is deliberately absent from this list.

## Tasks

- [x] Write failing order tests for `FrameMeta` in `molrs-python/tests/test_frame.py::TestFrameMeta` — non-alphabetical fixture, asserting `list(f.meta)`, `f.meta.keys()`, `f.meta.items()`, `f.meta.values()`, `dict(f.meta)` and `repr(f.meta)` all follow insertion order, plus `popitem()` returning the last-inserted key
- [x] Replace the non-discriminating fixture in `molrs-python/tests/test_frame.py:219-229` (`test_mapping_protocol`) — non-alphabetical keys, `list(f.meta)` instead of `sorted(f.meta)`
- [x] Write failing `TEST_F(MolrsTest, FrameMetadataOrder)` in `molrs-capi/tests/cpp/test_molrs_capi.cpp` beside `FrameMetadata` (`:99`), covering `molrs_frame_meta_count` and `molrs_frame_meta_key` over a non-alphabetical insertion sequence plus an out-of-range index
- [x] Write failing `frame_meta_entries_follow_insertion_order` in the existing `#[cfg(test)] mod tests` of `molrs-cxxapi/src/lib.rs` (`:1570`), asserting the full key sequence of `frame_meta_entries` rather than `.find()`
- [x] Tighten `meta_names_contains_all_keys` → `meta_names_follow_insertion_order` in `molrs-wasm/src/core/frame.rs:756-763` to a `Vec<String>` equality against `frame_with_meta`'s insertion order
- [x] Remove `names.sort_unstable()` from `PyFrameMeta::keys` in `molrs-python/src/core/store/frame.rs:238-242`
- [x] Remove `keys.sort()` from `molrs_frame_meta_key` in `molrs-capi/src/frame.rs:872-873` and rewrite its rustdoc at `:858`
- [x] Remove `values.sort_by_key` from `frame_meta_entries` in `molrs-cxxapi/src/lib.rs:1023-1025`
- [x] Update the order-bearing docs per `doc.style`: the `metaNames` rustdoc (`molrs-wasm/src/core/frame.rs:477-487`) and the `FrameMeta` docstring (`molrs-python/python/molrs/_lib.pyi:333-343`), keeping this spec's scope wording out of every rustdoc
- [x] Record the found debt in `.claude/notes/notes.md`: the dead `molrs-capi` Rust suite with `molrs-capi/src/schema.rs:150-168` (→ `/mol:fix`), the capi whole-frame clone with `molrs-ffi/src/store.rs:117` as target (→ `/mol:refactor`), and `PyFrameMeta::map` (→ link 04)
- [ ] Run full check + test suite

## Testing strategy

Per `.claude/notes/testing.md` and `CLAUDE.md` § Testing Rules: unit tests live in `#[cfg(test)]` next to the code, there is no `molrs/tests/` tree, fixtures are inline, and **bindings prove only the seam** — order at the boundary is exactly a seam property, so these tests belong in the binders and nowhere else. No numeric science is re-derived here.

- **Python** — `molrs-python/tests/test_frame.py::TestFrameMeta`, new case: build a frame, write `zeta`, `alpha`, `mu` in that order, assert `list(f.meta) == ["zeta", "alpha", "mu"]` and the same sequence from `.keys()`, `.items()`, `.values()`, `dict(f.meta)` and `repr(f.meta)`. Runs under `tox -e py` in pre-push `python-tox` and `ci-python`.
- **Edge case, Python** — `popitem()` returns `("mu", …)`, the last-inserted key, not the lexicographically last. This is the one user-visible behaviour change and it must be asserted, not merely documented.
- **Correction, Python** — `test_mapping_protocol` (`:219-229`) currently cannot fail; both the fixture and the `sorted()` wrapper are fixed.
- **capi** — `TEST_F(MolrsTest, FrameMetadataOrder)`: `molrs_frame_put_meta` for `zeta`, `alpha`, `mu`; `molrs_frame_meta_count` == 3; `molrs_frame_meta_key(frame, i, &out)` for i = 0,1,2 gives that sequence, each freed with `molrs_free_string`; frame dropped. Runs under both `ctest` gates. **Edge case:** index 3 still returns a non-OK status.
- **cxxapi** — `frame_meta_entries_follow_insertion_order`: insert `tag`, then `stress`, then `run`; assert the returned `Vec<MetaEntry>` keys are `["tag", "stress", "run"]`. Runs under `cargo test --manifest-path molrs-cxxapi/Cargo.toml`.
- **wasm** — `meta_names_follow_insertion_order`: `assert_eq!(frame.meta_names(), vec!["energy".to_string(), "config".to_string()])`. **This assertion is CI-only**: the pre-push `wasm-pack` hook builds but does not test, so only `.github/workflows/ci-wasm.yml:69` executes it.

**Discriminating-fixture requirement.** Every fixture above is non-alphabetical in insertion order (`zeta, alpha, mu`; `tag, stress, run`; `energy, config`), so reintroducing a sort at any of the four surfaces fails a test rather than passing silently.

**No `regressions/` example, and why.** This repository has no `regressions/` tree, and `CLAUDE.md` § Build & Test Commands states that the benchmark and regression systems are being redesigned outside this repo; `.claude/specs/cgsmiles-03-release.md:177` is the standing precedent. The equivalent public-API demonstration for each binding is its seam test above, run by that binding's own gate. The omission is a cited exception, not a finding. Consequently no `type: runtime` regression criterion appears in the acceptance contract; `ac-003`, `ac-006`, `ac-008`, `ac-009` and `ac-013` are the executing bars instead.

**No third-party software and no external oracle** is involved at any point.

## Out of scope

- **`molrs/src/serialize.rs` (the serde wire).** A real fifth enumeration surface, owned and closed by **link 01**. Not edited and not tasked here.
- **`molrs/src/io/zarr/frame_io.rs` (Zarr round-trip order).** Owned by link 01, verified by `ac-001`.
- **`PyFrameMeta::map()`'s clone-per-read** — link 04.
- **Adding `cargo test --manifest-path molrs-capi/Cargo.toml` to the gates** — named found debt, routed to `/mol:fix`. This link places its capi assertion where a gate already runs it rather than changing gate topology.
- **capi's `clone_frame`-per-metadata-read** — named found debt, routed to `/mol:refactor`, target `Store::with_frame`.
- **Block name ordering.** `Frame`'s blocks remain a `HashMap`, and the `blockNames` rustdoc at `molrs-wasm/src/core/frame.rs:499-504` correctly says so. Nothing here makes block enumeration ordered, and that doc is deliberately left alone — a reader of the wasm file should not conclude the whole file became insertion-ordered.
- **`molrs-ffi`** — no metadata enumeration surface exists there.
- **`molrs-capi/include/molrs.h`** — cbindgen output, signatures unchanged, no doc comments carried.
