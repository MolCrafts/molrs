# FFI Scope-Narrowing: a language-agnostic fix for spurious handle invalidation

Status: proposal (2026-04-23)
Owner: FFI
Affects: `molrs-ffi`, `molrs-wasm`, `molrs-capi`, `molrs-cxxapi`, `molrs-python`

## TL;DR

`Store::with_frame_mut` conservatively invalidates every `BlockHandle`
on a frame after the closure returns, even when the closure only
touched frame meta or grids. This produces spurious
`InvalidBlockHandle` errors in downstream consumers.

Because the four binding crates (`molrs-wasm`, `molrs-capi`,
`molrs-cxxapi`, `molrs-python`) all route through the same
`molrs-ffi::Store`, **one fix in `Store` benefits every consumer
language simultaneously**.

Proposal: split `with_frame_mut(&mut Frame)` into three scope-narrow
closures borrowing disjoint slices (`&mut FrameMeta`, `&mut GridMap`,
`&mut BlockMap`). Rust's type system then guarantees the invalidation
set — no runtime instrumentation required.

## 1. Background — why this problem is not Rust-specific

Molrs exposes Frame/Block data to non-Rust callers via integer
handles (`FrameId`, `BlockHandle`). Rust's compile-time borrow
checker cannot cross the FFI boundary; once a handle reaches
JavaScript / Python / C / C++, the caller is free to:

- retain the handle across mutation calls,
- hand it to another thread or coroutine,
- use it after the underlying data has been rewritten,

all of which the Rust compiler is invisible to.

The existing `(frame_id, key, version)` scheme in
`molrs-ffi/src/handle.rs` and `molrs-ffi/src/store.rs` is how we
**simulate Rust's borrow rules at runtime** across that boundary.
Every mutating call bumps a version counter; every read validates the
caller's version against the current one (`store.rs:398-418`). Stale
handles fail loudly with `FfiError::InvalidBlockHandle` instead of
silently reading corrupted data — the latter being unacceptable for
scientific workloads where a bad byte in an atom position produces
an MD explosion, not a visible error.

The version mechanism is therefore **mandatory** for any FFI that
wants to preserve zero-copy access to mutable data across a non-Rust
consumer. It is not Rust-idiosyncratic; the same design recurs in
MVCC databases (version columns), generational arenas (slotmap
itself), and epoch-based reclamation (crossbeam-epoch).

## 2. Observed problem — conservative version bumping

### 2.1 Mechanism

```rust
// molrs-ffi/src/store.rs:126-149
pub fn with_frame_mut<R>(
    &mut self,
    id: FrameId,
    f: impl FnOnce(&mut Frame) -> R,
) -> Result<R, FfiError> {
    let entry = self.frames.get_mut(id).ok_or(FfiError::InvalidFrameId)?;

    let keys_before: HashSet<String> = entry.frame.keys().map(|k| k.to_string()).collect();
    let result = f(&mut entry.frame);

    let mut all_keys = keys_before;
    all_keys.extend(entry.frame.keys().map(|k| k.to_string()));
    all_keys.extend(entry.block_versions.keys().cloned());

    for key in all_keys {
        let version = entry.block_versions.entry(key).or_insert(0);
        *version += 1;  // <-- ALL keys bump, regardless of what f() did
    }

    Ok(result)
}
```

Because the closure argument is `&mut Frame`, the store cannot
statically rule out that the closure modified some `Block`. It
therefore bumps every block's version unconditionally. Correct, but
over-coarse.

### 2.2 Over-eager callsites in `molrs-wasm`

All five wasm-side callers currently route through `with_frame_mut`:

| Caller (molrs-wasm) | File:line | Does the closure touch blocks? |
|---|---|---|
| `Frame::set_meta` | `core/frame.rs:506` | **No** |
| `Frame::insert_grid` | `core/frame.rs:394` | **No** |
| `Frame::remove_grid` | `core/frame.rs:419` | **No** |
| `Frame::rename_block` | `core/frame.rs:260` | Yes (legitimate) |
| `Frame::rename_column` | `core/frame.rs:295` | Yes (legitimate) |

Three of five are false positives. Analogous patterns exist in
`molrs-capi`, `molrs-cxxapi`, and `molrs-python`, all of which go
through `molrs-ffi::FrameRef::with_mut` (`shared.rs:87-94`), which
itself calls `Store::with_frame_mut`.

### 2.3 Downstream symptom (molvis)

Concrete symptom observed in the molvis web consumer:

1. App caches `frame.getBlock("atoms")` in a long-lived field.
2. A separate code path calls `frame.setMeta("labels", …)` (e.g. the
   RPC handler for `scene.set_frame_labels`).
3. The cached `Block` handle is now silently stale.
4. Next access (e.g. `mark_atom` reading an atom position, or a
   hover picker resolving atom id → coordinate) throws
   `InvalidBlockHandle`.

The same bug reproduces in any consumer that caches `Block` handles
across a `set_meta`/`insert_grid`/`remove_grid` call. It affects
every binding language equally.

## 3. Design principles for language-agnostic FFI

These are the invariants the current FFI design already upholds.
The proposal below preserves all of them.

1. **All cross-boundary references are integer handles** — never
   raw pointers. Copy semantics, FFI-safe width, survive process
   serialization (worker messages, IPC).
2. **All handles carry a generation/version epoch** — O(1) staleness
   detection, loud failure on misuse.
3. **All mutations narrow the borrow scope to exactly the data they
   touch** — the type system, not runtime instrumentation, decides
   which handles become stale. *(This is the principle currently
   violated by `with_frame_mut`.)*
4. **All errors are `FfiError` variants with 1:1 mappings per
   language** — JS exception / Python exception / C status int /
   C++ exception. One error taxonomy, four translations.
5. **All native resources have explicit free/drop functions** —
   GC/finalizer support is opt-in convenience, never required for
   correctness.
6. **No callbacks cross the FFI boundary** — `with_*` closures live
   entirely inside Rust; the public ABI is flat function calls.

Principle 3 is the lever this proposal pulls.

## 4. Proposal

### 4.1 `Store` API change (internal to `molrs-ffi`)

Replace one over-coarse `with_frame_mut` with three scope-narrow
mutators borrowing disjoint Frame slices:

```rust
// molrs-ffi/src/store.rs

/// Mutate frame meta. Does not bump block versions — the &mut
/// FrameMeta borrow proves the closure cannot reach blocks.
pub fn with_frame_meta_mut<R>(
    &mut self,
    id: FrameId,
    f: impl FnOnce(&mut FrameMeta) -> R,
) -> Result<R, FfiError>;

/// Mutate frame grids. Does not bump block versions.
pub fn with_frame_grids_mut<R>(
    &mut self,
    id: FrameId,
    f: impl FnOnce(&mut GridMap) -> R,
) -> Result<R, FfiError>;

/// Mutate frame blocks. Bumps all block versions on return
/// (same semantics as today's with_frame_mut).
pub fn with_frame_blocks_mut<R>(
    &mut self,
    id: FrameId,
    f: impl FnOnce(&mut BlockMap) -> R,
) -> Result<R, FfiError>;
```

`with_frame_mut` itself remains, initially as an alias for
`with_frame_blocks_mut` with a deprecation warning, removed in the
next major.

The prerequisite is that `Frame` exposes split-borrow accessors:

```rust
// molrs/src/frame/mod.rs (or wherever Frame is defined)
impl Frame {
    pub fn meta_mut(&mut self) -> &mut FrameMeta;
    pub fn grids_mut(&mut self) -> &mut GridMap;
    pub fn blocks_mut(&mut self) -> &mut BlockMap;
}
```

If `Frame`'s internal fields already are `meta: FrameMeta`,
`grids: GridMap`, `blocks: BlockMap`, these are trivial accessor
additions. If the fields are tangled (e.g. blocks indexed inside a
unified map alongside grids), a small refactor precedes this work.

### 4.2 `FrameRef` API change (shared across bindings)

Mirror the split in the shared helper:

```rust
// molrs-ffi/src/shared.rs

impl FrameRef {
    pub fn with_meta_mut<R>(&self, f: impl FnOnce(&mut FrameMeta) -> R)
        -> Result<R, FfiError>;
    pub fn with_grids_mut<R>(&self, f: impl FnOnce(&mut GridMap) -> R)
        -> Result<R, FfiError>;
    pub fn with_blocks_mut<R>(&self, f: impl FnOnce(&mut BlockMap) -> R)
        -> Result<R, FfiError>;
}
```

### 4.3 Binding-crate migration

The FFI surface (JS / Python / C / C++ functions) **does not
change**. Only the internal implementation routes differently.

`molrs-wasm`:

| File:line | Change |
|---|---|
| `core/frame.rs:506` `Frame::set_meta` | `with_frame_mut` → `with_frame_meta_mut` |
| `core/frame.rs:394` `Frame::insert_grid` | `with_frame_mut` → `with_frame_grids_mut` |
| `core/frame.rs:419` `Frame::remove_grid` | `with_frame_mut` → `with_frame_grids_mut` |
| `core/frame.rs:260` `Frame::rename_block` | `with_frame_mut` → `with_frame_blocks_mut` (behavior unchanged) |
| `core/frame.rs:295` `Frame::rename_column` | `with_frame_mut` → `with_frame_blocks_mut` (behavior unchanged) |

Apply analogous rewrites in `molrs-capi`, `molrs-cxxapi`,
`molrs-python`. The audit procedure: grep each crate for
`with_frame_mut` / `with_mut`, classify each closure by what it
actually mutates, route to the narrowest accessor.

### 4.4 Test contract

Add to `molrs-ffi/src/store.rs` tests:

```rust
#[test]
fn meta_mut_preserves_block_handles() {
    let mut store = Store::new();
    let id = store.frame_new();
    let mut block = Block::new();
    block.insert("x", Array1::from_vec(vec![1.0 as F, 2.0]).into_dyn()).unwrap();
    store.set_block(id, "atoms", block).unwrap();
    let handle = store.get_block(id, "atoms").unwrap();

    store.with_frame_meta_mut(id, |meta| {
        meta.insert("energy".into(), "-3.14".into());
    }).unwrap();

    // Contract: meta-only mutation MUST NOT invalidate block handles.
    assert!(store.clone_block(&handle).is_ok());
}

#[test]
fn grids_mut_preserves_block_handles() { /* analogous */ }

#[test]
fn blocks_mut_invalidates_block_handles() {
    // LEGITIMATE invalidation — documents the semantics of blocks_mut.
}

#[test]
fn deprecated_with_frame_mut_still_invalidates_everything() {
    // Back-compat test until removal in next major.
}
```

These tests are **the cross-language contract**. Every downstream
binding inherits them without having to re-test per language,
because every binding dispatches through `molrs-ffi::Store`.

## 5. Migration plan

| Phase | Action | Back-compat |
|---|---|---|
| 1 | Add `meta_mut` / `grids_mut` / `blocks_mut` accessors on `Frame` | no break |
| 2 | Add `with_frame_meta_mut` / `_grids_mut` / `_blocks_mut` in `Store`; mirror on `FrameRef` | no break |
| 3 | Migrate `molrs-wasm` / `molrs-capi` / `molrs-cxxapi` / `molrs-python` call sites | no break; behavior *improves* for meta/grids paths |
| 4 | Mark `Store::with_frame_mut` / `FrameRef::with_mut` as `#[deprecated]` → `with_frame_blocks_mut` | warnings only |
| 5 | Remove deprecated methods in next major | breaking (internal ABI only; public bindings unchanged) |

No public FFI function signature changes. Consumer applications (JS,
Python, C, C++) require no code changes — they observe only the
*removal* of spurious `InvalidBlockHandle` errors.

## 6. Rejected alternatives

| Alternative | Why rejected |
|---|---|
| Drop version check entirely | Silent data corruption on stale handles; unacceptable for scientific data |
| Dynamically track closure mutation set (instrument `&mut Frame`) | Loses type-system guarantee; runtime overhead; same surface area as the current bug |
| Add an FFI `refresh_handle` API | Pushes bookkeeping to every consumer; easy to forget; multiplies error paths |
| Return a new handle from every mutation | Violates stable-id property; breaks caches keyed on handle identity |
| Use Rust lifetimes across FFI | Lifetimes don't survive the C ABI — the entire version scheme exists *because* this is impossible |

## 7. Consumer guidance until this lands

Workaround for downstream code (validated in molvis):

- **Cache `Frame`, not `Block`.** Frame handles survive until
  `frame_drop`; Block handles die on any `with_frame_mut` call.
- **Lazy-fetch blocks at point of use.**
  `frame.getBlock("atoms").viewColF("x")`, not
  `cachedBlock.viewColF("x")`. `Store::get_block` (`store.rs:81-92`)
  is read-only and never bumps versions, so it always yields a
  fresh-version handle.

Reference: `molvis/core/src/entity_source.ts`, which now stores
`Frame` and exposes `frameBlock` as a lazy getter.

## 8. Out of scope

- Reworking the grid-handle model. Grids currently have no cross-FFI
  handle type (they are fetched by name each time), so grid mutation
  touches no handle versions. If grid handles are introduced later,
  they will need their own version stream; the split proposed here
  is forward-compatible with that.
- Revisiting the C ABI error-code mapping. `FfiError` → language-
  specific error translation is adequate today.
- Adding sub-block (column-level) version granularity. Block-level
  invalidation is already fine-grained enough in practice; deeper
  granularity would multiply version state for no observed benefit.
