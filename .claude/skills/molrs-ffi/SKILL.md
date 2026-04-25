---
name: molrs-ffi
description: FFI safety standards for molrs — handle-based design, version-tracked invalidation, no panics across language boundaries. Covers the active CXX bridge in molrs-cxxapi. Reference document only; no procedural workflow.
---

Reference standard for molrs FFI safety. Apply when writing or reviewing any code that crosses the Rust ↔ C++ / Python / WASM boundary.

## Active FFI Crate

`molrs-cxxapi` — CXX bridge to the Atomiverse C++ project. Zero-copy I/O via `FrameView` (borrowed) into existing `write_xyz_frame` / `write_molrec_file`. Owned `Frame` only built when persisting to MolRec (Zarr).

Legacy crates `molrs-ffi/` and `molrs-wasm/` exist on disk but are NOT in the workspace; their patterns are referenced below for completeness but do not currently ship.

## Architecture

```
Rust (molrs-core, molrs-io)
   ▲ borrowed views (FrameView, BlockView, ColumnView)
   │
molrs-cxxapi  (#[cxx::bridge] in bridge.rs)
   ▼ zero-copy slices
C++ (Atomiverse)
```

For a future handle-based store (e.g. Python bindings), the proven pattern is:

```rust
slotmap::new_key_type! { pub struct FrameId; }   // (index: u32, generation: u32)

pub struct BlockHandle {
    pub frame_id: FrameId,
    pub key: String,       // block name (e.g., "atoms")
    pub version: u64,      // snapshot of block version at creation
}

pub struct Store {
    frames: SlotMap<FrameId, FrameEntry>,
}
struct FrameEntry {
    frame: Frame,
    block_versions: HashMap<String, u64>,
}
```

## Safety Rules

### Rule 1 — No raw pointers across the boundary

```rust
// WRONG
#[no_mangle] pub extern "C" fn get_frame() -> *mut Frame { ... }

// CORRECT
#[no_mangle] pub extern "C" fn frame_new(store: &mut Store) -> FrameId {
    store.frame_new()
}
```

CXX bridge: pass slices and shared structs from `cxx::CxxString` / `Vec<T>` — never raw `*mut`.

### Rule 2 — Version-tracked invalidation

Every block mutation increments its version counter. Consumers compare before use; stale handles fail with an error code.

### Rule 3 — No panics in extern functions

`extern "C"` and `#[cxx::bridge]` panics are undefined behavior across the FFI seam.

```rust
// WRONG
#[no_mangle]
pub extern "C" fn frame_get_natoms(store: &Store, id: FrameId) -> i64 {
    store.get(id).unwrap().get("atoms").unwrap().nrows() as i64
}

// CORRECT
#[no_mangle]
pub extern "C" fn frame_get_natoms(store: &Store, id: FrameId) -> i64 {
    match store.get(id).and_then(|f| f.get("atoms")) {
        Some(b) => b.nrows() as i64,
        None => -1,
    }
}
```

For WASM (`wasm_bindgen`): return `Result<T, JsValue>`.

### Rule 4 — Ownership stays in Rust

The FFI layer never transfers ownership of Rust objects. Store owns everything; handles are lightweight references.

### Rule 5 — String handling

- C ABI: accept `*const c_char`, copy immediately into Rust `String`.
- CXX bridge: use `cxx::CxxString` and `let_cxx_string!` for transfer.
- WASM: `&str` works directly via `wasm_bindgen`.

## Naming

| Surface | Convention | Example |
|---|---|---|
| `extern "C"` | `molrs_<noun>_<verb>` | `molrs_frame_new`, `molrs_block_get_f64` |
| CXX bridge | snake_case Rust → snake_case C++ | `write_xyz_frame` |
| `wasm_bindgen` | snake_case Rust → camelCase JS | `link_cell_new` → `linkCellNew` |

## WASM-Specific (when re-activated)

- Set `console_error_panic_hook::set_once()`.
- Use Serde for complex types: `serde_wasm_bindgen::to_value`.
- Use typed arrays for large data: `Float64Array` (matches `F = f64`).
- Always return `Result<T, JsValue>` for fallible operations.

## Compliance Checklist

- [ ] No raw pointers cross the FFI boundary
- [ ] All mutations increment version counters
- [ ] No `unwrap()` / `expect()` in `extern "C"` or CXX-bridge functions
- [ ] Error conditions return error indicators (not panics)
- [ ] Strings are copied on the Rust side immediately
- [ ] Ownership stays in Rust
- [ ] `#[no_mangle]` + `extern "C"` on C ABI functions
- [ ] CXX bridge uses `#[cxx::bridge]`, no manual pointer marshalling
- [ ] WASM functions return `Result<T, JsValue>`
- [ ] Large arrays use typed arrays (`Float64Array`, etc.)
- [ ] BlockHandle validity checked before use
- [ ] FrameId generation checked (stale handles detected)
