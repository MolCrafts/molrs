# FFI Safety Standards

Project standard for any code that crosses the Rust ↔ C++ / Python / WASM
boundary. Applied by `/mol:review --axis=ffi` (the `mol:ffi-guard` agent).

## Active binders (0.12+)

| Crate | Seam | Ships |
|---|---|---|
| `molrs-ffi` | Shared handle/store (`FrameRef`, slotmap generations) | path dep of all binders |
| `molrs-python` | PyO3 → PyPI `molcrafts-molrs` | yes |
| `molrs-wasm` | wasm-bindgen → npm `@molcrafts/molrs` | yes |
| `molrs-capi` | C ABI | yes |
| `molrs-cxxapi` | CXX bridge → Atomiverse | yes |

All four language binders depend on `molcrafts-molrs` + `molrs-ffi`. Do **not**
describe binders as “inactive” or “cxxapi only”.

## Architecture

```
molcrafts-molrs (single science crate)
        ▲
   molrs-ffi (FrameId / BlockHandle / Store)
        ▲
   ┌────┼────────────┬────────────┐
python  wasm       capi        cxxapi
```

Schema vocabulary is owned by Rust `core::store::schema`; binders **project**
it (Python/WASM/C expose JSON tables; cxxapi carries schema-version capability).

## Safety Rules

### Rule 1 — No panics on fallible paths in extern seams

Every CXX / C / wasm export that can fail returns `Result` (or a status code).
No `.unwrap()` / `.expect()` / `panic!` on I/O, insert, empty store, or bad
path. Chemistry errors (e.g. AM1-BCC) already return `Result`.

Gate (allowlist tests only):

```bash
# production body of molrs-cxxapi/src/lib.rs before #[cfg(test)]
rg '\.unwrap\(|\.expect\(|panic!' molrs-cxxapi/src/lib.rs
```

### Rule 2 — Handles and versions

`molrs-ffi` is the exemplar: `FrameId` (slotmap gen) + `BlockHandle.version`.
Mutations bump version; stale handles fail cleanly.

### Rule 3 — String ownership

C API: copy on ingress (`CStr` → owned). Free with paired `molrs_free_*`.

### Rule 4 — Zero-copy lifetimes

CAPI zero-copy getters release the store mutex after returning interior
pointers; multi-threaded C callers must external-sync or treat as invalid after
any mutate.

## Known product differences

- WASM enables `ff` transitively via `conformer`; Python/C enable `ff` directly.
- cxxapi exposes `frame_schema_version` envelope, not full vocab JSON (OK for C++ role).
