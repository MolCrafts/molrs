//! WebAssembly bindings for the molrs molecular simulation toolkit.
//!
//! Provides a JavaScript/TypeScript-friendly API for molecular data
//! manipulation, file I/O, 3D coordinate generation, and trajectory
//! analysis. Built on top of [`molrs_ffi`] handle-based architecture
//! for safe, single-threaded WASM usage.
//!
//! # Architecture
//!
//! The WASM API mirrors the core Rust data model:
//!
//! - **[`Frame`]** -- hierarchical container mapping string keys
//!   (e.g., `"atoms"`, `"bonds"`) to typed [`Block`]s.
//! - **[`Block`]** -- column-oriented data store with typed columns
//!   (`F`, `i32`, `u32`, `string`). Float columns map to
//!   `Float32Array` in default builds and `Float64Array` with the
//!   `f64` feature.
//! - **[`Box`]** (exported as `Box` in JS) -- simulation box defining
//!   periodic boundary conditions and coordinate transformations.
//! - **[`WasmArray`]** -- owned float array with ndarray-compatible shape
//!   metadata for passing multi-dimensional data across the WASM boundary.
//!
//! # Modules
//!
//! | JS module  | Purpose |
//! |------------|---------|
//! | `core`     | Frame, Block, Box, WasmArray |
//! | `io`       | File readers/writers (XYZ, PDB, LAMMPS, SMILES, Zarr) |
//! | `embed`    | 3D coordinate generation from molecular graphs |
//! | `compute`  | Analysis: RDF, MSD, Cluster, neighbor search |
//!
//! # Quick start (JavaScript)
//!
//! ```js
//! import init, { parseSMILES, generate3D, writeFrame } from "molrs-wasm";
//!
//! await init();
//!
//! const ir    = parseSMILES("CCO");
//! const frame = ir.toFrame();
//! const mol3d = generate3D(frame, "fast");
//! const xyz   = writeFrame(mol3d, "xyz");
//! console.log(xyz);
//! ```

use js_sys::WebAssembly::Memory;
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

/// WASM module entry point. Installs the panic hook so that Rust panics
/// are forwarded to the browser console as readable stack traces.
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

/// Return a handle to the WASM linear memory.
///
/// Useful for advanced interop where JS code needs direct access to the
/// WASM memory buffer (e.g., for zero-copy typed-array views).
///
/// # Example (JavaScript)
///
/// ```js
/// const mem = wasmMemory();
/// const buf = new Float64Array(mem.buffer, ptr, len); // or Float32Array in default builds
/// ```
#[wasm_bindgen(js_name = wasmMemory)]
pub fn wasm_memory() -> Memory {
    wasm_bindgen::memory().unchecked_into()
}

// Module declarations
#[cfg(feature = "compute")]
mod compute;
mod core;
#[cfg(feature = "embed")]
mod embed;
#[cfg(feature = "io")]
mod io;
#[cfg(feature = "smiles")]
mod smiles;

// Re-exports following molrs-core layout.
#[cfg(feature = "compute")]
pub use compute::*;
pub use core::{Block, Box, Frame, Grid, WasmArray};
#[cfg(feature = "embed")]
pub use embed::*;
#[cfg(feature = "io")]
pub use io::*;
#[cfg(feature = "smiles")]
pub use smiles::*;
