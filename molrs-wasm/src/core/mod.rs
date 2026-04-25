//! Core data model types exported to JavaScript.
//!
//! This module re-exports the four fundamental types that make up the
//! molrs WASM data layer:
//!
//! - [`Frame`] -- hierarchical container of named [`Block`]s, plus an
//!   optional [`Box`] (simulation box).
//! - [`Block`] -- column-oriented data store with typed arrays.
//! - [`Box`] -- parallelepiped simulation box with periodic boundary
//!   conditions (PBC).
//! - [`WasmArray`] -- owned float array with shape metadata for passing
//!   multi-dimensional numeric data across the WASM boundary.
//!
//! # Internal details
//!
//! All mutable state is managed through a [`SharedStore`] (an
//! `Rc<RefCell<Store>>`) that is **not** `Send + Sync`. This is
//! intentional: WebAssembly is single-threaded, so no locking overhead
//! is required. Native multi-threaded consumers should use
//! `Arc<Mutex<Store>>` instead.
//!
//! The [`SharedStore`] alias and its paired [`FrameRef`](molrs_ffi::FrameRef) /
//! [`BlockRef`](molrs_ffi::BlockRef) wrappers live in `molrs-ffi` so every
//! binding layer (wasm, python, capi) consumes the same canonical
//! lifetime-management plumbing.

use wasm_bindgen::JsValue;

use molrs_ffi::FfiError;

pub mod block;
pub mod frame;
pub mod grid;
pub mod region;
pub mod types;

pub use block::Block;
pub use frame::Frame;
pub use grid::Grid;
pub use region::simbox::Box;
pub use types::WasmArray;

/// Convert an [`FfiError`] into a [`JsValue`] string for propagation
/// to JavaScript as a thrown exception.
pub(crate) fn js_err(err: FfiError) -> JsValue {
    JsValue::from_str(&err.to_string())
}
