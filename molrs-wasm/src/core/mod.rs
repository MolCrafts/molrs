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
//! `Rc<RefCell<FFIStore>>`) that is **not** `Send + Sync`. This is
//! intentional: WebAssembly is single-threaded, so no locking overhead
//! is required. Native multi-threaded consumers should use
//! `Arc<Mutex<FFIStore>>` instead.

use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::JsValue;

use molrs_ffi::{FfiError, Store as FFIStore};

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

/// Shared store for single-threaded WASM use.
///
/// Wraps the FFI [`Store`](molrs_ffi::Store) in `Rc<RefCell<...>>` so
/// that multiple WASM-side handles ([`Frame`], [`Block`]) can share
/// ownership without `Send + Sync` bounds (which are unnecessary in
/// the single-threaded WASM environment).
pub(crate) type SharedStore = Rc<RefCell<FFIStore>>;

/// Convert an [`FfiError`] into a [`JsValue`] string for propagation
/// to JavaScript as a thrown exception.
pub(crate) fn js_err(err: FfiError) -> JsValue {
    JsValue::from_str(&err.to_string())
}
