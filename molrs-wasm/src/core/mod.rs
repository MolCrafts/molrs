//! Core data model exported to JavaScript — the WASM face of molrs' `store`,
//! `system` and `spatial` domains.
//!
//! - [`Frame`] -- hierarchical container of named [`Block`]s, plus an
//!   optional [`Box`] (simulation box).
//! - [`Block`] -- column-oriented data store with typed arrays.
//! - [`NDArray`] -- owned float array with shape metadata for passing
//!   multi-dimensional numeric data across the WASM boundary.
//! - `schema` -- the Frame schema vocabulary (`schemaDocument`, …).
//! - [`Topology`] -- the bond graph of a frame (`molrs::system::Topology`).
//! - `covalentRadius` -- per-element data (`molrs::system::Element`).
//! - `spatial` -- the simulation [`Box`], regions, [`Mesh`] and neighbor
//!   search.
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
//!
//! [`SharedStore`]: molrs_ffi::SharedStore

use wasm_bindgen::JsValue;

use molrs_ffi::FfiError;

pub(crate) mod block;
pub(crate) mod element;
pub(crate) mod frame;
pub(crate) mod schema;
pub(crate) mod spatial;
pub(crate) mod topology;
pub(crate) mod types;

pub use block::Block;
pub use element::covalent_radius;
pub use frame::Frame;
pub use schema::*;
pub use spatial::*;
pub use topology::Topology;
pub use types::NDArray;

/// Convert an [`FfiError`] into a [`JsValue`] string for propagation
/// to JavaScript as a thrown exception.
pub(crate) fn js_err(err: FfiError) -> JsValue {
    JsValue::from_str(&err.to_string())
}
