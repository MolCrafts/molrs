//! FFI layer for molrs with handle-based abstractions.
//!
//! This crate provides a stable, handle-based API for Python and WASM bindings.
//! It separates the Rust-idiomatic core API from cross-language FFI concerns.
//!
//! # Usage
//!
//! A consumer holds a [`FrameRef`] (a frame id paired with a shared `Store`) and
//! borrows columns zero-copy through [`BlockRef`]. See `docs/interop.md` for the
//! full recipe and the data contract.
//!
//! ```no_run
//! use molrs_ffi::FrameRef;
//!
//! let frame = FrameRef::new_standalone();      // a frame inside a fresh SharedStore
//! // ... populate it via frame.with_mut(|f| ...) ...
//! if let Ok(atoms) = frame.block("atoms") {
//!     // zero-copy borrow of the uint atom-id column (the uint-index contract)
//!     let n_ids = atoms.borrow_u("id", |ids, _shape| ids.len()).ok().flatten();
//!     let _ = n_ids;
//! }
//! ```

pub mod abi;
mod error;
#[cfg(feature = "ff")]
mod forcefield;
mod handle;
mod shared;
mod store;

pub use error::FfiError;
#[cfg(feature = "ff")]
pub use forcefield::ForceFieldRef;
pub use handle::{BlockHandle, FrameId};
pub use shared::{BlockRef, FrameRef, SharedStore, new_shared};
pub use store::Store;
