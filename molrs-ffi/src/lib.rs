//! FFI layer for molrs with handle-based abstractions.
//!
//! This crate provides a stable, handle-based API for Python and WASM bindings.
//! It separates the Rust-idiomatic core API from cross-language FFI concerns.

mod error;
mod handle;
mod store;

pub use error::FfiError;
pub use handle::{BlockHandle, FrameId};
pub use store::Store;
