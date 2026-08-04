//! WebSocket Frame streaming and bidirectional control.
//!
//! Enable with the `net` Cargo feature (intentionally **not** part of `full`).
//!
//! - [`message`] — WASM-clean [`ControlCommand`] wire types (serde only).
//! - [`FrameServer`] — native WebSocket broadcaster (not available on `wasm32`).
//!
//! Wire encoding reuses [`crate::stream`] (`frame_to_bytes` / `bytes_to_frame`);
//! there is no separate WireFrame type.

pub mod message;

pub use message::ControlCommand;

#[cfg(not(target_arch = "wasm32"))]
mod server;

#[cfg(not(target_arch = "wasm32"))]
pub use server::{FrameServer, SendError, ServerConfig};
