//! Live streaming of `Frame`s.
//!
//! The core model serializes **directly**: the real
//! [`Frame`](crate::core::store::frame::Frame) — with its `Block`s, `Column`s,
//! and `SimBox` — implements `serde::Serialize`/`Deserialize` via the `serde`
//! feature (which `stream` enables); see [`crate::serialize`]. There is no
//! separate wire type. This module only adds the transport encoding:
//! MessagePack (default) or JSON.

use crate::core::store::frame::Frame;

/// Encoding used for a streamed `Frame`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageFormat {
    /// MessagePack (compact binary; default).
    MessagePack,
    /// JSON (text; debugging / interop).
    Json,
}

/// Error from encoding or decoding a streamed `Frame`.
#[derive(Debug)]
pub enum StreamError {
    /// Serialization to bytes failed.
    Encode(String),
    /// Deserialization from bytes failed (bad bytes, or a payload that does not
    /// rebuild into a valid `Frame`).
    Decode(String),
}

impl std::fmt::Display for StreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StreamError::Encode(m) => write!(f, "stream encode error: {m}"),
            StreamError::Decode(m) => write!(f, "stream decode error: {m}"),
        }
    }
}

impl std::error::Error for StreamError {}

/// Encode a [`Frame`] to bytes in `format`.
pub fn frame_to_bytes(frame: &Frame, format: MessageFormat) -> Result<Vec<u8>, StreamError> {
    match format {
        MessageFormat::MessagePack => {
            rmp_serde::to_vec_named(frame).map_err(|e| StreamError::Encode(e.to_string()))
        }
        MessageFormat::Json => {
            serde_json::to_vec(frame).map_err(|e| StreamError::Encode(e.to_string()))
        }
    }
}

/// Decode bytes in `format` back into a [`Frame`].
pub fn bytes_to_frame(bytes: &[u8], format: MessageFormat) -> Result<Frame, StreamError> {
    match format {
        MessageFormat::MessagePack => {
            rmp_serde::from_slice(bytes).map_err(|e| StreamError::Decode(e.to_string()))
        }
        MessageFormat::Json => {
            serde_json::from_slice(bytes).map_err(|e| StreamError::Decode(e.to_string()))
        }
    }
}
