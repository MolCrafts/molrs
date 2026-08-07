//! Live streaming of `Frame`s.
//!
//! The core model serializes **directly**: the real [`Frame`] — with its
//! `Block`s, `Column`s, and `SimBox` — implements
//! `serde::Serialize`/`Deserialize` via the `serde` feature (which `stream`
//! enables); the impls live in the crate-private `serialize` module. There is no
//! separate wire type. This module only adds the transport encoding: MessagePack
//! (default) or JSON.

//! Beyond the encoding this module also carries the live transport that uses
//! it: [`ControlCommand`] (WASM-clean) and [`FrameServer`] (native only). They
//! live here rather than under `io` because they pull third-party runtime
//! dependencies — tokio, tungstenite, rmp-serde — that `io` must not acquire.

pub mod message;

pub use message::ControlCommand;

#[cfg(not(target_arch = "wasm32"))]
mod server;

#[cfg(not(target_arch = "wasm32"))]
pub use server::{FrameServer, SendError, ServerConfig};

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::spatial::simbox::SimBox;
    use crate::core::store::block::Block;
    use crate::types::{F, I, U};
    use ndarray::{Array1, array};

    /// Build a full Frame used by the net-streaming lossless round-trip contract
    /// (ac-002): atoms x/y/z + serial + type, bonds i/j + order, SimBox, meta.
    fn rich_frame() -> Frame {
        let mut atoms = Block::new();
        atoms
            .insert(
                "x",
                Array1::from_vec(vec![1.0 as F, 2.0 as F, 3.0 as F]).into_dyn(),
            )
            .unwrap();
        atoms
            .insert(
                "y",
                Array1::from_vec(vec![0.0 as F, 1.0 as F, 2.0 as F]).into_dyn(),
            )
            .unwrap();
        atoms
            .insert(
                "z",
                Array1::from_vec(vec![0.5 as F, 1.5 as F, 2.5 as F]).into_dyn(),
            )
            .unwrap();
        atoms
            .insert(
                "serial",
                Array1::from_vec(vec![10 as I, 20 as I, 30 as I]).into_dyn(),
            )
            .unwrap();
        atoms
            .insert("atype", Array1::from_vec(vec![6u8, 1u8, 8u8]).into_dyn())
            .unwrap();

        let mut bonds = Block::new();
        bonds
            .insert("i", Array1::from_vec(vec![0 as U, 1 as U]).into_dyn())
            .unwrap();
        bonds
            .insert("j", Array1::from_vec(vec![1 as U, 2 as U]).into_dyn())
            .unwrap();
        bonds
            .insert("order", Array1::from_vec(vec![1u8, 1u8]).into_dyn())
            .unwrap();

        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame.insert("bonds", bonds);
        frame.simbox =
            Some(SimBox::cube(10.0, array![0.0, 0.0, 0.0], [true, true, true]).expect("simbox"));
        frame.meta.insert("title", "stream-roundtrip");
        frame.meta.insert("step", 42i64);
        frame
    }

    fn assert_frame_eq(a: &Frame, b: &Frame) {
        assert_eq!(a.len(), b.len());
        assert!(a.contains_key("atoms"));
        assert!(b.contains_key("atoms"));
        assert!(a.contains_key("bonds"));
        assert!(b.contains_key("bonds"));

        let ax = a["atoms"].get_float("x").unwrap();
        let bx = b["atoms"].get_float("x").unwrap();
        assert_eq!(ax.len(), bx.len());
        for (u, v) in ax.iter().zip(bx.iter()) {
            assert!((u - v).abs() < f64::EPSILON);
        }
        let ay = a["atoms"].get_float("y").unwrap();
        let by = b["atoms"].get_float("y").unwrap();
        for (u, v) in ay.iter().zip(by.iter()) {
            assert!((u - v).abs() < f64::EPSILON);
        }
        let az = a["atoms"].get_float("z").unwrap();
        let bz = b["atoms"].get_float("z").unwrap();
        for (u, v) in az.iter().zip(bz.iter()) {
            assert!((u - v).abs() < f64::EPSILON);
        }

        let aserial = a["atoms"].get_int("serial").unwrap();
        let bserial = b["atoms"].get_int("serial").unwrap();
        assert_eq!(aserial.as_slice().unwrap(), bserial.as_slice().unwrap());

        let atype = a["atoms"].get_u8("atype").unwrap();
        let btype = b["atoms"].get_u8("atype").unwrap();
        assert_eq!(atype.as_slice().unwrap(), btype.as_slice().unwrap());

        let bi = a["bonds"].get_uint("i").unwrap();
        let bj = b["bonds"].get_uint("i").unwrap();
        assert_eq!(bi.as_slice().unwrap(), bj.as_slice().unwrap());

        assert!(a.simbox.is_some());
        assert!(b.simbox.is_some());
        assert_eq!(
            a.meta.get("title").and_then(|m| m.as_str()),
            b.meta.get("title").and_then(|m| m.as_str())
        );
        assert_eq!(
            a.meta.get("step").and_then(|m| m.as_i64()),
            b.meta.get("step").and_then(|m| m.as_i64())
        );
    }

    #[test]
    fn frame_messagepack_roundtrip() {
        let frame = rich_frame();
        let bytes = frame_to_bytes(&frame, MessageFormat::MessagePack).expect("encode");
        let back = bytes_to_frame(&bytes, MessageFormat::MessagePack).expect("decode");
        assert_frame_eq(&frame, &back);
    }

    #[test]
    fn frame_json_roundtrip() {
        let frame = rich_frame();
        let bytes = frame_to_bytes(&frame, MessageFormat::Json).expect("encode");
        let back = bytes_to_frame(&bytes, MessageFormat::Json).expect("decode");
        assert_frame_eq(&frame, &back);
    }
}
