//! A [`Frame`] as bytes or text in memory, in the encodings
//! [`crate::stream`] puts on the wire: MessagePack (the default) and JSON.
//!
//! The codecs live in `stream` (it builds without `io`); these are their one
//! public path.

use crate::core::Frame;
use crate::io::invalid_data;

/// Encode a [`Frame`] as MessagePack bytes — the wire encoding a
/// [`Publisher`](crate::stream::Publisher) sends by default.
pub fn write_msgpack_frame_bytes(frame: &Frame) -> std::io::Result<Vec<u8>> {
    crate::stream::encode_msgpack_frame(frame).map_err(invalid_data)
}

/// Decode a [`Frame`] from MessagePack bytes — the inverse of
/// [`write_msgpack_frame_bytes`].
pub fn read_msgpack_frame_bytes(bytes: &[u8]) -> std::io::Result<Frame> {
    crate::stream::decode_msgpack_frame(bytes).map_err(invalid_data)
}

/// Encode a [`Frame`] as JSON text — the debugging / interop wire encoding.
pub fn write_json_frame_str(frame: &Frame) -> std::io::Result<String> {
    crate::stream::encode_json_frame(frame).map_err(invalid_data)
}

/// Decode a [`Frame`] from JSON text — the inverse of
/// [`write_json_frame_str`].
pub fn read_json_frame_str(text: &str) -> std::io::Result<Frame> {
    crate::stream::decode_json_frame(text).map_err(invalid_data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_frame_round_trips_through_both_encodings() {
        let frame = crate::io::read_xyz_bytes(b"1\nH\nH 0 0 0\n").unwrap();
        let back = read_msgpack_frame_bytes(&write_msgpack_frame_bytes(&frame).unwrap()).unwrap();
        assert_eq!(back.keys().count(), frame.keys().count());
        let back = read_json_frame_str(&write_json_frame_str(&frame).unwrap()).unwrap();
        assert_eq!(back.keys().count(), frame.keys().count());
    }

    #[test]
    fn malformed_input_is_invalid_data() {
        let err = read_msgpack_frame_bytes(b"\xc1").unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        let err = read_json_frame_str("{").unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }
}
