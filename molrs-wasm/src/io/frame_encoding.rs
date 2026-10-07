//! The `molrs::stream` wire encodings for the WASM API — the face of
//! `molrs::io::frame_encoding` (`stream` feature).
//!
//! A page subscribed to a live run decodes payloads with these and never
//! re-derives the layout in JavaScript.
//!
//! | JS | molrs |
//! |----|-------|
//! | `readMsgpackFrameBytes`, `writeMsgpackFrameBytes` | MessagePack wire bytes |
//! | `readJsonFrameStr`, `writeJsonFrameStr` | JSON wire text |

use wasm_bindgen::prelude::*;

use crate::core::frame::Frame;

/// Rebuild a [`Frame`] from `molrs::stream` MessagePack wire bytes — what a
/// publisher puts on the socket, so a page subscribed to a live run decodes
/// payloads with this and never re-derives the layout in JavaScript. The
/// inverse of `writeMsgpackFrameBytes`.
///
/// # Example (JavaScript)
///
/// ```js
/// socket.onmessage = (ev) => {
///   const frame = readMsgpackFrameBytes(new Uint8Array(ev.data));
/// };
/// ```
#[wasm_bindgen(js_name = readMsgpackFrameBytes)]
pub fn read_msgpack_frame_bytes(data: &[u8]) -> Result<Frame, JsValue> {
    let rs_frame =
        molrs::io::read_msgpack_frame_bytes(data).map_err(|e| JsValue::from_str(&e.to_string()))?;
    Frame::from_rs(rs_frame)
}

/// Encode `frame` in the `molrs::stream` MessagePack wire encoding — what a
/// publisher puts on the socket. The inverse of `readMsgpackFrameBytes`.
#[wasm_bindgen(js_name = writeMsgpackFrameBytes)]
pub fn write_msgpack_frame_bytes(frame: &Frame) -> Result<Vec<u8>, JsValue> {
    frame.with_frame(|f| {
        molrs::io::write_msgpack_frame_bytes(f).map_err(|e| JsValue::from_str(&e.to_string()))
    })
}

/// Rebuild a [`Frame`] from `molrs::stream` JSON wire text. The inverse of
/// `writeJsonFrameStr`.
#[wasm_bindgen(js_name = readJsonFrameStr)]
pub fn read_json_frame_str(text: &str) -> Result<Frame, JsValue> {
    let rs_frame =
        molrs::io::read_json_frame_str(text).map_err(|e| JsValue::from_str(&e.to_string()))?;
    Frame::from_rs(rs_frame)
}

/// Encode `frame` in the `molrs::stream` JSON wire encoding. The inverse of
/// `readJsonFrameStr`.
#[wasm_bindgen(js_name = writeJsonFrameStr)]
pub fn write_json_frame_str(frame: &Frame) -> Result<String, JsValue> {
    frame.with_frame(|f| {
        molrs::io::write_json_frame_str(f).map_err(|e| JsValue::from_str(&e.to_string()))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::test_support::float_col;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn stream_bytes_round_trip_through_io() {
        use crate::core::nd_array::JsFloatArray;

        let frame = Frame::new();
        let mut atoms = frame.create_block("atoms").expect("atoms block");
        let x = JsFloatArray::from(&[1.0, 4.0][..]);
        atoms
            .set(
                "x",
                wasm_bindgen::JsCast::unchecked_into(JsValue::from(x)),
                None,
            )
            .expect("x");

        let from_msgpack =
            read_msgpack_frame_bytes(&write_msgpack_frame_bytes(&frame).expect("encode"))
                .expect("decode");
        let from_json =
            read_json_frame_str(&write_json_frame_str(&frame).expect("encode")).expect("decode");
        for back in [from_msgpack, from_json] {
            let x = float_col(&back.get("atoms").expect("atoms"), "x");
            assert_eq!(x.length(), 2);
            assert_eq!(x.get_index(0), 1.0);
            assert_eq!(x.get_index(1), 4.0);
        }
    }
}
