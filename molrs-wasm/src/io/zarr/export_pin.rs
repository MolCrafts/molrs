//! Source pin for the wasm Zarr adapter's public JS name and Rust import.
//!
//! Scans `mod.rs` only — this file's literals must not live in the production
//! unit, or the pin would be vacuously green.

const SRC: &str = include_str!("mod.rs");

#[test]
fn js_name_is_trajectory_reader() {
    assert!(
        SRC.contains("js_name = TrajectoryReader"),
        "wasm JS export must be js_name = TrajectoryReader (wraps FrameSequence)"
    );
}

#[test]
fn js_name_is_not_record_reader() {
    assert!(
        !SRC.contains("js_name = RecordReader"),
        "js_name = RecordReader is the retired JS export; use TrajectoryReader"
    );
}

#[test]
fn js_export_is_not_frame_reader() {
    assert!(
        !SRC.contains("js_name = FrameReader"),
        "do not export a FrameReader JS name; the wrapper is TrajectoryReader"
    );
    assert!(
        !SRC.contains("js_class = FrameReader"),
        "do not export a FrameReader JS class; the wrapper is TrajectoryReader"
    );
}

#[test]
fn frame_sequence_imported_from_io_mrec() {
    assert!(
        SRC.contains("use molrs::io::mrec::FrameSequence"),
        "FrameSequence must be imported from molrs::io::mrec"
    );
    assert!(
        !SRC.contains("use molrs::io::zarr::FrameSequence"),
        "FrameSequence must not be imported from molrs::io::zarr"
    );
}
