//! Round-trip tests for the `stream` wire codec.
//!
//! The wire format is an in-memory serialization, not a file format, so
//! building synthetic `Frame`s here is correct — the tests-data rule applies to
//! file-format readers, not to this codec.

use molrs::spatial::region::simbox::SimBox;
use molrs::store::block::{Block, DType};
use molrs::store::frame::Frame;
use molrs::stream::{MessageFormat, bytes_to_frame, frame_to_bytes};
use molrs::types::{F, U};
use ndarray::{Array1, arr1, arr2};

/// A frame exercising every dtype, a multi-axis (volumetric) block, a box, and
/// metadata — all under arbitrary block/column names (no privileged fields).
fn sample_frame() -> Frame {
    let mut frame = Frame::new();

    let mut atoms = Block::new();
    atoms
        .insert("x", Array1::from_vec(vec![0.0 as F, 1.5, 3.0]).into_dyn())
        .unwrap();
    atoms
        .insert("y", Array1::from_vec(vec![0.0 as F, 0.5, 1.0]).into_dyn())
        .unwrap();
    atoms
        .insert("z", Array1::from_vec(vec![0.0 as F, -0.5, -1.0]).into_dyn())
        .unwrap();
    atoms
        .insert("id", Array1::from_vec(vec![0 as U, 1, 2]).into_dyn())
        .unwrap();
    atoms
        .insert("kind", Array1::from_vec(vec![6u8, 1, 8]).into_dyn())
        .unwrap();
    atoms
        .insert(
            "frozen",
            Array1::from_vec(vec![true, false, true]).into_dyn(),
        )
        .unwrap();
    atoms
        .insert(
            "name",
            Array1::from_vec(vec!["C".to_string(), "H".into(), "O".into()]).into_dyn(),
        )
        .unwrap();
    frame.insert("atoms", atoms);

    let mut bonds = Block::new();
    bonds
        .insert("atomi", Array1::from_vec(vec![0 as U, 0]).into_dyn())
        .unwrap();
    bonds
        .insert("atomj", Array1::from_vec(vec![1 as U, 2]).into_dyn())
        .unwrap();
    bonds
        .insert("order", Array1::from_vec(vec![1.0 as F, 2.0]).into_dyn())
        .unwrap();
    frame.insert("bonds", bonds);

    // A volumetric block: structural shape [2,2,2], one flat rho column of len 8.
    let mut density = Block::new();
    let rho: Vec<F> = (0..8).map(|i| i as F * 0.25).collect();
    density
        .insert("rho", Array1::from_vec(rho).into_dyn())
        .unwrap();
    density.set_shape(&[2, 2, 2]).unwrap();
    frame.insert("density", density);

    let h = arr2(&[[10.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 8.0]]);
    frame.simbox = Some(SimBox::new(h, arr1(&[0.0, 0.0, 0.0]), [true, true, false]).unwrap());

    frame.meta.insert("title", "sample");
    frame
}

fn assert_roundtrip(fmt: MessageFormat) {
    let frame = sample_frame();
    let bytes = frame_to_bytes(&frame, fmt).unwrap();
    let back = bytes_to_frame(&bytes, fmt).unwrap();

    // Block set + per-block count.
    assert_eq!(back.len(), 3);
    for name in ["atoms", "bonds", "density"] {
        assert!(back.contains_key(name), "missing block {name}");
        assert_eq!(
            back.get(name).unwrap().nrows(),
            frame.get(name).unwrap().nrows(),
            "nrows mismatch for {name}"
        );
    }

    // Dtypes survive.
    let a = back.get("atoms").unwrap();
    assert_eq!(a.get("x").unwrap().dtype(), DType::Float);
    assert_eq!(a.get("id").unwrap().dtype(), DType::UInt);
    assert_eq!(a.get("kind").unwrap().dtype(), DType::U8);
    assert_eq!(a.get("frozen").unwrap().dtype(), DType::Bool);
    assert_eq!(a.get("name").unwrap().dtype(), DType::String);

    // Float values within epsilon.
    let x0 = frame.get("atoms").unwrap().get_float("x").unwrap();
    let x1 = a.get_float("x").unwrap();
    assert_eq!(x0.shape(), x1.shape());
    for (p, q) in x0.iter().zip(x1.iter()) {
        assert!((p - q).abs() < F::EPSILON, "x mismatch {p} vs {q}");
    }

    // Uint / bool / string values.
    assert_eq!(
        frame.get("bonds").unwrap().get_uint("atomi").unwrap(),
        back.get("bonds").unwrap().get_uint("atomi").unwrap()
    );
    assert_eq!(
        a.get_bool("frozen")
            .unwrap()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        vec![true, false, true]
    );
    assert_eq!(
        a.get_string("name")
            .unwrap()
            .iter()
            .cloned()
            .collect::<Vec<_>>(),
        vec!["C", "H", "O"]
    );

    // Volumetric block: structural shape preserved and values intact.
    assert_eq!(back.get("density").unwrap().shape(), vec![2, 2, 2]);
    let r0 = frame.get("density").unwrap().get_float("rho").unwrap();
    let r1 = back.get("density").unwrap().get_float("rho").unwrap();
    for (p, q) in r0.iter().zip(r1.iter()) {
        assert!((p - q).abs() < F::EPSILON);
    }

    // Box: vectors == H, boundary == pbc.
    let sb0 = frame.simbox.as_ref().unwrap();
    let sb1 = back.simbox.as_ref().unwrap();
    assert_eq!(sb1.pbc(), sb0.pbc());
    let (h0, h1) = (sb0.h_view(), sb1.h_view());
    for i in 0..3 {
        for j in 0..3 {
            assert!((h0[[i, j]] - h1[[i, j]]).abs() < F::EPSILON);
        }
    }

    // Metadata.
    assert_eq!(
        back.meta.get("title").and_then(|value| value.as_str()),
        Some("sample")
    );
}

#[test]
fn roundtrip_messagepack() {
    assert_roundtrip(MessageFormat::MessagePack);
}

#[test]
fn roundtrip_json() {
    assert_roundtrip(MessageFormat::Json);
}

#[test]
fn empty_frame_roundtrips() {
    let frame = Frame::new();
    for fmt in [MessageFormat::MessagePack, MessageFormat::Json] {
        let bytes = frame_to_bytes(&frame, fmt).unwrap();
        let back = bytes_to_frame(&bytes, fmt).unwrap();
        assert!(back.is_empty());
        assert!(back.simbox.is_none());
    }
}
