//! The `serde` feature makes the core model serialize directly — verified here
//! with plain `serde_json`, no `stream` transport involved.

use molrs::store::block::Block;
use molrs::store::frame::Frame;
use molrs::types::{F, U};
use ndarray::Array1;

#[test]
fn frame_serde_json_roundtrip() {
    let mut frame = Frame::new();
    let mut atoms = Block::new();
    atoms
        .insert("x", Array1::from_vec(vec![1.0 as F, 2.0]).into_dyn())
        .unwrap();
    atoms
        .insert("id", Array1::from_vec(vec![0 as U, 1]).into_dyn())
        .unwrap();
    frame.insert("atoms", atoms);
    frame.meta.insert("title".into(), "t".into());

    let json = serde_json::to_string(&frame).unwrap();
    let back: Frame = serde_json::from_str(&json).unwrap();

    assert_eq!(back.get("atoms").unwrap().nrows(), Some(2));
    assert_eq!(
        back.get("atoms").unwrap().get_uint("id").unwrap(),
        frame.get("atoms").unwrap().get_uint("id").unwrap()
    );
    assert_eq!(back.meta.get("title").map(String::as_str), Some("t"));
}

#[test]
fn column_data_may_precede_dtype_for_numeric_json() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(1.0 as F).to_le_bytes());
    bytes.extend_from_slice(&(2.0 as F).to_le_bytes());
    let json = format!(
        r#"{{"blocks":{{"atoms":{{"columns":{{"x":{{"data":{:?},"shape":[2],"dtype":"float"}}}}}}}},"meta":{{}}}}"#,
        bytes
    );

    let back: Frame = serde_json::from_str(&json).unwrap();
    let x = back.get("atoms").unwrap().get_float("x").unwrap();
    assert_eq!(x.iter().copied().collect::<Vec<_>>(), vec![1.0 as F, 2.0]);
}

#[test]
fn column_data_may_precede_dtype_for_string_json() {
    let json = r#"{"blocks":{"atoms":{"columns":{"name":{"data":["C","H"],"shape":[2],"dtype":"string"}}}},"meta":{}}"#;

    let back: Frame = serde_json::from_str(json).unwrap();
    let names = back.get("atoms").unwrap().get_string("name").unwrap();
    assert_eq!(
        names.iter().cloned().collect::<Vec<_>>(),
        vec!["C".to_string(), "H".to_string()]
    );
}

#[test]
fn empty_block_preserves_explicit_shape() {
    let json = r#"{"blocks":{"grid":{"shape":[2,3],"columns":{}}},"meta":{}}"#;

    let back: Frame = serde_json::from_str(json).unwrap();
    let grid = back.get("grid").unwrap();
    assert_eq!(grid.shape(), vec![2, 3]);
    assert_eq!(grid.nrows(), Some(6));
    assert!(grid.is_empty());
}

#[test]
fn empty_table_block_preserves_explicit_nrows() {
    let json = r#"{"blocks":{"atoms":{"shape":[5],"columns":{}}},"meta":{}}"#;

    let back: Frame = serde_json::from_str(json).unwrap();
    let atoms = back.get("atoms").unwrap();
    assert_eq!(atoms.shape(), vec![5]);
    assert_eq!(atoms.nrows(), Some(5));
    assert!(atoms.is_empty());
}
