//! File I/O for the C++ engine — the CXX face of `molrs::io`: XYZ / ExtXYZ
//! frames, single-frame `*.mrec` stores and the streaming `*.mrec` trajectory
//! writer.
//!
//! Every writer builds a transient molrs `Frame` from the caller's flat buffers
//! (atomic numbers become `element` symbols through molrs' periodic table) and
//! hands it to the molrs writer; every reader returns a fresh [`FrameRef`].

use std::fs::{File, OpenOptions};
use std::io::BufWriter;

use molrs::io::data::xyz::write_xyz_frame;
use molrs::io::mrec::{
    FrameSequenceWriter, SequenceSchema, open_trajectory_sequence, write_trajectory_file,
};
use molrs::spatial::SimBox;
use molrs::store::{Block, Frame, Trajectory};
use molrs::system::Element;
use ndarray::{Array1, Array2, ArrayD};

use crate::bridge;
use crate::frame::{FrameRef, meta_from_entry};

fn symbol_for_z(z: i32) -> Result<&'static str, String> {
    let element = u8::try_from(z)
        .ok()
        .and_then(Element::by_number)
        .ok_or_else(|| format!("symbol_for_z: invalid atomic number {z}"))?;
    Ok(element.symbol())
}

/// Build a frame with `atoms.{element,x,y,z}` (+ simbox) for XYZ/Zarr writing.
///
/// `element` is derived from the atomic number `type_id` via [`symbol_for_z`].
fn frame_with_elements(
    type_id: &[i32],
    x: &[f64],
    y: &[f64],
    z: &[f64],
    box_mat: &[f64],
) -> Result<Frame, String> {
    let n = type_id.len();
    let symbols: Result<Vec<String>, String> = type_id
        .iter()
        .map(|&z| symbol_for_z(z).map(|s| s.to_string()))
        .collect();
    let symbols = symbols?;
    let mut atoms = Block::new();
    atoms
        .insert(
            "element",
            Array1::from_vec(symbols).into_dyn() as ArrayD<String>,
        )
        .map_err(|e| format!("frame_with_elements insert element: {e}"))?;
    atoms
        .insert("x", Array1::from_vec(x[..n].to_vec()).into_dyn())
        .map_err(|e| format!("frame_with_elements insert x: {e}"))?;
    atoms
        .insert("y", Array1::from_vec(y[..n].to_vec()).into_dyn())
        .map_err(|e| format!("frame_with_elements insert y: {e}"))?;
    atoms
        .insert("z", Array1::from_vec(z[..n].to_vec()).into_dyn())
        .map_err(|e| format!("frame_with_elements insert z: {e}"))?;
    let mut frame = Frame::new();
    frame.insert("atoms", atoms);
    if box_mat.len() >= 9 {
        let h = Array2::from_shape_vec((3, 3), box_mat[..9].to_vec())
            .map_err(|e| format!("frame_with_elements box reshape: {e}"))?;
        if let Ok(sb) = SimBox::new(h, Array1::zeros(3), [true, true, true]) {
            frame.simbox = Some(sb);
        }
    }
    Ok(frame)
}

fn write_xyz_path(path: &str, frame: &Frame, append: bool) -> Result<(), String> {
    let f = if append {
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|err| format!("open {path}: {err}"))?
    } else {
        File::create(path).map_err(|err| format!("create {path}: {err}"))?
    };
    let mut w = BufWriter::new(f);
    write_xyz_frame(&mut w, frame).map_err(|err| err.to_string())
}

/// Write one frame with exact-dtype metadata.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_frame_xyz_typed(
    path: &str,
    type_id: &[i32],
    x: &[f64],
    y: &[f64],
    z: &[f64],
    box_mat: &[f64],
    meta: Vec<bridge::ffi::MetaEntry>,
    append: bool,
) -> Result<(), String> {
    let mut frame = frame_with_elements(type_id, x, y, z, box_mat)?;
    for entry in meta {
        let (key, value) = meta_from_entry(entry)?;
        frame.meta.insert(key, value);
    }
    write_xyz_path(path, &frame, append)
}

/// Write one frame (+ named per-atom fields) to a single-frame Zarr store.
///
/// Builds a transient `Frame` (element+x/y/z + simbox + one column per
/// `field_names[i]` from `field_data` reshaped `[n_fields, n_atoms]`), wraps it
/// as a single-frame `Trajectory`, and persists via `write_trajectory_file`.
/// Replaces the old per-record Zarr writer (Atomiverse's polyethylene checkpoint).
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_frame(
    path: &str,
    type_id: &[i32],
    x: &[f64],
    y: &[f64],
    z: &[f64],
    box_mat: &[f64],
    field_names: Vec<String>,
    field_data: &[f64],
) -> Result<(), String> {
    let n = type_id.len();
    let mut frame = frame_with_elements(type_id, x, y, z, box_mat)?;
    if let Some(atoms) = frame.get_mut("atoms") {
        for (fi, name) in field_names.iter().enumerate() {
            let base = fi * n;
            if base + n <= field_data.len() {
                atoms
                    .insert(
                        name.as_str(),
                        Array1::from_vec(field_data[base..base + n].to_vec()).into_dyn(),
                    )
                    .map_err(|e| format!("write_frame insert {name}: {e}"))?;
            }
        }
    }
    let traj = Trajectory::from_frames(vec![frame]);
    write_trajectory_file(path, &traj, None).map_err(|e| format!("write_frame: {e}"))
}

/// Read the first frame of a store into a fresh `FrameRef`.
///
/// Used by Atomiverse checkpoint reload (`cpu::ZarrReader`): stage 1 of a long
/// bench writes its end-state via [`write_frame`], then later debug
/// iterations call this to skip stage 1. Only frame 0 is decoded — the store
/// is opened as a lazy cursor, never materialized. The returned `FrameRef` is
/// populated via `with_mut` on a fresh standalone store — readers
/// (`frame_column_f64`, `frame_box`, etc.) see exactly the columns and simbox
/// that were stored.
pub(crate) fn read_first_frame(path: &str) -> Result<Box<FrameRef>, String> {
    let sequence = open_trajectory_sequence(path).map_err(|e| format!("read_first_frame: {e}"))?;
    let frame = sequence
        .frame(0)
        .map_err(|e| format!("read_first_frame: {e}"))?
        .ok_or_else(|| "read_first_frame: empty trajectory".to_string())?;
    let inner = molrs_ffi::FrameRef::new_standalone();
    inner
        .with_mut(|f| {
            *f = frame;
        })
        .map_err(|e| format!("read_first_frame: populate: {e}"))?;
    Ok(Box::new(FrameRef(inner)))
}

/// The engine's streaming trajectory writer: a `FrameSequenceWriter` behind
/// an opaque CXX handle. `None` once closed, so a use after close is an error
/// rather than a panic across the seam.
pub struct TrajectoryWriterRef(Option<FrameSequenceWriter>);

fn configure_writer(
    writer: FrameSequenceWriter,
    flush_every: u64,
    durable: bool,
) -> Result<FrameSequenceWriter, String> {
    let writer = writer.with_durable(durable);
    if flush_every == 0 {
        return Ok(writer);
    }
    writer
        .with_flush_every(flush_every)
        .map_err(|e| format!("trajectory_writer: {e}"))
}

/// Mint a `*.mrec` trajectory at `path`, pinned to the blocks and columns of
/// `schema_from`.
pub(crate) fn trajectory_writer_create(
    path: &str,
    schema_from: &FrameRef,
    flush_every: u64,
    durable: bool,
) -> Result<Box<TrajectoryWriterRef>, String> {
    let schema = schema_from
        .0
        .with(SequenceSchema::from_frame)
        .map_err(|e| format!("trajectory_writer_create: {e}"))?
        .map_err(|e| format!("trajectory_writer_create: {e}"))?;
    let writer = FrameSequenceWriter::create_at(path, schema)
        .map_err(|e| format!("trajectory_writer_create: {e}"))?;
    Ok(Box::new(TrajectoryWriterRef(Some(configure_writer(
        writer,
        flush_every,
        durable,
    )?))))
}

/// Reattach to the trajectory at `path` and continue after its last committed
/// frame; whatever a crash left past the commit marker is rolled back first.
pub(crate) fn trajectory_writer_open(
    path: &str,
    flush_every: u64,
    durable: bool,
) -> Result<Box<TrajectoryWriterRef>, String> {
    let writer =
        FrameSequenceWriter::open_at(path).map_err(|e| format!("trajectory_writer_open: {e}"))?;
    Ok(Box::new(TrajectoryWriterRef(Some(configure_writer(
        writer,
        flush_every,
        durable,
    )?))))
}

/// Buffer one frame at `step` (with `time` in fs when `has_time`).
pub(crate) fn trajectory_writer_append(
    writer: &mut TrajectoryWriterRef,
    fref: &FrameRef,
    step: i64,
    time: f64,
    has_time: bool,
) -> Result<(), String> {
    let inner = writer
        .0
        .as_mut()
        .ok_or_else(|| "trajectory_writer_append: writer is closed".to_string())?;
    let time = has_time.then_some(time);
    fref.0
        .with(|frame| inner.append_at(frame, step, time))
        .map_err(|e| format!("trajectory_writer_append: {e}"))?
        .map_err(|e| format!("trajectory_writer_append: {e}"))
}

/// Commit every buffered frame (durably, unless the writer was opened with
/// `durable == false`).
pub(crate) fn trajectory_writer_flush(writer: &mut TrajectoryWriterRef) -> Result<(), String> {
    writer
        .0
        .as_mut()
        .ok_or_else(|| "trajectory_writer_flush: writer is closed".to_string())?
        .flush()
        .map_err(|e| format!("trajectory_writer_flush: {e}"))
}

/// Frames committed so far (0 for a closed writer).
pub(crate) fn trajectory_writer_committed(writer: &TrajectoryWriterRef) -> u64 {
    writer.0.as_ref().map_or(0, FrameSequenceWriter::committed)
}

/// Commit whatever is buffered and release the writer.
pub(crate) fn trajectory_writer_close(writer: Box<TrajectoryWriterRef>) -> Result<(), String> {
    match writer.0 {
        Some(inner) => inner
            .close()
            .map_err(|e| format!("trajectory_writer_close: {e}")),
        None => Ok(()),
    }
}

/// Atomic number for a chemical symbol — inverse of [`symbol_for_z`].
///
/// Delegates to molrs' canonical periodic table. No fallback: panics on an
/// unrecognized symbol, matching Atomiverse's explicit-error convention.
fn z_for_symbol(sym: &str) -> Result<i32, String> {
    Element::by_symbol(sym)
        .map(|element| i32::from(element.z()))
        .ok_or_else(|| format!("z_for_symbol: unknown element symbol '{sym}'"))
}

/// Read the first frame of an (ext)XYZ file into a materialize-ready `FrameRef`.
///
/// All parsing (atom table, `Lattice="..."` -> simbox) is done by the molrs
/// core ExtXYZ reader. The `element` column (the ExtXYZ `species` property) is
/// consumed and replaced by `atomic_number` (a UInt column), so the result satisfies the exact schema
/// that `cpu::materialize` requires (`atoms.{x,y,z,atomic_number}`). The
/// external-format column does not cross that boundary.
///
/// Z lives in `atomic_number`, not `type`: `type` is the force-field label a
/// caller owns (a String), and the vocabulary binds a key's dtype wherever it
/// appears — so writing Z there is refused outright.
pub(crate) fn xyz_read_first_frame(path: &str) -> Result<Box<FrameRef>, String> {
    let mut frame = molrs::io::data::xyz::read_xyz_frame(path)
        .map_err(|e| format!("xyz_read_first_frame: read: {e}"))?;
    let atoms = frame
        .get_mut("atoms")
        .ok_or_else(|| "xyz_read_first_frame: frame has no atoms block".to_string())?;
    let species = atoms
        .get("element")
        .and_then(|c| c.as_string())
        .ok_or_else(|| {
            "xyz_read_first_frame: atoms block has no element (ExtXYZ species) column".to_string()
        })?;
    let zs: Result<Vec<u64>, String> = species
        .iter()
        .map(|symbol| z_for_symbol(symbol).map(|z| z as u64))
        .collect();
    let zs = zs?;
    atoms
        .insert("atomic_number", Array1::from_vec(zs).into_dyn())
        .map_err(|e| format!("xyz_read_first_frame: insert atomic_number: {e}"))?;
    atoms.remove("element");

    let inner = molrs_ffi::FrameRef::new_standalone();
    inner
        .with_mut(|f| {
            *f = frame;
        })
        .map_err(|e| format!("xyz_read_first_frame: populate: {e}"))?;
    Ok(Box::new(FrameRef(inner)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::*;

    #[test]
    fn a_trajectory_writer_minted_from_a_precise_frame_rounds_its_column() {
        let dir = std::env::temp_dir().join(format!(
            "molrs-cxxapi-precision-{}.mrec",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let mut fref = frame_new();
        frame_set_column_f64(&mut fref, "atoms", "x", &[0.123_456, 1.000_49]).unwrap();
        assert!(frame_set_precision(&mut fref, "atoms", "nope", 1e-3).is_err());
        assert!(frame_set_precision(&mut fref, "atoms", "x", 0.0).is_err());
        frame_set_precision(&mut fref, "atoms", "x", 1e-3).unwrap();
        let path = dir.to_str().unwrap();
        let mut writer = trajectory_writer_create(path, &fref, 0, false).unwrap();
        trajectory_writer_append(&mut writer, &fref, 0, 0.0, false).unwrap();
        trajectory_writer_close(writer).unwrap();
        let back = read_first_frame(path).unwrap();
        let q = 2f64.powi(-10);
        assert_eq!(
            frame_column_f64(&back, "atoms", "x"),
            vec![(0.123_456 / q).round() * q, (1.000_49 / q).round() * q]
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn element_symbols_are_the_canonical_rust_table() {
        for element in Element::ALL {
            assert_eq!(
                symbol_for_z(i32::from(element.z())).unwrap(),
                element.symbol()
            );
            assert_eq!(
                z_for_symbol(element.symbol()).unwrap(),
                i32::from(element.z())
            );
        }
        assert!(symbol_for_z(0).is_err());
        assert!(symbol_for_z(119).is_err());
    }

    #[test]
    fn extxyz_boundary_emits_only_the_exchange_element_column() {
        let path = std::env::temp_dir().join(format!(
            "molrs-cxxapi-extxyz-schema-{}.xyz",
            std::process::id()
        ));
        std::fs::write(
            &path,
            concat!(
                "2\n",
                "Lattice=\"5 0 0 0 5 0 0 0 5\" ",
                "Properties=species:S:1:pos:R:3 pbc=\"T T T\"\n",
                "O 0 0 0\n",
                "H 1 0 0\n",
            ),
        )
        .unwrap();

        let frame = xyz_read_first_frame(path.to_str().unwrap()).expect("xyz_read_first_frame");
        std::fs::remove_file(path).unwrap();
        let columns = frame_block_columns(&frame, "atoms");
        assert!(columns.iter().any(|column| column == "atomic_number"));
        assert!(!columns.iter().any(|column| column == "species"));
        assert!(!columns.iter().any(|column| column == "element"));
        assert!(
            !columns.iter().any(|column| column == "type"),
            "Z is `atomic_number`; `type` is the caller's force-field label"
        );
        assert_eq!(
            frame_column_u32(&frame, "atoms", "atomic_number"),
            [8u64, 1]
        );
    }

    /// The core reader names the element column `element` whichever key the
    /// file used (ExtXYZ's `species`, or an `element` property), so the
    /// bridge sees one column either way.
    #[test]
    fn extxyz_boundary_reads_an_element_property_like_species() {
        let path = std::env::temp_dir().join(format!(
            "molrs-cxxapi-extxyz-element-key-{}.xyz",
            std::process::id()
        ));
        std::fs::write(
            &path,
            concat!(
                "1\n",
                "Lattice=\"5 0 0 0 5 0 0 0 5\" ",
                "Properties=element:S:1:pos:R:3 pbc=\"T T T\"\n",
                "H 0 0 0\n",
            ),
        )
        .unwrap();

        let frame = xyz_read_first_frame(path.to_str().unwrap()).expect("xyz_read_first_frame");
        std::fs::remove_file(path).unwrap();
        assert_eq!(frame_column_u32(&frame, "atoms", "atomic_number"), [1u64]);
    }

    #[test]
    fn typed_xyz_writer_preserves_numeric_metadata() {
        use bridge::ffi::MetaType;
        let path = std::env::temp_dir().join(format!(
            "molrs-cxxapi-typed-meta-{}.xyz",
            std::process::id()
        ));
        let mut step = empty_meta_entry("step".into(), MetaType::I64);
        step.i64_value = 9_007_199_254_740_993;
        let mut energy = empty_meta_entry("energy_eV".into(), MetaType::F64);
        energy.f64_value = -1.25;
        write_frame_xyz_typed(
            path.to_str().unwrap(),
            &[1],
            &[0.0],
            &[0.0],
            &[0.0],
            &[10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0],
            vec![step, energy],
            false,
        )
        .unwrap();

        let frame = molrs::io::data::xyz::read_xyz_frame(&path).unwrap();
        assert_eq!(
            frame.meta.get("step").unwrap().as_i64(),
            Some(9_007_199_254_740_993)
        );
        assert_eq!(frame.meta.get("energy_eV").unwrap().as_f64(), Some(-1.25));
        std::fs::remove_file(path).unwrap();
    }
}
