//! File I/O for the C++ engine — the CXX face of `molrs::io`: XYZ / ExtXYZ
//! frames, `*.mrec` frame records, frames of a `*.mrec` trajectory and the
//! streaming `*.mrec` trajectory writer.
//!
//! Every writer builds a transient molrs `Frame` from the caller's flat buffers
//! (atomic numbers become `element` symbols through molrs' periodic table) and
//! hands it to the molrs writer; every reader returns a fresh [`FrameRef`].

use std::fs::{File, OpenOptions};
use std::io::BufWriter;

use molrs::core::Element;
use molrs::core::Frame;
use molrs::io::mrec::{MrecReader, MrecWriter, SequenceSchema};
use molrs::io::writer::{FrameWriter, Writer};
use molrs::io::xyz::XyzWriter;
use ndarray::{Array1, ArrayD};

use crate::bridge;
use crate::frame::{FrameRef, coords_frame, meta_from_keyed_value};

fn symbol_for_z(z: i32) -> Result<&'static str, String> {
    let element = u8::try_from(z)
        .ok()
        .and_then(Element::by_number)
        .ok_or_else(|| format!("symbol_for_z: invalid atomic number {z}"))?;
    Ok(element.symbol())
}

/// A frame with `atoms.{element, x, y, z}` (+ box) for the XYZ / `*.mrec`
/// writers: [`coords_frame`] plus the `element` symbols of the atomic
/// numbers `atomic_number`.
fn frame_with_elements(
    atomic_number: &[i32],
    x: &[f64],
    y: &[f64],
    z: &[f64],
    h: &[f64],
) -> Result<Frame, String> {
    if atomic_number.len() != x.len() {
        return Err(format!(
            "{} atomic numbers for {} atoms",
            atomic_number.len(),
            x.len()
        ));
    }
    let symbols = atomic_number
        .iter()
        .map(|&z| symbol_for_z(z).map(str::to_string))
        .collect::<Result<Vec<String>, String>>()?;
    let mut frame = coords_frame(x, y, z, h)?;
    frame
        .get_mut("atoms")
        .expect("coords_frame always has an atoms block")
        .insert(
            "element",
            Array1::from_vec(symbols).into_dyn() as ArrayD<String>,
        )
        .map_err(|e| format!("insert element: {e}"))?;
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
    XyzWriter::new(BufWriter::new(f))
        .write(frame)
        .map_err(|err| err.to_string())
}

/// Write one frame (element + coords + box + exact-dtype metadata) to an
/// XYZ / ExtXYZ file through molrs `XyzWriter`; `append` adds it after the
/// frames already there, otherwise the file is truncated.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_xyz_frame(
    path: &str,
    atomic_number: &[i32],
    x: &[f64],
    y: &[f64],
    z: &[f64],
    h: &[f64],
    meta: Vec<bridge::ffi::KeyedMetaValue>,
    append: bool,
) -> Result<(), String> {
    let mut frame = frame_with_elements(atomic_number, x, y, z, h)
        .map_err(|e| format!("write_xyz_frame: {e}"))?;
    for entry in meta {
        let (key, value) = meta_from_keyed_value(entry)?;
        frame.meta.insert(key, value);
    }
    write_xyz_path(path, &frame, append)
}

/// Write one frame (+ named per-atom fields) as a `*.mrec` record whose
/// `frame` section is that frame — molrs `io::write_mrec_frame`.
///
/// `field_data` is `[n_fields, n_atoms]` row-major, one row per
/// `field_names[i]`; a length other than `n_fields * n_atoms` is an error.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_mrec_frame(
    path: &str,
    atomic_number: &[i32],
    x: &[f64],
    y: &[f64],
    z: &[f64],
    h: &[f64],
    field_names: Vec<String>,
    field_data: &[f64],
) -> Result<(), String> {
    let n = atomic_number.len();
    if field_data.len() != field_names.len() * n {
        return Err(format!(
            "write_mrec_frame: {} field values for {} fields of {n} atoms",
            field_data.len(),
            field_names.len()
        ));
    }
    let mut frame = frame_with_elements(atomic_number, x, y, z, h)
        .map_err(|e| format!("write_mrec_frame: {e}"))?;
    let atoms = frame
        .get_mut("atoms")
        .expect("frame_with_elements always has an atoms block");
    for (name, values) in field_names.iter().zip(field_data.chunks(n.max(1))) {
        atoms
            .insert(name.as_str(), Array1::from_vec(values.to_vec()).into_dyn())
            .map_err(|e| format!("write_mrec_frame: insert {name}: {e}"))?;
    }
    molrs::io::write_mrec_frame(path, &frame, None, None)
        .map_err(|e| format!("write_mrec_frame: {e}"))
}

/// Hand a frame read from disk to C++ as a fresh standalone [`FrameRef`].
fn frame_ref_holding(frame: Frame, what: &str) -> Result<Box<FrameRef>, String> {
    let inner = molrs_ffi::FrameRef::new_standalone();
    inner
        .with_mut(|f| *f = frame)
        .map_err(|e| format!("{what}: populate: {e}"))?;
    Ok(Box::new(FrameRef(inner)))
}

/// Read the `frame` section of a `*.mrec` record — molrs
/// `io::read_mrec_frame`, the inverse of [`write_mrec_frame`].
pub(crate) fn read_mrec_frame(path: &str) -> Result<Box<FrameRef>, String> {
    let frame = molrs::io::read_mrec_frame(path).map_err(|e| format!("read_mrec_frame: {e}"))?;
    frame_ref_holding(frame, "read_mrec_frame")
}

/// Read frame `index` of the trajectory in a `*.mrec` record — molrs
/// `MrecReader::open(path)?.frame(index)`. Only that frame is decoded; the
/// trajectory is opened as a lazy cursor. Reads what an [`MrecWriterRef`]
/// wrote.
pub(crate) fn read_mrec_trajectory_frame(path: &str, index: u64) -> Result<Box<FrameRef>, String> {
    let what = "read_mrec_trajectory_frame";
    let frame = MrecReader::open(path)
        .map_err(|e| format!("{what}: {e}"))?
        .frame(index)
        .map_err(|e| format!("{what}: {e}"))?
        .ok_or_else(|| format!("{what}: the trajectory has no frame {index}"))?;
    frame_ref_holding(frame, what)
}

/// The engine's streaming trajectory writer: a molrs `MrecWriter` behind
/// an opaque CXX handle. `None` once closed, so a use after close is an error
/// rather than a panic across the seam.
pub struct MrecWriterRef(Option<MrecWriter>);

fn configure_writer(
    writer: MrecWriter,
    flush_every: u64,
    durable: bool,
) -> Result<MrecWriter, String> {
    let writer = writer.with_durable(durable);
    if flush_every == 0 {
        return Ok(writer);
    }
    writer
        .with_flush_every(flush_every)
        .map_err(|e| format!("mrec_writer: {e}"))
}

/// Mint a `*.mrec` trajectory at `path`, pinned to the blocks and columns of
/// `schema_from`.
pub(crate) fn mrec_writer_create(
    path: &str,
    schema_from: &FrameRef,
    flush_every: u64,
    durable: bool,
) -> Result<Box<MrecWriterRef>, String> {
    let schema = schema_from
        .0
        .with(SequenceSchema::from_frame)
        .map_err(|e| format!("mrec_writer_create: {e}"))?
        .map_err(|e| format!("mrec_writer_create: {e}"))?;
    let writer =
        MrecWriter::create(path, schema).map_err(|e| format!("mrec_writer_create: {e}"))?;
    Ok(Box::new(MrecWriterRef(Some(configure_writer(
        writer,
        flush_every,
        durable,
    )?))))
}

/// Reattach to the trajectory at `path` and continue after its last committed
/// frame; whatever a crash left past the commit marker is rolled back first.
pub(crate) fn mrec_writer_open(
    path: &str,
    flush_every: u64,
    durable: bool,
) -> Result<Box<MrecWriterRef>, String> {
    let writer = MrecWriter::open(path).map_err(|e| format!("mrec_writer_open: {e}"))?;
    Ok(Box::new(MrecWriterRef(Some(configure_writer(
        writer,
        flush_every,
        durable,
    )?))))
}

/// Buffer one frame at `step` (with `time` in fs when `has_time`).
pub(crate) fn mrec_writer_append(
    writer: &mut MrecWriterRef,
    fref: &FrameRef,
    step: i64,
    time: f64,
    has_time: bool,
) -> Result<(), String> {
    let inner = writer
        .0
        .as_mut()
        .ok_or_else(|| "mrec_writer_append: writer is closed".to_string())?;
    let time = has_time.then_some(time);
    fref.0
        .with(|frame| inner.append_at(frame, step, time))
        .map_err(|e| format!("mrec_writer_append: {e}"))?
        .map_err(|e| format!("mrec_writer_append: {e}"))
}

/// Commit every buffered frame (durably, unless the writer was opened with
/// `durable == false`).
pub(crate) fn mrec_writer_flush(writer: &mut MrecWriterRef) -> Result<(), String> {
    writer
        .0
        .as_mut()
        .ok_or_else(|| "mrec_writer_flush: writer is closed".to_string())?
        .flush()
        .map_err(|e| format!("mrec_writer_flush: {e}"))
}

/// Frames committed so far (0 for a closed writer).
pub(crate) fn mrec_writer_committed(writer: &MrecWriterRef) -> u64 {
    writer.0.as_ref().map_or(0, MrecWriter::committed)
}

/// Commit whatever is buffered and release the writer.
pub(crate) fn mrec_writer_close(writer: Box<MrecWriterRef>) -> Result<(), String> {
    match writer.0 {
        Some(inner) => inner.close().map_err(|e| format!("mrec_writer_close: {e}")),
        None => Ok(()),
    }
}

/// Atomic number for a chemical symbol — inverse of [`symbol_for_z`].
///
/// Delegates to molrs' canonical periodic table. No fallback: an
/// unrecognized symbol is an error.
fn z_for_symbol(sym: &str) -> Result<i32, String> {
    Element::by_symbol(sym)
        .map(|element| i32::from(element.z()))
        .ok_or_else(|| format!("z_for_symbol: unknown element symbol '{sym}'"))
}

/// Read the first frame of an (ext)XYZ file into a materialize-ready `FrameRef`.
///
/// All parsing (atom table, `Lattice="..."` -> box) is molrs `io::read_xyz`.
/// The `element` column (the ExtXYZ `species` property) is consumed and
/// replaced by `atomic_number` (a UInt column), so the result satisfies the
/// exact schema that `cpu::materialize` requires (`atoms.{x,y,z,atomic_number}`).
/// The external-format column does not cross that boundary.
///
/// Z lives in `atomic_number`, not `type`: `type` is the force-field label a
/// caller owns (a String), and the vocabulary binds a key's dtype wherever it
/// appears — so writing Z there is refused outright.
pub(crate) fn read_xyz_frame(path: &str) -> Result<Box<FrameRef>, String> {
    let what = "read_xyz_frame";
    let mut frame = molrs::io::read_xyz(path).map_err(|e| format!("{what}: read: {e}"))?;
    let atoms = frame
        .get_mut("atoms")
        .ok_or_else(|| format!("{what}: frame has no atoms block"))?;
    let species = atoms
        .get("element")
        .and_then(|c| c.as_string())
        .ok_or_else(|| format!("{what}: atoms block has no element (ExtXYZ species) column"))?;
    let zs = species
        .iter()
        .map(|symbol| z_for_symbol(symbol).map(|z| z as u64))
        .collect::<Result<Vec<u64>, String>>()?;
    atoms
        .insert("atomic_number", Array1::from_vec(zs).into_dyn())
        .map_err(|e| format!("{what}: insert atomic_number: {e}"))?;
    atoms.remove("element");
    frame_ref_holding(frame, what)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::*;

    #[test]
    fn an_mrec_writer_minted_from_a_precise_frame_rounds_its_column() {
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
        let mut writer = mrec_writer_create(path, &fref, 0, false).unwrap();
        mrec_writer_append(&mut writer, &fref, 0, 0.0, false).unwrap();
        mrec_writer_close(writer).unwrap();
        let back = read_mrec_trajectory_frame(path, 0).unwrap();
        assert!(read_mrec_trajectory_frame(path, 1).is_err());
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

        let frame = read_xyz_frame(path.to_str().unwrap()).expect("read_xyz_frame");
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
            frame_column_u64(&frame, "atoms", "atomic_number"),
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

        let frame = read_xyz_frame(path.to_str().unwrap()).expect("read_xyz_frame");
        std::fs::remove_file(path).unwrap();
        assert_eq!(frame_column_u64(&frame, "atoms", "atomic_number"), [1u64]);
    }

    #[test]
    fn typed_xyz_writer_preserves_numeric_metadata() {
        use bridge::ffi::MetaType;
        let path = std::env::temp_dir().join(format!(
            "molrs-cxxapi-typed-meta-{}.xyz",
            std::process::id()
        ));
        let mut step = empty_keyed_meta_value("step".into(), MetaType::I64);
        step.i64_value = 9_007_199_254_740_993;
        let mut energy = empty_keyed_meta_value("energy_eV".into(), MetaType::F64);
        energy.f64_value = -1.25;
        write_xyz_frame(
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

        let frame = molrs::io::read_xyz(&path).unwrap();
        assert_eq!(
            frame.meta.get("step").unwrap().as_i64(),
            Some(9_007_199_254_740_993)
        );
        assert_eq!(frame.meta.get("energy_eV").unwrap().as_f64(), Some(-1.25));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn an_mrec_frame_record_round_trips_its_fields_and_refuses_a_ragged_field() {
        let dir = std::env::temp_dir().join(format!(
            "molrs-cxxapi-frame-record-{}.mrec",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.to_str().unwrap();
        let h = [10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0];
        assert!(
            write_mrec_frame(
                path,
                &[8, 1],
                &[0.0, 1.0],
                &[0.0; 2],
                &[0.0; 2],
                &h,
                vec!["q".into()],
                &[1.0]
            )
            .is_err()
        );
        assert!(
            write_mrec_frame(
                path,
                &[8, 1],
                &[0.0, 1.0],
                &[0.0; 2],
                &[0.0; 2],
                &[1.0; 4],
                vec![],
                &[]
            )
            .is_err(),
            "a malformed H is an error, not a frame without a box"
        );
        write_mrec_frame(
            path,
            &[8, 1],
            &[0.0, 1.0],
            &[0.0; 2],
            &[0.0; 2],
            &h,
            vec!["q".into()],
            &[-0.8, 0.4],
        )
        .unwrap();
        let back = read_mrec_frame(path).unwrap();
        assert_eq!(frame_column_f64(&back, "atoms", "q"), vec![-0.8, 0.4]);
        assert_eq!(frame_column_f64(&back, "atoms", "x"), vec![0.0, 1.0]);
        assert_eq!(frame_box_h(&back), h);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
