//! Whole-file readers for the WASM API: the formats with no `*Stream`
//! class.
//!
//! Each reader takes a file's whole content (text, or raw bytes for a
//! binary format) since WASM has no filesystem access, and produces
//! [`Frame`] objects through a uniform two-method interface:
//!
//! - `read(step)` -- read a specific frame by index
//! - `len()` -- return the number of available frames
//!
//! XYZ, PDB, SDF, LAMMPS data and dump, DCD, XTC and TRR are read through
//! their chunk-fed `*Stream` class ([`super::streaming`]) and only there:
//! a format has one reader door.
//!
//! # Supported formats
//!
//! | JS class | Format | Multi-frame? | Produces |
//! |----------|--------|-------------|----------|
//! | `CifReader` | Crystallographic Information File | Yes (per `data_` block) | `"atoms"` block + box from unit cell |
//! | `CubeReader` | Gaussian Cube | No (step=0 only) | `"atoms"` + `"grid"` block + box (Å) |
//! | `VaspChgcarReader` | VASP CHGCAR | No (step=0 only) | `"atoms"` + `"grid"` block + box (Å) |
//! | `GroReader` | GROMACS GRO | Yes | `"atoms"` block + box (**nm→Å on read**) |
//! | `Mol2Reader` | Tripos MOL2 | Yes (per molecule) | `"atoms"` + optional `"bonds"` block (Å) |
//! | `VaspPoscarReader` | VASP POSCAR / CONTCAR | No (step=0 only) | `"atoms"` block + box (Cartesian Å) |
//! | `XsfReader` | XCrySDen XSF | No (step=0 only) | `"atoms"` block + box (Å) |
//! | `AmberInpcrdReader` | AMBER inpcrd / restrt | No (step=0 only) | `"atoms"` block + optional box |
//! | `AmberAcReader` | Antechamber AC | No (step=0 only) | `"atoms"` + optional `"bonds"` |

use crate::core::frame::Frame;
use molrs::io::cif::CifReader as RsCifReader;
use molrs::io::gro::GroReader as RsGroReader;
use molrs::io::mol2::Mol2Reader as RsMol2Reader;
use molrs::io::reader::{FrameReader, Reader};
use molrs::io::{
    read_amber_ac_str, read_amber_inpcrd_str, read_cube_str, read_vasp_chgcar_str,
    read_vasp_poscar_str, read_xsf_str,
};
use std::io::Cursor;

/// The text a reader was built from (it was a JS string, so it is UTF-8).
fn text(content: &[u8]) -> Result<&str, JsValue> {
    std::str::from_utf8(content).map_err(|e| JsValue::from_str(&format!("UTF-8 error: {e}")))
}
use wasm_bindgen::prelude::*;

/// Crystallographic Information File (CIF / mmCIF) reader.
///
/// Each `data_*` block in the file becomes one [`Frame`]. Most CIF files
/// contain a single block (one structure), but multi-block files (e.g.
/// polymorphs of the same compound) are also supported and are exposed
/// as a multi-frame sequence. The unit cell parameters
/// (`_cell_length_a`, `_b`, `_c`, `_cell_angle_alpha`, `_beta`,
/// `_gamma`) are converted to a 3x3 h-matrix on the Rust side and
/// surface on the JS side as `frame.simbox`.
///
/// Produces a [`Frame`] with an `"atoms"` block containing
/// `element` (string), `x`, `y`, `z` (F, angstrom in Cartesian
/// coordinates) and (when present in the file) `label`, `occupancy`,
/// `bfactor` columns.
///
/// CIF parsing reads the entire file on each `read(step)` call --
/// random access is therefore O(file_size), but typical CIF files are
/// small (< 1 MB) and the molvis lazy trajectory caches frames at the
/// JS level, so this is rarely a bottleneck.
///
/// # Example (JavaScript)
///
/// ```js
/// const content = await file.text();
/// const reader = new CifReader(content);
/// const frame  = reader.read(0);
/// const atoms  = frame.get("atoms");
/// const box    = frame.simbox;        // populated from the unit cell
/// ```
#[wasm_bindgen(js_name = CifReader)]
pub struct CifReader {
    content: Vec<u8>,
    cached_len: Option<usize>,
}

#[wasm_bindgen(js_class = CifReader)]
impl CifReader {
    /// Create a new CIF reader from a string containing the file content.
    ///
    /// # Arguments
    ///
    /// * `content` - The full text content of a CIF file
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const reader = new CifReader(cifString);
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new(content: &str) -> CifReader {
        CifReader {
            content: content.as_bytes().to_vec(),
            cached_len: None,
        }
    }

    /// Read the frame at the given block index.
    ///
    /// # Arguments
    ///
    /// * `step` - 0-based index of the `data_*` block to return
    ///
    /// # Returns
    ///
    /// A [`Frame`] for the requested block, or `undefined` when
    /// `step >= len()`.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string on parse errors.
    #[wasm_bindgen]
    pub fn read(&mut self, step: usize) -> Result<Option<Frame>, JsValue> {
        let mut reader = RsCifReader::new(Cursor::new(self.content.as_slice()));
        let frames = molrs::io::reader::collect_frames(&mut reader)
            .map_err(|e| JsValue::from_str(&format!("CIF read error: {}", e)))?;
        if step >= frames.len() {
            return Ok(None);
        }
        let rs_frame = frames.into_iter().nth(step).expect("bounds checked");
        Ok(Some(Frame::from_rs(rs_frame)?))
    }

    /// Return the number of `data_*` blocks in the file.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string on parse errors.
    #[wasm_bindgen]
    pub fn len(&mut self) -> Result<usize, JsValue> {
        if let Some(n) = self.cached_len {
            return Ok(n);
        }
        let mut reader = RsCifReader::new(Cursor::new(self.content.as_slice()));
        let n = molrs::io::reader::collect_frames(&mut reader)
            .map_err(|e| JsValue::from_str(&format!("CIF len error: {}", e)))?
            .len();
        self.cached_len = Some(n);
        Ok(n)
    }

    /// Check whether the file contains no valid blocks.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string on parse errors.
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&mut self) -> Result<bool, JsValue> {
        Ok(self.len()? == 0)
    }
}

/// Gaussian Cube file reader.
///
/// Cube files describe a single voxel grid with embedded atom geometry.
/// The reader produces a [`Frame`] with:
/// - `"atoms"` block: `element` (string), `atomic_number` (i32),
///   `charge` (F), `x`/`y`/`z` (F, **always Å** — Bohr files are converted
///   on read).
/// - `"grid"` block: structural shape `[nx, ny, nz]` and one f64 column
///   per scalar field — `density` for single-density files,
///   `mo_<idx>` for negative-natoms multi-orbital files.
/// - `box`: voxel cell × dims in Å.
///
/// Cube is inherently single-frame (only `step = 0` is valid).
///
/// # Example (JavaScript)
///
/// ```js
/// const content = await file.text();
/// const reader  = new CubeReader(content);
/// const frame   = reader.read(0);
/// const grid    = frame.get("grid");   // shape [nx, ny, nz]
/// const density = grid.get("density"); // owned Float64Array
/// ```
#[wasm_bindgen(js_name = CubeReader)]
pub struct CubeReader {
    content: Vec<u8>,
    cached_len: Option<usize>,
}

#[wasm_bindgen(js_class = CubeReader)]
impl CubeReader {
    /// Create a new Cube reader from the file's text content.
    #[wasm_bindgen(constructor)]
    pub fn new(content: &str) -> CubeReader {
        CubeReader {
            content: content.as_bytes().to_vec(),
            cached_len: None,
        }
    }

    /// Read the frame at `step`. Cube files are single-frame, so any
    /// `step != 0` returns `undefined`.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string on parse errors.
    #[wasm_bindgen]
    pub fn read(&mut self, step: usize) -> Result<Option<Frame>, JsValue> {
        if step > 0 {
            return Ok(None);
        }
        let rs_frame = read_cube_str(text(&self.content)?)
            .map_err(|e| JsValue::from_str(&format!("Cube read error: {}", e)))?;
        Ok(Some(Frame::from_rs(rs_frame)?))
    }

    /// Return the number of frames (always 0 or 1).
    #[wasm_bindgen]
    pub fn len(&mut self) -> Result<usize, JsValue> {
        if let Some(n) = self.cached_len {
            return Ok(n);
        }
        let n = if self.read(0)?.is_some() { 1 } else { 0 };
        self.cached_len = Some(n);
        Ok(n)
    }

    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&mut self) -> Result<bool, JsValue> {
        Ok(self.len()? == 0)
    }
}

/// VASP CHGCAR / CHGDIF volumetric data reader.
///
/// Reads VASP-format charge density files (extension-less canonical name
/// `CHGCAR` or `CHGCAR_*`). Produces a [`Frame`] with:
/// - `"atoms"` block: `element` (string), `x`/`y`/`z` (F, Cartesian Å).
/// - `"grid"` block: structural shape `[nx, ny, nz]`, columns
///   `total` (always) and `diff` (when ISPIN=2).
/// - `box`: triclinic POSCAR lattice in Å, fully periodic.
///
/// CHGCAR is single-frame; only `step = 0` is valid.
///
/// # Example (JavaScript)
///
/// ```js
/// const content = await file.text();
/// const reader  = new VaspChgcarReader(content);
/// const frame   = reader.read(0);
/// const grid    = frame.get("grid");
/// const total   = grid.get("total");
/// ```
#[wasm_bindgen(js_name = VaspChgcarReader)]
pub struct VaspChgcarReader {
    content: Vec<u8>,
    cached_len: Option<usize>,
}

#[wasm_bindgen(js_class = VaspChgcarReader)]
impl VaspChgcarReader {
    /// Create a new CHGCAR reader from the file's text content.
    #[wasm_bindgen(constructor)]
    pub fn new(content: &str) -> VaspChgcarReader {
        VaspChgcarReader {
            content: content.as_bytes().to_vec(),
            cached_len: None,
        }
    }

    /// Read the frame at `step`. CHGCAR is single-frame.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string on parse errors.
    #[wasm_bindgen]
    pub fn read(&mut self, step: usize) -> Result<Option<Frame>, JsValue> {
        if step > 0 {
            return Ok(None);
        }
        let rs_frame = read_vasp_chgcar_str(text(&self.content)?)
            .map_err(|e| JsValue::from_str(&format!("CHGCAR read error: {}", e)))?;
        Ok(Some(Frame::from_rs(rs_frame)?))
    }

    #[wasm_bindgen]
    pub fn len(&mut self) -> Result<usize, JsValue> {
        if let Some(n) = self.cached_len {
            return Ok(n);
        }
        let n = if self.read(0)?.is_some() { 1 } else { 0 };
        self.cached_len = Some(n);
        Ok(n)
    }

    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&mut self) -> Result<bool, JsValue> {
        Ok(self.len()? == 0)
    }
}

/// GROMACS GRO structure / trajectory reader.
///
/// GRO is a fixed-column text format for GROMACS structures and
/// single-precision trajectories. Multi-frame files expose each frame via
/// `read(step)`. Coordinates and box are GROMACS-native nm in the file and
/// arrive in angstrom — the molrs GRO reader normalises at its own boundary,
/// so this binder scales nothing. Each frame produces an `"atoms"` block
/// (`res_id`, `res_name`, `name`, `element`, `id`, `x`/`y`/`z`, optional
/// `vx`/`vy`/`vz`) and a `box` from the box-vector line.
#[wasm_bindgen(js_name = GroReader)]
pub struct GroReader {
    content: Vec<u8>,
    cached_len: Option<usize>,
}

#[wasm_bindgen(js_class = GroReader)]
impl GroReader {
    /// Create a new GRO reader from a string containing the file content.
    #[wasm_bindgen(constructor)]
    pub fn new(content: &str) -> GroReader {
        GroReader {
            content: content.as_bytes().to_vec(),
            cached_len: None,
        }
    }

    /// Read the frame at the given step index (0-based). Coordinates arrive
    /// in angstrom (the molrs reader converts from the file's nm).
    #[wasm_bindgen]
    pub fn read(&mut self, step: usize) -> Result<Option<Frame>, JsValue> {
        let mut reader = RsGroReader::new(Cursor::new(self.content.as_slice()));
        for current in 0..=step {
            let rs_frame = reader
                .read()
                .map_err(|e| JsValue::from_str(&format!("GRO read error: {}", e)))?;
            match rs_frame {
                Some(frame) if current == step => {
                    return Ok(Some(Frame::from_rs(frame)?));
                }
                Some(_) => continue,
                None => return Ok(None),
            }
        }
        Ok(None)
    }

    /// Return the number of frames in the GRO file.
    #[wasm_bindgen]
    pub fn len(&mut self) -> Result<usize, JsValue> {
        if let Some(n) = self.cached_len {
            return Ok(n);
        }
        let mut reader = RsGroReader::new(Cursor::new(self.content.as_slice()));
        let mut count = 0usize;
        while reader
            .read()
            .map_err(|e| JsValue::from_str(&format!("GRO len error: {}", e)))?
            .is_some()
        {
            count += 1;
        }
        self.cached_len = Some(count);
        Ok(count)
    }

    /// Check whether the file contains no frames.
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&mut self) -> Result<bool, JsValue> {
        Ok(self.len()? == 0)
    }
}

/// Tripos MOL2 reader.
///
/// MOL2 is a section-delimited (`@<TRIPOS>...`) text format. Multi-molecule
/// files expose each `MOLECULE` record as a frame via `read(step)`.
/// Coordinates are already in angstrom. Produces an `"atoms"` block (`id`,
/// `name`, `x`/`y`/`z`, `atom_type`, optional `subst_id`/`subst_name`/
/// `charge`) and, when present, a `"bonds"` block (`atomi`/`atomj` 0-based,
/// `bond_type`).
#[wasm_bindgen(js_name = Mol2Reader)]
pub struct Mol2Reader {
    content: Vec<u8>,
    cached_len: Option<usize>,
}

#[wasm_bindgen(js_class = Mol2Reader)]
impl Mol2Reader {
    /// Create a new MOL2 reader from a string containing the file content.
    #[wasm_bindgen(constructor)]
    pub fn new(content: &str) -> Mol2Reader {
        Mol2Reader {
            content: content.as_bytes().to_vec(),
            cached_len: None,
        }
    }

    /// Read the molecule record at the given step index (0-based).
    #[wasm_bindgen]
    pub fn read(&mut self, step: usize) -> Result<Option<Frame>, JsValue> {
        let mut reader = RsMol2Reader::new(Cursor::new(self.content.as_slice()));
        for current in 0..=step {
            let rs_frame = reader
                .read()
                .map_err(|e| JsValue::from_str(&format!("MOL2 read error: {}", e)))?;
            match rs_frame {
                Some(frame) if current == step => return Ok(Some(Frame::from_rs(frame)?)),
                Some(_) => continue,
                None => return Ok(None),
            }
        }
        Ok(None)
    }

    /// Return the number of molecule records in the MOL2 file.
    #[wasm_bindgen]
    pub fn len(&mut self) -> Result<usize, JsValue> {
        if let Some(n) = self.cached_len {
            return Ok(n);
        }
        let mut reader = RsMol2Reader::new(Cursor::new(self.content.as_slice()));
        let mut count = 0usize;
        while reader
            .read()
            .map_err(|e| JsValue::from_str(&format!("MOL2 len error: {}", e)))?
            .is_some()
        {
            count += 1;
        }
        self.cached_len = Some(count);
        Ok(count)
    }

    /// Check whether the file contains no records.
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&mut self) -> Result<bool, JsValue> {
        Ok(self.len()? == 0)
    }
}

/// VASP POSCAR / CONTCAR structure reader.
///
/// POSCAR describes a single crystalline cell. Coordinates are returned as
/// Cartesian angstrom (`Direct` files are converted on read by molrs).
/// Produces an `"atoms"` block (`x`/`y`/`z`, optional `symbol`,
/// selective-dynamics flags, velocities) and a periodic `box`.
/// Single-frame: any `step != 0` returns `undefined`.
#[wasm_bindgen(js_name = VaspPoscarReader)]
pub struct VaspPoscarReader {
    content: Vec<u8>,
    cached_len: Option<usize>,
}

#[wasm_bindgen(js_class = VaspPoscarReader)]
impl VaspPoscarReader {
    /// Create a new POSCAR reader from a string containing the file content.
    #[wasm_bindgen(constructor)]
    pub fn new(content: &str) -> VaspPoscarReader {
        VaspPoscarReader {
            content: content.as_bytes().to_vec(),
            cached_len: None,
        }
    }

    /// Read the frame at `step`. POSCAR is single-frame, so any `step != 0`
    /// returns `undefined`.
    #[wasm_bindgen]
    pub fn read(&mut self, step: usize) -> Result<Option<Frame>, JsValue> {
        if step > 0 {
            return Ok(None);
        }
        let rs_frame = read_vasp_poscar_str(text(&self.content)?)
            .map_err(|e| JsValue::from_str(&format!("POSCAR read error: {}", e)))?;
        Ok(Some(Frame::from_rs(rs_frame)?))
    }

    /// Return the number of frames (always 0 or 1 for POSCAR files).
    #[wasm_bindgen]
    pub fn len(&mut self) -> Result<usize, JsValue> {
        if let Some(n) = self.cached_len {
            return Ok(n);
        }
        let n = if self.read(0)?.is_some() { 1 } else { 0 };
        self.cached_len = Some(n);
        Ok(n)
    }

    /// Check whether the file contains no valid frame.
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&mut self) -> Result<bool, JsValue> {
        Ok(self.len()? == 0)
    }
}

/// XCrySDen XSF structure reader.
#[wasm_bindgen(js_name = XsfReader)]
pub struct XsfReader {
    content: Vec<u8>,
    cached_len: Option<usize>,
}

#[wasm_bindgen(js_class = XsfReader)]
impl XsfReader {
    #[wasm_bindgen(constructor)]
    pub fn new(content: &str) -> XsfReader {
        XsfReader {
            content: content.as_bytes().to_vec(),
            cached_len: None,
        }
    }

    #[wasm_bindgen]
    pub fn read(&mut self, step: usize) -> Result<Option<Frame>, JsValue> {
        if step > 0 {
            return Ok(None);
        }
        let rs_frame = read_xsf_str(text(&self.content)?)
            .map_err(|e| JsValue::from_str(&format!("XSF read error: {}", e)))?;
        Ok(Some(Frame::from_rs(rs_frame)?))
    }

    #[wasm_bindgen]
    pub fn len(&mut self) -> Result<usize, JsValue> {
        if let Some(n) = self.cached_len {
            return Ok(n);
        }
        let n = if self.read(0)?.is_some() { 1 } else { 0 };
        self.cached_len = Some(n);
        Ok(n)
    }

    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&mut self) -> Result<bool, JsValue> {
        Ok(self.len()? == 0)
    }
}

/// AMBER ASCII inpcrd / restrt coordinate reader.
#[wasm_bindgen(js_name = AmberInpcrdReader)]
pub struct AmberInpcrdReader {
    content: Vec<u8>,
    cached_len: Option<usize>,
}

#[wasm_bindgen(js_class = AmberInpcrdReader)]
impl AmberInpcrdReader {
    #[wasm_bindgen(constructor)]
    pub fn new(content: &str) -> AmberInpcrdReader {
        AmberInpcrdReader {
            content: content.as_bytes().to_vec(),
            cached_len: None,
        }
    }

    #[wasm_bindgen]
    pub fn read(&mut self, step: usize) -> Result<Option<Frame>, JsValue> {
        if step > 0 {
            return Ok(None);
        }
        let rs_frame = read_amber_inpcrd_str(text(&self.content)?)
            .map_err(|e| JsValue::from_str(&format!("AMBER inpcrd read error: {}", e)))?;
        Ok(Some(Frame::from_rs(rs_frame)?))
    }

    #[wasm_bindgen]
    pub fn len(&mut self) -> Result<usize, JsValue> {
        if let Some(n) = self.cached_len {
            return Ok(n);
        }
        let n = if self.read(0)?.is_some() { 1 } else { 0 };
        self.cached_len = Some(n);
        Ok(n)
    }

    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&mut self) -> Result<bool, JsValue> {
        Ok(self.len()? == 0)
    }
}

/// Antechamber `.ac` structure reader.
#[wasm_bindgen(js_name = AmberAcReader)]
pub struct AmberAcReader {
    content: String,
    cached_len: Option<usize>,
}

#[wasm_bindgen(js_class = AmberAcReader)]
impl AmberAcReader {
    #[wasm_bindgen(constructor)]
    pub fn new(content: &str) -> AmberAcReader {
        AmberAcReader {
            content: content.to_string(),
            cached_len: None,
        }
    }

    #[wasm_bindgen]
    pub fn read(&mut self, step: usize) -> Result<Option<Frame>, JsValue> {
        if step > 0 {
            return Ok(None);
        }
        let rs_frame = read_amber_ac_str(&self.content)
            .map_err(|e| JsValue::from_str(&format!("AC read error: {}", e)))?;
        Ok(Some(Frame::from_rs(rs_frame)?))
    }

    #[wasm_bindgen]
    pub fn len(&mut self) -> Result<usize, JsValue> {
        if let Some(n) = self.cached_len {
            return Ok(n);
        }
        let n = if self.read(0)?.is_some() { 1 } else { 0 };
        self.cached_len = Some(n);
        Ok(n)
    }

    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&mut self) -> Result<bool, JsValue> {
        Ok(self.len()? == 0)
    }
}

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
#[cfg(feature = "stream")]
#[wasm_bindgen(js_name = readMsgpackFrameBytes)]
pub fn read_msgpack_frame_bytes(data: &[u8]) -> Result<Frame, JsValue> {
    let rs_frame = molrs::stream::read_msgpack_frame_bytes(data)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Frame::from_rs(rs_frame)
}

/// Rebuild a [`Frame`] from `molrs::stream` JSON wire text. The inverse of
/// `writeJsonFrameStr`.
#[cfg(feature = "stream")]
#[wasm_bindgen(js_name = readJsonFrameStr)]
pub fn read_json_frame_str(text: &str) -> Result<Frame, JsValue> {
    let rs_frame =
        molrs::stream::read_json_frame_str(text).map_err(|e| JsValue::from_str(&e.to_string()))?;
    Frame::from_rs(rs_frame)
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    /// Owned `f64` column `key` of `block`.
    fn float_col(block: &crate::core::Block, key: &str) -> js_sys::Float64Array {
        wasm_bindgen::JsCast::unchecked_into(JsValue::from(block.get(key, None).expect(key)))
    }

    #[cfg(feature = "stream")]
    #[wasm_bindgen_test]
    fn stream_bytes_round_trip_through_io() {
        use crate::core::nd_array::JsFloatArray;
        use crate::io::writer::{write_json_frame_str, write_msgpack_frame_bytes};

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
