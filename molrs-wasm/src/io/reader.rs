//! Molecular file readers for the WASM API.
//!
//! Each reader takes a file's text content as a string (since WASM has
//! no filesystem access) and produces [`Frame`] objects. All readers
//! implement a uniform two-method interface:
//!
//! - `read(step)` -- read a specific frame by index
//! - `len()` -- return the number of available frames
//!
//! # Supported formats
//!
//! | JS class | Format | Multi-frame? | Produces |
//! |----------|--------|-------------|----------|
//! | `XYZReader` | XYZ / ExtXYZ | Yes | `"atoms"` block with `element`, `x`, `y`, `z` |
//! | `PDBReader` | Protein Data Bank | No (step=0 only) | `"atoms"` block with `name`, `resname`, `x`, `y`, `z`, etc. |
//! | `CIFReader` | Crystallographic Information File | Yes (per `data_` block) | `"atoms"` block + simbox from unit cell |
//! | `LAMMPSReader` | LAMMPS data file | No (step=0 only) | `"atoms"` block + `"bonds"` block + simbox |
//! | `LAMMPSTrajReader` | LAMMPS dump trajectory | Yes | `"atoms"` block with columns from dump header |
//! | `SDFReader` | MDL molfile / SDF | Yes (per record) | `"atoms"` + optional `"bonds"` block |
//! | `CubeReader` | Gaussian Cube | No (step=0 only) | `"atoms"` + `"grid"` block + simbox (Å) |
//! | `CHGCARReader` | VASP CHGCAR | No (step=0 only) | `"atoms"` + `"grid"` block + simbox (Å) |
//! | `GROReader` | GROMACS GRO | Yes | `"atoms"` block + simbox (**nm→Å on read**) |
//! | `MOL2Reader` | Tripos MOL2 | Yes (per molecule) | `"atoms"` + optional `"bonds"` block (Å) |
//! | `POSCARReader` | VASP POSCAR / CONTCAR | No (step=0 only) | `"atoms"` block + simbox (Cartesian Å) |
//! | `DCDReader` | DCD trajectory (binary) | Yes | `"atoms"` block + optional simbox |
//! | `TRRReader` | GROMACS TRR (binary) | Yes | `"atoms"` block + simbox (**nm→Å on read**) |
//! | `XTCReader` | GROMACS XTC (binary) | Yes | `"atoms"` block + simbox (**nm→Å on read**) |

use crate::core::frame::Frame;
use molrs::io::data::chgcar::read_chgcar_from_reader;
use molrs::io::data::cif::CifReader as RsCifReader;
use molrs::io::data::cube::read_cube_from_reader;
use molrs::io::data::gro::GroReader as RsGroReader;
use molrs::io::data::lammps_data::LAMMPSDataReader;
use molrs::io::data::mol2::Mol2Reader as RsMol2Reader;
use molrs::io::data::pdb::PDBReader;
use molrs::io::data::poscar::read_poscar_from_reader;
use molrs::io::data::sdf::SDFReader;
use molrs::io::data::xyz::XYZReader;
use molrs::io::reader::{FrameReader, Reader, TrajReader};
use molrs::io::trajectory::dcd::DcdReader as RsDcdReader;
use molrs::io::trajectory::lammps_dump::LAMMPSTrajReader;
use molrs::io::trajectory::trr::TrrReader as RsTrrReader;
use molrs::io::trajectory::xtc::XtcReader as RsXtcReader;
use molrs::spatial::region::simbox::SimBox;
use molrs::store::frame::Frame as RsFrame;
use molrs::types::F;
use std::io::{BufReader, Cursor};
use wasm_bindgen::prelude::*;

/// Nanometre → ångström length scale.
pub(crate) const NM_TO_ANGSTROM: F = 10.0;

/// Multiply a frame's position columns (`x`/`y`/`z`) and simulation box by
/// `factor`.
///
/// GROMACS-native formats (GRO, TRR, XTC) store lengths in nm, while every
/// molvis consumer — renderer, bond perception, RDF — works in ångström, so
/// the WASM boundary converts on read (`NM_TO_ANGSTROM`) and back on write
/// (its reciprocal), mirroring how the Cube reader converts Bohr → Å.
/// Velocity / force columns are deliberately left untouched: they are never
/// interpreted as positions downstream.
pub(crate) fn scale_frame_lengths(frame: &mut RsFrame, factor: F) -> Result<(), JsValue> {
    if let Some(atoms) = frame.get_mut("atoms") {
        for key in ["x", "y", "z"] {
            if let Some(col) = atoms.get_float_mut(key) {
                col.mapv_inplace(|v| v * factor);
            }
        }
    }
    if let Some(sb) = frame.simbox.as_ref() {
        let h = sb.h_view().to_owned() * factor;
        let origin = sb.origin_view().to_owned() * factor;
        let pbc = sb.pbc();
        let scaled = SimBox::new(h, origin, pbc)
            .map_err(|e| JsValue::from_str(&format!("length scale error: {:?}", e)))?;
        frame.simbox = Some(scaled);
    }
    Ok(())
}

/// Convert a freshly-read GROMACS frame from nm to ångström.
fn scale_nm_to_angstrom(frame: &mut RsFrame) -> Result<(), JsValue> {
    scale_frame_lengths(frame, NM_TO_ANGSTROM)
}

/// XYZ / Extended XYZ file reader.
///
/// Supports multi-frame trajectory files. Each frame produces a
/// [`Frame`] with an `"atoms"` block containing `element` (string)
/// and `x`, `y`, `z` (F, coordinates in angstrom) columns.
///
/// # Example (JavaScript)
///
/// ```js
/// const content = await file.text(); // read file in browser
/// const reader = new XYZReader(content);
/// console.log(reader.len()); // number of frames
///
/// const frame = reader.read(0); // first frame
/// const atoms = frame.getBlock("atoms");
/// const x = atoms.copyColF("x");
/// ```
#[wasm_bindgen(js_name = XYZReader)]
pub struct XyzReader {
    inner: XYZReader<Cursor<Vec<u8>>>,
}

#[wasm_bindgen(js_class = XYZReader)]
impl XyzReader {
    /// Create a new XYZ reader from a string containing the file content.
    ///
    /// # Arguments
    ///
    /// * `content` - The full text content of an XYZ or Extended XYZ file
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const reader = new XYZReader(fileContent);
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new(content: &str) -> XyzReader {
        let bytes = content.as_bytes().to_vec();
        XyzReader {
            inner: XYZReader::new(Cursor::new(bytes)),
        }
    }

    /// Read a frame at the given step index.
    ///
    /// # Arguments
    ///
    /// * `step` - Zero-based frame index
    ///
    /// # Returns
    ///
    /// A [`Frame`] if the step exists, or `undefined` if `step` is
    /// out of range.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string on parse errors.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const frame = reader.read(0);
    /// if (frame) {
    ///   const atoms = frame.getBlock("atoms");
    /// }
    /// ```
    #[wasm_bindgen]
    pub fn read(&mut self, step: usize) -> Result<Option<Frame>, JsValue> {
        let rs_frame = self
            .inner
            .read_step(step)
            .map_err(|e| JsValue::from_str(&format!("XYZ read error: {}", e)))?;

        match rs_frame {
            Some(frame_data) => Ok(Some(Frame::from_rs(frame_data)?)),
            None => Ok(None),
        }
    }

    /// Return the number of frames in the file.
    ///
    /// # Returns
    ///
    /// The total frame count.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string if the file cannot be scanned.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// console.log(reader.len()); // e.g., 100
    /// ```
    #[wasm_bindgen]
    pub fn len(&mut self) -> Result<usize, JsValue> {
        self.inner
            .len()
            .map_err(|e| JsValue::from_str(&format!("XYZ len error: {}", e)))
    }

    /// Check whether the file contains no frames.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string if the file cannot be scanned.
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&mut self) -> Result<bool, JsValue> {
        Ok(self.len()? == 0)
    }
}

/// Protein Data Bank (PDB) file reader.
///
/// PDB files contain a single molecular structure. The reader produces
/// a [`Frame`] with an `"atoms"` block containing columns such as
/// `name` (string), `resname` (string), `x`, `y`, `z` (F, angstrom),
/// and optionally `occupancy` and `bfactor` (F).
///
/// # Example (JavaScript)
///
/// ```js
/// const reader = new PDBReader(pdbContent);
/// const frame = reader.read(0);
/// const atoms = frame.getBlock("atoms");
/// const names = atoms.copyColStr("name"); // ["CA", "CB", ...]
/// const x = atoms.copyColF("x");
/// ```
#[wasm_bindgen(js_name = PDBReader)]
pub struct PdbReader {
    content: Vec<u8>,
    cached_len: Option<usize>,
}

#[wasm_bindgen(js_class = PDBReader)]
impl PdbReader {
    /// Create a new PDB reader from a string containing the file content.
    ///
    /// # Arguments
    ///
    /// * `content` - The full text content of a PDB file
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const reader = new PDBReader(pdbString);
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new(content: &str) -> PdbReader {
        PdbReader {
            content: content.as_bytes().to_vec(),
            cached_len: None,
        }
    }

    /// Read the frame at the given step index.
    ///
    /// PDB files contain a single structure, so only `step = 0` is
    /// valid. Passing any other step returns `undefined`.
    ///
    /// # Arguments
    ///
    /// * `step` - Frame index (must be `0` for PDB files)
    ///
    /// # Returns
    ///
    /// A [`Frame`] if `step == 0` and the file is valid, or `undefined`.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string on parse errors.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const frame = reader.read(0);
    /// ```
    #[wasm_bindgen]
    pub fn read(&mut self, step: usize) -> Result<Option<Frame>, JsValue> {
        if step > 0 {
            return Ok(None);
        }

        let mut reader = PDBReader::new(Cursor::new(self.content.as_slice()));
        let rs_frame = reader
            .read_frame()
            .map_err(|e| JsValue::from_str(&format!("PDB read error: {}", e)))?;

        match rs_frame {
            Some(frame_data) => Ok(Some(Frame::from_rs(frame_data)?)),
            None => Ok(None),
        }
    }

    /// Return the number of frames (always 0 or 1 for PDB files).
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string on parse errors.
    #[wasm_bindgen]
    pub fn len(&mut self) -> Result<usize, JsValue> {
        if let Some(n) = self.cached_len {
            return Ok(n);
        }
        let has_frame = self.read(0)?.is_some();
        let n = if has_frame { 1 } else { 0 };
        self.cached_len = Some(n);
        Ok(n)
    }

    /// Check whether the file contains no valid frames.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string on parse errors.
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&mut self) -> Result<bool, JsValue> {
        Ok(self.len()? == 0)
    }
}

/// LAMMPS data file reader.
///
/// Reads LAMMPS data files (the format written by `write_data`). The
/// reader produces a [`Frame`] containing:
///
/// - `"atoms"` block: `type` (i32), `x`, `y`, `z` (F, angstrom),
///   and optionally `charge` (F)
/// - `"bonds"` block (if present): `i`, `j` (u32), `type` (i32)
/// - Simulation box (`simbox`) with PBC
///
/// Only a single frame is supported (`step = 0`).
///
/// # Example (JavaScript)
///
/// ```js
/// const reader = new LAMMPSReader(dataFileContent);
/// const frame = reader.read(0);
/// const atoms = frame.getBlock("atoms");
/// const bonds = frame.getBlock("bonds");
/// const box   = frame.simbox;
/// ```
#[wasm_bindgen(js_name = LAMMPSReader)]
pub struct LammpsReader {
    content: Vec<u8>,
    cached_len: Option<usize>,
}

#[wasm_bindgen(js_class = LAMMPSReader)]
impl LammpsReader {
    /// Create a new LAMMPS data file reader from string content.
    ///
    /// # Arguments
    ///
    /// * `content` - The full text content of a LAMMPS data file
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const reader = new LAMMPSReader(dataFileString);
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new(content: &str) -> LammpsReader {
        LammpsReader {
            content: content.as_bytes().to_vec(),
            cached_len: None,
        }
    }

    /// Read the frame at the given step index.
    ///
    /// LAMMPS data files contain a single configuration, so only
    /// `step = 0` is valid. Passing any other step returns `undefined`.
    ///
    /// # Arguments
    ///
    /// * `step` - Frame index (must be `0`)
    ///
    /// # Returns
    ///
    /// A [`Frame`] with atoms, optional bonds, and simbox, or `undefined`.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string on parse errors.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const frame = reader.read(0);
    /// ```
    #[wasm_bindgen]
    pub fn read(&mut self, step: usize) -> Result<Option<Frame>, JsValue> {
        if step > 0 {
            return Ok(None);
        }

        // Build a fresh rs reader each call: `FrameReader::read_frame` is a
        // single-shot cursor API (sets `returned=true` after first call), so
        // reusing `inner` across `len()` and `read(0)` returned None on the
        // second call. PDBReader already uses this pattern — mirror it here.
        let mut reader = LAMMPSDataReader::new(Cursor::new(self.content.as_slice()));
        let rs_frame = reader
            .read_frame()
            .map_err(|e| JsValue::from_str(&format!("LAMMPS read error: {}", e)))?;

        match rs_frame {
            Some(frame_data) => Ok(Some(Frame::from_rs(frame_data)?)),
            None => Ok(None),
        }
    }

    /// Return the number of frames (always 0 or 1 for LAMMPS data files).
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string on parse errors.
    #[wasm_bindgen]
    pub fn len(&mut self) -> Result<usize, JsValue> {
        if let Some(n) = self.cached_len {
            return Ok(n);
        }
        let has_frame = self.read(0)?.is_some();
        let n = if has_frame { 1 } else { 0 };
        self.cached_len = Some(n);
        Ok(n)
    }

    /// Check whether the file contains no valid frames.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string on parse errors.
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&mut self) -> Result<bool, JsValue> {
        Ok(self.len()? == 0)
    }
}

/// LAMMPS dump trajectory file reader.
///
/// Reads multi-frame LAMMPS dump files (the format produced by the
/// `dump` command). Each frame produces a [`Frame`] containing an
/// `"atoms"` block with columns matching the dump header (e.g.
/// `id`, `type`, `x`, `y`, `z`, `vx`, `vy`, `vz`).
///
/// # Example (JavaScript)
///
/// ```js
/// const reader = new LAMMPSTrajReader(dumpContent);
/// console.log(reader.len()); // number of timesteps
/// const frame = reader.read(0);
/// const atoms = frame.getBlock("atoms");
/// ```
#[wasm_bindgen(js_name = LAMMPSTrajReader)]
pub struct LammpsDumpReader {
    inner: LAMMPSTrajReader<Cursor<Vec<u8>>>,
}

#[wasm_bindgen(js_class = LAMMPSTrajReader)]
impl LammpsDumpReader {
    /// Create a new LAMMPS dump reader from string content.
    #[wasm_bindgen(constructor)]
    pub fn new(content: &str) -> LammpsDumpReader {
        let bytes = content.as_bytes().to_vec();
        LammpsDumpReader {
            inner: LAMMPSTrajReader::new(Cursor::new(bytes)),
        }
    }

    /// Read a frame at the given step index.
    #[wasm_bindgen]
    pub fn read(&mut self, step: usize) -> Result<Option<Frame>, JsValue> {
        let rs_frame = self
            .inner
            .read_step(step)
            .map_err(|e| JsValue::from_str(&format!("LAMMPS dump read error: {}", e)))?;

        match rs_frame {
            Some(frame_data) => Ok(Some(Frame::from_rs(frame_data)?)),
            None => Ok(None),
        }
    }

    /// Return the number of frames in the dump file.
    #[wasm_bindgen]
    pub fn len(&mut self) -> Result<usize, JsValue> {
        self.inner
            .len()
            .map_err(|e| JsValue::from_str(&format!("LAMMPS dump len error: {}", e)))
    }

    /// Check whether the file contains no frames.
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&mut self) -> Result<bool, JsValue> {
        Ok(self.len()? == 0)
    }
}

/// MDL molfile / SDF (V2000 CTAB) reader.
///
/// Parses the connection table found in `.mol` files and the record
/// blocks of `.sdf` files. Coordinates come directly from the file —
/// no 3D generation is performed. Only V2000 is supported; V3000
/// records throw on read.
///
/// Produces a [`Frame`] with:
/// - `"atoms"` block: `element` (string), `id` (u32, 1-based),
///   `x`, `y`, `z` (F, angstrom)
/// - `"bonds"` block (if present): `atomi`, `atomj` (u32, 0-based),
///   `order` (u32)
///
/// Multi-record SDF files expose each record as a separate frame via
/// `read(step)`.
///
/// # Example (JavaScript)
///
/// ```js
/// const reader = new SDFReader(sdfContent);
/// const frame = reader.read(0);
/// const atoms = frame.getBlock("atoms");
/// const x = atoms.copyColF("x");
/// ```
#[wasm_bindgen(js_name = SDFReader)]
pub struct SdfReader {
    content: Vec<u8>,
    cached_len: Option<usize>,
}

#[wasm_bindgen(js_class = SDFReader)]
impl SdfReader {
    /// Create a new SDF reader from a string containing the file content.
    #[wasm_bindgen(constructor)]
    pub fn new(content: &str) -> SdfReader {
        SdfReader {
            content: content.as_bytes().to_vec(),
            cached_len: None,
        }
    }

    /// Read the frame (SDF record) at the given step index.
    #[wasm_bindgen]
    pub fn read(&mut self, step: usize) -> Result<Option<Frame>, JsValue> {
        let mut reader = SDFReader::new(Cursor::new(self.content.as_slice()));
        for current in 0..=step {
            let rs_frame = reader
                .read_frame()
                .map_err(|e| JsValue::from_str(&format!("SDF read error: {}", e)))?;
            match rs_frame {
                Some(frame) if current == step => return Ok(Some(Frame::from_rs(frame)?)),
                Some(_) => continue, // drop records before the target
                None => return Ok(None),
            }
        }
        Ok(None)
    }

    /// Return the total number of records in the SDF file.
    #[wasm_bindgen]
    pub fn len(&mut self) -> Result<usize, JsValue> {
        if let Some(n) = self.cached_len {
            return Ok(n);
        }
        let mut reader = SDFReader::new(Cursor::new(self.content.as_slice()));
        let mut count = 0usize;
        while reader
            .read_frame()
            .map_err(|e| JsValue::from_str(&format!("SDF len error: {}", e)))?
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

/// DCD trajectory file reader.
///
/// DCD is a binary multi-frame trajectory format originally used by
/// CHARMM and now widely produced by NAMD / OpenMM / GROMACS-via-VMD.
/// This wrapper accepts the file as raw bytes (`Uint8Array`) since
/// DCD is not text-encoded — passing a JS string would corrupt the
/// fixed-width Fortran record markers.
///
/// Each frame produces a [`Frame`] with an `"atoms"` block carrying
/// `x`, `y`, `z` (F, angstrom). Box/cell information, when the DCD
/// header declares it present, is attached as the frame's `simbox`.
///
/// # Example (JavaScript)
///
/// ```js
/// const bytes = new Uint8Array(await blob.arrayBuffer());
/// const reader = new DCDReader(bytes);
/// console.log(reader.len()); // number of frames
///
/// const frame = reader.read(0); // first frame
/// const atoms = frame.getBlock("atoms");
/// const x = atoms.copyColF("x");
/// ```
#[wasm_bindgen(js_name = DCDReader)]
pub struct DcdReader {
    inner: RsDcdReader<Cursor<Vec<u8>>>,
}

#[wasm_bindgen(js_class = DCDReader)]
impl DcdReader {
    /// Create a new DCD reader from the file's raw bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - The full binary content of a DCD file. The reader
    ///   takes ownership of an internal copy, so the caller is free
    ///   to discard the buffer immediately after this returns.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const bytes = new Uint8Array(await file.arrayBuffer());
    /// const reader = new DCDReader(bytes);
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new(bytes: &[u8]) -> DcdReader {
        DcdReader {
            inner: RsDcdReader::new(Cursor::new(bytes.to_vec())),
        }
    }

    /// Read a frame at the given step index.
    ///
    /// # Arguments
    ///
    /// * `step` - Zero-based frame index
    ///
    /// # Returns
    ///
    /// A [`Frame`] if the step exists, or `undefined` if `step` is
    /// out of range.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string on parse errors (truncated record,
    /// malformed header, byte-order mismatch).
    #[wasm_bindgen]
    pub fn read(&mut self, step: usize) -> Result<Option<Frame>, JsValue> {
        let rs_frame = self
            .inner
            .read_step(step)
            .map_err(|e| JsValue::from_str(&format!("DCD read error: {}", e)))?;

        match rs_frame {
            Some(frame_data) => Ok(Some(Frame::from_rs(frame_data)?)),
            None => Ok(None),
        }
    }

    /// Return the number of frames in the DCD file.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string if the header cannot be parsed.
    #[wasm_bindgen]
    pub fn len(&mut self) -> Result<usize, JsValue> {
        self.inner
            .len()
            .map_err(|e| JsValue::from_str(&format!("DCD len error: {}", e)))
    }

    /// Check whether the file contains no frames.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string if the header cannot be parsed.
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&mut self) -> Result<bool, JsValue> {
        Ok(self.len()? == 0)
    }
}

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
/// const reader = new CIFReader(content);
/// const frame  = reader.read(0);
/// const atoms  = frame.getBlock("atoms");
/// const box    = frame.simbox;        // populated from the unit cell
/// ```
#[wasm_bindgen(js_name = CIFReader)]
pub struct CifReader {
    content: Vec<u8>,
    cached_len: Option<usize>,
}

#[wasm_bindgen(js_class = CIFReader)]
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
    /// const reader = new CIFReader(cifString);
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
        let frames = reader
            .read_all()
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
        let n = reader
            .read_all()
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
/// - `simbox`: voxel cell × dims in Å.
///
/// Cube is inherently single-frame (only `step = 0` is valid).
///
/// # Example (JavaScript)
///
/// ```js
/// const content = await file.text();
/// const reader  = new CubeReader(content);
/// const frame   = reader.read(0);
/// const grid    = frame.getBlock("grid");   // shape [nx, ny, nz]
/// const density = grid.copyColF("density"); // owned Float64Array
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
        let reader = BufReader::new(Cursor::new(self.content.as_slice()));
        let rs_frame = read_cube_from_reader(reader)
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
/// - `simbox`: triclinic POSCAR lattice in Å, fully periodic.
///
/// CHGCAR is single-frame; only `step = 0` is valid.
///
/// # Example (JavaScript)
///
/// ```js
/// const content = await file.text();
/// const reader  = new CHGCARReader(content);
/// const frame   = reader.read(0);
/// const grid    = frame.getBlock("grid");
/// const total   = grid.copyColF("total");
/// ```
#[wasm_bindgen(js_name = CHGCARReader)]
pub struct ChgcarReader {
    content: Vec<u8>,
    cached_len: Option<usize>,
}

#[wasm_bindgen(js_class = CHGCARReader)]
impl ChgcarReader {
    /// Create a new CHGCAR reader from the file's text content.
    #[wasm_bindgen(constructor)]
    pub fn new(content: &str) -> ChgcarReader {
        ChgcarReader {
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
        let reader = BufReader::new(Cursor::new(self.content.as_slice()));
        let rs_frame = read_chgcar_from_reader(reader)
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
/// are converted to angstrom on read (x10), matching every other molvis
/// reader. Each frame produces an `"atoms"` block (`resid`, `resname`,
/// `atom_name`, `atom_id`, `x`/`y`/`z`, optional `vx`/`vy`/`vz`) and a
/// `simbox` from the box-vector line.
#[wasm_bindgen(js_name = GROReader)]
pub struct GroReader {
    content: Vec<u8>,
    cached_len: Option<usize>,
}

#[wasm_bindgen(js_class = GROReader)]
impl GroReader {
    /// Create a new GRO reader from a string containing the file content.
    #[wasm_bindgen(constructor)]
    pub fn new(content: &str) -> GroReader {
        GroReader {
            content: content.as_bytes().to_vec(),
            cached_len: None,
        }
    }

    /// Read the frame at the given step index (0-based). Coordinates are
    /// converted nm -> angstrom before the frame is returned.
    #[wasm_bindgen]
    pub fn read(&mut self, step: usize) -> Result<Option<Frame>, JsValue> {
        let mut reader = RsGroReader::new(Cursor::new(self.content.as_slice()));
        for current in 0..=step {
            let rs_frame = reader
                .read_frame()
                .map_err(|e| JsValue::from_str(&format!("GRO read error: {}", e)))?;
            match rs_frame {
                Some(mut frame) if current == step => {
                    scale_nm_to_angstrom(&mut frame)?;
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
            .read_frame()
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
#[wasm_bindgen(js_name = MOL2Reader)]
pub struct Mol2Reader {
    content: Vec<u8>,
    cached_len: Option<usize>,
}

#[wasm_bindgen(js_class = MOL2Reader)]
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
                .read_frame()
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
            .read_frame()
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
/// selective-dynamics flags, velocities) and a periodic `simbox`.
/// Single-frame: any `step != 0` returns `undefined`.
#[wasm_bindgen(js_name = POSCARReader)]
pub struct PoscarReader {
    content: Vec<u8>,
    cached_len: Option<usize>,
}

#[wasm_bindgen(js_class = POSCARReader)]
impl PoscarReader {
    /// Create a new POSCAR reader from a string containing the file content.
    #[wasm_bindgen(constructor)]
    pub fn new(content: &str) -> PoscarReader {
        PoscarReader {
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
        let rs_frame = read_poscar_from_reader(Cursor::new(self.content.as_slice()))
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

/// GROMACS TRR binary trajectory reader.
///
/// TRR is the full-precision GROMACS trajectory (XDR, big-endian). Accepts the
/// file as raw bytes (`Uint8Array`). Each frame produces an `"atoms"` block
/// (`id`, `x`/`y`/`z`, optional `vx`/`vy`/`vz` and `fx`/`fy`/`fz`);
/// coordinates and box are converted nm -> angstrom on read. Box is attached
/// as `simbox` when the frame carries one.
#[wasm_bindgen(js_name = TRRReader)]
pub struct TrrReader {
    inner: RsTrrReader<Cursor<Vec<u8>>>,
}

#[wasm_bindgen(js_class = TRRReader)]
impl TrrReader {
    /// Create a new TRR reader from the file's raw bytes.
    #[wasm_bindgen(constructor)]
    pub fn new(bytes: &[u8]) -> TrrReader {
        TrrReader {
            inner: RsTrrReader::new(Cursor::new(bytes.to_vec())),
        }
    }

    /// Read a frame at the given step index (0-based). Coordinates are
    /// converted nm -> angstrom before the frame is returned.
    #[wasm_bindgen]
    pub fn read(&mut self, step: usize) -> Result<Option<Frame>, JsValue> {
        let rs_frame = self
            .inner
            .read_step(step)
            .map_err(|e| JsValue::from_str(&format!("TRR read error: {}", e)))?;
        match rs_frame {
            Some(mut frame) => {
                scale_nm_to_angstrom(&mut frame)?;
                Ok(Some(Frame::from_rs(frame)?))
            }
            None => Ok(None),
        }
    }

    /// Return the number of frames in the TRR file.
    #[wasm_bindgen]
    pub fn len(&mut self) -> Result<usize, JsValue> {
        self.inner
            .len()
            .map_err(|e| JsValue::from_str(&format!("TRR len error: {}", e)))
    }

    /// Check whether the file contains no frames.
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&mut self) -> Result<bool, JsValue> {
        Ok(self.len()? == 0)
    }
}

/// GROMACS XTC binary trajectory reader.
///
/// XTC is the compressed GROMACS trajectory (XDR, big-endian, lossy
/// coordinate compression). Accepts the file as raw bytes (`Uint8Array`).
/// Each frame produces an `"atoms"` block (`id`, `x`/`y`/`z`); coordinates
/// and box are converted nm -> angstrom on read. Box is attached as `simbox`.
#[wasm_bindgen(js_name = XTCReader)]
pub struct XtcReader {
    inner: RsXtcReader<Cursor<Vec<u8>>>,
}

#[wasm_bindgen(js_class = XTCReader)]
impl XtcReader {
    /// Create a new XTC reader from the file's raw bytes.
    #[wasm_bindgen(constructor)]
    pub fn new(bytes: &[u8]) -> XtcReader {
        XtcReader {
            inner: RsXtcReader::new(Cursor::new(bytes.to_vec())),
        }
    }

    /// Read a frame at the given step index (0-based). Coordinates are
    /// converted nm -> angstrom before the frame is returned.
    #[wasm_bindgen]
    pub fn read(&mut self, step: usize) -> Result<Option<Frame>, JsValue> {
        let rs_frame = self
            .inner
            .read_step(step)
            .map_err(|e| JsValue::from_str(&format!("XTC read error: {}", e)))?;
        match rs_frame {
            Some(mut frame) => {
                scale_nm_to_angstrom(&mut frame)?;
                Ok(Some(Frame::from_rs(frame)?))
            }
            None => Ok(None),
        }
    }

    /// Return the number of frames in the XTC file.
    #[wasm_bindgen]
    pub fn len(&mut self) -> Result<usize, JsValue> {
        self.inner
            .len()
            .map_err(|e| JsValue::from_str(&format!("XTC len error: {}", e)))
    }

    /// Check whether the file contains no frames.
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&mut self) -> Result<bool, JsValue> {
        Ok(self.len()? == 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn test_lammps_reader_len_then_read() {
        // Regression: `len()` used to call `read(0)` which flipped the
        // inner reader's `returned` flag, causing a subsequent `read(0)`
        // from the lazy FrameProvider in reader.ts to return None
        // ("lammps reader returned no frame at step 0").
        let data = "LAMMPS data\n\n\
                    2 atoms\n1 atom types\n\n\
                    0.0 10.0 xlo xhi\n0.0 10.0 ylo yhi\n0.0 10.0 zlo zhi\n\n\
                    Atoms\n\n\
                    1 1 1.0 2.0 3.0\n2 1 4.0 5.0 6.0\n";
        let mut reader = LammpsReader::new(data);
        assert_eq!(reader.len().expect("len"), 1);
        let frame = reader.read(0).expect("read").expect("frame");
        let atoms = frame.get_block("atoms").expect("atoms");
        assert_eq!(atoms.copy_col_f("x").expect("x").length(), 2);
    }

    #[wasm_bindgen_test]
    fn test_pdb_reader() {
        let pdb_content = r#"ATOM      1  C   MOL     1       1.000   2.000   3.000  1.00  0.00           C
ATOM      2  N   MOL     1       4.000   5.000   6.000  1.00  0.00           N
END"#;

        let mut reader = PdbReader::new(pdb_content);

        let frame = reader.read(0).expect("read failed").expect("no frame");

        let block = frame.get_block("atoms").expect("no atoms block");

        let x = block.copy_col_f("x").expect("no x column");
        assert_eq!(x.length(), 2);
        assert_eq!(x.get_index(0), 1.0);
        assert_eq!(x.get_index(1), 4.0);
    }
}
