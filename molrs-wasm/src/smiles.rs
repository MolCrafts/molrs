//! SMILES string parsing for the WASM API.
//!
//! Provides [`parseSMILES`](parse_smiles) to convert a SMILES notation
//! string into a [`SmilesIR`](WasmSmilesIR) intermediate representation,
//! which can then be converted to a [`Frame`] with atoms and bonds.
//!
//! # Typical workflow (JavaScript)
//!
//! ```js
//! import { parseSMILES, generate3D } from "molrs-wasm";
//!
//! const ir    = parseSMILES("c1ccccc1"); // benzene
//! const frame = ir.toFrame();            // 2D graph (no coords)
//! const mol3d = generate3D(frame, "fast"); // embed 3D coords
//! ```
//!
//! # References
//!
//! - Weininger, D. (1988). SMILES, a chemical language and information
//!   system. *J. Chem. Inf. Comput. Sci.*, 28(1), 31-36.

use crate::core::frame::Frame;
use wasm_bindgen::prelude::*;

/// Intermediate representation of a parsed SMILES string.
///
/// Holds the molecular graph(s) parsed from a SMILES string. A single
/// SMILES string can encode multiple disconnected molecules separated
/// by `.` (e.g., `"[Na+].[Cl-]"`).
///
/// Call [`toFrame()`](WasmSmilesIR::to_frame) to convert to a [`Frame`]
/// with `"atoms"` and `"bonds"` blocks.
///
/// # Example (JavaScript)
///
/// ```js
/// const ir = parseSMILES("CCO");
/// console.log(ir.nComponents); // 1
///
/// const frame = ir.toFrame();
/// const atoms = frame.getBlock("atoms");
/// console.log(atoms.copyColStr("element")); // ["C", "C", "O", "H", ...]
/// ```
#[wasm_bindgen(js_name = SmilesIR)]
pub struct WasmSmilesIR {
    inner: molrs_io::smiles::SmilesIR,
}

#[wasm_bindgen(js_class = SmilesIR)]
impl WasmSmilesIR {
    /// Return the number of disconnected components in the SMILES.
    ///
    /// Components are separated by `.` in the SMILES string. For
    /// example, `"[Na+].[Cl-]"` has 2 components.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const ir = parseSMILES("[Na+].[Cl-]");
    /// console.log(ir.nComponents); // 2
    /// ```
    #[wasm_bindgen(getter, js_name = nComponents)]
    pub fn n_components(&self) -> usize {
        self.inner.components.len()
    }

    /// Convert the intermediate representation to a [`Frame`].
    ///
    /// The resulting frame contains:
    ///
    /// - `"atoms"` block: `symbol` (string), and implicit hydrogens
    ///   are added. No 3D coordinates are present -- use
    ///   [`generate3D`](crate::generate_3d_wasm) to embed coordinates.
    /// - `"bonds"` block: `i`, `j` (u32, zero-based atom indices),
    ///   `order` (F, bond order: 1.0 = single, 1.5 = aromatic,
    ///   2.0 = double, 3.0 = triple).
    ///
    /// # Returns
    ///
    /// A new [`Frame`] with atoms and bonds.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string if the conversion fails (e.g.,
    /// invalid valence).
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const frame = ir.toFrame();
    /// const bonds = frame.getBlock("bonds");
    /// const order = bonds.copyColF("order");
    /// ```
    #[wasm_bindgen(js_name = toFrame)]
    pub fn to_frame(&self) -> Result<Frame, JsValue> {
        let mol = molrs_io::smiles::to_atomistic(&self.inner)
            .map_err(|e| JsValue::from_str(&format!("IR -> Atomistic: {e}")))?;
        Frame::from_rs(mol.to_frame())
    }
}

/// Parse a SMILES notation string into an intermediate representation.
///
/// Supports standard SMILES features including ring closures,
/// branching, stereochemistry markers, and aromatic atoms.
///
/// # Arguments
///
/// * `smiles` - SMILES notation string (e.g., `"CCO"` for ethanol,
///   `"c1ccccc1"` for benzene, `"[Na+].[Cl-]"` for NaCl)
///
/// # Returns
///
/// A [`SmilesIR`](WasmSmilesIR) object. Call `.toFrame()` to convert
/// to a [`Frame`] with atoms and bonds blocks.
///
/// # Errors
///
/// Throws a `JsValue` string if the SMILES string is malformed
/// (e.g., unmatched ring closure digits, invalid atom symbols).
///
/// # Example (JavaScript)
///
/// ```js
/// const ir = parseSMILES("CCO");
/// const frame = ir.toFrame();
/// const mol3d = generate3D(frame, "fast");
/// ```
#[wasm_bindgen(js_name = parseSMILES)]
pub fn parse_smiles(smiles: &str) -> Result<WasmSmilesIR, JsValue> {
    let inner = molrs_io::smiles::parse_smiles(smiles)
        .map_err(|e| JsValue::from_str(&format!("SMILES parse error: {e}")))?;
    Ok(WasmSmilesIR { inner })
}
