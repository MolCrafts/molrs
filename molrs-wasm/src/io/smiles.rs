//! SMILES string parsing for the WASM API.
//!
//! Provides [`parseSMILES`](parse_smiles) to convert a SMILES notation
//! string into a [`SmilesIR`](SmilesIR) intermediate representation,
//! which can then be converted to a [`Frame`] with atoms and bonds.
//!
//! # Typical workflow (JavaScript)
//!
//! ```js
//! import { parseSMILES, generate3D } from "@molcrafts/molrs";
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
/// Call [`toFrame()`](SmilesIR::to_frame) to convert to a [`Frame`]
/// with `"atoms"` and `"bonds"` blocks.
///
/// # Example (JavaScript)
///
/// ```js
/// const ir = parseSMILES("CCO");
/// console.log(ir.nComponents); // 1
///
/// const frame = ir.toFrame();
/// const atoms = frame.get("atoms");
/// console.log(atoms.copy("element")); // ["C", "C", "O", "H", ...], an owned copy
/// ```
#[wasm_bindgen(js_name = SmilesIR)]
pub struct SmilesIR {
    inner: molrs::io::smiles::SmilesIR,
}

#[wasm_bindgen(js_class = SmilesIR)]
impl SmilesIR {
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
    ///   [`generate3D`](crate::conformer::generate_3d_wasm) to embed coordinates.
    /// - `"bonds"` block: `atomi`, `atomj` (u64, zero-based atom indices),
    ///   `bond_type` (u64: 1 single, 2 double, 3 triple, 4 aromatic) and
    ///   `bond_number` (u64: the localized Lewis/Kekulé integer, 0 when the
    ///   notation declared aromaticity without a phase — call
    ///   `new Perceive().findAromaticity(frame)` to fill it in).
    ///
    /// # Returns
    ///
    /// A new [`Frame`] with atoms and bonds.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string if the conversion fails (e.g., invalid
    /// valence), or if a property of the result contradicts the Frame schema
    /// on the way out.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const frame = ir.toFrame();
    /// const bonds = frame.get("bonds");
    /// const types = bonds.get("bond_type");
    /// const numbers = bonds.get("bond_number");
    /// ```
    #[wasm_bindgen(js_name = toFrame)]
    pub fn to_frame(&self) -> Result<Frame, JsValue> {
        let mol = molrs::io::smiles::to_atomistic(&self.inner)
            .map_err(|e| JsValue::from_str(&format!("IR -> Atomistic: {e}")))?;
        Frame::from_rs(
            mol.to_frame()
                .map_err(|e| JsValue::from_str(&format!("toFrame: {e}")))?,
        )
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
/// A [`SmilesIR`](SmilesIR) object. Call `.toFrame()` to convert
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
pub fn parse_smiles(smiles: &str) -> Result<SmilesIR, JsValue> {
    let inner = molrs::io::smiles::parse_smiles(smiles)
        .map_err(|e| JsValue::from_str(&format!("SMILES parse error: {e}")))?;
    Ok(SmilesIR { inner })
}
