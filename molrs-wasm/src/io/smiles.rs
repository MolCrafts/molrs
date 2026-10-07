//! SMILES string parsing for the WASM API.
//!
//! [`readSmilesStr`](read_smiles_str) reads one molecule from a SMILES
//! string into a [`Frame`] (`molrs::io::read_smiles_str`);
//! [`SmilesIR.parse`](SmilesIR::parse) parses any SMILES string — a
//! `.`-separated set included — into its intermediate representation
//! (`molrs::io::smiles::SmilesIR::parse`), which `toFrame()` converts.
//!
//! # Typical workflow (JavaScript)
//!
//! ```js
//! import { readSmilesStr, generate3D } from "@molcrafts/molrs";
//!
//! const frame = readSmilesStr("c1ccccc1"); // benzene, 2D graph (no coords)
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
/// const ir = SmilesIR.parse("CCO");
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
    /// Parse a SMILES string — a `.`-separated set of molecules included —
    /// into its intermediate representation.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string if the SMILES string is malformed
    /// (e.g., unmatched ring closure digits, invalid atom symbols).
    #[wasm_bindgen(js_name = parse)]
    pub fn parse(smiles: &str) -> Result<SmilesIR, JsValue> {
        let inner = molrs::io::smiles::SmilesIR::parse(smiles)
            .map_err(|e| JsValue::from_str(&format!("SMILES parse error: {e}")))?;
        Ok(SmilesIR { inner })
    }

    /// Return the number of disconnected components in the SMILES.
    ///
    /// Components are separated by `.` in the SMILES string. For
    /// example, `"[Na+].[Cl-]"` has 2 components.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const ir = SmilesIR.parse("[Na+].[Cl-]");
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
        let mol = self
            .inner
            .to_atomistic()
            .map_err(|e| JsValue::from_str(&format!("IR -> Atomistic: {e}")))?;
        Frame::from_rs(
            mol.to_frame()
                .map_err(|e| JsValue::from_str(&format!("toFrame: {e}")))?,
        )
    }
}

/// Read one molecule from a SMILES string into a [`Frame`] with `"atoms"`
/// and `"bonds"` blocks — connectivity only, no implicit hydrogens added, no
/// coordinates. A `.`-separated set is refused: parse it with
/// `SmilesIR.parse` and convert its components.
///
/// # Errors
///
/// Throws a `JsValue` string if the SMILES string is malformed or names more
/// than one molecule.
///
/// # Example (JavaScript)
///
/// ```js
/// const frame = readSmilesStr("CCO");
/// const mol3d = generate3D(frame, "fast");
/// ```
#[wasm_bindgen(js_name = readSmilesStr)]
pub fn read_smiles_str(smiles: &str) -> Result<Frame, JsValue> {
    let mol = molrs::io::read_smiles_str(smiles)
        .map_err(|e| JsValue::from_str(&format!("SMILES read error: {e}")))?;
    Frame::from_rs(
        mol.to_frame()
            .map_err(|e| JsValue::from_str(&format!("readSmilesStr: {e}")))?,
    )
}
