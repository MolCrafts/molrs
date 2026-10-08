//! SMILES string parsing for the WASM API.
//!
//! [`readSmilesStr`](read_smiles_str) reads one molecule from a SMILES
//! string into a [`Frame`] (`molrs::io::read_smiles_str`);
//! [`SmilesIr.parse`](SmilesIr::parse) parses any SMILES string — a
//! `.`-separated set included — into its intermediate representation
//! (`molrs::io::smiles::SmilesIr::parse`), which `toFrame()` converts.
//!
//! # Typical workflow (JavaScript)
//!
//! ```js
//! import { readSmilesStr, Conformer } from "@molcrafts/molrs";
//!
//! const frame = readSmilesStr("c1ccccc1"); // benzene, 2D graph (no coords)
//! const mol3d = new Conformer("fast").generate(frame); // embed 3D coords
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
/// Call [`toFrame()`](SmilesIr::to_frame) to convert to a [`Frame`]
/// with `"atoms"` and `"bonds"` blocks.
///
/// # Example (JavaScript)
///
/// ```js
/// const ir = SmilesIr.parse("CCO");
/// console.log(ir.nComponents); // 1
///
/// const frame = ir.toFrame();
/// const atoms = frame.get("atoms");
/// console.log(atoms.copy("element")); // ["C", "C", "O", "H", ...], an owned copy
/// ```
#[wasm_bindgen(js_name = SmilesIr)]
pub struct SmilesIr {
    inner: molrs::io::smiles::SmilesIr,
}

#[wasm_bindgen(js_class = SmilesIr)]
impl SmilesIr {
    /// Parse a SMILES string — a `.`-separated set of molecules included —
    /// into its intermediate representation.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string if the SMILES string is malformed
    /// (e.g., unmatched ring closure digits, invalid atom symbols).
    #[wasm_bindgen(js_name = parse)]
    pub fn parse(smiles: &str) -> Result<SmilesIr, JsValue> {
        let inner = molrs::io::smiles::SmilesIr::parse(smiles)
            .map_err(|e| JsValue::from_str(&format!("SMILES parse error: {e}")))?;
        Ok(SmilesIr { inner })
    }

    /// Return the number of disconnected components in the SMILES.
    ///
    /// Components are separated by `.` in the SMILES string. For
    /// example, `"[Na+].[Cl-]"` has 2 components.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const ir = SmilesIr.parse("[Na+].[Cl-]");
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
    /// - `"atoms"` block: `element` (string), and implicit hydrogens
    ///   are added. No 3D coordinates are present -- use
    ///   [`Conformer`](crate::conformer::Conformer) to embed coordinates.
    /// - `"bonds"` block: `atomi`, `atomj` (u64, zero-based atom indices),
    ///   `bond_type` (u64: 1 single, 2 double, 3 triple, 4 aromatic) and
    ///   `bond_number` (u64: the localized Lewis/Kekulé integer, 0 when the
    ///   notation declared aromaticity without a phase — call
    ///   `assignAromaticity(frame)` to fill it in).
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
    /// const types = bonds.copy("bond_type");
    /// const numbers = bonds.copy("bond_number");
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
/// `SmilesIr.parse` and convert its components.
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
/// const mol3d = new Conformer("fast").generate(frame);
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

/// Write the molecule in `frame` (its `"atoms"` and `"bonds"`) as a SMILES
/// string — `molrs::io::write_smiles_str` with molrs's default emit options
/// (canonical, aromatic as marked, organic-subset hydrogens, no stereo, one
/// component), the defaults of Python's `write_smiles_str`.
///
/// # Example (JavaScript)
///
/// ```js
/// writeSmilesStr(readSmilesStr("OCC")); // "CCO"
/// ```
#[wasm_bindgen(js_name = writeSmilesStr)]
pub fn write_smiles_str(frame: &Frame) -> Result<String, JsValue> {
    let mol = frame.with_frame(|rs| {
        molrs::core::Atomistic::from_frame(rs)
            .map_err(|e| JsValue::from_str(&format!("Frame → Atomistic: {e}")))
    })?;
    molrs::io::write_smiles_str(&mol, &molrs::io::smiles::SmilesEmitOptions::default())
        .map_err(|e| JsValue::from_str(&format!("SMILES writing error: {e}")))
}

/// Read the molecule a CGsmiles string states into a [`Frame`] —
/// `molrs::io::read_cgsmiles_str`, its graph converted as `readSmilesStr`'s.
#[wasm_bindgen(js_name = readCgsmilesStr)]
pub fn read_cgsmiles_str(text: &str) -> Result<Frame, JsValue> {
    let mol = molrs::io::read_cgsmiles_str(text)
        .map_err(|e| JsValue::from_str(&format!("CGsmiles read error: {e}")))?;
    Frame::from_rs(
        mol.to_frame()
            .map_err(|e| JsValue::from_str(&format!("readCgsmilesStr: {e}")))?,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn write_smiles_str_writes_the_canonical_string() {
        let frame = read_smiles_str("OCC").expect("smiles");
        assert_eq!(write_smiles_str(&frame).expect("write"), "CCO");
    }

    #[wasm_bindgen_test]
    fn read_cgsmiles_str_reads_a_molecule() {
        let frame =
            read_cgsmiles_str("{[#OH][#PEO]|3[#OH]}.{#OH=[$]O,#PEO=[$]COC[$]}").expect("cgsmiles");
        assert!(frame.get("atoms").is_ok());
    }
}
