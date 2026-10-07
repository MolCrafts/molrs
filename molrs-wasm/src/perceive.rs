//! Chemical perception for the WASM / JS surface.
//!
//! Mirrors the Rust [`molrs::perceive::Perceive`] builder: one type, graph-in /
//! graph-out (here **Frame-in / Frame-out**), non-mutating methods. Free
//! functions like a standalone `addHydrogens(frame)` are intentionally not
//! exported — call through [`Perceive`].
//!
//! # Example (JavaScript)
//!
//! ```js
//! const p = new Perceive();
//! const withH = p.findHydrogens(frame);
//! const heavy = p.removeHydrogens(withH);
//! const rings = p.findRings(frame);   // atoms/bonds gain is_in_ring, n_rings
//! ```

use wasm_bindgen::prelude::*;

use molrs::core::Atomistic;
use molrs::perceive::Perceive as RsPerceive;
use molrs::perceive::hydrogens::remove_hydrogens;

use crate::core::frame::Frame;

/// Chemical perception builder (WASM face of [`molrs::perceive::Perceive`]).
///
/// Stateless today — construct once and call `find*` / `remove*` methods.
/// Each method returns a **new** [`Frame`]; the input is never modified.
///
/// # Example (JavaScript)
///
/// ```js
/// const p = new Perceive();
/// const withH = p.findHydrogens(frame);
/// ```
#[wasm_bindgen(js_name = Perceive)]
pub struct Perceive {
    inner: RsPerceive,
}

#[wasm_bindgen(js_class = Perceive)]
impl Perceive {
    /// Create a perception builder with default settings.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: RsPerceive::new(),
        }
    }

    /// Perceive rings (SSSR) and record them on the frame.
    ///
    /// Wraps [`molrs::perceive::Perceive::find_rings`]: every atom and every
    /// bond receives `is_in_ring` (`0` / `1`) and `n_rings`, the number of SSSR
    /// rings it belongs to — acyclic ones included, flagged `0` rather than
    /// left unset. Input needs `"atoms"` and `"bonds"` (`atomi` / `atomj`).
    ///
    /// # Errors
    ///
    /// Throws if the frame cannot be read as an atomistic molecule, or if a
    /// property of the result contradicts the Frame schema on the way out.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const rings = new Perceive().findRings(frame);
    /// rings.get("atoms").copy("is_in_ring"); // Int32Array, 1 for ring atoms
    /// ```
    #[wasm_bindgen(js_name = findRings)]
    pub fn find_rings(&self, frame: &Frame) -> Result<Frame, JsValue> {
        let mol = frame_to_atomistic(frame)?;
        let out = self.inner.find_rings(&mol);
        Frame::from_rs(
            out.to_frame()
                .map_err(|e| JsValue::from_str(&format!("toFrame: {e}")))?,
        )
    }

    /// Add explicit hydrogens for unfilled heavy-atom valence.
    ///
    /// Wraps [`molrs::perceive::Perceive::find_hydrogens`]. Input needs
    /// `"atoms"` (`element`, optional `x`/`y`/`z`) and preferably `"bonds"`
    /// (`atomi`/`atomj`, float `order`).
    ///
    /// When coordinates are present, H is placed at standard X–H lengths along
    /// tetrahedral valence-completing directions (geometry only — force fields
    /// may refine).
    ///
    /// # Errors
    ///
    /// Throws if the frame cannot be read as an atomistic molecule, if
    /// repletion reports a stale atom handle on the graph it built, or if a
    /// property of the result contradicts the Frame schema on the way out.
    #[wasm_bindgen(js_name = findHydrogens)]
    pub fn find_hydrogens(&self, frame: &Frame) -> Result<Frame, JsValue> {
        let mol = frame_to_atomistic(frame)?;
        let out = self
            .inner
            .find_hydrogens(&mol)
            .map_err(|e| JsValue::from_str(&format!("findHydrogens: {e}")))?;
        Frame::from_rs(
            out.to_frame()
                .map_err(|e| JsValue::from_str(&format!("toFrame: {e}")))?,
        )
    }

    /// Assign a localized (Kekulé) `bond_number` to every aromatic bond.
    ///
    /// Wraps [`molrs::perceive::Perceive::find_kekule_orders`]. Kekulization and
    /// nothing else — a frame whose aromatic bonds are not marked yet comes back
    /// unchanged, because deciding *which* bonds are aromatic belongs to
    /// [`findAromaticity`](Self::find_aromaticity).
    ///
    /// # Errors
    ///
    /// Throws if the frame cannot be read as an atomistic molecule, or if a
    /// property of the result contradicts the Frame schema on the way out.
    #[wasm_bindgen(js_name = findKekuleOrders)]
    pub fn find_kekule_orders(&self, frame: &Frame) -> Result<Frame, JsValue> {
        let mol = frame_to_atomistic(frame)?;
        let out = self.inner.find_kekule_orders(&mol);
        Frame::from_rs(
            out.to_frame()
                .map_err(|e| JsValue::from_str(&format!("toFrame: {e}")))?,
        )
    }

    /// Bring a frame to the standard aromatic representation.
    ///
    /// On return every aromatic atom carries `is_aromatic`, every aromatic bond
    /// carries `bond_type = 4`, and every bond carries an integer `bond_number`
    /// — the localized Lewis structure. Nothing carries a fractional order.
    ///
    /// A renderer reads `bond_type` **first**: `4` is aromatic and may be drawn
    /// either as a uniform aromatic style or, in Kekulé mode, using
    /// `bond_number`. Reading `bond_number` alone and calling anything above 1
    /// a double bond is what drew benzene as six double bonds.
    ///
    /// # Errors
    ///
    /// Throws if the frame cannot be read as an atomistic molecule, or if a
    /// property of the result contradicts the Frame schema on the way out.
    #[wasm_bindgen(js_name = findAromaticity)]
    pub fn find_aromaticity(&self, frame: &Frame) -> Result<Frame, JsValue> {
        let mol = frame_to_atomistic(frame)?;
        let out = self.inner.find_aromaticity(&mol);
        Frame::from_rs(
            out.to_frame()
                .map_err(|e| JsValue::from_str(&format!("toFrame: {e}")))?,
        )
    }

    /// Remove terminal (degree-1) explicit hydrogen atoms.
    ///
    /// Graph-in / graph-out; non-terminal H is left in place. Uses
    /// [`molrs::perceive::hydrogens::remove_hydrogens`] (not yet on the Rust builder — same contract).
    ///
    /// # Errors
    ///
    /// Throws if the frame cannot be read as an atomistic molecule, if
    /// stripping reports a stale atom handle on the graph it built, or if a
    /// property of the result contradicts the Frame schema on the way out.
    #[wasm_bindgen(js_name = removeHydrogens)]
    pub fn remove_hydrogens(&self, frame: &Frame) -> Result<Frame, JsValue> {
        let mol = frame_to_atomistic(frame)?;
        let out = remove_hydrogens(&mol)
            .map_err(|e| JsValue::from_str(&format!("removeHydrogens: {e}")))?;
        Frame::from_rs(
            out.to_frame()
                .map_err(|e| JsValue::from_str(&format!("toFrame: {e}")))?,
        )
    }
}

impl Default for Perceive {
    fn default() -> Self {
        Self::new()
    }
}

fn frame_to_atomistic(frame: &Frame) -> Result<Atomistic, JsValue> {
    frame.with_frame(|rs_frame| {
        Atomistic::from_frame(rs_frame)
            .map_err(|e| JsValue::from_str(&format!("Frame → Atomistic: {e}")))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::core::{Block, keys};
    use molrs::op::types::Idx;
    use ndarray::{Array1, ArrayD};
    use wasm_bindgen_test::*;

    /// Cyclopropane carbons plus one pendant carbon on atom 0.
    fn ring_with_tail() -> Frame {
        let mut atoms = Block::new();
        let elements: ArrayD<String> = Array1::from_vec(vec!["C".to_string(); 4]).into_dyn();
        atoms.insert(keys::ELEMENT, elements).unwrap();
        let mut bonds = Block::new();
        bonds
            .insert(
                keys::ATOMI,
                Array1::<Idx>::from_vec(vec![0, 1, 2, 0]).into_dyn(),
            )
            .unwrap();
        bonds
            .insert(
                keys::ATOMJ,
                Array1::<Idx>::from_vec(vec![1, 2, 0, 3]).into_dyn(),
            )
            .unwrap();
        let mut rs_frame = molrs::core::Frame::new();
        rs_frame.insert("atoms", atoms);
        rs_frame.insert("bonds", bonds);
        Frame::from_rs(rs_frame).unwrap()
    }

    #[wasm_bindgen_test]
    fn find_rings_flags_ring_atoms() {
        let out = Perceive::new().find_rings(&ring_with_tail()).unwrap();
        let flags: Vec<i32> = out
            .with_frame(|f| {
                let column = f.get("atoms").and_then(|a| a.get("is_in_ring"));
                Ok(column
                    .and_then(|c| c.as_int())
                    .map(|a| a.iter().copied().collect())
                    .unwrap_or_default())
            })
            .unwrap();
        assert_eq!(flags, vec![1, 1, 1, 0]);
    }
}
