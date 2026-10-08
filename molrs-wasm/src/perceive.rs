//! Chemical perception for the WASM / JS surface.
//!
//! Mirrors the Rust [`molrs::perceive`] free functions, Frame-in / Frame-out:
//! each `assign*` writes the perceived fact onto a **new** [`Frame`] and never
//! touches its input; `addHydrogens` / `removeHydrogens` are graph edits.
//!
//! # Example (JavaScript)
//!
//! ```js
//! const withH = addHydrogens(frame);
//! const heavy = removeHydrogens(withH);
//! const rings = assignRings(frame);   // atoms/bonds gain is_in_ring, n_rings
//! ```

use wasm_bindgen::prelude::*;

use molrs::core::Atomistic;

use crate::core::frame::Frame;

/// Run `perception` on the molecule `frame` holds and hand back the result as
/// a new frame.
fn frame_out(
    frame: &Frame,
    perception: impl FnOnce(&Atomistic) -> Result<Atomistic, String>,
) -> Result<Frame, JsValue> {
    let mol = frame_to_atomistic(frame)?;
    let out = perception(&mol).map_err(|e| JsValue::from_str(&e))?;
    Frame::from_rs(
        out.to_frame()
            .map_err(|e| JsValue::from_str(&format!("toFrame: {e}")))?,
    )
}

/// Perceive rings (SSSR) and record them on a new frame.
///
/// [`molrs::perceive::assign_rings`]: every atom and every bond receives
/// `is_in_ring` (`0` / `1`) and `n_rings`, the number of SSSR rings it belongs
/// to — acyclic ones included, flagged `0` rather than left unset. Input needs
/// `"atoms"` and `"bonds"` (`atomi` / `atomj`).
///
/// # Errors
///
/// Throws if the frame cannot be read as an atomistic molecule, or if a
/// property of the result contradicts the Frame schema on the way out.
///
/// # Example (JavaScript)
///
/// ```js
/// const rings = assignRings(frame);
/// rings.get("atoms").copy("is_in_ring"); // Int32Array, 1 for ring atoms
/// ```
#[wasm_bindgen(js_name = assignRings)]
pub fn assign_rings(frame: &Frame) -> Result<Frame, JsValue> {
    frame_out(frame, |mol| Ok(molrs::perceive::assign_rings(mol)))
}

/// Add explicit hydrogens for unfilled heavy-atom valence
/// ([`molrs::perceive::add_hydrogens`]). Input needs `"atoms"` (`element`,
/// optional `x`/`y`/`z`) and preferably `"bonds"`.
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
#[wasm_bindgen(js_name = addHydrogens)]
pub fn add_hydrogens(frame: &Frame) -> Result<Frame, JsValue> {
    frame_out(frame, |mol| {
        molrs::perceive::add_hydrogens(mol).map_err(|e| format!("addHydrogens: {e}"))
    })
}

/// Assign a localized (Kekulé) `bond_number` to every aromatic bond
/// ([`molrs::perceive::assign_kekule_bond_orders`]). Kekulization and nothing
/// else — a frame whose aromatic bonds are not marked yet comes back
/// unchanged, because deciding *which* bonds are aromatic belongs to
/// `assignAromaticity`.
///
/// # Errors
///
/// Throws if the frame cannot be read as an atomistic molecule, or if a
/// property of the result contradicts the Frame schema on the way out.
#[wasm_bindgen(js_name = assignKekuleBondOrders)]
pub fn assign_kekule_bond_orders(frame: &Frame) -> Result<Frame, JsValue> {
    frame_out(frame, |mol| {
        Ok(molrs::perceive::assign_kekule_bond_orders(mol))
    })
}

/// Bring a frame to the standard aromatic representation
/// ([`molrs::perceive::assign_aromaticity`]).
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
#[wasm_bindgen(js_name = assignAromaticity)]
pub fn assign_aromaticity(frame: &Frame) -> Result<Frame, JsValue> {
    frame_out(frame, |mol| Ok(molrs::perceive::assign_aromaticity(mol)))
}

/// Remove terminal (degree-1) explicit hydrogen atoms
/// ([`molrs::perceive::remove_hydrogens`]); non-terminal H is left in place.
///
/// # Errors
///
/// Throws if the frame cannot be read as an atomistic molecule, if
/// stripping reports a stale atom handle on the graph it built, or if a
/// property of the result contradicts the Frame schema on the way out.
#[wasm_bindgen(js_name = removeHydrogens)]
pub fn remove_hydrogens(frame: &Frame) -> Result<Frame, JsValue> {
    frame_out(frame, |mol| {
        molrs::perceive::remove_hydrogens(mol).map_err(|e| format!("removeHydrogens: {e}"))
    })
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
    use molrs::op::Idx;
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
    fn assign_rings_flags_ring_atoms() {
        let out = assign_rings(&ring_with_tail()).unwrap();
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
