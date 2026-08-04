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
//! ```

use wasm_bindgen::prelude::*;

use molrs::perceive::Perceive as RsPerceive;
use molrs::system::atomistic::Atomistic;
use molrs::remove_hydrogens;

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
    /// Throws if the frame cannot be read as an atomistic molecule.
    #[wasm_bindgen(js_name = findHydrogens)]
    pub fn find_hydrogens(&self, frame: &Frame) -> Result<Frame, JsValue> {
        let mol = frame_to_atomistic(frame)?;
        let out = self.inner.find_hydrogens(&mol);
        Frame::from_rs(out.to_frame())
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
    /// Throws if the frame cannot be read as an atomistic molecule.
    #[wasm_bindgen(js_name = findKekuleOrders)]
    pub fn find_kekule_orders(&self, frame: &Frame) -> Result<Frame, JsValue> {
        let mol = frame_to_atomistic(frame)?;
        let out = self.inner.find_kekule_orders(&mol);
        Frame::from_rs(out.to_frame())
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
    /// Throws if the frame cannot be read as an atomistic molecule.
    #[wasm_bindgen(js_name = findAromaticity)]
    pub fn find_aromaticity(&self, frame: &Frame) -> Result<Frame, JsValue> {
        let mol = frame_to_atomistic(frame)?;
        let out = self.inner.find_aromaticity(&mol);
        Frame::from_rs(out.to_frame())
    }

    /// Remove terminal (degree-1) explicit hydrogen atoms.
    ///
    /// Graph-in / graph-out; non-terminal H is left in place. Uses
    /// [`molrs::remove_hydrogens`] (not yet on the Rust builder — same contract).
    ///
    /// # Errors
    ///
    /// Throws if the frame cannot be read as an atomistic molecule.
    #[wasm_bindgen(js_name = removeHydrogens)]
    pub fn remove_hydrogens(&self, frame: &Frame) -> Result<Frame, JsValue> {
        let mol = frame_to_atomistic(frame)?;
        let out = remove_hydrogens(&mol);
        Frame::from_rs(out.to_frame())
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
