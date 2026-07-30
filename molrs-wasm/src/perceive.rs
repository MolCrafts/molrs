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
