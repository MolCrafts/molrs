//! Force-field WASM face — typifiers and the potentials they compile; mirrors
//! native molrs composition (`molrs::ff::typifier`, `molrs::ff::potential`).
//! Minimizing with those potentials is `optimize`'s job (`Lbfgs`).
//!
//! ```js
//! const typifier = new UFFTypifier();
//! const typed    = typifier.typify(frame);
//! const pots     = typifier.toPotentials(typed);   // no .forcefield()
//! ```
//!
//! No `typifyUff` / `insertIntramolecularPairs` façades, and no force-field
//! handle: each typifier class wraps a native `Typing<…>` whose accumulated
//! output (`forcefield()`) and library (`library()`) stay private. This is a
//! known asymmetry with Python, which exposes both as copies.

use std::sync::Arc;

use wasm_bindgen::prelude::*;

use molrs::core::Atomistic;
use molrs::ff::forcefield::ForceField as RsForceField;
use molrs::ff::potential::{PotentialCompiler, Potentials as RsPotentials};
use molrs::ff::typifier::Typing;
use molrs::ff::typifier::UFFTypifier as RsUFF;
use molrs::ff::typifier::mmff::{MMFF94STypifier as RsMMFF94S, MMFF94Typifier as RsMMFF94};

use crate::core::frame::Frame;

// ── Typifiers ───────────────────────────────────────────────────────────────

macro_rules! wasm_typifier {
    (
        $(#[$meta:meta])*
        $JsName:ident, $RsType:ty, $ctor:expr
    ) => {
        $(#[$meta])*
        #[wasm_bindgen(js_name = $JsName)]
        pub struct $JsName {
            inner: Typing<$RsType>,
        }

        #[wasm_bindgen(js_class = $JsName)]
        impl $JsName {
            #[wasm_bindgen(constructor)]
            pub fn new() -> $JsName {
                $JsName {
                    inner: Typing::new($ctor),
                }
            }

            /// Typify a molecular [`Frame`]. Returns a **new** labeled frame;
            /// `frame` is untouched. Every call accumulates its definitions
            /// into this typifier's private output force field; a conflicting
            /// definition is an error and leaves the output unchanged.
            ///
            /// Native: `Typing::new(typifier).typify(&mol)?.to_frame()?`.
            pub fn typify(&mut self, frame: &Frame) -> Result<Frame, JsValue> {
                let mol = frame.with_frame(|rs| {
                    Atomistic::from_frame(rs).map_err(|e| {
                        JsValue::from_str(&format!("Frame → Atomistic: {e}"))
                    })
                })?;
                let typed = self
                    .inner
                    .typify(&mol)
                    .map_err(|e| JsValue::from_str(&e))?;
                Frame::from_rs(
                    typed.to_frame().map_err(|e| {
                        JsValue::from_str(&format!("toFrame: {e}"))
                    })?,
                )
            }

            /// Compile molecule-bound potentials from a **typed** frame, using
            /// the output force field accumulated by [`typify`](Self::typify)
            /// (only the definitions typing has assigned — call `typify` first).
            ///
            /// Non-bonded terms need a `pairs` block; `Lbfgs.minimize` installs
            /// that list (from a caller-supplied [`Neighbors`] table or an
            /// internal bruteforce topology list) and recompiles before
            /// minimizing.
            /// Calling this alone with no `pairs` yields bonded-only kernels.
            ///
            /// Native: `PotentialCompiler::new(typing.forcefield()).compile(&frame)?` —
            /// the FF handle stays private, and WASM exposes no `PotentialCompiler`
            /// class; it collapses that to one method on the typifier.
            #[wasm_bindgen(js_name = toPotentials)]
            pub fn to_potentials(&self, frame: &Frame) -> Result<Potentials, JsValue> {
                let pots = frame.with_frame(|rs| {
                    PotentialCompiler::new(self.inner.forcefield())
                        .compile(rs)
                        .map_err(|e| JsValue::from_str(&format!("toPotentials: {e}")))
                })?;
                Ok(Potentials {
                    ff: self.inner.forcefield().clone(),
                    inner: Arc::new(pots),
                })
            }
        }

        impl Default for $JsName {
            fn default() -> Self {
                Self::new()
            }
        }
    };
}

wasm_typifier!(
    /// Universal Force Field typifier (full RDKit default table).
    UFFTypifier,
    RsUFF,
    RsUFF::new()
);

wasm_typifier!(
    /// MMFF94 typifier.
    MMFF94Typifier,
    RsMMFF94,
    RsMMFF94::new()
);

wasm_typifier!(
    /// MMFF94s typifier (static / planar amide N).
    MMFF94STypifier,
    RsMMFF94S,
    RsMMFF94S::new()
);

// ── Potentials ──────────────────────────────────────────────────────────────

/// Compiled kernels. Holds the force-field skeleton so [`Lbfgs`](crate::optimize::Lbfgs) can recompile
/// after installing a neighbour list.
#[wasm_bindgen(js_name = Potentials)]
pub struct Potentials {
    pub(crate) ff: RsForceField,
    pub(crate) inner: Arc<RsPotentials>,
}

#[wasm_bindgen(js_class = Potentials)]
impl Potentials {
    /// `{ energy: number, forces: Float64Array }` for flat 3N coordinates.
    #[wasm_bindgen(js_name = energyForces)]
    pub fn energy_forces(&self, coords: &js_sys::Float64Array) -> Result<JsValue, JsValue> {
        let mut buf = vec![0.0; coords.length() as usize];
        coords.copy_to(&mut buf);
        let (e, f) = self.inner.calc_energy_forces(&buf);
        let obj = js_sys::Object::new();
        js_sys::Reflect::set(&obj, &"energy".into(), &JsValue::from_f64(e))?;
        let fa = js_sys::Float64Array::new_with_length(f.len() as u32);
        fa.copy_from(&f);
        js_sys::Reflect::set(&obj, &"forces".into(), &fa)?;
        Ok(obj.into())
    }
}
