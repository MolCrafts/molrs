//! Static dielectric constant — WASM face of the `molrs::compute`
//! dielectric family: `static_dielectric_constant` and
//! `static_dielectric_constant_components`, free functions as in Rust and
//! Python.

use super::{array2, js_value};
use molrs::op::F;
use serde::Serialize;
use wasm_bindgen::prelude::*;

/// Neumann static dielectric constant from the fluctuation of the total
/// dipole — molrs `compute::static_dielectric_constant`.
///
/// `dipoleMoments` is the flat `(nFrames, 3)` total dipole per frame (e Å),
/// `volume` the box volume (Å³), `temperature` in K; `epsilonInf` defaults to
/// 1 (conducting boundary).
#[wasm_bindgen(js_name = staticDielectricConstant)]
pub fn static_dielectric_constant(
    dipole_moments: &[F],
    n_frames: usize,
    volume: F,
    temperature: F,
    epsilon_inf: Option<F>,
) -> Result<F, JsValue> {
    let dipoles = array2(
        dipole_moments,
        n_frames,
        3,
        "staticDielectricConstant dipoleMoments",
    )?;
    molrs::compute::static_dielectric_constant(
        &dipoles,
        volume,
        temperature,
        epsilon_inf.unwrap_or(1.0),
    )
    .map_err(|e| JsValue::from_str(&format!("staticDielectricConstant: {e}")))
}

/// The per-axis terms of [`staticDielectricConstant`](static_dielectric_constant)
/// — molrs `compute::static_dielectric_constant_components`, a
/// `StaticDielectricResult`: `{ eps, epsMean, fluctuation, dipoleMean,
/// dipoleSqMean, epsilonInf, nFrames }`.
#[wasm_bindgen(js_name = staticDielectricConstantComponents)]
pub fn static_dielectric_constant_components(
    dipole_moments: &[F],
    n_frames: usize,
    volume: F,
    temperature: F,
    epsilon_inf: Option<F>,
) -> Result<JsValue, JsValue> {
    #[derive(Serialize)]
    #[serde(rename_all = "camelCase")]
    struct Out {
        eps: Vec<F>,
        eps_mean: F,
        fluctuation: Vec<F>,
        dipole_mean: Vec<F>,
        dipole_sq_mean: Vec<F>,
        epsilon_inf: F,
        n_frames: usize,
    }
    let dipoles = array2(
        dipole_moments,
        n_frames,
        3,
        "staticDielectricConstantComponents dipoleMoments",
    )?;
    let c = molrs::compute::static_dielectric_constant_components(
        &dipoles,
        volume,
        temperature,
        epsilon_inf.unwrap_or(1.0),
    )
    .map_err(|e| JsValue::from_str(&format!("staticDielectricConstantComponents: {e}")))?;
    js_value(&Out {
        eps: c.eps.to_vec(),
        eps_mean: c.eps_mean,
        fluctuation: c.fluctuation.to_vec(),
        dipole_mean: c.dipole_mean.to_vec(),
        dipole_sq_mean: c.dipole_sq_mean.to_vec(),
        epsilon_inf: c.epsilon_inf,
        n_frames: c.n_frames,
    })
}
