//! Static dielectric constant — WASM face of the `molrs::compute`
//! dielectric family.

use super::{array2, js_value};
use molrs::op::F;
use serde::Serialize;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = StaticDielectric)]
pub struct StaticDielectric {
    volume: F,
    temperature: F,
    epsilon_inf: Option<F>,
}

#[wasm_bindgen(js_class = StaticDielectric)]
impl StaticDielectric {
    #[wasm_bindgen(constructor)]
    pub fn new(volume: F, temperature: F, epsilon_inf: Option<F>) -> Self {
        Self {
            volume,
            temperature,
            epsilon_inf,
        }
    }

    pub fn compute(&self, dipoles: &[F], n_frames: usize) -> Result<JsValue, JsValue> {
        let volume = self.volume;
        let temperature = self.temperature;
        let epsilon_inf = self.epsilon_inf;
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            epsilon: F,
            eps: Vec<F>,
            eps_mean: F,
            fluctuation: Vec<F>,
            dipole_mean: Vec<F>,
            dipole_sq_mean: Vec<F>,
            epsilon_inf: F,
            n_frames: usize,
        }
        let dipoles = array2(dipoles, n_frames, 3, "StaticDielectric dipoles")?;
        let eps_inf = epsilon_inf.unwrap_or(1.0);
        let epsilon =
            molrs::compute::static_dielectric_constant(&dipoles, volume, temperature, eps_inf)
                .map_err(|e| JsValue::from_str(&format!("StaticDielectric: {e}")))?;
        let c = molrs::compute::static_dielectric_constant_components(
            &dipoles,
            volume,
            temperature,
            eps_inf,
        )
        .map_err(|e| JsValue::from_str(&format!("StaticDielectric components: {e}")))?;
        js_value(&Out {
            epsilon,
            eps: c.eps.to_vec(),
            eps_mean: c.eps_mean,
            fluctuation: c.fluctuation.to_vec(),
            dipole_mean: c.dipole_mean.to_vec(),
            dipole_sq_mean: c.dipole_sq_mean.to_vec(),
            epsilon_inf: c.epsilon_inf,
            n_frames: c.n_frames,
        })
    }
}
