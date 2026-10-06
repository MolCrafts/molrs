//! Curve fitting and integration helpers — WASM face of the
//! `molrs::compute` fitting family.

use super::{array1, js_value};
use crate::core::types::JsFloatArray;
use molrs::compute::Fit;
use molrs::op::types::F;
use serde::Serialize;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = LinearFit)]
pub struct LinearFit {
    start_frac: F,
    end_frac: F,
}

#[wasm_bindgen(js_class = LinearFit)]
impl LinearFit {
    #[wasm_bindgen(constructor)]
    pub fn new(start_frac: F, end_frac: F) -> Self {
        Self {
            start_frac,
            end_frac,
        }
    }

    pub fn fit(&self, x: &[F], y: &[F]) -> Result<JsValue, JsValue> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            slope: F,
            intercept: F,
            r2: F,
            fit_start: usize,
            fit_end: usize,
        }
        let x = array1(x);
        let y = array1(y);
        let r = molrs::compute::LinearFit {
            window: (self.start_frac, self.end_frac),
        }
        .fit((&x, &y))
        .map_err(|e| JsValue::from_str(&format!("LinearFit: {e}")))?;
        js_value(&Out {
            slope: r.slope,
            intercept: r.intercept,
            r2: r.r2,
            fit_start: r.fit_start,
            fit_end: r.fit_end,
        })
    }
}

#[wasm_bindgen(js_name = CumulativeTrapezoid)]
pub struct CumulativeTrapezoid {
    dt: F,
    n_lags: Option<usize>,
}

#[wasm_bindgen(js_class = CumulativeTrapezoid)]
impl CumulativeTrapezoid {
    #[wasm_bindgen(constructor)]
    pub fn new(dt: F, n_lags: Option<usize>) -> Self {
        Self { dt, n_lags }
    }

    pub fn fit(&self, y: &[F]) -> Result<JsFloatArray, JsValue> {
        let dt = self.dt;
        let n_lags = self.n_lags;
        let y = array1(y);
        let r = molrs::compute::CumulativeTrapezoid
            .fit((&y, dt, n_lags))
            .map_err(|e| JsValue::from_str(&format!("CumulativeTrapezoid: {e}")))?;
        Ok(JsFloatArray::from(r.integral.as_slice().unwrap()))
    }
}

#[wasm_bindgen(js_name = Plateau)]
pub struct Plateau {
    start_frac: F,
    end_frac: F,
}

#[wasm_bindgen(js_class = Plateau)]
impl Plateau {
    #[wasm_bindgen(constructor)]
    pub fn new(start_frac: F, end_frac: F) -> Self {
        Self {
            start_frac,
            end_frac,
        }
    }

    pub fn fit(&self, y: &[F]) -> Result<JsValue, JsValue> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            value: F,
            n_samples: usize,
            std: F,
        }
        let y = array1(y);
        let r = molrs::compute::Plateau {
            window: (self.start_frac, self.end_frac),
        }
        .fit(&y)
        .map_err(|e| JsValue::from_str(&format!("Plateau: {e}")))?;
        js_value(&Out {
            value: r.value,
            n_samples: r.n_samples,
            std: r.std,
        })
    }
}
