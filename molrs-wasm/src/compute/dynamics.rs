//! Van Hove correlation and pair persistence — WASM face of the
//! `molrs::compute` dynamics family.

use super::{SeriesOut, array2, js_value};
use crate::core::frame::Frame;
use molrs::compute::Compute;
use molrs::op::types::F;
use ndarray::Array3;
use serde::Serialize;
use wasm_bindgen::prelude::*;

fn array3(
    data: &[F],
    dim0: usize,
    dim1: usize,
    dim2: usize,
    name: &str,
) -> Result<Array3<F>, JsValue> {
    if data.len() != dim0 * dim1 * dim2 {
        return Err(JsValue::from_str(&format!(
            "{name}: data length {} != dim0 * dim1 * dim2 = {} * {} * {}",
            data.len(),
            dim0,
            dim1,
            dim2
        )));
    }
    Array3::from_shape_vec((dim0, dim1, dim2), data.to_vec())
        .map_err(|e| JsValue::from_str(&format!("{name}: {e}")))
}

#[wasm_bindgen(js_name = VanHove)]
pub struct VanHove {
    frames: Vec<molrs::store::Frame>,
    n_r_bins: usize,
    r_max: F,
    lags: Vec<usize>,
    stride: usize,
}

#[wasm_bindgen(js_class = VanHove)]
impl VanHove {
    #[wasm_bindgen(constructor)]
    pub fn new(n_r_bins: usize, r_max: F, lags: Vec<usize>, stride: Option<usize>) -> Self {
        Self {
            frames: Vec::new(),
            n_r_bins,
            r_max,
            lags,
            stride: stride.unwrap_or(1),
        }
    }

    pub fn feed(&mut self, frame: &Frame) -> Result<(), JsValue> {
        frame.with_frame(|rs_frame| {
            self.frames.push(rs_frame.clone());
            Ok(())
        })
    }

    pub fn compute(&self) -> Result<JsValue, JsValue> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            r_edges: Vec<F>,
            r_centers: Vec<F>,
            lags: Vec<usize>,
            g_self: Vec<F>,
            g_distinct: Vec<F>,
            shape: [usize; 2],
            dr: F,
            has_distinct: bool,
        }
        let refs: Vec<&molrs::store::Frame> = self.frames.iter().collect();
        let calc = molrs::compute::VanHove::new(self.n_r_bins, self.r_max, self.lags.clone())
            .map_err(|e| JsValue::from_str(&format!("VanHove: {e}")))?
            .with_stride(self.stride);
        let r = calc
            .compute(&refs, ())
            .map_err(|e| JsValue::from_str(&format!("VanHove compute: {e}")))?;
        let shape = [r.g_self.nrows(), r.g_self.ncols()];
        js_value(&Out {
            r_edges: r.r_edges.to_vec(),
            r_centers: r.r_centers.to_vec(),
            lags: r.lags,
            g_self: r.g_self.iter().copied().collect(),
            g_distinct: r.g_distinct.iter().copied().collect(),
            shape,
            dr: r.dr,
            has_distinct: r.has_distinct,
        })
    }

    pub fn reset(&mut self) {
        self.frames.clear();
    }
}

#[wasm_bindgen(js_name = PairPersistence)]
pub struct PairPersistence {
    r0: F,
    r1: F,
    method: String,
    dt: F,
    max_lag: usize,
    exclude_self: bool,
}

#[wasm_bindgen(js_class = PairPersistence)]
impl PairPersistence {
    #[wasm_bindgen(constructor)]
    pub fn new(r0: F, r1: F, method: String, dt: F, max_lag: usize, exclude_self: bool) -> Self {
        Self {
            r0,
            r1,
            method,
            dt,
            max_lag,
            exclude_self,
        }
    }

    pub fn compute(
        &self,
        coords_i: &[F],
        n_frames: usize,
        n_i: usize,
        coords_j: &[F],
        n_j: usize,
        box_lengths: &[F],
    ) -> Result<JsValue, JsValue> {
        let r0 = self.r0;
        let r1 = self.r1;
        let method = self.method.clone();
        let dt = self.dt;
        let max_lag = self.max_lag;
        let exclude_self = self.exclude_self;
        let ci = array3(coords_i, n_frames, n_i, 3, "PairPersistence coords_i")?;
        let cj = array3(coords_j, n_frames, n_j, 3, "PairPersistence coords_j")?;
        let bl = array2(box_lengths, n_frames, 3, "PairPersistence box_lengths")?;
        let method = molrs::compute::SurvivalMethod::parse(&method)
            .map_err(|e| JsValue::from_str(&format!("PairPersistence method: {e}")))?;
        let r = molrs::compute::pair_survival_tcf(
            &ci,
            &cj,
            &bl,
            r0,
            r1,
            method,
            dt,
            max_lag,
            exclude_self,
        )
        .map_err(|e| JsValue::from_str(&format!("PairPersistence: {e}")))?;
        js_value(&SeriesOut {
            lag_times: r.lag_times.to_vec(),
            values: r.correlation.to_vec(),
        })
    }
}
