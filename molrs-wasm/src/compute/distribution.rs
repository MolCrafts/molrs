//! Distance / angle / dihedral and combined distributions — WASM face of
//! the `molrs::compute` distribution family.

use super::js_value;
use crate::core::frame::Frame;
use molrs::compute::Compute;
use molrs::op::types::F;
use serde::Serialize;
use wasm_bindgen::prelude::*;

fn distribution_compute<O: molrs::compute::Observable + Sync>(
    frame: &Frame,
    calc: molrs::compute::DistributionFunction<O>,
    groups: molrs::compute::AtomGroups,
) -> Result<JsValue, JsValue> {
    #[derive(Serialize)]
    #[serde(rename_all = "camelCase")]
    struct Out {
        bin_centers: Vec<F>,
        bin_edges: Vec<F>,
        counts: Vec<F>,
        density: Vec<F>,
        density_sin_corrected: Option<Vec<F>>,
        bin_width: F,
        n_binned: F,
        n_raw_samples: usize,
        n_frames: usize,
        angular: bool,
    }
    frame.with_frame(|rs_frame| {
        let r = calc
            .compute(&[rs_frame], &groups)
            .map_err(|e| JsValue::from_str(&format!("Distribution compute: {e}")))?;
        js_value(&Out {
            bin_centers: r.bin_centers.to_vec(),
            bin_edges: r.bin_edges.to_vec(),
            counts: r.counts.to_vec(),
            density: r.density.to_vec(),
            density_sin_corrected: r.density_sin_corrected.map(|v| v.to_vec()),
            bin_width: r.bin_width,
            n_binned: r.n_binned,
            n_raw_samples: r.n_raw_samples,
            n_frames: r.n_frames,
            angular: r.angular,
        })
    })
}

#[wasm_bindgen(js_name = WasmDistanceDistribution)]
pub struct WasmDistanceDistribution {
    n_bins: usize,
    min: F,
    max: F,
}

#[wasm_bindgen(js_class = WasmDistanceDistribution)]
impl WasmDistanceDistribution {
    #[wasm_bindgen(constructor)]
    pub fn new(n_bins: usize, min: F, max: F) -> Self {
        Self { n_bins, min, max }
    }

    pub fn compute(&self, frame: &Frame, pairs: &[u32]) -> Result<JsValue, JsValue> {
        let groups = molrs::compute::AtomGroups::new(2, pairs.iter().map(|&v| v as u64).collect())
            .map_err(|e| JsValue::from_str(&format!("DistanceDistribution groups: {e}")))?;
        let calc = molrs::compute::DistributionFunction::new(
            molrs::compute::DistanceObservable,
            self.n_bins,
            self.min,
            self.max,
        )
        .map_err(|e| JsValue::from_str(&format!("DistanceDistribution: {e}")))?;
        distribution_compute(frame, calc, groups)
    }
}

#[wasm_bindgen(js_name = WasmAngleDistribution)]
pub struct WasmAngleDistribution {
    n_bins: usize,
}

#[wasm_bindgen(js_class = WasmAngleDistribution)]
impl WasmAngleDistribution {
    #[wasm_bindgen(constructor)]
    pub fn new(n_bins: usize) -> Self {
        Self { n_bins }
    }

    pub fn compute(&self, frame: &Frame, triples: &[u32]) -> Result<JsValue, JsValue> {
        let groups =
            molrs::compute::AtomGroups::new(3, triples.iter().map(|&v| v as u64).collect())
                .map_err(|e| JsValue::from_str(&format!("AngleDistribution groups: {e}")))?;
        let calc = molrs::compute::DistributionFunction::over_natural_range(
            molrs::compute::AngleObservable,
            self.n_bins,
        )
        .map_err(|e| JsValue::from_str(&format!("AngleDistribution: {e}")))?;
        distribution_compute(frame, calc, groups)
    }
}

#[wasm_bindgen(js_name = WasmDihedralDistribution)]
pub struct WasmDihedralDistribution {
    n_bins: usize,
}

#[wasm_bindgen(js_class = WasmDihedralDistribution)]
impl WasmDihedralDistribution {
    #[wasm_bindgen(constructor)]
    pub fn new(n_bins: usize) -> Self {
        Self { n_bins }
    }

    pub fn compute(&self, frame: &Frame, quads: &[u32]) -> Result<JsValue, JsValue> {
        let groups = molrs::compute::AtomGroups::new(4, quads.iter().map(|&v| v as u64).collect())
            .map_err(|e| JsValue::from_str(&format!("DihedralDistribution groups: {e}")))?;
        let calc = molrs::compute::DistributionFunction::over_natural_range(
            molrs::compute::DihedralObservable,
            self.n_bins,
        )
        .map_err(|e| JsValue::from_str(&format!("DihedralDistribution: {e}")))?;
        distribution_compute(frame, calc, groups)
    }
}

#[wasm_bindgen(js_name = WasmCombinedDistribution)]
pub struct WasmCombinedDistribution {
    kinds: Vec<String>,
    axes: Vec<molrs::compute::AxisSpec>,
}

#[wasm_bindgen(js_class = WasmCombinedDistribution)]
impl WasmCombinedDistribution {
    /// `kinds[i]` is `"distance" | "angle" | "dihedral"`; axis `i` bins
    /// observable `i` into `bins[i]` bins over `[mins[i], maxs[i]]`. A non-zero
    /// `sinWeight[i]` marks axis `i` angular so its marginal carries the
    /// sin θ solid-angle correction.
    #[wasm_bindgen(constructor)]
    pub fn new(
        kinds: Vec<String>,
        bins: &[u32],
        mins: &[F],
        maxs: &[F],
        sin_weight: Option<Vec<u8>>,
    ) -> Result<Self, JsValue> {
        let n = kinds.len();
        if bins.len() != n || mins.len() != n || maxs.len() != n {
            return Err(JsValue::from_str(
                "CombinedDistribution: kinds, bins, mins and maxs must have equal length",
            ));
        }
        let sin = sin_weight.unwrap_or_default();
        let axes = (0..n)
            .map(|i| {
                let spec = molrs::compute::AxisSpec::new(bins[i] as usize, mins[i], maxs[i])
                    .map_err(|e| {
                        JsValue::from_str(&format!("CombinedDistribution axis {i}: {e}"))
                    })?;
                Ok(spec.with_sin_weight(sin.get(i).is_some_and(|&v| v != 0)))
            })
            .collect::<Result<Vec<_>, JsValue>>()?;
        Ok(Self { kinds, axes })
    }

    /// `groups` is `number[][]`: one flat atom-index array per observable, each
    /// of length `arity × nGroups` (arity 2/3/4 for distance/angle/dihedral).
    pub fn compute(&self, frame: &Frame, groups: JsValue) -> Result<JsValue, JsValue> {
        use molrs::compute::{AnyObservable, AtomGroups};

        let raw: Vec<Vec<u32>> = serde_wasm_bindgen::from_value(groups)
            .map_err(|e| JsValue::from_str(&format!("CombinedDistribution groups: {e}")))?;
        if raw.len() != self.kinds.len() {
            return Err(JsValue::from_str(
                "CombinedDistribution: one atom-index group array per observable is required",
            ));
        }

        let mut observables = Vec::with_capacity(self.kinds.len());
        let mut atom_groups = Vec::with_capacity(self.kinds.len());
        for (i, kind) in self.kinds.iter().enumerate() {
            let (obs, arity) = AnyObservable::from_kind(kind).map_err(|e| {
                JsValue::from_str(&format!("CombinedDistribution observable {i}: {e}"))
            })?;
            observables.push(obs);
            atom_groups.push(
                AtomGroups::new(arity, raw[i].iter().map(|&v| v as u64).collect()).map_err(
                    |e| JsValue::from_str(&format!("CombinedDistribution groups {i}: {e}")),
                )?,
            );
        }
        let calc = molrs::compute::CombinedDistribution::new(observables, self.axes.clone())
            .map_err(|e| JsValue::from_str(&format!("CombinedDistribution: {e}")))?;

        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            shape: Vec<usize>,
            edges: Vec<Vec<F>>,
            centers: Vec<Vec<F>>,
            counts: Vec<F>,
            density: Vec<F>,
            n_binned: F,
            n_raw_samples: usize,
            n_frames: usize,
        }
        frame.with_frame(|rs_frame| {
            let r = calc
                .compute(&[rs_frame], &atom_groups)
                .map_err(|e| JsValue::from_str(&format!("CombinedDistribution compute: {e}")))?;
            js_value(&Out {
                shape: r.centers.iter().map(|c| c.len()).collect(),
                edges: r.edges.iter().map(|e| e.to_vec()).collect(),
                centers: r.centers.iter().map(|c| c.to_vec()).collect(),
                counts: r.counts.to_vec(),
                density: r.density.to_vec(),
                n_binned: r.binned,
                n_raw_samples: r.n_raw_samples,
                n_frames: r.n_frames,
            })
        })
    }
}
