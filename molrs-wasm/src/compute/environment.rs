//! Local environments (bond order, descriptors, angular separation,
//! environment matching) — WASM face of the `molrs::compute` environment family.

use super::{Grid2Out, js_value, quats};
use crate::core::frame::Frame;
use crate::core::neighbors::Neighbors;
use crate::core::types::JsFloatArray;
use molrs::compute::Compute;
use molrs::op::F;
use serde::Serialize;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = BondOrientationalOrder)]
pub struct BondOrientationalOrder {
    inner: molrs::compute::BondOrientationalOrder,
}

#[wasm_bindgen(js_class = BondOrientationalOrder)]
impl BondOrientationalOrder {
    #[wasm_bindgen(constructor)]
    pub fn new(n_theta: usize, n_phi: usize) -> Result<Self, JsValue> {
        Ok(Self {
            inner: molrs::compute::BondOrientationalOrder::new(n_theta, n_phi)
                .map_err(|e| JsValue::from_str(&format!("BondOrientationalOrder: {e}")))?,
        })
    }

    pub fn compute(&self, frame: &Frame, neighbors: &Neighbors) -> Result<JsValue, JsValue> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            bond_order: Grid2Out,
            raw_counts: Vec<u64>,
            shape: [usize; 2],
            theta_edges: Vec<F>,
            phi_edges: Vec<F>,
        }
        frame.with_frame(|rs_frame| {
            let nlists = std::slice::from_ref(&neighbors.inner);
            let mut out = self
                .inner
                .compute(&[rs_frame], nlists)
                .map_err(|e| JsValue::from_str(&format!("BondOrientationalOrder compute: {e}")))?;
            let r = out
                .pop()
                .ok_or_else(|| JsValue::from_str("BondOrientationalOrder: empty result"))?;
            let shape = [r.bond_order.nrows(), r.bond_order.ncols()];
            js_value(&Out {
                bond_order: Grid2Out {
                    data: r.bond_order.iter().copied().collect(),
                    shape,
                },
                raw_counts: r.raw_counts.iter().copied().collect(),
                shape,
                theta_edges: r.theta_edges,
                phi_edges: r.phi_edges,
            })
        })
    }
}

#[wasm_bindgen(js_name = LocalDescriptors)]
pub struct LocalDescriptors {
    inner: molrs::compute::LocalDescriptors,
}

#[wasm_bindgen(js_class = LocalDescriptors)]
impl LocalDescriptors {
    #[wasm_bindgen(constructor)]
    pub fn new(l_max: u32) -> Self {
        Self {
            inner: molrs::compute::LocalDescriptors::new(l_max),
        }
    }

    pub fn compute(&self, frame: &Frame, neighbors: &Neighbors) -> Result<JsValue, JsValue> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            l_max: u32,
            n_sphs: usize,
            descriptors_re_im: Vec<F>,
        }
        frame.with_frame(|rs_frame| {
            let nlists = std::slice::from_ref(&neighbors.inner);
            let mut out = self
                .inner
                .compute(&[rs_frame], nlists)
                .map_err(|e| JsValue::from_str(&format!("LocalDescriptors compute: {e}")))?;
            let r = out
                .pop()
                .ok_or_else(|| JsValue::from_str("LocalDescriptors: empty result"))?;
            js_value(&Out {
                l_max: r.l_max,
                n_sphs: r.n_sphs,
                descriptors_re_im: r.descriptors.iter().flat_map(|c| [c.re, c.im]).collect(),
            })
        })
    }
}

#[wasm_bindgen(js_name = AngularSeparation)]
pub struct AngularSeparation {
    equivalent_orientations: bool,
}

#[wasm_bindgen(js_class = AngularSeparation)]
impl AngularSeparation {
    #[wasm_bindgen(constructor)]
    pub fn new(equivalent_orientations: Option<bool>) -> Self {
        Self {
            equivalent_orientations: equivalent_orientations.unwrap_or(true),
        }
    }

    #[wasm_bindgen(js_name = computeGlobal)]
    pub fn compute_global(&self, query: &[F], global: &[F]) -> Result<JsValue, JsValue> {
        let query = quats(query, "AngularSeparation query")?;
        let global = quats(global, "AngularSeparation global")?;
        let dummy = molrs::core::Frame::new();
        let calc = molrs::compute::AngularSeparationGlobal::new()
            .with_equivalent_orientations(self.equivalent_orientations);
        let args = molrs::compute::AngularSeparationGlobalArgs {
            query: &query,
            global: &global,
        };
        let mut out = calc
            .compute(&[&dummy], args)
            .map_err(|e| JsValue::from_str(&format!("AngularSeparationGlobal: {e}")))?;
        let r = out
            .pop()
            .ok_or_else(|| JsValue::from_str("AngularSeparationGlobal: empty result"))?;
        let shape = [r.angles.nrows(), r.angles.ncols()];
        js_value(&Grid2Out {
            data: r.angles.iter().copied().collect(),
            shape,
        })
    }

    #[wasm_bindgen(js_name = computeNeighbor)]
    pub fn compute_neighbor(
        &self,
        frame: &Frame,
        neighbors: &Neighbors,
        query: &[F],
        points: &[F],
    ) -> Result<JsFloatArray, JsValue> {
        let query = quats(query, "AngularSeparation query")?;
        let points = quats(points, "AngularSeparation points")?;
        frame.with_frame(|rs_frame| {
            let nlists = std::slice::from_ref(&neighbors.inner);
            let q = vec![query];
            let p = vec![points];
            let calc = molrs::compute::AngularSeparationNeighbor::new()
                .with_equivalent_orientations(self.equivalent_orientations);
            let args = molrs::compute::AngularSeparationNeighborArgs {
                nlists,
                query_orientations: &q,
                point_orientations: &p,
            };
            let mut out = calc
                .compute(&[rs_frame], args)
                .map_err(|e| JsValue::from_str(&format!("AngularSeparationNeighbor: {e}")))?;
            let r = out
                .pop()
                .ok_or_else(|| JsValue::from_str("AngularSeparationNeighbor: empty result"))?;
            Ok(JsFloatArray::from(r.angles.as_slice()))
        })
    }
}

#[wasm_bindgen(js_name = EnvironmentMatch)]
pub struct EnvironmentMatch {
    inner: molrs::compute::EnvironmentMatch,
}

#[wasm_bindgen(js_class = EnvironmentMatch)]
impl EnvironmentMatch {
    #[wasm_bindgen(constructor)]
    pub fn new(
        rmsd_threshold: F,
        registration: Option<bool>,
        max_neighbors_for_registration: Option<usize>,
    ) -> Result<Self, JsValue> {
        let mut inner = molrs::compute::EnvironmentMatch::new(rmsd_threshold)
            .map_err(|e| JsValue::from_str(&format!("EnvironmentMatch: {e}")))?;
        inner = inner.with_registration(registration.unwrap_or(false));
        if let Some(n) = max_neighbors_for_registration {
            inner = inner.with_max_neighbors_for_registration(n);
        }
        Ok(Self { inner })
    }

    pub fn compute(&self, frame: &Frame, neighbors: &Neighbors) -> Result<JsValue, JsValue> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            cluster_idx: Vec<u32>,
            n_clusters: usize,
            fingerprints: Vec<Vec<F>>,
        }
        frame.with_frame(|rs_frame| {
            let nlists = std::slice::from_ref(&neighbors.inner);
            let mut out = self
                .inner
                .compute(&[rs_frame], nlists)
                .map_err(|e| JsValue::from_str(&format!("EnvironmentMatch compute: {e}")))?;
            let r = out
                .pop()
                .ok_or_else(|| JsValue::from_str("EnvironmentMatch: empty result"))?;
            js_value(&Out {
                cluster_idx: r.cluster_idx,
                n_clusters: r.n_clusters,
                fingerprints: r.fingerprints,
            })
        })
    }
}
