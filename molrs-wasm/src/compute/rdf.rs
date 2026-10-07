//! Radial distribution function g(r) — WASM face of `molrs::compute::Rdf`.

use crate::core::frame::Frame;
use crate::core::frame::positions_from_frame;
use crate::core::types::JsFloatArray;
use molrs::compute::{Rdf as RsRdf, RdfResult as RsRdfResult};
use molrs::op::F;
use wasm_bindgen::prelude::*;

/// Radial distribution function g(r) analysis.
///
/// Bins neighbor-pair distances in `[rMin, rMax]` and normalizes by the
/// ideal-gas pair density. Defaults follow freud (`rMin = 0`). Periodic
/// systems take their normalization volume from `frame.simbox`; non-periodic
/// systems must supply it as the constructor's `volume`.
///
/// # Algorithm
///
/// g(r) = n(r) / (rho * V_shell(r) * N_ref)
///
/// where `n(r)` is the pair count in bin `r`, `rho = N/V` is the number
/// density, and `V_shell(r)` is the shell volume for that bin.
///
/// # Example (JavaScript)
///
/// ```js
/// const rdf = new RDF(100, 5.0);          // rMin defaults to 0
/// const result = rdf.compute(frame);      // streams its own neighbor search
///
/// // Non-periodic frame: supply the normalization volume.
/// const free = new RDF(100, 5.0, undefined, volumeA3).compute(frame);
///
/// const r  = result.binCenters();
/// const gr = result.rdf();
/// ```
#[wasm_bindgen(js_name = Rdf)]
pub struct Rdf {
    inner: RsRdf,
    /// Explicit normalization volume (A^3). When unset, `compute` takes the
    /// volume from `frame.simbox`.
    volume: Option<F>,
}

#[wasm_bindgen(js_class = Rdf)]
impl Rdf {
    /// Create a new RDF analysis.
    ///
    /// # Arguments
    ///
    /// * `n_bins` - Number of histogram bins
    /// * `r_max` - Upper radial cutoff in angstrom (A). Should be ≤ the
    ///   neighbor-search cutoff.
    /// * `r_min` - Lower radial cutoff in angstrom (A). Optional, defaults
    ///   to 0 (freud convention). Pairs with `d < rMin` or `d == 0` are
    ///   excluded from the histogram.
    /// * `volume` - Explicit normalization volume in A^3. Optional; when unset,
    ///   [`compute`](Self::compute) reads the volume from `frame.simbox`.
    ///   Required for a frame without a box.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const rdf  = new RDF(100, 5.0);                  // rMin = 0, box volume
    /// const rdf2 = new RDF(100, 5.0, 0.5);             // exclude d < 0.5 A
    /// const rdf3 = new RDF(100, 5.0, null, 1000.0);    // non-periodic frame
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new(
        n_bins: usize,
        r_max: F,
        r_min: Option<F>,
        volume: Option<F>,
    ) -> Result<Rdf, JsValue> {
        if let Some(v) = volume
            && !(v.is_finite() && v > 0.0)
        {
            return Err(JsValue::from_str("Rdf: volume must be finite and > 0"));
        }
        let inner = RsRdf::new(n_bins, r_max, r_min.unwrap_or(0.0))
            .map_err(|e| JsValue::from_str(&format!("Rdf: {e}")))?;
        Ok(Self { inner, volume })
    }

    /// Compute g(r) for one frame (self-query).
    ///
    /// **Single semantic path** for RDF: builds a cell-list **index** at
    /// `r_max` and streams pairs into the histogram (`build_index` +
    /// `visit_pairs`). A full [`Neighbors`] is never allocated.
    /// Memory is \(O(N + n_{\mathrm{bins}})\), not \(O(P)\).
    ///
    /// Volume comes from the constructor `volume` if set, else `frame.simbox`.
    /// Non-periodic frames must pass `volume` to the constructor.
    ///
    /// For A↔B cross-RDF use [`computeCross`](Self::compute_cross) (same
    /// streaming engine; wasm-bindgen cannot express `Option<&Frame>`).
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const rdf = new RDF(100, 5.0);       // r_max is required
    /// const result = rdf.compute(frame);
    /// ```
    pub fn compute(&self, frame: &Frame) -> Result<RdfResult, JsValue> {
        frame.with_frame(|rs_frame| {
            // If constructor set an explicit volume and the frame has no box,
            // synthesize a cubic box so compute_frame can proceed.
            if rs_frame.simbox.is_none() {
                let v = self.volume.ok_or_else(|| {
                    JsValue::from_str(
                        "Rdf compute: frame has no box — pass volume to the RDF constructor",
                    )
                })?;
                return self.compute_with_synth_box(rs_frame, v);
            }
            let mut result = self
                .inner
                .compute_frame(rs_frame)
                .map_err(|e| JsValue::from_str(&format!("Rdf compute: {e}")))?;
            if let Some(v) = self.volume {
                apply_volume_override(&mut result, v);
            }
            Ok(RdfResult { inner: result })
        })
    }

    /// Cross-query g(r) between two frames (same streaming engine as
    /// [`compute`](Self::compute)).
    ///
    /// * `ref_frame` — reference set (B, builds the cell index)
    /// * `query_frame` — query set (A)
    #[wasm_bindgen(js_name = computeCross)]
    pub fn compute_cross(
        &self,
        ref_frame: &Frame,
        query_frame: &Frame,
    ) -> Result<RdfResult, JsValue> {
        ref_frame.with_frame(|rs_ref| {
            let ref_pos = positions_from_frame(rs_ref)?;
            let owned_box;
            let bx = match rs_ref.simbox.as_ref() {
                Some(sb) => sb,
                None => {
                    let v = self.volume.ok_or_else(|| {
                        JsValue::from_str(
                            "Rdf computeCross: frame has no box — pass volume to the RDF constructor",
                        )
                    })?;
                    let box_len = v.cbrt();
                    owned_box = molrs::core::SimBox::cube(
                        box_len,
                        ndarray::array![0.0 as F, 0.0 as F, 0.0 as F],
                        [false, false, false],
                    )
                    .map_err(|e| JsValue::from_str(&format!("Rdf computeCross: {e:?}")))?;
                    &owned_box
                }
            };
            query_frame.with_frame(|rs_query| {
                let query_pos = positions_from_frame(rs_query)?;
                let mut result = self
                    .inner
                    .compute_cross(ref_pos.view(), query_pos.view(), bx)
                    .map_err(|e| JsValue::from_str(&format!("Rdf compute: {e}")))?;
                if let Some(v) = self.volume {
                    apply_volume_override(&mut result, v);
                }
                Ok(RdfResult { inner: result })
            })
        })
    }

    fn compute_with_synth_box(
        &self,
        rs_frame: &molrs::core::Frame,
        volume: F,
    ) -> Result<RdfResult, JsValue> {
        // Temporarily attach a cubic box for the streaming path, then restore.
        // Frame is behind shared store — we can't mutate easily. Interleave
        // positions and call compute_self with an owned box instead.
        let pos = positions_from_frame(rs_frame)?;
        let box_len = volume.cbrt();
        let bx = molrs::core::SimBox::cube(
            box_len,
            ndarray::array![0.0 as F, 0.0 as F, 0.0 as F],
            [false, false, false],
        )
        .map_err(|e| JsValue::from_str(&format!("Rdf compute: {e:?}")))?;
        let result = self
            .inner
            .compute_self(pos.view(), &bx)
            .map_err(|e| JsValue::from_str(&format!("Rdf compute: {e}")))?;
        Ok(RdfResult { inner: result })
    }
}

fn apply_volume_override(result: &mut molrs::compute::RdfResult, volume: F) {
    use molrs::compute::ComputeResult;
    result.volume = volume;
    result.finalized = false;
    result.finalize();
}

/// Result of a radial distribution function computation.
///
/// Contains the binned g(r) values, bin geometry, raw pair counts,
/// and normalization metadata.
///
/// # Example (JavaScript)
///
/// ```js
/// const result = rdf.compute(frame, nlist);
/// const r  = result.binCenters();  // Float64Array [0.025, 0.075, ...]
/// const gr = result.rdf();         // Float64Array, normalized g(r)
/// const nr = result.pairCounts();  // Float64Array, raw counts
/// console.log("Volume:", result.volume, "A^3");
/// console.log("N_ref:", result.numPoints);
/// ```
#[wasm_bindgen(js_name = RdfResult)]
pub struct RdfResult {
    inner: RsRdfResult,
}

#[wasm_bindgen(js_class = RdfResult)]
impl RdfResult {
    /// Zero-copy `Float64Array` view of bin center positions in A.
    /// Length equals `n_bins`. **Invalidated** on WASM memory growth;
    /// copy in JS if it needs to outlive later calls.
    #[wasm_bindgen(js_name = binCenters)]
    pub fn bin_centers(&self) -> JsFloatArray {
        // SAFETY: view borrows wasm memory; short-lived use only.
        unsafe { JsFloatArray::view(self.inner.bin_centers.as_slice().unwrap()) }
    }

    /// Zero-copy `Float64Array` view of bin edge positions in A.
    /// Length is `n_bins + 1`. Same invalidation caveat.
    #[wasm_bindgen(js_name = binEdges)]
    pub fn bin_edges(&self) -> JsFloatArray {
        // SAFETY: view borrows wasm memory; short-lived use only.
        unsafe { JsFloatArray::view(self.inner.bin_edges.as_slice().unwrap()) }
    }

    /// Zero-copy `Float64Array` view of normalized g(r). Same invalidation
    /// caveat.
    pub fn rdf(&self) -> JsFloatArray {
        // SAFETY: view borrows wasm memory; short-lived use only.
        unsafe { JsFloatArray::view(self.inner.rdf.as_slice().unwrap()) }
    }

    /// Zero-copy `Float64Array` view of raw (un-normalized) pair counts
    /// per bin. Same invalidation caveat.
    #[wasm_bindgen(js_name = pairCounts)]
    pub fn pair_counts(&self) -> JsFloatArray {
        // SAFETY: view borrows wasm memory; short-lived use only.
        unsafe { JsFloatArray::view(self.inner.n_r.as_slice().unwrap()) }
    }

    /// Number of reference points used in the normalization.
    #[wasm_bindgen(getter, js_name = nPoints)]
    pub fn n_points(&self) -> usize {
        self.inner.n_points
    }

    /// Normalization volume in A^3 (from the SimBox or the explicit caller value).
    #[wasm_bindgen(getter)]
    pub fn volume(&self) -> F {
        self.inner.volume
    }

    /// Inner cutoff in A (lower edge of bin 0).
    #[wasm_bindgen(getter, js_name = rMin)]
    pub fn r_min(&self) -> F {
        self.inner.r_min
    }
}
