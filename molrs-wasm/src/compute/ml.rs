//! Descriptor-space analysis (PCA, k-means) — WASM face of the
//! `molrs::compute` ml family.

use js_sys::Float64Array;
use js_sys::Int32Array;
use molrs::compute::Compute;
use molrs::compute::ComputeResult;
use molrs::compute::DescriptorRow;
use molrs::compute::{Kmeans as RsKmeans, Pca as RsPca, PcaResult as RsPcaResult};
use molrs::op::F;
use wasm_bindgen::prelude::*;

/// Stateless wrapper for [`molrs::compute::Pca`].
///
/// All configuration lives on [`fitTransform`](Self::fit_transform).
///
/// # Example (JavaScript)
///
/// ```js
/// const pca = new Pca();
/// const result = pca.fitTransform(matrix, nRows, nCols);
/// const coords   = result.coords();    // Float64Array, length 2 * nRows
/// const variance = result.variance();  // Float64Array, length 2
/// ```
#[wasm_bindgen]
pub struct Pca;

#[allow(clippy::new_without_default)]
#[wasm_bindgen(js_class = Pca)]
impl Pca {
    /// Create a new PCA calculator. The struct carries no state — all
    /// parameters are supplied on [`fitTransform`](Self::fit_transform).
    #[wasm_bindgen(constructor)]
    pub fn new() -> Pca {
        Pca
    }

    /// Fit 2-component PCA on a row-major observation matrix and return the
    /// projected coordinates + per-component variance.
    ///
    /// # Arguments
    ///
    /// * `matrix` — row-major `n_rows × n_cols` observation matrix.
    /// * `n_rows` — number of observations.
    /// * `n_cols` — number of features.
    ///
    /// # Errors
    ///
    /// Throws if `n_rows < 3`, `n_cols < 2`, the length does not match
    /// `n_rows * n_cols`, any element is non-finite, or any column has
    /// zero variance.
    #[wasm_bindgen(js_name = fitTransform)]
    pub fn fit_transform(
        &self,
        matrix: &[F],
        n_rows: usize,
        n_cols: usize,
    ) -> Result<PcaResult, JsValue> {
        if matrix.len() != n_rows * n_cols {
            return Err(JsValue::from_str(&format!(
                "PCA: matrix length {} != n_rows * n_cols = {} * {}",
                matrix.len(),
                n_rows,
                n_cols
            )));
        }
        let rows: Vec<PcaRow> = (0..n_rows)
            .map(|i| PcaRow(matrix[i * n_cols..(i + 1) * n_cols].to_vec()))
            .collect();
        let dummy = molrs::core::Frame::new();
        RsPca::<PcaRow>::new()
            .compute(&[&dummy], &rows)
            .map(|inner| PcaResult { inner })
            .map_err(|e| JsValue::from_str(&format!("PCA: {e}")))
    }
}

/// Row adapter so the stateless `Pca` can consume caller matrices without
/// requiring a downstream molrs-compute type.
#[derive(Clone)]
struct PcaRow(Vec<F>);

impl DescriptorRow for PcaRow {
    fn as_row(&self) -> &[F] {
        &self.0
    }
}

impl ComputeResult for PcaRow {}

/// Result of a [`Pca::fit_transform`] call.
///
/// Each accessor returns an **owned** `Float64Array` (copy of the underlying
/// `Vec`) so JS is free to let this wrapper be GC'd without dangling views.
#[wasm_bindgen]
pub struct PcaResult {
    inner: RsPcaResult,
}

#[wasm_bindgen(js_class = PcaResult)]
impl PcaResult {
    /// Projected 2D coordinates as a row-major `Float64Array` of length
    /// `2 * n_rows`. `coords[2 * i + 0]` is the PC1 score for row `i`,
    /// `coords[2 * i + 1]` is PC2.
    pub fn coords(&self) -> Float64Array {
        let out = Float64Array::new_with_length(self.inner.coords.len() as u32);
        out.copy_from(&self.inner.coords);
        out
    }

    /// Explained variance per component as `Float64Array` of length 2.
    /// `variance[0] >= variance[1]` by construction.
    pub fn variance(&self) -> Float64Array {
        let out = Float64Array::new_with_length(2);
        out.copy_from(&self.inner.variance);
        out
    }
}

// ===========================================================================
// k-means — with k-means++ init
// ===========================================================================

/// Wrapper for [`molrs::compute::Kmeans`].
///
/// # Example (JavaScript)
///
/// ```js
/// const km = new Kmeans(3, 100, 42);
/// const labels = km.fit(coords, nRows, 2);   // Int32Array
/// ```
#[wasm_bindgen]
pub struct Kmeans {
    inner: RsKmeans,
}

#[wasm_bindgen(js_class = Kmeans)]
impl Kmeans {
    /// Create a new k-means configuration.
    ///
    /// # Arguments
    ///
    /// * `k` — number of clusters (>= 1).
    /// * `max_iter` — maximum Lloyd iterations (>= 1).
    /// * `seed` — RNG seed for k-means++ initialization. Cast to `u64`
    ///   internally (JS numbers are `f64`; integers up to 2^53 pass
    ///   through losslessly).
    ///
    /// # Errors
    ///
    /// Throws if `k == 0` or `max_iter == 0`.
    #[wasm_bindgen(constructor)]
    pub fn new(k: usize, max_iter: usize, seed: f64) -> Result<Kmeans, JsValue> {
        let seed_u64 = seed as u64;
        RsKmeans::new(k, max_iter, seed_u64)
            .map(|inner| Kmeans { inner })
            .map_err(|e| JsValue::from_str(&format!("Kmeans: {e}")))
    }

    /// Cluster a row-major `n_rows × n_dims` coordinate matrix.
    ///
    /// # Returns
    ///
    /// Cluster labels in `0..k` as an owned `Int32Array`, one per row.
    ///
    /// # Errors
    ///
    /// Throws if `k > n_rows`, `n_dims == 0`, the length does not match
    /// `n_rows * n_dims`, or any element is non-finite.
    pub fn fit(&self, coords: &[F], n_rows: usize, n_dims: usize) -> Result<Int32Array, JsValue> {
        if n_dims != 2 {
            return Err(JsValue::from_str(
                "Kmeans wasm binding supports n_dims=2 only (PCA-score input)",
            ));
        }
        if coords.len() != n_rows * n_dims {
            return Err(JsValue::from_str(&format!(
                "Kmeans: coords length {} != n_rows * n_dims = {} * {}",
                coords.len(),
                n_rows,
                n_dims
            )));
        }
        let pca = molrs::compute::PcaResult {
            coords: coords.to_vec(),
            variance: [0.0 as F, 0.0 as F],
        };
        let dummy = molrs::core::Frame::new();
        let labels = self
            .inner
            .compute(&[&dummy], &pca)
            .map_err(|e| JsValue::from_str(&format!("Kmeans fit: {e}")))?;
        let out = Int32Array::new_with_length(labels.0.len() as u32);
        out.copy_from(&labels.0);
        Ok(out)
    }
}
