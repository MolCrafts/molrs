//! Mean squared displacement — WASM face of `molrs::compute::MSD`.

use crate::core::frame::Frame;
use crate::core::types::JsFloatArray;
use molrs::compute::Compute;
use molrs::compute::{MSD as RsMSD, MSDResult as RsMSDResult};
use molrs::op::types::F;
use wasm_bindgen::prelude::*;

/// Mean squared displacement (MSD) analysis.
///
/// Computes MSD = |r(t) - r(0)|^2 for each particle and the system
/// average. The first frame fed is automatically used as the reference.
/// Useful for measuring diffusion coefficients via D = MSD / (6t).
///
/// All distances are in angstrom (A), so MSD is in A^2.
///
/// # Example (JavaScript)
///
/// ```js
/// const msd = new MSD();
/// for (const frame of trajectory) {
///   msd.feed(frame);         // first frame = reference
/// }
/// const results = msd.results();  // MSDResult[] per frame
/// console.log(results[10].mean);  // MSD at frame 10 in A^2
/// ```
///
/// # References
///
/// - Einstein, A. (1905). *Annalen der Physik*, 322(8), 549-560.
#[wasm_bindgen(js_name = MSD)]
pub struct MSD {
    frames: Vec<molrs::store::Frame>,
}

#[allow(clippy::new_without_default)]
#[wasm_bindgen(js_class = MSD)]
impl MSD {
    /// Create an empty MSD analysis.
    ///
    /// The first frame passed to [`feed`] becomes the reference
    /// configuration (t=0).
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const msd = new MSD();
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self { frames: Vec::new() }
    }

    /// Feed a frame into the MSD analysis.
    ///
    /// Internally clones the frame's core data so subsequent mutations on
    /// the JS side (e.g. trajectory playback overwriting buffers) do not
    /// race against pending [`results`](Self::results) calls. The first
    /// frame sets the reference configuration.
    ///
    /// # Arguments
    ///
    /// * `frame` - Frame with `"atoms"` block containing
    ///   `x`, `y`, `z` (F) columns
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const msd = new MSD();
    /// msd.feed(frame0);  // sets reference
    /// msd.feed(frame1);  // added to trajectory
    /// const series = msd.results();
    /// ```
    pub fn feed(&mut self, frame: &Frame) -> Result<(), JsValue> {
        frame.with_frame(|rs_frame| {
            self.frames.push(rs_frame.clone());
            Ok(())
        })
    }

    /// Run the stateless [`molrs::compute::MSD`] over every fed frame and
    /// return the per-frame time series.
    ///
    /// The first frame is always the reference, so `results()[0].mean ≈ 0`.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const results = msd.results();
    /// results.forEach((r, t) => console.log(`t=${t}: MSD=${r.mean}`));
    /// ```
    pub fn results(&self) -> Result<Vec<MSDResult>, JsValue> {
        if self.frames.is_empty() {
            return Ok(Vec::new());
        }
        let refs: Vec<&molrs::store::Frame> = self.frames.iter().collect();
        let series = RsMSD::new()
            .compute(&refs, ())
            .map_err(|e| JsValue::from_str(&format!("MSD results: {e}")))?;
        Ok(series
            .data
            .iter()
            .map(|r| MSDResult { inner: r.clone() })
            .collect())
    }

    /// Number of frames accumulated.
    #[wasm_bindgen(getter)]
    pub fn count(&self) -> usize {
        self.frames.len()
    }

    /// Reset the analysis, clearing the trajectory buffer.
    pub fn reset(&mut self) {
        self.frames.clear();
    }
}

/// Result of a mean squared displacement computation.
///
/// # Example (JavaScript)
///
/// ```js
/// const result = msd.compute(frame);
/// console.log(result.mean);              // number (A^2)
/// console.log(result.perParticle());     // Float64Array (A^2)
/// ```
#[wasm_bindgen(js_name = MSDResult)]
pub struct MSDResult {
    inner: RsMSDResult,
}

#[wasm_bindgen(js_class = MSDResult)]
impl MSDResult {
    /// System-average mean squared displacement in A^2.
    ///
    /// This is the arithmetic mean of all per-particle squared
    /// displacements: `mean = sum(|r_i(t) - r_i(0)|^2) / N`.
    #[wasm_bindgen(getter)]
    pub fn mean(&self) -> F {
        self.inner.mean
    }

    /// Zero-copy `Float64Array` view of per-particle squared displacements
    /// in A². `perParticle()[i]` is `|r_i(t) - r_i(0)|²` for particle `i`.
    /// **Invalidated** on WASM memory growth; copy in JS if needed.
    #[wasm_bindgen(js_name = perParticle)]
    pub fn per_particle(&self) -> JsFloatArray {
        // SAFETY: view borrows wasm memory; short-lived use only.
        unsafe { JsFloatArray::view(self.inner.per_particle.as_slice().unwrap()) }
    }
}
