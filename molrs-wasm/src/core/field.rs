//! WASM wrapper for MolRec uniform-grid fields.

use wasm_bindgen::prelude::*;

use crate::core::frame::Frame;
use crate::core::types::{JsFloatArray, WasmArray};

/// Uniform scalar field sampled on a regular real-space grid.
#[wasm_bindgen(js_name = UniformGridField)]
pub struct UniformGridField {
    inner: molrs::UniformGridField,
}

impl UniformGridField {
    pub(crate) fn from_rs(inner: molrs::UniformGridField) -> Self {
        Self { inner }
    }
}

#[wasm_bindgen(js_class = UniformGridField)]
impl UniformGridField {
    /// Grid shape `[nx, ny, nz]`.
    pub fn shape(&self) -> Box<[usize]> {
        Box::new(self.inner.shape)
    }

    /// Grid origin in Angstrom as a 1x3 array.
    pub fn origin(&self) -> WasmArray {
        WasmArray::from_vec(self.inner.origin.to_vec(), Box::new([3]))
    }

    /// Cell matrix in Angstrom as a 3x3 row-major array.
    pub fn cell(&self) -> WasmArray {
        let data: Vec<molrs::types::F> = self
            .inner
            .cell
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        WasmArray::from_vec(data, Box::new([3, 3]))
    }

    /// Periodicity flags.
    pub fn pbc(&self) -> Box<[u8]> {
        Box::new(self.inner.pbc.map(|v| if v { 1 } else { 0 }))
    }

    /// Zero-copy `Float64Array` view of the sample values, flat row-major
    /// order (shape `[nx, ny, nz]`, use [`UniformGridField::shape`]).
    ///
    /// **Warning**: the view is invalidated on any WASM memory growth.
    /// Copy in JS (`new Float64Array(view)`) if it needs to outlive
    /// subsequent allocations.
    pub fn values(&self) -> JsFloatArray {
        // SAFETY: view borrows WASM linear memory; JS must not retain it
        // past any allocation that could trigger memory growth.
        unsafe { JsFloatArray::view(&self.inner.values) }
    }

    /// Materialize a coarse point-cloud frame for validation rendering.
    #[wasm_bindgen(js_name = toPointCloudFrame)]
    pub fn to_point_cloud_frame(&self, threshold: f64, stride: usize) -> Result<Frame, JsValue> {
        let rs_frame = self
            .inner
            .to_point_cloud_frame(threshold as molrs::types::F, stride)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Frame::from_rs(rs_frame)
    }
}
