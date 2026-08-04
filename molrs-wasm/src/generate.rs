//! WASM bindings for structure generators.

use molrs::CarbonTubeBuilder as RsCarbonTubeBuilder;
use wasm_bindgen::prelude::*;

use crate::core::Frame;

/// Exact single-wall carbon nanotube builder.
#[wasm_bindgen]
pub struct CarbonTubeBuilder {
    inner: RsCarbonTubeBuilder,
}

#[wasm_bindgen]
impl CarbonTubeBuilder {
    /// Start a builder for chiral indices `(n, m)`.
    #[wasm_bindgen(constructor)]
    pub fn new(n: u32, m: u32) -> Result<CarbonTubeBuilder, JsValue> {
        Ok(Self {
            inner: RsCarbonTubeBuilder::new(n, m).map_err(js_error)?,
        })
    }

    /// Use a fixed number of axial translational cells.
    #[wasm_bindgen(js_name = setCells)]
    pub fn set_cells(&mut self, cells: u32) -> Result<(), JsValue> {
        self.inner = self
            .inner
            .clone()
            .with_cells(cells as usize)
            .map_err(js_error)?;
        Ok(())
    }

    /// Round an axial length up to complete translational cells.
    #[wasm_bindgen(js_name = setLength)]
    pub fn set_length(&mut self, length: f64) -> Result<(), JsValue> {
        self.inner = self.inner.clone().with_length(length).map_err(js_error)?;
        Ok(())
    }

    /// Set the carbon-carbon bond length in angstrom.
    #[wasm_bindgen(js_name = setBondLength)]
    pub fn set_bond_length(&mut self, bond_length: f64) -> Result<(), JsValue> {
        self.inner = self
            .inner
            .clone()
            .with_bond_length(bond_length)
            .map_err(js_error)?;
        Ok(())
    }

    /// Enable or disable periodic closure along the tube axis.
    #[wasm_bindgen(js_name = setPeriodic)]
    pub fn set_periodic(&mut self, periodic: bool) {
        self.inner = self.inner.clone().with_periodic(periodic);
    }

    /// Set transverse vacuum padding in angstrom.
    #[wasm_bindgen(js_name = setVacuum)]
    pub fn set_vacuum(&mut self, vacuum: f64) -> Result<(), JsValue> {
        self.inner = self.inner.clone().with_vacuum(vacuum).map_err(js_error)?;
        Ok(())
    }

    /// Build a fresh frame containing atoms, bonds, and the simulation box.
    pub fn build(&self) -> Result<Frame, JsValue> {
        Frame::from_rs(self.inner.build().map_err(js_error)?)
    }

    /// Validate the exact lattice and periodic topology.
    pub fn validate(&self) -> Result<(), JsValue> {
        self.inner.validate().map_err(js_error)
    }
}

fn js_error(error: molrs::CarbonTubeError) -> JsValue {
    JsValue::from_str(&error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn builds_frame_for_javascript() {
        let mut builder = CarbonTubeBuilder::new(6, 0).unwrap();
        builder.set_cells(2).unwrap();
        let frame = builder.build().unwrap();
        assert!(frame.get_block("atoms").is_some());
        assert!(frame.get_block("bonds").is_some());
    }
}
