//! Field observables for MolRec-style continuous physical quantities.
//!
//! This module intentionally models the physical field first and keeps the
//! numerical storage form explicit through [`FieldEncoding`]. In 0.1 we only
//! implement `uniform_grid`, but the enum shape leaves room for
//! `plane_wave`, `basis_expansion`, and `density_matrix` later.

use ndarray::Array1;
use serde::{Deserialize, Serialize};

use crate::block::Block;
use crate::error::MolRsError;
use crate::frame::Frame;
use crate::types::F;

/// Top-level field observable.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FieldObservable {
    /// Human-readable field name, e.g. `electron_density`.
    pub name: String,
    /// Physical quantity identifier, e.g. `electron_density`.
    pub quantity: String,
    /// Semantic scope. 0.1 uses `field`.
    pub scope: String,
    /// Domain where the field lives, e.g. `real_space`.
    pub domain: String,
    /// Physical unit string, e.g. `e/Angstrom^3`.
    pub unit: String,
    /// Numerical encoding of the field.
    pub encoding: FieldEncoding,
}

impl FieldObservable {
    /// Build a field observable around a uniform grid.
    pub fn uniform_grid(
        name: impl Into<String>,
        quantity: impl Into<String>,
        unit: impl Into<String>,
        grid: UniformGridField,
    ) -> Self {
        Self {
            name: name.into(),
            quantity: quantity.into(),
            scope: "field".to_string(),
            domain: "real_space".to_string(),
            unit: unit.into(),
            encoding: FieldEncoding::UniformGrid(grid),
        }
    }

    /// Validate the field against the constraints of its encoding.
    pub fn validate(&self) -> Result<(), MolRsError> {
        if self.scope != "field" {
            return Err(MolRsError::validation(format!(
                "field observable scope must be 'field', got '{}'",
                self.scope
            )));
        }
        match &self.encoding {
            FieldEncoding::UniformGrid(grid) => grid.validate(),
        }
    }

    /// Convert the field into a coarse point-cloud frame for visualization.
    ///
    /// This is intentionally a validation/debug view, not a replacement for
    /// proper isosurface extraction. The caller controls the sparsity using
    /// `stride` and a density threshold.
    pub fn to_point_cloud_frame(&self, threshold: F, stride: usize) -> Result<Frame, MolRsError> {
        match &self.encoding {
            FieldEncoding::UniformGrid(grid) => grid.to_point_cloud_frame(threshold, stride),
        }
    }
}

/// Numerical storage form of a field observable.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FieldEncoding {
    UniformGrid(UniformGridField),
}

/// Uniform real-space grid samples of a scalar field.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UniformGridField {
    /// Number of samples along each cell direction.
    pub shape: [usize; 3],
    /// Cartesian origin of the grid in Angstrom.
    pub origin: [F; 3],
    /// Cell vectors in Angstrom. Positions are
    /// `origin + (i/nx) a + (j/ny) b + (k/nz) c`.
    pub cell: [[F; 3]; 3],
    /// Periodicity flags for the sampled domain.
    pub pbc: [bool; 3],
    /// Scalar field samples stored in row-major `[ix, iy, iz]` order.
    pub values: Vec<F>,
}

impl UniformGridField {
    /// Total number of samples.
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// Returns true if the grid contains no values.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Validate grid dimensions and storage.
    pub fn validate(&self) -> Result<(), MolRsError> {
        if self.shape.contains(&0) {
            return Err(MolRsError::validation(format!(
                "uniform_grid shape must be strictly positive, got {:?}",
                self.shape
            )));
        }
        let expected = self.len();
        if self.values.len() != expected {
            return Err(MolRsError::validation(format!(
                "uniform_grid values length mismatch: expected {}, got {}",
                expected,
                self.values.len()
            )));
        }
        Ok(())
    }

    /// Return the sample position in Cartesian Angstrom.
    pub fn sample_position(&self, ix: usize, iy: usize, iz: usize) -> [F; 3] {
        let fx = ix as F / self.shape[0] as F;
        let fy = iy as F / self.shape[1] as F;
        let fz = iz as F / self.shape[2] as F;
        [
            self.origin[0] + fx * self.cell[0][0] + fy * self.cell[1][0] + fz * self.cell[2][0],
            self.origin[1] + fx * self.cell[0][1] + fy * self.cell[1][1] + fz * self.cell[2][1],
            self.origin[2] + fx * self.cell[0][2] + fy * self.cell[1][2] + fz * self.cell[2][2],
        ]
    }

    /// Flattened sample index for row-major `[ix, iy, iz]` order.
    pub fn index(&self, ix: usize, iy: usize, iz: usize) -> usize {
        (ix * self.shape[1] + iy) * self.shape[2] + iz
    }

    /// Convert to a coarse point-cloud frame for debugging/validation.
    pub fn to_point_cloud_frame(&self, threshold: F, stride: usize) -> Result<Frame, MolRsError> {
        self.validate()?;
        let stride = stride.max(1);

        let mut xs = Vec::new();
        let mut ys = Vec::new();
        let mut zs = Vec::new();
        let mut density = Vec::new();
        let mut element = Vec::new();

        for ix in (0..self.shape[0]).step_by(stride) {
            for iy in (0..self.shape[1]).step_by(stride) {
                for iz in (0..self.shape[2]).step_by(stride) {
                    let idx = self.index(ix, iy, iz);
                    let value = self.values[idx];
                    if value.abs() < threshold {
                        continue;
                    }
                    let pos = self.sample_position(ix, iy, iz);
                    xs.push(pos[0]);
                    ys.push(pos[1]);
                    zs.push(pos[2]);
                    density.push(value);
                    // Reuse an existing lightweight element so the current
                    // renderer can display the sample points without special
                    // shader work.
                    element.push(String::from("He"));
                }
            }
        }

        let mut atoms = Block::new();
        atoms.insert("x", Array1::from_vec(xs).into_dyn())?;
        atoms.insert("y", Array1::from_vec(ys).into_dyn())?;
        atoms.insert("z", Array1::from_vec(zs).into_dyn())?;
        atoms.insert("density", Array1::from_vec(density).into_dyn())?;
        atoms.insert("element", Array1::from_vec(element).into_dyn())?;

        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame
            .meta
            .insert("molrec_view".into(), "field_point_cloud".into());
        Ok(frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_grid_validates_shape() {
        let grid = UniformGridField {
            shape: [2, 2, 2],
            origin: [0.0, 0.0, 0.0],
            cell: [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
            pbc: [false, false, false],
            values: vec![0.0; 7],
        };
        assert!(grid.validate().is_err());
    }

    #[test]
    fn uniform_grid_to_point_cloud_filters_values() {
        let grid = UniformGridField {
            shape: [2, 2, 2],
            origin: [0.0, 0.0, 0.0],
            cell: [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
            pbc: [false, false, false],
            values: vec![0.0, 0.2, 0.0, 0.3, 0.0, 0.0, 0.4, 0.0],
        };

        let frame = grid.to_point_cloud_frame(0.1, 1).unwrap();
        let atoms = frame.get("atoms").unwrap();
        assert_eq!(atoms.nrows(), Some(3));
        assert!(atoms.get_float("density").is_some());
    }
}
