//! Uniform spatial grid storing multiple named scalar arrays.
//!
//! A [`Grid`] holds any number of named scalar fields (e.g. `electron_density`,
//! `spin_density`) that all share the same spatial grid: same dimensions,
//! same origin, and same cell vectors. This maps naturally onto VASP CHGCAR
//! (2–4 arrays on one grid) and Gaussian cube files (one array per file,
//! possibly combined by the caller).
//!
//! Grid positions are **not stored** — they are computed on demand from the
//! cell and dim fields.

use std::collections::HashMap;

use ndarray::{Array3, ArrayD};

use crate::error::MolRsError;
use crate::types::F;

/// A collection of named scalar arrays on a shared uniform spatial grid.
///
/// Cell vectors are stored as columns (same convention as [`SimBox`][crate::region::simbox::SimBox]).
/// The Cartesian position of voxel `(i, j, k)` is:
/// ```text
/// origin + (i/nx)*cell_col0 + (j/ny)*cell_col1 + (k/nz)*cell_col2
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Grid {
    /// Grid dimensions `[nx, ny, nz]`.
    pub dim: [usize; 3],
    /// Cartesian origin in Ångström.
    pub origin: [F; 3],
    /// Cell matrix: columns are the three lattice vectors (Å).
    pub cell: [[F; 3]; 3],
    /// Periodic boundary flags for each axis.
    pub pbc: [bool; 3],
    /// Named scalar arrays stored in row-major `(ix, iy, iz)` order.
    arrays: HashMap<String, Vec<F>>,
}

impl Grid {
    /// Create an empty grid with the given spatial definition.
    pub fn new(dim: [usize; 3], origin: [F; 3], cell: [[F; 3]; 3], pbc: [bool; 3]) -> Self {
        Self {
            dim,
            origin,
            cell,
            pbc,
            arrays: HashMap::new(),
        }
    }

    /// Total number of voxels: `nx * ny * nz`.
    pub fn total(&self) -> usize {
        self.dim[0] * self.dim[1] * self.dim[2]
    }

    /// Insert (or replace) a named array.
    ///
    /// `data` must have length `nx * ny * nz` in row-major `(ix, iy, iz)` order.
    pub fn insert(&mut self, name: impl Into<String>, data: Vec<F>) -> Result<(), MolRsError> {
        let expected = self.total();
        let name = name.into();
        if data.len() != expected {
            return Err(MolRsError::validation(format!(
                "grid array '{}' length mismatch: expected {}, got {}",
                name, expected, data.len()
            )));
        }
        self.arrays.insert(name, data);
        Ok(())
    }

    /// Return a named array reshaped to `(nx, ny, nz)`, or `None` if absent.
    pub fn get(&self, name: &str) -> Option<ArrayD<F>> {
        self.arrays.get(name).map(|data| {
            Array3::from_shape_vec(
                [self.dim[0], self.dim[1], self.dim[2]],
                data.clone(),
            )
            .expect("grid shape matches stored data")
            .into_dyn()
        })
    }

    /// Borrow the raw flat slice for a named array, or `None` if absent.
    pub fn get_raw(&self, name: &str) -> Option<&[F]> {
        self.arrays.get(name).map(|v| v.as_slice())
    }

    /// Whether a named array is present.
    pub fn contains(&self, name: &str) -> bool {
        self.arrays.contains_key(name)
    }

    /// Number of named arrays stored in this grid.
    pub fn len(&self) -> usize {
        self.arrays.len()
    }

    /// Returns `true` if no arrays are stored.
    pub fn is_empty(&self) -> bool {
        self.arrays.is_empty()
    }

    /// Iterate over `(name, flat_data)` pairs.
    pub fn raw_arrays(&self) -> impl Iterator<Item = (&str, &[F])> {
        self.arrays.iter().map(|(k, v)| (k.as_str(), v.as_slice()))
    }

    /// Iterate over array names.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.arrays.keys().map(|s| s.as_str())
    }

    /// Compute Cartesian position of voxel `(ix, iy, iz)`.
    pub fn voxel_position(&self, ix: usize, iy: usize, iz: usize) -> [F; 3] {
        let fx = ix as F / self.dim[0] as F;
        let fy = iy as F / self.dim[1] as F;
        let fz = iz as F / self.dim[2] as F;
        [
            self.origin[0]
                + fx * self.cell[0][0]
                + fy * self.cell[1][0]
                + fz * self.cell[2][0],
            self.origin[1]
                + fx * self.cell[0][1]
                + fy * self.cell[1][1]
                + fz * self.cell[2][1],
            self.origin[2]
                + fx * self.cell[0][2]
                + fy * self.cell[1][2]
                + fz * self.cell[2][2],
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_validates_length() {
        let mut g = Grid::new([2, 2, 2], [0.0; 3], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [false; 3]);
        assert!(g.insert("rho", vec![0.0; 7]).is_err());
        assert!(g.insert("rho", vec![0.0; 8]).is_ok());
    }

    #[test]
    fn get_returns_shaped_array() {
        let mut g = Grid::new([2, 3, 4], [0.0; 3], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [false; 3]);
        g.insert("rho", (0..24).map(|x| x as F).collect()).unwrap();
        let arr = g.get("rho").unwrap();
        assert_eq!(arr.shape(), &[2, 3, 4]);
    }
}
