//! WASM wrapper for the molrs [`Grid`] type.
//!
//! A [`Grid`] holds one or more named scalar arrays that all share the same
//! uniform spatial grid (same dimensions, origin, and cell vectors). This is
//! the primary structure for volumetric data such as electron densities,
//! spin densities, or electrostatic potentials.

use js_sys::Array as JsArray;
use wasm_bindgen::prelude::*;

use crate::core::types::WasmArray;

/// A uniform spatial grid storing multiple named scalar arrays.
///
/// All arrays in a `Grid` share the same spatial definition: dimensions
/// (`[nx, ny, nz]`), Cartesian origin, cell matrix (columns are lattice
/// vectors, matching VASP/molrs convention), and periodic boundary flags.
///
/// # Example (JavaScript)
///
/// ```js
/// // Create a 10×10×10 cubic grid
/// const origin = new Float32Array([0, 0, 0]);
/// const cell = new Float32Array([
///   10, 0, 0,   // first column (a vector)
///    0,10, 0,   // second column (b vector)
///    0, 0,10,   // third column (c vector)
/// ]);
/// const grid = new Grid(10, 10, 10, origin, cell, true, true, true);
///
/// // Insert a density array (must have length = 10*10*10 = 1000)
/// const rho = new Float32Array(1000).fill(1.0);
/// grid.insertArray("rho", rho);
///
/// // Retrieve it
/// const arr = grid.getArray("rho");
/// console.log(arr.toCopy());
/// ```
#[wasm_bindgen(js_name = Grid)]
pub struct Grid {
    inner: molrs::Grid,
}

impl Grid {
    pub(crate) fn from_rs(inner: molrs::Grid) -> Self {
        Self { inner }
    }

    pub(crate) fn into_rs(self) -> molrs::Grid {
        self.inner
    }
}

#[wasm_bindgen(js_class = Grid)]
impl Grid {
    /// Create a new empty grid with the given spatial definition.
    ///
    /// # Arguments
    ///
    /// * `dim_x`, `dim_y`, `dim_z` — Number of grid points along each axis.
    /// * `origin` — Float32Array of length 3: Cartesian origin in Ångström.
    /// * `cell` — Float32Array of length 9: cell matrix in column-major order.
    ///   `cell[0..3]` is the first lattice vector (a), `cell[3..6]` is b,
    ///   `cell[6..9]` is c (matching VASP/molrs convention where columns are
    ///   lattice vectors).
    /// * `pbc_x`, `pbc_y`, `pbc_z` — Periodic boundary flags for each axis.
    ///
    /// # Errors
    ///
    /// Throws if `origin` does not have length 3, or `cell` does not have
    /// length 9.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const origin = new Float32Array([0, 0, 0]);
    /// const cell = new Float32Array([10,0,0, 0,10,0, 0,0,10]);
    /// const grid = new Grid(10, 10, 10, origin, cell, true, true, true);
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new(
        dim_x: usize,
        dim_y: usize,
        dim_z: usize,
        origin: &[f32],
        cell: &[f32],
        pbc_x: bool,
        pbc_y: bool,
        pbc_z: bool,
    ) -> Result<Grid, JsValue> {
        if origin.len() != 3 {
            return Err(JsValue::from_str(&format!(
                "Grid: origin must have length 3, got {}",
                origin.len()
            )));
        }
        if cell.len() != 9 {
            return Err(JsValue::from_str(&format!(
                "Grid: cell must have length 9, got {}",
                cell.len()
            )));
        }

        let origin_arr = [origin[0], origin[1], origin[2]];
        // cell is column-major: cell[0..3] = col0, cell[3..6] = col1, cell[6..9] = col2
        let cell_arr = [
            [cell[0], cell[1], cell[2]],
            [cell[3], cell[4], cell[5]],
            [cell[6], cell[7], cell[8]],
        ];
        let pbc_arr = [pbc_x, pbc_y, pbc_z];

        Ok(Grid {
            inner: molrs::Grid::new([dim_x, dim_y, dim_z], origin_arr, cell_arr, pbc_arr),
        })
    }

    /// Grid dimensions `[nx, ny, nz]`.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// console.log(grid.dim()); // [10, 10, 10]
    /// ```
    pub fn dim(&self) -> Box<[usize]> {
        Box::new(self.inner.dim)
    }

    /// Cartesian origin in Ångström as a 1-D array of length 3.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const o = grid.origin();
    /// const arr = o.toCopy(); // Float32Array [ox, oy, oz]
    /// ```
    pub fn origin(&self) -> WasmArray {
        WasmArray::from_vec(self.inner.origin.to_vec(), Box::new([3]))
    }

    /// Cell matrix in Ångström as a flat array of length 9 in column-major
    /// order (columns are lattice vectors, matching VASP/molrs convention).
    ///
    /// Layout: `[col0_x, col0_y, col0_z, col1_x, col1_y, col1_z, col2_x, col2_y, col2_z]`
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const c = grid.cell();
    /// const flat = c.toCopy(); // Float32Array of length 9
    /// ```
    pub fn cell(&self) -> WasmArray {
        // Stored as [[F;3];3] where rows are columns of the cell matrix.
        // Emit in the same column-major order as the constructor accepts.
        let data: Vec<molrs::types::F> = self
            .inner
            .cell
            .iter()
            .flat_map(|col| col.iter().copied())
            .collect();
        WasmArray::from_vec(data, Box::new([3, 3]))
    }

    /// Periodic boundary flags as a `Uint8Array`-compatible slice.
    ///
    /// Each element is `1` (periodic) or `0` (not periodic).
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// console.log(grid.pbc()); // [1, 1, 1]
    /// ```
    pub fn pbc(&self) -> Box<[u8]> {
        Box::new(self.inner.pbc.map(|v| if v { 1 } else { 0 }))
    }

    /// Total number of voxels: `nx * ny * nz`.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// console.log(grid.total()); // 1000 for a 10×10×10 grid
    /// ```
    pub fn total(&self) -> usize {
        self.inner.total()
    }

    /// Names of all scalar arrays stored in this grid.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const names = grid.arrayNames(); // e.g. ["rho", "spin"]
    /// ```
    #[wasm_bindgen(js_name = arrayNames)]
    pub fn array_names(&self) -> JsArray {
        let names = JsArray::new();
        for name in self.inner.keys() {
            names.push(&JsValue::from_str(name));
        }
        names
    }

    /// Returns `true` if a named array is present in this grid.
    ///
    /// # Arguments
    ///
    /// * `name` — Array name to look up.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// grid.hasArray("rho"); // true or false
    /// ```
    #[wasm_bindgen(js_name = hasArray)]
    pub fn has_array(&self, name: &str) -> bool {
        self.inner.contains(name)
    }

    /// Number of named arrays stored in this grid.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// console.log(grid.len()); // e.g. 2
    /// ```
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if no arrays are stored.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// console.log(grid.isEmpty()); // true for a freshly created grid
    /// ```
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Retrieve a named scalar array as a flat `WasmArray` with shape `[nx, ny, nz]`.
    ///
    /// Returns `undefined` if the named array does not exist.
    ///
    /// # Arguments
    ///
    /// * `name` — Array name to retrieve.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const arr = grid.getArray("rho");
    /// if (arr) {
    ///   const flat = arr.toCopy(); // Float32Array of length nx*ny*nz
    /// }
    /// ```
    #[wasm_bindgen(js_name = getArray)]
    pub fn get_array(&self, name: &str) -> Option<WasmArray> {
        let raw = self.inner.get_raw(name)?;
        let [nx, ny, nz] = self.inner.dim;
        Some(WasmArray::from_vec(raw.to_vec(), Box::new([nx, ny, nz])))
    }

    /// Insert (or replace) a named scalar array.
    ///
    /// The provided `data` must have exactly `nx * ny * nz` elements in
    /// row-major `(ix, iy, iz)` order.
    ///
    /// # Arguments
    ///
    /// * `name` — Array name.
    /// * `data` — Float32Array with length equal to `grid.total()`.
    ///
    /// # Errors
    ///
    /// Throws if `data.length != nx * ny * nz`.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const rho = new Float32Array(grid.total()).fill(0.5);
    /// grid.insertArray("rho", rho);
    /// ```
    #[wasm_bindgen(js_name = insertArray)]
    pub fn insert_array(&mut self, name: &str, data: &[f32]) -> Result<(), JsValue> {
        self.inner
            .insert(name, data.to_vec())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}
