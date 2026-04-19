//! Python wrapper for `Frame`, a hierarchical data container backed by the
//! shared FFI store.
//!
//! A [`PyFrame`] maps string keys (e.g. `"atoms"`, `"bonds"`, `"angles"`) to
//! [`PyBlock`] column stores. It may optionally carry a [`PyBox`] (simulation
//! box) and a string-to-string metadata dictionary.
//!
//! # Conventional Block Layout
//!
//! | Block key   | Expected columns                                      | Notes                                    |
//! |-------------|-------------------------------------------------------|------------------------------------------|
//! | `"atoms"`   | `symbol` (str), `x`/`y`/`z` (float), `mass` (float)  | Atom positions and properties             |
//! | `"bonds"`   | `i`/`j` (uint), `order` (float)                       | Bond topology (indices into atoms)        |
//! | `"angles"`  | `i`/`j`/`k` (uint), `type` (int)                      | Angle topology                            |
//!
//! The frame itself does **not** enforce cross-block row consistency; that is
//! the caller's responsibility (use [`PyFrame::validate`] to check).

use std::collections::HashMap;

use crate::block::PyBlock;
use crate::helpers::{NpF, molrs_error_to_pyerr};
use crate::simbox::PyBox;
use crate::store::ffi_error_to_pyerr;
use molrs::frame::Frame as CoreFrame;
use molrs::grid::Grid as CoreGrid;
use molrs::types::F;
use molrs_ffi::FrameRef;
use ndarray::Array4;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray4, PyArrayDyn, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn};
use pyo3::exceptions::{PyKeyError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Uniform spatial grid storing multiple named scalar arrays.
///
/// A `Grid` groups named scalar fields (e.g. `"electron_density"`,
/// `"spin_density"`) that share the same spatial definition: same `dim`,
/// `origin`, `cell`, and `pbc`. Grid positions are computed on demand via
/// `.grid` — they are not stored.
///
/// # Python Examples
///
/// ```python
/// import numpy as np
/// import molrs
///
/// grid = molrs.Grid(
///     dim=np.array([4, 4, 4], dtype=np.intp),
///     origin=np.zeros(3, dtype=np.float32),
///     cell=(np.eye(3) * 10.0).astype(np.float32),
///     pbc=np.array([True, True, True]),
/// )
/// grid['electron_density'] = np.ones((4, 4, 4), dtype=np.float32)
/// ```
#[pyclass(name = "Grid", unsendable)]
#[derive(Clone)]
pub struct PyGrid {
    pub(crate) inner: CoreGrid,
}

#[pymethods]
impl PyGrid {
    /// Create a new empty grid.
    #[new]
    fn new(
        dim: [usize; 3],
        origin: PyReadonlyArray1<'_, NpF>,
        cell: PyReadonlyArray2<'_, NpF>,
        pbc: [bool; 3],
    ) -> PyResult<Self> {
        let origin_slice = origin.as_slice().map_err(|e| {
            PyTypeError::new_err(format!("origin must be a contiguous 1-D array: {}", e))
        })?;
        if origin_slice.len() != 3 {
            return Err(PyTypeError::new_err("origin must have length 3"));
        }
        let o = [origin_slice[0] as F, origin_slice[1] as F, origin_slice[2] as F];

        let cell_arr = cell.as_array();
        if cell_arr.dim() != (3, 3) {
            return Err(PyTypeError::new_err("cell must have shape (3, 3)"));
        }
        let c = [
            [cell_arr[[0, 0]] as F, cell_arr[[0, 1]] as F, cell_arr[[0, 2]] as F],
            [cell_arr[[1, 0]] as F, cell_arr[[1, 1]] as F, cell_arr[[1, 2]] as F],
            [cell_arr[[2, 0]] as F, cell_arr[[2, 1]] as F, cell_arr[[2, 2]] as F],
        ];

        Ok(Self { inner: CoreGrid::new(dim, o, c, pbc) })
    }

    /// Grid dimensions `[nx, ny, nz]`.
    #[getter]
    fn dim<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<usize>> {
        let v = vec![self.inner.dim[0], self.inner.dim[1], self.inner.dim[2]];
        v.into_pyarray(py)
    }

    /// Cartesian origin `[x, y, z]` in Ångström.
    #[getter]
    fn origin<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<NpF>> {
        let v: Vec<NpF> = self.inner.origin.iter().map(|&v| v as NpF).collect();
        v.into_pyarray(py)
    }

    /// Cell matrix of shape `(3, 3)` — columns are lattice vectors.
    #[getter]
    fn cell<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<NpF>> {
        let flat: Vec<NpF> = self.inner.cell.iter().flat_map(|row| row.iter().map(|&v| v as NpF)).collect();
        ndarray::Array2::from_shape_vec((3, 3), flat)
            .unwrap()
            .into_pyarray(py)
    }

    /// Periodic boundary flags `[px, py, pz]`.
    #[getter]
    fn pbc<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        let v = vec![self.inner.pbc[0], self.inner.pbc[1], self.inner.pbc[2]];
        v.into_pyarray(py)
    }

    /// Cartesian position of every voxel. Shape `(nx, ny, nz, 3)`.
    ///
    /// Computed on demand: `position[i, j, k] = origin + (i/nx)*col0 + (j/ny)*col1 + (k/nz)*col2`.
    #[getter]
    fn grid<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray4<NpF>> {
        let [nx, ny, nz] = self.inner.dim;
        let mut data = Vec::with_capacity(nx * ny * nz * 3);
        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    let pos = self.inner.voxel_position(ix, iy, iz);
                    data.push(pos[0] as NpF);
                    data.push(pos[1] as NpF);
                    data.push(pos[2] as NpF);
                }
            }
        }
        Array4::from_shape_vec((nx, ny, nz, 3), data)
            .unwrap()
            .into_pyarray(py)
    }

    /// Retrieve a named scalar array as a shaped `(nx, ny, nz)` numpy array.
    fn __getitem__<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyArrayDyn<NpF>>> {
        let arr = self.inner.get(name).ok_or_else(|| PyKeyError::new_err(name.to_string()))?;
        Ok(arr.mapv(|v| v as NpF).into_pyarray(py))
    }

    /// Store a named scalar array. Shape must be `(nx, ny, nz)`.
    fn __setitem__(&mut self, name: &str, data: PyReadonlyArrayDyn<'_, NpF>) -> PyResult<()> {
        let arr = data.as_array();
        let [nx, ny, nz] = self.inner.dim;
        if arr.shape() != [nx, ny, nz] {
            return Err(PyTypeError::new_err(format!(
                "array shape {:?} does not match grid dim [{}, {}, {}]",
                arr.shape(), nx, ny, nz
            )));
        }
        let flat: Vec<F> = arr.iter().map(|&v| v as F).collect();
        self.inner.insert(name, flat).map_err(molrs_error_to_pyerr)
    }

    /// Test whether a named array is present.
    fn __contains__(&self, name: &str) -> bool {
        self.inner.contains(name)
    }

    /// Number of named arrays stored in this grid.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Names of stored arrays.
    fn keys(&self) -> Vec<String> {
        self.inner.keys().map(|s| s.to_string()).collect()
    }

    fn __repr__(&self) -> String {
        let [nx, ny, nz] = self.inner.dim;
        let keys: Vec<&str> = self.inner.keys().collect();
        format!("Grid(dim=[{}, {}, {}], arrays={:?})", nx, ny, nz, keys)
    }
}

/// Hierarchical data container exposed to Python as `molrs.Frame`.
///
/// A `Frame` is a dictionary of named [`Block`](crate::block::PyBlock)s with
/// optional simulation box and metadata. It is the primary exchange format for
/// molecular data across the molrs ecosystem.
///
/// # Python Examples
///
/// ```python
/// import numpy as np
/// from molrs import Frame, Block, Box
///
/// frame = Frame()
/// atoms = Block()
/// atoms.insert("symbol", ["O", "H", "H"])
/// atoms.insert("x", np.array([0.0, 0.76, -0.76], dtype=np.float32))
/// atoms.insert("y", np.array([0.0, 0.59,  0.59], dtype=np.float32))
/// atoms.insert("z", np.zeros(3, dtype=np.float32))
/// frame["atoms"] = atoms
///
/// frame.box = Box.cube(10.0)
/// print(frame)          # Frame(blocks=['atoms'], simbox=yes)
/// print(frame.keys())   # ['atoms']
/// ```
#[pyclass(name = "Frame", from_py_object, unsendable)]
#[derive(Clone)]
pub struct PyFrame {
    pub(crate) inner: FrameRef,
}

#[pymethods]
impl PyFrame {
    /// Create an empty frame with no blocks, no simulation box, and empty
    /// metadata.
    ///
    /// Returns
    /// -------
    /// Frame
    #[new]
    fn new() -> Self {
        Self {
            inner: FrameRef::new_standalone(),
        }
    }

    /// Retrieve a block by name.
    ///
    /// Parameters
    /// ----------
    /// key : str
    ///     Block name (e.g. ``"atoms"``).
    ///
    /// Returns
    /// -------
    /// Block
    ///
    /// Raises
    /// ------
    /// KeyError
    ///     If ``key`` does not exist.
    ///
    /// Examples
    /// --------
    /// >>> atoms = frame["atoms"]
    fn __getitem__<'py>(&self, py: Python<'py>, key: &str) -> PyResult<Py<PyAny>> {
        // Try block first
        if let Ok(inner) = self.inner.block(key) {
            return Ok(Py::new(py, PyBlock { inner })?.into_any());
        }
        // Try grid
        if let Some(grid) = self.with_frame(|f| f.get_grid(key).cloned())? {
            return Ok(Py::new(py, PyGrid { inner: grid })?.into_any());
        }
        Err(PyKeyError::new_err(key.to_string()))
    }

    /// Assign a block under the given name.
    ///
    /// If a block with the same key already exists it is replaced.
    ///
    /// Parameters
    /// ----------
    /// key : str
    ///     Block name.
    /// block : Block
    ///     The block to store.
    ///
    /// Examples
    /// --------
    /// >>> frame["atoms"] = atoms_block
    fn __setitem__(&mut self, key: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        if let Ok(grid) = value.extract::<PyRef<'_, PyGrid>>() {
            return self
                .inner
                .with_mut(|f| {
                    f.insert_grid(key, grid.inner.clone());
                })
                .map_err(ffi_error_to_pyerr);
        }
        if let Ok(block) = value.extract::<PyRef<'_, PyBlock>>() {
            let core_block = block.clone_core_block()?;
            return self
                .inner
                .store
                .borrow_mut()
                .set_block(self.inner.id, key, core_block)
                .map_err(ffi_error_to_pyerr);
        }
        Err(PyTypeError::new_err("value must be a Block or Grid"))
    }

    /// Delete a block by name.
    ///
    /// Parameters
    /// ----------
    /// key : str
    ///     Block name to remove.
    ///
    /// Raises
    /// ------
    /// KeyError
    ///     If ``key`` does not exist.
    ///
    /// Examples
    /// --------
    /// >>> del frame["bonds"]
    fn __delitem__(&mut self, key: &str) -> PyResult<()> {
        self.inner
            .store
            .borrow_mut()
            .remove_block(self.inner.id, key)
            .map_err(ffi_error_to_pyerr)
    }

    /// Test whether a block name is present.
    ///
    /// Parameters
    /// ----------
    /// key : str
    ///     Block name.
    ///
    /// Returns
    /// -------
    /// bool
    ///
    /// Examples
    /// --------
    /// >>> "atoms" in frame
    /// True
    fn __contains__(&self, key: &str) -> PyResult<bool> {
        self.with_frame(|f| f.contains_key(key) || f.has_grid(key))
    }

    /// Number of blocks stored in this frame.
    ///
    /// Returns
    /// -------
    /// int
    fn __len__(&self) -> PyResult<usize> {
        self.with_frame(|f| f.len())
    }

    /// List all block names.
    ///
    /// Returns
    /// -------
    /// list[str]
    fn keys(&self) -> PyResult<Vec<String>> {
        self.with_frame(|f| f.keys().map(|s| s.to_string()).collect())
    }

    /// Return all grids as a dict mapping name → Grid.
    ///
    /// Returns
    /// -------
    /// dict[str, Grid]
    fn grids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        let grids = self.with_frame(|f| {
            f.grids()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect::<Vec<_>>()
        })?;
        for (k, v) in grids {
            dict.set_item(k, Py::new(py, PyGrid { inner: v })?)?;
        }
        Ok(dict)
    }

    /// The simulation box attached to this frame, or ``None``.
    ///
    /// Returns
    /// -------
    /// Box | None
    ///     Periodic simulation box, if set.
    ///
    /// Examples
    /// --------
    /// >>> if frame.box is not None:
    /// ...     print(frame.box.volume())
    #[getter(simbox)]
    fn get_box(&self) -> PyResult<Option<PyBox>> {
        Ok(self
            .inner
            .simbox_clone()
            .map_err(ffi_error_to_pyerr)?
            .map(|inner| PyBox { inner }))
    }

    /// Set (or clear) the simulation box.
    ///
    /// Parameters
    /// ----------
    /// box : Box | None
    ///     Pass ``None`` to remove the simulation box.
    ///
    /// Examples
    /// --------
    /// >>> frame.box = Box.cube(20.0)
    /// >>> frame.box = None  # remove
    #[setter(simbox)]
    fn set_box(&mut self, simbox: Option<&PyBox>) -> PyResult<()> {
        self.inner
            .set_simbox(simbox.map(|sb| sb.inner.clone()))
            .map_err(ffi_error_to_pyerr)
    }

    /// Metadata dictionary (``dict[str, str]``).
    ///
    /// Returns
    /// -------
    /// dict[str, str]
    ///     String key-value metadata attached to this frame.
    #[getter]
    fn meta<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        let meta = self.with_frame(|f| f.meta.clone())?;
        for (k, v) in meta {
            dict.set_item(k, v)?;
        }
        Ok(dict)
    }

    /// Replace the metadata dictionary.
    ///
    /// Parameters
    /// ----------
    /// meta : dict[str, str]
    ///     New metadata. All keys and values must be strings.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If any key or value cannot be converted to ``str``.
    #[setter]
    fn set_meta(&mut self, meta: &Bound<'_, PyDict>) -> PyResult<()> {
        let mut map = HashMap::new();
        for (k, v) in meta.iter() {
            let key: String = k.extract()?;
            let val: String = v.extract()?;
            map.insert(key, val);
        }
        self.inner
            .with_mut(|f| {
                f.meta = map;
            })
            .map_err(ffi_error_to_pyerr)?;
        Ok(())
    }

    /// Validate cross-block consistency.
    ///
    /// Checks that referenced indices (e.g. bond atom indices) are within
    /// bounds and that required columns exist.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If validation fails.
    fn validate(&self) -> PyResult<()> {
        self.with_frame(|f| f.validate().map_err(molrs_error_to_pyerr))?
    }

    fn __repr__(&self) -> PyResult<String> {
        self.with_frame(|f| {
            let keys: Vec<&str> = f.keys().collect();
            format!(
                "Frame(blocks={:?}, simbox={})",
                keys,
                if f.simbox.is_some() { "yes" } else { "no" }
            )
        })
    }
}

impl PyFrame {
    /// Create a `PyFrame` from a Rust `CoreFrame`, allocating a new FFI store.
    pub(crate) fn from_core_frame(frame: CoreFrame) -> PyResult<Self> {
        let store = molrs_ffi::new_shared();
        let id = store.borrow_mut().frame_new();
        store
            .borrow_mut()
            .set_frame(id, frame)
            .map_err(ffi_error_to_pyerr)?;
        Ok(Self {
            inner: FrameRef::new(store, id),
        })
    }

    /// Clone the underlying `CoreFrame` out of the store (deep copy).
    pub(crate) fn clone_core_frame(&self) -> PyResult<CoreFrame> {
        self.inner.clone_frame().map_err(ffi_error_to_pyerr)
    }

    /// Run a read-only closure on the underlying `CoreFrame`.
    fn with_frame<R>(&self, f: impl FnOnce(&CoreFrame) -> R) -> PyResult<R> {
        self.inner.with(f).map_err(ffi_error_to_pyerr)
    }
}
