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

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::block::PyBlock;
use crate::helpers::molrs_error_to_pyerr;
use crate::simbox::PyBox;
use crate::store::{SharedStore, ffi_error_to_pyerr};
use molrs::frame::Frame as CoreFrame;
use molrs_ffi::FrameId;
use pyo3::prelude::*;
use pyo3::types::PyDict;

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
/// frame.simbox = Box.cube(10.0)
/// print(frame)          # Frame(blocks=['atoms'], simbox=yes)
/// print(frame.keys())   # ['atoms']
/// ```
#[pyclass(name = "Frame", from_py_object, unsendable)]
#[derive(Clone)]
pub struct PyFrame {
    pub(crate) id: FrameId,
    pub(crate) store: SharedStore,
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
        let store = Rc::new(RefCell::new(molrs_ffi::Store::new()));
        let id = store.borrow_mut().frame_new();
        Self { id, store }
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
    fn __getitem__(&self, key: &str) -> PyResult<PyBlock> {
        let handle = self
            .store
            .borrow()
            .get_block(self.id, key)
            .map_err(ffi_error_to_pyerr)?;
        Ok(PyBlock {
            handle,
            store: self.store.clone(),
        })
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
    fn __setitem__(&mut self, key: &str, block: &PyBlock) -> PyResult<()> {
        let block = block.clone_core_block()?;
        self.store
            .borrow_mut()
            .set_block(self.id, key, block)
            .map_err(ffi_error_to_pyerr)
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
        self.store
            .borrow_mut()
            .remove_block(self.id, key)
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
        self.with_frame(|f| f.contains_key(key))
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

    /// The simulation box attached to this frame, or ``None``.
    ///
    /// Returns
    /// -------
    /// Box | None
    ///     Periodic simulation box, if set.
    ///
    /// Examples
    /// --------
    /// >>> if frame.simbox is not None:
    /// ...     print(frame.simbox.volume())
    #[getter]
    fn simbox(&self) -> PyResult<Option<PyBox>> {
        self.store
            .borrow()
            .with_frame_simbox(self.id, |sb| sb.cloned().map(|inner| PyBox { inner }))
            .map_err(ffi_error_to_pyerr)
    }

    /// Set (or clear) the simulation box.
    ///
    /// Parameters
    /// ----------
    /// simbox : Box | None
    ///     Pass ``None`` to remove the simulation box.
    ///
    /// Examples
    /// --------
    /// >>> frame.simbox = Box.cube(20.0)
    /// >>> frame.simbox = None  # remove
    #[setter]
    fn set_simbox(&mut self, simbox: Option<&PyBox>) -> PyResult<()> {
        self.store
            .borrow_mut()
            .set_frame_simbox(self.id, simbox.map(|sb| sb.inner.clone()))
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
        self.store
            .borrow_mut()
            .with_frame_mut(self.id, |f| {
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
        let store = Rc::new(RefCell::new(molrs_ffi::Store::new()));
        let id = store.borrow_mut().frame_new();
        store
            .borrow_mut()
            .set_frame(id, frame)
            .map_err(ffi_error_to_pyerr)?;
        Ok(Self { id, store })
    }

    /// Clone the underlying `CoreFrame` out of the store (deep copy).
    pub(crate) fn clone_core_frame(&self) -> PyResult<CoreFrame> {
        self.store
            .borrow()
            .clone_frame(self.id)
            .map_err(ffi_error_to_pyerr)
    }

    /// Run a read-only closure on the underlying `CoreFrame`.
    fn with_frame<R>(&self, f: impl FnOnce(&CoreFrame) -> R) -> PyResult<R> {
        self.store
            .borrow()
            .with_frame(self.id, f)
            .map_err(ffi_error_to_pyerr)
    }
}
