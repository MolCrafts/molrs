//! Python wrapper for `Block`, a heterogeneous column store backed by the
//! shared FFI store.
//!
//! A [`PyBlock`] holds typed columns keyed by name. Each column is a
//! contiguous ndarray that maps directly to a numpy array on the Python side.
//!
//! # Supported Column Types
//!
//! | Rust type        | numpy dtype (default) | Typical usage                |
//! |------------------|-----------------------|------------------------------|
//! | `F`  (f32/f64)   | `float32` / `float64` | positions, masses, charges   |
//! | `I`  (i32/i64)   | `int32`   / `int64`   | atom type IDs                |
//! | `U`  (u32/u64)   | `uint32`  / `uint64`  | bond indices                 |
//! | `bool`           | `bool`                | selection masks               |
//! | `String`         | `list[str]`           | element symbols               |

use molrs::block::{Block as CoreBlock, BlockDtype, Column};
use molrs::types::{F, I, U};
use molrs_ffi::BlockRef;
use ndarray::Array1;
use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::store::ffi_error_to_pyerr;

/// Internal owner that prevents the backing ndarray from being freed while a
/// numpy view is alive.
///
/// # Safety
///
/// The `borrow_from_array` call in each `*_array_view` helper below creates a
/// numpy view whose lifetime is tied to this owner object. The `unsendable`
/// marker ensures the owner (and therefore the view) never crosses threads.
#[pyclass(unsendable)]
struct FloatArrayOwner {
    array: ndarray::ArrayD<F>,
}

/// See [`FloatArrayOwner`].
#[pyclass(unsendable)]
struct IntArrayOwner {
    array: ndarray::ArrayD<I>,
}

/// See [`FloatArrayOwner`].
#[pyclass(unsendable)]
struct BoolArrayOwner {
    array: ndarray::ArrayD<bool>,
}

/// See [`FloatArrayOwner`].
#[pyclass(unsendable)]
struct UIntArrayOwner {
    array: ndarray::ArrayD<U>,
}

/// Heterogeneous column store exposed to Python as `molrs.Block`.
///
/// Each column is a named, typed array. All columns share the same number of
/// rows (axis-0 length). The underlying storage lives in an FFI `Store` and is
/// accessed through a version-tracked [`BlockHandle`].
///
/// # Python Examples
///
/// ```python
/// import numpy as np
/// from molrs import Block
///
/// b = Block()
/// b.insert("x", np.array([1.0, 2.0, 3.0], dtype=np.float32))
/// b.insert("symbol", ["C", "H", "H"])
/// assert b.nrows == 3
/// assert "x" in b
/// arr = b.view("x")        # zero-copy numpy view
/// ```
#[pyclass(name = "Block", from_py_object, unsendable)]
#[derive(Clone)]
pub struct PyBlock {
    pub(crate) inner: BlockRef,
}

#[pymethods]
impl PyBlock {
    /// Create an empty standalone `Block`.
    ///
    /// The block starts with zero columns and zero rows.
    ///
    /// Returns
    /// -------
    /// Block
    ///     A new empty block.
    ///
    /// Examples
    /// --------
    /// >>> b = Block()
    /// >>> len(b)
    /// 0
    #[new]
    fn new() -> PyResult<Self> {
        Self::from_core_block(CoreBlock::new())
    }

    /// Insert a numpy array (or list of strings) as a named column.
    ///
    /// If a column with the same key already exists it is replaced. The array
    /// length must match the row count of existing columns, or the block must
    /// be empty.
    ///
    /// Parameters
    /// ----------
    /// key : str
    ///     Column name (e.g. ``"x"``, ``"symbol"``).
    /// array : numpy.ndarray | list[str]
    ///     Column data. Accepted dtypes: ``float32``, ``float64``, ``int32``,
    ///     ``int64``, ``uint32``, ``uint64``, ``bool``, or a Python
    ///     ``list[str]``.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If the array dtype is not supported.
    /// ValueError
    ///     If the row count does not match existing columns.
    ///
    /// Examples
    /// --------
    /// >>> b = Block()
    /// >>> b.insert("x", np.zeros(10, dtype=np.float32))
    /// >>> b.insert("y", np.ones(10, dtype=np.float32))
    fn insert(&mut self, key: &str, array: &Bound<'_, pyo3::types::PyAny>) -> PyResult<()> {
        if let Ok(arr) = array.extract::<PyReadonlyArrayDyn<'_, f32>>() {
            let converted = arr.as_array().mapv(|x| x as F);
            return self.insert_array(key, converted);
        }
        if let Ok(arr) = array.extract::<PyReadonlyArrayDyn<'_, f64>>() {
            let converted = arr.as_array().mapv(|x| x as F);
            return self.insert_array(key, converted);
        }
        if let Ok(arr) = array.extract::<PyReadonlyArrayDyn<'_, i32>>() {
            let converted = arr.as_array().mapv(|x| x as I);
            return self.insert_array(key, converted);
        }
        if let Ok(arr) = array.extract::<PyReadonlyArrayDyn<'_, i64>>() {
            let converted = arr.as_array().mapv(|x| x as I);
            return self.insert_array(key, converted);
        }
        if let Ok(arr) = array.extract::<PyReadonlyArrayDyn<'_, u32>>() {
            let converted = arr.as_array().mapv(|x| x as U);
            return self.insert_array(key, converted);
        }
        if let Ok(arr) = array.extract::<PyReadonlyArrayDyn<'_, u64>>() {
            let converted = arr.as_array().mapv(|x| x as U);
            return self.insert_array(key, converted);
        }
        if let Ok(arr) = array.extract::<PyReadonlyArrayDyn<'_, bool>>() {
            return self.insert_array(key, arr.as_array().to_owned().into_dyn());
        }
        if let Ok(strings) = array.extract::<Vec<String>>() {
            return self.insert_array(key, Array1::from(strings).into_dyn());
        }
        Err(PyTypeError::new_err(
            "unsupported dtype: expected float32, float64, int32, int64, bool, uint32, uint64, or list[str]",
        ))
    }

    /// Return a zero-copy numpy view of the column data.
    ///
    /// The returned array shares memory with the internal Rust storage.
    /// For string columns a Python ``list[str]`` is returned instead.
    ///
    /// Parameters
    /// ----------
    /// key : str
    ///     Column name to retrieve.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray | list[str]
    ///     Array view for numeric/bool columns, or a Python list for string
    ///     columns.
    ///
    /// Raises
    /// ------
    /// KeyError
    ///     If ``key`` does not exist in this block.
    ///
    /// Examples
    /// --------
    /// >>> arr = block.view("x")  # numpy float32 view
    /// >>> syms = block.view("symbol")  # list of str
    fn view<'py>(&self, py: Python<'py>, key: &str) -> PyResult<Py<pyo3::types::PyAny>> {
        self.inner
            .with(|b| -> PyResult<Py<pyo3::types::PyAny>> {
                let col = b
                    .get(key)
                    .ok_or_else(|| PyKeyError::new_err(key.to_string()))?;
                match col {
                    Column::Float(a) => float_array_view(py, a.clone()),
                    Column::Int(a) => int_array_view(py, a.clone()),
                    Column::Bool(a) => bool_array_view(py, a.clone()),
                    Column::UInt(a) => uint_array_view(py, a.clone()),
                    Column::U8(a) => {
                        let arr = numpy::PyArray1::from_iter(py, a.iter().copied());
                        Ok(arr.into_any().unbind())
                    }
                    Column::String(a) => {
                        let list: Vec<String> = a.iter().cloned().collect();
                        Ok(pyo3::types::PyList::new(py, &list)?.into_any().unbind())
                    }
                }
            })
            .map_err(ffi_error_to_pyerr)?
    }

    /// Number of rows (axis-0 length), or ``None`` if the block has no columns.
    ///
    /// Returns
    /// -------
    /// int | None
    #[getter]
    fn nrows(&self) -> PyResult<Option<usize>> {
        self.with_block(|b| b.nrows())
    }

    /// Number of columns in this block.
    ///
    /// Returns
    /// -------
    /// int
    fn __len__(&self) -> PyResult<usize> {
        self.with_block(|b| b.len())
    }

    /// List all column names.
    ///
    /// Returns
    /// -------
    /// list[str]
    fn keys(&self) -> PyResult<Vec<String>> {
        self.with_block(|b| b.keys().map(|s| s.to_string()).collect())
    }

    /// Check whether a column name exists in this block.
    ///
    /// Parameters
    /// ----------
    /// key : str
    ///     Column name to test.
    ///
    /// Returns
    /// -------
    /// bool
    fn __contains__(&self, key: &str) -> PyResult<bool> {
        self.with_block(|b| b.contains_key(key))
    }

    /// Remove a column by name.
    ///
    /// Parameters
    /// ----------
    /// key : str
    ///     Column name to remove.
    ///
    /// Raises
    /// ------
    /// KeyError
    ///     If ``key`` does not exist.
    fn remove(&mut self, key: &str) -> PyResult<()> {
        let removed = self
            .inner
            .with_mut(|b| b.remove(key).is_some())
            .map_err(ffi_error_to_pyerr)?;
        if removed {
            Ok(())
        } else {
            Err(PyKeyError::new_err(key.to_string()))
        }
    }

    /// Return the dtype string for the given column.
    ///
    /// Parameters
    /// ----------
    /// key : str
    ///     Column name.
    ///
    /// Returns
    /// -------
    /// str
    ///     One of ``"float"``, ``"int"``, ``"uint"``, ``"bool"``,
    ///     ``"u8"``, ``"string"``.
    ///
    /// Raises
    /// ------
    /// KeyError
    ///     If ``key`` does not exist.
    fn dtype(&self, key: &str) -> PyResult<String> {
        self.with_block(|b| {
            let col = b
                .get(key)
                .ok_or_else(|| PyKeyError::new_err(key.to_string()))?;
            Ok::<String, PyErr>(format!("{}", col.dtype()))
        })?
    }

    fn __repr__(&self) -> PyResult<String> {
        self.with_block(|b| {
            let keys: Vec<&str> = b.keys().collect();
            format!("Block(nrows={:?}, keys={:?})", b.nrows(), keys)
        })
    }
}

impl PyBlock {
    /// Create a `PyBlock` from a Rust `CoreBlock`, allocating a new
    /// single-frame FFI store.
    pub(crate) fn from_core_block(block: CoreBlock) -> PyResult<Self> {
        let store = molrs_ffi::new_shared();
        let frame = store.borrow_mut().frame_new();
        store
            .borrow_mut()
            .set_block(frame, "__block__", block)
            .map_err(ffi_error_to_pyerr)?;
        let handle = store
            .borrow()
            .get_block(frame, "__block__")
            .map_err(ffi_error_to_pyerr)?;
        Ok(Self {
            inner: BlockRef::new(store, handle),
        })
    }

    /// Clone the underlying `CoreBlock` out of the store (deep copy).
    pub(crate) fn clone_core_block(&self) -> PyResult<CoreBlock> {
        self.inner.clone_block().map_err(ffi_error_to_pyerr)
    }

    /// Run a read-only closure on the underlying `CoreBlock`.
    fn with_block<R>(&self, f: impl FnOnce(&CoreBlock) -> R) -> PyResult<R> {
        self.inner.with(f).map_err(ffi_error_to_pyerr)
    }

    /// Insert a typed ndarray column, validating row count.
    fn insert_array<T: BlockDtype>(
        &mut self,
        key: &str,
        array: ndarray::ArrayD<T>,
    ) -> PyResult<()> {
        self.inner
            .with_mut(|b| {
                b.insert(key, array)
                    .map_err(|e| PyValueError::new_err(e.to_string()))
            })
            .map_err(ffi_error_to_pyerr)??;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Zero-copy numpy views backed by PyO3-owned ndarray storage
// ---------------------------------------------------------------------------

/// Create a zero-copy numpy view of a float ndarray.
///
/// # Safety
///
/// `borrow_from_array` creates a numpy view whose lifetime is pinned to the
/// `FloatArrayOwner` Python object. As long as the owner is alive the data
/// pointer is valid.
fn float_array_view(py: Python<'_>, array: ndarray::ArrayD<F>) -> PyResult<Py<pyo3::types::PyAny>> {
    let owner = Py::new(py, FloatArrayOwner { array })?;
    let owner = owner.into_bound(py);
    let view = unsafe {
        PyArrayDyn::<F>::borrow_from_array(&owner.borrow().array, owner.clone().into_any())
    };
    Ok(view.into_any().unbind())
}

/// Create a zero-copy numpy view of an integer ndarray.
///
/// # Safety
///
/// See [`float_array_view`].
fn int_array_view(py: Python<'_>, array: ndarray::ArrayD<I>) -> PyResult<Py<pyo3::types::PyAny>> {
    let owner = Py::new(py, IntArrayOwner { array })?;
    let owner = owner.into_bound(py);
    let view = unsafe {
        PyArrayDyn::<I>::borrow_from_array(&owner.borrow().array, owner.clone().into_any())
    };
    Ok(view.into_any().unbind())
}

/// Create a zero-copy numpy view of a boolean ndarray.
///
/// # Safety
///
/// See [`float_array_view`].
fn bool_array_view(
    py: Python<'_>,
    array: ndarray::ArrayD<bool>,
) -> PyResult<Py<pyo3::types::PyAny>> {
    let owner = Py::new(py, BoolArrayOwner { array })?;
    let owner = owner.into_bound(py);
    let view = unsafe {
        PyArrayDyn::<bool>::borrow_from_array(&owner.borrow().array, owner.clone().into_any())
    };
    Ok(view.into_any().unbind())
}

/// Create a zero-copy numpy view of an unsigned integer ndarray.
///
/// # Safety
///
/// See [`float_array_view`].
fn uint_array_view(py: Python<'_>, array: ndarray::ArrayD<U>) -> PyResult<Py<pyo3::types::PyAny>> {
    let owner = Py::new(py, UIntArrayOwner { array })?;
    let owner = owner.into_bound(py);
    let view = unsafe {
        PyArrayDyn::<U>::borrow_from_array(&owner.borrow().array, owner.clone().into_any())
    };
    Ok(view.into_any().unbind())
}
