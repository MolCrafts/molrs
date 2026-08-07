//! Python wrapper for `Frame`, a hierarchical data container backed by the
//! shared FFI store.
//!
//! A [`PyFrame`] maps string keys (e.g. `"atoms"`, `"bonds"`, `"angles"`) to
//! [`PyBlock`] column stores. It may optionally carry a [`PyBox`] (simulation
//! box) and an exact-dtype metadata map.
//!
//! # Conventional Block Layout
//!
//! | Block key   | Expected columns                                      | Notes                                    |
//! |-------------|-------------------------------------------------------|------------------------------------------|
//! | `"atoms"`   | `symbol` (str), `x`/`y`/`z` (float), `mass` (float)  | Atom positions and properties             |
//! | `"bonds"`   | `atomi`/`atomj` (uint), `order` (float)               | Bond topology (indices into atoms)        |
//! | `"angles"`  | `atomi`/`atomj`/`atomk` (uint), `type` (int)          | Angle topology                            |
//!
//! The frame itself does **not** enforce cross-block row consistency; that is
//! the caller's responsibility (use [`PyFrame::validate`] to check).

use std::ffi::CString;

use crate::core::spatial::simbox::PyBox;
use crate::core::store::block::PyBlock;
use crate::helpers::{message_format, molrs_error_to_pyerr, py_value_err};
use crate::store::ffi_error_to_pyerr;
use molrs::store::block::Block as CoreBlock;
use molrs::store::frame::Frame as CoreFrame;
use molrs::store::meta::{MetaMap, MetaValue};
use molrs_ffi::FrameRef;
use pyo3::exceptions::{PyKeyError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyCapsule, PyDict, PyList};

/// Exact-dtype frame metadata value.
#[pyclass(module = "molrs", name = "MetaValue", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyMetaValue {
    pub(crate) inner: MetaValue,
}

#[pymethods]
impl PyMetaValue {
    /// Construct a metadata value from a stable dtype tag and payload.
    #[new]
    fn new(dtype: &str, value: &Bound<'_, PyAny>) -> PyResult<Self> {
        fn array<T, const N: usize>(value: &Bound<'_, PyAny>, dtype: &str) -> PyResult<[T; N]>
        where
            for<'a, 'py> T: FromPyObject<'a, 'py>,
        {
            let values: Vec<T> = value.extract()?;
            values.try_into().map_err(|values: Vec<T>| {
                PyTypeError::new_err(format!("{dtype} requires {N} values, got {}", values.len()))
            })
        }

        let inner = match dtype {
            "bool" => MetaValue::Bool(value.extract()?),
            "i32" => MetaValue::I32(value.extract()?),
            "i64" => MetaValue::I64(value.extract()?),
            "u32" => MetaValue::U32(value.extract()?),
            "u64" => MetaValue::U64(value.extract()?),
            "f32" => MetaValue::F32(value.extract()?),
            "f64" => MetaValue::F64(value.extract()?),
            "string" => MetaValue::String(value.extract()?),
            "bool3" => MetaValue::Bool3(array(value, dtype)?),
            "i32x3" => MetaValue::I32x3(array(value, dtype)?),
            "i64x3" => MetaValue::I64x3(array(value, dtype)?),
            "u32x3" => MetaValue::U32x3(array(value, dtype)?),
            "u64x3" => MetaValue::U64x3(array(value, dtype)?),
            "f32x3" => MetaValue::F32x3(array(value, dtype)?),
            "f64x3" => MetaValue::F64x3(array(value, dtype)?),
            "f32x6" => MetaValue::F32x6(array(value, dtype)?),
            "f64x6" => MetaValue::F64x6(array(value, dtype)?),
            "f32x9" => MetaValue::F32x9(array(value, dtype)?),
            "f64x9" => MetaValue::F64x9(array(value, dtype)?),
            _ => {
                return Err(PyTypeError::new_err(format!(
                    "unknown metadata dtype '{dtype}'"
                )));
            }
        };
        Ok(Self { inner })
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype()
    }

    #[getter]
    fn value(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        macro_rules! scalar {
            ($value:expr) => {
                $value.into_pyobject(py)?.into_any().unbind()
            };
        }
        macro_rules! list {
            ($value:expr) => {
                PyList::new(py, $value)?.into_any().unbind()
            };
        }
        Ok(match &self.inner {
            MetaValue::Bool(v) => v.into_pyobject(py)?.to_owned().into_any().unbind(),
            MetaValue::I32(v) => scalar!(*v),
            MetaValue::I64(v) => scalar!(*v),
            MetaValue::U32(v) => scalar!(*v),
            MetaValue::U64(v) => scalar!(*v),
            MetaValue::F32(v) => scalar!(*v),
            MetaValue::F64(v) => scalar!(*v),
            MetaValue::String(v) => scalar!(v),
            MetaValue::Bool3(v) => list!(v),
            MetaValue::I32x3(v) => list!(v),
            MetaValue::I64x3(v) => list!(v),
            MetaValue::U32x3(v) => list!(v),
            MetaValue::U64x3(v) => list!(v),
            MetaValue::F32x3(v) => list!(v),
            MetaValue::F64x3(v) => list!(v),
            MetaValue::F32x6(v) => list!(v),
            MetaValue::F64x6(v) => list!(v),
            MetaValue::F32x9(v) => list!(v),
            MetaValue::F64x9(v) => list!(v),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "MetaValue(dtype='{}', value={:?})",
            self.inner.dtype(),
            self.inner
        )
    }
}

/// Hierarchical data container exposed to Python as `molrs.Frame`.
///
/// A `Frame` is a dictionary of named [`Block`](crate::core::store::block::PyBlock)s with
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
/// print(frame)          # Frame(blocks=['atoms'], box=yes)
/// print(frame.keys())   # ['atoms']
/// ```
#[pyclass(
    module = "molrs._lib",
    name = "Frame",
    from_py_object,
    unsendable,
    subclass
)]
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

    /// Build a frame from a dictionary of blocks.
    ///
    /// Accepts the exact ``{"blocks": {...}, "meta": {...}}`` frame shape.
    /// Column values use the same accepted types as :meth:`Block.insert` and
    /// every metadata value must be a :class:`MetaValue`.
    ///
    /// Parameters
    /// ----------
    /// data : dict
    ///     Frame data in the shared ``to_dict`` / ``from_dict`` exchange shape.
    ///
    /// Returns
    /// -------
    /// Frame
    #[staticmethod]
    fn from_dict(data: &Bound<'_, PyDict>) -> PyResult<Self> {
        if data.len() != 2 || !data.contains("blocks")? || !data.contains("meta")? {
            return Err(PyTypeError::new_err(
                "frame dict must contain exactly 'blocks' and 'meta'",
            ));
        }
        let blocks = data
            .get_item("blocks")?
            .expect("presence checked above")
            .cast_into::<PyDict>()
            .map_err(|_| PyTypeError::new_err("'blocks' must be a dict"))?;

        let mut frame = Self::new();
        for (block_name, columns) in blocks.iter() {
            let name: String = block_name.extract()?;
            let columns = columns
                .cast::<PyDict>()
                .map_err(|_| PyTypeError::new_err(format!("block '{name}' must be a dict")))?;

            let mut block = PyBlock::from_core_block(CoreBlock::new())?;
            for (column_name, values) in columns.iter() {
                let key: String = column_name.extract()?;
                block.insert_py_column(&key, &values)?;
            }

            let core_block = block.clone_core_block()?;
            frame
                .inner
                .store
                .borrow_mut()
                .set_block(frame.inner.id, &name, core_block)
                .map_err(ffi_error_to_pyerr)?;
        }

        let meta = data
            .get_item("meta")?
            .expect("presence checked above")
            .cast_into::<PyDict>()
            .map_err(|_| PyTypeError::new_err("'meta' must be a dict"))?;
        frame.set_meta(&meta)?;

        Ok(frame)
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
        if let Ok(inner) = self.inner.block(key) {
            return Ok(Py::new(py, PyBlock { inner })?.into_any());
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
        if let Ok(block) = value.extract::<PyRef<'_, PyBlock>>() {
            let core_block = block.clone_core_block()?;
            return self
                .inner
                .store
                .borrow_mut()
                .set_block(self.inner.id, key, core_block)
                .map_err(ffi_error_to_pyerr);
        }
        Err(PyTypeError::new_err("value must be a Block"))
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

    /// The simulation :class:`Box` attached to this frame, or ``None``.
    ///
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
    #[getter]
    fn get_box(&self) -> PyResult<Option<PyBox>> {
        Ok(self
            .inner
            .box_clone()
            .map_err(ffi_error_to_pyerr)?
            .map(|inner| PyBox { inner }))
    }

    /// Set (or clear) the simulation :class:`Box`.
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
    #[setter]
    fn set_box(&mut self, box_: Option<&PyBox>) -> PyResult<()> {
        self.inner
            .set_box(box_.map(|sb| sb.inner.clone()))
            .map_err(ffi_error_to_pyerr)
    }

    /// Exact-dtype metadata dictionary (``dict[str, MetaValue]``).
    ///
    /// Returns
    /// -------
    /// dict[str, MetaValue]
    ///     Typed metadata attached to this frame.
    #[getter]
    fn meta<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        let meta = self.with_frame(|f| f.meta.clone())?;
        for (k, v) in meta {
            dict.set_item(k, Py::new(py, PyMetaValue { inner: v })?)?;
        }
        Ok(dict)
    }

    /// Replace the metadata dictionary.
    ///
    /// Parameters
    /// ----------
    /// meta : dict[str, MetaValue]
    ///     New metadata. Every value must carry an explicit dtype.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If a key is not ``str`` or a value is not ``MetaValue``.
    #[setter]
    fn set_meta(&mut self, meta: &Bound<'_, PyDict>) -> PyResult<()> {
        let mut map = MetaMap::with_capacity(meta.len());
        for (k, v) in meta.iter() {
            let key: String = k.extract()?;
            let val: PyRef<'_, PyMetaValue> = v.extract().map_err(|_| {
                PyTypeError::new_err(format!("metadata '{key}' must be a MetaValue"))
            })?;
            map.insert(key, val.inner.clone());
        }
        self.inner
            .with_mut(|f| {
                f.meta = map;
            })
            .map_err(ffi_error_to_pyerr)?;
        Ok(())
    }

    /// Judge this frame against the canonical Frame schema.
    ///
    /// Delegates to ``molrs``'s ``Validator::canonical`` — dtype, shape,
    /// required columns, and endpoint ranges. Callers must not re-implement
    /// those checks in Python.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the frame does not conform (message is the full report).
    fn validate(&self) -> PyResult<()> {
        self.with_frame(|f| f.validate().map_err(molrs_error_to_pyerr))?
    }

    /// Return a deep copy of this frame.
    ///
    /// All blocks, the simulation box, and typed metadata are cloned into
    /// a new, independent frame backed by its own store.
    ///
    /// Returns
    /// -------
    /// Frame
    ///     An independent copy.
    fn copy(&self) -> PyResult<Self> {
        Self::from_core_frame(self.clone_core_frame()?)
    }

    /// Encode this frame as streaming wire bytes.
    ///
    /// The inverse of :meth:`from_bytes`. This is the encoding
    /// :class:`FrameServer` puts on the wire, so a consumer decodes a live
    /// stream with :meth:`from_bytes` and never re-derives the layout.
    ///
    /// Parameters
    /// ----------
    /// format : {"msgpack", "json"}
    ///     Wire encoding. MessagePack (default) is compact binary; JSON is
    ///     text, for debugging and non-Rust peers.
    ///
    /// Returns
    /// -------
    /// bytes
    #[pyo3(signature = (format = "msgpack"))]
    fn to_bytes<'py>(&self, py: Python<'py>, format: &str) -> PyResult<Bound<'py, PyBytes>> {
        let fmt = message_format(format)?;
        let bytes = self
            .with_frame(|f| molrs::stream::frame_to_bytes(f, fmt))?
            .map_err(py_value_err)?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// Rebuild a frame from streaming wire bytes.
    ///
    /// Parameters
    /// ----------
    /// data : bytes
    ///     A payload produced by :meth:`to_bytes` or by a Rust
    ///     ``molrs::net::FrameServer``.
    /// format : {"msgpack", "json"}
    ///     Wire encoding the payload was written with.
    ///
    /// Returns
    /// -------
    /// Frame
    #[staticmethod]
    #[pyo3(signature = (data, format = "msgpack"))]
    fn from_bytes(data: &[u8], format: &str) -> PyResult<Self> {
        let fmt = message_format(format)?;
        let frame = molrs::stream::bytes_to_frame(data, fmt).map_err(py_value_err)?;
        Self::from_core_frame(frame)
    }

    fn __repr__(&self) -> PyResult<String> {
        self.with_frame(|f| {
            let keys: Vec<&str> = f.keys().collect();
            format!(
                "Frame(blocks={:?}, box={})",
                keys,
                if f.simbox.is_some() { "yes" } else { "no" }
            )
        })
    }

    /// Export this frame's FFI handle as a ``PyCapsule``.
    ///
    /// The capsule wraps a *clone* of this frame's ``FrameRef`` handle. The
    /// clone shares the same underlying ``Store`` (``Rc<RefCell<Store>>``),
    /// so a consumer (e.g. Atomiverse C++ via the molrs-cxxapi bridge) that
    /// resolves the capsule reads and writes the *same* frame data: no deep
    /// copy is made. The capsule's destructor reclaims the boxed
    /// ``FrameRef`` on capsule destruction, dropping its two ``Rc``
    /// references.
    ///
    /// Pointer indirection: PyO3's ``PyCapsule::new`` heap-boxes its
    /// payload, and the payload here is a ``#[repr(transparent)]``
    /// ``FrameRefPtr`` (itself ``*mut FrameRef``). The capsule's ``void*``
    /// is therefore ``*mut FrameRefPtr`` ≡ ``*mut *mut FrameRef``: one
    /// dereference yields the ``*mut FrameRef`` clone. Atomiverse's
    /// ``frame_clone_from_addr`` does exactly that double-resolve.
    ///
    /// The capsule name is the C string ``"molrs.FrameRef"``.
    ///
    /// Returns
    /// -------
    /// capsule
    ///     A ``PyCapsule`` named ``"molrs.FrameRef"`` whose pointer is
    ///     ``*mut *mut`` :class:`molrs_ffi.FrameRef` (a cloned handle).
    fn _ffi_frameref_capsule<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyCapsule>> {
        // Box a clone of the handle and hand the raw pointer to the capsule.
        // `FrameRef` holds an `Rc` and is therefore not `Send`; a bare
        // `*mut FrameRef` is not `Send` either, so wrap it in `FrameRefPtr`
        // which asserts `Send`. This is sound because the capsule is only
        // ever touched under the GIL (molrs FFI is single-threaded — see the
        // threading note in `molrs_ffi::shared`).
        let raw = FrameRefPtr(Box::into_raw(Box::new(self.inner.clone())));
        let name = CString::new("molrs.FrameRef").expect("static capsule name");
        PyCapsule::new_with_destructor(py, raw, Some(name), |ptr: FrameRefPtr, _ctx| {
            // SAFETY: `ptr.0` is the pointer produced by `Box::into_raw`
            // above and is reclaimed exactly once when the capsule dies.
            drop(unsafe { Box::from_raw(ptr.0) });
        })
    }

    /// Build a `Frame` from a ``"molrs.FrameRef"`` capsule — the **return path**
    /// for a downstream Rust consumer (e.g. molpack handing back a packed frame).
    ///
    /// Symmetric with [`_ffi_frameref_capsule`](Self::_ffi_frameref_capsule):
    /// the consumer wraps its result frame in a `FrameRef`, exports a capsule of
    /// the same shape and name, and calls this. The new `Frame` **shares** the
    /// producer's store (one `Rc` bump), so no deep copy is made.
    #[staticmethod]
    fn _from_ffi_frameref_capsule(capsule: &Bound<'_, PyCapsule>) -> PyResult<Self> {
        // `pointer_checked` validates the capsule name and rejects a null
        // payload in one step, returning the `*mut *mut FrameRef`.
        let ptr = capsule.pointer_checked(Some(c"molrs.FrameRef"))?;
        let pp = ptr.as_ptr() as *const *const FrameRef;
        // SAFETY: a `"molrs.FrameRef"` capsule's `void*` is `*mut *mut FrameRef`
        // (the exporter boxes a `*mut FrameRef`). Deref once to reach the cloned
        // handle and `.clone()` it (two `Rc` bumps) onto the same store. The
        // capsule is only ever touched under the GIL.
        let fref = unsafe { (**pp).clone() };
        Ok(Self { inner: fref })
    }
}

/// `Send` wrapper around a `*mut FrameRef` so it can ride inside a
/// `PyCapsule` (whose payload must be `Send`).
///
/// `FrameRef` is `!Send` (it holds an `Rc`), and raw pointers are `!Send` by
/// default. The capsule is only ever created, read, and destroyed while the
/// Python GIL is held, so no cross-thread access of the `Rc` ever occurs —
/// the `unsafe impl Send` is upheld by that single-threaded discipline.
///
/// `#[repr(transparent)]` guarantees this newtype has exactly the layout of
/// the wrapped `*mut FrameRef`. PyO3's `PyCapsule::new` heap-boxes the
/// payload, so the capsule's `void*` is `*mut FrameRefPtr`; the transparent
/// repr makes that pointer reinterpretable as `*mut *mut FrameRef`, which is
/// how Atomiverse's molrs-cxxapi bridge resolves it
/// (`frame_clone_from_addr`).
#[repr(transparent)]
struct FrameRefPtr(*mut FrameRef);

// SAFETY: see the type-level doc — single-threaded, GIL-guarded use only.
unsafe impl Send for FrameRefPtr {}

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
    pub(crate) fn with_frame<R>(&self, f: impl FnOnce(&CoreFrame) -> R) -> PyResult<R> {
        self.inner.with(f).map_err(ffi_error_to_pyerr)
    }
}
