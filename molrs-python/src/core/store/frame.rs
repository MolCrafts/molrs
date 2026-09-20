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

use crate::core::spatial::simbox::PyBox;
use crate::core::store::block::PyBlock;
use crate::helpers::molrs_error_to_pyerr;
use crate::store::ffi_error_to_pyerr;
use molrs::store::frame::Frame as CoreFrame;
use molrs::store::meta::{MetaMap, MetaValue};
use molrs_ffi::FrameRef;
use pyo3::exceptions::{PyKeyError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyCapsule, PyDict, PyFloat, PyInt, PyList, PyString};
use serde_json::Value as JsonValue;

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
        Ok(Self {
            inner: meta_value_from_dtype(dtype, value)?,
        })
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype()
    }

    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, pyo3::types::PyTuple>)> {
        let this = slf.borrow();
        crate::helpers::reduce_via_type(slf.as_any(), (this.dtype(), this.value(slf.py())?))
    }

    #[getter]
    fn value(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        meta_value_to_py(py, &self.inner)
    }

    fn __repr__(&self) -> String {
        format!(
            "MetaValue(dtype='{}', value={:?})",
            self.inner.dtype(),
            self.inner
        )
    }
}

/// Collect a mapping (or an iterable of pairs) into a fresh [`MetaMap`].
///
/// Used by the wholesale `frame.meta = ...` replacement, where no slot exists
/// yet, so every dtype is either pinned by a [`MetaValue`] or inferred.
fn mapping_to_meta_map(source: &Bound<'_, PyAny>) -> PyResult<MetaMap> {
    // Another frame's metadata copies across whole, tags and all: going through
    // plain values would silently widen every non-default dtype.
    if let Ok(view) = source.extract::<PyRef<'_, PyFrameMeta>>() {
        return view.map();
    }
    let mut map = MetaMap::new();
    if source.hasattr("keys")? {
        for key in source.call_method0("keys")?.try_iter()? {
            let key = key?;
            let name: String = key.extract()?;
            map.insert(name, infer_meta_value(&source.get_item(&key)?)?);
        }
        return Ok(map);
    }
    for pair in source.try_iter()? {
        let (name, value): (String, Bound<'_, PyAny>) = pair?.extract()?;
        map.insert(name, infer_meta_value(&value)?);
    }
    Ok(map)
}

/// Live, write-through view of a frame's metadata.
///
/// An ordinary Python mapping: reading a key yields a plain value — scalars
/// unwrap, fixed-length vectors become lists, JSON documents become the decoded
/// object — and writing one lands in the frame.
///
/// **The dtype belongs to the key, not to the value handed in.** That is how
/// `mrec` already models metadata (`SequenceSchema.declare_meta(key, dtype)`),
/// so writing a plain value to a key that already exists keeps that key's
/// dtype and refuses a value it cannot hold; `frame.meta["t"] = frame.meta["t"]`
/// is therefore an identity. [`MetaValue`](PyMetaValue) is how a key is given a
/// dtype other than the inferred default, and [`dtype`](Self::dtype) reads the
/// tag back.
///
/// A JSON document is handed back decoded, which makes it a **snapshot**:
/// `frame.meta["run"]["step"] = 3` mutates a copy. Read, modify, write back.
#[pyclass(module = "molrs._lib", name = "FrameMeta", unsendable)]
pub struct PyFrameMeta {
    inner: FrameRef,
}

impl PyFrameMeta {
    fn map(&self) -> PyResult<MetaMap> {
        self.inner
            .with(|f| f.meta.clone())
            .map_err(ffi_error_to_pyerr)
    }

    fn tag_of(&self, key: &str) -> PyResult<Option<&'static str>> {
        self.inner
            .with(|f| f.meta.get(key).map(MetaValue::dtype))
            .map_err(ffi_error_to_pyerr)
    }

    /// Typed-slot write: an existing key keeps its dtype, a new key infers one.
    fn typed_for(&self, key: &str, value: &Bound<'_, PyAny>) -> PyResult<MetaValue> {
        if value.extract::<PyRef<'_, PyMetaValue>>().is_ok() {
            return infer_meta_value(value);
        }
        match self.tag_of(key)? {
            Some(dtype) => meta_value_from_dtype(dtype, value).map_err(|err| {
                PyTypeError::new_err(format!(
                    "metadata key '{key}' is {dtype}; assign a MetaValue to change it ({err})"
                ))
            }),
            None => infer_meta_value(value),
        }
    }

    fn store(&mut self, key: &str, value: MetaValue) -> PyResult<()> {
        self.inner
            .with_mut(|f| {
                f.meta.insert(key, value);
            })
            .map_err(ffi_error_to_pyerr)
    }

    fn take(&mut self, key: &str) -> PyResult<Option<MetaValue>> {
        self.inner
            .with_mut(|f| f.meta.remove(key))
            .map_err(ffi_error_to_pyerr)
    }

    fn as_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (key, value) in self.map()?.iter() {
            dict.set_item(key, meta_value_to_py(py, value)?)?;
        }
        Ok(dict)
    }

    /// Absorb a mapping or an iterable of `(key, value)` pairs.
    fn absorb(&mut self, other: &Bound<'_, PyAny>) -> PyResult<()> {
        if other.hasattr("keys")? {
            for key in other.call_method0("keys")?.try_iter()? {
                let key = key?;
                let name: String = key.extract()?;
                let value = other.get_item(&key)?;
                let typed = self.typed_for(&name, &value)?;
                self.store(&name, typed)?;
            }
            return Ok(());
        }
        for pair in other.try_iter()? {
            let (name, value): (String, Bound<'_, PyAny>) = pair?.extract()?;
            let typed = self.typed_for(&name, &value)?;
            self.store(&name, typed)?;
        }
        Ok(())
    }
}

#[pymethods]
impl PyFrameMeta {
    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        match self.map()?.get(key) {
            Some(value) => meta_value_to_py(py, value),
            None => Err(PyKeyError::new_err(key.to_owned())),
        }
    }

    fn __setitem__(&mut self, key: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let typed = self.typed_for(key, value)?;
        self.store(key, typed)
    }

    fn __delitem__(&mut self, key: &str) -> PyResult<()> {
        match self.take(key)? {
            Some(_) => Ok(()),
            None => Err(PyKeyError::new_err(key.to_owned())),
        }
    }

    fn __contains__(&self, key: &str) -> PyResult<bool> {
        Ok(self.map()?.get(key).is_some())
    }

    fn __len__(&self) -> PyResult<usize> {
        Ok(self.map()?.len())
    }

    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyIterator>> {
        PyList::new(py, self.keys()?)?.try_iter()
    }

    fn __eq__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.as_dict(py)?.as_any().eq(other)
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!("{}", self.as_dict(py)?))
    }

    /// The dtype tag stored for `key`, or ``None`` when the key is absent.
    fn dtype(&self, key: &str) -> PyResult<Option<&'static str>> {
        self.tag_of(key)
    }

    fn keys(&self) -> PyResult<Vec<String>> {
        let mut names: Vec<String> = self.map()?.keys().cloned().collect();
        names.sort_unstable();
        Ok(names)
    }

    fn values(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        let map = self.map()?;
        self.keys()?
            .iter()
            .map(|key| meta_value_to_py(py, map.get(key).expect("key from this map")))
            .collect()
    }

    fn items(&self, py: Python<'_>) -> PyResult<Vec<(String, Py<PyAny>)>> {
        let map = self.map()?;
        self.keys()?
            .into_iter()
            .map(|key| {
                let value = meta_value_to_py(py, map.get(&key).expect("key from this map"))?;
                Ok((key, value))
            })
            .collect()
    }

    #[pyo3(signature = (key, default=None))]
    fn get(&self, py: Python<'_>, key: &str, default: Option<Py<PyAny>>) -> PyResult<Py<PyAny>> {
        match self.map()?.get(key) {
            Some(value) => meta_value_to_py(py, value),
            None => Ok(default.unwrap_or_else(|| py.None())),
        }
    }

    #[pyo3(signature = (key, *default))]
    fn pop(
        &mut self,
        py: Python<'_>,
        key: &str,
        default: &Bound<'_, pyo3::types::PyTuple>,
    ) -> PyResult<Py<PyAny>> {
        match self.take(key)? {
            Some(value) => meta_value_to_py(py, &value),
            None if default.len() == 1 => Ok(default.get_item(0)?.unbind()),
            None => Err(PyKeyError::new_err(key.to_owned())),
        }
    }

    fn popitem(&mut self, py: Python<'_>) -> PyResult<(String, Py<PyAny>)> {
        let key = self
            .keys()?
            .pop()
            .ok_or_else(|| PyKeyError::new_err("popitem(): metadata is empty"))?;
        let value = self.take(&key)?.expect("key came from this map");
        Ok((key, meta_value_to_py(py, &value)?))
    }

    fn clear(&mut self) -> PyResult<()> {
        self.inner
            .with_mut(|f| f.meta.clear())
            .map_err(ffi_error_to_pyerr)
    }

    #[pyo3(signature = (key, default=None))]
    fn setdefault(
        &mut self,
        py: Python<'_>,
        key: &str,
        default: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        if let Some(value) = self.map()?.get(key) {
            return meta_value_to_py(py, value);
        }
        // `dict.setdefault(k)` inserts None; so does this, now that None is a
        // JSON null rather than a rejection.
        let value = match default {
            Some(value) => value,
            None => py.None().into_bound(py),
        };
        let typed = self.typed_for(key, &value)?;
        self.store(key, typed)?;
        self.__getitem__(py, key)
    }

    #[pyo3(signature = (other=None, **kwargs))]
    fn update(
        &mut self,
        other: Option<&Bound<'_, PyAny>>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        if let Some(other) = other {
            self.absorb(other)?;
        }
        if let Some(kwargs) = kwargs {
            self.absorb(kwargs.as_any())?;
        }
        Ok(())
    }

    /// A plain `dict` snapshot; mutating it does not touch the frame.
    fn copy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.as_dict(py)
    }

    /// The same snapshot with every dtype kept, as `dict[str, MetaValue]`.
    ///
    /// `dict(meta)` throws the tags away, which is the right default for
    /// reading; this is what a copy or a pickle has to carry instead.
    fn typed<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (key, value) in self.map()?.iter() {
            dict.set_item(
                key,
                Py::new(
                    py,
                    PyMetaValue {
                        inner: value.clone(),
                    },
                )?,
            )?;
        }
        Ok(dict)
    }

    fn __or__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let merged = self.as_dict(py)?;
        merged.update(other.cast::<pyo3::types::PyMapping>()?)?;
        Ok(merged)
    }

    fn __ror__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let merged = other.cast::<PyDict>()?.copy()?;
        merged.update(self.as_dict(py)?.as_mapping())?;
        Ok(merged)
    }

    fn __ior__(&mut self, other: &Bound<'_, PyAny>) -> PyResult<()> {
        self.absorb(other)
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
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(_args: &Bound<'_, PyAny>, _kwargs: Option<&Bound<'_, PyAny>>) -> Self {
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

    /// Live, write-through view of this frame's metadata.
    ///
    /// Returns
    /// -------
    /// FrameMeta
    ///     A mapping of plain Python values. Mutations persist; the dtype of an
    ///     existing key is kept (assign a :class:`MetaValue` to change it).
    #[getter]
    fn meta(&self) -> PyFrameMeta {
        PyFrameMeta {
            inner: self.inner.clone(),
        }
    }

    /// Replace the metadata dictionary.
    ///
    /// Values may be :class:`MetaValue` or any JSON-serializable object
    /// (``str``, ``int``, ``float``, ``bool``, ``list``, ``dict``).
    #[setter]
    fn set_meta(&mut self, meta: &Bound<'_, PyAny>) -> PyResult<()> {
        // Build the replacement first: `frame.meta = frame.meta` and
        // `Frame.copy` both read the map they are about to overwrite.
        let map = mapping_to_meta_map(meta)?;
        self.inner
            .with_mut(|f| {
                f.meta = map;
            })
            .map_err(ffi_error_to_pyerr)
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
    /// The capsule name is ``molrs_ffi::abi::frameref_capsule_name()`` —
    /// ``"molrs.FrameRef/<major.minor>"``. The name carries the ABI line so a
    /// consumer built on a different molrs minor fails the name check cleanly
    /// instead of dereferencing a possibly drifted layout.
    ///
    /// Returns
    /// -------
    /// capsule
    ///     A ``PyCapsule`` named ``"molrs.FrameRef/<major.minor>"`` whose
    ///     pointer is ``*mut *mut`` :class:`molrs_ffi.FrameRef` (a cloned
    ///     handle).
    fn _ffi_frameref_capsule<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyCapsule>> {
        // Box a clone of the handle and hand the raw pointer to the capsule.
        // `FrameRef` holds an `Rc` and is therefore not `Send`; a bare
        // `*mut FrameRef` is not `Send` either, so wrap it in `FrameRefPtr`
        // which asserts `Send`. This is sound because the capsule is only
        // ever touched under the GIL (molrs FFI is single-threaded — see the
        // threading note in `molrs_ffi::shared`).
        let raw = FrameRefPtr(Box::into_raw(Box::new(self.inner.clone())));
        let name = molrs_ffi::abi::frameref_capsule_name().to_owned();
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
        // payload in one step, returning the `*mut *mut FrameRef`. The name
        // carries the ABI line, so a producer on another molrs minor line
        // (including pre-0.14 unversioned `molrs.FrameRef` capsules) fails
        // here cleanly instead of being dereferenced.
        let expected = molrs_ffi::abi::frameref_capsule_name();
        let ptr = capsule.pointer_checked(Some(expected)).map_err(|err| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "{err} — this build of molcrafts-molrs speaks FFI ABI line \
                 {line} (capsule name {expected:?}); the producing extension \
                 embeds a different molrs minor line. Align both packages on \
                 one minor line.",
                line = molrs_ffi::abi::abi_line(),
            ))
        })?;
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

/// The plain Python counterpart of a stored value: scalars unwrap, fixed-length
/// vectors become lists, and a JSON document becomes the decoded object.
fn meta_value_to_py(py: Python<'_>, value: &MetaValue) -> PyResult<Py<PyAny>> {
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
    Ok(match value {
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
        MetaValue::Json(v) => json_to_py(py, v)?,
    })
}

/// Build a [`MetaValue`] carrying an explicit dtype tag, coercing `value` into it.
///
/// This is the one place a tag becomes storage, so a tag read back off an
/// existing slot ([`MetaValue::dtype`]) round-trips through it exactly.
/// Coercion never truncates: a value the tag cannot hold raises.
fn meta_value_from_dtype(dtype: &str, value: &Bound<'_, PyAny>) -> PyResult<MetaValue> {
    fn array<T, const N: usize>(value: &Bound<'_, PyAny>, dtype: &str) -> PyResult<[T; N]>
    where
        for<'a, 'py> T: FromPyObject<'a, 'py>,
    {
        let values: Vec<T> = value.extract()?;
        values.try_into().map_err(|values: Vec<T>| {
            PyTypeError::new_err(format!("{dtype} requires {N} values, got {}", values.len()))
        })
    }

    Ok(match dtype {
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
        "json" => MetaValue::Json(py_to_json(value)?),
        _ => {
            return Err(PyTypeError::new_err(format!(
                "unknown metadata dtype '{dtype}'"
            )));
        }
    })
}

/// Pick a dtype for a value written to a key that has none yet.
///
/// Scalars take their exact Python counterpart; a numeric sequence of 3, 6 or 9
/// becomes the matching fixed-length vector — the shapes `mrec` declares — and
/// everything else (objects, ragged or non-numeric sequences, `None`) is a JSON
/// document. A [`MetaValue`](PyMetaValue) passes through with its tag intact.
fn infer_meta_value(value: &Bound<'_, PyAny>) -> PyResult<MetaValue> {
    if let Ok(typed) = value.extract::<PyRef<'_, PyMetaValue>>() {
        return Ok(typed.inner.clone());
    }
    // bool before int: Python bools are ints.
    if let Ok(v) = value.cast::<PyBool>() {
        return Ok(MetaValue::Bool(v.is_true()));
    }
    if let Ok(v) = value.cast::<PyInt>() {
        if let Ok(n) = v.extract::<i64>() {
            return Ok(MetaValue::I64(n));
        }
        if let Ok(n) = v.extract::<u64>() {
            return Ok(MetaValue::U64(n));
        }
        return Err(PyTypeError::new_err(
            "integer metadata does not fit i64/u64",
        ));
    }
    if let Ok(v) = value.cast::<PyFloat>() {
        return Ok(MetaValue::F64(v.extract()?));
    }
    if let Ok(v) = value.cast::<PyString>() {
        return Ok(MetaValue::String(v.extract()?));
    }
    if let Ok(list) = value.cast::<PyList>() {
        let vector = match list.len() {
            3 => Some("f64x3"),
            6 => Some("f64x6"),
            9 => Some("f64x9"),
            _ => None,
        };
        if let Some(dtype) = vector
            && let Ok(typed) = meta_value_from_dtype(dtype, value)
        {
            return Ok(typed);
        }
    }
    Ok(MetaValue::Json(py_to_json(value)?))
}

fn json_to_py(py: Python<'_>, value: &JsonValue) -> PyResult<Py<PyAny>> {
    Ok(match value {
        JsonValue::Null => py.None(),
        JsonValue::Bool(b) => b.into_pyobject(py)?.to_owned().into_any().unbind(),
        JsonValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.into_pyobject(py)?.into_any().unbind()
            } else if let Some(u) = n.as_u64() {
                u.into_pyobject(py)?.into_any().unbind()
            } else {
                n.as_f64()
                    .unwrap_or(f64::NAN)
                    .into_pyobject(py)?
                    .into_any()
                    .unbind()
            }
        }
        JsonValue::String(s) => s.into_pyobject(py)?.into_any().unbind(),
        JsonValue::Array(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(json_to_py(py, item)?)?;
            }
            list.into_any().unbind()
        }
        JsonValue::Object(map) => {
            let dict = PyDict::new(py);
            for (key, item) in map {
                dict.set_item(key, json_to_py(py, item)?)?;
            }
            dict.into_any().unbind()
        }
    })
}

fn py_to_json(value: &Bound<'_, PyAny>) -> PyResult<JsonValue> {
    if value.is_none() {
        return Ok(JsonValue::Null);
    }
    if let Ok(b) = value.cast::<PyBool>() {
        return Ok(JsonValue::Bool(b.is_true()));
    }
    if let Ok(i) = value.cast::<PyInt>() {
        return Ok(JsonValue::from(i.extract::<i64>()?));
    }
    if let Ok(f) = value.cast::<PyFloat>() {
        return Ok(serde_json::Number::from_f64(f.extract::<f64>()?)
            .map(JsonValue::Number)
            .unwrap_or(JsonValue::Null));
    }
    if let Ok(s) = value.cast::<PyString>() {
        return Ok(JsonValue::String(s.extract::<String>()?));
    }
    if let Ok(dict) = value.cast::<PyDict>() {
        let mut map = serde_json::Map::new();
        for (k, v) in dict.iter() {
            let key: String = k.extract()?;
            map.insert(key, py_to_json(&v)?);
        }
        return Ok(JsonValue::Object(map));
    }
    if let Ok(list) = value.cast::<PyList>() {
        let mut items = Vec::with_capacity(list.len());
        for item in list.iter() {
            items.push(py_to_json(&item)?);
        }
        return Ok(JsonValue::Array(items));
    }
    Err(PyTypeError::new_err(format!(
        "metadata value is not JSON-serializable: {value}"
    )))
}
