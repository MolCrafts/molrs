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
use crate::helpers::molrs_error_to_pyerr;
use crate::store::ffi_error_to_pyerr;
use molrs::store::block::Block as CoreBlock;
use molrs::store::frame::Frame as CoreFrame;
use molrs::store::meta::{MetaMap, MetaValue};
use molrs_ffi::FrameRef;
use pyo3::exceptions::{PyKeyError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{
    PyBool, PyCapsule, PyDict, PyFloat, PyInt, PyIterator, PyList, PySequence, PyString, PyTuple,
};

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
    })
}

fn numpy_scalar_item<'py>(value: &Bound<'py, PyAny>) -> Option<Bound<'py, PyAny>> {
    let shape = value.getattr("shape").ok()?;
    let dims: Vec<usize> = shape.extract().ok()?;
    if !dims.is_empty() {
        return None;
    }
    value.call_method0("item").ok()
}

fn extract_array<T, const N: usize>(seq: &Bound<'_, PySequence>) -> PyResult<[T; N]>
where
    for<'a, 'py> T: FromPyObject<'a, 'py>,
{
    let mut values = Vec::with_capacity(N);
    for i in 0..N {
        let item = seq.get_item(i)?;
        if let Ok(value) = item.extract::<T>() {
            values.push(value);
            continue;
        }
        if let Some(item) = numpy_scalar_item(&item) {
            values.push(
                item.extract()
                    .map_err(|_| PyTypeError::new_err("sequence item has the wrong type"))?,
            );
            continue;
        }
        return Err(PyTypeError::new_err("sequence item has the wrong type"));
    }
    values.try_into().map_err(|values: Vec<T>| {
        PyTypeError::new_err(format!("expected {N} values, got {}", values.len()))
    })
}

fn sequence_to_meta_value(seq: &Bound<'_, PySequence>) -> PyResult<MetaValue> {
    match seq.len()? {
        3 => {
            if let Ok(values) = extract_array::<bool, 3>(seq) {
                return Ok(MetaValue::Bool3(values));
            }
            if let Ok(values) = extract_array::<i64, 3>(seq) {
                return Ok(MetaValue::I64x3(values));
            }
            if let Ok(values) = extract_array::<f64, 3>(seq) {
                return Ok(MetaValue::F64x3(values));
            }
        }
        6 => {
            if let Ok(values) = extract_array::<f64, 6>(seq) {
                return Ok(MetaValue::F64x6(values));
            }
        }
        9 => {
            if let Ok(values) = extract_array::<f64, 9>(seq) {
                return Ok(MetaValue::F64x9(values));
            }
        }
        _ => {}
    }
    Err(PyTypeError::new_err(
        "metadata sequences must be length 3, 6, or 9 of bool/int/float",
    ))
}

fn py_to_meta_value(value: &Bound<'_, PyAny>) -> PyResult<MetaValue> {
    if let Ok(typed) = value.extract::<PyRef<'_, PyMetaValue>>() {
        return Ok(typed.inner.clone());
    }
    if let Some(item) = numpy_scalar_item(value) {
        return py_to_meta_value(&item);
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
        return Ok(MetaValue::String(v.to_str()?.to_owned()));
    }
    if value.cast::<PyDict>().is_ok() {
        return Err(PyTypeError::new_err(
            "metadata values must be bool, int, float, str, a fixed-length sequence, or MetaValue",
        ));
    }
    if let Ok(seq) = value.cast::<PySequence>() {
        return sequence_to_meta_value(&seq);
    }
    Err(PyTypeError::new_err(format!(
        "metadata values must be bool, int, float, str, a fixed-length sequence, or MetaValue, got {}",
        value.get_type().name()?
    )))
}

fn mapping_to_meta_map(value: &Bound<'_, PyAny>) -> PyResult<MetaMap> {
    if let Ok(meta) = value.extract::<PyRef<'_, PyFrameMeta>>() {
        return meta.clone_map();
    }
    let items = value
        .call_method0("items")
        .map_err(|_| PyTypeError::new_err("meta must be a mapping of str to values"))?;
    let mut map = MetaMap::new();
    for pair in items.try_iter()? {
        let pair = pair?;
        let (key, raw) = pair_to_entry(&pair)?;
        map.insert(key, py_to_meta_value(&raw)?);
    }
    Ok(map)
}

fn pair_to_entry<'py>(pair: &Bound<'py, PyAny>) -> PyResult<(String, Bound<'py, PyAny>)> {
    if let Ok(tuple) = pair.cast::<PyTuple>() {
        if tuple.len() == 2 {
            let key: String = tuple.get_item(0)?.extract()?;
            return Ok((key, tuple.get_item(1)?));
        }
    }
    if let Ok(seq) = pair.cast::<PySequence>() {
        if seq.len()? == 2 {
            let key: String = seq.get_item(0)?.extract()?;
            return Ok((key, seq.get_item(1)?));
        }
    }
    Err(PyTypeError::new_err(
        "meta items must be (str, value) pairs",
    ))
}

/// Write-through view of a frame's metadata. Reads unwrap to Python scalars;
/// writes coerce bool/int/float/str (and length-3/6/9 sequences) into the
/// existing [`MetaValue`] set. No new payload types.
#[pyclass(module = "molrs", name = "FrameMeta", mapping, unsendable)]
pub struct PyFrameMeta {
    inner: FrameRef,
}

impl PyFrameMeta {
    fn clone_map(&self) -> PyResult<MetaMap> {
        self.inner
            .with(|f| f.meta.clone())
            .map_err(ffi_error_to_pyerr)
    }

    fn insert_value(&mut self, key: String, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let typed = py_to_meta_value(value)?;
        self.inner
            .with_mut(|f| {
                f.meta.insert(key, typed);
            })
            .map_err(ffi_error_to_pyerr)
    }

    fn as_pydict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (key, value) in self.clone_map()? {
            dict.set_item(key, meta_value_to_py(py, &value)?)?;
        }
        Ok(dict)
    }

    fn absorb_pairs(&mut self, iterable: &Bound<'_, PyAny>) -> PyResult<()> {
        for pair in iterable.try_iter()? {
            let pair = pair?;
            let (key, raw) = pair_to_entry(&pair)?;
            self.insert_value(key, &raw)?;
        }
        Ok(())
    }

    fn absorb_mapping(&mut self, mapping: &Bound<'_, PyAny>) -> PyResult<()> {
        if let Ok(meta) = mapping.extract::<PyRef<'_, PyFrameMeta>>() {
            let extra = meta.clone_map()?;
            return self
                .inner
                .with_mut(|f| {
                    for (key, value) in extra {
                        f.meta.insert(key, value);
                    }
                })
                .map_err(ffi_error_to_pyerr);
        }
        let items = mapping.call_method0("items").map_err(|_| {
            PyTypeError::new_err("update() argument must be a mapping or iterable of pairs")
        })?;
        self.absorb_pairs(&items)
    }

    fn eq_mapping(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let py = other.py();
        let left = self.as_pydict(py)?;
        if let Ok(meta) = other.extract::<PyRef<'_, PyFrameMeta>>() {
            return Ok(left.eq(meta.as_pydict(py)?)?);
        }
        if let Ok(dict) = other.cast::<PyDict>() {
            return Ok(left.eq(dict)?);
        }
        if other.hasattr("items")? {
            let right = PyDict::new(py);
            right.call_method1("update", (other,))?;
            return Ok(left.eq(&right)?);
        }
        Ok(false)
    }
}

#[pymethods]
impl PyFrameMeta {
    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        let value = self
            .inner
            .with(|f| f.meta.get(key).cloned())
            .map_err(ffi_error_to_pyerr)?;
        match value {
            Some(value) => meta_value_to_py(py, &value),
            None => Err(PyKeyError::new_err(key.to_string())),
        }
    }

    fn __setitem__(&mut self, key: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.insert_value(key.to_owned(), value)
    }

    fn __delitem__(&mut self, key: &str) -> PyResult<()> {
        let removed = self
            .inner
            .with_mut(|f| f.meta.remove(key))
            .map_err(ffi_error_to_pyerr)?;
        if removed.is_none() {
            return Err(PyKeyError::new_err(key.to_string()));
        }
        Ok(())
    }

    fn __contains__(&self, key: &str) -> PyResult<bool> {
        self.inner
            .with(|f| f.meta.contains_key(key))
            .map_err(ffi_error_to_pyerr)
    }

    fn __len__(&self) -> PyResult<usize> {
        self.inner
            .with(|f| f.meta.len())
            .map_err(ffi_error_to_pyerr)
    }

    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyIterator>> {
        let keys = self.keys()?;
        PyList::new(py, keys)?.try_iter()
    }

    fn keys(&self) -> PyResult<Vec<String>> {
        self.inner
            .with(|f| f.meta.keys().cloned().collect())
            .map_err(ffi_error_to_pyerr)
    }

    fn values(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        let meta = self.clone_map()?;
        meta.values()
            .map(|value| meta_value_to_py(py, value))
            .collect()
    }

    fn items(&self, py: Python<'_>) -> PyResult<Vec<(String, Py<PyAny>)>> {
        let meta = self.clone_map()?;
        meta.into_iter()
            .map(|(key, value)| Ok((key, meta_value_to_py(py, &value)?)))
            .collect()
    }

    #[pyo3(signature = (key, default=None))]
    fn get(&self, py: Python<'_>, key: &str, default: Option<Py<PyAny>>) -> PyResult<Py<PyAny>> {
        let value = self
            .inner
            .with(|f| f.meta.get(key).cloned())
            .map_err(ffi_error_to_pyerr)?;
        match value {
            Some(value) => meta_value_to_py(py, &value),
            None => Ok(default.unwrap_or_else(|| py.None())),
        }
    }

    #[pyo3(signature = (key, *default))]
    fn pop(
        &mut self,
        py: Python<'_>,
        key: &str,
        default: &Bound<'_, PyTuple>,
    ) -> PyResult<Py<PyAny>> {
        let removed = self
            .inner
            .with_mut(|f| f.meta.remove(key))
            .map_err(ffi_error_to_pyerr)?;
        match removed {
            Some(value) => meta_value_to_py(py, &value),
            None if default.is_empty() => Err(PyKeyError::new_err(key.to_string())),
            None => Ok(default.get_item(0)?.unbind()),
        }
    }

    fn popitem(&mut self, py: Python<'_>) -> PyResult<(String, Py<PyAny>)> {
        let pair = self
            .inner
            .with_mut(|f| {
                let key = f.meta.keys().next().cloned();
                key.and_then(|key| f.meta.remove(&key).map(|value| (key, value)))
            })
            .map_err(ffi_error_to_pyerr)?;
        match pair {
            Some((key, value)) => Ok((key, meta_value_to_py(py, &value)?)),
            None => Err(PyKeyError::new_err("popitem(): metadata is empty")),
        }
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
        default: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        if let Some(value) = self
            .inner
            .with(|f| f.meta.get(key).cloned())
            .map_err(ffi_error_to_pyerr)?
        {
            return meta_value_to_py(py, &value);
        }
        let default = default
            .ok_or_else(|| PyTypeError::new_err("setdefault() default must be a metadata value"))?;
        self.insert_value(key.to_owned(), default)?;
        Ok(default.clone().unbind())
    }

    #[pyo3(signature = (other=None, **kwargs))]
    fn update(
        &mut self,
        other: Option<&Bound<'_, PyAny>>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        if let Some(other) = other {
            if other.extract::<PyRef<'_, PyFrameMeta>>().is_ok() || other.hasattr("keys")? {
                self.absorb_mapping(other)?;
            } else {
                self.absorb_pairs(other)?;
            }
        }
        if let Some(kwargs) = kwargs {
            self.absorb_mapping(kwargs.as_any())?;
        }
        Ok(())
    }

    fn copy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.as_pydict(py)
    }

    fn __or__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let out = self.as_pydict(py)?;
        out.call_method1("update", (other,))?;
        Ok(out)
    }

    fn __ror__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        out.call_method1("update", (other,))?;
        out.call_method1("update", (self.as_pydict(py)?,))?;
        Ok(out)
    }

    fn __ior__(&mut self, other: &Bound<'_, PyAny>) -> PyResult<()> {
        self.update(Some(other), None)
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.eq_mapping(other)
    }

    fn __ne__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(!self.eq_mapping(other)?)
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(self.as_pydict(py)?.repr()?.to_string())
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
    /// Column values use the same accepted types as :meth:`Block.insert`.
    /// Metadata values may be Python scalars or :class:`MetaValue`.
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

    /// Write-through metadata mapping. Reads unwrap to Python scalars;
    /// writes coerce bool/int/float/str (or an explicit :class:`MetaValue`).
    ///
    /// Returns
    /// -------
    /// FrameMeta
    ///     Live view of this frame's metadata. Mutations persist.
    #[getter]
    fn meta(&self) -> PyFrameMeta {
        PyFrameMeta {
            inner: self.inner.clone(),
        }
    }

    /// Replace the metadata mapping.
    ///
    /// Parameters
    /// ----------
    /// meta : mapping
    ///     New metadata. Values may be Python scalars, length-3/6/9
    ///     sequences, or :class:`MetaValue` (exact dtype). Assigning
    ///     another :class:`FrameMeta` copies typed values as-is.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If a key is not ``str`` or a value cannot be stored.
    #[setter]
    fn set_meta(&mut self, meta: &Bound<'_, PyAny>) -> PyResult<()> {
        let map = mapping_to_meta_map(meta)?;
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
