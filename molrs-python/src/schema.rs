//! `molrs.schema` — the Frame vocabulary, inspectable from Python.
//!
//! Every value here is projected from the compiled-in Rust tables at import
//! time. Nothing is transcribed, so `molrs.schema` and
//! `molrs::store::schema` cannot describe different contracts.
//!
//! `molrs.keys` is the field-name convention projected from the same tables:
//! each constant is a :class:`Key` (not a bare ``str``).

use std::hash::{Hash, Hasher};

use pyo3::basic::CompareOp;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use molrs::store::schema;

// ── Key ──────────────────────────────────────────────────────────────────────

/// Canonical Frame / Block column name.
///
/// Projected from the Rust schema tables as ``molrs.keys.<CONST>``. Use
/// ``.key`` (or ``str(key)``) wherever an API still takes a plain string.
#[pyclass(module = "molrs.keys", name = "Key", frozen, from_py_object)]
#[derive(Clone, Copy)]
pub struct PyKey {
    name: &'static str,
}

impl PyKey {
    pub(crate) const fn new(name: &'static str) -> Self {
        Self { name }
    }

    pub(crate) fn as_str(self) -> &'static str {
        self.name
    }
}

#[pymethods]
impl PyKey {
    /// The canonical column name string (e.g. ``"x"``).
    #[getter]
    fn key(&self) -> &'static str {
        self.name
    }

    fn __str__(&self) -> &'static str {
        self.name
    }

    fn __repr__(&self) -> String {
        format!("Key({:?})", self.name)
    }

    fn __hash__(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.name.hash(&mut h);
        h.finish()
    }

    fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: CompareOp) -> PyResult<bool> {
        let other_s = if let Ok(k) = other.extract::<PyRef<'_, PyKey>>() {
            k.name.to_owned()
        } else if let Ok(s) = other.extract::<String>() {
            s
        } else {
            return Ok(matches!(op, CompareOp::Ne));
        };
        Ok(match op {
            CompareOp::Eq => self.name == other_s,
            CompareOp::Ne => self.name != other_s,
            CompareOp::Lt => self.name < other_s.as_str(),
            CompareOp::Le => self.name <= other_s.as_str(),
            CompareOp::Gt => self.name > other_s.as_str(),
            CompareOp::Ge => self.name >= other_s.as_str(),
        })
    }
}

/// Accept either a :class:`Key` or a ``str`` as a column name.
pub(crate) fn extract_column_key(ob: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(k) = ob.extract::<PyRef<'_, PyKey>>() {
        return Ok(k.name.to_owned());
    }
    if let Ok(s) = ob.extract::<String>() {
        return Ok(s);
    }
    Err(PyTypeError::new_err(
        "column key must be molrs.keys.Key or str",
    ))
}

/// One canonical column of the Frame vocabulary.
#[pyclass(module = "molrs.schema", name = "ColumnSpec", frozen, get_all)]
#[derive(Clone)]
pub struct PyColumnSpec {
    /// Canonical key as it appears in a Block.
    pub key: String,
    /// Constant name, also exported as `molrs.keys.<const_name>`.
    pub const_name: String,
    /// `"float"` | `"int"` | `"uint"` | `"bool"` | `"u8"` | `"string"`.
    pub dtype: String,
    /// `"scalar"` or `"vec(n)"`.
    pub shape: String,
    /// Unit symbol; empty when dimensionless or unit-free.
    pub unit: String,
    /// One-line meaning.
    pub doc: String,
}

#[pymethods]
impl PyColumnSpec {
    fn __repr__(&self) -> String {
        format!("ColumnSpec(key='{}', dtype='{}')", self.key, self.dtype)
    }

    /// The numpy dtype string this column maps to.
    #[getter]
    fn numpy_dtype(&self) -> &'static str {
        match self.dtype.as_str() {
            "float" => "float64",
            "int" => "int32",
            "uint" => "uint32",
            "bool" => "bool",
            "u8" => "uint8",
            _ => "str",
        }
    }
}

/// One canonical block of the Frame vocabulary.
#[pyclass(module = "molrs.schema", name = "BlockSpec", frozen, get_all)]
#[derive(Clone)]
pub struct PyBlockSpec {
    /// Canonical block name.
    pub name: String,
    /// `"node"`, `"relation(k)"`, or `"grid"`.
    pub row_kind: String,
    /// Block the endpoints index into, for relation blocks.
    pub endpoint_target: Option<String>,
    /// Endpoint column keys, in position order.
    pub endpoint_columns: Vec<String>,
    /// Columns that must be present.
    pub required: Vec<String>,
    /// Conventional but optional columns.
    pub optional: Vec<String>,
    /// Whether columns outside the vocabulary are admissible here.
    pub open: bool,
    /// One-line meaning.
    pub doc: String,
}

#[pymethods]
impl PyBlockSpec {
    fn __repr__(&self) -> String {
        format!("BlockSpec(name='{}', rows='{}')", self.name, self.row_kind)
    }
}

fn column_specs() -> Vec<PyColumnSpec> {
    schema::document()
        .columns
        .into_iter()
        .map(|c| PyColumnSpec {
            key: c.key,
            const_name: c.const_name,
            dtype: c.dtype,
            shape: c.shape,
            unit: c.unit,
            doc: c.doc,
        })
        .collect()
}

fn block_specs() -> Vec<PyBlockSpec> {
    schema::document()
        .blocks
        .into_iter()
        .map(|b| PyBlockSpec {
            name: b.name,
            row_kind: b.row_kind,
            endpoint_target: b.endpoint_target,
            endpoint_columns: b.endpoint_columns,
            required: b.required,
            optional: b.optional,
            open: b.open,
            doc: b.doc,
        })
        .collect()
}

/// Spec for a column key, or `None` if the key is unconstrained.
///
/// *key* may be a ``str`` or a :class:`molrs.keys.Key`.
#[pyfunction]
#[pyo3(name = "column")]
fn py_column(key: &Bound<'_, PyAny>) -> PyResult<Option<PyColumnSpec>> {
    let key = extract_column_key(key)?;
    Ok(column_specs().into_iter().find(|c| c.key == key))
}

/// Spec for a block name, or `None` if the block is not in the vocabulary.
#[pyfunction]
#[pyo3(name = "block")]
fn py_block(name: &str) -> Option<PyBlockSpec> {
    block_specs().into_iter().find(|b| b.name == name)
}

/// The whole vocabulary as canonical JSON — stable across runs, so two
/// releases can be diffed.
#[pyfunction]
fn to_json() -> String {
    schema::document().to_json()
}

/// The whole vocabulary as Markdown tables.
#[pyfunction]
fn to_markdown() -> String {
    schema::document().to_markdown()
}

/// Register `molrs.schema`.
pub fn register_schema(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "schema")?;
    m.add_class::<PyColumnSpec>()?;
    m.add_class::<PyBlockSpec>()?;
    m.add("columns", column_specs())?;
    m.add("blocks", block_specs())?;
    m.add("VOCAB_VERSION", schema::FRAME_VOCAB_VERSION)?;
    m.add_function(wrap_pyfunction!(py_column, &m)?)?;
    m.add_function(wrap_pyfunction!(py_block, &m)?)?;
    m.add_function(wrap_pyfunction!(to_json, &m)?)?;
    m.add_function(wrap_pyfunction!(to_markdown, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}

/// Register `molrs.keys`, projected from the same tables.
///
/// A loop, not a hand-written list: adding a column to the Rust table adds
/// `molrs.keys.<CONST>` with no edit here, so the two cannot drift.
///
/// Each constant is a :class:`Key`. Ordered groups (`COORDS`, …) are tuples of
/// :class:`Key`.
pub fn register_keys(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    use molrs::store::keys;
    let py = parent.py();
    let m = PyModule::new(py, "keys")?;
    m.add_class::<PyKey>()?;

    for spec in schema::SCHEMA_COLUMNS {
        m.add(spec.const_name, PyKey::new(spec.key))?;
    }

    fn key_tuple(names: &[&'static str]) -> Vec<PyKey> {
        names.iter().map(|n| PyKey::new(n)).collect()
    }

    m.add("COORDS", key_tuple(&keys::COORDS))?;
    m.add("VELOCITIES", key_tuple(&keys::VELOCITIES))?;
    m.add("QUAT", key_tuple(&keys::QUAT))?;
    m.add("DIPOLE", key_tuple(&keys::DIPOLE))?;
    m.add("ENDPOINTS", key_tuple(&keys::ENDPOINTS))?;

    parent.add_submodule(&m)?;
    parent.setattr("keys", &m)?;
    Ok(())
}
