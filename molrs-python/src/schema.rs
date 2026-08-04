//! `molrs.schema` — the Frame vocabulary, inspectable from Python.
//!
//! Every value here is projected from the compiled-in Rust tables at import
//! time. Nothing is transcribed, so `molrs.schema` and
//! `molrs::store::schema` cannot describe different contracts.

use pyo3::prelude::*;
use pyo3::types::PyModule;

use molrs::store::schema;

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
#[pyfunction]
#[pyo3(name = "column")]
fn py_column(key: &str) -> Option<PyColumnSpec> {
    column_specs().into_iter().find(|c| c.key == key)
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
pub fn register_keys(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    use molrs::store::keys;
    let m = PyModule::new(parent.py(), "keys")?;
    for spec in schema::SCHEMA_COLUMNS {
        m.add(spec.const_name, spec.key)?;
    }
    // Ordered groups have no single key, so they are named explicitly.
    m.add("COORDS", keys::COORDS.to_vec())?;
    m.add("VELOCITIES", keys::VELOCITIES.to_vec())?;
    m.add("QUAT", keys::QUAT.to_vec())?;
    m.add("DIPOLE", keys::DIPOLE.to_vec())?;
    m.add("ENDPOINTS", keys::ENDPOINTS.to_vec())?;
    parent.add_submodule(&m)?;
    Ok(())
}
