//! Python binding for the Rust-owned periodic table.

use molrs::Element;
use pyo3::exceptions::{PyKeyError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::PyType;

/// Immutable chemical-element record backed by [`Element`].
#[pyclass(module = "molrs", name = "Element", frozen, eq, hash, skip_from_py_object)]
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct PyElement {
    inner: Element,
}

impl PyElement {
    fn from_identifier(identifier: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(number) = identifier.extract::<i64>() {
            let element = u8::try_from(number)
                .ok()
                .and_then(Element::by_number)
                .ok_or_else(|| element_not_found(identifier))?;
            return Ok(Self { inner: element });
        }

        let Ok(text) = identifier.extract::<String>() else {
            return Err(PyTypeError::new_err(
                "element identifier must be a name, symbol, or atomic number",
            ));
        };
        let element = Element::by_symbol(&text)
            .or_else(|| {
                Element::ALL
                    .iter()
                    .copied()
                    .find(|element| element.name().eq_ignore_ascii_case(&text))
            })
            .ok_or_else(|| element_not_found(identifier))?;
        Ok(Self { inner: element })
    }
}

fn element_not_found(identifier: &Bound<'_, PyAny>) -> PyErr {
    let value = identifier
        .repr()
        .map(|repr| repr.to_string())
        .unwrap_or_else(|_| "<unprintable>".to_string());
    PyKeyError::new_err(format!("Element not found: {value}"))
}

#[pymethods]
impl PyElement {
    /// Look up an element by case-insensitive name/symbol or atomic number.
    #[new]
    fn new(identifier: &Bound<'_, PyAny>) -> PyResult<Self> {
        Self::from_identifier(identifier)
    }

    /// Atomic number.
    #[getter]
    fn number(&self) -> u8 {
        self.inner.z()
    }

    /// Canonical English name.
    #[getter]
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    /// Canonical chemical symbol.
    #[getter]
    fn symbol(&self) -> &'static str {
        self.inner.symbol()
    }

    /// Standard atomic mass in daltons.
    #[getter]
    fn mass(&self) -> f64 {
        self.inner.atomic_mass() as f64
    }

    /// Van der Waals radius in angstroms.
    #[getter]
    fn vdw(&self) -> f64 {
        self.inner.vdw_radius() as f64
    }

    /// Single-bond covalent radius in angstroms.
    #[getter]
    fn covalent(&self) -> f64 {
        self.inner.covalent_radius() as f64
    }

    /// Convert element identifiers to canonical symbols.
    #[classmethod]
    fn get_symbols(
        _cls: &Bound<'_, PyType>,
        identifiers: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<&'static str>> {
        identifiers
            .try_iter()?
            .map(|identifier| {
                let element = Self::from_identifier(&identifier?)?;
                Ok(element.inner.symbol())
            })
            .collect()
    }

    /// Return the atomic number for a name or symbol.
    #[classmethod]
    fn get_atomic_number(_cls: &Bound<'_, PyType>, identifier: &Bound<'_, PyAny>) -> PyResult<u8> {
        Ok(Self::from_identifier(identifier)?.inner.z())
    }

    fn __repr__(&self) -> String {
        format!("<Element {}>", self.inner.symbol())
    }
}
