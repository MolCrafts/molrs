//! Python bindings for molrs' native unit engine.

use crate::error::units_error;
use molrs::units::{Dimension, Quantity, Unit, UnitDef, UnitRegistry};
use pyo3::exceptions::{PyAttributeError, PyTypeError};
use pyo3::prelude::*;

#[pyclass(name = "Unit", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyUnit {
    inner: Unit,
}

impl PyUnit {
    fn new(inner: Unit) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyUnit {
    #[getter]
    fn dimension(&self) -> [i32; 7] {
        self.inner.dimension().exponents()
    }

    #[getter]
    fn dimensionality(&self) -> [i32; 7] {
        self.dimension()
    }

    fn is_affine(&self) -> bool {
        self.inner.is_affine()
    }

    fn factor_to(&self, other: &PyUnit) -> PyResult<f64> {
        self.inner.factor_to(&other.inner).map_err(units_error)
    }

    fn __rmul__(&self, value: f64) -> PyQuantity {
        PyQuantity::new(Quantity::new(value, self.inner.clone()))
    }

    fn __mul__(&self, value: f64) -> PyQuantity {
        self.__rmul__(value)
    }

    fn __eq__(&self, other: &PyUnit) -> bool {
        if self.inner.is_affine() || other.inner.is_affine() {
            return self.inner == other.inner;
        }
        self.inner
            .factor_to(&other.inner)
            .is_ok_and(|factor| factor == 1.0)
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!("<Unit('{}')>", self.inner)
    }
}

#[pyclass(name = "Quantity", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyQuantity {
    inner: Quantity,
}

impl PyQuantity {
    fn new(inner: Quantity) -> Self {
        Self { inner }
    }

    fn quantity_operand(value: &Bound<'_, PyAny>) -> Option<Quantity> {
        value
            .extract::<PyRef<'_, PyQuantity>>()
            .ok()
            .map(|quantity| quantity.inner.clone())
    }
}

#[pymethods]
impl PyQuantity {
    #[getter]
    fn magnitude(&self) -> f64 {
        self.inner.value()
    }

    #[getter]
    fn value(&self) -> f64 {
        self.inner.value()
    }

    #[getter]
    fn units(&self) -> PyUnit {
        PyUnit::new(self.inner.unit().clone())
    }

    #[getter]
    fn unit(&self) -> PyUnit {
        self.units()
    }

    fn to(&self, target: &Bound<'_, PyAny>) -> PyResult<Self> {
        let converted = if let Ok(unit) = target.extract::<PyRef<'_, PyUnit>>() {
            self.inner.to(&unit.inner)
        } else if let Ok(expression) = target.extract::<String>() {
            self.inner.to_parsed(&expression)
        } else {
            return Err(PyTypeError::new_err(
                "target must be a Unit or unit expression",
            ));
        };
        converted.map(Self::new).map_err(units_error)
    }

    fn to_base_units(&self) -> Self {
        Self::new(self.inner.to_base_units())
    }

    fn __add__(&self, rhs: &PyQuantity) -> PyResult<Self> {
        self.inner
            .try_add(&rhs.inner)
            .map(Self::new)
            .map_err(units_error)
    }

    fn __sub__(&self, rhs: &PyQuantity) -> PyResult<Self> {
        self.inner
            .try_sub(&rhs.inner)
            .map(Self::new)
            .map_err(units_error)
    }

    fn __mul__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Some(quantity) = Self::quantity_operand(rhs) {
            return self
                .inner
                .try_mul(&quantity)
                .map(Self::new)
                .map_err(units_error);
        }
        if let Ok(value) = rhs.extract::<f64>() {
            return Ok(Self::new(self.inner.clone() * value));
        }
        Err(PyTypeError::new_err(
            "quantity can only multiply a scalar or Quantity",
        ))
    }

    fn __rmul__(&self, lhs: f64) -> Self {
        Self::new(self.inner.clone() * lhs)
    }

    fn __truediv__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Some(quantity) = Self::quantity_operand(rhs) {
            return self
                .inner
                .try_div(&quantity)
                .map(Self::new)
                .map_err(units_error);
        }
        if let Ok(value) = rhs.extract::<f64>() {
            return Ok(Self::new(self.inner.clone() / value));
        }
        Err(PyTypeError::new_err(
            "quantity can only divide by a scalar or Quantity",
        ))
    }

    fn __neg__(&self) -> Self {
        Self::new(-self.inner.clone())
    }

    fn __eq__(&self, other: &PyQuantity) -> bool {
        self.inner == other.inner
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "<Quantity({}, '{}')>",
            self.inner.value(),
            self.inner.unit()
        )
    }
}

#[pyclass(name = "UnitRegistry", subclass, dict)]
pub struct PyUnitRegistry {
    inner: UnitRegistry,
}

impl PyUnitRegistry {
    fn parse_inner(&self, expression: &str) -> PyResult<PyUnit> {
        self.inner
            .parse(expression)
            .map(PyUnit::new)
            .map_err(units_error)
    }
}

#[pymethods]
impl PyUnitRegistry {
    #[new]
    #[pyo3(signature = (*, empty = false))]
    fn new(empty: bool) -> Self {
        Self {
            inner: if empty {
                UnitRegistry::empty()
            } else {
                UnitRegistry::new()
            },
        }
    }

    fn parse(&self, expression: &str) -> PyResult<PyUnit> {
        self.parse_inner(expression)
    }

    #[pyo3(name = "Unit")]
    fn unit(&self, expression: &str) -> PyResult<PyUnit> {
        self.parse_inner(expression)
    }

    fn quantity(&self, value: f64, expression: &str) -> PyResult<PyQuantity> {
        self.inner
            .quantity(value, expression)
            .map(PyQuantity::new)
            .map_err(units_error)
    }

    #[pyo3(name = "Quantity")]
    fn quantity_alias(&self, value: f64, expression: &str) -> PyResult<PyQuantity> {
        self.quantity(value, expression)
    }

    #[pyo3(signature = (name, factor, dimension, *, aliases = Vec::new(), symbol = None, offset = 0.0, prefixable = false))]
    fn define(
        &mut self,
        name: String,
        factor: f64,
        dimension: [i32; 7],
        aliases: Vec<String>,
        symbol: Option<String>,
        offset: f64,
        prefixable: bool,
    ) -> PyResult<()> {
        self.inner
            .define(UnitDef {
                symbol: symbol.unwrap_or_else(|| name.clone()),
                name,
                aliases,
                factor,
                offset,
                dimension: Dimension::from_exponents(dimension),
                prefixable,
            })
            .map_err(units_error)
    }

    fn define_lj_units(
        &mut self,
        mass: &PyQuantity,
        sigma: &PyQuantity,
        epsilon: &PyQuantity,
    ) -> PyResult<()> {
        self.inner
            .define_lj_units(&mass.inner, &sigma.inner, &epsilon.inner)
            .map_err(units_error)
    }

    fn __getattr__(&self, name: &str) -> PyResult<PyUnit> {
        self.parse_inner(name)
            .map_err(|_| PyAttributeError::new_err(format!("unknown unit: {name}")))
    }

    fn __repr__(&self) -> &'static str {
        "<molrs.UnitRegistry>"
    }
}
