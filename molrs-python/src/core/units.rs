//! Python bindings for molrs' native unit engine.

use crate::error::units_error;
use molrs::units::{Dimension, Quantity, Unit, UnitDef, UnitPreset, UnitRegistry, lookup_preset};
use pyo3::exceptions::{PyAttributeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

#[pyclass(module = "molrs", name = "Unit", frozen, skip_from_py_object)]
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
    #[new]
    fn py_new(factor: f64, offset: f64, dimension: [i32; 7], name: String) -> Self {
        Self::new(Unit::from_parts(
            factor,
            offset,
            Dimension::from_exponents(dimension),
            name,
        ))
    }

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

    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, pyo3::types::PyTuple>)> {
        let unit = &slf.borrow().inner;
        crate::helpers::reduce_via_type(
            slf.as_any(),
            (
                unit.factor(),
                unit.offset(),
                unit.dimension().exponents(),
                unit.name().to_owned(),
            ),
        )
    }
}

#[pyclass(module = "molrs", name = "Quantity", frozen, skip_from_py_object)]
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
    #[new]
    fn py_new(magnitude: f64, unit: &PyUnit) -> Self {
        Self::new(Quantity::new(magnitude, unit.inner.clone()))
    }

    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, pyo3::types::PyTuple>)> {
        let this = slf.borrow();
        crate::helpers::reduce_via_type(slf.as_any(), (this.magnitude(), this.unit()))
    }

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

/// One unit definition the way Python hands it over — the positional shape of
/// [`UnitDef`]: `(name, aliases, symbol, factor, offset, dimension, prefixable)`.
type UnitDefTuple = (String, Vec<String>, String, f64, f64, [i32; 7], bool);

#[pyclass(module = "molrs", name = "UnitRegistry", subclass, dict)]
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
    #[pyo3(signature = (definitions=None, *, empty=false))]
    fn new(definitions: Option<Vec<UnitDefTuple>>, empty: bool) -> PyResult<Self> {
        if let Some(definitions) = definitions {
            let definitions = definitions
                .into_iter()
                .map(
                    |(name, aliases, symbol, factor, offset, dimension, prefixable)| UnitDef {
                        name,
                        aliases,
                        symbol,
                        factor,
                        offset,
                        dimension: Dimension::from_exponents(dimension),
                        prefixable,
                    },
                )
                .collect();
            return UnitRegistry::from_definitions(definitions)
                .map(|inner| Self { inner })
                .map_err(units_error);
        }
        Ok(Self {
            inner: if empty {
                UnitRegistry::empty()
            } else {
                UnitRegistry::new()
            },
        })
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
    #[allow(clippy::too_many_arguments, reason = "Public Python keyword arguments")]
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

    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>, Bound<'py, PyAny>)> {
        let definitions: Vec<_> = slf
            .borrow()
            .inner
            .definitions()
            .map(|definition| {
                (
                    definition.name.clone(),
                    definition.aliases.clone(),
                    definition.symbol.clone(),
                    definition.factor,
                    definition.offset,
                    definition.dimension.exponents(),
                    definition.prefixable,
                )
            })
            .collect();
        Ok((
            slf.get_type().into_any(),
            PyTuple::new(slf.py(), [definitions])?,
            slf.getattr("__dict__")?,
        ))
    }
}

/// Named unit-system view (`"real"`, `"metal"`, …). Constants live in core;
/// this is the Python spelling of `molrs::units::UnitPreset`.
#[pyclass(module = "molrs", name = "UnitPreset", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyUnitPreset {
    inner: UnitPreset,
}

#[pymethods]
impl PyUnitPreset {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        lookup_preset(name)
            .map(|inner| Self { inner })
            .ok_or_else(|| PyValueError::new_err(format!("unknown unit preset {name:?}")))
    }

    #[staticmethod]
    fn real() -> Self {
        Self {
            inner: UnitPreset::real(),
        }
    }

    #[getter]
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, pyo3::types::PyTuple>)> {
        crate::helpers::reduce_via_type(slf.as_any(), (slf.borrow().name().to_owned(),))
    }

    /// Boltzmann constant **in this preset's energy / temperature units**
    /// (`"real"` gives kcal/mol/K), not in the amu / angstrom / fs system
    /// `molrs.md` integrates in. Convert it the same way you convert an
    /// energy, or the temperature the engine sees is off by the ratio.
    fn boltzmann(&self) -> f64 {
        self.inner.boltzmann()
    }

    /// Coulomb constant in this preset's own units — same caveat as
    /// [`boltzmann`](Self::boltzmann).
    fn coulomb(&self) -> f64 {
        self.inner.coulomb()
    }

    fn mass(&self) -> &str {
        self.inner.mass()
    }
    fn length(&self) -> &str {
        self.inner.length()
    }
    fn time(&self) -> &str {
        self.inner.time()
    }
    fn energy(&self) -> &str {
        self.inner.energy()
    }
    fn temperature(&self) -> &str {
        self.inner.temperature()
    }
    fn charge(&self) -> &str {
        self.inner.charge()
    }
    fn pressure(&self) -> &str {
        self.inner.pressure()
    }
    fn velocity(&self) -> &str {
        self.inner.velocity()
    }
    fn force(&self) -> &str {
        self.inner.force()
    }
    fn density(&self) -> &str {
        self.inner.density()
    }

    fn __repr__(&self) -> String {
        format!("UnitPreset({:?})", self.inner.name())
    }
}
