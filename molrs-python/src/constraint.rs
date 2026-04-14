//! Python wrappers for molecular packing restraints.
//!
//! Each concrete restraint (``InsideBox``, ``InsideSphere``, ``OutsideSphere``,
//! ``AbovePlane``, ``BelowPlane``) can be composed via ``.and_()`` to build a
//! ``MoleculeConstraint`` — a bundle of restraints applied together.
//!
//! When a ``MoleculeConstraint`` (or a single restraint) is passed to
//! ``Target.with_constraint`` / ``with_constraint_for_atoms``, every restraint
//! in the bundle is attached to the target independently.

use crate::helpers::NpF;
use molrs_pack::restraint::Restraint;
use molrs_pack::restraint::{
    AbovePlaneRestraint, BelowPlaneRestraint, InsideBoxRestraint, InsideSphereRestraint,
    OutsideSphereRestraint,
};
use pyo3::prelude::*;

/// Concrete enum over the built-in restraints. Implements `Restraint` via
/// dispatch so callers can hand a single value into `Target::with_restraint`.
#[derive(Clone, Debug)]
pub(crate) enum AnyRestraint {
    InsideBox(InsideBoxRestraint),
    InsideSphere(InsideSphereRestraint),
    OutsideSphere(OutsideSphereRestraint),
    AbovePlane(AbovePlaneRestraint),
    BelowPlane(BelowPlaneRestraint),
}

impl Restraint for AnyRestraint {
    fn f(
        &self,
        x: &[molrs_pack::F; 3],
        scale: molrs_pack::F,
        scale2: molrs_pack::F,
    ) -> molrs_pack::F {
        match self {
            Self::InsideBox(r) => r.f(x, scale, scale2),
            Self::InsideSphere(r) => r.f(x, scale, scale2),
            Self::OutsideSphere(r) => r.f(x, scale, scale2),
            Self::AbovePlane(r) => r.f(x, scale, scale2),
            Self::BelowPlane(r) => r.f(x, scale, scale2),
        }
    }
    fn fg(
        &self,
        x: &[molrs_pack::F; 3],
        scale: molrs_pack::F,
        scale2: molrs_pack::F,
        g: &mut [molrs_pack::F; 3],
    ) -> molrs_pack::F {
        match self {
            Self::InsideBox(r) => r.fg(x, scale, scale2, g),
            Self::InsideSphere(r) => r.fg(x, scale, scale2, g),
            Self::OutsideSphere(r) => r.fg(x, scale, scale2, g),
            Self::AbovePlane(r) => r.fg(x, scale, scale2, g),
            Self::BelowPlane(r) => r.fg(x, scale, scale2, g),
        }
    }
}

fn extract_single(obj: &Bound<'_, pyo3::types::PyAny>) -> PyResult<AnyRestraint> {
    if let Ok(c) = obj.extract::<PyInsideBox>() {
        return Ok(AnyRestraint::InsideBox(c.inner));
    }
    if let Ok(c) = obj.extract::<PyInsideSphere>() {
        return Ok(AnyRestraint::InsideSphere(c.inner));
    }
    if let Ok(c) = obj.extract::<PyOutsideSphere>() {
        return Ok(AnyRestraint::OutsideSphere(c.inner));
    }
    if let Ok(c) = obj.extract::<PyAbovePlane>() {
        return Ok(AnyRestraint::AbovePlane(c.inner));
    }
    if let Ok(c) = obj.extract::<PyBelowPlane>() {
        return Ok(AnyRestraint::BelowPlane(c.inner));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "expected a restraint (InsideBox, InsideSphere, OutsideSphere, AbovePlane, BelowPlane, MoleculeConstraint)",
    ))
}

/// Extract a flat list of restraints from any supported Python object.
///
/// Single restraints become a one-element list; a `MoleculeConstraint` is
/// unwrapped to its contained restraints.
pub(crate) fn extract_restraints(
    obj: &Bound<'_, pyo3::types::PyAny>,
) -> PyResult<Vec<AnyRestraint>> {
    if let Ok(mc) = obj.extract::<PyMoleculeConstraint>() {
        return Ok(mc.restraints);
    }
    Ok(vec![extract_single(obj)?])
}

// ---------------------------------------------------------------------------
// Concrete restraint wrappers
// ---------------------------------------------------------------------------

macro_rules! restraint_and {
    ($self_:expr, $other:expr) => {{
        let mut rs = vec![AnyRestraint::from($self_.inner.clone())];
        rs.extend(extract_restraints($other)?);
        Ok(PyMoleculeConstraint { restraints: rs })
    }};
}

/// Box restraint: atoms penalized outside an axis-aligned bounding box.
#[pyclass(name = "InsideBox", from_py_object)]
#[derive(Clone)]
pub struct PyInsideBox {
    pub(crate) inner: InsideBoxRestraint,
}

#[pymethods]
impl PyInsideBox {
    #[new]
    fn new(min: [NpF; 3], max: [NpF; 3]) -> Self {
        Self {
            inner: InsideBoxRestraint::new(min, max),
        }
    }

    #[pyo3(name = "and_")]
    fn and_(
        &self,
        other: &Bound<'_, pyo3::types::PyAny>,
    ) -> PyResult<PyMoleculeConstraint> {
        restraint_and!(self, other)
    }

    fn __repr__(&self) -> String {
        "InsideBox(...)".to_string()
    }
}

impl From<InsideBoxRestraint> for AnyRestraint {
    fn from(r: InsideBoxRestraint) -> Self {
        AnyRestraint::InsideBox(r)
    }
}

/// Sphere restraint: atoms penalized outside a sphere.
#[pyclass(name = "InsideSphere", from_py_object)]
#[derive(Clone)]
pub struct PyInsideSphere {
    pub(crate) inner: InsideSphereRestraint,
}

#[pymethods]
impl PyInsideSphere {
    #[new]
    fn new(radius: NpF, center: [NpF; 3]) -> Self {
        Self {
            inner: InsideSphereRestraint::new(center, radius),
        }
    }

    #[pyo3(name = "and_")]
    fn and_(
        &self,
        other: &Bound<'_, pyo3::types::PyAny>,
    ) -> PyResult<PyMoleculeConstraint> {
        restraint_and!(self, other)
    }

    fn __repr__(&self) -> String {
        "InsideSphere(...)".to_string()
    }
}

impl From<InsideSphereRestraint> for AnyRestraint {
    fn from(r: InsideSphereRestraint) -> Self {
        AnyRestraint::InsideSphere(r)
    }
}

/// Sphere restraint: atoms penalized inside a sphere.
#[pyclass(name = "OutsideSphere", from_py_object)]
#[derive(Clone)]
pub struct PyOutsideSphere {
    pub(crate) inner: OutsideSphereRestraint,
}

#[pymethods]
impl PyOutsideSphere {
    #[new]
    fn new(radius: NpF, center: [NpF; 3]) -> Self {
        Self {
            inner: OutsideSphereRestraint::new(center, radius),
        }
    }

    #[pyo3(name = "and_")]
    fn and_(
        &self,
        other: &Bound<'_, pyo3::types::PyAny>,
    ) -> PyResult<PyMoleculeConstraint> {
        restraint_and!(self, other)
    }

    fn __repr__(&self) -> String {
        "OutsideSphere(...)".to_string()
    }
}

impl From<OutsideSphereRestraint> for AnyRestraint {
    fn from(r: OutsideSphereRestraint) -> Self {
        AnyRestraint::OutsideSphere(r)
    }
}

/// Half-space: atoms penalized below the plane `n . x >= d`.
#[pyclass(name = "AbovePlane", from_py_object)]
#[derive(Clone)]
pub struct PyAbovePlane {
    pub(crate) inner: AbovePlaneRestraint,
}

#[pymethods]
impl PyAbovePlane {
    #[new]
    fn new(normal: [NpF; 3], distance: NpF) -> Self {
        Self {
            inner: AbovePlaneRestraint::new(normal, distance),
        }
    }

    #[pyo3(name = "and_")]
    fn and_(
        &self,
        other: &Bound<'_, pyo3::types::PyAny>,
    ) -> PyResult<PyMoleculeConstraint> {
        restraint_and!(self, other)
    }

    fn __repr__(&self) -> String {
        "AbovePlane(...)".to_string()
    }
}

impl From<AbovePlaneRestraint> for AnyRestraint {
    fn from(r: AbovePlaneRestraint) -> Self {
        AnyRestraint::AbovePlane(r)
    }
}

/// Half-space: atoms penalized above the plane `n . x <= d`.
#[pyclass(name = "BelowPlane", from_py_object)]
#[derive(Clone)]
pub struct PyBelowPlane {
    pub(crate) inner: BelowPlaneRestraint,
}

#[pymethods]
impl PyBelowPlane {
    #[new]
    fn new(normal: [NpF; 3], distance: NpF) -> Self {
        Self {
            inner: BelowPlaneRestraint::new(normal, distance),
        }
    }

    #[pyo3(name = "and_")]
    fn and_(
        &self,
        other: &Bound<'_, pyo3::types::PyAny>,
    ) -> PyResult<PyMoleculeConstraint> {
        restraint_and!(self, other)
    }

    fn __repr__(&self) -> String {
        "BelowPlane(...)".to_string()
    }
}

impl From<BelowPlaneRestraint> for AnyRestraint {
    fn from(r: BelowPlaneRestraint) -> Self {
        AnyRestraint::BelowPlane(r)
    }
}

// ---------------------------------------------------------------------------
// MoleculeConstraint (composite)
// ---------------------------------------------------------------------------

/// Bundle of restraints that all apply to the same target. Built by chaining
/// `.and_()` on primitive constraints or on another ``MoleculeConstraint``.
#[pyclass(name = "MoleculeConstraint", from_py_object)]
#[derive(Clone)]
pub struct PyMoleculeConstraint {
    pub(crate) restraints: Vec<AnyRestraint>,
}

#[pymethods]
impl PyMoleculeConstraint {
    #[pyo3(name = "and_")]
    fn and_(
        &self,
        other: &Bound<'_, pyo3::types::PyAny>,
    ) -> PyResult<PyMoleculeConstraint> {
        let mut rs = self.restraints.clone();
        rs.extend(extract_restraints(other)?);
        Ok(PyMoleculeConstraint { restraints: rs })
    }

    fn __repr__(&self) -> String {
        format!(
            "MoleculeConstraint(restraints={})",
            self.restraints.len()
        )
    }
}
