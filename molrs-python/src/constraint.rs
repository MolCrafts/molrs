//! Python wrappers for molecular packing constraints.
//!
//! Constraints restrict where molecules may be placed during packing. They
//! can be composed with :meth:`and_` to build compound constraints.
//!
//! | Constraint      | Condition               | Parameters              |
//! |-----------------|-------------------------|-------------------------|
//! | `InsideBox`     | Atoms within AABB       | ``min``, ``max`` corners|
//! | `InsideSphere`  | Atoms within sphere     | ``radius``, ``center``  |
//! | `OutsideSphere` | Atoms outside sphere    | ``radius``, ``center``  |
//! | `AbovePlane`    | ``n . x >= d``          | ``normal``, ``distance``|
//! | `BelowPlane`    | ``n . x <= d``          | ``normal``, ``distance``|
//!
//! All length quantities are in the same unit as the input coordinates
//! (typically angstroms).

use crate::helpers::NpF;
use molrs_pack::constraint::{
    AbovePlaneConstraint, BelowPlaneConstraint, InsideBoxConstraint, InsideSphereConstraint,
    MoleculeConstraint, OutsideSphereConstraint,
};
use pyo3::prelude::*;

/// Box constraint: all atoms must lie within an axis-aligned bounding box.
///
/// Exposed to Python as `molrs.InsideBox`.
///
/// Parameters
/// ----------
/// min : list[float] | tuple[float, float, float]
///     Minimum corner ``[xmin, ymin, zmin]``.
/// max : list[float] | tuple[float, float, float]
///     Maximum corner ``[xmax, ymax, zmax]``.
///
/// Examples
/// --------
/// >>> c = InsideBox([0, 0, 0], [10, 10, 10])
/// >>> target = target.with_constraint(c)
#[pyclass(name = "InsideBox", from_py_object)]
#[derive(Clone)]
pub struct PyInsideBox {
    pub(crate) inner: InsideBoxConstraint,
}

#[pymethods]
impl PyInsideBox {
    /// Create a box constraint from minimum and maximum corner coordinates.
    ///
    /// Parameters
    /// ----------
    /// min : list[float]
    ///     ``[xmin, ymin, zmin]``.
    /// max : list[float]
    ///     ``[xmax, ymax, zmax]``.
    ///
    /// Returns
    /// -------
    /// InsideBox
    #[new]
    fn new(min: [NpF; 3], max: [NpF; 3]) -> Self {
        PyInsideBox {
            inner: InsideBoxConstraint::new(min, max),
        }
    }

    /// Compose this constraint with another, returning a new compound
    /// constraint that requires **both** to be satisfied.
    ///
    /// Parameters
    /// ----------
    /// other : InsideBox | InsideSphere | OutsideSphere | AbovePlane | BelowPlane | MoleculeConstraint
    ///     The constraint to combine with.
    ///
    /// Returns
    /// -------
    /// MoleculeConstraint
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If ``other`` is not a recognized constraint type.
    ///
    /// Examples
    /// --------
    /// >>> compound = InsideBox([0,0,0], [10,10,10]).and_(InsideSphere(5.0, [5,5,5]))
    #[pyo3(name = "and_")]
    fn and_constraint(
        &self,
        other: &Bound<'_, pyo3::types::PyAny>,
    ) -> PyResult<PyMoleculeConstraint> {
        let other_mc = extract_molecule_constraint(other)?;
        let mc: MoleculeConstraint = self.inner.clone().into();
        Ok(PyMoleculeConstraint {
            inner: mc.and(other_mc),
        })
    }

    fn __repr__(&self) -> String {
        "InsideBox(...)".to_string()
    }
}

/// Sphere constraint: all atoms must lie **inside** a sphere.
///
/// Exposed to Python as `molrs.InsideSphere`.
///
/// Parameters
/// ----------
/// radius : float
///     Sphere radius.
/// center : list[float] | tuple[float, float, float]
///     Sphere center ``[cx, cy, cz]``.
///
/// Examples
/// --------
/// >>> c = InsideSphere(5.0, [0.0, 0.0, 0.0])
#[pyclass(name = "InsideSphere", from_py_object)]
#[derive(Clone)]
pub struct PyInsideSphere {
    pub(crate) inner: InsideSphereConstraint,
}

#[pymethods]
impl PyInsideSphere {
    /// Create a sphere-interior constraint.
    ///
    /// Parameters
    /// ----------
    /// radius : float
    ///     Sphere radius.
    /// center : list[float]
    ///     ``[cx, cy, cz]``.
    ///
    /// Returns
    /// -------
    /// InsideSphere
    #[new]
    fn new(radius: NpF, center: [NpF; 3]) -> Self {
        PyInsideSphere {
            inner: InsideSphereConstraint::new(radius, center),
        }
    }

    /// Compose with another constraint (see :meth:`InsideBox.and_`).
    ///
    /// Returns
    /// -------
    /// MoleculeConstraint
    #[pyo3(name = "and_")]
    fn and_constraint(
        &self,
        other: &Bound<'_, pyo3::types::PyAny>,
    ) -> PyResult<PyMoleculeConstraint> {
        let other_mc = extract_molecule_constraint(other)?;
        let mc: MoleculeConstraint = self.inner.clone().into();
        Ok(PyMoleculeConstraint {
            inner: mc.and(other_mc),
        })
    }

    fn __repr__(&self) -> String {
        "InsideSphere(...)".to_string()
    }
}

/// Sphere constraint: all atoms must lie **outside** a sphere.
///
/// Exposed to Python as `molrs.OutsideSphere`.
///
/// Parameters
/// ----------
/// radius : float
///     Exclusion sphere radius.
/// center : list[float] | tuple[float, float, float]
///     Sphere center ``[cx, cy, cz]``.
///
/// Examples
/// --------
/// >>> c = OutsideSphere(2.0, [5.0, 5.0, 5.0])
#[pyclass(name = "OutsideSphere", from_py_object)]
#[derive(Clone)]
pub struct PyOutsideSphere {
    pub(crate) inner: OutsideSphereConstraint,
}

#[pymethods]
impl PyOutsideSphere {
    /// Create a sphere-exterior constraint.
    ///
    /// Parameters
    /// ----------
    /// radius : float
    ///     Exclusion sphere radius.
    /// center : list[float]
    ///     ``[cx, cy, cz]``.
    ///
    /// Returns
    /// -------
    /// OutsideSphere
    #[new]
    fn new(radius: NpF, center: [NpF; 3]) -> Self {
        PyOutsideSphere {
            inner: OutsideSphereConstraint::new(radius, center),
        }
    }

    /// Compose with another constraint (see :meth:`InsideBox.and_`).
    ///
    /// Returns
    /// -------
    /// MoleculeConstraint
    #[pyo3(name = "and_")]
    fn and_constraint(
        &self,
        other: &Bound<'_, pyo3::types::PyAny>,
    ) -> PyResult<PyMoleculeConstraint> {
        let other_mc = extract_molecule_constraint(other)?;
        let mc: MoleculeConstraint = self.inner.clone().into();
        Ok(PyMoleculeConstraint {
            inner: mc.and(other_mc),
        })
    }

    fn __repr__(&self) -> String {
        "OutsideSphere(...)".to_string()
    }
}

/// Half-space constraint: all atoms must satisfy ``n . x >= d``.
///
/// Exposed to Python as `molrs.AbovePlane`.
///
/// Parameters
/// ----------
/// normal : list[float] | tuple[float, float, float]
///     Plane normal vector ``[nx, ny, nz]`` (need not be unit length).
/// distance : float
///     Signed distance threshold.
///
/// Examples
/// --------
/// >>> c = AbovePlane([0, 0, 1], 5.0)  # z >= 5
#[pyclass(name = "AbovePlane", from_py_object)]
#[derive(Clone)]
pub struct PyAbovePlane {
    pub(crate) inner: AbovePlaneConstraint,
}

#[pymethods]
impl PyAbovePlane {
    /// Create an above-plane constraint.
    ///
    /// Parameters
    /// ----------
    /// normal : list[float]
    ///     ``[nx, ny, nz]``.
    /// distance : float
    ///     Signed distance from origin along normal.
    ///
    /// Returns
    /// -------
    /// AbovePlane
    #[new]
    fn new(normal: [NpF; 3], distance: NpF) -> Self {
        PyAbovePlane {
            inner: AbovePlaneConstraint::new(normal, distance),
        }
    }

    /// Compose with another constraint (see :meth:`InsideBox.and_`).
    ///
    /// Returns
    /// -------
    /// MoleculeConstraint
    #[pyo3(name = "and_")]
    fn and_constraint(
        &self,
        other: &Bound<'_, pyo3::types::PyAny>,
    ) -> PyResult<PyMoleculeConstraint> {
        let other_mc = extract_molecule_constraint(other)?;
        let mc: MoleculeConstraint = self.inner.clone().into();
        Ok(PyMoleculeConstraint {
            inner: mc.and(other_mc),
        })
    }

    fn __repr__(&self) -> String {
        "AbovePlane(...)".to_string()
    }
}

/// Half-space constraint: all atoms must satisfy ``n . x <= d``.
///
/// Exposed to Python as `molrs.BelowPlane`.
///
/// Parameters
/// ----------
/// normal : list[float] | tuple[float, float, float]
///     Plane normal vector ``[nx, ny, nz]``.
/// distance : float
///     Signed distance threshold.
///
/// Examples
/// --------
/// >>> c = BelowPlane([0, 0, 1], 15.0)  # z <= 15
#[pyclass(name = "BelowPlane", from_py_object)]
#[derive(Clone)]
pub struct PyBelowPlane {
    pub(crate) inner: BelowPlaneConstraint,
}

#[pymethods]
impl PyBelowPlane {
    /// Create a below-plane constraint.
    ///
    /// Parameters
    /// ----------
    /// normal : list[float]
    ///     ``[nx, ny, nz]``.
    /// distance : float
    ///     Signed distance from origin along normal.
    ///
    /// Returns
    /// -------
    /// BelowPlane
    #[new]
    fn new(normal: [NpF; 3], distance: NpF) -> Self {
        PyBelowPlane {
            inner: BelowPlaneConstraint::new(normal, distance),
        }
    }

    /// Compose with another constraint (see :meth:`InsideBox.and_`).
    ///
    /// Returns
    /// -------
    /// MoleculeConstraint
    #[pyo3(name = "and_")]
    fn and_constraint(
        &self,
        other: &Bound<'_, pyo3::types::PyAny>,
    ) -> PyResult<PyMoleculeConstraint> {
        let other_mc = extract_molecule_constraint(other)?;
        let mc: MoleculeConstraint = self.inner.clone().into();
        Ok(PyMoleculeConstraint {
            inner: mc.and(other_mc),
        })
    }

    fn __repr__(&self) -> String {
        "BelowPlane(...)".to_string()
    }
}

/// Composed constraint formed by combining primitive constraints via
/// :meth:`and_`.
///
/// Exposed to Python as `molrs.MoleculeConstraint`.
///
/// All constituent restraints must be satisfied simultaneously.
///
/// Examples
/// --------
/// >>> c = InsideBox([0,0,0], [10,10,10]).and_(OutsideSphere(2.0, [5,5,5]))
/// >>> c  # MoleculeConstraint(restraints=2)
#[pyclass(name = "MoleculeConstraint", from_py_object)]
#[derive(Clone)]
pub struct PyMoleculeConstraint {
    pub(crate) inner: MoleculeConstraint,
}

#[pymethods]
impl PyMoleculeConstraint {
    /// Compose with another constraint, returning a new compound
    /// constraint requiring **all** constituents.
    ///
    /// Parameters
    /// ----------
    /// other : InsideBox | InsideSphere | OutsideSphere | AbovePlane | BelowPlane | MoleculeConstraint
    ///     The constraint to add.
    ///
    /// Returns
    /// -------
    /// MoleculeConstraint
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If ``other`` is not a recognized constraint type.
    #[pyo3(name = "and_")]
    fn and_constraint(
        &self,
        other: &Bound<'_, pyo3::types::PyAny>,
    ) -> PyResult<PyMoleculeConstraint> {
        let other_mc = extract_molecule_constraint(other)?;
        Ok(PyMoleculeConstraint {
            inner: self.inner.clone().and(other_mc),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "MoleculeConstraint(restraints={})",
            self.inner.restraints.len()
        )
    }
}

/// Extract a `MoleculeConstraint` from any supported Python constraint
/// object.
///
/// Accepts `InsideBox`, `InsideSphere`, `OutsideSphere`, `AbovePlane`,
/// `BelowPlane`, or `MoleculeConstraint`.
pub(crate) fn extract_molecule_constraint(
    obj: &Bound<'_, pyo3::types::PyAny>,
) -> PyResult<MoleculeConstraint> {
    if let Ok(c) = obj.extract::<PyInsideBox>() {
        return Ok(c.inner.into());
    }
    if let Ok(c) = obj.extract::<PyInsideSphere>() {
        return Ok(c.inner.into());
    }
    if let Ok(c) = obj.extract::<PyOutsideSphere>() {
        return Ok(c.inner.into());
    }
    if let Ok(c) = obj.extract::<PyAbovePlane>() {
        return Ok(c.inner.into());
    }
    if let Ok(c) = obj.extract::<PyBelowPlane>() {
        return Ok(c.inner.into());
    }
    if let Ok(c) = obj.extract::<PyMoleculeConstraint>() {
        return Ok(c.inner);
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "expected a constraint type (InsideBox, InsideSphere, OutsideSphere, AbovePlane, BelowPlane, or MoleculeConstraint)",
    ))
}
