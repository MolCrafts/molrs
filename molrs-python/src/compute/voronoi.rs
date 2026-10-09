//! Radical (Laguerre) Voronoi tessellation and integration
//! (`molrs::compute::voronoi`).

use crate::core::simbox::PyBox;
use crate::error::py_value_err;
use molrs::compute::{
    DensityGrid, MolecularMoments, RadicalVoronoi, VoronoiCells, VoronoiDomainAnalysis,
    VoronoiIntegration, VoronoiVoidAnalysis, polarizability_finite_field,
};
use molrs::op::F;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};

// ---------------------------------------------------------------------------
// Radical (Laguerre) Voronoi tessellation + domain / void analysis
// ---------------------------------------------------------------------------

/// Per-cell radical-Voronoi tessellation result.
#[pyclass(module = "molrs.compute", name = "VoronoiCells")]
pub struct PyVoronoiCells {
    inner: VoronoiCells,
}

#[pymethods]
impl PyVoronoiCells {
    /// Per-cell volumes (Å³), one per input point.
    #[getter]
    fn volumes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.volumes)
    }
    #[getter]
    fn total_volume(&self) -> f64 {
        self.inner.total_volume()
    }
    fn __len__(&self) -> usize {
        self.inner.len()
    }
    /// Face-adjacent neighbour cell indices of cell `i` (negative = boundary).
    fn neighbors(&self, i: usize) -> Vec<i64> {
        self.inner.neighbors(i)
    }
}

/// Radical (Laguerre / power) Voronoi tessellation — native periodic builder.
#[pyclass(module = "molrs.compute", name = "RadicalVoronoi")]
pub struct PyRadicalVoronoi;

#[pymethods]
impl PyRadicalVoronoi {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Tessellate `positions` `(N, 3)` with per-point `radii` `(N,)` inside the
    /// periodic `box`.
    fn build(
        &self,
        positions: PyReadonlyArray2<'_, f64>,
        radii: PyReadonlyArray1<'_, f64>,
        box_: &Bound<'_, PyBox>,
    ) -> PyResult<PyVoronoiCells> {
        let pts = positions.as_array();
        if pts.ncols() != 3 {
            return Err(PyValueError::new_err("positions must be (N, 3)"));
        }
        let radii_slice = radii.as_slice()?;
        if radii_slice.len() != pts.nrows() {
            return Err(PyValueError::new_err(
                "len(radii) must equal len(positions)",
            ));
        }
        let simbox = &box_.borrow().inner;
        let inner = RadicalVoronoi
            .build(pts, radii_slice, simbox)
            .map_err(py_value_err)?;
        Ok(PyVoronoiCells { inner })
    }
}

/// Partition Voronoi cells into same-label face-adjacent domains.
#[pyclass(module = "molrs.compute", name = "VoronoiDomainAnalysis", frozen)]
pub struct PyVoronoiDomainAnalysis;

#[pymethods]
impl PyVoronoiDomainAnalysis {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Merge face-adjacent cells sharing a label into domains. ``labels`` has
    /// one entry per cell. Returns
    /// ``{"sizes", "count", "largest_fraction", "domain_of"}``.
    fn analyze<'py>(
        &self,
        py: Python<'py>,
        cells: &PyVoronoiCells,
        labels: Vec<i64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let r = VoronoiDomainAnalysis
            .analyze(&cells.inner, &labels)
            .map_err(py_value_err)?;
        let d = PyDict::new(py);
        d.set_item("sizes", r.sizes)?;
        d.set_item("count", r.count)?;
        d.set_item("largest_fraction", r.largest_fraction)?;
        d.set_item("domain_of", r.domain_of)?;
        Ok(d)
    }
}

/// Merge adjacent void-probe Voronoi cells into cavities.
#[pyclass(module = "molrs.compute", name = "VoronoiVoidAnalysis", frozen)]
pub struct PyVoronoiVoidAnalysis;

#[pymethods]
impl PyVoronoiVoidAnalysis {
    #[new]
    fn new() -> Self {
        Self
    }

    /// ``is_void`` is a per-cell bool mask; ``box_volume`` normalizes the void
    /// fraction. Returns
    /// ``{"cavity_volumes", "total_void_volume", "void_fraction"}``.
    fn analyze<'py>(
        &self,
        py: Python<'py>,
        cells: &PyVoronoiCells,
        is_void: Vec<bool>,
        box_volume: F,
    ) -> PyResult<Bound<'py, PyDict>> {
        let r = VoronoiVoidAnalysis
            .analyze(&cells.inner, &is_void, box_volume)
            .map_err(py_value_err)?;
        let d = PyDict::new(py);
        d.set_item("cavity_volumes", r.cavity_volumes)?;
        d.set_item("total_void_volume", r.total_void_volume)?;
        d.set_item("void_fraction", r.void_fraction)?;
        Ok(d)
    }
}

// ---------------------------------------------------------------------------
// Voronoi electron-density integration → per-molecule moments → polarizability
// ---------------------------------------------------------------------------

/// A volumetric electron density on a (generally non-orthogonal) voxel grid.
#[pyclass(module = "molrs.compute", name = "DensityGrid")]
pub struct PyDensityGrid {
    inner: DensityGrid,
}

#[pymethods]
impl PyDensityGrid {
    /// Parameters
    /// ----------
    /// origin : (3,) float array — grid origin (Å).
    /// basis : (3, 3) float array — voxel edge vectors (rows, Å).
    /// dims : (int, int, int) — voxel counts per axis.
    /// density : (D,) float array — row-major densities, `len == prod(dims)`.
    #[new]
    fn new(
        origin: PyReadonlyArray1<'_, f64>,
        basis: PyReadonlyArray2<'_, f64>,
        dims: [usize; 3],
        density: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Self> {
        let o = origin.as_slice()?;
        if o.len() != 3 {
            return Err(PyValueError::new_err("origin must have length 3"));
        }
        let b = basis.as_array();
        if b.shape() != [3, 3] {
            return Err(PyValueError::new_err("basis must be (3, 3)"));
        }
        let basis_arr = [
            [b[[0, 0]], b[[0, 1]], b[[0, 2]]],
            [b[[1, 0]], b[[1, 1]], b[[1, 2]]],
            [b[[2, 0]], b[[2, 1]], b[[2, 2]]],
        ];
        let dens = density.as_slice()?;
        let expected = dims[0] * dims[1] * dims[2];
        if dens.len() != expected {
            return Err(PyValueError::new_err(format!(
                "len(density)={} must equal prod(dims)={expected}",
                dens.len()
            )));
        }
        Ok(Self {
            inner: DensityGrid::new([o[0], o[1], o[2]], basis_arr, dims, dens.to_vec()),
        })
    }
}

/// Per-molecule electromagnetic moments for one frame.
#[pyclass(module = "molrs.compute", name = "MolecularMoments")]
pub struct PyMolecularMoments {
    inner: MolecularMoments,
}

#[pymethods]
impl PyMolecularMoments {
    /// Molecular charges `Q_m` (e), length `n_mol`.
    #[getter]
    fn charges<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.charges)
    }
    /// Molecular dipoles `μ_m` (e·Å), shape `(n_mol, 3)`.
    #[getter]
    fn dipoles<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.dipoles.clone().into_pyarray(py)
    }
    /// Per-molecule reference points (centre of nuclear charge), `(n_mol, 3)`.
    #[getter]
    fn references<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.references.clone().into_pyarray(py)
    }
}

/// Integrate an electron density over radical-Voronoi cells into per-molecule
/// charges + dipoles.
#[pyclass(module = "molrs.compute", name = "VoronoiIntegration")]
pub struct PyVoronoiIntegration;

#[pymethods]
impl PyVoronoiIntegration {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Parameters
    /// ----------
    /// positions : (N, 3) float array — generator (atom) positions.
    /// radii : (N,) float array — radical-Voronoi radii.
    /// atomic_numbers : (N,) int array — nuclear charges `Z_a`.
    /// atom_to_mol : (N,) int array — atom→molecule index in `0..n_mol`.
    /// n_mol : int — number of molecules.
    /// grid : DensityGrid — the electron density.
    /// box_ : Box — periodic cell.
    #[allow(clippy::too_many_arguments)]
    fn integrate(
        &self,
        positions: PyReadonlyArray2<'_, f64>,
        radii: PyReadonlyArray1<'_, f64>,
        atomic_numbers: PyReadonlyArray1<'_, i64>,
        atom_to_mol: PyReadonlyArray1<'_, i64>,
        n_mol: usize,
        grid: &PyDensityGrid,
        box_: &Bound<'_, PyBox>,
    ) -> PyResult<PyMolecularMoments> {
        let pts = positions.as_array();
        if pts.ncols() != 3 {
            return Err(PyValueError::new_err("positions must be (N, 3)"));
        }
        let radii_slice = radii.as_slice()?;
        let z: Vec<i32> = atomic_numbers
            .as_array()
            .iter()
            .map(|&v| v as i32)
            .collect();
        let a2m: Vec<usize> = atom_to_mol
            .as_array()
            .iter()
            .map(|&v| {
                if v < 0 {
                    Err(PyValueError::new_err(
                        "atom_to_mol indices must be non-negative",
                    ))
                } else {
                    Ok(v as usize)
                }
            })
            .collect::<PyResult<_>>()?;
        let simbox = &box_.borrow().inner;
        let inner = VoronoiIntegration
            .integrate(pts, radii_slice, &z, &a2m, n_mol, &grid.inner, simbox)
            .map_err(py_value_err)?;
        Ok(PyMolecularMoments { inner })
    }
}

/// Finite-field molecular polarizability `α` (Å³) from three moment sets at
/// field `0`, `+field`, `−field` (central difference of the dipoles).
/// Returns a `(n_mol·3, 3)` block of per-molecule 3×3 tensors stacked by row.
#[pyfunction(name = "polarizability_finite_field")]
fn polarizability_finite_field_py<'py>(
    py: Python<'py>,
    moments_zero: &PyMolecularMoments,
    plus: &PyMolecularMoments,
    minus: &PyMolecularMoments,
    field: F,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let alpha = polarizability_finite_field(&moments_zero.inner, &plus.inner, &minus.inner, field)
        .map_err(py_value_err)?;
    Ok(alpha.into_pyarray(py))
}

/// Register this domain's classes and functions.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVoronoiCells>()?;
    m.add_class::<PyRadicalVoronoi>()?;
    m.add_class::<PyDensityGrid>()?;
    m.add_class::<PyMolecularMoments>()?;
    m.add_class::<PyVoronoiIntegration>()?;
    m.add_class::<PyVoronoiDomainAnalysis>()?;
    m.add_class::<PyVoronoiVoidAnalysis>()?;
    crate::add_function(
        m,
        "molrs.compute",
        wrap_pyfunction!(polarizability_finite_field_py, m)?,
    )?;
    Ok(())
}
