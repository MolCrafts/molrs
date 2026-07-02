// PyO3 method signatures have unavoidably-complex return types
// (`Vec<(Bound<'py, PyArray…>, …)>`). Refactoring to type aliases would
// require `for<'py>` HKTs that PyO3 doesn't model.
#![allow(clippy::type_complexity)]

//! Python wrappers for the freud-ported analyzers added in the
//! `molcrafts-molrs-compute` 0.0.16 → 0.0.17 cycle.
//!
//! Keeps each analyzer to the minimum useful surface: constructor +
//! `compute(...)` returning a numpy-friendly tuple or array. Result
//! shapes mirror the freud Python API so notebooks can be ported with
//! near-mechanical renaming.
//!
//! | Python class                  | Args                                | Output                                  |
//! |-------------------------------|-------------------------------------|------------------------------------------|
//! | `Steinhardt`                  | list[NeighborList]                  | dict {ql, wl?, qlm}                      |
//! | `Nematic`                     | list[director]                       | (order, eigenvalues, director, q_tensor) |
//! | `Hexatic`                     | list[NeighborList]                  | complex ψ_k per particle                 |
//! | `SolidLiquid`                 | list[NeighborList]                  | (n_solid_bonds, is_solid)                |
//! | `ClusterProperties`           | list[ClusterResult]                 | dict of cluster scalars / tensors        |
//! | `LocalDensity`                | list[NeighborList]                  | (num_neighbors, density)                 |
//! | `GaussianDensity`             | —                                   | 3-D density grid                         |
//! | `BondOrder`                   | list[NeighborList]                  | (raw_counts, bond_order, edges)          |
//! | `StaticStructureFactorDebye`  | —                                   | (k_values, S(k))                         |
//! | `PMFTXY`                      | list[NeighborList], orientations?   | (raw_counts, density, pmf)               |

use crate::compute::{PyClusterResult, py_value_err};
use crate::helpers::{NpF, collect_frames, collect_nlists};

use molrs::compute::{
    BondOrder, ClusterProperties, Compute, GaussianDensity, Hexatic, LocalDensity, Nematic, PMFTXY,
    PMFTXYArgs, SolidLiquid, StaticStructureFactorDebye, Steinhardt,
};
use molrs::store::frame::Frame as CoreFrame;
use molrs::types::F;

use ndarray::{Array1, Array2, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

// ---------------------------------------------------------------------------
// Steinhardt
// ---------------------------------------------------------------------------

#[pyclass(name = "Steinhardt")]
pub struct PySteinhardt {
    inner: Steinhardt,
}

#[pymethods]
impl PySteinhardt {
    #[new]
    #[pyo3(signature = (l, average=false, wl=false, wl_normalize=false))]
    fn new(l: Vec<u32>, average: bool, wl: bool, wl_normalize: bool) -> PyResult<Self> {
        let inner = Steinhardt::new(&l)
            .map_err(py_value_err)?
            .with_average(average)
            .with_wl(wl)
            .with_wl_normalize(wl_normalize);
        Ok(Self { inner })
    }

    fn compute<'py>(
        &self,
        py: Python<'py>,
        frames: &Bound<'py, PyAny>,
        nlists: &Bound<'py, PyAny>,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let owned = collect_frames(frames)?;
        let refs: Vec<&CoreFrame> = owned.iter().collect();
        let nl = collect_nlists(nlists)?;
        let results = self.inner.compute(&refs, &nl).map_err(py_value_err)?;
        results
            .into_iter()
            .map(|r| {
                let d = PyDict::new(py);
                let ls: Vec<u32> = r.l.clone();
                d.set_item("l", ls)?;
                let ql: Vec<Bound<'py, PyArray1<NpF>>> =
                    r.ql.into_iter()
                        .map(|v| Array1::from_vec(v).into_pyarray(py))
                        .collect();
                d.set_item("ql", ql)?;
                if let Some(wl) = r.wl {
                    let wls: Vec<Bound<'py, PyArray1<NpF>>> = wl
                        .into_iter()
                        .map(|v| Array1::from_vec(v).into_pyarray(py))
                        .collect();
                    d.set_item("wl", wls)?;
                }
                Ok(d)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Nematic
// ---------------------------------------------------------------------------

#[pyclass(name = "Nematic")]
pub struct PyNematic {
    inner: Nematic,
}

#[pymethods]
impl PyNematic {
    #[new]
    fn new() -> Self {
        Self {
            inner: Nematic::new(),
        }
    }

    /// Returns `(order: float, eigenvalues: ndarray[3], director: ndarray[3], q_tensor: ndarray[3,3])`.
    fn compute<'py>(
        &self,
        py: Python<'py>,
        frames: &Bound<'py, PyAny>,
        directors: Vec<[NpF; 3]>,
    ) -> PyResult<(
        NpF,
        Bound<'py, PyArray1<NpF>>,
        Bound<'py, PyArray1<NpF>>,
        Bound<'py, PyArray2<NpF>>,
    )> {
        let owned = collect_frames(frames)?;
        let refs: Vec<&CoreFrame> = owned.iter().collect();
        let mut results = self
            .inner
            .compute(&refs, directors.as_slice())
            .map_err(py_value_err)?;
        let r = results.remove(0);
        let q = Array2::from_shape_vec(
            (3, 3),
            r.q_tensor.iter().flatten().copied().collect::<Vec<F>>(),
        )
        .map_err(py_value_err)?;
        Ok((
            r.order,
            Array1::from_vec(r.eigenvalues.to_vec()).into_pyarray(py),
            Array1::from_vec(r.director.to_vec()).into_pyarray(py),
            q.into_pyarray(py),
        ))
    }
}

// ---------------------------------------------------------------------------
// Hexatic
// ---------------------------------------------------------------------------

#[pyclass(name = "Hexatic")]
pub struct PyHexatic {
    inner: Hexatic,
}

#[pymethods]
impl PyHexatic {
    #[new]
    fn new(k: u32) -> PyResult<Self> {
        Ok(Self {
            inner: Hexatic::new(k).map_err(py_value_err)?,
        })
    }

    /// Returns one numpy array per frame: `complex64` per-particle ψ_k
    /// (interleaved real/imag pairs in an `(N, 2)` `float64` array).
    fn compute<'py>(
        &self,
        py: Python<'py>,
        frames: &Bound<'py, PyAny>,
        nlists: &Bound<'py, PyAny>,
    ) -> PyResult<Vec<Bound<'py, PyArray2<NpF>>>> {
        let owned = collect_frames(frames)?;
        let refs: Vec<&CoreFrame> = owned.iter().collect();
        let nl = collect_nlists(nlists)?;
        let results = self.inner.compute(&refs, &nl).map_err(py_value_err)?;
        Ok(results
            .into_iter()
            .map(|r| {
                let n = r.psi.len();
                let mut arr = Array2::<F>::zeros((n, 2));
                for (i, c) in r.psi.into_iter().enumerate() {
                    arr[[i, 0]] = c.re;
                    arr[[i, 1]] = c.im;
                }
                arr.into_pyarray(py)
            })
            .collect())
    }
}

// ---------------------------------------------------------------------------
// SolidLiquid
// ---------------------------------------------------------------------------

#[pyclass(name = "SolidLiquid")]
pub struct PySolidLiquid {
    inner: SolidLiquid,
}

#[pymethods]
impl PySolidLiquid {
    #[new]
    #[pyo3(signature = (l, q_threshold=0.7, n_threshold=6))]
    fn new(l: u32, q_threshold: NpF, n_threshold: u32) -> Self {
        let inner = SolidLiquid::new(l)
            .with_q_threshold(q_threshold)
            .with_n_threshold(n_threshold);
        Self { inner }
    }

    /// Returns `(n_solid_bonds[i] : u32 ndarray, is_solid[i] : bool list)` per frame.
    fn compute<'py>(
        &self,
        py: Python<'py>,
        frames: &Bound<'py, PyAny>,
        nlists: &Bound<'py, PyAny>,
    ) -> PyResult<Vec<(Bound<'py, PyArray1<u32>>, Vec<bool>)>> {
        let owned = collect_frames(frames)?;
        let refs: Vec<&CoreFrame> = owned.iter().collect();
        let nl = collect_nlists(nlists)?;
        let results = self.inner.compute(&refs, &nl).map_err(py_value_err)?;
        Ok(results
            .into_iter()
            .map(|r| {
                (
                    Array1::from_vec(r.n_solid_bonds).into_pyarray(py),
                    r.is_solid,
                )
            })
            .collect())
    }
}

// ---------------------------------------------------------------------------
// ClusterProperties
// ---------------------------------------------------------------------------

#[pyclass(name = "ClusterProperties")]
pub struct PyClusterProperties {
    inner: ClusterProperties,
}

#[pymethods]
impl PyClusterProperties {
    #[new]
    fn new() -> Self {
        Self {
            inner: ClusterProperties::new(),
        }
    }

    fn with_masses(&self, masses: Vec<NpF>) -> Self {
        Self {
            inner: self.inner.clone().with_masses(&masses),
        }
    }

    /// Returns a dict per frame with keys
    /// `sizes`, `centers`, `centers_of_mass`, `cluster_masses`,
    /// `gyration_tensors`, `radii_of_gyration`.
    fn compute<'py>(
        &self,
        py: Python<'py>,
        frames: &Bound<'py, PyAny>,
        clusters: Vec<PyRef<'_, PyClusterResult>>,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let owned = collect_frames(frames)?;
        let refs: Vec<&CoreFrame> = owned.iter().collect();
        let cl: Vec<_> = clusters.iter().map(|c| c.inner.clone()).collect();
        let results = self.inner.compute(&refs, &cl).map_err(py_value_err)?;
        results
            .into_iter()
            .map(|r| {
                let d = PyDict::new(py);
                d.set_item("sizes", r.sizes)?;
                d.set_item(
                    "centers",
                    Array2::from_shape_vec(
                        (r.centers.len(), 3),
                        r.centers.iter().flatten().copied().collect::<Vec<F>>(),
                    )
                    .map_err(py_value_err)?
                    .into_pyarray(py),
                )?;
                d.set_item(
                    "centers_of_mass",
                    Array2::from_shape_vec(
                        (r.centers_of_mass.len(), 3),
                        r.centers_of_mass
                            .iter()
                            .flatten()
                            .copied()
                            .collect::<Vec<F>>(),
                    )
                    .map_err(py_value_err)?
                    .into_pyarray(py),
                )?;
                d.set_item(
                    "cluster_masses",
                    Array1::from_vec(r.cluster_masses).into_pyarray(py),
                )?;
                let n = r.gyration_tensors.len();
                let mut g = Array3::<F>::zeros((n, 3, 3));
                for (c, t) in r.gyration_tensors.iter().enumerate() {
                    for a in 0..3 {
                        for b in 0..3 {
                            g[[c, a, b]] = t[a][b];
                        }
                    }
                }
                d.set_item("gyration_tensors", g.into_pyarray(py))?;
                d.set_item(
                    "radii_of_gyration",
                    Array1::from_vec(r.radii_of_gyration).into_pyarray(py),
                )?;
                Ok(d)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// LocalDensity
// ---------------------------------------------------------------------------

#[pyclass(name = "LocalDensity")]
pub struct PyLocalDensity {
    inner: LocalDensity,
}

#[pymethods]
impl PyLocalDensity {
    #[new]
    #[pyo3(signature = (r_max, diameter=0.0))]
    fn new(r_max: NpF, diameter: NpF) -> PyResult<Self> {
        let inner = LocalDensity::new(r_max)
            .map_err(py_value_err)?
            .with_diameter(diameter);
        Ok(Self { inner })
    }

    /// Returns `(num_neighbors, density)` ndarrays per frame.
    fn compute<'py>(
        &self,
        py: Python<'py>,
        frames: &Bound<'py, PyAny>,
        nlists: &Bound<'py, PyAny>,
    ) -> PyResult<Vec<(Bound<'py, PyArray1<NpF>>, Bound<'py, PyArray1<NpF>>)>> {
        let owned = collect_frames(frames)?;
        let refs: Vec<&CoreFrame> = owned.iter().collect();
        let nl = collect_nlists(nlists)?;
        let results = self.inner.compute(&refs, &nl).map_err(py_value_err)?;
        Ok(results
            .into_iter()
            .map(|r| {
                (
                    Array1::from_vec(r.num_neighbors).into_pyarray(py),
                    Array1::from_vec(r.density).into_pyarray(py),
                )
            })
            .collect())
    }
}

// ---------------------------------------------------------------------------
// GaussianDensity
// ---------------------------------------------------------------------------

#[pyclass(name = "GaussianDensity")]
pub struct PyGaussianDensity {
    inner: GaussianDensity,
}

#[pymethods]
impl PyGaussianDensity {
    #[new]
    fn new(nx: usize, ny: usize, nz: usize, sigma: NpF) -> PyResult<Self> {
        Ok(Self {
            inner: GaussianDensity::new(nx, ny, nz, sigma).map_err(py_value_err)?,
        })
    }

    /// Returns a list of 3-D density grids `(nx, ny, nz)`, one per frame.
    fn compute<'py>(
        &self,
        py: Python<'py>,
        frames: &Bound<'py, PyAny>,
    ) -> PyResult<Vec<Bound<'py, PyArray3<NpF>>>> {
        let owned = collect_frames(frames)?;
        let refs: Vec<&CoreFrame> = owned.iter().collect();
        let results = self.inner.compute(&refs, ()).map_err(py_value_err)?;
        Ok(results
            .into_iter()
            .map(|r| r.density.into_pyarray(py))
            .collect())
    }
}

// ---------------------------------------------------------------------------
// BondOrder
// ---------------------------------------------------------------------------

#[pyclass(name = "BondOrder")]
pub struct PyBondOrder {
    inner: BondOrder,
}

#[pymethods]
impl PyBondOrder {
    #[new]
    fn new(n_theta: usize, n_phi: usize) -> PyResult<Self> {
        Ok(Self {
            inner: BondOrder::new(n_theta, n_phi).map_err(py_value_err)?,
        })
    }

    /// Returns per-frame `(raw_counts, bond_order, theta_edges, phi_edges)`.
    fn compute<'py>(
        &self,
        py: Python<'py>,
        frames: &Bound<'py, PyAny>,
        nlists: &Bound<'py, PyAny>,
    ) -> PyResult<
        Vec<(
            Bound<'py, PyArray2<u64>>,
            Bound<'py, PyArray2<NpF>>,
            Bound<'py, PyArray1<NpF>>,
            Bound<'py, PyArray1<NpF>>,
        )>,
    > {
        let owned = collect_frames(frames)?;
        let refs: Vec<&CoreFrame> = owned.iter().collect();
        let nl = collect_nlists(nlists)?;
        let results = self.inner.compute(&refs, &nl).map_err(py_value_err)?;
        Ok(results
            .into_iter()
            .map(|r| {
                (
                    r.raw_counts.into_pyarray(py),
                    r.bond_order.into_pyarray(py),
                    Array1::from_vec(r.theta_edges).into_pyarray(py),
                    Array1::from_vec(r.phi_edges).into_pyarray(py),
                )
            })
            .collect())
    }
}

// ---------------------------------------------------------------------------
// StaticStructureFactorDebye
// ---------------------------------------------------------------------------

#[pyclass(name = "StaticStructureFactorDebye")]
pub struct PyStaticStructureFactorDebye {
    inner: StaticStructureFactorDebye,
}

#[pymethods]
impl PyStaticStructureFactorDebye {
    #[new]
    fn new(k_values: Vec<NpF>) -> PyResult<Self> {
        Ok(Self {
            inner: StaticStructureFactorDebye::new(&k_values).map_err(py_value_err)?,
        })
    }

    #[staticmethod]
    fn linspace(k_min: NpF, k_max: NpF, n: usize) -> PyResult<Self> {
        Ok(Self {
            inner: StaticStructureFactorDebye::linspace(k_min, k_max, n).map_err(py_value_err)?,
        })
    }

    /// Returns per-frame `(k_values, S(k), n_particles)`.
    fn compute<'py>(
        &self,
        py: Python<'py>,
        frames: &Bound<'py, PyAny>,
    ) -> PyResult<Vec<(Bound<'py, PyArray1<NpF>>, Bound<'py, PyArray1<NpF>>, usize)>> {
        let owned = collect_frames(frames)?;
        let refs: Vec<&CoreFrame> = owned.iter().collect();
        let results = self.inner.compute(&refs, ()).map_err(py_value_err)?;
        Ok(results
            .into_iter()
            .map(|r| {
                (
                    r.k_values.into_pyarray(py),
                    r.sk.into_pyarray(py),
                    r.n_particles,
                )
            })
            .collect())
    }
}

// ---------------------------------------------------------------------------
// PMFTXY
// ---------------------------------------------------------------------------

#[pyclass(name = "PMFTXY")]
pub struct PyPMFTXY {
    inner: PMFTXY,
}

#[pymethods]
impl PyPMFTXY {
    #[new]
    fn new(x_max: NpF, y_max: NpF, n_x: usize, n_y: usize) -> PyResult<Self> {
        Ok(Self {
            inner: PMFTXY::new(x_max, y_max, n_x, n_y).map_err(py_value_err)?,
        })
    }

    /// Returns per-frame `(raw_counts, density, pmf)`. `orientations` is
    /// an optional `list[list[float]]` of per-particle 2-D angles
    /// (radians) — if supplied, every bond is rotated into the query
    /// particle's local frame.
    #[pyo3(signature = (frames, nlists, orientations=None))]
    fn compute<'py>(
        &self,
        py: Python<'py>,
        frames: &Bound<'py, PyAny>,
        nlists: &Bound<'py, PyAny>,
        orientations: Option<Vec<Vec<NpF>>>,
    ) -> PyResult<
        Vec<(
            Bound<'py, PyArray2<u64>>,
            Bound<'py, PyArray2<NpF>>,
            Bound<'py, PyArray2<NpF>>,
        )>,
    > {
        let owned = collect_frames(frames)?;
        let refs: Vec<&CoreFrame> = owned.iter().collect();
        let nl = collect_nlists(nlists)?;
        let orient_ref = orientations.as_deref();
        let args = PMFTXYArgs {
            nlists: &nl,
            query_orientations: orient_ref,
        };
        let results = self.inner.compute(&refs, args).map_err(py_value_err)?;
        Ok(results
            .into_iter()
            .map(|r| {
                (
                    r.raw_counts.into_pyarray(py),
                    r.density.into_pyarray(py),
                    r.pmf.into_pyarray(py),
                )
            })
            .collect())
    }
}

// ---------------------------------------------------------------------------
// Module-level helper to register everything.
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySteinhardt>()?;
    m.add_class::<PyNematic>()?;
    m.add_class::<PyHexatic>()?;
    m.add_class::<PySolidLiquid>()?;
    m.add_class::<PyClusterProperties>()?;
    m.add_class::<PyLocalDensity>()?;
    m.add_class::<PyGaussianDensity>()?;
    m.add_class::<PyBondOrder>()?;
    m.add_class::<PyStaticStructureFactorDebye>()?;
    m.add_class::<PyPMFTXY>()?;
    Ok(())
}

// Suppress unused-import lints for collectors that may be wrapped via
// other helpers in the future.
#[allow(dead_code)]
fn _suppress(_: Bound<'_, PyList>) {}
