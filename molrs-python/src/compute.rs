//! Python wrappers for structural analysis routines.
//!
//! Three analysis tools are provided:
//!
//! | Class     | Description                                          |
//! |-----------|------------------------------------------------------|
//! | `RDF`     | Radial distribution function g(r)                    |
//! | `MSD`     | Mean squared displacement relative to a reference    |
//! | `Cluster` | Distance-based cluster analysis via BFS              |
//!
//! Each analysis produces a typed result object with numpy-array attributes.

use crate::frame::PyFrame;
use crate::helpers::NpF;
use crate::linkedcell::{PyLinkedCell, PyNeighborList};
use crate::simbox::PyBox;

use molrs_compute::{
    CenterOfMass, Cluster, ClusterCenters, ClusterResult, Compute, GyrationTensor, InertiaTensor,
    MSD, MSDResult, RDF, RDFResult, RadiusOfGyration,
};
use molrs::neighbors::NbListAlgo;
use molrs::types::F;

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayDyn, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn py_value_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

// ---------------------------------------------------------------------------
// RDF
// ---------------------------------------------------------------------------

/// Result of a radial distribution function computation.
///
/// Exposed to Python as `molrs.RDFResult`.
///
/// Attributes
/// ----------
/// bin_centers : numpy.ndarray, shape (n_bins,), dtype float
///     Radial distance at each bin center.
/// rdf : numpy.ndarray, shape (n_bins,), dtype float
///     Normalized g(r) values.
/// bin_edges : numpy.ndarray, shape (n_bins + 1,), dtype float
///     Bin edges.
/// n_r : numpy.ndarray, shape (n_bins,), dtype float
///     Raw (unnormalized) pair counts per bin.
#[pyclass(name = "RDFResult", unsendable)]
pub struct PyRDFResult {
    inner: RDFResult,
}

#[pymethods]
impl PyRDFResult {
    /// Bin center distances.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (n_bins,), dtype float
    #[getter]
    fn bin_centers<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<NpF>> {
        self.inner.bin_centers.clone().into_pyarray(py)
    }

    /// Normalized g(r) values.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (n_bins,), dtype float
    #[getter]
    fn rdf<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<NpF>> {
        self.inner.rdf.clone().into_pyarray(py)
    }

    /// Bin edge positions.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (n_bins + 1,), dtype float
    #[getter]
    fn bin_edges<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<NpF>> {
        self.inner.bin_edges.clone().into_pyarray(py)
    }

    /// Raw pair counts per bin (before normalization).
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (n_bins,), dtype float
    #[getter]
    fn n_r<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<NpF>> {
        self.inner.n_r.clone().into_pyarray(py)
    }

    /// Normalization volume in A^3.
    #[getter]
    fn volume(&self) -> NpF {
        self.inner.volume
    }

    /// Lower radial cutoff in A (lower edge of bin 0).
    #[getter]
    fn r_min(&self) -> NpF {
        self.inner.r_min
    }

    /// Number of reference points used for normalization.
    #[getter]
    fn n_points(&self) -> usize {
        self.inner.n_points
    }

    fn __repr__(&self) -> String {
        format!(
            "RDFResult(n_bins={}, n_points={})",
            self.inner.bin_centers.len(),
            self.inner.n_points,
        )
    }
}

/// Radial distribution function g(r) calculator.
///
/// Exposed to Python as `molrs.RDF`.
///
/// Parameters
/// ----------
/// n_bins : int
///     Number of histogram bins.
/// r_max : float
///     Upper radial cutoff (same unit as coordinates).
/// r_min : float, optional
///     Lower radial cutoff. Defaults to 0 (freud convention). Pairs with
///     ``d < r_min`` or ``d == 0`` are excluded.
///
/// Examples
/// --------
/// >>> rdf_calc = RDF(n_bins=100, r_max=10.0)
/// >>> # Periodic: volume comes from the box
/// >>> result = rdf_calc.compute(nlist, box)
/// >>> # Non-periodic: pass an explicit volume (A^3)
/// >>> result = rdf_calc.compute_with_volume(nlist, 1000.0)
/// >>> import matplotlib.pyplot as plt
/// >>> plt.plot(result.bin_centers, result.rdf)
#[pyclass(name = "RDF", unsendable)]
pub struct PyRDF {
    inner: RDF,
}

#[pymethods]
impl PyRDF {
    /// Create an RDF calculator.
    ///
    /// Parameters
    /// ----------
    /// n_bins : int
    ///     Number of histogram bins.
    /// r_max : float
    ///     Upper radial cutoff.
    /// r_min : float, optional
    ///     Lower radial cutoff (default 0).
    ///
    /// Returns
    /// -------
    /// RDF
    #[new]
    #[pyo3(signature = (n_bins, r_max, r_min = 0.0))]
    fn new(n_bins: usize, r_max: NpF, r_min: NpF) -> PyResult<Self> {
        let inner = RDF::new(n_bins, r_max, r_min).map_err(py_value_err)?;
        Ok(Self { inner })
    }

    /// Compute g(r) from a neighbor list and simulation box.
    ///
    /// Parameters
    /// ----------
    /// nlist : NeighborList
    ///     Pre-built neighbor list (from :class:`NeighborQuery`).
    /// box : Box
    ///     Simulation box. Its volume is used for normalization.
    ///
    /// Returns
    /// -------
    /// RDFResult
    #[pyo3(name = "compute")]
    fn compute_nlist(&self, nlist: &PyNeighborList, r#box: &PyBox) -> PyResult<PyRDFResult> {
        let result = self
            .inner
            .compute_with_volume(&nlist.inner, r#box.inner.volume())
            .map_err(py_value_err)?;
        Ok(PyRDFResult { inner: result })
    }

    /// Compute g(r) from a neighbor list with an explicit normalization volume.
    ///
    /// Use this for non-periodic systems or to override the box volume.
    ///
    /// Parameters
    /// ----------
    /// nlist : NeighborList
    ///     Pre-built neighbor list.
    /// volume : float
    ///     Normalization volume in A^3 (must be finite and > 0).
    ///
    /// Returns
    /// -------
    /// RDFResult
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If volume is non-finite or non-positive.
    #[pyo3(name = "compute_with_volume")]
    fn compute_with_volume(
        &self,
        nlist: &PyNeighborList,
        volume: NpF,
    ) -> PyResult<PyRDFResult> {
        let result = self
            .inner
            .compute_with_volume(&nlist.inner, volume)
            .map_err(py_value_err)?;
        Ok(PyRDFResult { inner: result })
    }

    /// Compute g(r) from a Frame and a LinkedCell.
    ///
    /// The frame must have an ``"atoms"`` block with ``x``, ``y``, ``z``
    /// columns and a :attr:`~Frame.simbox`.
    ///
    /// Parameters
    /// ----------
    /// frame : Frame
    ///     Molecular data with positions and a simulation box.
    /// linked_cell : LinkedCell
    ///     Pre-built neighbor list with ``cutoff >= r_max``.
    ///
    /// Returns
    /// -------
    /// RDFResult
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the frame has no simulation box or required columns are missing.
    #[pyo3(name = "compute_from_frame")]
    fn compute_frame(&self, frame: &PyFrame, linked_cell: &PyLinkedCell) -> PyResult<PyRDFResult> {
        let core_frame = frame.clone_core_frame()?;
        let neighbors = linked_cell.inner.query();
        let result = self
            .inner
            .compute(&core_frame, neighbors)
            .map_err(py_value_err)?;
        Ok(PyRDFResult { inner: result })
    }

    fn __repr__(&self) -> String {
        "RDF(...)".to_string()
    }
}

// ---------------------------------------------------------------------------
// MSD
// ---------------------------------------------------------------------------

/// Result of a mean squared displacement computation.
///
/// Exposed to Python as `molrs.MSDResult`.
///
/// Attributes
/// ----------
/// mean : float
///     System-average MSD in length_unit^2 (e.g. angstrom^2).
/// per_particle : numpy.ndarray, shape (N,), dtype float
///     Per-particle squared displacement from the reference positions.
#[pyclass(name = "MSDResult", unsendable)]
pub struct PyMSDResult {
    inner: MSDResult,
}

#[pymethods]
impl PyMSDResult {
    /// System-average mean squared displacement.
    ///
    /// Returns
    /// -------
    /// float
    ///     MSD in length_unit^2.
    #[getter]
    fn mean(&self) -> f64 {
        self.inner.mean as f64
    }

    /// Per-particle squared displacement from the reference.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (N,), dtype float
    #[getter]
    fn per_particle<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<NpF>> {
        self.inner.per_particle.clone().into_pyarray(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "MSDResult(mean={:.4}, n_particles={})",
            self.inner.mean,
            self.inner.per_particle.len(),
        )
    }
}

/// Mean squared displacement (MSD) calculator relative to a reference frame.
///
/// Exposed to Python as `molrs.MSD`.
///
/// The reference positions are captured at construction time. Call
/// :meth:`compute` with subsequent frames to measure displacement.
///
/// Parameters
/// ----------
/// ref_frame : Frame
///     Reference frame with an ``"atoms"`` block containing ``x``, ``y``,
///     ``z`` columns. Positions define r_ref.
///
/// Examples
/// --------
/// >>> msd_calc = MSD(ref_frame)
/// >>> result = msd_calc.compute(current_frame)
/// >>> result.mean   # average squared displacement
/// 12.34
#[pyclass(name = "MSD", unsendable)]
pub struct PyMSD {
    inner: MSD,
}

#[pymethods]
impl PyMSD {
    /// Create an MSD calculator from a reference frame.
    ///
    /// Parameters
    /// ----------
    /// ref_frame : Frame
    ///     Frame whose ``"atoms"`` block defines the reference positions.
    ///
    /// Returns
    /// -------
    /// MSD
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the frame lacks an ``"atoms"`` block or ``x``/``y``/``z``
    ///     columns.
    #[new]
    #[allow(deprecated)]
    fn new(ref_frame: &PyFrame) -> PyResult<Self> {
        let core_frame = ref_frame.clone_core_frame()?;
        let msd =
            MSD::from_reference(&core_frame).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner: msd })
    }

    /// Compute MSD for a frame relative to the reference.
    ///
    /// Parameters
    /// ----------
    /// frame : Frame
    ///     Frame with an ``"atoms"`` block containing ``x``, ``y``, ``z``
    ///     columns. Must have the same number of atoms as the reference.
    ///
    /// Returns
    /// -------
    /// MSDResult
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the atom count does not match the reference or required columns
    ///     are missing.
    fn compute(&self, frame: &PyFrame) -> PyResult<PyMSDResult> {
        let core_frame = frame.clone_core_frame()?;
        let result = self
            .inner
            .compute(&core_frame, ())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyMSDResult { inner: result })
    }

    fn __repr__(&self) -> String {
        "MSD(...)".to_string()
    }
}

// ---------------------------------------------------------------------------
// Cluster
// ---------------------------------------------------------------------------

/// Result of a distance-based cluster analysis.
///
/// Exposed to Python as `molrs.ClusterResult`.
///
/// Attributes
/// ----------
/// num_clusters : int
///     Number of clusters found (after ``min_cluster_size`` filtering).
/// cluster_idx : numpy.ndarray, shape (N,), dtype int64
///     Per-particle cluster ID. Particles in clusters smaller than
///     ``min_cluster_size`` are labeled ``-1``.
/// cluster_sizes : list[int]
///     Number of particles in each cluster, indexed by cluster ID.
#[pyclass(name = "ClusterResult", unsendable)]
pub struct PyClusterResult {
    inner: ClusterResult,
}

#[pymethods]
impl PyClusterResult {
    /// Number of clusters found.
    ///
    /// Returns
    /// -------
    /// int
    #[getter]
    fn num_clusters(&self) -> usize {
        self.inner.num_clusters
    }

    /// Per-particle cluster ID (``-1`` for unassigned).
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (N,), dtype int64
    #[getter]
    fn cluster_idx<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        self.inner.cluster_idx.clone().into_pyarray(py)
    }

    /// Size of each cluster, indexed by cluster ID.
    ///
    /// Returns
    /// -------
    /// list[int]
    #[getter]
    fn cluster_sizes(&self) -> Vec<usize> {
        self.inner.cluster_sizes.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "ClusterResult(num_clusters={}, largest={})",
            self.inner.num_clusters,
            self.inner.cluster_sizes.iter().max().unwrap_or(&0),
        )
    }
}

/// Distance-based cluster analysis using BFS on the neighbor graph.
///
/// Exposed to Python as `molrs.Cluster`.
///
/// Two atoms belong to the same cluster if they are connected by a chain of
/// neighbor-list edges.
///
/// Parameters
/// ----------
/// min_cluster_size : int
///     Clusters smaller than this are relabeled to ``-1``.
///
/// Examples
/// --------
/// >>> cluster_calc = Cluster(min_cluster_size=5)
/// >>> result = cluster_calc.compute(frame, nlist)
/// >>> result.num_clusters
/// 3
#[pyclass(name = "Cluster", unsendable)]
pub struct PyCluster {
    inner: Cluster,
}

#[pymethods]
impl PyCluster {
    /// Create a cluster analysis calculator.
    ///
    /// Parameters
    /// ----------
    /// min_cluster_size : int
    ///     Minimum number of particles for a cluster to be retained.
    ///
    /// Returns
    /// -------
    /// Cluster
    #[new]
    fn new(min_cluster_size: usize) -> Self {
        Self {
            inner: Cluster::new(min_cluster_size),
        }
    }

    /// Compute cluster assignment from a NeighborList (freud-style API).
    ///
    /// Parameters
    /// ----------
    /// frame : Frame
    ///     Frame with an ``"atoms"`` block.
    /// nlist : NeighborList
    ///     Pre-built neighbor list defining connectivity.
    ///
    /// Returns
    /// -------
    /// ClusterResult
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the frame lacks required data.
    #[pyo3(name = "compute")]
    fn compute_nlist(&self, frame: &PyFrame, nlist: &PyNeighborList) -> PyResult<PyClusterResult> {
        let core_frame = frame.clone_core_frame()?;
        let result = self
            .inner
            .compute(&core_frame, &nlist.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyClusterResult { inner: result })
    }

    /// Compute cluster assignment from a LinkedCell (legacy API).
    ///
    /// Parameters
    /// ----------
    /// frame : Frame
    ///     Frame with an ``"atoms"`` block.
    /// linked_cell : LinkedCell
    ///     Pre-built linked-cell neighbor list.
    ///
    /// Returns
    /// -------
    /// ClusterResult
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the frame lacks required data.
    #[pyo3(name = "compute_from_frame")]
    fn compute_frame(
        &self,
        frame: &PyFrame,
        linked_cell: &PyLinkedCell,
    ) -> PyResult<PyClusterResult> {
        let core_frame = frame.clone_core_frame()?;
        let neighbors = linked_cell.inner.query();
        let result = self
            .inner
            .compute(&core_frame, neighbors)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyClusterResult { inner: result })
    }

    fn __repr__(&self) -> String {
        "Cluster(...)".to_string()
    }
}

// ---------------------------------------------------------------------------
// ClusterCenters
// ---------------------------------------------------------------------------

/// Geometric cluster centers (MIC-aware).
///
/// Exposed to Python as `molrs.ClusterCenters`.
///
/// Examples
/// --------
/// >>> centers = molrs.ClusterCenters().compute(frame, cluster_result)
#[pyclass(name = "ClusterCenters", unsendable)]
pub struct PyClusterCenters {
    inner: ClusterCenters,
}

#[pymethods]
impl PyClusterCenters {
    #[new]
    fn new() -> Self {
        Self {
            inner: ClusterCenters::new(),
        }
    }

    /// Compute geometric centers for each cluster.
    ///
    /// Parameters
    /// ----------
    /// frame : Frame
    /// cluster_result : ClusterResult
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (num_clusters, 3)
    fn compute<'py>(
        &self,
        py: Python<'py>,
        frame: &PyFrame,
        cluster_result: &PyClusterResult,
    ) -> PyResult<Bound<'py, PyArray2<NpF>>> {
        let core_frame = frame.clone_core_frame()?;
        let centers = self
            .inner
            .compute(&core_frame, &cluster_result.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let nc = centers.len();
        let flat: Vec<NpF> = centers
            .iter()
            .flat_map(|c| [c[0] as NpF, c[1] as NpF, c[2] as NpF])
            .collect();
        Ok(ndarray::Array2::from_shape_vec((nc, 3), flat)
            .expect("centers shape")
            .into_pyarray(py))
    }

    fn __repr__(&self) -> String {
        "ClusterCenters()".to_string()
    }
}

// ---------------------------------------------------------------------------
// CenterOfMass
// ---------------------------------------------------------------------------

/// Mass-weighted cluster centers (center of mass, MIC-aware).
///
/// Exposed to Python as `molrs.CenterOfMass`.
///
/// Examples
/// --------
/// >>> com = molrs.CenterOfMass(masses=masses).compute(frame, cluster_result)
/// >>> com.centers_of_mass  # (nc, 3)
/// >>> com.cluster_masses   # (nc,)
#[pyclass(name = "CenterOfMassResult", unsendable)]
pub struct PyCenterOfMassResult {
    inner: molrs_compute::CenterOfMassResult,
}

#[pymethods]
impl PyCenterOfMassResult {
    /// Mass-weighted centers, shape (num_clusters, 3).
    #[getter]
    fn centers_of_mass<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<NpF>> {
        let nc = self.inner.centers_of_mass.len();
        let flat: Vec<NpF> = self
            .inner
            .centers_of_mass
            .iter()
            .flat_map(|c| [c[0] as NpF, c[1] as NpF, c[2] as NpF])
            .collect();
        ndarray::Array2::from_shape_vec((nc, 3), flat)
            .expect("com shape")
            .into_pyarray(py)
    }

    /// Total mass per cluster, shape (num_clusters,).
    #[getter]
    fn cluster_masses<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<NpF>> {
        let v: Vec<NpF> = self
            .inner
            .cluster_masses
            .iter()
            .map(|&m| m as NpF)
            .collect();
        ndarray::Array1::from_vec(v).into_pyarray(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "CenterOfMassResult(num_clusters={})",
            self.inner.centers_of_mass.len()
        )
    }
}

#[pyclass(name = "CenterOfMass", unsendable)]
pub struct PyCenterOfMass {
    masses: Option<Vec<F>>,
}

#[pymethods]
impl PyCenterOfMass {
    /// Create a center-of-mass calculator.
    ///
    /// Parameters
    /// ----------
    /// masses : numpy.ndarray, optional
    ///     Per-particle masses. If None, all masses = 1.0.
    #[new]
    #[pyo3(signature = (masses=None))]
    fn new(masses: Option<PyReadonlyArray1<'_, NpF>>) -> Self {
        let masses = masses.map(|m| {
            m.as_slice()
                .unwrap()
                .iter()
                .map(|&v| v as F)
                .collect::<Vec<F>>()
        });
        Self { masses }
    }

    /// Compute centers of mass for each cluster.
    ///
    /// Parameters
    /// ----------
    /// frame : Frame
    /// cluster_result : ClusterResult
    ///
    /// Returns
    /// -------
    /// CenterOfMassResult
    fn compute(
        &self,
        frame: &PyFrame,
        cluster_result: &PyClusterResult,
    ) -> PyResult<PyCenterOfMassResult> {
        let core_frame = frame.clone_core_frame()?;
        let calc = if let Some(ref ms) = self.masses {
            CenterOfMass::new().with_masses(ms)
        } else {
            CenterOfMass::new()
        };
        let result = calc
            .compute(&core_frame, &cluster_result.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyCenterOfMassResult { inner: result })
    }

    fn __repr__(&self) -> String {
        "CenterOfMass(...)".to_string()
    }
}

// ---------------------------------------------------------------------------
// GyrationTensor
// ---------------------------------------------------------------------------

/// Gyration tensor per cluster.
///
/// Exposed to Python as `molrs.GyrationTensor`.
///
/// Examples
/// --------
/// >>> tensors = molrs.GyrationTensor().compute(frame, cluster_result)
/// >>> tensors.shape  # (nc, 3, 3)
#[pyclass(name = "GyrationTensor", unsendable)]
pub struct PyGyrationTensor {
    inner: GyrationTensor,
}

#[pymethods]
impl PyGyrationTensor {
    #[new]
    fn new() -> Self {
        Self {
            inner: GyrationTensor::new(),
        }
    }

    /// Compute gyration tensors.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (num_clusters, 3, 3)
    fn compute<'py>(
        &self,
        py: Python<'py>,
        frame: &PyFrame,
        cluster_result: &PyClusterResult,
    ) -> PyResult<Bound<'py, PyArrayDyn<NpF>>> {
        let core_frame = frame.clone_core_frame()?;
        let tensors = self
            .inner
            .compute(&core_frame, &cluster_result.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let nc = tensors.len();
        let flat: Vec<NpF> = tensors
            .iter()
            .flat_map(|t| t.iter().flat_map(|row| row.iter().map(|&v| v as NpF)))
            .collect();
        Ok(ndarray::ArrayD::from_shape_vec(vec![nc, 3, 3], flat)
            .expect("gyration shape")
            .into_pyarray(py))
    }

    fn __repr__(&self) -> String {
        "GyrationTensor()".to_string()
    }
}

// ---------------------------------------------------------------------------
// InertiaTensor
// ---------------------------------------------------------------------------

/// Moment of inertia tensor per cluster.
///
/// Exposed to Python as `molrs.InertiaTensor`.
///
/// Examples
/// --------
/// >>> tensors = molrs.InertiaTensor(masses=masses).compute(frame, cluster_result)
/// >>> tensors.shape  # (nc, 3, 3)
#[pyclass(name = "InertiaTensor", unsendable)]
pub struct PyInertiaTensor {
    masses: Option<Vec<F>>,
}

#[pymethods]
impl PyInertiaTensor {
    #[new]
    #[pyo3(signature = (masses=None))]
    fn new(masses: Option<PyReadonlyArray1<'_, NpF>>) -> Self {
        let masses = masses.map(|m| {
            m.as_slice()
                .unwrap()
                .iter()
                .map(|&v| v as F)
                .collect::<Vec<F>>()
        });
        Self { masses }
    }

    /// Compute inertia tensors.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (num_clusters, 3, 3)
    fn compute<'py>(
        &self,
        py: Python<'py>,
        frame: &PyFrame,
        cluster_result: &PyClusterResult,
    ) -> PyResult<Bound<'py, PyArrayDyn<NpF>>> {
        let core_frame = frame.clone_core_frame()?;
        let calc = if let Some(ref ms) = self.masses {
            InertiaTensor::new().with_masses(ms)
        } else {
            InertiaTensor::new()
        };
        let tensors = calc
            .compute(&core_frame, &cluster_result.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let nc = tensors.len();
        let flat: Vec<NpF> = tensors
            .iter()
            .flat_map(|t| t.iter().flat_map(|row| row.iter().map(|&v| v as NpF)))
            .collect();
        Ok(ndarray::ArrayD::from_shape_vec(vec![nc, 3, 3], flat)
            .expect("inertia shape")
            .into_pyarray(py))
    }

    fn __repr__(&self) -> String {
        "InertiaTensor(...)".to_string()
    }
}

// ---------------------------------------------------------------------------
// RadiusOfGyration
// ---------------------------------------------------------------------------

/// Radius of gyration per cluster.
///
/// Exposed to Python as `molrs.RadiusOfGyration`.
///
/// Examples
/// --------
/// >>> rg = molrs.RadiusOfGyration(masses=masses).compute(frame, cluster_result)
/// >>> rg  # numpy.ndarray, shape (nc,)
#[pyclass(name = "RadiusOfGyration", unsendable)]
pub struct PyRadiusOfGyration {
    masses: Option<Vec<F>>,
}

#[pymethods]
impl PyRadiusOfGyration {
    #[new]
    #[pyo3(signature = (masses=None))]
    fn new(masses: Option<PyReadonlyArray1<'_, NpF>>) -> Self {
        let masses = masses.map(|m| {
            m.as_slice()
                .unwrap()
                .iter()
                .map(|&v| v as F)
                .collect::<Vec<F>>()
        });
        Self { masses }
    }

    /// Compute radii of gyration.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (num_clusters,)
    fn compute<'py>(
        &self,
        py: Python<'py>,
        frame: &PyFrame,
        cluster_result: &PyClusterResult,
    ) -> PyResult<Bound<'py, PyArray1<NpF>>> {
        let core_frame = frame.clone_core_frame()?;
        let calc = if let Some(ref ms) = self.masses {
            RadiusOfGyration::new().with_masses(ms)
        } else {
            RadiusOfGyration::new()
        };
        let radii = calc
            .compute(&core_frame, &cluster_result.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let v: Vec<NpF> = radii.iter().map(|&r| r as NpF).collect();
        Ok(ndarray::Array1::from_vec(v).into_pyarray(py))
    }

    fn __repr__(&self) -> String {
        "RadiusOfGyration(...)".to_string()
    }
}
