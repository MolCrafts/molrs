//! Python wrappers for the unified `molrs-compute` API.
//!
//! The Rust crate exposes a single [`Compute`] trait that consumes a batch of
//! frames (`&[&Frame]`) and returns one typed result per call. These Python
//! wrappers expose the same shape: every `compute(...)` method accepts either a
//! single frame or a list of frames and returns either a single result or a
//! list, matching the Rust output type.
//!
//! | Python class      | Args                                        | Output                       |
//! |-------------------|---------------------------------------------|------------------------------|
//! | `RDF`             | list[NeighborList]                          | `RDFResult` (accumulated)    |
//! | `MSD`             | —                                           | `MSDTimeSeries`              |
//! | `Cluster`         | list[NeighborList]                          | list[`ClusterResult`]        |
//! | `ClusterCenters`  | list[`ClusterResult`]                       | list[`ClusterCentersResult`] |
//! | `CenterOfMass`    | list[`ClusterResult`]                       | list[`CenterOfMassResult`]   |
//! | `GyrationTensor`  | list[`ClusterResult`], list[`ClusterCenters…`] | list[3×3 ndarray]         |
//! | `InertiaTensor`   | list[`ClusterResult`], list[`CenterOfMass…`]   | list[3×3 ndarray]         |
//! | `RadiusOfGyration`| list[`ClusterResult`], list[`CenterOfMass…`]   | list[float]               |
//! | `Pca2`            | list[descriptor]                            | `PcaResult`                  |
//! | `KMeans`          | `PcaResult`                                 | `KMeansResult`               |

use crate::frame::PyFrame;
use crate::helpers::NpF;
use crate::linkedcell::PyNeighborList;

use molrs::frame::Frame as CoreFrame;
use molrs::neighbors::NeighborList;
use molrs::types::F;
use molrs_compute::{
    CenterOfMass, COMResult, Cluster, ClusterCenters, ClusterCentersResult, ClusterResult,
    Compute, GyrationTensor, InertiaTensor, KMeans, KMeansResult, MSD, MSDResult, MSDTimeSeries,
    Pca2, PcaResult, RDF, RDFResult, RadiusOfGyration, RgResult,
};

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayDyn, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn py_value_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

// Clone owned CoreFrames out of a single PyFrame or a Python list of PyFrames.
// The returned Vec<Frame> is then wrapped into Vec<&Frame> at the call site.
fn clone_frames(frames: &Bound<'_, PyAny>) -> PyResult<Vec<CoreFrame>> {
    if let Ok(single) = frames.extract::<PyRef<'_, PyFrame>>() {
        return Ok(vec![single.clone_core_frame()?]);
    }
    let list: Vec<PyRef<'_, PyFrame>> = frames.extract()?;
    list.iter().map(|f| f.clone_core_frame()).collect()
}

// Collect &NeighborList references from a single wrapper or a list.
fn collect_nlists<'py>(
    arg: &'py Bound<'py, PyAny>,
) -> PyResult<Vec<molrs::neighbors::NeighborList>> {
    if let Ok(single) = arg.extract::<PyRef<'_, PyNeighborList>>() {
        return Ok(vec![single.inner.clone()]);
    }
    let list: Vec<PyRef<'_, PyNeighborList>> = arg.extract()?;
    Ok(list.iter().map(|n| n.inner.clone()).collect())
}

fn was_batched(frames: &Bound<'_, PyAny>) -> bool {
    frames.extract::<PyRef<'_, PyFrame>>().is_err()
}

// ---------------------------------------------------------------------------
// RDF
// ---------------------------------------------------------------------------

/// Radial distribution function g(r) result.
///
/// `rdf` is already normalized (RDF.compute finalizes eagerly).
#[pyclass(name = "RDFResult", unsendable)]
pub struct PyRDFResult {
    inner: RDFResult,
}

#[pymethods]
impl PyRDFResult {
    #[getter]
    fn bin_centers<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<NpF>> {
        self.inner.bin_centers.clone().into_pyarray(py)
    }

    #[getter]
    fn rdf<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<NpF>> {
        self.inner.rdf.clone().into_pyarray(py)
    }

    #[getter]
    fn bin_edges<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<NpF>> {
        self.inner.bin_edges.clone().into_pyarray(py)
    }

    #[getter]
    fn n_r<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<NpF>> {
        self.inner.n_r.clone().into_pyarray(py)
    }

    #[getter]
    fn volume(&self) -> NpF {
        self.inner.volume
    }

    #[getter]
    fn r_min(&self) -> NpF {
        self.inner.r_min
    }

    #[getter]
    fn n_points(&self) -> usize {
        self.inner.n_points
    }

    #[getter]
    fn n_frames(&self) -> usize {
        self.inner.n_frames
    }

    fn __repr__(&self) -> String {
        format!(
            "RDFResult(n_bins={}, n_frames={}, n_points={})",
            self.inner.bin_centers.len(),
            self.inner.n_frames,
            self.inner.n_points,
        )
    }
}

/// Radial distribution function calculator.
///
/// Accepts either a single `(frame, nlist)` pair or a list of each. Results
/// accumulate across frames and are ideal-gas normalized on return.
#[pyclass(name = "RDF", unsendable)]
pub struct PyRDF {
    inner: RDF,
}

#[pymethods]
impl PyRDF {
    #[new]
    #[pyo3(signature = (n_bins, r_max, r_min = 0.0))]
    fn new(n_bins: usize, r_max: NpF, r_min: NpF) -> PyResult<Self> {
        let inner = RDF::new(n_bins, r_max, r_min).map_err(py_value_err)?;
        Ok(Self { inner })
    }

    /// Compute g(r) from a batch of frames + neighbor lists.
    ///
    /// Parameters
    /// ----------
    /// frames : Frame | list[Frame]
    /// nlists : NeighborList | list[NeighborList]
    ///     One neighbor list per frame.
    ///
    /// Returns
    /// -------
    /// RDFResult
    fn compute(
        &self,
        frames: &Bound<'_, PyAny>,
        nlists: &Bound<'_, PyAny>,
    ) -> PyResult<PyRDFResult> {
        let owned = clone_frames(frames)?;
        let refs: Vec<&CoreFrame> = owned.iter().collect();
        let nlists_vec: Vec<NeighborList> = collect_nlists(nlists)?;
        if nlists_vec.len() != refs.len() {
            return Err(PyValueError::new_err(format!(
                "len(nlists)={} must equal len(frames)={}",
                nlists_vec.len(),
                refs.len()
            )));
        }
        let result = self
            .inner
            .compute(&refs, &nlists_vec)
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

/// Per-frame MSD result (from a single time point).
#[pyclass(name = "MSDResult", unsendable)]
pub struct PyMSDResult {
    inner: MSDResult,
}

#[pymethods]
impl PyMSDResult {
    #[getter]
    fn mean(&self) -> f64 {
        self.inner.mean as f64
    }

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

/// MSD time series aligned with the input frame list.
///
/// `series.data[0]` is the reference frame (mean = 0); `series.data[i]`
/// compares frame `i` against frame `0`.
#[pyclass(name = "MSDTimeSeries", unsendable)]
pub struct PyMSDTimeSeries {
    inner: MSDTimeSeries,
}

#[pymethods]
impl PyMSDTimeSeries {
    #[getter]
    fn mean<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<NpF>> {
        let v: Vec<NpF> = self.inner.data.iter().map(|r| r.mean as NpF).collect();
        ndarray::Array1::from_vec(v).into_pyarray(py)
    }

    #[getter]
    fn per_particle<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<NpF>>> {
        let t = self.inner.data.len();
        if t == 0 {
            return Ok(ndarray::Array2::from_shape_vec((0, 0), vec![])
                .unwrap()
                .into_pyarray(py));
        }
        let n = self.inner.data[0].per_particle.len();
        let mut flat: Vec<NpF> = Vec::with_capacity(t * n);
        for row in &self.inner.data {
            if row.per_particle.len() != n {
                return Err(PyValueError::new_err(
                    "MSD per-particle width not constant across frames",
                ));
            }
            flat.extend(row.per_particle.iter().map(|&v| v as NpF));
        }
        Ok(ndarray::Array2::from_shape_vec((t, n), flat)
            .expect("MSD shape")
            .into_pyarray(py))
    }

    fn __len__(&self) -> usize {
        self.inner.data.len()
    }

    fn __getitem__(&self, i: isize) -> PyResult<PyMSDResult> {
        let n = self.inner.data.len() as isize;
        let idx = if i < 0 { i + n } else { i };
        if idx < 0 || idx >= n {
            return Err(PyValueError::new_err("MSD index out of range"));
        }
        Ok(PyMSDResult {
            inner: self.inner.data[idx as usize].clone(),
        })
    }

    fn __repr__(&self) -> String {
        format!("MSDTimeSeries(n_frames={})", self.inner.data.len())
    }
}

/// Stateless MSD calculator.
///
/// ``compute(frames)`` uses ``frames[0]`` as the reference and returns a
/// ``MSDTimeSeries`` of the same length as ``frames``.
#[pyclass(name = "MSD", unsendable)]
pub struct PyMSD {
    inner: MSD,
}

#[pymethods]
impl PyMSD {
    #[new]
    fn new() -> Self {
        Self { inner: MSD::new() }
    }

    /// Compute MSD time series against frames[0].
    fn compute(&self, frames: &Bound<'_, PyAny>) -> PyResult<PyMSDTimeSeries> {
        let owned = clone_frames(frames)?;
        if owned.is_empty() {
            return Err(PyValueError::new_err("MSD.compute requires >= 1 frame"));
        }
        let refs: Vec<&CoreFrame> = owned.iter().collect();
        let series = self.inner.compute(&refs, ()).map_err(py_value_err)?;
        Ok(PyMSDTimeSeries { inner: series })
    }

    fn __repr__(&self) -> String {
        "MSD()".to_string()
    }
}

// ---------------------------------------------------------------------------
// Cluster
// ---------------------------------------------------------------------------

/// Per-frame cluster assignment.
#[pyclass(name = "ClusterResult", unsendable)]
#[derive(Clone)]
pub struct PyClusterResult {
    pub(crate) inner: ClusterResult,
}

#[pymethods]
impl PyClusterResult {
    #[getter]
    fn num_clusters(&self) -> usize {
        self.inner.num_clusters
    }

    #[getter]
    fn cluster_idx<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        self.inner.cluster_idx.clone().into_pyarray(py)
    }

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

/// Distance-based cluster analysis.
#[pyclass(name = "Cluster", unsendable)]
pub struct PyCluster {
    inner: Cluster,
}

#[pymethods]
impl PyCluster {
    #[new]
    fn new(min_cluster_size: usize) -> Self {
        Self {
            inner: Cluster::new(min_cluster_size),
        }
    }

    /// Compute one cluster result per input frame.
    ///
    /// Returns a single `ClusterResult` when a single frame is passed, or a
    /// `list[ClusterResult]` when a list is passed.
    fn compute<'py>(
        &self,
        py: Python<'py>,
        frames: &Bound<'py, PyAny>,
        nlists: &Bound<'py, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let batched = was_batched(frames);
        let owned = clone_frames(frames)?;
        let refs: Vec<&CoreFrame> = owned.iter().collect();
        let nlists_vec = collect_nlists(nlists)?;
        if nlists_vec.len() != refs.len() {
            return Err(PyValueError::new_err(format!(
                "len(nlists)={} must equal len(frames)={}",
                nlists_vec.len(),
                refs.len()
            )));
        }
        let out = self
            .inner
            .compute(&refs, &nlists_vec)
            .map_err(py_value_err)?;
        if !batched {
            let single = out.into_iter().next().unwrap();
            return Ok(Py::new(py, PyClusterResult { inner: single })?.into_any());
        }
        let wrapped: Vec<Py<PyClusterResult>> = out
            .into_iter()
            .map(|r| Py::new(py, PyClusterResult { inner: r }))
            .collect::<PyResult<_>>()?;
        Ok(pyo3::types::PyList::new(py, wrapped)?.into_any().unbind())
    }

    fn __repr__(&self) -> String {
        "Cluster(...)".to_string()
    }
}

// ---------------------------------------------------------------------------
// ClusterCenters
// ---------------------------------------------------------------------------

/// Geometric cluster centers for a single frame.
#[pyclass(name = "ClusterCentersResult", unsendable)]
#[derive(Clone)]
pub struct PyClusterCentersResult {
    pub(crate) inner: ClusterCentersResult,
}

#[pymethods]
impl PyClusterCentersResult {
    #[getter]
    fn centers<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<NpF>> {
        let nc = self.inner.centers.len();
        let flat: Vec<NpF> = self
            .inner
            .centers
            .iter()
            .flat_map(|c| [c[0] as NpF, c[1] as NpF, c[2] as NpF])
            .collect();
        ndarray::Array2::from_shape_vec((nc, 3), flat)
            .expect("centers shape")
            .into_pyarray(py)
    }

    fn __len__(&self) -> usize {
        self.inner.centers.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "ClusterCentersResult(num_clusters={})",
            self.inner.centers.len()
        )
    }
}

fn extract_cluster_vec(arg: &Bound<'_, PyAny>) -> PyResult<(bool, Vec<ClusterResult>)> {
    if let Ok(single) = arg.extract::<PyRef<'_, PyClusterResult>>() {
        return Ok((false, vec![single.inner.clone()]));
    }
    let list: Vec<PyRef<'_, PyClusterResult>> = arg.extract()?;
    Ok((true, list.iter().map(|r| r.inner.clone()).collect()))
}

fn extract_centers_vec(arg: &Bound<'_, PyAny>) -> PyResult<Vec<ClusterCentersResult>> {
    if let Ok(single) = arg.extract::<PyRef<'_, PyClusterCentersResult>>() {
        return Ok(vec![single.inner.clone()]);
    }
    let list: Vec<PyRef<'_, PyClusterCentersResult>> = arg.extract()?;
    Ok(list.iter().map(|r| r.inner.clone()).collect())
}

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

    fn compute<'py>(
        &self,
        py: Python<'py>,
        frames: &Bound<'py, PyAny>,
        clusters: &Bound<'py, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let batched = was_batched(frames);
        let owned = clone_frames(frames)?;
        let refs: Vec<&CoreFrame> = owned.iter().collect();
        let (_, cl_vec) = extract_cluster_vec(clusters)?;
        if cl_vec.len() != refs.len() {
            return Err(PyValueError::new_err(
                "len(clusters) must equal len(frames)",
            ));
        }
        let out = self.inner.compute(&refs, &cl_vec).map_err(py_value_err)?;
        if !batched {
            let single = out.into_iter().next().unwrap();
            return Ok(Py::new(
                py,
                PyClusterCentersResult { inner: single },
            )?
            .into_any());
        }
        let wrapped: Vec<Py<PyClusterCentersResult>> = out
            .into_iter()
            .map(|r| Py::new(py, PyClusterCentersResult { inner: r }))
            .collect::<PyResult<_>>()?;
        Ok(pyo3::types::PyList::new(py, wrapped)?.into_any().unbind())
    }

    fn __repr__(&self) -> String {
        "ClusterCenters()".to_string()
    }
}

// ---------------------------------------------------------------------------
// CenterOfMass
// ---------------------------------------------------------------------------

/// Per-frame mass-weighted cluster centers and total cluster masses.
#[pyclass(name = "CenterOfMassResult", unsendable)]
#[derive(Clone)]
pub struct PyCenterOfMassResult {
    pub(crate) inner: COMResult,
}

#[pymethods]
impl PyCenterOfMassResult {
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

fn extract_com_vec(arg: &Bound<'_, PyAny>) -> PyResult<Vec<COMResult>> {
    if let Ok(single) = arg.extract::<PyRef<'_, PyCenterOfMassResult>>() {
        return Ok(vec![single.inner.clone()]);
    }
    let list: Vec<PyRef<'_, PyCenterOfMassResult>> = arg.extract()?;
    Ok(list.iter().map(|r| r.inner.clone()).collect())
}

#[pyclass(name = "CenterOfMass", unsendable)]
pub struct PyCenterOfMass {
    masses: Option<Vec<F>>,
}

#[pymethods]
impl PyCenterOfMass {
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

    fn compute<'py>(
        &self,
        py: Python<'py>,
        frames: &Bound<'py, PyAny>,
        clusters: &Bound<'py, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let batched = was_batched(frames);
        let owned = clone_frames(frames)?;
        let refs: Vec<&CoreFrame> = owned.iter().collect();
        let (_, cl_vec) = extract_cluster_vec(clusters)?;
        if cl_vec.len() != refs.len() {
            return Err(PyValueError::new_err(
                "len(clusters) must equal len(frames)",
            ));
        }
        let calc = if let Some(ref ms) = self.masses {
            CenterOfMass::new().with_masses(ms)
        } else {
            CenterOfMass::new()
        };
        let out = calc.compute(&refs, &cl_vec).map_err(py_value_err)?;
        if !batched {
            let single = out.into_iter().next().unwrap();
            return Ok(Py::new(
                py,
                PyCenterOfMassResult { inner: single },
            )?
            .into_any());
        }
        let wrapped: Vec<Py<PyCenterOfMassResult>> = out
            .into_iter()
            .map(|r| Py::new(py, PyCenterOfMassResult { inner: r }))
            .collect::<PyResult<_>>()?;
        Ok(pyo3::types::PyList::new(py, wrapped)?.into_any().unbind())
    }

    fn __repr__(&self) -> String {
        "CenterOfMass(...)".to_string()
    }
}

// ---------------------------------------------------------------------------
// GyrationTensor
// ---------------------------------------------------------------------------

fn tensor_list_into_pyarray<'py>(
    py: Python<'py>,
    batched: bool,
    tensors: Vec<[[F; 3]; 3]>,
) -> Bound<'py, PyArrayDyn<NpF>> {
    let n = tensors.len();
    if !batched {
        let flat: Vec<NpF> = tensors[0]
            .iter()
            .flat_map(|row| row.iter().map(|&v| v as NpF))
            .collect();
        return ndarray::ArrayD::from_shape_vec(vec![3, 3], flat)
            .expect("tensor shape")
            .into_pyarray(py);
    }
    let flat: Vec<NpF> = tensors
        .iter()
        .flat_map(|t| t.iter().flat_map(|row| row.iter().map(|&v| v as NpF)))
        .collect();
    ndarray::ArrayD::from_shape_vec(vec![n, 3, 3], flat)
        .expect("tensor batch shape")
        .into_pyarray(py)
}

/// Gyration tensor per cluster.
///
/// ``compute(frames, clusters, centers)``:
/// - single frame → shape `(num_clusters, 3, 3)` **but wrapped as a list of length 1**
///   when a list of frames is passed. For a single frame you get a `(num_clusters, 3, 3)` ndarray.
/// - list of frames → ndarray of shape `(n_frames, num_clusters, 3, 3)` only if all frames
///   have identical cluster counts; otherwise a Python list of per-frame ndarrays.
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

    fn compute<'py>(
        &self,
        py: Python<'py>,
        frames: &Bound<'py, PyAny>,
        clusters: &Bound<'py, PyAny>,
        centers: &Bound<'py, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let batched = was_batched(frames);
        let owned = clone_frames(frames)?;
        let refs: Vec<&CoreFrame> = owned.iter().collect();
        let (_, cl_vec) = extract_cluster_vec(clusters)?;
        let cc_vec = extract_centers_vec(centers)?;
        if cl_vec.len() != refs.len() || cc_vec.len() != refs.len() {
            return Err(PyValueError::new_err(
                "clusters and centers must match len(frames)",
            ));
        }
        let out = self
            .inner
            .compute(&refs, (&cl_vec, &cc_vec))
            .map_err(py_value_err)?;
        if !batched {
            let tensors = out.into_iter().next().unwrap().0;
            return Ok(tensor_list_into_pyarray(py, false, tensors).into_any().unbind());
        }
        let arrays: Vec<Py<PyArrayDyn<NpF>>> = out
            .into_iter()
            .map(|r| tensor_list_into_pyarray(py, false, r.0).unbind())
            .collect();
        Ok(pyo3::types::PyList::new(py, arrays)?.into_any().unbind())
    }

    fn __repr__(&self) -> String {
        "GyrationTensor()".to_string()
    }
}

// ---------------------------------------------------------------------------
// InertiaTensor
// ---------------------------------------------------------------------------

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

    fn compute<'py>(
        &self,
        py: Python<'py>,
        frames: &Bound<'py, PyAny>,
        clusters: &Bound<'py, PyAny>,
        com: &Bound<'py, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let batched = was_batched(frames);
        let owned = clone_frames(frames)?;
        let refs: Vec<&CoreFrame> = owned.iter().collect();
        let (_, cl_vec) = extract_cluster_vec(clusters)?;
        let com_vec = extract_com_vec(com)?;
        if cl_vec.len() != refs.len() || com_vec.len() != refs.len() {
            return Err(PyValueError::new_err(
                "clusters and com must match len(frames)",
            ));
        }
        let calc = if let Some(ref ms) = self.masses {
            InertiaTensor::new().with_masses(ms)
        } else {
            InertiaTensor::new()
        };
        let out = calc
            .compute(&refs, (&cl_vec, &com_vec))
            .map_err(py_value_err)?;
        if !batched {
            let tensors = out.into_iter().next().unwrap().0;
            return Ok(tensor_list_into_pyarray(py, false, tensors).into_any().unbind());
        }
        let arrays: Vec<Py<PyArrayDyn<NpF>>> = out
            .into_iter()
            .map(|r| tensor_list_into_pyarray(py, false, r.0).unbind())
            .collect();
        Ok(pyo3::types::PyList::new(py, arrays)?.into_any().unbind())
    }

    fn __repr__(&self) -> String {
        "InertiaTensor(...)".to_string()
    }
}

// ---------------------------------------------------------------------------
// RadiusOfGyration
// ---------------------------------------------------------------------------

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

    /// Compute Rg per cluster.
    ///
    /// Returns a `(num_clusters,)` ndarray for a single frame, or a
    /// `(n_frames, num_clusters)` ndarray for a batch (clusters per frame
    /// must be identical; otherwise a Python list is returned).
    fn compute<'py>(
        &self,
        py: Python<'py>,
        frames: &Bound<'py, PyAny>,
        clusters: &Bound<'py, PyAny>,
        com: &Bound<'py, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let batched = was_batched(frames);
        let owned = clone_frames(frames)?;
        let refs: Vec<&CoreFrame> = owned.iter().collect();
        let (_, cl_vec) = extract_cluster_vec(clusters)?;
        let com_vec = extract_com_vec(com)?;
        if cl_vec.len() != refs.len() || com_vec.len() != refs.len() {
            return Err(PyValueError::new_err(
                "clusters and com must match len(frames)",
            ));
        }
        let calc = if let Some(ref ms) = self.masses {
            RadiusOfGyration::new().with_masses(ms)
        } else {
            RadiusOfGyration::new()
        };
        let out: Vec<RgResult> = calc
            .compute(&refs, (&cl_vec, &com_vec))
            .map_err(py_value_err)?;

        if !batched {
            let v: Vec<NpF> = out
                .into_iter()
                .next()
                .unwrap()
                .0
                .iter()
                .map(|&r| r as NpF)
                .collect();
            return Ok(ndarray::Array1::from_vec(v)
                .into_pyarray(py)
                .into_any()
                .unbind());
        }

        // Try a rectangular (n_frames, nc) packing if widths agree.
        let widths: Vec<usize> = out.iter().map(|r| r.0.len()).collect();
        let uniform = widths.iter().all(|&w| w == widths[0]);
        if uniform {
            let n_frames = out.len();
            let nc = widths.first().copied().unwrap_or(0);
            let mut flat: Vec<NpF> = Vec::with_capacity(n_frames * nc);
            for row in &out {
                flat.extend(row.0.iter().map(|&v| v as NpF));
            }
            let arr = ndarray::Array2::from_shape_vec((n_frames, nc), flat)
                .expect("rg batch shape");
            return Ok(arr.into_pyarray(py).into_any().unbind());
        }

        let arrays: Vec<Py<PyArray1<NpF>>> = out
            .into_iter()
            .map(|r| {
                let v: Vec<NpF> = r.0.iter().map(|&x| x as NpF).collect();
                ndarray::Array1::from_vec(v).into_pyarray(py).unbind()
            })
            .collect();
        Ok(pyo3::types::PyList::new(py, arrays)?.into_any().unbind())
    }

    fn __repr__(&self) -> String {
        "RadiusOfGyration(...)".to_string()
    }
}

// ---------------------------------------------------------------------------
// PCA
// ---------------------------------------------------------------------------

/// Row-based descriptor wrapper for PCA/KMeans input.
///
/// Wrap each row (a 1-D float array) with `DescriptorRow(row)`; then pass a
/// Python list of them to ``Pca2.compute`` / ``KMeans.compute``.
#[pyclass(name = "DescriptorRow", unsendable)]
#[derive(Clone)]
pub struct PyDescriptorRow {
    row: Vec<F>,
}

#[pymethods]
impl PyDescriptorRow {
    #[new]
    fn new(values: PyReadonlyArray1<'_, NpF>) -> PyResult<Self> {
        let slice = values.as_slice()?;
        Ok(Self {
            row: slice.iter().map(|&v| v as F).collect(),
        })
    }

    fn __len__(&self) -> usize {
        self.row.len()
    }
}

impl molrs_compute::DescriptorRow for PyDescriptorRow {
    fn as_row(&self) -> &[F] {
        &self.row
    }
}

/// Two-component PCA result.
#[pyclass(name = "PcaResult", unsendable)]
#[derive(Clone)]
pub struct PyPcaResult {
    pub(crate) inner: PcaResult,
}

#[pymethods]
impl PyPcaResult {
    /// Row-major `(n_rows, 2)` projected coordinates.
    #[getter]
    fn coords<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<NpF>>> {
        let n = self.inner.coords.len() / 2;
        let flat: Vec<NpF> = self.inner.coords.iter().map(|&v| v as NpF).collect();
        Ok(ndarray::Array2::from_shape_vec((n, 2), flat)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .into_pyarray(py))
    }

    /// `[var_pc1, var_pc2]` — explained variance per component.
    #[getter]
    fn variance(&self) -> (NpF, NpF) {
        (self.inner.variance[0] as NpF, self.inner.variance[1] as NpF)
    }

    fn __repr__(&self) -> String {
        format!(
            "PcaResult(n_rows={}, variance=[{:.4}, {:.4}])",
            self.inner.coords.len() / 2,
            self.inner.variance[0],
            self.inner.variance[1],
        )
    }
}

/// Two-component PCA calculator.
#[pyclass(name = "Pca2", unsendable)]
pub struct PyPca2 {
    inner: Pca2<PyDescriptorRow>,
}

#[pymethods]
impl PyPca2 {
    #[new]
    fn new() -> Self {
        Self {
            inner: Pca2::<PyDescriptorRow>::new(),
        }
    }

    /// Compute PCA over a list of `DescriptorRow` objects.
    fn compute(&self, rows: Vec<PyRef<'_, PyDescriptorRow>>) -> PyResult<PyPcaResult> {
        let owned: Vec<PyDescriptorRow> = rows.iter().map(|r| (*r).clone()).collect();
        // Wrap an empty FrameAccess slice — Pca2 does not touch frames.
        let frames: [&CoreFrame; 0] = [];
        let pca = self
            .inner
            .compute(&frames, &owned)
            .map_err(py_value_err)?;
        Ok(PyPcaResult { inner: pca })
    }

    fn __repr__(&self) -> String {
        "Pca2()".to_string()
    }
}

// ---------------------------------------------------------------------------
// KMeans
// ---------------------------------------------------------------------------

/// k-means cluster labels.
#[pyclass(name = "KMeansResult", unsendable)]
pub struct PyKMeansResult {
    inner: KMeansResult,
}

#[pymethods]
impl PyKMeansResult {
    #[getter]
    fn labels<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        ndarray::Array1::from_vec(self.inner.0.clone()).into_pyarray(py)
    }

    fn __len__(&self) -> usize {
        self.inner.0.len()
    }

    fn __repr__(&self) -> String {
        format!("KMeansResult(n={})", self.inner.0.len())
    }
}

/// k-means clustering over a PCA projection (2-D).
#[pyclass(name = "KMeans", unsendable)]
pub struct PyKMeans {
    inner: KMeans,
}

#[pymethods]
impl PyKMeans {
    #[new]
    #[pyo3(signature = (k, max_iter = 100, seed = 0))]
    fn new(k: usize, max_iter: usize, seed: u64) -> PyResult<Self> {
        Ok(Self {
            inner: KMeans::new(k, max_iter, seed).map_err(py_value_err)?,
        })
    }

    /// Cluster a `PcaResult`.
    fn compute(&self, pca: &PyPcaResult) -> PyResult<PyKMeansResult> {
        let frames: [&CoreFrame; 0] = [];
        let out = self
            .inner
            .compute(&frames, &pca.inner)
            .map_err(py_value_err)?;
        Ok(PyKMeansResult { inner: out })
    }

    fn __repr__(&self) -> String {
        "KMeans(...)".to_string()
    }
}
