//! Python wrappers for the analysis-parity computes (geometric distributions,
//! Van Hove, Legendre reorientation, hydrogen bonds, spatial distribution,
//! radical-Voronoi tessellation + domain/void analysis).
//!
//! Same shape as the rest of the compute bindings: each `compute(...)` accepts a
//! single frame or a list of frames and returns one typed result. The chiral /
//! resonance spectra (`VcdSpectrum` / `RoaSpectrum` / `ResonanceRamanSpectrum`)
//! live in [`crate::compute::fit`] alongside the other spectral transforms.

use molrs::compute::Compute;
use molrs::compute::distribution::{
    AngleObservable, AnyObservable, AtomGroups, AxisSpec, CombinedDistribution,
    CombinedDistributionResult, DihedralObservable, DistanceObservable, DistributionFunction,
    DistributionResult,
};
use molrs::compute::{
    DensityGrid, DistKind, DomainAnalysis, GridSpec, HBondCriterion, HBonds, HBondsResult,
    LegendreReorientation, LegendreReorientationResult, MolecularMoments, RadicalVoronoi,
    SpatialDistribution, SpatialDistributionResult, VanHove, VanHoveResult, VoidAnalysis,
    VoronoiCells, VoronoiIntegration, polarizability_finite_field,
};
use molrs::store::frame::Frame as CoreFrame;
use molrs::store::frame_access::FrameAccess;
use molrs::types::F;

use ndarray::Array2;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyArray4, PyReadonlyArray1, PyReadonlyArray2,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::core::spatial::simbox::PyBox;
use crate::helpers::{NpF, collect_frames, py_value_err};

// ---------------------------------------------------------------------------
// Distribution functions (ADF / DDF / distance-DF)
// ---------------------------------------------------------------------------

/// Shared result of the geometric distribution functions.
#[pyclass(name = "DistributionResult")]
pub struct PyDistributionResult {
    inner: DistributionResult,
}

#[pymethods]
impl PyDistributionResult {
    #[getter]
    fn bin_centers<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<NpF>> {
        self.inner.bin_centers.clone().into_pyarray(py)
    }
    #[getter]
    fn bin_edges<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<NpF>> {
        self.inner.bin_edges.clone().into_pyarray(py)
    }
    #[getter]
    fn counts<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<NpF>> {
        self.inner.counts.clone().into_pyarray(py)
    }
    #[getter]
    fn density<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<NpF>> {
        self.inner.density.clone().into_pyarray(py)
    }
    #[getter]
    fn density_sin_corrected<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<NpF>>> {
        self.inner
            .density_sin_corrected
            .as_ref()
            .map(|d| d.clone().into_pyarray(py))
    }
    #[getter]
    fn bin_width(&self) -> NpF {
        self.inner.bin_width
    }
    #[getter]
    fn n_binned(&self) -> NpF {
        self.inner.n_binned
    }
    #[getter]
    fn n_raw_samples(&self) -> usize {
        self.inner.n_raw_samples
    }
    #[getter]
    fn n_frames(&self) -> usize {
        self.inner.n_frames
    }
    #[getter]
    fn angular(&self) -> bool {
        self.inner.angular
    }
}

/// Angular distribution function (ADF) over atom triplets (angle at the middle
/// atom). Ported from the reference implementation; the sin θ correction is exposed separately.
#[pyclass(name = "AngleDistribution")]
pub struct PyAngleDistribution {
    inner: DistributionFunction<AngleObservable>,
}

#[pymethods]
impl PyAngleDistribution {
    #[new]
    #[pyo3(signature = (n_bins, min=0.0, max=180.0))]
    fn new(n_bins: usize, min: NpF, max: NpF) -> PyResult<Self> {
        let inner =
            DistributionFunction::new(AngleObservable, n_bins, min, max).map_err(py_value_err)?;
        Ok(Self { inner })
    }

    /// Atom triplets (angle vertex in the middle) are read from the `angles`
    /// topology block of the first frame.
    fn compute(&self, frames: &Bound<'_, PyAny>) -> PyResult<PyDistributionResult> {
        let owned = collect_frames(frames)?;
        let refs: Vec<&CoreFrame> = owned.iter().collect();
        let first = refs
            .first()
            .copied()
            .ok_or_else(|| PyValueError::new_err("no frames provided"))?;
        let groups = AtomGroups::from_frame(first, "angles", 3).map_err(py_value_err)?;
        let inner = self.inner.compute(&refs, &groups).map_err(py_value_err)?;
        Ok(PyDistributionResult { inner })
    }
}

/// Dihedral distribution function (DDF) over atom quadruplets.
#[pyclass(name = "DihedralDistribution")]
pub struct PyDihedralDistribution {
    inner: DistributionFunction<DihedralObservable>,
}

#[pymethods]
impl PyDihedralDistribution {
    #[new]
    #[pyo3(signature = (n_bins, min=-180.0, max=180.0))]
    fn new(n_bins: usize, min: NpF, max: NpF) -> PyResult<Self> {
        let inner = DistributionFunction::new(DihedralObservable, n_bins, min, max)
            .map_err(py_value_err)?;
        Ok(Self { inner })
    }

    /// Atom quadruplets are read from the `dihedrals` topology block of the
    /// first frame.
    fn compute(&self, frames: &Bound<'_, PyAny>) -> PyResult<PyDistributionResult> {
        let owned = collect_frames(frames)?;
        let refs: Vec<&CoreFrame> = owned.iter().collect();
        let first = refs
            .first()
            .copied()
            .ok_or_else(|| PyValueError::new_err("no frames provided"))?;
        let groups = AtomGroups::from_frame(first, "dihedrals", 4).map_err(py_value_err)?;
        let inner = self.inner.compute(&refs, &groups).map_err(py_value_err)?;
        Ok(PyDistributionResult { inner })
    }
}

/// Distance distribution function over atom pairs.
#[pyclass(name = "DistanceDistribution")]
pub struct PyDistanceDistribution {
    inner: DistributionFunction<DistanceObservable>,
}

#[pymethods]
impl PyDistanceDistribution {
    #[new]
    #[pyo3(signature = (n_bins, min, max))]
    fn new(n_bins: usize, min: NpF, max: NpF) -> PyResult<Self> {
        let inner = DistributionFunction::new(DistanceObservable, n_bins, min, max)
            .map_err(py_value_err)?;
        Ok(Self { inner })
    }

    /// Atom pairs are read from the `bonds` topology block of the first frame.
    fn compute(&self, frames: &Bound<'_, PyAny>) -> PyResult<PyDistributionResult> {
        let owned = collect_frames(frames)?;
        let refs: Vec<&CoreFrame> = owned.iter().collect();
        let first = refs
            .first()
            .copied()
            .ok_or_else(|| PyValueError::new_err("no frames provided"))?;
        let groups = AtomGroups::from_frame(first, "bonds", 2).map_err(py_value_err)?;
        let inner = self.inner.compute(&refs, &groups).map_err(py_value_err)?;
        Ok(PyDistributionResult { inner })
    }
}

// ---------------------------------------------------------------------------
// Combined (multi-axis) distribution
// ---------------------------------------------------------------------------

/// Map an observable-kind string to its `AnyObservable` variant + arity.
#[pyclass(name = "CombinedDistributionResult")]
pub struct PyCombinedDistributionResult {
    inner: CombinedDistributionResult,
}

#[pymethods]
impl PyCombinedDistributionResult {
    /// Per-axis bin edges (`bins + 1` each).
    #[getter]
    fn edges<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyArray1<NpF>>> {
        self.inner
            .edges
            .iter()
            .map(|e| e.clone().into_pyarray(py))
            .collect()
    }
    /// Per-axis bin centers (`bins` each).
    #[getter]
    fn centers<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyArray1<NpF>>> {
        self.inner
            .centers
            .iter()
            .map(|c| c.clone().into_pyarray(py))
            .collect()
    }
    /// Flat row-major counts (axis 0 fastest).
    #[getter]
    fn counts<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<NpF>> {
        self.inner.counts.clone().into_pyarray(py)
    }
    /// Flat row-major normalized joint density.
    #[getter]
    fn density<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<NpF>> {
        self.inner.density.clone().into_pyarray(py)
    }
    #[getter]
    fn binned(&self) -> NpF {
        self.inner.binned
    }
    #[getter]
    fn n_raw_samples(&self) -> usize {
        self.inner.n_raw_samples
    }
    #[getter]
    fn n_frames(&self) -> usize {
        self.inner.n_frames
    }
    #[getter]
    fn ndim(&self) -> usize {
        self.inner.ndim()
    }
    /// Flat row-major index for a multi-axis bin coordinate.
    fn flat_index(&self, idx: Vec<usize>) -> usize {
        self.inner.flat_index(&idx)
    }
    /// Product of all axis bin widths (the N-D cell "volume").
    fn bin_width_product(&self) -> NpF {
        self.inner.bin_width_product()
    }
}

/// Joint multi-axis distribution over several geometric observables (the reference implementation
/// combined-DF). Each axis is `(kind, bins, min, max, sin_weight)` where `kind`
/// is ``"distance"`` / ``"angle"`` / ``"dihedral"``.
#[pyclass(name = "CombinedDistribution")]
pub struct PyCombinedDistribution {
    inner: CombinedDistribution,
    arities: Vec<usize>,
}

#[pymethods]
impl PyCombinedDistribution {
    /// `axes`: list of `(kind, bins, min, max, sin_weight)` — one per dimension.
    #[new]
    fn new(axes: Vec<(String, usize, NpF, NpF, bool)>) -> PyResult<Self> {
        let mut observables = Vec::with_capacity(axes.len());
        let mut specs = Vec::with_capacity(axes.len());
        let mut arities = Vec::with_capacity(axes.len());
        for (kind, bins, min, max, sin_weight) in &axes {
            let (obs, arity) = AnyObservable::from_kind(kind).map_err(py_value_err)?;
            observables.push(obs);
            arities.push(arity);
            specs.push(
                AxisSpec::new(*bins, *min, *max)
                    .map_err(py_value_err)?
                    .with_sin_weight(*sin_weight),
            );
        }
        let inner = CombinedDistribution::new(observables, specs).map_err(py_value_err)?;
        Ok(Self { inner, arities })
    }

    /// Per-axis atom groups are read from the first frame's topology block that
    /// matches each axis arity (2 → `bonds`, 3 → `angles`, 4 → `dihedrals`).
    fn compute(&self, frames: &Bound<'_, PyAny>) -> PyResult<PyCombinedDistributionResult> {
        let owned = collect_frames(frames)?;
        let refs: Vec<&CoreFrame> = owned.iter().collect();
        let first = refs
            .first()
            .copied()
            .ok_or_else(|| PyValueError::new_err("no frames provided"))?;
        let group_objs: Vec<AtomGroups> = self
            .arities
            .iter()
            .map(|&arity| {
                let block = match arity {
                    2 => "bonds",
                    3 => "angles",
                    4 => "dihedrals",
                    _ => {
                        return Err(PyValueError::new_err(format!(
                            "no topology block for axis arity {arity}"
                        )));
                    }
                };
                AtomGroups::from_frame(first, block, arity).map_err(py_value_err)
            })
            .collect::<PyResult<_>>()?;
        let inner = self
            .inner
            .compute(&refs, &group_objs)
            .map_err(py_value_err)?;
        Ok(PyCombinedDistributionResult { inner })
    }
}

// ---------------------------------------------------------------------------
// Van Hove correlation function
// ---------------------------------------------------------------------------

#[pyclass(name = "VanHoveResult")]
pub struct PyVanHoveResult {
    inner: VanHoveResult,
}

#[pymethods]
impl PyVanHoveResult {
    #[getter]
    fn r_edges<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<NpF>> {
        self.inner.r_edges.clone().into_pyarray(py)
    }
    #[getter]
    fn r_centers<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<NpF>> {
        self.inner.r_centers.clone().into_pyarray(py)
    }
    #[getter]
    fn lags(&self) -> Vec<usize> {
        self.inner.lags.clone()
    }
    #[getter]
    fn g_self<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<NpF>> {
        self.inner.g_self.clone().into_pyarray(py)
    }
    #[getter]
    fn g_distinct<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<NpF>> {
        self.inner.g_distinct.clone().into_pyarray(py)
    }
    #[getter]
    fn dr(&self) -> NpF {
        self.inner.dr
    }
    #[getter]
    fn has_distinct(&self) -> bool {
        self.inner.has_distinct
    }
}

/// Van Hove correlation function `G(r, t)` (self + distinct parts).
#[pyclass(name = "VanHove")]
pub struct PyVanHove {
    inner: VanHove,
}

#[pymethods]
impl PyVanHove {
    #[new]
    #[pyo3(signature = (n_rbins, r_max, lags, stride=1))]
    fn new(n_rbins: usize, r_max: NpF, lags: Vec<usize>, stride: usize) -> PyResult<Self> {
        let inner = VanHove::new(n_rbins, r_max, lags)
            .map_err(py_value_err)?
            .with_stride(stride);
        Ok(Self { inner })
    }

    /// Compute `G(r, t)` from a trajectory (list of frames, time-ordered).
    fn compute(&self, frames: &Bound<'_, PyAny>) -> PyResult<PyVanHoveResult> {
        let owned = collect_frames(frames)?;
        let refs: Vec<&CoreFrame> = owned.iter().collect();
        let inner = self.inner.compute(&refs, ()).map_err(py_value_err)?;
        Ok(PyVanHoveResult { inner })
    }
}

// ---------------------------------------------------------------------------
// Legendre reorientation correlation (C1 / C2)
// ---------------------------------------------------------------------------

#[pyclass(name = "LegendreReorientationResult")]
pub struct PyLegendreReorientationResult {
    inner: LegendreReorientationResult,
}

#[pymethods]
impl PyLegendreReorientationResult {
    #[getter]
    fn lags(&self) -> Vec<usize> {
        self.inner.lags.clone()
    }
    #[getter]
    fn c1<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<NpF>> {
        self.inner.c1.clone().into_pyarray(py)
    }
    #[getter]
    fn c2<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<NpF>> {
        self.inner.c2.clone().into_pyarray(py)
    }
}

/// First/second Legendre reorientational TCFs `C₁(t)`, `C₂(t)` of bond vectors.
#[pyclass(name = "LegendreReorientation")]
pub struct PyLegendreReorientation {
    inner: LegendreReorientation,
}

#[pymethods]
impl PyLegendreReorientation {
    #[new]
    #[pyo3(signature = (max_lag, stride=1))]
    fn new(max_lag: usize, stride: usize) -> Self {
        Self {
            inner: LegendreReorientation::new(max_lag).with_stride(stride),
        }
    }

    /// The `(tail, head)` atom-index pairs defining each tracked bond vector are
    /// read from the `bonds` topology block of the first frame. `frames` are
    /// time-ordered.
    fn compute(
        &self,
        frames: &Bound<'_, PyAny>,
    ) -> PyResult<PyLegendreReorientationResult> {
        let owned = collect_frames(frames)?;
        let refs: Vec<&CoreFrame> = owned.iter().collect();
        let first = refs
            .first()
            .copied()
            .ok_or_else(|| PyValueError::new_err("no frames provided"))?;
        let groups = AtomGroups::from_frame(first, "bonds", 2).map_err(py_value_err)?;
        let tuples: Vec<(u32, u32)> = (0..groups.len())
            .map(|i| {
                let t = groups.tuple(i);
                (t[0], t[1])
            })
            .collect();
        let inner = self.inner.compute(&refs, &tuples).map_err(py_value_err)?;
        Ok(PyLegendreReorientationResult { inner })
    }
}

// ---------------------------------------------------------------------------
// Hydrogen bonds
// ---------------------------------------------------------------------------

/// Geometric hydrogen-bond criterion (Luzar–Chandler defaults).
#[pyclass(name = "HBondCriterion", from_py_object)]
#[derive(Clone)]
pub struct PyHBondCriterion {
    inner: HBondCriterion,
}

#[pymethods]
impl PyHBondCriterion {
    /// ``HBondCriterion(dist_cutoff=3.5, dist_kind="donor_acceptor", angle_cutoff=150.0)``.
    /// `dist_kind` is ``"donor_acceptor"`` or ``"hydrogen_acceptor"``.
    #[new]
    #[pyo3(signature = (dist_cutoff=3.5, dist_kind="donor_acceptor".to_string(), angle_cutoff=150.0))]
    fn new(dist_cutoff: NpF, dist_kind: String, angle_cutoff: NpF) -> PyResult<Self> {
        let kind = match dist_kind.as_str() {
            "donor_acceptor" => DistKind::DonorAcceptor,
            "hydrogen_acceptor" => DistKind::HydrogenAcceptor,
            other => {
                return Err(PyValueError::new_err(format!(
                    "dist_kind must be 'donor_acceptor' or 'hydrogen_acceptor', got {other:?}"
                )));
            }
        };
        Ok(Self {
            inner: HBondCriterion::new(dist_cutoff, kind, angle_cutoff),
        })
    }
}

#[pyclass(name = "HBondsResult")]
pub struct PyHBondsResult {
    inner: HBondsResult,
}

/// Per-frame hydrogen bonds: lists of `(donor, hydrogen, acceptor, distance, angle)`.
type PerFrameHBonds = Vec<Vec<(u32, u32, u32, NpF, NpF)>>;

#[pymethods]
impl PyHBondsResult {
    /// Per-frame hydrogen bonds as a list of lists of
    /// `(donor, hydrogen, acceptor, distance, angle)` tuples.
    #[getter]
    fn per_frame(&self) -> PerFrameHBonds {
        self.inner
            .per_frame
            .iter()
            .map(|frame| {
                frame
                    .iter()
                    .map(|b| (b.donor, b.hydrogen, b.acceptor, b.distance, b.angle))
                    .collect()
            })
            .collect()
    }

    /// H-bond count per frame.
    #[getter]
    fn counts(&self) -> Vec<usize> {
        self.inner.counts.clone()
    }
}

/// Detect hydrogen bonds per frame from explicit donor `(D, H)` pairs and
/// acceptor atoms under a geometric criterion.
#[pyclass(name = "HBonds")]
pub struct PyHBonds {
    inner: HBonds,
}

#[pymethods]
impl PyHBonds {
    /// `donors`: `(N, 2)` int array of `(donor, hydrogen)` pairs.
    /// `acceptors`: 1-D int array of acceptor atom indices.
    #[new]
    #[pyo3(signature = (donors, acceptors, criterion=None))]
    fn new(
        donors: PyReadonlyArray2<'_, i64>,
        acceptors: PyReadonlyArray1<'_, i64>,
        criterion: Option<PyHBondCriterion>,
    ) -> PyResult<Self> {
        let da = donors.as_array();
        if da.ncols() != 2 {
            return Err(PyValueError::new_err(
                "donors must have 2 columns (donor, hydrogen)",
            ));
        }
        let mut donor_pairs: Vec<(u32, u32)> = Vec::with_capacity(da.nrows());
        for row in da.rows() {
            if row[0] < 0 || row[1] < 0 {
                return Err(PyValueError::new_err("atom indices must be non-negative"));
            }
            donor_pairs.push((row[0] as u32, row[1] as u32));
        }
        let mut acc: Vec<u32> = Vec::with_capacity(acceptors.len()?);
        for &v in acceptors.as_array().iter() {
            if v < 0 {
                return Err(PyValueError::new_err("atom indices must be non-negative"));
            }
            acc.push(v as u32);
        }
        let crit = criterion.map(|c| c.inner).unwrap_or_default();
        Ok(Self {
            inner: HBonds::new(donor_pairs, acc, crit),
        })
    }

    /// Detect hydrogen bonds in each frame.
    fn compute(&self, frames: &Bound<'_, PyAny>) -> PyResult<PyHBondsResult> {
        let owned = collect_frames(frames)?;
        let refs: Vec<&CoreFrame> = owned.iter().collect();
        let inner = self.inner.compute(&refs, ()).map_err(py_value_err)?;
        Ok(PyHBondsResult { inner })
    }
}

// ---------------------------------------------------------------------------
// Spatial distribution function (SDF)
// ---------------------------------------------------------------------------

#[pyclass(name = "SpatialDistributionResult")]
pub struct PySpatialDistributionResult {
    inner: SpatialDistributionResult,
}

#[pymethods]
impl PySpatialDistributionResult {
    #[getter]
    fn counts<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<NpF>> {
        self.inner.counts.clone().into_pyarray(py)
    }
    #[getter]
    fn density<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<NpF>> {
        self.inner.density.clone().into_pyarray(py)
    }
    #[getter]
    fn g_sdf<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray3<NpF>>> {
        self.inner
            .g_sdf
            .as_ref()
            .map(|g| g.clone().into_pyarray(py))
    }
    /// Per-voxel mean body-frame orientation `(nx, ny, nz, 3)` (only present
    /// when the frames carried an `"orientations"` topology block).
    #[getter]
    fn orientation<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray4<NpF>>> {
        self.inner
            .orientation
            .as_ref()
            .map(|o| o.clone().into_pyarray(py))
    }
    #[getter]
    fn n(&self) -> [usize; 3] {
        self.inner.n
    }
    #[getter]
    fn extent(&self) -> [NpF; 3] {
        self.inner.extent
    }
    #[getter]
    fn voxel_volume(&self) -> NpF {
        self.inner.voxel_volume
    }
    #[getter]
    fn n_frames(&self) -> usize {
        self.inner.n_frames
    }
}

/// Spatial distribution function: target-atom density on a body-fixed grid,
/// aligned to a reference template via Kabsch superposition.
#[pyclass(name = "SpatialDistribution")]
pub struct PySpatialDistribution {
    inner: SpatialDistribution,
}

#[pymethods]
impl PySpatialDistribution {
    /// Parameters
    /// ----------
    /// reference, target : list[int]
    ///     Reference (≥3, for alignment) and target (binned) atom indices.
    /// template : (R, 3) float array
    ///     Body-frame template coordinates for the reference atoms.
    /// n : (int, int, int)
    ///     Grid voxel counts per axis.
    /// extent : (float, float, float)
    ///     Grid extent (Å) per axis.
    /// bulk_density : float, optional
    ///     If set, also produce ``g_sdf = density / bulk_density``.
    ///
    /// A per-voxel mean orientation field is produced when the frames carry an
    /// ``"orientations"`` topology block — one `(head, tail)` atom pair per
    /// target atom — read at compute time (no constructor array).
    #[new]
    #[pyo3(signature = (reference, template, target, n, extent, bulk_density=None))]
    fn new(
        reference: Vec<usize>,
        template: PyReadonlyArray2<'_, f64>,
        target: Vec<usize>,
        n: [usize; 3],
        extent: [F; 3],
        bulk_density: Option<F>,
    ) -> PyResult<Self> {
        let tmpl: Array2<F> = template.as_array().to_owned();
        let grid = GridSpec { n, extent };
        let mut sdf =
            SpatialDistribution::new(reference, tmpl, target, grid).map_err(py_value_err)?;
        if let Some(rho) = bulk_density {
            sdf = sdf.with_bulk_density(rho);
        }
        Ok(Self { inner: sdf })
    }

    /// Accumulate the SDF over a trajectory (list of frames).
    ///
    /// If the first frame carries an `"orientations"` topology block (one
    /// `(head, tail)` atom pair per target atom, in `target` order), a per-voxel
    /// mean body-frame orientation of the unit `head − tail` vector is
    /// accumulated; otherwise the SDF is orientation-free (the old `None` case).
    fn compute(&self, frames: &Bound<'_, PyAny>) -> PyResult<PySpatialDistributionResult> {
        let owned = collect_frames(frames)?;
        let refs: Vec<&CoreFrame> = owned.iter().collect();
        let first = refs
            .first()
            .copied()
            .ok_or_else(|| PyValueError::new_err("no frames provided"))?;
        let inner = if first.contains_block("orientations") {
            let groups = AtomGroups::from_frame(first, "orientations", 2).map_err(py_value_err)?;
            // The block stores `(head, tail)`; `with_orientation` expects
            // `(tail, head)` and forms the unit `head − tail` vector internally
            // (with minimum-image), so swap the endpoint order.
            let tuples: Vec<(usize, usize)> = (0..groups.len())
                .map(|i| {
                    let t = groups.tuple(i);
                    (t[1] as usize, t[0] as usize)
                })
                .collect();
            self.inner
                .clone()
                .with_orientation(tuples)
                .compute(&refs, ())
                .map_err(py_value_err)?
        } else {
            self.inner.compute(&refs, ()).map_err(py_value_err)?
        };
        Ok(PySpatialDistributionResult { inner })
    }
}

// ---------------------------------------------------------------------------
// Radical (Laguerre) Voronoi tessellation + domain / void analysis
// ---------------------------------------------------------------------------

/// Per-cell radical-Voronoi tessellation result.
#[pyclass(name = "VoronoiCells")]
pub struct PyVoronoiCells {
    inner: VoronoiCells,
}

#[pymethods]
impl PyVoronoiCells {
    /// Per-cell volumes (Å³), one per input point.
    #[getter]
    fn volumes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<NpF>> {
        PyArray1::from_slice(py, &self.inner.volumes)
    }
    #[getter]
    fn total_volume(&self) -> NpF {
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
#[pyclass(name = "RadicalVoronoi")]
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

/// Merge face-adjacent cells sharing the same label into domains. `labels` has
/// one integer per cell. Returns ``{"sizes", "count", "largest_fraction",
/// "domain_of"}``.
#[pyfunction]
fn voronoi_domains<'py>(
    py: Python<'py>,
    cells: &PyVoronoiCells,
    labels: Vec<i64>,
) -> PyResult<Bound<'py, PyDict>> {
    let r = DomainAnalysis
        .analyze(&cells.inner, &labels)
        .map_err(py_value_err)?;
    let d = PyDict::new(py);
    d.set_item("sizes", r.sizes)?;
    d.set_item("count", r.count)?;
    d.set_item("largest_fraction", r.largest_fraction)?;
    d.set_item("domain_of", r.domain_of)?;
    Ok(d)
}

/// Merge adjacent void-probe cells into cavities. `is_void` is a per-cell bool
/// mask; `box_volume` normalizes the void fraction. Returns
/// ``{"cavity_volumes", "total_void_volume", "void_fraction"}``.
#[pyfunction]
fn voronoi_voids<'py>(
    py: Python<'py>,
    cells: &PyVoronoiCells,
    is_void: Vec<bool>,
    box_volume: F,
) -> PyResult<Bound<'py, PyDict>> {
    let r = VoidAnalysis
        .analyze(&cells.inner, &is_void, box_volume)
        .map_err(py_value_err)?;
    let d = PyDict::new(py);
    d.set_item("cavity_volumes", r.cavity_volumes)?;
    d.set_item("total_void_volume", r.total_void_volume)?;
    d.set_item("void_fraction", r.void_fraction)?;
    Ok(d)
}

// ---------------------------------------------------------------------------
// Voronoi electron-density integration → per-molecule moments → polarizability
// ---------------------------------------------------------------------------

/// A volumetric electron density on a (generally non-orthogonal) voxel grid.
#[pyclass(name = "DensityGrid")]
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
#[pyclass(name = "MolecularMoments")]
pub struct PyMolecularMoments {
    inner: MolecularMoments,
}

#[pymethods]
impl PyMolecularMoments {
    /// Molecular charges `Q_m` (e), length `n_mol`.
    #[getter]
    fn charges<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<NpF>> {
        PyArray1::from_slice(py, &self.inner.charges)
    }
    /// Molecular dipoles `μ_m` (e·Å), shape `(n_mol, 3)`.
    #[getter]
    fn dipoles<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<NpF>> {
        self.inner.dipoles.clone().into_pyarray(py)
    }
    /// Per-molecule reference points (centre of nuclear charge), `(n_mol, 3)`.
    #[getter]
    fn references<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<NpF>> {
        self.inner.references.clone().into_pyarray(py)
    }
}

/// Integrate an electron density over radical-Voronoi cells into per-molecule
/// charges + dipoles.
#[pyclass(name = "VoronoiIntegration")]
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
    /// box : Box — periodic cell.
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
) -> PyResult<Bound<'py, PyArray2<NpF>>> {
    let alpha = polarizability_finite_field(&moments_zero.inner, &plus.inner, &minus.inner, field)
        .map_err(py_value_err)?;
    Ok(alpha.into_pyarray(py))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDistributionResult>()?;
    m.add_class::<PyAngleDistribution>()?;
    m.add_class::<PyDihedralDistribution>()?;
    m.add_class::<PyDistanceDistribution>()?;
    m.add_class::<PyCombinedDistribution>()?;
    m.add_class::<PyCombinedDistributionResult>()?;
    m.add_class::<PyVanHove>()?;
    m.add_class::<PyVanHoveResult>()?;
    m.add_class::<PyLegendreReorientation>()?;
    m.add_class::<PyLegendreReorientationResult>()?;
    m.add_class::<PyHBondCriterion>()?;
    m.add_class::<PyHBonds>()?;
    m.add_class::<PyHBondsResult>()?;
    m.add_class::<PySpatialDistribution>()?;
    m.add_class::<PySpatialDistributionResult>()?;
    m.add_class::<PyRadicalVoronoi>()?;
    m.add_class::<PyVoronoiCells>()?;
    m.add_class::<PyDensityGrid>()?;
    m.add_class::<PyMolecularMoments>()?;
    m.add_class::<PyVoronoiIntegration>()?;
    m.add_function(wrap_pyfunction!(voronoi_domains, m)?)?;
    m.add_function(wrap_pyfunction!(voronoi_voids, m)?)?;
    m.add_function(wrap_pyfunction!(polarizability_finite_field_py, m)?)?;
    Ok(())
}
