//! WASM bindings for trajectory analysis: neighbor search, RDF, MSD, and cluster detection.
//!
//! This module provides freud-style analysis classes that operate on
//! [`Frame`] objects. The typical workflow is:
//!
//! 1. Build a neighbor list using [`LinkedCell`].
//! 2. Pass the [`NeighborList`] to an analysis class ([`RDF`], [`Cluster`]).
//! 3. Read the result object.
//!
//! [`MSD`] does not require a neighbor list -- it only needs a reference
//! frame and a current frame.
//!
//! # Example (JavaScript)
//!
//! ```js
//! // Build neighbor list
//! const lc = new LinkedCell(5.0);
//! const nlist = lc.build(frame);
//!
//! // Compute RDF
//! const rdf = new RDF(100, 5.0);
//! const result = rdf.compute(frame, nlist);
//! const gr = result.rdf();           // Float32Array or Float64Array
//! const r  = result.binCenters();    // Float32Array or Float64Array
//!
//! // Compute MSD
//! const msd = new MSD();
//! for (const frame of trajectory) {
//!     msd.feed(frame);
//! }
//! console.log(msd.results()[1].mean); // MSD at frame 1 in A^2
//! ```
//!
//! # References
//!
//! - Ramasubramani, V. et al. (2020). freud: A software suite for
//!   high throughput analysis of particle simulation data. *Computer
//!   Physics Communications*, 254, 107275.

use wasm_bindgen::prelude::*;

use molrs::topology::{Topology as RsTopology, TopologyRingInfo as RsTopologyRingInfo};

use molrs::neighbors::{
    LinkCell as RsLinkCell, NbListAlgo, NeighborList as RsNeighborList,
    NeighborQuery as RsNeighborQuery, QueryMode,
};
use molrs::types::F;
use molrs_compute::center_of_mass::{COMResult as RsCOMResult, CenterOfMass as RsCenterOfMass};
use molrs_compute::cluster::{Cluster as RsCluster, ClusterResult as RsClusterResult};
use molrs_compute::cluster_centers::ClusterCenters as RsClusterCenters;
use molrs_compute::gyration_tensor::GyrationTensor as RsGyrationTensor;
use molrs_compute::inertia_tensor::InertiaTensor as RsInertiaTensor;
use molrs_compute::kmeans::KMeans as RsKMeans;
use molrs_compute::msd::{MSD as RsMSD, MSDResult as RsMSDResult};
use molrs_compute::pca::{Pca2 as RsPca2, PcaResult as RsPcaResult};
use molrs_compute::radius_of_gyration::RadiusOfGyration as RsRadiusOfGyration;
use molrs_compute::rdf::{RDF as RsRDF, RDFResult as RsRDFResult};
use molrs_compute::result::{ComputeResult, DescriptorRow};
use molrs_compute::traits::Compute;

use crate::core::frame::Frame;
use crate::core::types::JsFloatArray;
use js_sys::{Float64Array, Int32Array, Uint32Array};

// ---------------------------------------------------------------------------
// Helper: extract Nx3 position matrix from a core Frame
// ---------------------------------------------------------------------------

/// Extract an Nx3 position matrix from the `"atoms"` block of a core
/// [`Frame`](molrs::frame::Frame).
///
/// Reads the `x`, `y`, `z` columns (F, angstrom) and assembles
/// them into a contiguous row-major matrix.
fn positions_from_frame(frame: &molrs::frame::Frame) -> Result<ndarray::Array2<F>, JsValue> {
    let atoms = frame
        .get("atoms")
        .ok_or_else(|| JsValue::from_str("Frame has no 'atoms' block"))?;
    let get = |col: &str| -> Result<&[F], JsValue> {
        use molrs::block::BlockDtype;
        let c = atoms
            .get(col)
            .ok_or_else(|| JsValue::from_str(&format!("atoms block missing '{col}' column")))?;
        let arr = <F as BlockDtype>::from_column(c)
            .ok_or_else(|| JsValue::from_str(&format!("'{col}' column has wrong dtype")))?;
        arr.as_slice()
            .ok_or_else(|| JsValue::from_str(&format!("'{col}' column is not contiguous")))
    };
    let xs = get("x")?;
    let ys = get("y")?;
    let zs = get("z")?;
    let n = xs.len();
    let mut pos = ndarray::Array2::<F>::zeros((n, 3));
    for i in 0..n {
        pos[[i, 0]] = xs[i];
        pos[[i, 1]] = ys[i];
        pos[[i, 2]] = zs[i];
    }
    Ok(pos)
}

// ===========================================================================
// LinkedCell — cell-list based neighbor search
// ===========================================================================

/// Cell-list (linked-cell) based neighbor search.
///
/// Creates a spatial index from a [`Frame`]'s atom positions and
/// simulation box, then finds all neighbor pairs within the cutoff
/// distance.
///
/// All distances are in angstrom (A).
///
/// # Example (JavaScript)
///
/// ```js
/// const lc = new LinkedCell(3.0);       // cutoff = 3.0 A
/// const nlist = lc.build(frame);         // self-query (unique pairs i < j)
/// const cross = lc.query(ref, other);    // cross-query
///
/// console.log(nlist.numPairs);
/// ```
#[wasm_bindgen(js_name = LinkedCell)]
pub struct LinkedCell {
    cutoff: F,
}

#[wasm_bindgen(js_class = LinkedCell)]
impl LinkedCell {
    /// Create a linked-cell neighbor search with the given distance cutoff.
    ///
    /// # Arguments
    ///
    /// * `cutoff` - Maximum neighbor distance in angstrom (A)
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const lc = new LinkedCell(5.0);
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new(cutoff: F) -> Self {
        Self { cutoff }
    }

    /// Build a neighbor list from a [`Frame`] (self-query).
    ///
    /// Finds all unique pairs `(i < j)` of atoms within the cutoff
    /// distance using the cell-list algorithm.
    ///
    /// The frame must have an `"atoms"` block with `x`, `y`, `z` (F) columns.
    /// If the frame has a `simbox`, periodic boundary conditions are used.
    /// Otherwise, a free-boundary bounding box is auto-generated.
    ///
    /// # Arguments
    ///
    /// * `frame` - Frame with atom positions
    ///
    /// # Returns
    ///
    /// A [`NeighborList`] containing all unique pairs within the cutoff.
    ///
    /// # Errors
    ///
    /// Throws if the frame is missing required data.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const lc = new LinkedCell(3.0);
    /// const nlist = lc.build(frame);
    /// const dists = nlist.distances(); // Float32Array or Float64Array
    /// ```
    pub fn build(&self, frame: &Frame) -> Result<NeighborList, JsValue> {
        frame.with_frame(|rs_frame| {
            let pos = positions_from_frame(rs_frame)?;

            let simbox;
            let bx_ref = match rs_frame.simbox.as_ref() {
                Some(sb) => sb,
                None => {
                    simbox = molrs::region::simbox::SimBox::free(pos.view(), self.cutoff)
                        .map_err(|e| JsValue::from_str(&format!("free-boundary box: {e:?}")))?;
                    &simbox
                }
            };

            let mut lc = RsLinkCell::new().cutoff(self.cutoff);
            lc.build(pos.view(), bx_ref);
            let result = lc.query().clone();

            Ok(NeighborList { inner: result })
        })
    }

    /// Cross-query: find all pairs where `i` indexes query points and
    /// `j` indexes the reference points.
    ///
    /// # Arguments
    ///
    /// * `ref_frame` - Frame with reference atom positions
    /// * `query_frame` - Frame with query atom positions (must have
    ///   `"atoms"` block with `x`, `y`, `z` columns)
    ///
    /// # Returns
    ///
    /// A [`NeighborList`] containing all `(i, j, distance)` pairs
    /// within the cutoff.
    ///
    /// # Errors
    ///
    /// Throws if either frame is missing required columns.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const lc = new LinkedCell(3.0);
    /// const crossPairs = lc.query(refFrame, otherFrame);
    /// console.log(crossPairs.numPairs);
    /// ```
    pub fn query(&self, ref_frame: &Frame, query_frame: &Frame) -> Result<NeighborList, JsValue> {
        ref_frame.with_frame(|rs_ref| {
            let ref_pos = positions_from_frame(rs_ref)?;

            let aabb = match rs_ref.simbox.as_ref() {
                Some(sb) => RsNeighborQuery::new(sb, ref_pos.view(), self.cutoff),
                None => RsNeighborQuery::free(ref_pos.view(), self.cutoff),
            };

            query_frame.with_frame(|rs_query| {
                let query_pos = positions_from_frame(rs_query)?;
                let result = aabb.query(query_pos.view());
                Ok(NeighborList { inner: result })
            })
        })
    }
}

// ===========================================================================
// NeighborList
// ===========================================================================

/// Result of a neighbor search: all atom pairs within a distance cutoff.
///
/// Contains pair indices, distances, and squared distances for every
/// neighbor pair found. This object is produced by [`LinkedCell`]
/// and consumed by analysis classes like [`RDF`] and [`Cluster`].
///
/// # Properties
///
/// | Property | Type | Description |
/// |----------|------|-------------|
/// | `numPairs` | `number` | Total number of neighbor pairs |
/// | `numPoints` | `number` | Number of reference points |
/// | `numQueryPoints` | `number` | Number of query points |
/// | `isSelfQuery` | `boolean` | Whether this is a self-query result |
///
/// # Example (JavaScript)
///
/// ```js
/// const nlist = lc.build(frame);
/// console.log(nlist.numPairs);
///
/// const i = nlist.queryPointIndices(); // Uint32Array
/// const j = nlist.pointIndices();      // Uint32Array
/// const d = nlist.distances();         // Float32Array or Float64Array (in A)
/// ```
#[wasm_bindgen(js_name = NeighborList)]
pub struct NeighborList {
    inner: RsNeighborList,
}

#[wasm_bindgen(js_class = NeighborList)]
impl NeighborList {
    /// Total number of neighbor pairs found.
    #[wasm_bindgen(getter, js_name = numPairs)]
    pub fn num_pairs(&self) -> usize {
        self.inner.n_pairs()
    }

    /// Number of reference (target) points in the search.
    #[wasm_bindgen(getter, js_name = numPoints)]
    pub fn num_points(&self) -> usize {
        self.inner.num_points()
    }

    /// Number of query points in the search.
    ///
    /// For self-queries, this equals `numPoints`.
    #[wasm_bindgen(getter, js_name = numQueryPoints)]
    pub fn num_query_points(&self) -> usize {
        self.inner.num_query_points()
    }

    /// Whether this result came from a self-query (`build()`).
    ///
    /// In self-queries, only unique pairs `(i < j)` are reported.
    #[wasm_bindgen(getter, js_name = isSelfQuery)]
    pub fn is_self_query(&self) -> bool {
        self.inner.mode() == QueryMode::SelfQuery
    }

    /// Zero-copy `Uint32Array` view of query point indices (`i`) over
    /// WASM memory. **Invalidated** on any WASM memory growth — copy
    /// in JS (`new Uint32Array(view)`) if it needs to outlive later calls.
    #[wasm_bindgen(js_name = queryPointIndices)]
    pub fn query_point_indices(&self) -> Uint32Array {
        // SAFETY: view borrows wasm memory; short-lived use only.
        unsafe { Uint32Array::view(self.inner.query_point_indices()) }
    }

    /// Zero-copy `Uint32Array` view of reference point indices (`j`).
    /// Same invalidation caveat as [`queryPointIndices`](Self::query_point_indices).
    #[wasm_bindgen(js_name = pointIndices)]
    pub fn point_indices(&self) -> Uint32Array {
        // SAFETY: view borrows wasm memory; short-lived use only.
        unsafe { Uint32Array::view(self.inner.point_indices()) }
    }

    /// Pairwise distances in angstrom (A). Computed lazily from `distSq`.
    ///
    /// Returns an owned copy because distances are derived on the fly
    /// (`sqrt` per pair) rather than stored.
    pub fn distances(&self) -> Vec<F> {
        self.inner.distances()
    }

    /// Zero-copy `Float64Array` view of squared pairwise distances in A^2.
    /// Same invalidation caveat as [`queryPointIndices`](Self::query_point_indices).
    #[wasm_bindgen(js_name = distSq)]
    pub fn dist_sq(&self) -> JsFloatArray {
        // SAFETY: view borrows wasm memory; short-lived use only.
        unsafe { JsFloatArray::view(self.inner.dist_sq()) }
    }
}

// ===========================================================================
// RDF — Radial Distribution Function
// ===========================================================================

/// Radial distribution function g(r) analysis.
///
/// Bins neighbor-pair distances in `[rMin, rMax]` and normalizes by the
/// ideal-gas pair density. Defaults follow freud (`rMin = 0`). Periodic
/// systems take their normalization volume from `frame.simbox`; non-periodic
/// systems must supply it explicitly via [`computeWithVolume`].
///
/// # Algorithm
///
/// g(r) = n(r) / (rho * V_shell(r) * N_ref)
///
/// where `n(r)` is the pair count in bin `r`, `rho = N/V` is the number
/// density, and `V_shell(r)` is the shell volume for that bin.
///
/// # Example (JavaScript)
///
/// ```js
/// const lc = new LinkedCell(5.0);
/// const nlist = lc.build(frame);
///
/// const rdf = new RDF(100, 5.0);          // rMin defaults to 0
/// const result = rdf.compute(frame, nlist);
///
/// // Non-periodic frame: supply the normalization volume.
/// const resultFree = rdf.computeWithVolume(nlist, volumeA3);
///
/// const r  = result.binCenters();
/// const gr = result.rdf();
/// ```
#[wasm_bindgen(js_name = RDF)]
pub struct RDF {
    inner: RsRDF,
}

#[wasm_bindgen(js_class = RDF)]
impl RDF {
    /// Create a new RDF analysis.
    ///
    /// # Arguments
    ///
    /// * `n_bins` - Number of histogram bins
    /// * `r_max` - Upper radial cutoff in angstrom (A). Should be ≤ the
    ///   neighbor-search cutoff.
    /// * `r_min` - Lower radial cutoff in angstrom (A). Optional, defaults
    ///   to 0 (freud convention). Pairs with `d < rMin` or `d == 0` are
    ///   excluded from the histogram.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const rdf = new RDF(100, 5.0);       // rMin = 0
    /// const rdf2 = new RDF(100, 5.0, 0.5); // exclude d < 0.5 A
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new(n_bins: usize, r_max: F, r_min: Option<F>) -> Result<RDF, JsValue> {
        let inner = RsRDF::new(n_bins, r_max, r_min.unwrap_or(0.0))
            .map_err(|e| JsValue::from_str(&format!("RDF: {e}")))?;
        Ok(Self { inner })
    }

    /// Compute g(r) using the simulation-box volume from `frame.simbox`.
    ///
    /// # Arguments
    ///
    /// * `frame` - Frame with a `simbox` set (used only for volume)
    /// * `neighbors` - Pre-built [`NeighborList`] from [`LinkedCell`]
    ///
    /// # Errors
    ///
    /// Throws if the frame has no `simbox` — use
    /// [`computeWithVolume`](Self::compute_with_volume) for non-periodic frames.
    pub fn compute(&self, frame: &Frame, neighbors: &NeighborList) -> Result<RDFResult, JsValue> {
        frame.with_frame(|rs_frame| {
            let nlist_vec = vec![neighbors.inner.clone()];
            let result = self
                .inner
                .compute(&[rs_frame], &nlist_vec)
                .map_err(|e| JsValue::from_str(&format!("RDF compute: {e}")))?;
            Ok(RDFResult { inner: result })
        })
    }

    /// Compute g(r) using an explicit normalization volume (A^3).
    ///
    /// Use this for non-periodic systems or to override the box volume.
    /// Internally wraps the supplied volume as a cubic SimBox since the
    /// underlying [`molrs_compute::RDF`] pulls its normalization volume from
    /// `frame.simbox`.
    ///
    /// # Arguments
    ///
    /// * `neighbors` - Pre-built [`NeighborList`]
    /// * `volume` - Normalization volume in A^3 (must be finite and > 0)
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const result = rdf.computeWithVolume(nlist, 1000.0);
    /// ```
    #[wasm_bindgen(js_name = computeWithVolume)]
    pub fn compute_with_volume(
        &self,
        neighbors: &NeighborList,
        volume: F,
    ) -> Result<RDFResult, JsValue> {
        if !(volume.is_finite() && volume > 0.0) {
            return Err(JsValue::from_str(
                "RDF computeWithVolume: volume must be finite and > 0",
            ));
        }
        let box_len = volume.cbrt();
        let simbox = molrs::region::simbox::SimBox::cube(
            box_len,
            ndarray::array![0.0 as F, 0.0 as F, 0.0 as F],
            [false, false, false],
        )
        .map_err(|e| JsValue::from_str(&format!("RDF computeWithVolume: {e:?}")))?;
        let mut synth = molrs::frame::Frame::new();
        synth.simbox = Some(simbox);
        let nlist_vec = vec![neighbors.inner.clone()];
        let result = self
            .inner
            .compute(&[&synth], &nlist_vec)
            .map_err(|e| JsValue::from_str(&format!("RDF computeWithVolume: {e}")))?;
        Ok(RDFResult { inner: result })
    }
}

/// Result of a radial distribution function computation.
///
/// Contains the binned g(r) values, bin geometry, raw pair counts,
/// and normalization metadata.
///
/// # Example (JavaScript)
///
/// ```js
/// const result = rdf.compute(frame, nlist);
/// const r  = result.binCenters();  // Float32Array or Float64Array [0.025, 0.075, ...]
/// const gr = result.rdf();         // Float32Array or Float64Array, normalized g(r)
/// const nr = result.pairCounts();  // Float32Array or Float64Array, raw counts
/// console.log("Volume:", result.volume, "A^3");
/// console.log("N_ref:", result.numPoints);
/// ```
#[wasm_bindgen(js_name = RDFResult)]
pub struct RDFResult {
    inner: RsRDFResult,
}

#[wasm_bindgen(js_class = RDFResult)]
impl RDFResult {
    /// Zero-copy `Float64Array` view of bin center positions in A.
    /// Length equals `n_bins`. **Invalidated** on WASM memory growth;
    /// copy in JS if it needs to outlive later calls.
    #[wasm_bindgen(js_name = binCenters)]
    pub fn bin_centers(&self) -> JsFloatArray {
        // SAFETY: view borrows wasm memory; short-lived use only.
        unsafe { JsFloatArray::view(self.inner.bin_centers.as_slice().unwrap()) }
    }

    /// Zero-copy `Float64Array` view of bin edge positions in A.
    /// Length is `n_bins + 1`. Same invalidation caveat.
    #[wasm_bindgen(js_name = binEdges)]
    pub fn bin_edges(&self) -> JsFloatArray {
        // SAFETY: view borrows wasm memory; short-lived use only.
        unsafe { JsFloatArray::view(self.inner.bin_edges.as_slice().unwrap()) }
    }

    /// Zero-copy `Float64Array` view of normalized g(r). Same invalidation
    /// caveat.
    pub fn rdf(&self) -> JsFloatArray {
        // SAFETY: view borrows wasm memory; short-lived use only.
        unsafe { JsFloatArray::view(self.inner.rdf.as_slice().unwrap()) }
    }

    /// Zero-copy `Float64Array` view of raw (un-normalized) pair counts
    /// per bin. Same invalidation caveat.
    #[wasm_bindgen(js_name = pairCounts)]
    pub fn pair_counts(&self) -> JsFloatArray {
        // SAFETY: view borrows wasm memory; short-lived use only.
        unsafe { JsFloatArray::view(self.inner.n_r.as_slice().unwrap()) }
    }

    /// Number of reference points used in the normalization.
    #[wasm_bindgen(getter, js_name = numPoints)]
    pub fn num_points(&self) -> usize {
        self.inner.n_points
    }

    /// Normalization volume in A^3 (from the SimBox or the explicit caller value).
    #[wasm_bindgen(getter)]
    pub fn volume(&self) -> F {
        self.inner.volume
    }

    /// Inner cutoff in A (lower edge of bin 0).
    #[wasm_bindgen(getter, js_name = rMin)]
    pub fn r_min(&self) -> F {
        self.inner.r_min
    }
}

// ===========================================================================
// MSD — Mean Squared Displacement
// ===========================================================================

/// Mean squared displacement (MSD) analysis.
///
/// Computes MSD = |r(t) - r(0)|^2 for each particle and the system
/// average. The first frame fed is automatically used as the reference.
/// Useful for measuring diffusion coefficients via D = MSD / (6t).
///
/// All distances are in angstrom (A), so MSD is in A^2.
///
/// # Example (JavaScript)
///
/// ```js
/// const msd = new MSD();
/// for (const frame of trajectory) {
///   msd.feed(frame);         // first frame = reference
/// }
/// const results = msd.results();  // MSDResult[] per frame
/// console.log(results[10].mean);  // MSD at frame 10 in A^2
/// ```
///
/// # References
///
/// - Einstein, A. (1905). *Annalen der Physik*, 322(8), 549-560.
#[wasm_bindgen(js_name = MSD)]
pub struct MSD {
    frames: Vec<molrs::frame::Frame>,
}

#[allow(clippy::new_without_default)]
#[wasm_bindgen(js_class = MSD)]
impl MSD {
    /// Create an empty MSD analysis.
    ///
    /// The first frame passed to [`feed`] becomes the reference
    /// configuration (t=0).
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const msd = new MSD();
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self { frames: Vec::new() }
    }

    /// Feed a frame into the MSD analysis.
    ///
    /// Internally clones the frame's core data so subsequent mutations on
    /// the JS side (e.g. trajectory playback overwriting buffers) do not
    /// race against pending [`results`](Self::results) calls. The first
    /// frame sets the reference configuration.
    ///
    /// # Arguments
    ///
    /// * `frame` - Frame with `"atoms"` block containing
    ///   `x`, `y`, `z` (F) columns
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const msd = new MSD();
    /// msd.feed(frame0);  // sets reference
    /// msd.feed(frame1);  // added to trajectory
    /// const series = msd.results();
    /// ```
    pub fn feed(&mut self, frame: &Frame) -> Result<(), JsValue> {
        frame.with_frame(|rs_frame| {
            self.frames.push(rs_frame.clone());
            Ok(())
        })
    }

    /// Run the stateless [`molrs_compute::MSD`] over every fed frame and
    /// return the per-frame time series.
    ///
    /// The first frame is always the reference, so `results()[0].mean ≈ 0`.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const results = msd.results();
    /// results.forEach((r, t) => console.log(`t=${t}: MSD=${r.mean}`));
    /// ```
    pub fn results(&self) -> Result<Vec<MSDResult>, JsValue> {
        if self.frames.is_empty() {
            return Ok(Vec::new());
        }
        let refs: Vec<&molrs::frame::Frame> = self.frames.iter().collect();
        let series = RsMSD::new()
            .compute(&refs, ())
            .map_err(|e| JsValue::from_str(&format!("MSD results: {e}")))?;
        Ok(series
            .data
            .iter()
            .map(|r| MSDResult { inner: r.clone() })
            .collect())
    }

    /// Number of frames accumulated.
    #[wasm_bindgen(getter)]
    pub fn count(&self) -> usize {
        self.frames.len()
    }

    /// Reset the analysis, clearing the trajectory buffer.
    pub fn reset(&mut self) {
        self.frames.clear();
    }
}

/// Result of a mean squared displacement computation.
///
/// # Example (JavaScript)
///
/// ```js
/// const result = msd.compute(frame);
/// console.log(result.mean);              // number (A^2)
/// console.log(result.perParticle());     // Float32Array or Float64Array (A^2)
/// ```
#[wasm_bindgen(js_name = MSDResult)]
pub struct MSDResult {
    inner: RsMSDResult,
}

#[wasm_bindgen(js_class = MSDResult)]
impl MSDResult {
    /// System-average mean squared displacement in A^2.
    ///
    /// This is the arithmetic mean of all per-particle squared
    /// displacements: `mean = sum(|r_i(t) - r_i(0)|^2) / N`.
    #[wasm_bindgen(getter)]
    pub fn mean(&self) -> F {
        self.inner.mean
    }

    /// Zero-copy `Float64Array` view of per-particle squared displacements
    /// in A². `perParticle()[i]` is `|r_i(t) - r_i(0)|²` for particle `i`.
    /// **Invalidated** on WASM memory growth; copy in JS if needed.
    #[wasm_bindgen(js_name = perParticle)]
    pub fn per_particle(&self) -> JsFloatArray {
        // SAFETY: view borrows wasm memory; short-lived use only.
        unsafe { JsFloatArray::view(self.inner.per_particle.as_slice().unwrap()) }
    }
}

// ===========================================================================
// Cluster — Distance-based cluster analysis
// ===========================================================================

/// Distance-based cluster analysis using BFS on the neighbor graph.
///
/// Particles that are connected (directly or transitively) through
/// neighbor-list pairs are grouped into clusters. Clusters smaller
/// than `minClusterSize` are filtered out (their particles get
/// cluster ID = -1).
///
/// # Example (JavaScript)
///
/// ```js
/// const lc = new LinkedCell(2.0);
/// const nlist = lc.build(frame);
///
/// const cluster = new Cluster(5); // min 5 particles per cluster
/// const result = cluster.compute(frame, nlist);
///
/// console.log(result.numClusters);     // number of valid clusters
/// console.log(result.clusterIdx());    // Int32Array, per-particle IDs
/// console.log(result.clusterSizes());  // Uint32Array, size of each cluster
/// ```
#[wasm_bindgen(js_name = Cluster)]
pub struct Cluster {
    inner: RsCluster,
}

#[wasm_bindgen(js_class = Cluster)]
impl Cluster {
    /// Create a cluster analysis with a minimum cluster size filter.
    ///
    /// # Arguments
    ///
    /// * `min_cluster_size` - Minimum number of particles for a cluster
    ///   to be considered valid. Clusters with fewer particles are
    ///   discarded (their particles get cluster ID = -1).
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const cluster = new Cluster(5); // ignore clusters < 5 particles
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new(min_cluster_size: usize) -> Self {
        Self {
            inner: RsCluster::new(min_cluster_size),
        }
    }

    /// Run cluster analysis on a frame with pre-built neighbor pairs.
    ///
    /// # Arguments
    ///
    /// * `frame` - Frame with atom positions
    /// * `neighbors` - Pre-built [`NeighborList`] defining connectivity
    ///
    /// # Returns
    ///
    /// A [`ClusterResult`] with per-particle cluster IDs and cluster sizes.
    ///
    /// # Errors
    ///
    /// Throws if the frame cannot be cloned or the analysis fails.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const result = cluster.compute(frame, nlist);
    /// ```
    pub fn compute(
        &self,
        frame: &Frame,
        neighbors: &NeighborList,
    ) -> Result<ClusterResult, JsValue> {
        frame.with_frame(|rs_frame| {
            let nlist_vec = vec![neighbors.inner.clone()];
            let mut results = self
                .inner
                .compute(&[rs_frame], &nlist_vec)
                .map_err(|e| JsValue::from_str(&format!("Cluster compute: {e}")))?;
            let first = results
                .pop()
                .ok_or_else(|| JsValue::from_str("Cluster compute: empty result"))?;
            Ok(ClusterResult { inner: first })
        })
    }
}

/// Result of a distance-based cluster analysis.
///
/// # Example (JavaScript)
///
/// ```js
/// const result = cluster.compute(frame, nlist);
/// console.log(result.numClusters);       // number
///
/// const ids   = result.clusterIdx();     // Int32Array (per-particle)
/// const sizes = result.clusterSizes();   // Uint32Array (per-cluster)
///
/// // Particles in filtered-out clusters have id = -1
/// for (let i = 0; i < ids.length; i++) {
///   if (ids[i] === -1) console.log(`Particle ${i} not in any valid cluster`);
/// }
/// ```
#[wasm_bindgen(js_name = ClusterResult)]
pub struct ClusterResult {
    inner: RsClusterResult,
}

#[wasm_bindgen(js_class = ClusterResult)]
impl ClusterResult {
    /// Number of valid clusters found (after min-size filtering).
    #[wasm_bindgen(getter, js_name = numClusters)]
    pub fn num_clusters(&self) -> usize {
        self.inner.num_clusters
    }

    /// Per-particle cluster ID assignment as `Int32Array`.
    ///
    /// `clusterIdx()[i]` is the cluster ID for particle `i`.
    /// Particles in clusters smaller than `minClusterSize` are
    /// assigned ID = -1 (filtered out).
    ///
    /// Cluster IDs are zero-based and contiguous: `0, 1, ..., numClusters-1`.
    #[wasm_bindgen(js_name = clusterIdx)]
    pub fn cluster_idx(&self) -> Vec<i32> {
        self.inner.cluster_idx.iter().map(|&id| id as i32).collect()
    }

    /// Size (particle count) of each valid cluster as `Uint32Array`.
    ///
    /// `clusterSizes()[c]` is the number of particles in cluster `c`.
    /// Length equals `numClusters`.
    #[wasm_bindgen(js_name = clusterSizes)]
    pub fn cluster_sizes(&self) -> Vec<u32> {
        self.inner.cluster_sizes.iter().map(|&s| s as u32).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    /// Helper: create a Frame with N particles at given positions + cubic simbox.
    fn make_frame(positions: &[[F; 3]], box_len: F) -> Frame {
        use molrs::block::Block;
        use molrs::region::simbox::SimBox;
        use ndarray::{Array1, array};

        let x = Array1::from_iter(positions.iter().map(|p| p[0]));
        let y = Array1::from_iter(positions.iter().map(|p| p[1]));
        let z = Array1::from_iter(positions.iter().map(|p| p[2]));

        let mut block = Block::new();
        block.insert("x", x.into_dyn()).unwrap();
        block.insert("y", y.into_dyn()).unwrap();
        block.insert("z", z.into_dyn()).unwrap();

        let mut rs_frame = molrs::frame::Frame::new();
        rs_frame.insert("atoms", block);
        rs_frame.simbox =
            Some(SimBox::cube(box_len, array![0.0 as F, 0.0, 0.0], [false, false, false]).unwrap());

        Frame::from_rs(rs_frame).unwrap()
    }

    #[wasm_bindgen_test]
    fn linked_cell_build_finds_pairs() {
        let positions = [[1.0, 1.0, 1.0], [1.5, 1.0, 1.0], [8.0, 8.0, 8.0]];
        let frame = make_frame(&positions, 20.0);

        let lc = LinkedCell::new(2.0);
        let nbrs = lc.build(&frame).unwrap();
        assert!(nbrs.num_pairs() >= 1);
    }

    #[wasm_bindgen_test]
    fn rdf_runs() {
        let positions: Vec<[F; 3]> = (0..50)
            .map(|i| {
                let v = i as F * 0.2;
                [v % 10.0, (v * 1.3) % 10.0, (v * 1.7) % 10.0]
            })
            .collect();
        let frame = make_frame(&positions, 10.0);

        let lc = LinkedCell::new(4.0);
        let nbrs = lc.build(&frame).unwrap();

        let rdf = RDF::new(20, 4.0, None).unwrap();
        let result = rdf.compute(&frame, &nbrs).unwrap();

        assert_eq!(result.bin_centers().length(), 20);
        assert_eq!(result.rdf().length(), 20);
        assert_eq!(result.bin_edges().length(), 21);
    }

    #[wasm_bindgen_test]
    fn msd_feed_trajectory() {
        let ref_pos = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let cur_pos = [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
        let ref_frame = make_frame(&ref_pos, 20.0);
        let cur_frame = make_frame(&cur_pos, 20.0);

        let mut msd = MSD::new();
        msd.feed(&ref_frame).unwrap();
        msd.feed(&cur_frame).unwrap();

        assert_eq!(msd.count(), 2);
        let results = msd.results().unwrap();
        assert_eq!(results.len(), 2);

        // frame 0 vs itself = 0
        assert!(results[0].mean() < 1e-6);

        // frame 1: particle 0: d^2 = 1, particle 1: d^2 = 4, mean = 2.5
        assert!((results[1].mean() - 2.5).abs() < 1e-5);
    }

    #[wasm_bindgen_test]
    fn cluster_two_groups() {
        let positions = [
            [1.0, 1.0, 1.0],
            [1.5, 1.0, 1.0],
            [8.0, 8.0, 8.0],
            [8.5, 8.0, 8.0],
        ];
        let frame = make_frame(&positions, 20.0);

        let lc = LinkedCell::new(2.0);
        let nbrs = lc.build(&frame).unwrap();

        let cluster = Cluster::new(1);
        let result = cluster.compute(&frame, &nbrs).unwrap();

        assert_eq!(result.num_clusters(), 2);
        let idx = result.cluster_idx();
        assert_eq!(idx.len(), 4);
        assert_eq!(idx[0], idx[1]);
        assert_eq!(idx[2], idx[3]);
        assert_ne!(idx[0], idx[2]);
    }

    #[wasm_bindgen_test]
    fn cluster_min_size_filters() {
        let positions = [
            [1.0, 1.0, 1.0],
            [1.5, 1.0, 1.0],
            [8.0, 8.0, 8.0], // isolated
        ];
        let frame = make_frame(&positions, 20.0);

        let lc = LinkedCell::new(2.0);
        let nbrs = lc.build(&frame).unwrap();

        let cluster = Cluster::new(2);
        let result = cluster.compute(&frame, &nbrs).unwrap();

        assert_eq!(result.num_clusters(), 1);
        let idx = result.cluster_idx();
        assert_eq!(idx[2], -1); // filtered out
        assert!(idx[0] >= 0);
    }
}

// ===========================================================================
// ClusterCenters — Geometric cluster centers (MIC-aware)
// ===========================================================================

/// Geometric cluster centers with minimum image convention.
///
/// # Example (JavaScript)
///
/// ```js
/// const centers = new ClusterCenters().compute(frame, clusterResult);
/// // Float32Array or Float64Array [x0,y0,z0, x1,y1,z1, ...]
/// ```
#[wasm_bindgen(js_name = ClusterCenters)]
pub struct ClusterCenters {
    inner: RsClusterCenters,
}

#[allow(clippy::new_without_default)]
#[wasm_bindgen(js_class = ClusterCenters)]
impl ClusterCenters {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: RsClusterCenters::new(),
        }
    }

    /// Compute geometric centers. Returns a flat float typed array `[x0,y0,z0, ...]`.
    pub fn compute(
        &self,
        frame: &Frame,
        cluster_result: &ClusterResult,
    ) -> Result<Vec<F>, JsValue> {
        frame.with_frame(|rs_frame| {
            let clusters_vec = vec![cluster_result.inner.clone()];
            let mut results = self
                .inner
                .compute(&[rs_frame], &clusters_vec)
                .map_err(|e| JsValue::from_str(&format!("ClusterCenters: {e}")))?;
            let first = results
                .pop()
                .ok_or_else(|| JsValue::from_str("ClusterCenters: empty result"))?;
            Ok(first
                .centers
                .iter()
                .flat_map(|c| [c[0], c[1], c[2]])
                .collect())
        })
    }
}

// ===========================================================================
// CenterOfMass — Mass-weighted cluster centers
// ===========================================================================

/// Result of center-of-mass computation.
///
/// # Example (JavaScript)
///
/// ```js
/// const com = new CenterOfMass().compute(frame, clusterResult);
/// com.centersOfMass();   // Float32Array or Float64Array [x0,y0,z0, ...]
/// com.clusterMasses();   // Float32Array or Float64Array
/// ```
#[wasm_bindgen(js_name = CenterOfMassResult)]
pub struct CenterOfMassResult {
    inner: RsCOMResult,
}

#[wasm_bindgen(js_class = CenterOfMassResult)]
impl CenterOfMassResult {
    /// Zero-copy `Float64Array` view of mass-weighted centers, flat
    /// `[x0,y0,z0, x1,y1,z1, ...]`. **Invalidated** on WASM memory growth.
    #[wasm_bindgen(js_name = centersOfMass)]
    pub fn centers_of_mass(&self) -> JsFloatArray {
        // SAFETY: Vec<[F; 3]> is contiguous; `as_flattened` is safe.
        unsafe { JsFloatArray::view(self.inner.centers_of_mass.as_flattened()) }
    }

    /// Zero-copy `Float64Array` view of total mass per cluster.
    /// **Invalidated** on WASM memory growth.
    #[wasm_bindgen(js_name = clusterMasses)]
    pub fn cluster_masses(&self) -> JsFloatArray {
        // SAFETY: view borrows wasm memory; short-lived use only.
        unsafe { JsFloatArray::view(&self.inner.cluster_masses) }
    }

    /// Number of clusters.
    #[wasm_bindgen(getter, js_name = numClusters)]
    pub fn num_clusters(&self) -> usize {
        self.inner.centers_of_mass.len()
    }
}

/// Mass-weighted cluster center calculator.
#[wasm_bindgen(js_name = CenterOfMass)]
pub struct CenterOfMass {
    masses: Option<Vec<F>>,
}

#[wasm_bindgen(js_class = CenterOfMass)]
impl CenterOfMass {
    /// Create a center-of-mass calculator.
    ///
    /// Pass `null` for uniform masses, or a float typed array of per-particle masses.
    #[wasm_bindgen(constructor)]
    pub fn new(masses: Option<Vec<F>>) -> Self {
        Self { masses }
    }

    /// Compute centers of mass.
    pub fn compute(
        &self,
        frame: &Frame,
        cluster_result: &ClusterResult,
    ) -> Result<CenterOfMassResult, JsValue> {
        frame.with_frame(|rs_frame| {
            let calc = if let Some(ref ms) = self.masses {
                RsCenterOfMass::new().with_masses(ms)
            } else {
                RsCenterOfMass::new()
            };
            let clusters_vec = vec![cluster_result.inner.clone()];
            let mut results = calc
                .compute(&[rs_frame], &clusters_vec)
                .map_err(|e| JsValue::from_str(&format!("CenterOfMass: {e}")))?;
            let first = results
                .pop()
                .ok_or_else(|| JsValue::from_str("CenterOfMass: empty result"))?;
            Ok(CenterOfMassResult { inner: first })
        })
    }
}

// ===========================================================================
// GyrationTensor
// ===========================================================================

/// Gyration tensor per cluster.
///
/// Returns flat array: `[g00,g01,g02, g10,g11,g12, g20,g21,g22, ...]` per cluster.
#[wasm_bindgen(js_name = GyrationTensor)]
pub struct GyrationTensor {
    inner: RsGyrationTensor,
}

#[allow(clippy::new_without_default)]
#[wasm_bindgen(js_class = GyrationTensor)]
impl GyrationTensor {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: RsGyrationTensor::new(),
        }
    }

    /// Compute gyration tensors. Returns a flat float typed array (9 values per cluster).
    ///
    /// Internally computes the cluster geometric centers (via
    /// [`RsClusterCenters`]) since the new compute trait exposes them as a
    /// required upstream — the old single-frame wasm API hides this detail.
    pub fn compute(
        &self,
        frame: &Frame,
        cluster_result: &ClusterResult,
    ) -> Result<Vec<F>, JsValue> {
        frame.with_frame(|rs_frame| {
            let clusters_vec = vec![cluster_result.inner.clone()];
            let centers = RsClusterCenters::new()
                .compute(&[rs_frame], &clusters_vec)
                .map_err(|e| JsValue::from_str(&format!("GyrationTensor centers: {e}")))?;
            let mut tensors = self
                .inner
                .compute(&[rs_frame], (&clusters_vec, &centers))
                .map_err(|e| JsValue::from_str(&format!("GyrationTensor: {e}")))?;
            let first = tensors
                .pop()
                .ok_or_else(|| JsValue::from_str("GyrationTensor: empty result"))?;
            Ok(first
                .0
                .iter()
                .flat_map(|t| t.iter().flat_map(|row| row.iter().copied()))
                .collect())
        })
    }
}

// ===========================================================================
// InertiaTensor
// ===========================================================================

/// Moment of inertia tensor per cluster.
#[wasm_bindgen(js_name = InertiaTensor)]
pub struct InertiaTensor {
    masses: Option<Vec<F>>,
}

#[wasm_bindgen(js_class = InertiaTensor)]
impl InertiaTensor {
    #[wasm_bindgen(constructor)]
    pub fn new(masses: Option<Vec<F>>) -> Self {
        Self { masses }
    }

    /// Compute inertia tensors. Returns a flat float typed array (9 values per cluster).
    ///
    /// Internally computes the cluster centers of mass (via
    /// [`RsCenterOfMass`]) since the new compute trait consumes them as a
    /// required upstream — the old single-frame wasm API hides this detail.
    pub fn compute(
        &self,
        frame: &Frame,
        cluster_result: &ClusterResult,
    ) -> Result<Vec<F>, JsValue> {
        frame.with_frame(|rs_frame| {
            let com_calc = if let Some(ref ms) = self.masses {
                RsCenterOfMass::new().with_masses(ms)
            } else {
                RsCenterOfMass::new()
            };
            let clusters_vec = vec![cluster_result.inner.clone()];
            let coms = com_calc
                .compute(&[rs_frame], &clusters_vec)
                .map_err(|e| JsValue::from_str(&format!("InertiaTensor COM: {e}")))?;
            let calc = if let Some(ref ms) = self.masses {
                RsInertiaTensor::new().with_masses(ms)
            } else {
                RsInertiaTensor::new()
            };
            let mut tensors = calc
                .compute(&[rs_frame], (&clusters_vec, &coms))
                .map_err(|e| JsValue::from_str(&format!("InertiaTensor: {e}")))?;
            let first = tensors
                .pop()
                .ok_or_else(|| JsValue::from_str("InertiaTensor: empty result"))?;
            Ok(first
                .0
                .iter()
                .flat_map(|t| t.iter().flat_map(|row| row.iter().copied()))
                .collect())
        })
    }
}

// ===========================================================================
// RadiusOfGyration
// ===========================================================================

/// Radius of gyration per cluster.
#[wasm_bindgen(js_name = RadiusOfGyration)]
pub struct RadiusOfGyration {
    masses: Option<Vec<F>>,
}

#[wasm_bindgen(js_class = RadiusOfGyration)]
impl RadiusOfGyration {
    #[wasm_bindgen(constructor)]
    pub fn new(masses: Option<Vec<F>>) -> Self {
        Self { masses }
    }

    /// Compute radii of gyration. Returns a float typed array of length `numClusters`.
    ///
    /// Internally computes the cluster centers of mass so the single-frame
    /// wasm signature `(frame, cluster)` stays stable despite the new
    /// compute trait needing explicit COM upstream.
    pub fn compute(
        &self,
        frame: &Frame,
        cluster_result: &ClusterResult,
    ) -> Result<Vec<F>, JsValue> {
        frame.with_frame(|rs_frame| {
            let com_calc = if let Some(ref ms) = self.masses {
                RsCenterOfMass::new().with_masses(ms)
            } else {
                RsCenterOfMass::new()
            };
            let clusters_vec = vec![cluster_result.inner.clone()];
            let coms = com_calc
                .compute(&[rs_frame], &clusters_vec)
                .map_err(|e| JsValue::from_str(&format!("RadiusOfGyration COM: {e}")))?;
            let calc = if let Some(ref ms) = self.masses {
                RsRadiusOfGyration::new().with_masses(ms)
            } else {
                RsRadiusOfGyration::new()
            };
            let mut radii = calc
                .compute(&[rs_frame], (&clusters_vec, &coms))
                .map_err(|e| JsValue::from_str(&format!("RadiusOfGyration: {e}")))?;
            let first = radii
                .pop()
                .ok_or_else(|| JsValue::from_str("RadiusOfGyration: empty result"))?;
            Ok(first.0)
        })
    }
}

// ===========================================================================
// Topology — Graph-based molecular topology (igraph-style API)
// ===========================================================================

/// Graph-based molecular topology with automated detection of angles,
/// dihedrals, impropers, connected components, and rings (SSSR).
///
/// API mirrors igraph / molpy conventions.
///
/// # Example (JavaScript)
///
/// ```js
/// const topo = Topology.fromFrame(frame);
/// console.log(topo.nAtoms, topo.nBonds);
///
/// const angles = topo.angles();       // Uint32Array [i,j,k, ...]
/// const dihedrals = topo.dihedrals(); // Uint32Array [i,j,k,l, ...]
/// const cc = topo.connectedComponents(); // Int32Array per-atom labels
///
/// const rings = topo.findRings();
/// console.log(rings.numRings);
/// ```
#[wasm_bindgen(js_name = Topology)]
pub struct WasmTopology {
    inner: RsTopology,
}

#[wasm_bindgen(js_class = Topology)]
impl WasmTopology {
    /// Create a topology with `n` atoms and no bonds.
    #[wasm_bindgen(constructor)]
    pub fn new(n_atoms: usize) -> Self {
        Self {
            inner: RsTopology::with_atoms(n_atoms),
        }
    }

    /// Build a topology from a Frame's `bonds` block.
    ///
    /// Reads the `atoms` block for atom count and `bonds` block for
    /// `i`, `j` columns (Uint32).
    #[wasm_bindgen(js_name = fromFrame)]
    pub fn from_frame(frame: &Frame) -> Result<WasmTopology, JsValue> {
        frame.with_frame(|rs_frame| {
            let atoms = rs_frame
                .get("atoms")
                .ok_or_else(|| JsValue::from_str("Frame has no 'atoms' block"))?;
            let n_atoms = atoms
                .nrows()
                .ok_or_else(|| JsValue::from_str("atoms block is empty"))?;

            let mut topo = RsTopology::with_atoms(n_atoms);

            if let Some(bonds) = rs_frame.get("bonds") {
                use molrs::block::BlockDtype;
                let col_i = bonds
                    .get("i")
                    .and_then(|c| <u32 as BlockDtype>::from_column(c))
                    .and_then(|a| a.as_slice());
                let col_j = bonds
                    .get("j")
                    .and_then(|c| <u32 as BlockDtype>::from_column(c))
                    .and_then(|a| a.as_slice());

                if let (Some(is), Some(js)) = (col_i, col_j) {
                    let pairs: Vec<[usize; 2]> = is
                        .iter()
                        .zip(js.iter())
                        .map(|(&i, &j)| [i as usize, j as usize])
                        .collect();
                    topo.add_bonds(&pairs);
                }
            }

            Ok(Self { inner: topo })
        })
    }

    /// Number of atoms (vertices).
    #[wasm_bindgen(getter, js_name = nAtoms)]
    pub fn n_atoms(&self) -> usize {
        self.inner.n_atoms()
    }

    /// Number of bonds (edges).
    #[wasm_bindgen(getter, js_name = nBonds)]
    pub fn n_bonds(&self) -> usize {
        self.inner.n_bonds()
    }

    /// Number of unique angles.
    #[wasm_bindgen(getter, js_name = nAngles)]
    pub fn n_angles(&self) -> usize {
        self.inner.n_angles()
    }

    /// Number of unique proper dihedrals.
    #[wasm_bindgen(getter, js_name = nDihedrals)]
    pub fn n_dihedrals(&self) -> usize {
        self.inner.n_dihedrals()
    }

    /// Number of connected components.
    #[wasm_bindgen(getter, js_name = nComponents)]
    pub fn n_components(&self) -> usize {
        self.inner.n_components()
    }

    /// All bond pairs as flat `Uint32Array` `[i0,j0, i1,j1, ...]`.
    pub fn bonds(&self) -> Vec<u32> {
        self.inner
            .bonds()
            .iter()
            .flat_map(|b| [b[0] as u32, b[1] as u32])
            .collect()
    }

    /// All angle triplets as flat `Uint32Array` `[i,j,k, ...]`.
    pub fn angles(&self) -> Vec<u32> {
        self.inner
            .angles()
            .iter()
            .flat_map(|a| [a[0] as u32, a[1] as u32, a[2] as u32])
            .collect()
    }

    /// All proper dihedral quartets as flat `Uint32Array` `[i,j,k,l, ...]`.
    pub fn dihedrals(&self) -> Vec<u32> {
        self.inner
            .dihedrals()
            .iter()
            .flat_map(|d| [d[0] as u32, d[1] as u32, d[2] as u32, d[3] as u32])
            .collect()
    }

    /// All improper dihedral quartets as flat `Uint32Array` `[center,i,j,k, ...]`.
    pub fn impropers(&self) -> Vec<u32> {
        self.inner
            .impropers()
            .iter()
            .flat_map(|d| [d[0] as u32, d[1] as u32, d[2] as u32, d[3] as u32])
            .collect()
    }

    /// Per-atom connected component labels as `Int32Array`.
    ///
    /// Labels are 0-based and contiguous. Each atom gets a component ID.
    /// Atoms in the same connected subgraph share the same label.
    #[wasm_bindgen(js_name = connectedComponents)]
    pub fn connected_components(&self) -> Vec<i32> {
        self.inner
            .connected_components()
            .iter()
            .map(|&c| c as i32)
            .collect()
    }

    /// Neighbor atom indices of atom `idx` as `Uint32Array`.
    pub fn neighbors(&self, idx: usize) -> Vec<u32> {
        self.inner
            .neighbors(idx)
            .iter()
            .map(|&n| n as u32)
            .collect()
    }

    /// Degree (number of bonds) of atom `idx`.
    pub fn degree(&self, idx: usize) -> usize {
        self.inner.degree(idx)
    }

    /// Whether atoms `i` and `j` are directly bonded.
    #[wasm_bindgen(js_name = areBonded)]
    pub fn are_bonded(&self, i: usize, j: usize) -> bool {
        self.inner.are_bonded(i, j)
    }

    /// Add a single atom.
    #[wasm_bindgen(js_name = addAtom)]
    pub fn add_atom(&mut self) {
        self.inner.add_atom();
    }

    /// Add a bond between atoms `i` and `j`.
    #[wasm_bindgen(js_name = addBond)]
    pub fn add_bond(&mut self, i: usize, j: usize) {
        self.inner.add_bond(i, j);
    }

    /// Delete an atom by index.
    #[wasm_bindgen(js_name = deleteAtom)]
    pub fn delete_atom(&mut self, idx: usize) {
        self.inner.delete_atom(idx);
    }

    /// Delete a bond by edge index.
    #[wasm_bindgen(js_name = deleteBond)]
    pub fn delete_bond(&mut self, idx: usize) {
        self.inner.delete_bond(idx);
    }

    /// Compute the Smallest Set of Smallest Rings (SSSR).
    #[wasm_bindgen(js_name = findRings)]
    pub fn find_rings(&self) -> TopologyRingInfo {
        TopologyRingInfo {
            inner: self.inner.find_rings(),
        }
    }
}

// ===========================================================================
// TopologyRingInfo — SSSR ring detection result
// ===========================================================================

/// Result of ring detection (SSSR) on a topology graph.
///
/// # Example (JavaScript)
///
/// ```js
/// const rings = topo.findRings();
/// console.log(rings.numRings);
/// console.log(rings.ringSizes());   // Uint32Array
/// console.log(rings.isAtomInRing(0));
///
/// // Get all rings as flat array [size0, idx0_0, idx0_1, ..., size1, ...]
/// const data = rings.rings();
/// ```
#[wasm_bindgen(js_name = TopologyRingInfo)]
pub struct TopologyRingInfo {
    inner: RsTopologyRingInfo,
}

#[wasm_bindgen(js_class = TopologyRingInfo)]
impl TopologyRingInfo {
    /// Total number of rings detected.
    #[wasm_bindgen(getter, js_name = numRings)]
    pub fn num_rings(&self) -> usize {
        self.inner.num_rings()
    }

    /// Size of each ring as `Uint32Array`.
    #[wasm_bindgen(js_name = ringSizes)]
    pub fn ring_sizes(&self) -> Vec<u32> {
        self.inner.ring_sizes().iter().map(|&s| s as u32).collect()
    }

    /// Whether atom `idx` belongs to any ring.
    #[wasm_bindgen(js_name = isAtomInRing)]
    pub fn is_atom_in_ring(&self, idx: usize) -> bool {
        self.inner.is_atom_in_ring(idx)
    }

    /// Number of rings containing atom `idx`.
    #[wasm_bindgen(js_name = numAtomRings)]
    pub fn num_atom_rings(&self, idx: usize) -> usize {
        self.inner.num_atom_rings(idx)
    }

    /// Per-atom boolean mask as `Uint8Array` (0 or 1). 1 if atom is in any ring.
    #[wasm_bindgen(js_name = atomRingMask)]
    pub fn atom_ring_mask(&self, n_atoms: usize) -> Vec<u8> {
        self.inner
            .atom_ring_mask(n_atoms)
            .iter()
            .map(|&b| b as u8)
            .collect()
    }

    /// All rings as flat `Uint32Array` with length-prefixed encoding:
    /// `[size0, atom0, atom1, ..., size1, atom0, atom1, ...]`.
    pub fn rings(&self) -> Vec<u32> {
        let mut out = Vec::new();
        for ring in self.inner.rings() {
            out.push(ring.len() as u32);
            for &idx in ring {
                out.push(idx as u32);
            }
        }
        out
    }
}

// ===========================================================================
// PCA — 2-component Principal Component Analysis
// ===========================================================================

/// Stateless wrapper for [`molrs_compute::pca::Pca2`].
///
/// All configuration lives on [`fitTransform`](Self::fit_transform).
///
/// # Example (JavaScript)
///
/// ```js
/// const pca = new WasmPca2();
/// const result = pca.fitTransform(matrix, nRows, nCols);
/// const coords   = result.coords();    // Float64Array, length 2 * nRows
/// const variance = result.variance();  // Float64Array, length 2
/// ```
#[wasm_bindgen]
pub struct WasmPca2;

#[allow(clippy::new_without_default)]
#[wasm_bindgen(js_class = WasmPca2)]
impl WasmPca2 {
    /// Create a new PCA calculator. The struct carries no state — all
    /// parameters are supplied on [`fitTransform`](Self::fit_transform).
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmPca2 {
        WasmPca2
    }

    /// Fit 2-component PCA on a row-major observation matrix and return the
    /// projected coordinates + per-component variance.
    ///
    /// # Arguments
    ///
    /// * `matrix` — row-major `n_rows × n_cols` observation matrix.
    /// * `n_rows` — number of observations.
    /// * `n_cols` — number of features.
    ///
    /// # Errors
    ///
    /// Throws if `n_rows < 3`, `n_cols < 2`, the length does not match
    /// `n_rows * n_cols`, any element is non-finite, or any column has
    /// zero variance.
    #[wasm_bindgen(js_name = fitTransform)]
    pub fn fit_transform(
        &self,
        matrix: &[F],
        n_rows: usize,
        n_cols: usize,
    ) -> Result<WasmPcaResult, JsValue> {
        if matrix.len() != n_rows * n_cols {
            return Err(JsValue::from_str(&format!(
                "PCA: matrix length {} != n_rows * n_cols = {} * {}",
                matrix.len(),
                n_rows,
                n_cols
            )));
        }
        let rows: Vec<PcaRow> = (0..n_rows)
            .map(|i| PcaRow(matrix[i * n_cols..(i + 1) * n_cols].to_vec()))
            .collect();
        let dummy = molrs::frame::Frame::new();
        RsPca2::<PcaRow>::new()
            .compute(&[&dummy], &rows)
            .map(|inner| WasmPcaResult { inner })
            .map_err(|e| JsValue::from_str(&format!("PCA: {e}")))
    }
}

/// Row adapter so the stateless `Pca2` can consume caller matrices without
/// requiring a downstream molrs-compute type.
#[derive(Clone)]
struct PcaRow(Vec<F>);

impl DescriptorRow for PcaRow {
    fn as_row(&self) -> &[F] {
        &self.0
    }
}

impl ComputeResult for PcaRow {}

/// Result of a [`WasmPca2::fit_transform`] call.
///
/// Each accessor returns an **owned** `Float64Array` (copy of the underlying
/// `Vec`) so JS is free to let this wrapper be GC'd without dangling views.
#[wasm_bindgen]
pub struct WasmPcaResult {
    inner: RsPcaResult,
}

#[wasm_bindgen(js_class = WasmPcaResult)]
impl WasmPcaResult {
    /// Projected 2D coordinates as a row-major `Float64Array` of length
    /// `2 * n_rows`. `coords[2 * i + 0]` is the PC1 score for row `i`,
    /// `coords[2 * i + 1]` is PC2.
    pub fn coords(&self) -> Float64Array {
        let out = Float64Array::new_with_length(self.inner.coords.len() as u32);
        out.copy_from(&self.inner.coords);
        out
    }

    /// Explained variance per component as `Float64Array` of length 2.
    /// `variance[0] >= variance[1]` by construction.
    pub fn variance(&self) -> Float64Array {
        let out = Float64Array::new_with_length(2);
        out.copy_from(&self.inner.variance);
        out
    }
}

// ===========================================================================
// k-means — with k-means++ init
// ===========================================================================

/// Wrapper for [`molrs_compute::kmeans::KMeans`].
///
/// # Example (JavaScript)
///
/// ```js
/// const km = new WasmKMeans(3, 100, 42);
/// const labels = km.fit(coords, nRows, 2);   // Int32Array
/// ```
#[wasm_bindgen]
pub struct WasmKMeans {
    inner: RsKMeans,
}

#[wasm_bindgen(js_class = WasmKMeans)]
impl WasmKMeans {
    /// Create a new k-means configuration.
    ///
    /// # Arguments
    ///
    /// * `k` — number of clusters (>= 1).
    /// * `max_iter` — maximum Lloyd iterations (>= 1).
    /// * `seed` — RNG seed for k-means++ initialization. Cast to `u64`
    ///   internally (JS numbers are `f64`; integers up to 2^53 pass
    ///   through losslessly).
    ///
    /// # Errors
    ///
    /// Throws if `k == 0` or `max_iter == 0`.
    #[wasm_bindgen(constructor)]
    pub fn new(k: usize, max_iter: usize, seed: f64) -> Result<WasmKMeans, JsValue> {
        let seed_u64 = seed as u64;
        RsKMeans::new(k, max_iter, seed_u64)
            .map(|inner| WasmKMeans { inner })
            .map_err(|e| JsValue::from_str(&format!("KMeans: {e}")))
    }

    /// Cluster a row-major `n_rows × n_dims` coordinate matrix.
    ///
    /// # Returns
    ///
    /// Cluster labels in `0..k` as an owned `Int32Array`, one per row.
    ///
    /// # Errors
    ///
    /// Throws if `k > n_rows`, `n_dims == 0`, the length does not match
    /// `n_rows * n_dims`, or any element is non-finite.
    pub fn fit(&self, coords: &[F], n_rows: usize, n_dims: usize) -> Result<Int32Array, JsValue> {
        if n_dims != 2 {
            return Err(JsValue::from_str(
                "KMeans wasm binding supports n_dims=2 only (PCA-score input)",
            ));
        }
        if coords.len() != n_rows * n_dims {
            return Err(JsValue::from_str(&format!(
                "KMeans: coords length {} != n_rows * n_dims = {} * {}",
                coords.len(),
                n_rows,
                n_dims
            )));
        }
        let pca = molrs_compute::pca::PcaResult {
            coords: coords.to_vec(),
            variance: [0.0 as F, 0.0 as F],
        };
        let dummy = molrs::frame::Frame::new();
        let labels = self
            .inner
            .compute(&[&dummy], &pca)
            .map_err(|e| JsValue::from_str(&format!("KMeans fit: {e}")))?;
        let out = Int32Array::new_with_length(labels.0.len() as u32);
        out.copy_from(&labels.0);
        Ok(out)
    }
}
