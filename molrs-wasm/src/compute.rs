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

use molrs::compute::center_of_mass::{
    CenterOfMass as RsCenterOfMass, CenterOfMassResult as RsCenterOfMassResult,
};
use molrs::compute::cluster::{Cluster as RsCluster, ClusterResult as RsClusterResult};
use molrs::compute::cluster_centers::ClusterCenters as RsClusterCenters;
use molrs::compute::gyration_tensor::GyrationTensor as RsGyrationTensor;
use molrs::compute::inertia_tensor::InertiaTensor as RsInertiaTensor;
use molrs::compute::msd::{MSD as RsMSD, MSDResult as RsMSDResult};
use molrs::compute::radius_of_gyration::RadiusOfGyration as RsRadiusOfGyration;
use molrs::compute::rdf::{RDF as RsRDF, RDFResult as RsRDFResult};
use molrs::compute::traits::Compute;
use molrs::neighbors::{
    LinkCell as RsLinkCell, NbListAlgo, NeighborList as RsNeighborList,
    NeighborQuery as RsNeighborQuery, QueryMode,
};
use molrs::types::F;

use crate::core::frame::Frame;

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
        let rs_frame = frame.clone_core_frame()?;
        let pos = positions_from_frame(&rs_frame)?;

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
        let rs_ref = ref_frame.clone_core_frame()?;
        let ref_pos = positions_from_frame(&rs_ref)?;

        let aabb = match rs_ref.simbox.as_ref() {
            Some(sb) => RsNeighborQuery::new(sb, ref_pos.view(), self.cutoff),
            None => RsNeighborQuery::free(ref_pos.view(), self.cutoff),
        };

        let rs_query = query_frame.clone_core_frame()?;
        let query_pos = positions_from_frame(&rs_query)?;
        let result = aabb.query(query_pos.view());

        Ok(NeighborList { inner: result })
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

    /// Query point indices (`i`) for each pair, as a `Uint32Array`.
    ///
    /// The `k`-th pair connects query point `queryPointIndices()[k]`
    /// to reference point `pointIndices()[k]`.
    #[wasm_bindgen(js_name = queryPointIndices)]
    pub fn query_point_indices(&self) -> Vec<u32> {
        self.inner.query_point_indices().to_vec()
    }

    /// Reference point indices (`j`) for each pair, as a `Uint32Array`.
    #[wasm_bindgen(js_name = pointIndices)]
    pub fn point_indices(&self) -> Vec<u32> {
        self.inner.point_indices().to_vec()
    }

    /// Pairwise distances in angstrom (A), as a float typed array.
    ///
    /// `distances()[k]` is the distance between query point
    /// `queryPointIndices()[k]` and reference point `pointIndices()[k]`.
    pub fn distances(&self) -> Vec<F> {
        self.inner.distances()
    }

    /// Squared pairwise distances in A^2, as a float typed array.
    ///
    /// More efficient than `distances()` when you only need to
    /// compare or threshold distances.
    #[wasm_bindgen(js_name = distSq)]
    pub fn dist_sq(&self) -> Vec<F> {
        self.inner.dist_sq().to_vec()
    }
}

// ===========================================================================
// RDF — Radial Distribution Function
// ===========================================================================

/// Radial distribution function g(r) analysis.
///
/// Computes the pair correlation function by binning neighbor-pair
/// distances into a histogram and normalizing by the ideal-gas
/// density (spherical shell volume normalization).
///
/// # Algorithm
///
/// g(r) = n(r) / (rho * V_shell(r) * N_ref)
///
/// where `n(r)` is the pair count in bin `r`, `rho` is the number
/// density, and `V_shell(r)` is the shell volume for that bin.
///
/// # Example (JavaScript)
///
/// ```js
/// const lc = new LinkedCell(5.0);
/// const nlist = lc.build(frame);
///
/// const rdf = new RDF(100, 5.0);
/// const result = rdf.compute(frame, nlist);
///
/// const r  = result.binCenters();  // Float32Array or Float64Array, bin centers in A
/// const gr = result.rdf();         // Float32Array or Float64Array, g(r) values
/// ```
#[wasm_bindgen(js_name = RDF)]
pub struct RDF {
    inner: RsRDF,
}

#[wasm_bindgen(js_class = RDF)]
impl RDF {
    /// Create a new RDF analysis with specified binning.
    ///
    /// # Arguments
    ///
    /// * `n_bins` - Number of histogram bins
    /// * `r_max` - Maximum radial distance in angstrom (A). Should
    ///   match or be less than the neighbor-search cutoff.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const rdf = new RDF(100, 5.0); // 100 bins up to 5 A
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new(n_bins: usize, r_max: F) -> Self {
        Self {
            inner: RsRDF::new(n_bins, r_max),
        }
    }

    /// Compute the RDF from a pre-built neighbor list.
    ///
    /// The frame is needed to read the simulation box volume for
    /// normalization.
    ///
    /// # Arguments
    ///
    /// * `frame` - Frame with a `simbox` set (for volume normalization)
    /// * `neighbors` - Pre-built [`NeighborList`] from [`LinkedCell`]
    ///
    /// # Returns
    ///
    /// An [`RDFResult`] containing bin centers, g(r) values, and raw
    /// pair counts.
    ///
    /// # Errors
    ///
    /// Throws if the frame cannot be cloned or the computation fails.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const result = rdf.compute(frame, nlist);
    /// const gr = result.rdf(); // Float32Array or Float64Array
    /// ```
    pub fn compute(&self, frame: &Frame, neighbors: &NeighborList) -> Result<RDFResult, JsValue> {
        let rs_frame = frame.clone_core_frame()?;
        let result = self
            .inner
            .compute(&rs_frame, &neighbors.inner)
            .map_err(|e| JsValue::from_str(&format!("RDF compute: {e}")))?;
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
    /// Bin center positions as a float typed array in angstrom (A).
    ///
    /// Length equals `n_bins` (the value passed to the `RDF` constructor).
    #[wasm_bindgen(js_name = binCenters)]
    pub fn bin_centers(&self) -> Vec<F> {
        self.inner.bin_centers.to_vec()
    }

    /// Bin edge positions as a float typed array in angstrom (A).
    ///
    /// Length is `n_bins + 1` (one more than bin centers).
    #[wasm_bindgen(js_name = binEdges)]
    pub fn bin_edges(&self) -> Vec<F> {
        self.inner.bin_edges.to_vec()
    }

    /// Normalized g(r) values as a float typed array (dimensionless).
    ///
    /// A uniform ideal gas has g(r) = 1.0 everywhere. Peaks indicate
    /// preferred interatomic distances (coordination shells).
    pub fn rdf(&self) -> Vec<F> {
        self.inner.rdf.to_vec()
    }

    /// Raw (un-normalized) pair counts per bin as a float typed array.
    #[wasm_bindgen(js_name = pairCounts)]
    pub fn pair_counts(&self) -> Vec<F> {
        self.inner.n_r.to_vec()
    }

    /// Number of reference points used in the normalization.
    #[wasm_bindgen(getter, js_name = numPoints)]
    pub fn num_points(&self) -> usize {
        self.inner.n_points
    }

    /// Simulation box volume used in the normalization, in A^3.
    #[wasm_bindgen(getter)]
    pub fn volume(&self) -> F {
        self.inner.volume
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
    inner: RsMSD,
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
        Self {
            inner: RsMSD::new(),
        }
    }

    /// Feed a frame into the MSD analysis.
    ///
    /// The first frame sets the reference configuration.
    /// Subsequent frames compute MSD relative to that reference.
    ///
    /// # Arguments
    ///
    /// * `frame` - Frame with `"atoms"` block containing
    ///   `x`, `y`, `z` (F) columns
    ///
    /// # Errors
    ///
    /// Throws if the frame is missing required columns or has a
    /// different number of atoms than the reference.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const msd = new MSD();
    /// msd.feed(frame0);  // sets reference
    /// msd.feed(frame1);  // computes MSD vs frame0
    /// ```
    pub fn feed(&mut self, frame: &Frame) -> Result<(), JsValue> {
        let rs_frame = frame.clone_core_frame()?;
        self.inner
            .feed(&rs_frame)
            .map_err(|e| JsValue::from_str(&format!("MSD feed: {e}")))?;
        Ok(())
    }

    /// Return all accumulated MSD results as an array.
    ///
    /// Returns one [`MSDResult`] per frame fed (including the
    /// reference frame, which has MSD = 0).
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const results = msd.results();
    /// results.forEach((r, t) => console.log(`t=${t}: MSD=${r.mean}`));
    /// ```
    pub fn results(&self) -> Vec<MSDResult> {
        self.inner
            .results()
            .iter()
            .map(|r| MSDResult { inner: r.clone() })
            .collect()
    }

    /// Number of frames accumulated.
    #[wasm_bindgen(getter)]
    pub fn count(&self) -> usize {
        self.inner.count()
    }

    /// Reset the analysis, clearing reference and all results.
    pub fn reset(&mut self) {
        self.inner.reset();
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

    /// Per-particle squared displacements as a float typed array in A^2.
    ///
    /// `perParticle()[i]` is `|r_i(t) - r_i(0)|^2` for particle `i`.
    /// Length equals the number of atoms.
    #[wasm_bindgen(js_name = perParticle)]
    pub fn per_particle(&self) -> Vec<F> {
        self.inner.per_particle.to_vec()
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
        let rs_frame = frame.clone_core_frame()?;
        let result = self
            .inner
            .compute(&rs_frame, &neighbors.inner)
            .map_err(|e| JsValue::from_str(&format!("Cluster compute: {e}")))?;
        Ok(ClusterResult { inner: result })
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

        Frame::from_rs_frame(rs_frame).unwrap()
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

        let rdf = RDF::new(20, 4.0);
        let result = rdf.compute(&frame, &nbrs).unwrap();

        assert_eq!(result.bin_centers().len(), 20);
        assert_eq!(result.rdf().len(), 20);
        assert_eq!(result.bin_edges().len(), 21);
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
        let results = msd.results();
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
        let rs_frame = frame.clone_core_frame()?;
        let centers = self
            .inner
            .compute(&rs_frame, &cluster_result.inner)
            .map_err(|e| JsValue::from_str(&format!("ClusterCenters: {e}")))?;
        Ok(centers.iter().flat_map(|c| [c[0], c[1], c[2]]).collect())
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
    inner: RsCenterOfMassResult,
}

#[wasm_bindgen(js_class = CenterOfMassResult)]
impl CenterOfMassResult {
    /// Mass-weighted centers, flat `[x0,y0,z0, x1,y1,z1, ...]`.
    #[wasm_bindgen(js_name = centersOfMass)]
    pub fn centers_of_mass(&self) -> Vec<F> {
        self.inner
            .centers_of_mass
            .iter()
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect()
    }

    /// Total mass per cluster.
    #[wasm_bindgen(js_name = clusterMasses)]
    pub fn cluster_masses(&self) -> Vec<F> {
        self.inner.cluster_masses.clone()
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
        let rs_frame = frame.clone_core_frame()?;
        let calc = if let Some(ref ms) = self.masses {
            RsCenterOfMass::new().with_masses(ms)
        } else {
            RsCenterOfMass::new()
        };
        let result = calc
            .compute(&rs_frame, &cluster_result.inner)
            .map_err(|e| JsValue::from_str(&format!("CenterOfMass: {e}")))?;
        Ok(CenterOfMassResult { inner: result })
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
    pub fn compute(
        &self,
        frame: &Frame,
        cluster_result: &ClusterResult,
    ) -> Result<Vec<F>, JsValue> {
        let rs_frame = frame.clone_core_frame()?;
        let tensors = self
            .inner
            .compute(&rs_frame, &cluster_result.inner)
            .map_err(|e| JsValue::from_str(&format!("GyrationTensor: {e}")))?;
        Ok(tensors
            .iter()
            .flat_map(|t| t.iter().flat_map(|row| row.iter().copied()))
            .collect())
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
    pub fn compute(
        &self,
        frame: &Frame,
        cluster_result: &ClusterResult,
    ) -> Result<Vec<F>, JsValue> {
        let rs_frame = frame.clone_core_frame()?;
        let calc = if let Some(ref ms) = self.masses {
            RsInertiaTensor::new().with_masses(ms)
        } else {
            RsInertiaTensor::new()
        };
        let tensors = calc
            .compute(&rs_frame, &cluster_result.inner)
            .map_err(|e| JsValue::from_str(&format!("InertiaTensor: {e}")))?;
        Ok(tensors
            .iter()
            .flat_map(|t| t.iter().flat_map(|row| row.iter().copied()))
            .collect())
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
    pub fn compute(
        &self,
        frame: &Frame,
        cluster_result: &ClusterResult,
    ) -> Result<Vec<F>, JsValue> {
        let rs_frame = frame.clone_core_frame()?;
        let calc = if let Some(ref ms) = self.masses {
            RsRadiusOfGyration::new().with_masses(ms)
        } else {
            RsRadiusOfGyration::new()
        };
        let radii = calc
            .compute(&rs_frame, &cluster_result.inner)
            .map_err(|e| JsValue::from_str(&format!("RadiusOfGyration: {e}")))?;
        Ok(radii)
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
        let rs_frame = frame.clone_core_frame()?;

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
                .and_then(|a| a.as_slice().map(|s| s.to_vec()));
            let col_j = bonds
                .get("j")
                .and_then(|c| <u32 as BlockDtype>::from_column(c))
                .and_then(|a| a.as_slice().map(|s| s.to_vec()));

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
