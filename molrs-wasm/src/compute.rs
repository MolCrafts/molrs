//! WASM bindings for trajectory analysis: neighbor search, RDF, MSD, and cluster detection.
//!
//! This module provides freud-style analysis classes that operate on
//! [`Frame`] objects. The typical workflow is:
//!
//! 1. Index the coordinates with [`NeighborList`] (`build`), then materialize
//!    the pairs into a [`Neighbors`] table (`neighbors`).
//! 2. Pass that table to an analysis class ([`RDF`], [`Cluster`]).
//! 3. Read the result object.
//!
//! [`MSD`] does not require a neighbor table -- it only needs a reference
//! frame and a current frame.
//!
//! # Example (JavaScript)
//!
//! ```js
//! // Build the pair table
//! const nl = new NeighborList(5.0);
//! nl.build(frame);
//! const nlist = nl.neighbors();
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

use serde::Serialize;
use wasm_bindgen::prelude::*;

use molrs::system::topology::{Topology as RsTopology, TopologyRingInfo as RsTopologyRingInfo};

use molrs::compute::cluster::{Cluster as RsCluster, ClusterResult as RsClusterResult};
use molrs::compute::ml::{KMeans as RsKMeans, Pca2 as RsPca2, PcaResult as RsPcaResult};
use molrs::compute::msd::{MSD as RsMSD, MSDResult as RsMSDResult};
use molrs::compute::rdf::{RDF as RsRDF, RDFResult as RsRDFResult};
use molrs::compute::result::{ComputeResult, DescriptorRow};
use molrs::compute::shape::{
    COMResult as RsCOMResult, CenterOfMass as RsCenterOfMass, ClusterCenters as RsClusterCenters,
    GyrationTensor as RsGyrationTensor, InertiaTensor as RsInertiaTensor,
    RadiusOfGyration as RsRadiusOfGyration,
};
use molrs::compute::traits::{Compute, Fit};
use molrs::spatial::neighbors::{
    NeighborList as RsNeighborList, NeighborQuery as RsNeighborQuery, Neighbors as RsNeighbors,
    NeighborsStorage as RsNeighborsStorage, QueryMode,
};
use molrs::store::keys;
use molrs::types::F;
use ndarray::{Array1, Array2, Array3};

use crate::core::frame::Frame;
use crate::core::types::JsFloatArray;
use js_sys::{Float64Array, Int32Array, Uint32Array};

// ---------------------------------------------------------------------------
// Helper: extract Nx3 position matrix from a core Frame
// ---------------------------------------------------------------------------

/// Extract an Nx3 position matrix from the `"atoms"` block of a core
/// [`Frame`](molrs::store::frame::Frame).
///
/// Reads the `x`, `y`, `z` columns (F, angstrom) and assembles
/// them into a contiguous row-major matrix.
fn positions_from_frame(frame: &molrs::store::frame::Frame) -> Result<ndarray::Array2<F>, JsValue> {
    let atoms = frame
        .get("atoms")
        .ok_or_else(|| JsValue::from_str("Frame has no 'atoms' block"))?;
    let get = |col: &str| -> Result<&[F], JsValue> {
        use molrs::store::block::BlockDtype;
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
// NeighborList — the engine: coordinates in, pairs out
// ===========================================================================

/// Neighbor search over one point set: index the coordinates, then read pairs.
///
/// This is the door to every self search. It owns the cutoff and the backend
/// that indexes space, and keeps the two halves of the job apart: `build` and
/// `update` place the atoms in space and enumerate **nothing**, while
/// `neighbors()` materializes the pairs into a [`Neighbors`] table.
///
/// The pairs are a **half-shell** self list: every unordered pair appears
/// exactly once, with `i < j`, and never as `i == j`. A directed search against
/// a *second* point set is a different question — see [`LinkedCell::query`].
///
/// All distances are in angstrom (Å).
///
/// # Example (JavaScript)
///
/// ```js
/// const nl = new NeighborList(3.0);      // O(N) cell list, cutoff 3 Å
/// nl.build(frame);                       // index only — no pair table
/// const neigh = nl.neighbors();          // both columns (the default)
/// const lean  = nl.neighbors({ disp: false });   // indices + d² only
///
/// nl.update(movedFrame);                 // re-index in the box from `build`
/// ```
#[wasm_bindgen(js_name = NeighborList)]
pub struct NeighborList {
    inner: RsNeighborList,
}

#[wasm_bindgen(js_class = NeighborList)]
impl NeighborList {
    /// Create a search with the O(N) cell-list backend — the production choice.
    ///
    /// `cutoff` is the interaction radius in angstrom (Å). It is fixed here
    /// rather than passed per query because it sets the cell width of the
    /// index.
    ///
    /// # Errors
    ///
    /// Throws if `cutoff` is not a positive length.
    #[wasm_bindgen(constructor)]
    pub fn new(cutoff: F) -> Result<NeighborList, JsValue> {
        check_cutoff(cutoff)?;
        Ok(NeighborList {
            inner: RsNeighborList::new(cutoff),
        })
    }

    /// Create a search with the O(N²) all-pairs backend.
    ///
    /// Finds exactly the same pairs as the cell list — that is what makes it
    /// useful as a reference — at a cost that grows with the square of the
    /// particle count. Prefer it only for very small systems.
    ///
    /// # Errors
    ///
    /// Throws if `cutoff` is not a positive length.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const nl = NeighborList.bruteForce(12.5);
    /// ```
    #[wasm_bindgen(js_name = bruteForce)]
    pub fn brute_force(cutoff: F) -> Result<NeighborList, JsValue> {
        check_cutoff(cutoff)?;
        Ok(NeighborList {
            inner: RsNeighborList::brute_force(cutoff),
        })
    }

    /// The cutoff distance (Å) fixed at construction.
    #[wasm_bindgen(getter)]
    pub fn cutoff(&self) -> F {
        self.inner.cutoff()
    }

    /// Index a [`Frame`]'s atom positions — coordinates **and** box.
    ///
    /// Builds the spatial index and nothing else: no pairs are enumerated and
    /// no table is allocated. A frame without a `simbox` is treated as a free
    /// (non-periodic) system and gets a bounding box padded by the cutoff, so
    /// no pair can wrap around an edge.
    ///
    /// The box is retained, so a later [`update`](Self::update) can re-index
    /// new coordinates in the same box.
    ///
    /// # Errors
    ///
    /// Throws if the frame has no `atoms` block with `x` / `y` / `z` columns,
    /// or if a free-boundary box cannot be derived from the coordinates.
    pub fn build(&mut self, frame: &Frame) -> Result<(), JsValue> {
        let cutoff = self.inner.cutoff();
        frame.with_frame(|rs_frame| {
            let pos = positions_from_frame(rs_frame)?;
            let simbox;
            let bx_ref = match rs_frame.simbox.as_ref() {
                Some(sb) => sb,
                None => {
                    simbox = molrs::spatial::simbox::SimBox::free(pos.view(), cutoff)
                        .map_err(|e| JsValue::from_str(&format!("free-boundary box: {e:?}")))?;
                    &simbox
                }
            };
            self.inner.build(pos.view(), bx_ref);
            Ok(())
        })?;
        Ok(())
    }

    /// Re-index new coordinates in the box captured by the last
    /// [`build`](Self::build).
    ///
    /// The natural per-step call at fixed volume: the positions move, the box
    /// does not. Index-only, like `build`. There is no skin — this re-indexes
    /// every time rather than deciding for you that the previous index was
    /// still good enough. **If the box itself changed** (a barostat), call
    /// `build` again: `update` keeps the old box and would fold minimum images
    /// against a stale cell.
    ///
    /// # Errors
    ///
    /// Throws if no `build` has run yet — the box is then unknown, and guessing
    /// one silently changes every minimum-image distance. Throws too if the
    /// frame carries no readable positions.
    pub fn update(&mut self, frame: &Frame) -> Result<(), JsValue> {
        // Core panics on update-before-build; a panic aborts wasm, so check
        // the engine's own state and throw instead.
        if !self.inner.is_built() {
            return Err(JsValue::from_str(
                "NeighborList.update reuses the box of the previous build: \
                 call build(frame) first",
            ));
        }
        frame.with_frame(|rs_frame| {
            let pos = positions_from_frame(rs_frame)?;
            self.inner.update(pos.view());
            Ok(())
        })
    }

    /// Materialize the pairs into a [`Neighbors`] table.
    ///
    /// `storage` is an optional `{ distSq?: boolean, disp?: boolean }`. Both
    /// columns are kept by default, so no analysis is surprised by a missing
    /// one; pass `false` for a column this call site will not read. A dropped
    /// column cannot be added afterwards — materialize again instead.
    ///
    /// Keeping both columns names the *columns*, not the pair direction: a self
    /// search stays half-shell (`i < j`) either way.
    ///
    /// Row order is unspecified — the cell-list backend materializes in
    /// parallel and the work split decides the order.
    ///
    /// # Errors
    ///
    /// Throws if `storage` is neither nullish nor an object, or if one of its
    /// two fields is present but not a boolean.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const neigh = nl.neighbors();                  // distSq + disp
    /// const lean  = nl.neighbors({ disp: false });   // indices + d²
    /// ```
    pub fn neighbors(
        &self,
        storage: Option<NeighborsStorageOptions>,
    ) -> Result<Neighbors, JsValue> {
        let policy = match storage.as_deref() {
            // Omitted, `undefined` or `null`: keep every column — the safe
            // default, and the one an order parameter needs.
            None => RsNeighborsStorage::FULL,
            Some(options) if options.is_undefined() || options.is_null() => {
                RsNeighborsStorage::FULL
            }
            Some(options) if !options.is_object() => {
                return Err(JsValue::from_str(
                    "neighbors(storage) expects an object like { distSq: true, disp: true }",
                ));
            }
            Some(options) => RsNeighborsStorage {
                dist_sq: storage_flag(options, "distSq")?,
                disp: storage_flag(options, "disp")?,
            },
        };
        Ok(self.materialize(policy))
    }
}

impl NeighborList {
    /// Materialize under an already-resolved column policy — the shared body
    /// behind [`neighbors`](Self::neighbors) and the compatibility aliases.
    fn materialize(&self, storage: RsNeighborsStorage) -> Neighbors {
        Neighbors {
            inner: self.inner.neighbors(storage),
        }
    }
}

/// Reject a non-positive cutoff before the core asserts on it: a panic must not
/// cross this seam.
fn check_cutoff(cutoff: F) -> Result<(), JsValue> {
    if cutoff.is_nan() || cutoff <= 0.0 {
        return Err(JsValue::from_str("cutoff must be a positive length in Å"));
    }
    Ok(())
}

#[wasm_bindgen(typescript_custom_section)]
const NEIGHBORS_STORAGE_OPTIONS: &'static str = r#"
/**
 * Which optional columns `NeighborList.neighbors` keeps. Both default to
 * `true`: a column omitted here cannot be read back, and cannot be added
 * afterwards without materializing again.
 */
export interface NeighborsStorageOptions {
    /** Keep squared minimum-image distances (Å²), 8 B/pair. */
    distSq?: boolean;
    /** Keep minimum-image displacement vectors (Å), 24 B/pair. */
    disp?: boolean;
}
"#;

#[wasm_bindgen]
extern "C" {
    /// The `{ distSq?, disp? }` object accepted by
    /// [`NeighborList::neighbors`] — typed on the JS side rather than `any`,
    /// so a misspelt flag is a compile error for a TypeScript caller.
    #[wasm_bindgen(typescript_type = "NeighborsStorageOptions")]
    pub type NeighborsStorageOptions;
}

/// One boolean field of the storage options object; absent means `true`.
///
/// A field that is present but not a boolean is an error rather than a
/// coercion: `{ disp: 0 }` silently dropping the displacement column is exactly
/// the failure this surface exists to prevent.
fn storage_flag(storage: &JsValue, key: &str) -> Result<bool, JsValue> {
    let value = js_sys::Reflect::get(storage, &JsValue::from_str(key))?;
    if value.is_undefined() || value.is_null() {
        return Ok(true);
    }
    value
        .as_bool()
        .ok_or_else(|| JsValue::from_str(&format!("neighbors(storage): '{key}' must be a boolean")))
}

// ===========================================================================
// Neighbors — the materialized pair table
// ===========================================================================

/// Materialized neighbor pair table: every atom pair within the cutoff.
///
/// A column store — two index columns that are always present, plus whichever
/// physical columns the search was told to keep. Row `k` of every column
/// describes the same pair. Produced by [`NeighborList::neighbors`] and
/// consumed by the analysis classes ([`RDF`], [`Cluster`], `Steinhardt`, …).
///
/// A **self** search is half-shell: each unordered pair appears exactly once,
/// with `i < j`. A cross search ([`LinkedCell::query`]) is directed and has no
/// such rule.
///
/// # Properties
///
/// | Property | Type | Description |
/// |----------|------|-------------|
/// | `numPairs` | `number` | Number of pairs — the row count every column shares |
/// | `numPoints` | `number` | Number of reference points |
/// | `numQueryPoints` | `number` | Number of query points (= `numPoints` for a self search) |
/// | `isSelfQuery` | `boolean` | Whether both index columns address the same point set |
///
/// # Optional columns
///
/// `distSq()` and `disp()` return `undefined` when the search did not store
/// that column. That is deliberately different from an empty or zero-filled
/// array: a zero displacement is a physically meaningful value (two coincident
/// particles), so fabricating one would turn a missing column into a wrong
/// answer.
///
/// # Example (JavaScript)
///
/// ```js
/// const neigh = nl.neighbors();
/// console.log(neigh.numPairs);
///
/// const i  = neigh.queryPointIndices(); // Uint32Array
/// const j  = neigh.pointIndices();      // Uint32Array
/// const d2 = neigh.distSq();            // Float64Array (Å²) or undefined
/// const dr = neigh.disp();              // Float64Array (Å), 3 per pair
/// ```
#[wasm_bindgen(js_name = Neighbors)]
pub struct Neighbors {
    /// Crate-visible so the force-field optimizer can read pair indices.
    pub(crate) inner: RsNeighbors,
}

#[wasm_bindgen(js_class = Neighbors)]
impl Neighbors {
    /// Number of neighbor pairs — the row count every column shares.
    #[wasm_bindgen(getter, js_name = numPairs)]
    pub fn num_pairs(&self) -> usize {
        self.inner.n_pairs()
    }

    /// Number of reference (target) points the search indexed.
    #[wasm_bindgen(getter, js_name = numPoints)]
    pub fn num_points(&self) -> usize {
        self.inner.num_points()
    }

    /// Number of query points; equal to `numPoints` for a self search.
    #[wasm_bindgen(getter, js_name = numQueryPoints)]
    pub fn num_query_points(&self) -> usize {
        self.inner.num_query_points()
    }

    /// Whether both index columns address the same point set (half-shell,
    /// `i < j`).
    #[wasm_bindgen(getter, js_name = isSelfQuery)]
    pub fn is_self_query(&self) -> bool {
        matches!(self.inner.mode(), QueryMode::SelfQuery { .. })
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

    /// Squared minimum-image distances in Å², one per pair, or `undefined`
    /// when this table never stored the column.
    ///
    /// Zero-copy view; same invalidation caveat as
    /// [`queryPointIndices`](Self::query_point_indices). The square root is
    /// left to the caller — most consumers compare against a squared cutoff,
    /// and an accessor hiding a per-pair `sqrt` would make that cost invisible.
    #[wasm_bindgen(js_name = distSq)]
    pub fn dist_sq(&self) -> Option<JsFloatArray> {
        // SAFETY: view borrows wasm memory; short-lived use only.
        self.inner
            .dist_sq()
            .map(|column| unsafe { JsFloatArray::view(column) })
    }

    /// Minimum-image displacements `r_j - r_i` in Å, flattened as
    /// `[dx0, dy0, dz0, dx1, …]` — three values per pair, `3 * numPairs` long.
    /// `undefined` when this table never stored the column.
    ///
    /// The vector is **not** normalized: its length is the pair distance, and
    /// it points from `i` to `j`, so swapping the indices flips its sign. Both
    /// physical columns come from the same minimum-image evaluation, so
    /// `distSq[k]` is exactly the squared length of row `k`.
    ///
    /// Zero-copy view; same invalidation caveat as
    /// [`queryPointIndices`](Self::query_point_indices).
    ///
    /// # Errors
    ///
    /// Throws if the stored column is not contiguous, which would make a flat
    /// view silently misaligned rather than merely slow.
    pub fn disp(&self) -> Result<Option<JsFloatArray>, JsValue> {
        let Some(view) = self.inner.disp() else {
            return Ok(None);
        };
        let flat = view.as_slice().ok_or_else(|| {
            JsValue::from_str("disp column is not contiguous; cannot expose a flat view")
        })?;
        // SAFETY: view borrows wasm memory; short-lived use only.
        Ok(Some(unsafe { JsFloatArray::view(flat) }))
    }
}

// ===========================================================================
// LinkedCell / BruteForce — compatibility aliases over the engine
// ===========================================================================

/// **Compatibility alias.** Cell-list neighbor search in one call.
///
/// The primary door is [`NeighborList`], which separates indexing (`build` /
/// `update`) from materialization (`neighbors`). This class stays for the
/// molvis linkage, which constructs a search and a table in a single step, and
/// it is a thin wrapper over the same engine — the pairs are identical.
///
/// It also carries [`query`](Self::query), the **cross** search (query points
/// against a separate reference set), which the engine does not answer; that
/// is the second reason it is still here.
///
/// All distances are in angstrom (Å).
///
/// # Example (JavaScript)
///
/// ```js
/// const lc = new LinkedCell(3.0);        // cutoff = 3.0 Å
/// const neigh = lc.build(frame);         // self search (half-shell, i < j)
/// const cross = lc.query(ref, other);    // cross search (directed)
///
/// // Preferred spelling:
/// const nl = new NeighborList(3.0);
/// nl.build(frame);
/// const same = nl.neighbors();
/// ```
#[wasm_bindgen(js_name = LinkedCell)]
pub struct LinkedCell {
    cutoff: F,
    storage: RsNeighborsStorage,
}

#[wasm_bindgen(js_class = LinkedCell)]
impl LinkedCell {
    /// Create a cell-list search with the given distance cutoff.
    ///
    /// # Arguments
    ///
    /// * `cutoff` - Maximum neighbor distance in angstrom (Å)
    /// * `store_dist_sq` - Retain squared distances when materializing
    ///   (default `true`). Set `false` to save 8 B/pair.
    /// * `store_diff` - Retain MIC displacement vectors (default `true`).
    ///   Set `false` to save 24 B/pair.
    ///
    /// # Errors
    ///
    /// Throws if `cutoff` is not a positive length.
    #[wasm_bindgen(constructor)]
    pub fn new(
        cutoff: F,
        store_dist_sq: Option<bool>,
        store_diff: Option<bool>,
    ) -> Result<LinkedCell, JsValue> {
        check_cutoff(cutoff)?;
        Ok(LinkedCell {
            cutoff,
            storage: RsNeighborsStorage {
                dist_sq: store_dist_sq.unwrap_or(true),
                disp: store_diff.unwrap_or(true),
            },
        })
    }

    /// Index `frame` and materialize its half-shell pairs in one call.
    ///
    /// Equivalent to `new NeighborList(cutoff)` + `build(frame)` +
    /// `neighbors(...)` with the constructor's column flags.
    pub fn build(&self, frame: &Frame) -> Result<Neighbors, JsValue> {
        let mut engine = NeighborList::new(self.cutoff)?;
        engine.build(frame)?;
        Ok(engine.materialize(self.storage))
    }

    /// Cross search: all pairs where `i` indexes the query frame's atoms and
    /// `j` indexes the reference frame's atoms.
    ///
    /// Directed and full-shell — every query point reports all of its
    /// reference neighbors, with no `i < j` rule — so the result is tagged
    /// `isSelfQuery === false`.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const cross = new LinkedCell(3.0).query(refFrame, otherFrame);
    /// console.log(cross.numPairs);
    /// ```
    pub fn query(&self, ref_frame: &Frame, query_frame: &Frame) -> Result<Neighbors, JsValue> {
        ref_frame.with_frame(|rs_ref| {
            let ref_pos = positions_from_frame(rs_ref)?;

            let index = match rs_ref.simbox.as_ref() {
                Some(sb) => RsNeighborQuery::new(sb, ref_pos.view(), self.cutoff),
                None => RsNeighborQuery::free(ref_pos.view(), self.cutoff),
            };

            query_frame.with_frame(|rs_query| {
                let query_pos = positions_from_frame(rs_query)?;
                let table = index.query(query_pos.view());
                // A cross query always materializes every column; drop the
                // ones this search was told not to keep.
                let table = if self.storage == RsNeighborsStorage::FULL {
                    table
                } else {
                    table.repack(self.storage)
                };
                Ok(Neighbors { inner: table })
            })
        })
    }
}

/// **Compatibility alias.** O(N²) all-pairs neighbor search in one call.
///
/// The primary door is [`NeighborList::bruteForce`](NeighborList::brute_force).
/// This class stays for the molvis linkage; it is a thin wrapper over the same
/// engine and produces the same [`Neighbors`] table as [`LinkedCell`], so the
/// two are interchangeable for RDF, Cluster, LBFGS and the order parameters.
///
/// Prefer it over the cell list only for **small** systems (a few hundred
/// atoms) where building cells costs more than the search itself.
///
/// # Example (JavaScript)
///
/// ```js
/// const neigh = new BruteForce(12.5).build(frame);
/// ```
#[wasm_bindgen(js_name = BruteForce)]
pub struct BruteForce {
    cutoff: F,
    storage: RsNeighborsStorage,
}

#[wasm_bindgen(js_class = BruteForce)]
impl BruteForce {
    /// Create a brute-force search with the given cutoff (Å).
    ///
    /// * `store_dist_sq` — keep d² (default `true`)
    /// * `store_diff` — keep MIC displacement vectors (default `true`)
    ///
    /// # Errors
    ///
    /// Throws if `cutoff` is not a positive length.
    #[wasm_bindgen(constructor)]
    pub fn new(
        cutoff: F,
        store_dist_sq: Option<bool>,
        store_diff: Option<bool>,
    ) -> Result<BruteForce, JsValue> {
        check_cutoff(cutoff)?;
        Ok(BruteForce {
            cutoff,
            storage: RsNeighborsStorage {
                dist_sq: store_dist_sq.unwrap_or(true),
                disp: store_diff.unwrap_or(true),
            },
        })
    }

    /// Index `frame` and materialize its half-shell pairs in one call.
    ///
    /// # Errors
    ///
    /// Throws above `MAX_ATOMS` atoms: materializing O(N²) pairs in a browser
    /// tab at that size is almost always a mistake, and the cell list answers
    /// the same question.
    pub fn build(&self, frame: &Frame) -> Result<Neighbors, JsValue> {
        // Guard: materializing O(N²) pairs above this is almost always a bug.
        const MAX_ATOMS: usize = 8_000;
        let n = frame.with_frame(|rs_frame| {
            let atoms = rs_frame
                .get("atoms")
                .ok_or_else(|| JsValue::from_str("Frame has no 'atoms' block"))?;
            // `nrows` is None for a block with no columns — that is zero atoms.
            Ok(atoms.nrows().unwrap_or(0))
        })?;
        if n > MAX_ATOMS {
            return Err(JsValue::from_str(&format!(
                "BruteForce refused N={n} (max {MAX_ATOMS}): use NeighborList for large systems"
            )));
        }

        let mut engine = NeighborList::brute_force(self.cutoff)?;
        engine.build(frame)?;
        Ok(engine.materialize(self.storage))
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
    /// Explicit normalization volume (A^3). When unset, `compute` takes the
    /// volume from `frame.simbox`.
    volume: Option<F>,
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
    /// * `volume` - Explicit normalization volume in A^3. Optional; when unset,
    ///   [`compute`](Self::compute) reads the volume from `frame.simbox`.
    ///   Required by [`computeWithVolume`](Self::compute_with_volume).
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const rdf  = new RDF(100, 5.0);                  // rMin = 0, box volume
    /// const rdf2 = new RDF(100, 5.0, 0.5);             // exclude d < 0.5 A
    /// const rdf3 = new RDF(100, 5.0, null, 1000.0);    // non-periodic frame
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new(
        n_bins: usize,
        r_max: F,
        r_min: Option<F>,
        volume: Option<F>,
    ) -> Result<RDF, JsValue> {
        if let Some(v) = volume
            && !(v.is_finite() && v > 0.0)
        {
            return Err(JsValue::from_str("RDF: volume must be finite and > 0"));
        }
        let inner = RsRDF::new(n_bins, r_max, r_min.unwrap_or(0.0))
            .map_err(|e| JsValue::from_str(&format!("RDF: {e}")))?;
        Ok(Self { inner, volume })
    }

    /// Compute g(r) for one frame (self-query).
    ///
    /// **Single semantic path** for RDF: builds a cell-list **index** at
    /// `r_max` and streams pairs into the histogram (`build_index` +
    /// `visit_pairs`). A full [`Neighbors`] is never allocated.
    /// Memory is \(O(N + n_{\mathrm{bins}})\), not \(O(P)\).
    ///
    /// Volume comes from the constructor `volume` if set, else `frame.simbox`.
    /// Non-periodic frames must pass `volume` to the constructor.
    ///
    /// For A↔B cross-RDF use [`computeCross`](Self::compute_cross) (same
    /// streaming engine; wasm-bindgen cannot express `Option<&Frame>`).
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const rdf = new RDF(100, 5.0);       // r_max is required
    /// const result = rdf.compute(frame);
    /// ```
    pub fn compute(&self, frame: &Frame) -> Result<RDFResult, JsValue> {
        frame.with_frame(|rs_frame| {
            // If constructor set an explicit volume and the frame has no box,
            // synthesize a cubic box so compute_frame can proceed.
            if rs_frame.simbox.is_none() {
                let v = self.volume.ok_or_else(|| {
                    JsValue::from_str(
                        "RDF compute: frame has no box — pass volume to the RDF constructor",
                    )
                })?;
                return self.compute_with_synth_box(rs_frame, v);
            }
            let mut result = self
                .inner
                .compute_frame(rs_frame)
                .map_err(|e| JsValue::from_str(&format!("RDF compute: {e}")))?;
            if let Some(v) = self.volume {
                apply_volume_override(&mut result, v);
            }
            Ok(RDFResult { inner: result })
        })
    }

    /// Cross-query g(r) between two frames (same streaming engine as
    /// [`compute`](Self::compute)).
    ///
    /// * `ref_frame` — reference set (B, builds the cell index)
    /// * `query_frame` — query set (A)
    #[wasm_bindgen(js_name = computeCross)]
    pub fn compute_cross(
        &self,
        ref_frame: &Frame,
        query_frame: &Frame,
    ) -> Result<RDFResult, JsValue> {
        ref_frame.with_frame(|rs_ref| {
            let ref_pos = positions_from_frame(rs_ref)?;
            let owned_box;
            let bx = match rs_ref.simbox.as_ref() {
                Some(sb) => sb,
                None => {
                    let v = self.volume.ok_or_else(|| {
                        JsValue::from_str(
                            "RDF computeCross: frame has no box — pass volume to the RDF constructor",
                        )
                    })?;
                    let box_len = v.cbrt();
                    owned_box = molrs::spatial::simbox::SimBox::cube(
                        box_len,
                        ndarray::array![0.0 as F, 0.0 as F, 0.0 as F],
                        [false, false, false],
                    )
                    .map_err(|e| JsValue::from_str(&format!("RDF computeCross: {e:?}")))?;
                    &owned_box
                }
            };
            query_frame.with_frame(|rs_query| {
                let query_pos = positions_from_frame(rs_query)?;
                let mut result = self
                    .inner
                    .compute_cross(ref_pos.view(), query_pos.view(), bx)
                    .map_err(|e| JsValue::from_str(&format!("RDF compute: {e}")))?;
                if let Some(v) = self.volume {
                    apply_volume_override(&mut result, v);
                }
                Ok(RDFResult { inner: result })
            })
        })
    }

    fn compute_with_synth_box(
        &self,
        rs_frame: &molrs::store::frame::Frame,
        volume: F,
    ) -> Result<RDFResult, JsValue> {
        // Temporarily attach a cubic box for the streaming path, then restore.
        // Frame is behind shared store — we can't mutate easily. Interleave
        // positions and call compute_self with an owned box instead.
        let pos = positions_from_frame(rs_frame)?;
        let box_len = volume.cbrt();
        let bx = molrs::spatial::simbox::SimBox::cube(
            box_len,
            ndarray::array![0.0 as F, 0.0 as F, 0.0 as F],
            [false, false, false],
        )
        .map_err(|e| JsValue::from_str(&format!("RDF compute: {e:?}")))?;
        let result = self
            .inner
            .compute_self(pos.view(), &bx)
            .map_err(|e| JsValue::from_str(&format!("RDF compute: {e}")))?;
        Ok(RDFResult { inner: result })
    }
}

fn apply_volume_override(result: &mut molrs::compute::rdf::RDFResult, volume: F) {
    use molrs::compute::result::ComputeResult;
    result.volume = volume;
    result.finalized = false;
    result.finalize();
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
    frames: Vec<molrs::store::frame::Frame>,
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

    /// Run the stateless [`molrs::compute::MSD`] over every fed frame and
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
        let refs: Vec<&molrs::store::frame::Frame> = self.frames.iter().collect();
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
    /// * `neighbors` - Pre-built [`Neighbors`] defining connectivity
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
    pub fn compute(&self, frame: &Frame, neighbors: &Neighbors) -> Result<ClusterResult, JsValue> {
        frame.with_frame(|rs_frame| {
            let nlist_vec = std::slice::from_ref(&neighbors.inner);
            let mut results = self
                .inner
                .compute(&[rs_frame], nlist_vec)
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
        use molrs::spatial::simbox::SimBox;
        use molrs::store::block::Block;
        use ndarray::{Array1, array};

        let x = Array1::from_iter(positions.iter().map(|p| p[0]));
        let y = Array1::from_iter(positions.iter().map(|p| p[1]));
        let z = Array1::from_iter(positions.iter().map(|p| p[2]));

        let mut block = Block::new();
        block.insert("x", x.into_dyn()).unwrap();
        block.insert("y", y.into_dyn()).unwrap();
        block.insert("z", z.into_dyn()).unwrap();

        let mut rs_frame = molrs::store::frame::Frame::new();
        rs_frame.insert("atoms", block);
        rs_frame.simbox =
            Some(SimBox::cube(box_len, array![0.0 as F, 0.0, 0.0], [false, false, false]).unwrap());

        Frame::from_rs(rs_frame).unwrap()
    }

    #[wasm_bindgen_test]
    fn linked_cell_build_finds_pairs() {
        let positions = [[1.0, 1.0, 1.0], [1.5, 1.0, 1.0], [8.0, 8.0, 8.0]];
        let frame = make_frame(&positions, 20.0);

        let lc = LinkedCell::new(2.0, None, None).unwrap();
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

        // Streaming path: no NeighborList materialization.
        let rdf = RDF::new(20, 4.0, None, None).unwrap();
        let result = rdf.compute(&frame).unwrap();

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

        let lc = LinkedCell::new(2.0, None, None).unwrap();
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

        let lc = LinkedCell::new(2.0, None, None).unwrap();
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
                use molrs::store::block::BlockDtype;
                let col_i = bonds
                    .get("i")
                    .and_then(|c| <u64 as BlockDtype>::from_column(c))
                    .and_then(|a| a.as_slice());
                let col_j = bonds
                    .get("j")
                    .and_then(|c| <u64 as BlockDtype>::from_column(c))
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

/// Stateless wrapper for [`molrs::compute::ml::pca::Pca2`].
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
        let dummy = molrs::store::frame::Frame::new();
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

/// Wrapper for [`molrs::compute::ml::kmeans::KMeans`].
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
        let pca = molrs::compute::PcaResult {
            coords: coords.to_vec(),
            variance: [0.0 as F, 0.0 as F],
        };
        let dummy = molrs::store::frame::Frame::new();
        let labels = self
            .inner
            .compute(&[&dummy], &pca)
            .map_err(|e| JsValue::from_str(&format!("KMeans fit: {e}")))?;
        let out = Int32Array::new_with_length(labels.0.len() as u32);
        out.copy_from(&labels.0);
        Ok(out)
    }
}

// ===========================================================================
// Extended molrs compute API
// ===========================================================================

fn js_value<T: Serialize>(value: &T) -> Result<JsValue, JsValue> {
    serde_wasm_bindgen::to_value(value)
        .map_err(|e| JsValue::from_str(&format!("serialize wasm result: {e}")))
}

fn array1(data: &[F]) -> Array1<F> {
    Array1::from_vec(data.to_vec())
}

fn array2(data: &[F], rows: usize, cols: usize, name: &str) -> Result<Array2<F>, JsValue> {
    if data.len() != rows * cols {
        return Err(JsValue::from_str(&format!(
            "{name}: data length {} != rows * cols = {} * {}",
            data.len(),
            rows,
            cols
        )));
    }
    Array2::from_shape_vec((rows, cols), data.to_vec())
        .map_err(|e| JsValue::from_str(&format!("{name}: {e}")))
}

fn array3(
    data: &[F],
    dim0: usize,
    dim1: usize,
    dim2: usize,
    name: &str,
) -> Result<Array3<F>, JsValue> {
    if data.len() != dim0 * dim1 * dim2 {
        return Err(JsValue::from_str(&format!(
            "{name}: data length {} != dim0 * dim1 * dim2 = {} * {} * {}",
            data.len(),
            dim0,
            dim1,
            dim2
        )));
    }
    Array3::from_shape_vec((dim0, dim1, dim2), data.to_vec())
        .map_err(|e| JsValue::from_str(&format!("{name}: {e}")))
}

fn vectors3(data: &[F], name: &str) -> Result<Vec<[F; 3]>, JsValue> {
    if !data.len().is_multiple_of(3) {
        return Err(JsValue::from_str(&format!(
            "{name}: expected flat [x,y,z,...] length divisible by 3"
        )));
    }
    Ok(data.chunks_exact(3).map(|v| [v[0], v[1], v[2]]).collect())
}

fn quats(data: &[F], name: &str) -> Result<Vec<[F; 4]>, JsValue> {
    if !data.len().is_multiple_of(4) {
        return Err(JsValue::from_str(&format!(
            "{name}: expected flat [w,x,y,z,...] length divisible by 4"
        )));
    }
    Ok(data
        .chunks_exact(4)
        .map(|v| [v[0], v[1], v[2], v[3]])
        .collect())
}

fn u32_pairs(data: &[u32], name: &str) -> Result<Vec<(u32, u32)>, JsValue> {
    if !data.len().is_multiple_of(2) {
        return Err(JsValue::from_str(&format!("{name}: expected pairs")));
    }
    Ok(data.chunks_exact(2).map(|p| (p[0], p[1])).collect())
}

fn usize_pairs(data: &[u32], name: &str) -> Result<Vec<(usize, usize)>, JsValue> {
    Ok(u32_pairs(data, name)?
        .into_iter()
        .map(|(a, b)| (a as usize, b as usize))
        .collect())
}

fn usize_vec(data: &[u32]) -> Vec<usize> {
    data.iter().map(|&v| v as usize).collect()
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SeriesOut {
    lag_times: Vec<F>,
    values: Vec<F>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SpectrumOut {
    frequencies: Vec<F>,
    intensities: Vec<F>,
    resolution: usize,
    n_frames: usize,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RamanSpectrumOut {
    frequencies: Vec<F>,
    isotropic: Vec<F>,
    anisotropic: Vec<F>,
    parallel: Option<Vec<F>>,
    perpendicular: Option<Vec<F>>,
    resolution: usize,
    n_frames: usize,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct DielectricSpectrumOut {
    frequencies: Vec<F>,
    eps_real: Vec<F>,
    eps_imag: Vec<F>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Grid2Out {
    data: Vec<F>,
    shape: [usize; 2],
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Grid3Out<T: Serialize> {
    data: Vec<T>,
    shape: [usize; 3],
}

fn spectrum_out(r: molrs::compute::SpectrumResult) -> SpectrumOut {
    SpectrumOut {
        frequencies: r.frequencies_cm1.to_vec(),
        intensities: r.intensities.to_vec(),
        resolution: r.resolution,
        n_frames: r.n_frames,
    }
}

fn raman_out(r: molrs::compute::RamanSpectrumResult) -> RamanSpectrumOut {
    RamanSpectrumOut {
        frequencies: r.frequencies_cm1.to_vec(),
        isotropic: r.isotropic.to_vec(),
        anisotropic: r.anisotropic.to_vec(),
        parallel: r.parallel.map(|v| v.to_vec()),
        perpendicular: r.perpendicular.map(|v| v.to_vec()),
        resolution: r.resolution,
        n_frames: r.n_frames,
    }
}

fn dielectric_spectrum_out(r: molrs::compute::DielectricSpectrumResult) -> DielectricSpectrumOut {
    DielectricSpectrumOut {
        frequencies: r.frequencies.to_vec(),
        eps_real: r.eps_real.to_vec(),
        eps_imag: r.eps_imag.to_vec(),
    }
}

// ===========================================================================
// Compute catalog — the single source of truth for downstream UIs
// ===========================================================================
//
// Everything a caller needs to present, configure and dispatch an analysis
// lives here, so no consumer has to keep a parallel hand-written table that
// can drift out of sync with the bindings above.

/// Default value of a [`ParamSpec`], serialized untagged so JS sees a bare
/// `number`, `boolean` or `string`.
#[derive(Serialize, Clone, Copy)]
#[serde(untagged)]
enum ParamDefault {
    Num(F),
    Bool(bool),
    Text(&'static str),
}

/// One user-facing knob of an analysis.
///
/// These are **UI-level** parameters, not a literal mirror of the WASM
/// constructor: `diffraction.static_structure_factor` exposes `kMin`/`kMax`/`nK`
/// and the caller expands them into the `k_values` array the constructor wants.
/// `kind` tells the caller how to render and coerce the value:
///
/// | `kind` | JS value |
/// |--------|----------|
/// | `int`, `float` | `number` |
/// | `bool` | `boolean` |
/// | `select` | one of `options` |
/// | `intList`, `floatList` | comma-separated `string` → typed array |
#[derive(Serialize, Clone, Copy)]
#[serde(rename_all = "camelCase")]
struct ParamSpec {
    key: &'static str,
    label: &'static str,
    kind: &'static str,
    default: ParamDefault,
    /// `true` when the binding accepts `null` for this argument.
    optional: bool,
    /// `"ctor"` — a positional constructor argument, in declaration order,
    /// after any leading arguments the dispatch shape supplies itself. Every
    /// piece of configuration lives here: `compute` / `fit` take only data.
    /// `"call"` — the knob configures a *different* object the caller builds
    /// first (`LinkedCell`'s cutoff, `Cluster`'s min size), never this one.
    slot: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    min: Option<F>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max: Option<F>,
    #[serde(skip_serializing_if = "Option::is_none")]
    unit: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<&'static [&'static str]>,
}

const fn base(
    key: &'static str,
    label: &'static str,
    kind: &'static str,
    default: ParamDefault,
) -> ParamSpec {
    ParamSpec {
        key,
        label,
        kind,
        default,
        optional: false,
        slot: "ctor",
        min: None,
        max: None,
        unit: None,
        options: None,
    }
}

fn p_int(key: &'static str, label: &'static str, default: u32, min: F, max: F) -> ParamSpec {
    ParamSpec {
        min: Some(min),
        max: Some(max),
        ..base(key, label, "int", ParamDefault::Num(F::from(default)))
    }
}

fn p_float(
    key: &'static str,
    label: &'static str,
    default: F,
    unit: Option<&'static str>,
) -> ParamSpec {
    ParamSpec {
        unit,
        ..base(key, label, "float", ParamDefault::Num(default))
    }
}

fn p_bool(key: &'static str, label: &'static str, default: bool) -> ParamSpec {
    base(key, label, "bool", ParamDefault::Bool(default))
}

fn p_select(
    key: &'static str,
    label: &'static str,
    default: &'static str,
    options: &'static [&'static str],
) -> ParamSpec {
    ParamSpec {
        options: Some(options),
        ..base(key, label, "select", ParamDefault::Text(default))
    }
}

fn p_list(
    key: &'static str,
    label: &'static str,
    kind: &'static str,
    default: &'static str,
) -> ParamSpec {
    base(key, label, kind, ParamDefault::Text(default))
}

fn optional(spec: ParamSpec) -> ParamSpec {
    ParamSpec {
        optional: true,
        ..spec
    }
}

/// Mark a knob as configuring a helper object the caller builds, not this one.
fn call(spec: ParamSpec) -> ParamSpec {
    ParamSpec {
        slot: "call",
        ..spec
    }
}

/// The `cutoff` knob every neighbor-driven analysis needs to build its
/// `LinkedCell` before `compute(frame, nlist)`.
fn p_cutoff(default: F) -> ParamSpec {
    call(p_float("cutoff", "Neighbor cutoff", default, Some("Å")))
}

/// A menu category, in the order a picker should present it.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct CatalogCategory {
    id: &'static str,
    label: &'static str,
}

/// One analysis: what it is, how to call it, and what it needs.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ComputeCatalogEntry {
    id: &'static str,
    category: &'static str,
    label: &'static str,
    /// Class exported from this module. Always present — the catalog never
    /// names a binding that does not exist.
    wasm_export: &'static str,
    /// How to drive the binding. The caller dispatches on this, not on `id`.
    ///
    /// | `input_kind` | invocation |
    /// |--------------|-----------|
    /// | `frame` | `compute(frame)` |
    /// | `frameNeighbors` | `compute(frame, nlist)`, `nlist` from `cutoff` |
    /// | `frameClusters` | `compute(frame, clusterResult)` |
    /// | `frameGroups` | `compute(frame, atomIndexTuples)` |
    /// | `frameGroupSets` | `compute(frame, number[][])` |
    /// | `frameRadii` | `compute(frame, radii, …)` — Voronoi family |
    /// | `accumulate` | `feed(frame)` per frame, then `compute()` / `results()` |
    /// | `series` | `compute(…)` / `fit(…)` over raw arrays; no `Frame` |
    input_kind: &'static str,
    /// Shape of the payload, for picking a renderer.
    result_kind: &'static str,
    /// Per-atom or per-trajectory inputs needed **beyond positions**. A caller
    /// that cannot supply one of these should disable the entry and say which.
    requires: &'static [&'static str],
    params: Vec<ParamSpec>,
}

/// Menu categories — freud top-level analysis modules first, then molrs
/// extensions. Not 1:1 with every Rust `compute/` folder.
///
/// freud mappings:
/// - `density` — g(r) (`freud.density.RDF`) + Local/Gaussian density, …
/// - `locality` — Voronoi (`freud.locality.Voronoi`); neighbor queries are infra
/// - `msd` / `cluster` / `order` / `environment` / `diffraction` / `pmft` — 1:1
///
/// molrs extensions (no freud top-level):
/// - `transport` — VACF/diffusion/conductivity **and** van Hove / pair persistence
/// - `spectroscopy` — IR/Raman/… **and** static dielectric constant
/// - `distribution` / `hbond` / `shape` / `fit` / `ml`
const CATEGORIES: [CatalogCategory; 15] = [
    // --- freud core ---------------------------------------------------------
    CatalogCategory {
        id: "density",
        label: "Density",
    },
    CatalogCategory {
        id: "locality",
        label: "Locality",
    },
    CatalogCategory {
        id: "msd",
        label: "MSD",
    },
    CatalogCategory {
        id: "cluster",
        label: "Cluster",
    },
    CatalogCategory {
        id: "order",
        label: "Order",
    },
    CatalogCategory {
        id: "environment",
        label: "Environment",
    },
    CatalogCategory {
        id: "diffraction",
        label: "Diffraction",
    },
    CatalogCategory {
        id: "pmft",
        label: "PMFT",
    },
    // --- molrs extensions ---------------------------------------------------
    CatalogCategory {
        id: "transport",
        label: "Transport",
    },
    CatalogCategory {
        id: "spectroscopy",
        label: "Spectroscopy",
    },
    CatalogCategory {
        id: "hbond",
        label: "Hydrogen Bonds",
    },
    CatalogCategory {
        id: "distribution",
        label: "Distribution",
    },
    CatalogCategory {
        id: "shape",
        label: "Shape",
    },
    CatalogCategory {
        id: "fit",
        label: "Fit",
    },
    CatalogCategory {
        id: "ml",
        label: "ML",
    },
];

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ComputeCatalog {
    /// Bump whenever an entry's `id`, `input_kind` or param keys change.
    version: u32,
    categories: &'static [CatalogCategory],
    analyses: Vec<ComputeCatalogEntry>,
}

fn entry(
    id: &'static str,
    category: &'static str,
    label: &'static str,
    wasm_export: &'static str,
    input_kind: &'static str,
    result_kind: &'static str,
    requires: &'static [&'static str],
    params: Vec<ParamSpec>,
) -> ComputeCatalogEntry {
    ComputeCatalogEntry {
        id,
        category,
        label,
        wasm_export,
        input_kind,
        result_kind,
        requires,
        params,
    }
}

/// Describe every analysis this module exports.
///
/// Returns `{ version, categories, analyses }`. Consumers should group
/// `analyses` by `category` in `categories` order to build a menu.
#[wasm_bindgen(js_name = molrsComputeCatalog)]
pub fn molrs_compute_catalog() -> Result<JsValue, JsValue> {
    let analyses = vec![
        // --- density (freud.density: RDF + local/gaussian density, …) -------
        // Analysis id keeps the `rdf.*` prefix for stable clients; the menu
        // category is `density` to match freud.density.RDF.
        entry(
            "rdf.radial_distribution",
            "density",
            "Radial distribution g(r)",
            "RDF",
            "frameNeighbors",
            "lineSeries",
            &[],
            vec![
                p_cutoff(10.0),
                p_int("nBins", "Bins", 100, 1.0, 4096.0),
                p_float("rMax", "r max", 10.0, Some("Å")),
                optional(p_float("rMin", "r min", 0.0, Some("Å"))),
            ],
        ),
        // --- msd ------------------------------------------------------------
        entry(
            "msd.mean_squared_displacement",
            "msd",
            "Mean squared displacement",
            "MSD",
            "accumulate",
            "trajectorySeries",
            &[],
            vec![],
        ),
        // --- transport ------------------------------------------------------
        entry(
            "transport.vacf",
            "transport",
            "VACF",
            "WasmVACF",
            "series",
            "lineSeries",
            &["velocity"],
            // `n_dof` is the column count of the velocity matrix (3 x atoms),
            // so the caller derives it from the data rather than asking.
            vec![
                p_float("dt", "Timestep", 1.0, Some("fs")),
                p_int("resolution", "Max lag", 200, 1.0, 1e6),
            ],
        ),
        entry(
            "transport.einstein_diffusion",
            "transport",
            "Einstein diffusion",
            "WasmEinsteinDiffusion",
            "accumulate",
            "lineSeries",
            &[],
            vec![p_float("dt", "Timestep", 1.0, Some("fs"))],
        ),
        entry(
            "transport.green_kubo_diffusion",
            "transport",
            "Green-Kubo diffusion",
            "WasmGreenKuboDiffusion",
            "series",
            "lineSeries",
            &["velocity"],
            // `n_dof` is the column count of the velocity matrix (3 x atoms),
            // so the caller derives it from the data rather than asking.
            vec![
                p_float("dt", "Timestep", 1.0, Some("fs")),
                p_int("resolution", "Max lag", 200, 1.0, 1e6),
            ],
        ),
        entry(
            "transport.conductivity",
            "transport",
            "Conductivity",
            "WasmGreenKuboConductivity",
            "series",
            "lineSeries",
            &["charge", "velocity"],
            vec![
                p_float("dt", "Timestep", 1.0, Some("fs")),
                p_int("maxLag", "Max lag", 200, 1.0, 1e6),
            ],
        ),
        entry(
            "transport.einstein_conductivity",
            "transport",
            "Einstein conductivity",
            "WasmEinsteinConductivity",
            "series",
            "lineSeries",
            &["charge"],
            vec![
                p_float("dt", "Timestep", 1.0, Some("fs")),
                p_int("maxLag", "Max lag", 200, 1.0, 1e6),
            ],
        ),
        entry(
            "transport.onsager_correlation",
            "transport",
            "Onsager correlation",
            "WasmOnsagerCorrelation",
            "series",
            "lineSeries",
            &["charge", "velocity"],
            vec![
                p_float("dt", "Timestep", 1.0, Some("fs")),
                p_int("maxLag", "Max lag", 200, 1.0, 1e6),
            ],
        ),
        // --- dynamics -------------------------------------------------------
        entry(
            "dynamics.van_hove_function",
            "transport",
            "Van Hove function",
            "WasmVanHove",
            "accumulate",
            "matrix",
            &[],
            vec![
                p_int("nRBins", "r bins", 100, 1.0, 4096.0),
                p_float("rMax", "r max", 10.0, Some("Å")),
                p_list("lags", "Lags", "intList", "1,2,5,10"),
                optional(p_int("stride", "Stride", 1, 1.0, 1e6)),
            ],
        ),
        entry(
            "dynamics.pair_persistence",
            "transport",
            "Pair persistence",
            "WasmPairPersistence",
            "series",
            "lineSeries",
            &["atomPairs"],
            vec![
                p_float("r0", "Birth radius r0", 3.0, Some("Å")),
                p_float("r1", "Break radius r1", 3.5, Some("Å")),
                p_select(
                    "method",
                    "Survival method",
                    "continuous",
                    &["continuous", "intermittent", "ssp"],
                ),
                p_float("dt", "Timestep", 1.0, Some("fs")),
                p_int("maxLag", "Max lag", 200, 1.0, 1e6),
                p_bool("excludeSelf", "Exclude self pairs", true),
            ],
        ),
        // --- spectroscopy ---------------------------------------------------
        entry(
            "spectroscopy.power_spectrum",
            "spectroscopy",
            "Power spectrum",
            "WasmPowerSpectrum",
            "series",
            "lineSeries",
            &["velocity"],
            vec![
                p_float("dtFs", "Timestep", 1.0, Some("fs")),
                call(p_int("resolution", "Max lag", 200, 1.0, 1e6)),
            ],
        ),
        entry(
            "spectroscopy.ir_spectrum",
            "spectroscopy",
            "IR spectrum",
            "WasmIRSpectrum",
            "series",
            "lineSeries",
            &["dipole"],
            vec![
                p_float("dtFs", "Timestep", 1.0, Some("fs")),
                call(p_int("resolution", "Max lag", 200, 1.0, 1e6)),
            ],
        ),
        entry(
            "spectroscopy.raman_spectrum",
            "spectroscopy",
            "Raman spectrum",
            "WasmRamanSpectrum",
            "series",
            "lineSeries",
            &["polarizability"],
            vec![
                optional(p_float(
                    "incidentFrequencyCm1",
                    "Incident frequency",
                    0.0,
                    Some("cm^-1"),
                )),
                optional(p_float("temperatureK", "Temperature", 0.0, Some("K"))),
                optional(p_bool("averaged", "Orientation averaged", false)),
                p_float("dtFs", "Timestep", 1.0, Some("fs")),
                call(p_int("resolution", "Max lag", 200, 1.0, 1e6)),
            ],
        ),
        entry(
            "spectroscopy.vcd_spectrum",
            "spectroscopy",
            "VCD spectrum",
            "WasmVcdSpectrum",
            "series",
            "lineSeries",
            &["dipole", "magneticDipole"],
            vec![
                p_float("dtFs", "Timestep", 1.0, Some("fs")),
                call(p_int("resolution", "Max lag", 200, 1.0, 1e6)),
            ],
        ),
        entry(
            "spectroscopy.roa_spectrum",
            "spectroscopy",
            "ROA spectrum",
            "WasmRoaSpectrum",
            "series",
            "lineSeries",
            &["polarizability", "gTensor"],
            vec![
                optional(p_float(
                    "incidentFrequencyCm1",
                    "Incident frequency",
                    0.0,
                    Some("cm^-1"),
                )),
                optional(p_float("temperatureK", "Temperature", 0.0, Some("K"))),
                optional(p_bool("averaged", "Orientation averaged", false)),
                p_float("dtFs", "Timestep", 1.0, Some("fs")),
                call(p_int("resolution", "Max lag", 200, 1.0, 1e6)),
            ],
        ),
        entry(
            "spectroscopy.dielectric_spectrum",
            "spectroscopy",
            "Dielectric spectrum",
            "WasmGreenKuboDielectricSpectrum",
            "series",
            "lineSeries",
            &["charge", "velocity"],
            vec![
                p_float("dt", "Timestep", 1.0, Some("fs")),
                p_float("volume", "Box volume", 0.0, Some("Å³")),
                p_float("temperature", "Temperature", 300.0, Some("K")),
                optional(p_float("epsilonInf", "ε∞", 1.0, None)),
            ],
        ),
        // --- dielectric -----------------------------------------------------
        entry(
            "dielectric.static_dielectric_constant",
            "spectroscopy",
            "Static dielectric constant",
            "WasmStaticDielectric",
            "series",
            "scalar",
            &["dipole"],
            vec![
                p_float("volume", "Box volume", 0.0, Some("Å³")),
                p_float("temperature", "Temperature", 300.0, Some("K")),
                optional(p_float("epsilonInf", "ε∞", 1.0, None)),
            ],
        ),
        // --- fit ------------------------------------------------------------
        entry(
            "fit.linear_fit",
            "fit",
            "Linear fit",
            "WasmLinearFit",
            "series",
            "scalar",
            &["xySeries"],
            vec![
                p_float("startFrac", "Window start", 0.2, None),
                p_float("endFrac", "Window end", 0.8, None),
            ],
        ),
        entry(
            "fit.running_integral",
            "fit",
            "Running integral",
            "WasmCumulativeTrapezoid",
            "series",
            "lineSeries",
            &["series"],
            vec![
                p_float("dt", "Timestep", 1.0, Some("fs")),
                optional(p_int("nLags", "Lags", 200, 1.0, 1e6)),
            ],
        ),
        entry(
            "fit.plateau",
            "fit",
            "Plateau",
            "WasmPlateau",
            "series",
            "scalar",
            &["series"],
            vec![
                p_float("startFrac", "Window start", 0.2, None),
                p_float("endFrac", "Window end", 0.8, None),
            ],
        ),
        entry(
            "fit.debye_fit",
            "fit",
            "Debye fit",
            "WasmDebyeFit",
            "series",
            "scalar",
            &["series"],
            vec![p_float("dt", "Timestep", 1.0, Some("fs"))],
        ),
        // --- cluster --------------------------------------------------------
        entry(
            "cluster.connected_components",
            "cluster",
            "Cluster analysis",
            "Cluster",
            "frameNeighbors",
            "barSeries",
            &[],
            vec![
                p_cutoff(3.0),
                p_int("minClusterSize", "Min cluster size", 1, 1.0, 1e6),
            ],
        ),
        // --- shape ----------------------------------------------------------
        entry(
            "shape.cluster_properties",
            "cluster",
            "Radius of gyration",
            "RadiusOfGyration",
            "frameClusters",
            "table",
            &[],
            vec![
                p_cutoff(3.0),
                call(p_int("minClusterSize", "Min cluster size", 1, 1.0, 1e6)),
            ],
        ),
        entry(
            "shape.center_of_mass",
            "shape",
            "Center of mass",
            "CenterOfMass",
            "frameClusters",
            "table",
            &[],
            vec![
                p_cutoff(3.0),
                call(p_int("minClusterSize", "Min cluster size", 1, 1.0, 1e6)),
            ],
        ),
        entry(
            "shape.gyration_tensor",
            "shape",
            "Gyration tensor",
            "GyrationTensor",
            "frameClusters",
            "matrix",
            &[],
            vec![
                p_cutoff(3.0),
                call(p_int("minClusterSize", "Min cluster size", 1, 1.0, 1e6)),
            ],
        ),
        entry(
            "shape.inertia_tensor",
            "shape",
            "Inertia tensor",
            "InertiaTensor",
            "frameClusters",
            "matrix",
            &[],
            vec![
                p_cutoff(3.0),
                call(p_int("minClusterSize", "Min cluster size", 1, 1.0, 1e6)),
            ],
        ),
        // --- density --------------------------------------------------------
        entry(
            "density.correlation_function",
            "density",
            "Correlation function",
            "WasmCorrelationFunction",
            "frameNeighbors",
            "lineSeries",
            &["scalarField"],
            vec![
                p_cutoff(10.0),
                p_int("nBins", "Bins", 100, 1.0, 4096.0),
                p_float("rMax", "r max", 10.0, Some("Å")),
                optional(p_float("rMin", "r min", 0.0, Some("Å"))),
            ],
        ),
        entry(
            "density.gaussian_density",
            "density",
            "Gaussian density",
            "WasmGaussianDensity",
            "frame",
            "grid3",
            &[],
            vec![
                p_int("nx", "Grid x", 32, 1.0, 512.0),
                p_int("ny", "Grid y", 32, 1.0, 512.0),
                p_int("nz", "Grid z", 32, 1.0, 512.0),
                p_float("sigma", "Sigma", 1.0, Some("Å")),
                optional(p_float("rMax", "Cutoff", 5.0, Some("Å"))),
            ],
        ),
        entry(
            "density.local_density",
            "density",
            "Local density",
            "WasmLocalDensity",
            "frameNeighbors",
            "lineSeries",
            &[],
            vec![
                p_cutoff(5.0),
                p_float("rMax", "r max", 5.0, Some("Å")),
                optional(p_float("diameter", "Particle diameter", 1.0, Some("Å"))),
            ],
        ),
        entry(
            "density.spatial_distribution",
            "density",
            "Spatial distribution",
            "WasmSpatialDistribution",
            "accumulate",
            "grid3",
            &["referenceAtoms", "template", "targetAtoms"],
            vec![
                p_int("nx", "Grid x", 32, 1.0, 512.0),
                p_int("ny", "Grid y", 32, 1.0, 512.0),
                p_int("nz", "Grid z", 32, 1.0, 512.0),
                p_float("extentX", "Extent x", 10.0, Some("Å")),
                p_float("extentY", "Extent y", 10.0, Some("Å")),
                p_float("extentZ", "Extent z", 10.0, Some("Å")),
                optional(p_float("bulkDensity", "Bulk density", 0.0, Some("Å⁻³"))),
            ],
        ),
        entry(
            "density.sphere_voxelization",
            "density",
            "Sphere voxelization",
            "WasmSphereVoxelization",
            "frame",
            "grid3",
            &[],
            vec![
                p_int("nx", "Grid x", 32, 1.0, 512.0),
                p_int("ny", "Grid y", 32, 1.0, 512.0),
                p_int("nz", "Grid z", 32, 1.0, 512.0),
                p_float("rMax", "Sphere radius", 2.0, Some("Å")),
            ],
        ),
        // --- order ----------------------------------------------------------
        entry(
            "order.steinhardt",
            "order",
            "Steinhardt",
            "WasmSteinhardt",
            "frameNeighbors",
            "matrix",
            &[],
            vec![
                p_cutoff(3.0),
                p_list("lValues", "l values", "intList", "6"),
                optional(p_bool("average", "Averaged", false)),
                optional(p_bool("wl", "Compute w_l", false)),
                optional(p_bool("wlNormalize", "Normalize w_l", false)),
            ],
        ),
        entry(
            "order.hexatic",
            "order",
            "Hexatic",
            "WasmHexatic",
            "frameNeighbors",
            "lineSeries",
            &[],
            vec![p_cutoff(3.0), p_int("k", "Symmetry k", 6, 1.0, 32.0)],
        ),
        entry(
            "order.nematic",
            "order",
            "Nematic",
            "WasmNematic",
            "series",
            "scalar",
            &["orientation"],
            vec![],
        ),
        entry(
            "order.cubatic",
            "order",
            "Cubatic",
            "WasmCubatic",
            "series",
            "scalar",
            &["orientation"],
            vec![
                optional(p_int("seed", "Seed", 0, 0.0, 4.294e9)),
                optional(p_float("initialTemp", "Initial temperature", 5.0, None)),
                optional(p_float("coolingRate", "Cooling rate", 0.9, None)),
                optional(p_int("nSteps", "Steps", 100, 1.0, 1e6)),
                optional(p_int("nChains", "Chains", 10, 1.0, 1e4)),
            ],
        ),
        entry(
            "order.solid_liquid",
            "order",
            "Solid-liquid",
            "WasmSolidLiquid",
            "frameNeighbors",
            "table",
            &[],
            vec![
                p_cutoff(3.0),
                p_int("l", "l", 6, 0.0, 32.0),
                optional(p_float("qThreshold", "q threshold", 0.7, None)),
                optional(p_int("nThreshold", "Neighbor threshold", 6, 0.0, 64.0)),
                optional(p_bool("normalizeQ", "Normalize q", true)),
            ],
        ),
        entry(
            "order.rotational_autocorrelation",
            "order",
            "Rotational autocorrelation",
            "WasmRotationalAutocorrelation",
            "series",
            "lineSeries",
            &["orientation"],
            vec![p_int("l", "l", 2, 0.0, 32.0)],
        ),
        // --- environment ----------------------------------------------------
        entry(
            "environment.bond_order",
            "environment",
            "Bond order",
            "WasmBondOrder",
            "frameNeighbors",
            "matrix",
            &[],
            vec![
                p_cutoff(3.0),
                p_int("nTheta", "θ bins", 60, 1.0, 512.0),
                p_int("nPhi", "φ bins", 30, 1.0, 512.0),
            ],
        ),
        entry(
            "environment.local_descriptors",
            "environment",
            "Local descriptors",
            "WasmLocalDescriptors",
            "frameNeighbors",
            "matrix",
            &[],
            vec![p_cutoff(3.0), p_int("lMax", "l max", 6, 0.0, 32.0)],
        ),
        entry(
            "environment.angular_separation",
            "environment",
            "Angular separation",
            "WasmAngularSeparation",
            "series",
            "lineSeries",
            &["orientation"],
            vec![optional(p_bool(
                "equivalentOrientations",
                "Fold equivalent orientations",
                false,
            ))],
        ),
        entry(
            "environment.environment_matching",
            "environment",
            "Environment matching",
            "WasmMatchEnv",
            "frameNeighbors",
            "table",
            &[],
            vec![
                p_cutoff(3.0),
                p_float("rmsdThreshold", "RMSD threshold", 0.1, Some("Å")),
                optional(p_bool("registration", "Registration", false)),
                optional(p_int(
                    "maxNeighborsForRegistration",
                    "Max neighbors",
                    12,
                    1.0,
                    128.0,
                )),
            ],
        ),
        // --- diffraction ----------------------------------------------------
        entry(
            "diffraction.static_structure_factor",
            "diffraction",
            "Static structure factor S(k)",
            "WasmStaticStructureFactorDebye",
            "frame",
            "lineSeries",
            &[],
            vec![
                p_float("kMin", "k min", 0.1, Some("Å⁻¹")),
                p_float("kMax", "k max", 10.0, Some("Å⁻¹")),
                p_int("nK", "k samples", 100, 1.0, 4096.0),
            ],
        ),
        entry(
            "diffraction.diffraction_pattern",
            "diffraction",
            "Diffraction pattern",
            "WasmDiffractionPattern",
            "frame",
            "matrix",
            &[],
            vec![
                p_int("nGrid", "Grid", 512, 8.0, 4096.0),
                p_float("sigma", "Sigma", 1.0, None),
                optional(p_int("axis", "Zone axis", 2, 0.0, 2.0)),
            ],
        ),
        // --- distribution ---------------------------------------------------
        entry(
            "distribution.distance_distribution",
            "distribution",
            "Distance distribution",
            "WasmDistanceDistribution",
            "frameGroups",
            "lineSeries",
            &["atomPairs"],
            vec![
                p_int("nBins", "Bins", 100, 1.0, 4096.0),
                p_float("min", "Min", 0.0, Some("Å")),
                p_float("max", "Max", 10.0, Some("Å")),
            ],
        ),
        entry(
            "distribution.angle_distribution",
            "distribution",
            "Angle distribution",
            "WasmAngleDistribution",
            "frameGroups",
            "lineSeries",
            &["atomTriples"],
            vec![p_int("nBins", "Bins", 100, 1.0, 4096.0)],
        ),
        entry(
            "distribution.dihedral_distribution",
            "distribution",
            "Dihedral distribution",
            "WasmDihedralDistribution",
            "frameGroups",
            "lineSeries",
            &["atomQuads"],
            vec![p_int("nBins", "Bins", 100, 1.0, 4096.0)],
        ),
        entry(
            "distribution.combined_distribution",
            "distribution",
            "Combined distribution",
            "WasmCombinedDistribution",
            "frameGroupSets",
            "matrix",
            &["atomGroups"],
            vec![
                p_list("kinds", "Observables", "textList", "distance,angle"),
                p_list("bins", "Bins per axis", "intList", "50,50"),
                p_list("mins", "Axis minima", "floatList", "0,0"),
                p_list("maxs", "Axis maxima", "floatList", "10,3.14159265"),
                optional(p_list("sinWeight", "sin θ weighting", "intList", "0,1")),
            ],
        ),
        // --- pmft -----------------------------------------------------------
        entry(
            "pmft.pmft_r12",
            "pmft",
            "PMFT R12",
            "WasmPMFTR12",
            "frameNeighbors",
            "matrix",
            &["orientation"],
            vec![
                p_cutoff(5.0),
                p_float("rMax", "r max", 5.0, Some("Å")),
                p_int("nR", "r bins", 50, 1.0, 1024.0),
                p_int("nT1", "θ₁ bins", 36, 1.0, 1024.0),
                p_int("nT2", "θ₂ bins", 36, 1.0, 1024.0),
            ],
        ),
        entry(
            "pmft.pmft_xy",
            "pmft",
            "PMFT XY",
            "WasmPMFTXY",
            "frameNeighbors",
            "matrix",
            &[],
            vec![
                p_cutoff(5.0),
                p_float("xMax", "x max", 5.0, Some("Å")),
                p_float("yMax", "y max", 5.0, Some("Å")),
                p_int("nX", "x bins", 50, 1.0, 1024.0),
                p_int("nY", "y bins", 50, 1.0, 1024.0),
            ],
        ),
        entry(
            "pmft.pmft_xyt",
            "pmft",
            "PMFT XYT",
            "WasmPMFTXYT",
            "frameNeighbors",
            "matrix",
            &["orientation"],
            vec![
                p_cutoff(5.0),
                p_float("xMax", "x max", 5.0, Some("Å")),
                p_float("yMax", "y max", 5.0, Some("Å")),
                p_int("nX", "x bins", 50, 1.0, 1024.0),
                p_int("nY", "y bins", 50, 1.0, 1024.0),
                p_int("nT", "θ bins", 36, 1.0, 1024.0),
            ],
        ),
        entry(
            "pmft.pmft_xyz",
            "pmft",
            "PMFT XYZ",
            "WasmPMFTXYZ",
            "frameNeighbors",
            "matrix",
            &[],
            vec![
                p_cutoff(5.0),
                p_float("xMax", "x max", 5.0, Some("Å")),
                p_float("yMax", "y max", 5.0, Some("Å")),
                p_float("zMax", "z max", 5.0, Some("Å")),
                p_int("nX", "x bins", 30, 1.0, 512.0),
                p_int("nY", "y bins", 30, 1.0, 512.0),
                p_int("nZ", "z bins", 30, 1.0, 512.0),
            ],
        ),
        // --- hbond ----------------------------------------------------------
        entry(
            "hbond.hydrogen_bond_detection",
            "hbond",
            "Hydrogen-bond detection",
            "WasmHBonds",
            "accumulate",
            "table",
            &["donors", "acceptors"],
            vec![
                optional(p_float("distCutoff", "Distance cutoff", 3.5, Some("Å"))),
                optional(p_select(
                    "distKind",
                    "Distance criterion",
                    "donor_acceptor",
                    &["donor_acceptor", "hydrogen_acceptor"],
                )),
                optional(p_float("angleCutoff", "Angle cutoff", 150.0, Some("°"))),
            ],
        ),
        entry(
            "hbond.lifetime",
            "hbond",
            "Lifetime",
            "WasmHBondLifetime",
            "series",
            "lineSeries",
            &["hbondPresence"],
            vec![
                p_float("dt", "Timestep", 1.0, Some("fs")),
                p_int("maxLag", "Max lag", 200, 1.0, 1e6),
            ],
        ),
        entry(
            "hbond.network_components",
            "hbond",
            "Network components",
            "WasmHBondNetwork",
            "series",
            "table",
            &["hbondEdges"],
            vec![],
        ),
        // --- locality (freud.locality: Voronoi; neighbor queries are infra) --
        // Analysis ids keep the `voronoi.*` prefix for stable clients.
        entry(
            "voronoi.radical_voronoi",
            "locality",
            "Radical Voronoi",
            "WasmRadicalVoronoi",
            "frameRadii",
            "table",
            &[],
            vec![p_bool("useAtomRadii", "Weight by covalent radii", true)],
        ),
        entry(
            "voronoi.domain_analysis",
            "locality",
            "Domain analysis",
            "WasmVoronoiDomainAnalysis",
            "frameRadii",
            "table",
            &["labels"],
            vec![
                p_bool("useAtomRadii", "Weight by covalent radii", true),
                call(p_select(
                    "labelBy",
                    "Label cells by",
                    "element",
                    &[keys::ELEMENT, keys::MOL_ID, keys::TYPE],
                )),
            ],
        ),
        entry(
            "voronoi.void_analysis",
            "locality",
            "Void analysis",
            "WasmVoronoiVoidAnalysis",
            "frameRadii",
            "table",
            &["voidMask"],
            vec![
                p_bool("useAtomRadii", "Weight by covalent radii", true),
                optional(p_float("boxVolume", "Box volume", 0.0, Some("Å³"))),
            ],
        ),
        // --- ml -------------------------------------------------------------
        entry(
            "ml.pca",
            "ml",
            "PCA",
            "WasmPca2",
            "series",
            "custom",
            &["descriptorMatrix"],
            vec![],
        ),
        entry(
            "ml.kmeans",
            "ml",
            "k-means",
            "WasmKMeans",
            "series",
            "custom",
            &["descriptorMatrix"],
            vec![
                p_int("k", "Clusters", 3, 1.0, 1024.0),
                p_int("maxIter", "Max iterations", 100, 1.0, 1e5),
                p_int("seed", "Seed", 0, 0.0, 4.294e9),
            ],
        ),
    ];

    js_value(&ComputeCatalog {
        // v3: freud core order + molrs extensions — dynamics→transport,
        // static dielectric→spectroscopy; cluster_properties→cluster.
        version: 3,
        categories: &CATEGORIES,
        analyses,
    })
}

#[wasm_bindgen(js_name = WasmVACF)]
pub struct WasmVACF {
    dt: F,
    resolution: usize,
}

#[wasm_bindgen(js_class = WasmVACF)]
impl WasmVACF {
    #[wasm_bindgen(constructor)]
    pub fn new(dt: F, resolution: usize) -> Self {
        Self { dt, resolution }
    }

    pub fn compute(
        &self,
        velocities: &[F],
        n_frames: usize,
        n_dof: usize,
    ) -> Result<JsValue, JsValue> {
        let dt = self.dt;
        let resolution = self.resolution;
        let v = array2(velocities, n_frames, n_dof, "VACF velocities")?;
        let frames: [&molrs::store::frame::Frame; 0] = [];
        let calc = molrs::compute::VACF;
        let r = calc
            .compute(&frames, (&v, dt, resolution))
            .map_err(|e| JsValue::from_str(&format!("VACF: {e}")))?;
        js_value(&SeriesOut {
            lag_times: r.lag_times.to_vec(),
            values: r.acf.to_vec(),
        })
    }
}

#[wasm_bindgen(js_name = WasmGreenKuboDiffusion)]
pub struct WasmGreenKuboDiffusion {
    dt: F,
    resolution: usize,
}

#[wasm_bindgen(js_class = WasmGreenKuboDiffusion)]
impl WasmGreenKuboDiffusion {
    #[wasm_bindgen(constructor)]
    pub fn new(dt: F, resolution: usize) -> Self {
        Self { dt, resolution }
    }

    pub fn compute(
        &self,
        velocities: &[F],
        n_frames: usize,
        n_dof: usize,
    ) -> Result<JsValue, JsValue> {
        let dt = self.dt;
        let resolution = self.resolution;
        let v = array2(velocities, n_frames, n_dof, "GreenKuboDiffusion velocities")?;
        let frames: [&molrs::store::frame::Frame; 0] = [];
        let calc = molrs::compute::GreenKuboDiffusion;
        let r = calc
            .compute(&frames, (&v, dt, resolution))
            .map_err(|e| JsValue::from_str(&format!("GreenKuboDiffusion: {e}")))?;
        js_value(&SeriesOut {
            lag_times: r.lag_times.to_vec(),
            values: r.acf.to_vec(),
        })
    }
}

#[wasm_bindgen(js_name = WasmGreenKuboConductivity)]
pub struct WasmGreenKuboConductivity {
    dt: F,
    max_lag: usize,
}

#[wasm_bindgen(js_class = WasmGreenKuboConductivity)]
impl WasmGreenKuboConductivity {
    #[wasm_bindgen(constructor)]
    pub fn new(dt: F, max_lag: usize) -> Self {
        Self { dt, max_lag }
    }

    pub fn compute(&self, current: &[F], n_frames: usize) -> Result<JsValue, JsValue> {
        let dt = self.dt;
        let max_lag = self.max_lag;
        let c = array2(current, n_frames, 3, "GreenKuboConductivity current")?;
        let frames: [&molrs::store::frame::Frame; 0] = [];
        let calc = molrs::compute::GreenKuboConductivity;
        let r = calc
            .compute(&frames, (&c, dt, max_lag))
            .map_err(|e| JsValue::from_str(&format!("GreenKuboConductivity: {e}")))?;
        js_value(&SeriesOut {
            lag_times: r.lag_times.to_vec(),
            values: r.jacf.to_vec(),
        })
    }
}

#[wasm_bindgen(js_name = WasmEinsteinConductivity)]
pub struct WasmEinsteinConductivity {
    dt: F,
    max_lag: usize,
}

#[wasm_bindgen(js_class = WasmEinsteinConductivity)]
impl WasmEinsteinConductivity {
    #[wasm_bindgen(constructor)]
    pub fn new(dt: F, max_lag: usize) -> Self {
        Self { dt, max_lag }
    }

    pub fn compute(&self, translational_dipole: &[F], n_frames: usize) -> Result<JsValue, JsValue> {
        let dt = self.dt;
        let max_lag = self.max_lag;
        let d = array2(
            translational_dipole,
            n_frames,
            3,
            "EinsteinConductivity translationalDipole",
        )?;
        let frames: [&molrs::store::frame::Frame; 0] = [];
        let calc = molrs::compute::EinsteinConductivity;
        let r = calc
            .compute(&frames, (&d, dt, max_lag))
            .map_err(|e| JsValue::from_str(&format!("EinsteinConductivity: {e}")))?;
        js_value(&SeriesOut {
            lag_times: r.lag_times.to_vec(),
            values: r.msd.to_vec(),
        })
    }
}

#[wasm_bindgen(js_name = WasmOnsagerCorrelation)]
pub struct WasmOnsagerCorrelation {
    dt: F,
    max_lag: usize,
}

#[wasm_bindgen(js_class = WasmOnsagerCorrelation)]
impl WasmOnsagerCorrelation {
    #[wasm_bindgen(constructor)]
    pub fn new(dt: F, max_lag: usize) -> Self {
        Self { dt, max_lag }
    }

    pub fn compute(&self, pi: &[F], pj: &[F], n_frames: usize) -> Result<JsValue, JsValue> {
        let dt = self.dt;
        let max_lag = self.max_lag;
        let pi = array2(pi, n_frames, 3, "Onsager p_i")?;
        let pj = array2(pj, n_frames, 3, "Onsager p_j")?;
        let frames: [&molrs::store::frame::Frame; 0] = [];
        let calc = molrs::compute::OnsagerCorrelation;
        let r = calc
            .compute(&frames, (&pi, &pj, dt, max_lag))
            .map_err(|e| JsValue::from_str(&format!("OnsagerCorrelation: {e}")))?;
        js_value(&SeriesOut {
            lag_times: r.lag_times.to_vec(),
            values: r.correlation.to_vec(),
        })
    }
}

#[wasm_bindgen(js_name = WasmEinsteinDiffusion)]
pub struct WasmEinsteinDiffusion {
    dt: F,
    frames: Vec<molrs::store::frame::Frame>,
}

#[wasm_bindgen(js_class = WasmEinsteinDiffusion)]
impl WasmEinsteinDiffusion {
    #[wasm_bindgen(constructor)]
    pub fn new(dt: F) -> Self {
        Self {
            dt,
            frames: Vec::new(),
        }
    }

    pub fn feed(&mut self, frame: &Frame) -> Result<(), JsValue> {
        frame.with_frame(|rs_frame| {
            self.frames.push(rs_frame.clone());
            Ok(())
        })
    }

    pub fn compute(&self) -> Result<JsValue, JsValue> {
        let dt = self.dt;
        let refs: Vec<&molrs::store::frame::Frame> = self.frames.iter().collect();
        let calc = molrs::compute::EinsteinDiffusion;
        let r = calc
            .compute(&refs, molrs::compute::EinsteinDiffusionArgs { dt })
            .map_err(|e| JsValue::from_str(&format!("EinsteinDiffusion: {e}")))?;
        js_value(&SeriesOut {
            lag_times: r.lag_times.to_vec(),
            values: r.msd.to_vec(),
        })
    }

    pub fn reset(&mut self) {
        self.frames.clear();
    }
}

#[wasm_bindgen(js_name = WasmDebyeRelaxation)]
pub struct WasmDebyeRelaxation {
    dt: F,
    max_lag: usize,
    volume: F,
    temperature: F,
    boundary: String,
}

#[wasm_bindgen(js_class = WasmDebyeRelaxation)]
impl WasmDebyeRelaxation {
    #[wasm_bindgen(constructor)]
    pub fn new(volume: F, temperature: F, boundary: Option<String>, dt: F, max_lag: usize) -> Self {
        Self {
            dt,
            max_lag,
            volume,
            temperature,
            boundary: boundary.unwrap_or_else(|| "tinfoil".to_string()),
        }
    }

    pub fn compute(&self, dipoles: &[F], n_frames: usize) -> Result<JsValue, JsValue> {
        let dt = self.dt;
        let max_lag = self.max_lag;
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            lag_times: Vec<F>,
            acf: Vec<F>,
            zero_lag_variance: F,
            volume: F,
            temperature: F,
            boundary: String,
        }

        let dipoles = array2(dipoles, n_frames, 3, "DebyeRelaxation dipoles")?;
        let boundary = molrs::compute::EwaldBoundary::from_name(&self.boundary)
            .map_err(|e| JsValue::from_str(&format!("DebyeRelaxation boundary: {e}")))?;
        let calc = molrs::compute::DebyeRelaxation {
            volume: self.volume,
            temperature: self.temperature,
            boundary,
        };
        let frames: [&molrs::store::frame::Frame; 0] = [];
        let r = calc
            .compute(&frames, (&dipoles, dt, max_lag))
            .map_err(|e| JsValue::from_str(&format!("DebyeRelaxation: {e}")))?;
        js_value(&Out {
            lag_times: r.lag_times.to_vec(),
            acf: r.acf.to_vec(),
            zero_lag_variance: r.zero_lag_variance,
            volume: r.volume,
            temperature: r.temperature,
            boundary: self.boundary.clone(),
        })
    }
}

#[wasm_bindgen(js_name = WasmIRFlux)]
pub struct WasmIRFlux {
    dt: F,
    resolution: usize,
}

#[wasm_bindgen(js_class = WasmIRFlux)]
impl WasmIRFlux {
    #[wasm_bindgen(constructor)]
    pub fn new(dt: F, resolution: usize) -> Self {
        Self { dt, resolution }
    }

    pub fn compute(&self, dipoles: &[F], n_frames: usize) -> Result<JsValue, JsValue> {
        let dt = self.dt;
        let resolution = self.resolution;
        let dipoles = array2(dipoles, n_frames, 3, "IRFlux dipoles")?;
        let frames: [&molrs::store::frame::Frame; 0] = [];
        let calc = molrs::compute::IRFlux;
        let r = calc
            .compute(&frames, (&dipoles, dt, resolution))
            .map_err(|e| JsValue::from_str(&format!("IRFlux: {e}")))?;
        js_value(&SeriesOut {
            lag_times: r.lag_times.to_vec(),
            values: r.acf.to_vec(),
        })
    }
}

#[wasm_bindgen(js_name = WasmRamanTensor)]
pub struct WasmRamanTensor {
    dt: F,
    resolution: usize,
}

#[wasm_bindgen(js_class = WasmRamanTensor)]
impl WasmRamanTensor {
    #[wasm_bindgen(constructor)]
    pub fn new(dt: F, resolution: usize) -> Self {
        Self { dt, resolution }
    }

    pub fn compute(&self, polarizabilities: &[F], n_frames: usize) -> Result<JsValue, JsValue> {
        let dt = self.dt;
        let resolution = self.resolution;
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            lag_times: Vec<F>,
            acf_iso: Vec<F>,
            acf_aniso: Vec<F>,
        }
        let p = array2(
            polarizabilities,
            n_frames,
            6,
            "RamanTensor polarizabilities",
        )?;
        let frames: [&molrs::store::frame::Frame; 0] = [];
        let calc = molrs::compute::RamanTensor;
        let r = calc
            .compute(&frames, (&p, dt, resolution))
            .map_err(|e| JsValue::from_str(&format!("RamanTensor: {e}")))?;
        js_value(&Out {
            lag_times: r.lag_times.to_vec(),
            acf_iso: r.acf_iso.to_vec(),
            acf_aniso: r.acf_aniso.to_vec(),
        })
    }
}

#[wasm_bindgen(js_name = WasmVcdCrossFlux)]
pub struct WasmVcdCrossFlux {
    dt: F,
    resolution: usize,
}

#[wasm_bindgen(js_class = WasmVcdCrossFlux)]
impl WasmVcdCrossFlux {
    #[wasm_bindgen(constructor)]
    pub fn new(dt: F, resolution: usize) -> Self {
        Self { dt, resolution }
    }

    pub fn compute(
        &self,
        electric: &[F],
        magnetic: &[F],
        n_frames: usize,
    ) -> Result<JsValue, JsValue> {
        let dt = self.dt;
        let resolution = self.resolution;
        let e = array2(electric, n_frames, 3, "VcdCrossFlux electric")?;
        let m = array2(magnetic, n_frames, 3, "VcdCrossFlux magnetic")?;
        let frames: [&molrs::store::frame::Frame; 0] = [];
        let calc = molrs::compute::VcdCrossFlux;
        let r = calc
            .compute(&frames, (&e, &m, dt, resolution))
            .map_err(|e| JsValue::from_str(&format!("VcdCrossFlux: {e}")))?;
        js_value(&SeriesOut {
            lag_times: r.lag_times.to_vec(),
            values: r.acf.to_vec(),
        })
    }
}

#[wasm_bindgen(js_name = WasmRoaCrossTensor)]
pub struct WasmRoaCrossTensor {
    dt: F,
    resolution: usize,
}

#[wasm_bindgen(js_class = WasmRoaCrossTensor)]
impl WasmRoaCrossTensor {
    #[wasm_bindgen(constructor)]
    pub fn new(dt: F, resolution: usize) -> Self {
        Self { dt, resolution }
    }

    pub fn compute(
        &self,
        electric_pol: &[F],
        g_tensor: &[F],
        n_frames: usize,
    ) -> Result<JsValue, JsValue> {
        let dt = self.dt;
        let resolution = self.resolution;
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            lag_times: Vec<F>,
            acf_iso: Vec<F>,
            acf_aniso: Vec<F>,
        }
        let a = array2(electric_pol, n_frames, 6, "RoaCrossTensor electricPol")?;
        let g = array2(g_tensor, n_frames, 6, "RoaCrossTensor gTensor")?;
        let frames: [&molrs::store::frame::Frame; 0] = [];
        let calc = molrs::compute::RoaCrossTensor;
        let r = calc
            .compute(&frames, (&a, &g, dt, resolution))
            .map_err(|e| JsValue::from_str(&format!("RoaCrossTensor: {e}")))?;
        js_value(&Out {
            lag_times: r.lag_times.to_vec(),
            acf_iso: r.acf_iso.to_vec(),
            acf_aniso: r.acf_aniso.to_vec(),
        })
    }
}

macro_rules! spectrum_fit_class {
    ($name:ident, $calc:expr) => {
        #[wasm_bindgen(js_name = $name)]
        pub struct $name {
            dt_fs: F,
        }
        #[wasm_bindgen(js_class = $name)]
        impl $name {
            /// `dt_fs` is the trajectory timestep in femtoseconds.
            #[wasm_bindgen(constructor)]
            pub fn new(dt_fs: F) -> Self {
                Self { dt_fs }
            }
            pub fn fit(&self, acf: &[F]) -> Result<JsValue, JsValue> {
                let y = array1(acf);
                let r = $calc
                    .fit((&y, self.dt_fs))
                    .map_err(|e| JsValue::from_str(&format!("spectrum fit: {e}")))?;
                js_value(&spectrum_out(r))
            }
        }
    };
}

spectrum_fit_class!(WasmPowerSpectrum, molrs::compute::PowerSpectrum);
spectrum_fit_class!(WasmIRSpectrum, molrs::compute::IRSpectrum);
spectrum_fit_class!(WasmVcdSpectrum, molrs::compute::VcdSpectrum);

#[wasm_bindgen(js_name = WasmRamanSpectrum)]
pub struct WasmRamanSpectrum {
    dt_fs: F,
    incident_frequency_cm1: F,
    temperature_k: F,
    averaged: bool,
}

#[wasm_bindgen(js_class = WasmRamanSpectrum)]
impl WasmRamanSpectrum {
    #[wasm_bindgen(constructor)]
    pub fn new(
        incident_frequency_cm1: Option<F>,
        temperature_k: Option<F>,
        averaged: Option<bool>,
        dt_fs: F,
    ) -> Self {
        Self {
            dt_fs,
            incident_frequency_cm1: incident_frequency_cm1.unwrap_or(0.0),
            temperature_k: temperature_k.unwrap_or(0.0),
            averaged: averaged.unwrap_or(false),
        }
    }

    pub fn fit(&self, acf_iso: &[F], acf_aniso: &[F]) -> Result<JsValue, JsValue> {
        let dt_fs = self.dt_fs;
        let iso = array1(acf_iso);
        let aniso = array1(acf_aniso);
        let calc = molrs::compute::RamanSpectrum {
            incident_frequency_cm1: self.incident_frequency_cm1,
            temperature_k: self.temperature_k,
            averaged: self.averaged,
        };
        let r = calc
            .fit((&iso, &aniso, dt_fs))
            .map_err(|e| JsValue::from_str(&format!("RamanSpectrum: {e}")))?;
        js_value(&raman_out(r))
    }
}

#[wasm_bindgen(js_name = WasmRoaSpectrum)]
pub struct WasmRoaSpectrum(WasmRamanSpectrum);

#[wasm_bindgen(js_class = WasmRoaSpectrum)]
impl WasmRoaSpectrum {
    /// Same optical configuration as [`WasmRamanSpectrum`]; ROA differs only in
    /// which cross-correlations are supplied to [`fit`](Self::fit).
    #[wasm_bindgen(constructor)]
    pub fn new(
        incident_frequency_cm1: Option<F>,
        temperature_k: Option<F>,
        averaged: Option<bool>,
        dt_fs: F,
    ) -> Self {
        Self(WasmRamanSpectrum::new(
            incident_frequency_cm1,
            temperature_k,
            averaged,
            dt_fs,
        ))
    }

    pub fn fit(&self, acf_iso: &[F], acf_aniso: &[F]) -> Result<JsValue, JsValue> {
        let iso = array1(acf_iso);
        let aniso = array1(acf_aniso);
        let calc = molrs::compute::RoaSpectrum {
            incident_frequency_cm1: self.0.incident_frequency_cm1,
            temperature_k: self.0.temperature_k,
            averaged: self.0.averaged,
        };
        let r = calc
            .fit((&iso, &aniso, self.0.dt_fs))
            .map_err(|e| JsValue::from_str(&format!("RoaSpectrum: {e}")))?;
        js_value(&raman_out(r))
    }
}

#[wasm_bindgen(js_name = WasmEinsteinHelfandDielectricSpectrum)]
pub struct WasmEinsteinHelfandDielectricSpectrum {
    dt: F,
    volume: F,
    temperature: F,
    epsilon_inf: F,
    zero_lag_variance: F,
}

#[wasm_bindgen(js_class = WasmEinsteinHelfandDielectricSpectrum)]
impl WasmEinsteinHelfandDielectricSpectrum {
    #[wasm_bindgen(constructor)]
    pub fn new(
        dt: F,
        volume: F,
        temperature: F,
        epsilon_inf: Option<F>,
        zero_lag_variance: F,
    ) -> Self {
        Self {
            dt,
            volume,
            temperature,
            epsilon_inf: epsilon_inf.unwrap_or(1.0),
            zero_lag_variance,
        }
    }

    pub fn fit(&self, acf: &[F]) -> Result<JsValue, JsValue> {
        let acf = array1(acf);
        let calc = molrs::compute::EinsteinHelfandSpectrum {
            dt: self.dt,
            volume: self.volume,
            temperature: self.temperature,
            epsilon_inf: self.epsilon_inf,
            zero_lag_variance: self.zero_lag_variance,
        };
        let r = calc
            .fit(&acf)
            .map_err(|e| JsValue::from_str(&format!("EinsteinHelfandDielectricSpectrum: {e}")))?;
        js_value(&dielectric_spectrum_out(r))
    }
}

#[wasm_bindgen(js_name = WasmGreenKuboDielectricSpectrum)]
pub struct WasmGreenKuboDielectricSpectrum {
    dt: F,
    volume: F,
    temperature: F,
    epsilon_inf: F,
    window_type: String,
}

#[wasm_bindgen(js_class = WasmGreenKuboDielectricSpectrum)]
impl WasmGreenKuboDielectricSpectrum {
    #[wasm_bindgen(constructor)]
    pub fn new(
        dt: F,
        volume: F,
        temperature: F,
        epsilon_inf: Option<F>,
        window_type: Option<String>,
    ) -> Self {
        Self {
            dt,
            volume,
            temperature,
            epsilon_inf: epsilon_inf.unwrap_or(1.0),
            window_type: window_type.unwrap_or_else(|| "cosine_sq".to_string()),
        }
    }

    pub fn fit(&self, jacf: &[F]) -> Result<JsValue, JsValue> {
        let jacf = array1(jacf);
        let calc = molrs::compute::GreenKuboSpectrum {
            dt: self.dt,
            volume: self.volume,
            temperature: self.temperature,
            epsilon_inf: self.epsilon_inf,
            window_type: self.window_type.clone(),
        };
        let r = calc
            .fit(&jacf)
            .map_err(|e| JsValue::from_str(&format!("GreenKuboDielectricSpectrum: {e}")))?;
        js_value(&dielectric_spectrum_out(r))
    }
}

#[wasm_bindgen(js_name = WasmLinearFit)]
pub struct WasmLinearFit {
    start_frac: F,
    end_frac: F,
}

#[wasm_bindgen(js_class = WasmLinearFit)]
impl WasmLinearFit {
    #[wasm_bindgen(constructor)]
    pub fn new(start_frac: F, end_frac: F) -> Self {
        Self {
            start_frac,
            end_frac,
        }
    }

    pub fn fit(&self, x: &[F], y: &[F]) -> Result<JsValue, JsValue> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            slope: F,
            intercept: F,
            r2: F,
            fit_start: usize,
            fit_end: usize,
        }
        let x = array1(x);
        let y = array1(y);
        let r = molrs::compute::LinearFit {
            window: (self.start_frac, self.end_frac),
        }
        .fit((&x, &y))
        .map_err(|e| JsValue::from_str(&format!("LinearFit: {e}")))?;
        js_value(&Out {
            slope: r.slope,
            intercept: r.intercept,
            r2: r.r2,
            fit_start: r.fit_start,
            fit_end: r.fit_end,
        })
    }
}

#[wasm_bindgen(js_name = WasmCumulativeTrapezoid)]
pub struct WasmCumulativeTrapezoid {
    dt: F,
    n_lags: Option<usize>,
}

#[wasm_bindgen(js_class = WasmCumulativeTrapezoid)]
impl WasmCumulativeTrapezoid {
    #[wasm_bindgen(constructor)]
    pub fn new(dt: F, n_lags: Option<usize>) -> Self {
        Self { dt, n_lags }
    }

    pub fn fit(&self, y: &[F]) -> Result<JsFloatArray, JsValue> {
        let dt = self.dt;
        let n_lags = self.n_lags;
        let y = array1(y);
        let r = molrs::compute::CumulativeTrapezoid
            .fit((&y, dt, n_lags))
            .map_err(|e| JsValue::from_str(&format!("CumulativeTrapezoid: {e}")))?;
        Ok(JsFloatArray::from(r.integral.as_slice().unwrap()))
    }
}

#[wasm_bindgen(js_name = WasmPlateau)]
pub struct WasmPlateau {
    start_frac: F,
    end_frac: F,
}

#[wasm_bindgen(js_class = WasmPlateau)]
impl WasmPlateau {
    #[wasm_bindgen(constructor)]
    pub fn new(start_frac: F, end_frac: F) -> Self {
        Self {
            start_frac,
            end_frac,
        }
    }

    pub fn fit(&self, y: &[F]) -> Result<JsValue, JsValue> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            value: F,
            n_samples: usize,
            std: F,
        }
        let y = array1(y);
        let r = molrs::compute::Plateau {
            window: (self.start_frac, self.end_frac),
        }
        .fit(&y)
        .map_err(|e| JsValue::from_str(&format!("Plateau: {e}")))?;
        js_value(&Out {
            value: r.value,
            n_samples: r.n_samples,
            std: r.std,
        })
    }
}

#[wasm_bindgen(js_name = WasmDebyeFit)]
pub struct WasmDebyeFit {
    dt: F,
}

#[wasm_bindgen(js_class = WasmDebyeFit)]
impl WasmDebyeFit {
    #[wasm_bindgen(constructor)]
    pub fn new(dt: F) -> Self {
        Self { dt }
    }

    pub fn fit(&self, phi: &[F]) -> Result<JsValue, JsValue> {
        let dt = self.dt;
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            tau: F,
            amplitude: F,
            n_samples: usize,
        }
        let phi = array1(phi);
        let r = molrs::compute::DebyeFit
            .fit((&phi, dt))
            .map_err(|e| JsValue::from_str(&format!("DebyeFit: {e}")))?;
        js_value(&Out {
            tau: r.tau,
            amplitude: r.amplitude,
            n_samples: r.n_samples,
        })
    }
}

#[wasm_bindgen(js_name = WasmStaticDielectric)]
pub struct WasmStaticDielectric {
    volume: F,
    temperature: F,
    epsilon_inf: Option<F>,
}

#[wasm_bindgen(js_class = WasmStaticDielectric)]
impl WasmStaticDielectric {
    #[wasm_bindgen(constructor)]
    pub fn new(volume: F, temperature: F, epsilon_inf: Option<F>) -> Self {
        Self {
            volume,
            temperature,
            epsilon_inf,
        }
    }

    pub fn compute(&self, dipoles: &[F], n_frames: usize) -> Result<JsValue, JsValue> {
        let volume = self.volume;
        let temperature = self.temperature;
        let epsilon_inf = self.epsilon_inf;
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            epsilon: F,
            eps: Vec<F>,
            eps_mean: F,
            fluctuation: Vec<F>,
            dipole_mean: Vec<F>,
            dipole_sq_mean: Vec<F>,
            epsilon_inf: F,
            n_frames: usize,
        }
        let dipoles = array2(dipoles, n_frames, 3, "StaticDielectric dipoles")?;
        let eps_inf = epsilon_inf.unwrap_or(1.0);
        let epsilon =
            molrs::compute::static_dielectric_constant(&dipoles, volume, temperature, eps_inf)
                .map_err(|e| JsValue::from_str(&format!("StaticDielectric: {e}")))?;
        let c = molrs::compute::static_dielectric_constant_components(
            &dipoles,
            volume,
            temperature,
            eps_inf,
        )
        .map_err(|e| JsValue::from_str(&format!("StaticDielectric components: {e}")))?;
        js_value(&Out {
            epsilon,
            eps: c.eps.to_vec(),
            eps_mean: c.eps_mean,
            fluctuation: c.fluctuation.to_vec(),
            dipole_mean: c.dipole_mean.to_vec(),
            dipole_sq_mean: c.dipole_sq_mean.to_vec(),
            epsilon_inf: c.epsilon_inf,
            n_frames: c.n_frames,
        })
    }
}

#[wasm_bindgen(js_name = WasmVanHove)]
pub struct WasmVanHove {
    frames: Vec<molrs::store::frame::Frame>,
    n_r_bins: usize,
    r_max: F,
    lags: Vec<usize>,
    stride: usize,
}

#[wasm_bindgen(js_class = WasmVanHove)]
impl WasmVanHove {
    #[wasm_bindgen(constructor)]
    pub fn new(n_r_bins: usize, r_max: F, lags: Vec<usize>, stride: Option<usize>) -> Self {
        Self {
            frames: Vec::new(),
            n_r_bins,
            r_max,
            lags,
            stride: stride.unwrap_or(1),
        }
    }

    pub fn feed(&mut self, frame: &Frame) -> Result<(), JsValue> {
        frame.with_frame(|rs_frame| {
            self.frames.push(rs_frame.clone());
            Ok(())
        })
    }

    pub fn compute(&self) -> Result<JsValue, JsValue> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            r_edges: Vec<F>,
            r_centers: Vec<F>,
            lags: Vec<usize>,
            g_self: Vec<F>,
            g_distinct: Vec<F>,
            shape: [usize; 2],
            dr: F,
            has_distinct: bool,
        }
        let refs: Vec<&molrs::store::frame::Frame> = self.frames.iter().collect();
        let calc = molrs::compute::VanHove::new(self.n_r_bins, self.r_max, self.lags.clone())
            .map_err(|e| JsValue::from_str(&format!("VanHove: {e}")))?
            .with_stride(self.stride);
        let r = calc
            .compute(&refs, ())
            .map_err(|e| JsValue::from_str(&format!("VanHove compute: {e}")))?;
        let shape = [r.g_self.nrows(), r.g_self.ncols()];
        js_value(&Out {
            r_edges: r.r_edges.to_vec(),
            r_centers: r.r_centers.to_vec(),
            lags: r.lags,
            g_self: r.g_self.iter().copied().collect(),
            g_distinct: r.g_distinct.iter().copied().collect(),
            shape,
            dr: r.dr,
            has_distinct: r.has_distinct,
        })
    }

    pub fn reset(&mut self) {
        self.frames.clear();
    }
}

#[wasm_bindgen(js_name = WasmPairPersistence)]
pub struct WasmPairPersistence {
    r0: F,
    r1: F,
    method: String,
    dt: F,
    max_lag: usize,
    exclude_self: bool,
}

#[wasm_bindgen(js_class = WasmPairPersistence)]
impl WasmPairPersistence {
    #[wasm_bindgen(constructor)]
    pub fn new(r0: F, r1: F, method: String, dt: F, max_lag: usize, exclude_self: bool) -> Self {
        Self {
            r0,
            r1,
            method,
            dt,
            max_lag,
            exclude_self,
        }
    }

    pub fn compute(
        &self,
        coords_i: &[F],
        n_frames: usize,
        n_i: usize,
        coords_j: &[F],
        n_j: usize,
        box_lengths: &[F],
    ) -> Result<JsValue, JsValue> {
        let r0 = self.r0;
        let r1 = self.r1;
        let method = self.method.clone();
        let dt = self.dt;
        let max_lag = self.max_lag;
        let exclude_self = self.exclude_self;
        let ci = array3(coords_i, n_frames, n_i, 3, "PairPersistence coords_i")?;
        let cj = array3(coords_j, n_frames, n_j, 3, "PairPersistence coords_j")?;
        let bl = array2(box_lengths, n_frames, 3, "PairPersistence box_lengths")?;
        let method = molrs::compute::SurvivalMethod::parse(&method)
            .map_err(|e| JsValue::from_str(&format!("PairPersistence method: {e}")))?;
        let r = molrs::compute::pair_survival_tcf(
            &ci,
            &cj,
            &bl,
            r0,
            r1,
            method,
            dt,
            max_lag,
            exclude_self,
        )
        .map_err(|e| JsValue::from_str(&format!("PairPersistence: {e}")))?;
        js_value(&SeriesOut {
            lag_times: r.lag_times.to_vec(),
            values: r.correlation.to_vec(),
        })
    }
}

#[wasm_bindgen(js_name = WasmCorrelationFunction)]
pub struct WasmCorrelationFunction {
    inner: molrs::compute::CorrelationFunction,
}

#[wasm_bindgen(js_class = WasmCorrelationFunction)]
impl WasmCorrelationFunction {
    #[wasm_bindgen(constructor)]
    pub fn new(n_bins: usize, r_max: F, r_min: Option<F>) -> Result<Self, JsValue> {
        Ok(Self {
            inner: molrs::compute::CorrelationFunction::new(n_bins, r_max, r_min.unwrap_or(0.0))
                .map_err(|e| JsValue::from_str(&format!("CorrelationFunction: {e}")))?,
        })
    }

    pub fn compute(
        &self,
        frame: &Frame,
        neighbors: &Neighbors,
        values_a: &[F],
        values_b: &[F],
    ) -> Result<JsValue, JsValue> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            bin_edges: Vec<F>,
            bin_centers: Vec<F>,
            bin_counts: Vec<u64>,
            correlation: Vec<F>,
        }
        frame.with_frame(|rs_frame| {
            let nlists = std::slice::from_ref(&neighbors.inner);
            let va = vec![values_a.to_vec()];
            let vb = vec![values_b.to_vec()];
            let args = molrs::compute::density::correlation_function::CorrelationArgs {
                nlists,
                values_a: &va,
                values_b: &vb,
            };
            let mut out = self
                .inner
                .compute(&[rs_frame], args)
                .map_err(|e| JsValue::from_str(&format!("CorrelationFunction compute: {e}")))?;
            let r = out
                .pop()
                .ok_or_else(|| JsValue::from_str("CorrelationFunction: empty result"))?;
            js_value(&Out {
                bin_edges: r.bin_edges.to_vec(),
                bin_centers: r.bin_centers.to_vec(),
                bin_counts: r.bin_counts.to_vec(),
                correlation: r.correlation.to_vec(),
            })
        })
    }
}

#[wasm_bindgen(js_name = WasmLocalDensity)]
pub struct WasmLocalDensity {
    inner: molrs::compute::LocalDensity,
}

#[wasm_bindgen(js_class = WasmLocalDensity)]
impl WasmLocalDensity {
    #[wasm_bindgen(constructor)]
    pub fn new(r_max: F, diameter: Option<F>) -> Result<Self, JsValue> {
        let mut inner = molrs::compute::LocalDensity::new(r_max)
            .map_err(|e| JsValue::from_str(&format!("LocalDensity: {e}")))?;
        if let Some(d) = diameter {
            inner = inner.with_diameter(d);
        }
        Ok(Self { inner })
    }

    pub fn compute(&self, frame: &Frame, neighbors: &Neighbors) -> Result<JsValue, JsValue> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            num_neighbors: Vec<F>,
            density: Vec<F>,
        }
        frame.with_frame(|rs_frame| {
            let nlists = std::slice::from_ref(&neighbors.inner);
            let mut out = self
                .inner
                .compute(&[rs_frame], nlists)
                .map_err(|e| JsValue::from_str(&format!("LocalDensity compute: {e}")))?;
            let r = out
                .pop()
                .ok_or_else(|| JsValue::from_str("LocalDensity: empty result"))?;
            js_value(&Out {
                num_neighbors: r.num_neighbors,
                density: r.density,
            })
        })
    }
}

#[wasm_bindgen(js_name = WasmGaussianDensity)]
pub struct WasmGaussianDensity {
    inner: molrs::compute::GaussianDensity,
}

#[wasm_bindgen(js_class = WasmGaussianDensity)]
impl WasmGaussianDensity {
    #[wasm_bindgen(constructor)]
    pub fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        sigma: F,
        r_max: Option<F>,
    ) -> Result<Self, JsValue> {
        let mut inner = molrs::compute::GaussianDensity::new(nx, ny, nz, sigma)
            .map_err(|e| JsValue::from_str(&format!("GaussianDensity: {e}")))?;
        if let Some(r) = r_max {
            inner = inner.with_r_max(r);
        }
        Ok(Self { inner })
    }

    pub fn compute(&self, frame: &Frame) -> Result<JsValue, JsValue> {
        frame.with_frame(|rs_frame| {
            let mut out = self
                .inner
                .compute(&[rs_frame], ())
                .map_err(|e| JsValue::from_str(&format!("GaussianDensity compute: {e}")))?;
            let r = out
                .pop()
                .ok_or_else(|| JsValue::from_str("GaussianDensity: empty result"))?;
            let shape = [
                r.density.shape()[0],
                r.density.shape()[1],
                r.density.shape()[2],
            ];
            js_value(&Grid3Out {
                data: r.density.iter().copied().collect::<Vec<F>>(),
                shape,
            })
        })
    }
}

#[wasm_bindgen(js_name = WasmSphereVoxelization)]
pub struct WasmSphereVoxelization {
    inner: molrs::compute::SphereVoxelization,
}

#[wasm_bindgen(js_class = WasmSphereVoxelization)]
impl WasmSphereVoxelization {
    #[wasm_bindgen(constructor)]
    pub fn new(nx: usize, ny: usize, nz: usize, r_max: F) -> Result<Self, JsValue> {
        Ok(Self {
            inner: molrs::compute::SphereVoxelization::new(nx, ny, nz, r_max)
                .map_err(|e| JsValue::from_str(&format!("SphereVoxelization: {e}")))?,
        })
    }

    pub fn compute(&self, frame: &Frame) -> Result<JsValue, JsValue> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            voxels: Grid3Out<u8>,
            raw_counts: Grid3Out<u32>,
        }
        frame.with_frame(|rs_frame| {
            let mut out = self
                .inner
                .compute(&[rs_frame], ())
                .map_err(|e| JsValue::from_str(&format!("SphereVoxelization compute: {e}")))?;
            let r = out
                .pop()
                .ok_or_else(|| JsValue::from_str("SphereVoxelization: empty result"))?;
            let shape = [
                r.voxels.shape()[0],
                r.voxels.shape()[1],
                r.voxels.shape()[2],
            ];
            js_value(&Out {
                voxels: Grid3Out {
                    data: r.voxels.iter().copied().collect::<Vec<u8>>(),
                    shape,
                },
                raw_counts: Grid3Out {
                    data: r.raw_counts.iter().copied().collect::<Vec<u32>>(),
                    shape,
                },
            })
        })
    }
}

#[wasm_bindgen(js_name = WasmSpatialDistribution)]
pub struct WasmSpatialDistribution {
    inner: molrs::compute::SpatialDistribution,
    frames: Vec<molrs::store::frame::Frame>,
    bulk_density: Option<F>,
}

#[wasm_bindgen(js_class = WasmSpatialDistribution)]
impl WasmSpatialDistribution {
    #[wasm_bindgen(constructor)]
    pub fn new(
        reference: &[u32],
        template: &[F],
        target: &[u32],
        nx: usize,
        ny: usize,
        nz: usize,
        extent_x: F,
        extent_y: F,
        extent_z: F,
        bulk_density: Option<F>,
    ) -> Result<Self, JsValue> {
        let reference_usize = usize_vec(reference);
        let target_usize = usize_vec(target);
        let template = array2(
            template,
            reference_usize.len(),
            3,
            "SpatialDistribution template",
        )?;
        let grid = molrs::compute::GridSpec {
            n: [nx, ny, nz],
            extent: [extent_x, extent_y, extent_z],
        };
        let mut inner =
            molrs::compute::SpatialDistribution::new(reference_usize, template, target_usize, grid)
                .map_err(|e| JsValue::from_str(&format!("SpatialDistribution: {e}")))?;
        if let Some(rho) = bulk_density {
            inner = inner.with_bulk_density(rho);
        }
        Ok(Self {
            inner,
            frames: Vec::new(),
            bulk_density,
        })
    }

    #[wasm_bindgen(js_name = setOrientationPairs)]
    pub fn set_orientation_pairs(&mut self, pairs: &[u32]) -> Result<(), JsValue> {
        self.inner = self
            .inner
            .clone()
            .with_orientation(usize_pairs(pairs, "SpatialDistribution orientationPairs")?);
        Ok(())
    }

    pub fn feed(&mut self, frame: &Frame) -> Result<(), JsValue> {
        frame.with_frame(|rs_frame| {
            self.frames.push(rs_frame.clone());
            Ok(())
        })
    }

    pub fn compute(&self) -> Result<JsValue, JsValue> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            counts: Grid3Out<F>,
            density: Grid3Out<F>,
            g_sdf: Option<Grid3Out<F>>,
            orientation: Option<Vec<F>>,
            orientation_shape: Option<[usize; 4]>,
            voxel_volume: F,
            n_frames: usize,
            bulk_density: Option<F>,
        }
        let refs: Vec<&molrs::store::frame::Frame> = self.frames.iter().collect();
        let r = self
            .inner
            .compute(&refs, ())
            .map_err(|e| JsValue::from_str(&format!("SpatialDistribution compute: {e}")))?;
        let shape = [r.n[0], r.n[1], r.n[2]];
        let g_sdf = r.g_sdf.as_ref().map(|g| Grid3Out {
            data: g.iter().copied().collect::<Vec<F>>(),
            shape,
        });
        let (orientation, orientation_shape) = match r.orientation.as_ref() {
            Some(o) => (
                Some(o.iter().copied().collect::<Vec<F>>()),
                Some([r.n[0], r.n[1], r.n[2], 3]),
            ),
            None => (None, None),
        };
        js_value(&Out {
            counts: Grid3Out {
                data: r.counts.iter().copied().collect::<Vec<F>>(),
                shape,
            },
            density: Grid3Out {
                data: r.density.iter().copied().collect::<Vec<F>>(),
                shape,
            },
            g_sdf,
            orientation,
            orientation_shape,
            voxel_volume: r.voxel_volume,
            n_frames: r.n_frames,
            bulk_density: self.bulk_density,
        })
    }

    pub fn reset(&mut self) {
        self.frames.clear();
    }
}

#[wasm_bindgen(js_name = WasmSteinhardt)]
pub struct WasmSteinhardt {
    inner: molrs::compute::Steinhardt,
}

#[wasm_bindgen(js_class = WasmSteinhardt)]
impl WasmSteinhardt {
    #[wasm_bindgen(constructor)]
    pub fn new(
        l_values: &[u32],
        average: Option<bool>,
        wl: Option<bool>,
        wl_normalize: Option<bool>,
    ) -> Result<Self, JsValue> {
        let mut inner = molrs::compute::Steinhardt::new(l_values)
            .map_err(|e| JsValue::from_str(&format!("Steinhardt: {e}")))?;
        inner = inner
            .with_average(average.unwrap_or(false))
            .with_wl(wl.unwrap_or(false))
            .with_wl_normalize(wl_normalize.unwrap_or(false));
        Ok(Self { inner })
    }

    pub fn compute(&self, frame: &Frame, neighbors: &Neighbors) -> Result<JsValue, JsValue> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            l: Vec<u32>,
            ql: Vec<Vec<F>>,
            wl: Option<Vec<Vec<F>>>,
            qlm_re_im: Vec<Vec<F>>,
        }
        frame.with_frame(|rs_frame| {
            let nlists = std::slice::from_ref(&neighbors.inner);
            let mut out = self
                .inner
                .compute(&[rs_frame], nlists)
                .map_err(|e| JsValue::from_str(&format!("Steinhardt compute: {e}")))?;
            let r = out
                .pop()
                .ok_or_else(|| JsValue::from_str("Steinhardt: empty result"))?;
            let qlm_re_im = r
                .qlm
                .iter()
                .map(|band| band.iter().flat_map(|c| [c.re, c.im]).collect())
                .collect();
            js_value(&Out {
                l: r.l,
                ql: r.ql,
                wl: r.wl,
                qlm_re_im,
            })
        })
    }
}

#[wasm_bindgen(js_name = WasmHexatic)]
pub struct WasmHexatic {
    inner: molrs::compute::Hexatic,
}

#[wasm_bindgen(js_class = WasmHexatic)]
impl WasmHexatic {
    #[wasm_bindgen(constructor)]
    pub fn new(k: u32) -> Result<Self, JsValue> {
        Ok(Self {
            inner: molrs::compute::Hexatic::new(k)
                .map_err(|e| JsValue::from_str(&format!("Hexatic: {e}")))?,
        })
    }

    pub fn compute(&self, frame: &Frame, neighbors: &Neighbors) -> Result<JsValue, JsValue> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            k: u32,
            psi_re_im: Vec<F>,
        }
        frame.with_frame(|rs_frame| {
            let nlists = std::slice::from_ref(&neighbors.inner);
            let mut out = self
                .inner
                .compute(&[rs_frame], nlists)
                .map_err(|e| JsValue::from_str(&format!("Hexatic compute: {e}")))?;
            let r = out
                .pop()
                .ok_or_else(|| JsValue::from_str("Hexatic: empty result"))?;
            js_value(&Out {
                k: r.k,
                psi_re_im: r.psi.iter().flat_map(|c| [c.re, c.im]).collect(),
            })
        })
    }
}

#[wasm_bindgen(js_name = WasmNematic)]
pub struct WasmNematic;

#[wasm_bindgen(js_class = WasmNematic)]
impl WasmNematic {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self
    }

    pub fn compute(&self, directors: &[F]) -> Result<JsValue, JsValue> {
        let directors = vectors3(directors, "Nematic directors")?;
        let dummy = molrs::store::frame::Frame::new();
        let calc = molrs::compute::Nematic::new();
        let mut out = calc
            .compute(&[&dummy], &directors)
            .map_err(|e| JsValue::from_str(&format!("Nematic: {e}")))?;
        let r = out
            .pop()
            .ok_or_else(|| JsValue::from_str("Nematic: empty result"))?;
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            order: F,
            eigenvalues: [F; 3],
            director: [F; 3],
            q_tensor: [[F; 3]; 3],
        }
        js_value(&Out {
            order: r.order,
            eigenvalues: r.eigenvalues,
            director: r.director,
            q_tensor: r.q_tensor,
        })
    }
}

#[wasm_bindgen(js_name = WasmCubatic)]
pub struct WasmCubatic {
    seed: u64,
    initial_temp: F,
    cooling_rate: F,
    n_steps: usize,
    n_chains: usize,
}

#[wasm_bindgen(js_class = WasmCubatic)]
impl WasmCubatic {
    #[wasm_bindgen(constructor)]
    pub fn new(
        seed: Option<f64>,
        initial_temp: Option<F>,
        cooling_rate: Option<F>,
        n_steps: Option<usize>,
        n_chains: Option<usize>,
    ) -> Self {
        Self {
            seed: seed.unwrap_or(0.0) as u64,
            initial_temp: initial_temp.unwrap_or(1.0),
            cooling_rate: cooling_rate.unwrap_or(0.95),
            n_steps: n_steps.unwrap_or(500),
            n_chains: n_chains.unwrap_or(4),
        }
    }

    pub fn compute(&self, directors: &[F]) -> Result<JsValue, JsValue> {
        let directors = vectors3(directors, "Cubatic directors")?;
        let dummy = molrs::store::frame::Frame::new();
        let calc = molrs::compute::Cubatic::new()
            .with_seed(self.seed)
            .with_initial_temp(self.initial_temp)
            .with_cooling_rate(self.cooling_rate)
            .with_n_steps(self.n_steps)
            .with_n_chains(self.n_chains);
        let mut out = calc
            .compute(&[&dummy], &directors)
            .map_err(|e| JsValue::from_str(&format!("Cubatic: {e}")))?;
        let r = out
            .pop()
            .ok_or_else(|| JsValue::from_str("Cubatic: empty result"))?;
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            order: F,
            director_basis: [[F; 3]; 3],
        }
        js_value(&Out {
            order: r.order,
            director_basis: r.director_basis,
        })
    }
}

#[wasm_bindgen(js_name = WasmSolidLiquid)]
pub struct WasmSolidLiquid {
    inner: molrs::compute::SolidLiquid,
}

#[wasm_bindgen(js_class = WasmSolidLiquid)]
impl WasmSolidLiquid {
    #[wasm_bindgen(constructor)]
    pub fn new(
        l: u32,
        q_threshold: Option<F>,
        n_threshold: Option<u32>,
        normalize_q: Option<bool>,
    ) -> Self {
        let mut inner = molrs::compute::SolidLiquid::new(l);
        if let Some(t) = q_threshold {
            inner = inner.with_q_threshold(t);
        }
        if let Some(n) = n_threshold {
            inner = inner.with_n_threshold(n);
        }
        if let Some(on) = normalize_q {
            inner = inner.with_normalize_q(on);
        }
        Self { inner }
    }

    pub fn compute(&self, frame: &Frame, neighbors: &Neighbors) -> Result<JsValue, JsValue> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            l: u32,
            n_solid_bonds: Vec<u32>,
            is_solid: Vec<bool>,
        }
        frame.with_frame(|rs_frame| {
            let nlists = std::slice::from_ref(&neighbors.inner);
            let mut out = self
                .inner
                .compute(&[rs_frame], nlists)
                .map_err(|e| JsValue::from_str(&format!("SolidLiquid compute: {e}")))?;
            let r = out
                .pop()
                .ok_or_else(|| JsValue::from_str("SolidLiquid: empty result"))?;
            js_value(&Out {
                l: r.l,
                n_solid_bonds: r.n_solid_bonds,
                is_solid: r.is_solid,
            })
        })
    }
}

#[wasm_bindgen(js_name = WasmRotationalAutocorrelation)]
pub struct WasmRotationalAutocorrelation {
    l: u32,
}

#[wasm_bindgen(js_class = WasmRotationalAutocorrelation)]
impl WasmRotationalAutocorrelation {
    #[wasm_bindgen(constructor)]
    pub fn new(l: u32) -> Self {
        Self { l }
    }

    pub fn compute(&self, reference: &[F], orientations: &[F]) -> Result<JsValue, JsValue> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            l: u32,
            psi: Vec<F>,
            mean: F,
        }
        let reference = quats(reference, "RotationalAutocorrelation reference")?;
        let orientations = quats(orientations, "RotationalAutocorrelation orientations")?;
        let dummy = molrs::store::frame::Frame::new();
        let calc = molrs::compute::RotationalAutocorrelation::new(self.l);
        let args =
            molrs::compute::order::rotational_autocorrelation::RotationalAutocorrelationArgs {
                ref_orientations: &reference,
                orientations: &orientations,
            };
        let mut out = calc
            .compute(&[&dummy], args)
            .map_err(|e| JsValue::from_str(&format!("RotationalAutocorrelation: {e}")))?;
        let r = out
            .pop()
            .ok_or_else(|| JsValue::from_str("RotationalAutocorrelation: empty result"))?;
        js_value(&Out {
            l: r.l,
            psi: r.psi,
            mean: r.mean,
        })
    }
}

#[wasm_bindgen(js_name = WasmBondOrder)]
pub struct WasmBondOrder {
    inner: molrs::compute::BondOrder,
}

#[wasm_bindgen(js_class = WasmBondOrder)]
impl WasmBondOrder {
    #[wasm_bindgen(constructor)]
    pub fn new(n_theta: usize, n_phi: usize) -> Result<Self, JsValue> {
        Ok(Self {
            inner: molrs::compute::BondOrder::new(n_theta, n_phi)
                .map_err(|e| JsValue::from_str(&format!("BondOrder: {e}")))?,
        })
    }

    pub fn compute(&self, frame: &Frame, neighbors: &Neighbors) -> Result<JsValue, JsValue> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            bond_order: Grid2Out,
            raw_counts: Vec<u64>,
            shape: [usize; 2],
            theta_edges: Vec<F>,
            phi_edges: Vec<F>,
        }
        frame.with_frame(|rs_frame| {
            let nlists = std::slice::from_ref(&neighbors.inner);
            let mut out = self
                .inner
                .compute(&[rs_frame], nlists)
                .map_err(|e| JsValue::from_str(&format!("BondOrder compute: {e}")))?;
            let r = out
                .pop()
                .ok_or_else(|| JsValue::from_str("BondOrder: empty result"))?;
            let shape = [r.bond_order.nrows(), r.bond_order.ncols()];
            js_value(&Out {
                bond_order: Grid2Out {
                    data: r.bond_order.iter().copied().collect(),
                    shape,
                },
                raw_counts: r.raw_counts.iter().copied().collect(),
                shape,
                theta_edges: r.theta_edges,
                phi_edges: r.phi_edges,
            })
        })
    }
}

#[wasm_bindgen(js_name = WasmLocalDescriptors)]
pub struct WasmLocalDescriptors {
    inner: molrs::compute::LocalDescriptors,
}

#[wasm_bindgen(js_class = WasmLocalDescriptors)]
impl WasmLocalDescriptors {
    #[wasm_bindgen(constructor)]
    pub fn new(l_max: u32) -> Self {
        Self {
            inner: molrs::compute::LocalDescriptors::new(l_max),
        }
    }

    pub fn compute(&self, frame: &Frame, neighbors: &Neighbors) -> Result<JsValue, JsValue> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            l_max: u32,
            n_sphs: usize,
            descriptors_re_im: Vec<F>,
        }
        frame.with_frame(|rs_frame| {
            let nlists = std::slice::from_ref(&neighbors.inner);
            let mut out = self
                .inner
                .compute(&[rs_frame], nlists)
                .map_err(|e| JsValue::from_str(&format!("LocalDescriptors compute: {e}")))?;
            let r = out
                .pop()
                .ok_or_else(|| JsValue::from_str("LocalDescriptors: empty result"))?;
            js_value(&Out {
                l_max: r.l_max,
                n_sphs: r.n_sphs,
                descriptors_re_im: r.descriptors.iter().flat_map(|c| [c.re, c.im]).collect(),
            })
        })
    }
}

#[wasm_bindgen(js_name = WasmAngularSeparation)]
pub struct WasmAngularSeparation {
    equivalent_orientations: bool,
}

#[wasm_bindgen(js_class = WasmAngularSeparation)]
impl WasmAngularSeparation {
    #[wasm_bindgen(constructor)]
    pub fn new(equivalent_orientations: Option<bool>) -> Self {
        Self {
            equivalent_orientations: equivalent_orientations.unwrap_or(true),
        }
    }

    #[wasm_bindgen(js_name = computeGlobal)]
    pub fn compute_global(&self, query: &[F], global: &[F]) -> Result<JsValue, JsValue> {
        let query = quats(query, "AngularSeparation query")?;
        let global = quats(global, "AngularSeparation global")?;
        let dummy = molrs::store::frame::Frame::new();
        let calc = molrs::compute::AngularSeparationGlobal::new()
            .with_equivalent_orientations(self.equivalent_orientations);
        let args = molrs::compute::environment::angular_separation::AngularSeparationGlobalArgs {
            query: &query,
            global: &global,
        };
        let mut out = calc
            .compute(&[&dummy], args)
            .map_err(|e| JsValue::from_str(&format!("AngularSeparationGlobal: {e}")))?;
        let r = out
            .pop()
            .ok_or_else(|| JsValue::from_str("AngularSeparationGlobal: empty result"))?;
        let shape = [r.angles.nrows(), r.angles.ncols()];
        js_value(&Grid2Out {
            data: r.angles.iter().copied().collect(),
            shape,
        })
    }

    #[wasm_bindgen(js_name = computeNeighbor)]
    pub fn compute_neighbor(
        &self,
        frame: &Frame,
        neighbors: &Neighbors,
        query: &[F],
        points: &[F],
    ) -> Result<JsFloatArray, JsValue> {
        let query = quats(query, "AngularSeparation query")?;
        let points = quats(points, "AngularSeparation points")?;
        frame.with_frame(|rs_frame| {
            let nlists = std::slice::from_ref(&neighbors.inner);
            let q = vec![query];
            let p = vec![points];
            let calc = molrs::compute::AngularSeparationNeighbor::new()
                .with_equivalent_orientations(self.equivalent_orientations);
            let args =
                molrs::compute::environment::angular_separation::AngularSeparationNeighborArgs {
                    nlists,
                    query_orientations: &q,
                    point_orientations: &p,
                };
            let mut out = calc
                .compute(&[rs_frame], args)
                .map_err(|e| JsValue::from_str(&format!("AngularSeparationNeighbor: {e}")))?;
            let r = out
                .pop()
                .ok_or_else(|| JsValue::from_str("AngularSeparationNeighbor: empty result"))?;
            Ok(JsFloatArray::from(r.angles.as_slice()))
        })
    }
}

#[wasm_bindgen(js_name = WasmMatchEnv)]
pub struct WasmMatchEnv {
    inner: molrs::compute::MatchEnv,
}

#[wasm_bindgen(js_class = WasmMatchEnv)]
impl WasmMatchEnv {
    #[wasm_bindgen(constructor)]
    pub fn new(
        rmsd_threshold: F,
        registration: Option<bool>,
        max_neighbors_for_registration: Option<usize>,
    ) -> Result<Self, JsValue> {
        let mut inner = molrs::compute::MatchEnv::new(rmsd_threshold)
            .map_err(|e| JsValue::from_str(&format!("MatchEnv: {e}")))?;
        inner = inner.with_registration(registration.unwrap_or(false));
        if let Some(n) = max_neighbors_for_registration {
            inner = inner.with_max_neighbors_for_registration(n);
        }
        Ok(Self { inner })
    }

    pub fn compute(&self, frame: &Frame, neighbors: &Neighbors) -> Result<JsValue, JsValue> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            cluster_idx: Vec<u32>,
            n_clusters: usize,
            fingerprints: Vec<Vec<F>>,
        }
        frame.with_frame(|rs_frame| {
            let nlists = std::slice::from_ref(&neighbors.inner);
            let mut out = self
                .inner
                .compute(&[rs_frame], nlists)
                .map_err(|e| JsValue::from_str(&format!("MatchEnv compute: {e}")))?;
            let r = out
                .pop()
                .ok_or_else(|| JsValue::from_str("MatchEnv: empty result"))?;
            js_value(&Out {
                cluster_idx: r.cluster_idx,
                n_clusters: r.n_clusters,
                fingerprints: r.fingerprints,
            })
        })
    }
}

#[wasm_bindgen(js_name = WasmStaticStructureFactorDebye)]
pub struct WasmStaticStructureFactorDebye {
    inner: molrs::compute::StaticStructureFactorDebye,
}

#[wasm_bindgen(js_class = WasmStaticStructureFactorDebye)]
impl WasmStaticStructureFactorDebye {
    /// Sample `n_k` scattering vectors evenly over `[k_min, k_max]` (A^-1).
    #[wasm_bindgen(constructor)]
    pub fn new(k_min: F, k_max: F, n_k: usize) -> Result<Self, JsValue> {
        Ok(Self {
            inner: molrs::compute::StaticStructureFactorDebye::linspace(k_min, k_max, n_k)
                .map_err(|e| JsValue::from_str(&format!("StaticStructureFactorDebye: {e}")))?,
        })
    }

    /// Sample an explicit, possibly non-uniform, set of scattering vectors.
    #[wasm_bindgen(js_name = fromKValues)]
    pub fn from_k_values(k_values: &[F]) -> Result<Self, JsValue> {
        Ok(Self {
            inner: molrs::compute::StaticStructureFactorDebye::new(k_values)
                .map_err(|e| JsValue::from_str(&format!("StaticStructureFactorDebye: {e}")))?,
        })
    }

    pub fn compute(&self, frame: &Frame) -> Result<JsValue, JsValue> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            k_values: Vec<F>,
            sk: Vec<F>,
            n_particles: usize,
        }
        frame.with_frame(|rs_frame| {
            let mut out = self.inner.compute(&[rs_frame], ()).map_err(|e| {
                JsValue::from_str(&format!("StaticStructureFactorDebye compute: {e}"))
            })?;
            let r = out
                .pop()
                .ok_or_else(|| JsValue::from_str("StaticStructureFactorDebye: empty result"))?;
            js_value(&Out {
                k_values: r.k_values.to_vec(),
                sk: r.sk.to_vec(),
                n_particles: r.n_particles,
            })
        })
    }
}

#[wasm_bindgen(js_name = WasmDiffractionPattern)]
pub struct WasmDiffractionPattern {
    inner: molrs::compute::DiffractionPattern,
}

#[wasm_bindgen(js_class = WasmDiffractionPattern)]
impl WasmDiffractionPattern {
    #[wasm_bindgen(constructor)]
    pub fn new(n_grid: usize, sigma: F, axis: Option<usize>) -> Result<Self, JsValue> {
        let mut inner = molrs::compute::DiffractionPattern::new(n_grid, sigma)
            .map_err(|e| JsValue::from_str(&format!("DiffractionPattern: {e}")))?;
        if let Some(axis) = axis {
            inner = inner.with_axis(axis);
        }
        Ok(Self { inner })
    }

    pub fn compute(&self, frame: &Frame) -> Result<JsValue, JsValue> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            diffraction: Grid2Out,
            image: Grid2Out,
        }
        frame.with_frame(|rs_frame| {
            let mut out = self
                .inner
                .compute(&[rs_frame], ())
                .map_err(|e| JsValue::from_str(&format!("DiffractionPattern compute: {e}")))?;
            let r = out
                .pop()
                .ok_or_else(|| JsValue::from_str("DiffractionPattern: empty result"))?;
            let shape = [r.diffraction.nrows(), r.diffraction.ncols()];
            js_value(&Out {
                diffraction: Grid2Out {
                    data: r.diffraction.iter().copied().collect(),
                    shape,
                },
                image: Grid2Out {
                    data: r.image.iter().copied().collect(),
                    shape,
                },
            })
        })
    }
}

fn distribution_compute<O: molrs::compute::distribution::Observable + Sync>(
    frame: &Frame,
    calc: molrs::compute::distribution::DistributionFunction<O>,
    groups: molrs::compute::distribution::AtomGroups,
) -> Result<JsValue, JsValue> {
    #[derive(Serialize)]
    #[serde(rename_all = "camelCase")]
    struct Out {
        bin_centers: Vec<F>,
        bin_edges: Vec<F>,
        counts: Vec<F>,
        density: Vec<F>,
        density_sin_corrected: Option<Vec<F>>,
        bin_width: F,
        n_binned: F,
        n_raw_samples: usize,
        n_frames: usize,
        angular: bool,
    }
    frame.with_frame(|rs_frame| {
        let r = calc
            .compute(&[rs_frame], &groups)
            .map_err(|e| JsValue::from_str(&format!("Distribution compute: {e}")))?;
        js_value(&Out {
            bin_centers: r.bin_centers.to_vec(),
            bin_edges: r.bin_edges.to_vec(),
            counts: r.counts.to_vec(),
            density: r.density.to_vec(),
            density_sin_corrected: r.density_sin_corrected.map(|v| v.to_vec()),
            bin_width: r.bin_width,
            n_binned: r.n_binned,
            n_raw_samples: r.n_raw_samples,
            n_frames: r.n_frames,
            angular: r.angular,
        })
    })
}

#[wasm_bindgen(js_name = WasmDistanceDistribution)]
pub struct WasmDistanceDistribution {
    n_bins: usize,
    min: F,
    max: F,
}

#[wasm_bindgen(js_class = WasmDistanceDistribution)]
impl WasmDistanceDistribution {
    #[wasm_bindgen(constructor)]
    pub fn new(n_bins: usize, min: F, max: F) -> Self {
        Self { n_bins, min, max }
    }

    pub fn compute(&self, frame: &Frame, pairs: &[u32]) -> Result<JsValue, JsValue> {
        let groups = molrs::compute::distribution::AtomGroups::new(2, pairs.iter().map(|&v| v as u64).collect())
            .map_err(|e| JsValue::from_str(&format!("DistanceDistribution groups: {e}")))?;
        let calc = molrs::compute::distribution::DistributionFunction::new(
            molrs::compute::distribution::DistanceObservable,
            self.n_bins,
            self.min,
            self.max,
        )
        .map_err(|e| JsValue::from_str(&format!("DistanceDistribution: {e}")))?;
        distribution_compute(frame, calc, groups)
    }
}

#[wasm_bindgen(js_name = WasmAngleDistribution)]
pub struct WasmAngleDistribution {
    n_bins: usize,
}

#[wasm_bindgen(js_class = WasmAngleDistribution)]
impl WasmAngleDistribution {
    #[wasm_bindgen(constructor)]
    pub fn new(n_bins: usize) -> Self {
        Self { n_bins }
    }

    pub fn compute(&self, frame: &Frame, triples: &[u32]) -> Result<JsValue, JsValue> {
        let groups = molrs::compute::distribution::AtomGroups::new(3, triples.iter().map(|&v| v as u64).collect())
            .map_err(|e| JsValue::from_str(&format!("AngleDistribution groups: {e}")))?;
        let calc = molrs::compute::distribution::DistributionFunction::over_natural_range(
            molrs::compute::distribution::AngleObservable,
            self.n_bins,
        )
        .map_err(|e| JsValue::from_str(&format!("AngleDistribution: {e}")))?;
        distribution_compute(frame, calc, groups)
    }
}

#[wasm_bindgen(js_name = WasmDihedralDistribution)]
pub struct WasmDihedralDistribution {
    n_bins: usize,
}

#[wasm_bindgen(js_class = WasmDihedralDistribution)]
impl WasmDihedralDistribution {
    #[wasm_bindgen(constructor)]
    pub fn new(n_bins: usize) -> Self {
        Self { n_bins }
    }

    pub fn compute(&self, frame: &Frame, quads: &[u32]) -> Result<JsValue, JsValue> {
        let groups = molrs::compute::distribution::AtomGroups::new(4, quads.iter().map(|&v| v as u64).collect())
            .map_err(|e| JsValue::from_str(&format!("DihedralDistribution groups: {e}")))?;
        let calc = molrs::compute::distribution::DistributionFunction::over_natural_range(
            molrs::compute::distribution::DihedralObservable,
            self.n_bins,
        )
        .map_err(|e| JsValue::from_str(&format!("DihedralDistribution: {e}")))?;
        distribution_compute(frame, calc, groups)
    }
}

#[wasm_bindgen(js_name = WasmHBonds)]
pub struct WasmHBonds {
    donors: Vec<(u32, u32)>,
    acceptors: Vec<u32>,
    dist_cutoff: F,
    dist_kind: String,
    angle_cutoff: F,
    frames: Vec<molrs::store::frame::Frame>,
}

#[wasm_bindgen(js_class = WasmHBonds)]
impl WasmHBonds {
    #[wasm_bindgen(constructor)]
    pub fn new(
        donors: &[u32],
        acceptors: &[u32],
        dist_cutoff: Option<F>,
        dist_kind: Option<String>,
        angle_cutoff: Option<F>,
    ) -> Result<Self, JsValue> {
        Ok(Self {
            donors: u32_pairs(donors, "HBonds donors")?,
            acceptors: acceptors.to_vec(),
            dist_cutoff: dist_cutoff.unwrap_or(3.5),
            dist_kind: dist_kind.unwrap_or_else(|| "donor_acceptor".to_string()),
            angle_cutoff: angle_cutoff.unwrap_or(150.0),
            frames: Vec::new(),
        })
    }

    pub fn feed(&mut self, frame: &Frame) -> Result<(), JsValue> {
        frame.with_frame(|rs_frame| {
            self.frames.push(rs_frame.clone());
            Ok(())
        })
    }

    pub fn compute(&self) -> Result<JsValue, JsValue> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct BondOut {
            donor: u32,
            hydrogen: u32,
            acceptor: u32,
            distance: F,
            angle: F,
        }
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            per_frame: Vec<Vec<BondOut>>,
            counts: Vec<usize>,
        }
        let dist_kind = match self.dist_kind.to_ascii_lowercase().as_str() {
            "donor_acceptor" | "donor-acceptor" | "da" => molrs::compute::DistKind::DonorAcceptor,
            "hydrogen_acceptor" | "hydrogen-acceptor" | "ha" => {
                molrs::compute::DistKind::HydrogenAcceptor
            }
            other => {
                return Err(JsValue::from_str(&format!(
                    "HBonds distKind: unknown {other}"
                )));
            }
        };
        let criterion =
            molrs::compute::HBondCriterion::new(self.dist_cutoff, dist_kind, self.angle_cutoff);
        let calc =
            molrs::compute::HBonds::new(self.donors.clone(), self.acceptors.clone(), criterion);
        let refs: Vec<&molrs::store::frame::Frame> = self.frames.iter().collect();
        let r = calc
            .compute(&refs, ())
            .map_err(|e| JsValue::from_str(&format!("HBonds: {e}")))?;
        let per_frame = r
            .per_frame
            .into_iter()
            .map(|frame| {
                frame
                    .into_iter()
                    .map(|b| BondOut {
                        donor: b.donor,
                        hydrogen: b.hydrogen,
                        acceptor: b.acceptor,
                        distance: b.distance,
                        angle: b.angle,
                    })
                    .collect()
            })
            .collect();
        js_value(&Out {
            per_frame,
            counts: r.counts,
        })
    }

    pub fn reset(&mut self) {
        self.frames.clear();
    }
}

#[wasm_bindgen(js_name = WasmHBondLifetime)]
pub struct WasmHBondLifetime {
    dt: F,
    max_lag: usize,
}

#[wasm_bindgen(js_class = WasmHBondLifetime)]
impl WasmHBondLifetime {
    #[wasm_bindgen(constructor)]
    pub fn new(dt: F, max_lag: usize) -> Self {
        Self { dt, max_lag }
    }

    pub fn compute(
        &self,
        presence: &[u8],
        n_bonds: usize,
        n_frames: usize,
    ) -> Result<JsValue, JsValue> {
        let dt = self.dt;
        let max_lag = self.max_lag;
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            lag_times: Vec<F>,
            continuous: Vec<F>,
            intermittent: Vec<F>,
            tau_continuous: F,
            tau_intermittent: F,
        }
        if presence.len() != n_bonds * n_frames {
            return Err(JsValue::from_str("HBondLifetime: presence length mismatch"));
        }
        let present: Vec<Vec<bool>> = presence
            .chunks_exact(n_frames)
            .map(|row| row.iter().map(|&v| v != 0).collect())
            .collect();
        let r = molrs::compute::hbond_lifetimes(&present, dt, max_lag)
            .map_err(|e| JsValue::from_str(&format!("HBondLifetime: {e}")))?;
        js_value(&Out {
            lag_times: r.lag_times.to_vec(),
            continuous: r.continuous.to_vec(),
            intermittent: r.intermittent.to_vec(),
            tau_continuous: r.tau_continuous,
            tau_intermittent: r.tau_intermittent,
        })
    }
}

#[wasm_bindgen(js_name = WasmHBondNetwork)]
pub struct WasmHBondNetwork;

#[wasm_bindgen(js_class = WasmHBondNetwork)]
impl WasmHBondNetwork {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self
    }

    pub fn compute(&self, n_nodes: usize, edges: &[u32]) -> Result<JsValue, JsValue> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            component_sizes: Vec<usize>,
            num_components: usize,
        }
        let edges = usize_pairs(edges, "HBondNetwork edges")?;
        let r = molrs::compute::hbond_components(n_nodes, &edges);
        js_value(&Out {
            component_sizes: r.component_sizes,
            num_components: r.num_components,
        })
    }
}

// ===========================================================================
// Per-atom orientations — the input PMFT needs beyond bare positions
// ===========================================================================

/// Borrow an `F`-typed column of a block as a contiguous slice.
fn f_col<'a>(atoms: &'a molrs::store::block::Block, col: &str) -> Option<&'a [F]> {
    use molrs::store::block::BlockDtype;
    <F as BlockDtype>::from_column(atoms.get(col)?)?.as_slice()
}

/// Per-atom unit quaternions `(w, i, j, k)`, normalized. `None` when the atoms
/// block does not carry the canonical [`keys::QUAT`] columns.
fn quaternions_from_frame(frame: &molrs::store::frame::Frame) -> Option<Vec<[F; 4]>> {
    let atoms = frame.get("atoms")?;
    let [w, i, j, k] = keys::QUAT.map(|col| f_col(atoms, col));
    let (w, i, j, k) = (w?, i?, j?, k?);
    Some(
        (0..w.len())
            .map(|n| {
                let q = [w[n], i[n], j[n], k[n]];
                let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
                if norm > 0.0 {
                    [q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm]
                } else {
                    [1.0, 0.0, 0.0, 0.0]
                }
            })
            .collect(),
    )
}

/// Per-atom 2-D orientation angle (radians), the z-rotation of the stored
/// quaternion: `θ = 2·atan2(q_k, q_w)`.
///
/// There is deliberately no separate angle column — the quaternion already
/// encodes the orientation, and a second column would be a second truth.
fn angles_from_frame(frame: &molrs::store::frame::Frame) -> Option<Vec<F>> {
    quaternions_from_frame(frame)
        .map(|quats| quats.iter().map(|q| 2.0 * q[3].atan2(q[0])).collect())
}

/// `angles_from_frame` for the analyses whose orientations are mandatory.
fn require_angles(frame: &molrs::store::frame::Frame, what: &str) -> Result<Vec<F>, JsValue> {
    angles_from_frame(frame).ok_or_else(|| {
        JsValue::from_str(&format!(
            "{what} needs per-atom orientations: add the {} columns to the atoms block",
            keys::QUAT.join(", ")
        ))
    })
}

// ===========================================================================
// PMFT — potentials of mean force and torque (freud.pmft)
// ===========================================================================

/// Binned free-energy surface, shared by every PMFT variant. `density`,
/// `rawCounts` and `pmf` are row-major over `shape`; `edges[k]` holds the
/// `shape[k] + 1` bin edges of axis `axes[k]`.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct PmftOut {
    axes: Vec<&'static str>,
    shape: Vec<usize>,
    edges: Vec<Vec<F>>,
    density: Vec<F>,
    raw_counts: Vec<u64>,
    pmf: Vec<F>,
}

#[wasm_bindgen(js_name = WasmPMFTR12)]
pub struct WasmPMFTR12 {
    inner: molrs::compute::PMFTR12,
}

#[wasm_bindgen(js_class = WasmPMFTR12)]
impl WasmPMFTR12 {
    /// Radial range `r_max` (A); `n_r × n_t1 × n_t2` bins over `(r, θ₁, θ₂)`.
    #[wasm_bindgen(constructor)]
    pub fn new(r_max: F, n_r: usize, n_t1: usize, n_t2: usize) -> Result<Self, JsValue> {
        Ok(Self {
            inner: molrs::compute::PMFTR12::new(r_max, n_r, n_t1, n_t2)
                .map_err(|e| JsValue::from_str(&format!("PMFTR12: {e}")))?,
        })
    }

    /// Requires per-atom orientation angles on the frame.
    pub fn compute(&self, frame: &Frame, neighbors: &Neighbors) -> Result<JsValue, JsValue> {
        frame.with_frame(|rs_frame| {
            let orientations = vec![require_angles(rs_frame, "PMFTR12")?];
            let nlists = std::slice::from_ref(&neighbors.inner);
            let mut out = self
                .inner
                .compute(
                    &[rs_frame],
                    molrs::compute::PMFTR12Args {
                        nlists,
                        orientations: &orientations,
                    },
                )
                .map_err(|e| JsValue::from_str(&format!("PMFTR12 compute: {e}")))?;
            let r = out
                .pop()
                .ok_or_else(|| JsValue::from_str("PMFTR12: empty result"))?;
            js_value(&PmftOut {
                axes: vec!["r", "theta1", "theta2"],
                shape: r.density.shape().to_vec(),
                edges: vec![r.r_edges, r.t1_edges, r.t2_edges],
                density: r.density.iter().copied().collect(),
                raw_counts: r.raw_counts.iter().copied().collect(),
                pmf: r.pmf.iter().copied().collect(),
            })
        })
    }
}

#[wasm_bindgen(js_name = WasmPMFTXY)]
pub struct WasmPMFTXY {
    inner: molrs::compute::PMFTXY,
}

#[wasm_bindgen(js_class = WasmPMFTXY)]
impl WasmPMFTXY {
    /// Body-frame window `±x_max × ±y_max` (A); `n_x × n_y` bins.
    #[wasm_bindgen(constructor)]
    pub fn new(x_max: F, y_max: F, n_x: usize, n_y: usize) -> Result<Self, JsValue> {
        Ok(Self {
            inner: molrs::compute::PMFTXY::new(x_max, y_max, n_x, n_y)
                .map_err(|e| JsValue::from_str(&format!("PMFTXY: {e}")))?,
        })
    }

    /// Orientations are optional: without them every query particle is treated
    /// as unrotated, which is the isotropic reference the freud docs describe.
    pub fn compute(&self, frame: &Frame, neighbors: &Neighbors) -> Result<JsValue, JsValue> {
        frame.with_frame(|rs_frame| {
            let orientations = angles_from_frame(rs_frame).map(|angles| vec![angles]);
            let nlists = std::slice::from_ref(&neighbors.inner);
            let mut out = self
                .inner
                .compute(
                    &[rs_frame],
                    molrs::compute::PMFTXYArgs {
                        nlists,
                        query_orientations: orientations.as_deref(),
                    },
                )
                .map_err(|e| JsValue::from_str(&format!("PMFTXY compute: {e}")))?;
            let r = out
                .pop()
                .ok_or_else(|| JsValue::from_str("PMFTXY: empty result"))?;
            js_value(&PmftOut {
                axes: vec!["x", "y"],
                shape: r.density.shape().to_vec(),
                edges: vec![r.x_edges, r.y_edges],
                density: r.density.iter().copied().collect(),
                raw_counts: r.raw_counts.iter().copied().collect(),
                pmf: r.pmf.iter().copied().collect(),
            })
        })
    }
}

#[wasm_bindgen(js_name = WasmPMFTXYT)]
pub struct WasmPMFTXYT {
    inner: molrs::compute::PMFTXYT,
}

#[wasm_bindgen(js_class = WasmPMFTXYT)]
impl WasmPMFTXYT {
    /// Body-frame window `±x_max × ±y_max` (A); `n_x × n_y × n_t` bins over
    /// `(x, y, θ)` where `θ` is the relative orientation.
    #[wasm_bindgen(constructor)]
    pub fn new(x_max: F, y_max: F, n_x: usize, n_y: usize, n_t: usize) -> Result<Self, JsValue> {
        Ok(Self {
            inner: molrs::compute::PMFTXYT::new(x_max, y_max, n_x, n_y, n_t)
                .map_err(|e| JsValue::from_str(&format!("PMFTXYT: {e}")))?,
        })
    }

    /// Requires per-atom orientation angles on the frame.
    pub fn compute(&self, frame: &Frame, neighbors: &Neighbors) -> Result<JsValue, JsValue> {
        frame.with_frame(|rs_frame| {
            let orientations = vec![require_angles(rs_frame, "PMFTXYT")?];
            let nlists = std::slice::from_ref(&neighbors.inner);
            let mut out = self
                .inner
                .compute(
                    &[rs_frame],
                    molrs::compute::PMFTXYTArgs {
                        nlists,
                        orientations: &orientations,
                    },
                )
                .map_err(|e| JsValue::from_str(&format!("PMFTXYT compute: {e}")))?;
            let r = out
                .pop()
                .ok_or_else(|| JsValue::from_str("PMFTXYT: empty result"))?;
            js_value(&PmftOut {
                axes: vec!["x", "y", "theta"],
                shape: r.density.shape().to_vec(),
                edges: vec![r.x_edges, r.y_edges, r.t_edges],
                density: r.density.iter().copied().collect(),
                raw_counts: r.raw_counts.iter().copied().collect(),
                pmf: r.pmf.iter().copied().collect(),
            })
        })
    }
}

#[wasm_bindgen(js_name = WasmPMFTXYZ)]
pub struct WasmPMFTXYZ {
    inner: molrs::compute::PMFTXYZ,
}

#[wasm_bindgen(js_class = WasmPMFTXYZ)]
impl WasmPMFTXYZ {
    /// Body-frame window `±x_max × ±y_max × ±z_max` (A); `n_x × n_y × n_z` bins.
    #[wasm_bindgen(constructor)]
    pub fn new(
        x_max: F,
        y_max: F,
        z_max: F,
        n_x: usize,
        n_y: usize,
        n_z: usize,
    ) -> Result<Self, JsValue> {
        Ok(Self {
            inner: molrs::compute::PMFTXYZ::new(x_max, y_max, z_max, n_x, n_y, n_z)
                .map_err(|e| JsValue::from_str(&format!("PMFTXYZ: {e}")))?,
        })
    }

    /// Uses the frame's per-atom quaternions when present; otherwise every
    /// query particle is treated as unrotated.
    pub fn compute(&self, frame: &Frame, neighbors: &Neighbors) -> Result<JsValue, JsValue> {
        frame.with_frame(|rs_frame| {
            let orientations = quaternions_from_frame(rs_frame).map(|quats| vec![quats]);
            let nlists = std::slice::from_ref(&neighbors.inner);
            let mut out = self
                .inner
                .compute(
                    &[rs_frame],
                    molrs::compute::PMFTXYZArgs {
                        nlists,
                        query_orientations: orientations.as_deref(),
                    },
                )
                .map_err(|e| JsValue::from_str(&format!("PMFTXYZ compute: {e}")))?;
            let r = out
                .pop()
                .ok_or_else(|| JsValue::from_str("PMFTXYZ: empty result"))?;
            js_value(&PmftOut {
                axes: vec!["x", "y", "z"],
                shape: r.density.shape().to_vec(),
                edges: vec![r.x_edges, r.y_edges, r.z_edges],
                density: r.density.iter().copied().collect(),
                raw_counts: r.raw_counts.iter().copied().collect(),
                pmf: r.pmf.iter().copied().collect(),
            })
        })
    }
}

// ===========================================================================
// CombinedDistribution — N-dimensional joint distribution over observables
// ===========================================================================

#[wasm_bindgen(js_name = WasmCombinedDistribution)]
pub struct WasmCombinedDistribution {
    kinds: Vec<String>,
    axes: Vec<molrs::compute::AxisSpec>,
}

#[wasm_bindgen(js_class = WasmCombinedDistribution)]
impl WasmCombinedDistribution {
    /// `kinds[i]` is `"distance" | "angle" | "dihedral"`; axis `i` bins
    /// observable `i` into `bins[i]` bins over `[mins[i], maxs[i]]`. A non-zero
    /// `sinWeight[i]` marks axis `i` angular so its marginal carries the
    /// sin θ solid-angle correction.
    #[wasm_bindgen(constructor)]
    pub fn new(
        kinds: Vec<String>,
        bins: &[u32],
        mins: &[F],
        maxs: &[F],
        sin_weight: Option<Vec<u8>>,
    ) -> Result<Self, JsValue> {
        let n = kinds.len();
        if bins.len() != n || mins.len() != n || maxs.len() != n {
            return Err(JsValue::from_str(
                "CombinedDistribution: kinds, bins, mins and maxs must have equal length",
            ));
        }
        let sin = sin_weight.unwrap_or_default();
        let axes = (0..n)
            .map(|i| {
                let spec = molrs::compute::AxisSpec::new(bins[i] as usize, mins[i], maxs[i])
                    .map_err(|e| {
                        JsValue::from_str(&format!("CombinedDistribution axis {i}: {e}"))
                    })?;
                Ok(spec.with_sin_weight(sin.get(i).is_some_and(|&v| v != 0)))
            })
            .collect::<Result<Vec<_>, JsValue>>()?;
        Ok(Self { kinds, axes })
    }

    /// `groups` is `number[][]`: one flat atom-index array per observable, each
    /// of length `arity × nGroups` (arity 2/3/4 for distance/angle/dihedral).
    pub fn compute(&self, frame: &Frame, groups: JsValue) -> Result<JsValue, JsValue> {
        use molrs::compute::distribution::{AnyObservable, AtomGroups};

        let raw: Vec<Vec<u32>> = serde_wasm_bindgen::from_value(groups)
            .map_err(|e| JsValue::from_str(&format!("CombinedDistribution groups: {e}")))?;
        if raw.len() != self.kinds.len() {
            return Err(JsValue::from_str(
                "CombinedDistribution: one atom-index group array per observable is required",
            ));
        }

        let mut observables = Vec::with_capacity(self.kinds.len());
        let mut atom_groups = Vec::with_capacity(self.kinds.len());
        for (i, kind) in self.kinds.iter().enumerate() {
            let (obs, arity) = AnyObservable::from_kind(kind).map_err(|e| {
                JsValue::from_str(&format!("CombinedDistribution observable {i}: {e}"))
            })?;
            observables.push(obs);
            atom_groups.push(AtomGroups::new(arity, raw[i].iter().map(|&v| v as u64).collect()).map_err(|e| {
                JsValue::from_str(&format!("CombinedDistribution groups {i}: {e}"))
            })?);
        }
        let calc = molrs::compute::CombinedDistribution::new(observables, self.axes.clone())
            .map_err(|e| JsValue::from_str(&format!("CombinedDistribution: {e}")))?;

        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            shape: Vec<usize>,
            edges: Vec<Vec<F>>,
            centers: Vec<Vec<F>>,
            counts: Vec<F>,
            density: Vec<F>,
            n_binned: F,
            n_raw_samples: usize,
            n_frames: usize,
        }
        frame.with_frame(|rs_frame| {
            let r = calc
                .compute(&[rs_frame], &atom_groups)
                .map_err(|e| JsValue::from_str(&format!("CombinedDistribution compute: {e}")))?;
            js_value(&Out {
                shape: r.centers.iter().map(|c| c.len()).collect(),
                edges: r.edges.iter().map(|e| e.to_vec()).collect(),
                centers: r.centers.iter().map(|c| c.to_vec()).collect(),
                counts: r.counts.to_vec(),
                density: r.density.to_vec(),
                n_binned: r.binned,
                n_raw_samples: r.n_raw_samples,
                n_frames: r.n_frames,
            })
        })
    }
}

// ===========================================================================
// Radical (Laguerre) Voronoi + its domain / void consumers
// ===========================================================================

/// Per-atom covalent radii from the frame's `element` column.
#[cfg(feature = "voronoi")]
fn covalent_radii_from_frame(frame: &molrs::store::frame::Frame) -> Result<Vec<F>, JsValue> {
    let atoms = frame
        .get("atoms")
        .ok_or_else(|| JsValue::from_str("Frame has no 'atoms' block"))?;
    let column = atoms
        .get("element")
        .and_then(|c| c.as_string())
        .ok_or_else(|| {
            JsValue::from_str(
                "radii weighting needs a string 'element' column on the atoms block; \
             construct with useAtomRadii = false for a plain Voronoi diagram",
            )
        })?;
    column
        .iter()
        .map(|symbol| {
            molrs::Element::by_symbol(symbol)
                .map(|el| F::from(el.covalent_radius()))
                .ok_or_else(|| JsValue::from_str(&format!("unknown element symbol {symbol}")))
        })
        .collect()
}

/// Tessellate a frame and report the box volume used for normalization.
///
/// With `use_atom_radii` the cells are Laguerre-weighted by covalent radius;
/// without it every generator has radius zero, which is a plain Voronoi diagram.
#[cfg(feature = "voronoi")]
fn voronoi_cells(
    frame: &molrs::store::frame::Frame,
    use_atom_radii: bool,
) -> Result<(molrs::compute::VoronoiCells, F), JsValue> {
    let positions = positions_from_frame(frame)?;
    let n = positions.nrows();
    let simbox = frame.simbox.as_ref().ok_or_else(|| {
        JsValue::from_str("Radical Voronoi needs a periodic simulation box (frame.simbox is unset)")
    })?;
    let radii = if use_atom_radii {
        covalent_radii_from_frame(frame)?
    } else {
        vec![0.0; n]
    };
    if radii.len() != n {
        return Err(JsValue::from_str(
            "Radical Voronoi: the element column length does not match the atom count",
        ));
    }
    let cells = molrs::compute::RadicalVoronoi
        .build(positions.view(), &radii, simbox)
        .map_err(|e| JsValue::from_str(&format!("RadicalVoronoi: {e}")))?;
    Ok((cells, simbox.volume()))
}

#[cfg(feature = "voronoi")]
#[wasm_bindgen(js_name = WasmRadicalVoronoi)]
pub struct WasmRadicalVoronoi {
    use_atom_radii: bool,
}

#[cfg(feature = "voronoi")]
#[wasm_bindgen(js_class = WasmRadicalVoronoi)]
impl WasmRadicalVoronoi {
    /// With `use_atom_radii` the tessellation is Laguerre-weighted by each
    /// atom's covalent radius, read from the frame's `element` column.
    #[wasm_bindgen(constructor)]
    pub fn new(use_atom_radii: bool) -> Self {
        Self { use_atom_radii }
    }

    /// Cell volumes plus the face graph. `faceNeighbors` / `faceAreas` are the
    /// concatenation of every cell's faces; cell `i` owns the slice
    /// `[faceOffsets[i], faceOffsets[i + 1])`. A negative neighbour id is a box
    /// boundary rather than another cell.
    pub fn compute(&self, frame: &Frame) -> Result<JsValue, JsValue> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            volumes: Vec<F>,
            total_volume: F,
            box_volume: F,
            face_neighbors: Vec<i64>,
            face_areas: Vec<F>,
            face_offsets: Vec<usize>,
        }
        frame.with_frame(|rs_frame| {
            let (cells, box_volume) = voronoi_cells(rs_frame, self.use_atom_radii)?;
            let mut face_neighbors = Vec::new();
            let mut face_areas = Vec::new();
            let mut face_offsets = Vec::with_capacity(cells.len() + 1);
            face_offsets.push(0);
            for faces in &cells.faces {
                for face in faces {
                    face_neighbors.push(face.neighbor);
                    face_areas.push(face.area);
                }
                face_offsets.push(face_neighbors.len());
            }
            js_value(&Out {
                total_volume: cells.total_volume(),
                volumes: cells.volumes.clone(),
                box_volume,
                face_neighbors,
                face_areas,
                face_offsets,
            })
        })
    }
}

#[cfg(feature = "voronoi")]
#[wasm_bindgen(js_name = WasmVoronoiDomainAnalysis)]
pub struct WasmVoronoiDomainAnalysis {
    use_atom_radii: bool,
}

#[cfg(feature = "voronoi")]
#[wasm_bindgen(js_class = WasmVoronoiDomainAnalysis)]
impl WasmVoronoiDomainAnalysis {
    #[wasm_bindgen(constructor)]
    pub fn new(use_atom_radii: bool) -> Self {
        Self { use_atom_radii }
    }

    /// Merge face-adjacent cells that share a `labels` value into domains.
    pub fn compute(&self, frame: &Frame, labels: &[i32]) -> Result<JsValue, JsValue> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            sizes: Vec<usize>,
            count: usize,
            largest_fraction: F,
            domain_of: Vec<usize>,
        }
        frame.with_frame(|rs_frame| {
            let (cells, _) = voronoi_cells(rs_frame, self.use_atom_radii)?;
            if labels.len() != cells.len() {
                return Err(JsValue::from_str(
                    "VoronoiDomainAnalysis: labels must have one entry per atom",
                ));
            }
            let labels: Vec<i64> = labels.iter().map(|&v| i64::from(v)).collect();
            let r = molrs::compute::DomainAnalysis
                .analyze(&cells, &labels)
                .map_err(|e| JsValue::from_str(&format!("VoronoiDomainAnalysis: {e}")))?;
            js_value(&Out {
                sizes: r.sizes,
                count: r.count,
                largest_fraction: r.largest_fraction,
                domain_of: r.domain_of,
            })
        })
    }
}

#[cfg(feature = "voronoi")]
#[wasm_bindgen(js_name = WasmVoronoiVoidAnalysis)]
pub struct WasmVoronoiVoidAnalysis {
    use_atom_radii: bool,
    box_volume: Option<F>,
}

#[cfg(feature = "voronoi")]
#[wasm_bindgen(js_class = WasmVoronoiVoidAnalysis)]
impl WasmVoronoiVoidAnalysis {
    /// `box_volume` overrides the frame's box volume when normalizing the void
    /// fraction; pass `null` to use the frame's own box.
    #[wasm_bindgen(constructor)]
    pub fn new(use_atom_radii: bool, box_volume: Option<F>) -> Self {
        Self {
            use_atom_radii,
            box_volume,
        }
    }

    /// A non-zero `isVoid[i]` marks cell `i` a void probe; adjacent probe cells
    /// merge into one cavity. `boxVolume` defaults to the frame's box volume.
    pub fn compute(&self, frame: &Frame, is_void: &[u8]) -> Result<JsValue, JsValue> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            cavity_volumes: Vec<F>,
            total_void_volume: F,
            void_fraction: F,
        }
        frame.with_frame(|rs_frame| {
            let (cells, frame_volume) = voronoi_cells(rs_frame, self.use_atom_radii)?;
            if is_void.len() != cells.len() {
                return Err(JsValue::from_str(
                    "VoronoiVoidAnalysis: isVoid must have one entry per atom",
                ));
            }
            let mask: Vec<bool> = is_void.iter().map(|&v| v != 0).collect();
            let r = molrs::compute::VoidAnalysis
                .analyze(&cells, &mask, self.box_volume.unwrap_or(frame_volume))
                .map_err(|e| JsValue::from_str(&format!("VoronoiVoidAnalysis: {e}")))?;
            js_value(&Out {
                cavity_volumes: r.cavity_volumes,
                total_void_volume: r.total_void_volume,
                void_fraction: r.void_fraction,
            })
        })
    }
}
