//! Radial distribution function g(r) computation.
//!
//! Accumulates pair distances from neighbor lists (one per frame) into a
//! histogram between `r_min` and `r_max`, then normalizes by the ideal-gas
//! shell volume at the system number density during
//! [`ComputeResult::finalize`](crate::compute::ComputeResult::finalize).
//!
//! Each frame contributes its own `SimBox` volume; non-periodic frames error
//! out — the compute never fabricates a bounding box.

mod accumulator;
mod result;

pub use accumulator::RDFAccumulator;
pub use result::RDFResult;

use molrs::spatial::neighbors::{Backend, LinkCell, Neighbors, QueryMode};
use molrs::spatial::simbox::SimBox;
use molrs::store::frame_access::FrameAccess;
use molrs::types::{F, FNx3View};
use ndarray::Array1;

use crate::compute::error::ComputeError;
use crate::compute::traits::Compute;
use crate::compute::util::get_positions_ref;

/// Radial distribution function g(r) calculator.
///
/// Stateless parameter container: bin count, radial cutoffs, and precomputed
/// bin edges/centers. Actual histograms are built inside each
/// [`compute`](Compute::compute) call.
#[derive(Debug, Clone)]
pub struct RDF {
    n_bins: usize,
    r_min: F,
    r_max: F,
    r_min_sq: F,
    r_max_sq: F,
    bin_width: F,
    bin_edges: Array1<F>,
    bin_centers: Array1<F>,
    dimensionality: u8,
}

impl RDF {
    /// Create an RDF analysis binning pair distances in `[r_min, r_max]`
    /// (angstrom) into `n_bins` bins.
    ///
    /// # Arguments
    ///
    /// **Note the argument order — `r_max` before `r_min`** (matches freud's
    /// convention, kept for cross-check compatibility):
    ///
    /// * `n_bins` — histogram bin count. Must be ≥ 1.
    /// * `r_max` — upper edge of the last bin, Å. Must be > `r_min`.
    /// * `r_min` — lower edge of bin 0, Å. Must be ≥ 0. Often 0.0.
    ///
    /// # References
    ///
    /// Allen & Tildesley, *Computer Simulation of Liquids*, 2nd ed., §2.6.
    pub fn new(n_bins: usize, r_max: F, r_min: F) -> Result<Self, ComputeError> {
        if n_bins == 0 {
            return Err(ComputeError::OutOfRange {
                field: "RDF::n_bins",
                value: n_bins.to_string(),
            });
        }
        if r_min.is_nan() || r_min < 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "RDF::r_min",
                value: r_min.to_string(),
            });
        }
        if r_max.is_nan() || r_max <= r_min {
            return Err(ComputeError::OutOfRange {
                field: "RDF::r_max",
                value: format!("r_max={r_max}, r_min={r_min}"),
            });
        }
        let bin_width = (r_max - r_min) / n_bins as F;
        let bin_edges = Array1::from_iter((0..=n_bins).map(|i| r_min + i as F * bin_width));
        let bin_centers =
            Array1::from_iter((0..n_bins).map(|i| r_min + (i as F + 0.5) * bin_width));
        Ok(Self {
            n_bins,
            r_min,
            r_max,
            r_min_sq: r_min * r_min,
            r_max_sq: r_max * r_max,
            bin_width,
            bin_edges,
            bin_centers,
            dimensionality: 3,
        })
    }

    /// Switch the shell-volume normalization to 2-D (`2π r dr`) instead of
    /// the default 3-D (`(4/3) π (r_o³ − r_i³)`).
    ///
    /// For a 2-D system the `SimBox` should be set up so that `volume()`
    /// returns the in-plane area (e.g. `Lz = 1`). Matches the convention
    /// of `freud.density.RDF` when `freud.box.Box.is2D` is true.
    pub fn with_dimensionality(mut self, dim: u8) -> Self {
        assert!(dim == 2 || dim == 3, "RDF dimensionality must be 2 or 3");
        self.dimensionality = dim;
        self
    }

    pub fn dimensionality(&self) -> u8 {
        self.dimensionality
    }

    pub fn bin_width(&self) -> F {
        self.bin_width
    }
    pub fn n_bins(&self) -> usize {
        self.n_bins
    }
    pub fn r_min(&self) -> F {
        self.r_min
    }
    pub fn r_max(&self) -> F {
        self.r_max
    }

    /// Bin one squared distance into `n_r` (self/cross agnostic).
    #[inline(always)]
    fn accumulate_d2(&self, d2: F, n_r: &mut Array1<F>) {
        if d2 <= 0.0 {
            return;
        }
        if d2 < self.r_min_sq || d2 >= self.r_max_sq {
            return;
        }
        let d = d2.sqrt();
        let bin = ((d - self.r_min) / self.bin_width) as usize;
        if bin < self.n_bins {
            n_r[bin] += 1.0;
        }
    }

    fn accumulate_into(&self, nlist: &Neighbors, n_r: &mut Array1<F>) {
        // Indices-only lists cannot be binned — caller should stream instead.
        let Some(dist_sq) = nlist.dist_sq() else {
            debug_assert!(
                nlist.n_pairs() == 0,
                "RDF::accumulate_into needs dist_sq storage or a streaming path"
            );
            return;
        };
        for &d2 in dist_sq {
            self.accumulate_d2(d2, n_r);
        }
    }

    /// Self-query RDF via cell-list **index + visit** (no `Neighbors` heap).
    ///
    /// This is the production path for large \(N\): memory is \(O(N)\) for the
    /// cell index plus \(O(n_{\mathrm{bins}})\) for the histogram, not
    /// \(O(P)\) for materialized pairs.
    pub fn compute_self(
        &self,
        points: FNx3View<'_>,
        bx: &SimBox,
    ) -> Result<RDFResult, ComputeError> {
        let vol = bx.volume();
        if !(vol.is_finite() && vol > 0.0) {
            return Err(ComputeError::OutOfRange {
                field: "RDF::volume",
                value: vol.to_string(),
            });
        }
        let n = points.nrows();
        if n < 2 {
            return Err(ComputeError::EmptyInput);
        }
        let mut lc = LinkCell::new().cutoff(self.r_max);
        lc.build_index(points, bx);
        let mut n_r = Array1::zeros(self.n_bins);
        lc.visit_pairs(&mut |_, _, d2, _| {
            self.accumulate_d2(d2, &mut n_r);
        });
        self.finalize_histogram(n_r, n, n, QueryMode::SelfQuery { num_points: n }, vol)
    }

    /// Cross-query RDF (group A vs group B) via cell-list index + per-query
    /// neighbor visits — still no full pair materialization.
    ///
    /// `ref_points` become the spatial index (B); `query_points` are A.
    pub fn compute_cross(
        &self,
        ref_points: FNx3View<'_>,
        query_points: FNx3View<'_>,
        bx: &SimBox,
    ) -> Result<RDFResult, ComputeError> {
        let vol = bx.volume();
        if !(vol.is_finite() && vol > 0.0) {
            return Err(ComputeError::OutOfRange {
                field: "RDF::volume",
                value: vol.to_string(),
            });
        }
        let n_ref = ref_points.nrows();
        let n_query = query_points.nrows();
        if n_ref < 1 || n_query < 1 {
            return Err(ComputeError::EmptyInput);
        }
        let mut n_r = Array1::zeros(self.n_bins);
        let cutoff_sq = self.r_max * self.r_max;
        let mut lc = LinkCell::new().cutoff(self.r_max);
        lc.build_index(ref_points, bx);
        for qi in 0..n_query {
            let qp = [
                query_points[[qi, 0]],
                query_points[[qi, 1]],
                query_points[[qi, 2]],
            ];
            lc.visit_neighbors_of_pt(qp, bx, |_, d2, _| {
                if d2 <= cutoff_sq {
                    self.accumulate_d2(d2, &mut n_r);
                }
            });
        }
        self.finalize_histogram(
            n_r,
            n_ref,
            n_query,
            QueryMode::CrossQuery {
                num_query_points: n_query,
                num_points: n_ref,
            },
            vol,
        )
    }

    /// Self-query RDF from a [`FrameAccess`] (reads `atoms.x/y/z` + simbox).
    pub fn compute_frame<FA: FrameAccess + Sync>(
        &self,
        frame: &FA,
    ) -> Result<RDFResult, ComputeError> {
        let bx = frame.simbox_ref().ok_or(ComputeError::MissingSimBox)?;
        let (xs, ys, zs) = get_positions_ref(frame)?;
        let xs = xs.slice();
        let ys = ys.slice();
        let zs = zs.slice();
        let vol = bx.volume();
        if !(vol.is_finite() && vol > 0.0) {
            return Err(ComputeError::OutOfRange {
                field: "RDF::volume",
                value: vol.to_string(),
            });
        }
        let n = xs.len();
        if n < 2 {
            return Err(ComputeError::EmptyInput);
        }
        let mut lc = LinkCell::new().cutoff(self.r_max);
        lc.build_index_soa(xs, ys, zs, bx);
        let mut n_r = Array1::zeros(self.n_bins);
        lc.visit_pairs(&mut |_, _, d2, _| {
            self.accumulate_d2(d2, &mut n_r);
        });
        self.finalize_histogram(n_r, n, n, QueryMode::SelfQuery { num_points: n }, vol)
    }

    fn finalize_histogram(
        &self,
        n_r: Array1<F>,
        n_points: usize,
        n_query_points: usize,
        mode: QueryMode,
        volume: F,
    ) -> Result<RDFResult, ComputeError> {
        let mut result = RDFResult {
            bin_edges: self.bin_edges.clone(),
            bin_centers: self.bin_centers.clone(),
            rdf: Array1::zeros(self.n_bins),
            n_r,
            n_points,
            n_query_points,
            mode,
            volume,
            r_min: self.r_min,
            n_frames: 1,
            dimensionality: self.dimensionality,
            finalized: false,
        };
        use crate::compute::result::ComputeResult;
        result.finalize();
        Ok(result)
    }
}

impl Compute for RDF {
    type Args<'a> = &'a Vec<Neighbors>;
    type Output = RDFResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        neighbors: &'a Vec<Neighbors>,
    ) -> Result<RDFResult, ComputeError> {
        if frames.is_empty() {
            return Err(ComputeError::EmptyInput);
        }
        if neighbors.len() != frames.len() {
            return Err(ComputeError::DimensionMismatch {
                expected: frames.len(),
                got: neighbors.len(),
                what: "neighbor-list count",
            });
        }

        // The batch path is the streaming accumulator driven over the frame
        // slice — one source of truth for the accumulation + normalization.
        let mut acc = RDFAccumulator::new(self.clone());
        for (frame, nlist) in frames.iter().zip(neighbors.iter()) {
            acc.accumulate(*frame, nlist)?;
        }
        // `finalize` normalizes eagerly so direct callers (outside Graph) can
        // read `rdf` without having to call `finalize` themselves.
        acc.finalize()
    }
}

#[cfg(test)]
mod tests {
    use super::super::util::get_positions;
    use super::*;
    use crate::compute::test_support::nlist_from_frame;
    use molrs::Frame;
    use molrs::spatial::simbox::SimBox;
    use molrs::store::block::Block;
    use ndarray::{Array1 as A1, array};
    use rand::RngExt;

    fn random_frame(n: usize, box_len: F, seed: u64) -> Frame {
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let mut rng = StdRng::seed_from_u64(seed);
        let mut block = Block::new();
        let x = A1::from_iter((0..n).map(|_| rng.random::<F>() * box_len));
        let y = A1::from_iter((0..n).map(|_| rng.random::<F>() * box_len));
        let z = A1::from_iter((0..n).map(|_| rng.random::<F>() * box_len));
        block.insert("x", x.into_dyn()).unwrap();
        block.insert("y", y.into_dyn()).unwrap();
        block.insert("z", z.into_dyn()).unwrap();

        let mut frame = Frame::new();
        frame.insert("atoms", block);
        frame.simbox = Some(
            SimBox::cube(
                box_len,
                array![0.0 as F, 0.0 as F, 0.0 as F],
                [true, true, true],
            )
            .unwrap(),
        );
        frame
    }

    fn build_nlist(frame: &Frame, r_max: F) -> Neighbors {
        nlist_from_frame(frame, r_max)
    }

    #[test]
    fn ideal_gas_rdf_approaches_one() {
        let n = 500;
        let box_len: F = 10.0;
        let r_max: F = 4.0;
        let n_bins = 40;

        let frame = random_frame(n, box_len, 42);
        let nlist = build_nlist(&frame, r_max);

        let rdf = RDF::new(n_bins, r_max, 0.0).unwrap();
        let result = rdf.compute(&[&frame], &vec![nlist]).unwrap();

        for i in 5..n_bins {
            assert!(
                (result.rdf[i] - 1.0).abs() < 0.5,
                "g(r={:.2}) = {:.3}, expected ~1.0",
                result.bin_centers[i],
                result.rdf[i]
            );
        }
    }

    #[test]
    fn multi_frame_reduces_variance() {
        // Batch compute over 10 frames; variance of g(r) - 1 should be smaller
        // than any single-frame variance at the same density.
        let n = 200;
        let box_len: F = 10.0;
        let r_max: F = 4.0;
        let n_bins = 20;

        let rdf = RDF::new(n_bins, r_max, 0.0).unwrap();

        // Single-frame baseline.
        let frame0 = random_frame(n, box_len, 100);
        let nlist0 = build_nlist(&frame0, r_max);
        let single = rdf.compute(&[&frame0], &vec![nlist0]).unwrap();
        let var_single: F = single
            .rdf
            .iter()
            .skip(3)
            .map(|g| (g - 1.0).powi(2))
            .sum::<F>()
            / (n_bins - 3) as F;

        // Multi-frame batch.
        let frames_owned: Vec<Frame> = (100..110u64).map(|s| random_frame(n, box_len, s)).collect();
        let nlists: Vec<Neighbors> = frames_owned.iter().map(|f| build_nlist(f, r_max)).collect();
        let frame_refs: Vec<&Frame> = frames_owned.iter().collect();
        let multi = rdf.compute(&frame_refs, &nlists).unwrap();
        let var_multi: F = multi
            .rdf
            .iter()
            .skip(3)
            .map(|g| (g - 1.0).powi(2))
            .sum::<F>()
            / (n_bins - 3) as F;

        assert!(
            var_multi < var_single,
            "multi-frame variance ({var_multi:.6}) should be less than single-frame ({var_single:.6})"
        );
    }

    #[test]
    fn streaming_accumulator_matches_batch_bitwise() {
        // The batch compute is implemented on the accumulator, but assert the
        // public streaming contract explicitly: frame-by-frame accumulate +
        // finalize == one-shot compute, bit-for-bit, including under per-frame
        // box changes (NPT-style).
        let n = 200;
        let r_max: F = 4.0;
        let n_bins = 20;

        let frames_owned: Vec<Frame> = (0..8u64)
            .map(|s| random_frame(n, 10.0 + 0.05 * s as F, 200 + s))
            .collect();
        let nlists: Vec<Neighbors> = frames_owned.iter().map(|f| build_nlist(f, r_max)).collect();

        let rdf = RDF::new(n_bins, r_max, 0.0).unwrap();

        let frame_refs: Vec<&Frame> = frames_owned.iter().collect();
        let batch = rdf.compute(&frame_refs, &nlists).unwrap();

        let mut acc = RDFAccumulator::new(rdf);
        for (f, nl) in frames_owned.iter().zip(nlists.iter()) {
            acc.accumulate(f, nl).unwrap();
        }
        assert_eq!(acc.n_frames(), frames_owned.len());
        let streamed = acc.finalize().unwrap();

        assert_eq!(streamed.n_r, batch.n_r);
        assert_eq!(streamed.rdf, batch.rdf);
        assert_eq!(streamed.n_points, batch.n_points);
        assert_eq!(streamed.volume, batch.volume);
        assert_eq!(streamed.n_frames, batch.n_frames);
    }

    #[test]
    fn accumulator_finalize_before_any_frame_is_error() {
        let rdf = RDF::new(10, 4.0, 0.0).unwrap();
        let acc = RDFAccumulator::new(rdf);
        assert!(matches!(
            acc.finalize().unwrap_err(),
            ComputeError::EmptyInput
        ));
    }

    #[test]
    fn empty_frames_is_error() {
        let rdf = RDF::new(10, 4.0, 0.0).unwrap();
        let frames: Vec<&Frame> = Vec::new();
        let nlists: Vec<Neighbors> = Vec::new();
        let err = rdf.compute(&frames, &nlists).unwrap_err();
        assert!(matches!(err, ComputeError::EmptyInput));
    }

    #[test]
    fn mismatched_nlist_count_is_error() {
        let frame = random_frame(50, 10.0, 1);
        let rdf = RDF::new(10, 4.0, 0.0).unwrap();
        let err = rdf
            .compute(&[&frame], &Vec::<Neighbors>::new())
            .unwrap_err();
        assert!(matches!(err, ComputeError::DimensionMismatch { .. }));
    }

    #[test]
    fn missing_simbox_is_error() {
        let mut frame = random_frame(50, 10.0, 1);
        frame.simbox = None;
        let nlist = {
            use molrs::spatial::neighbors::NeighborQuery;
            let (xs, ys, zs) = get_positions(&frame).unwrap();
            NeighborQuery::free_columns(xs, ys, zs, 4.0).query_self()
        };
        let rdf = RDF::new(10, 4.0, 0.0).unwrap();
        let err = rdf.compute(&[&frame], &vec![nlist]).unwrap_err();
        assert!(matches!(err, ComputeError::MissingSimBox));
    }

    #[test]
    fn r_min_shifts_bins_and_filters_pairs() {
        let box_len: F = 10.0;
        let frame = random_frame(200, box_len, 99);
        let nlist = build_nlist(&frame, 4.0);

        let r_min: F = 1.5;
        let r_max: F = 4.0;
        let n_bins = 25;
        let rdf = RDF::new(n_bins, r_max, r_min).unwrap();
        let result = rdf.compute(&[&frame], &vec![nlist]).unwrap();

        assert!((result.bin_edges[0] - r_min).abs() < 1e-12);
        assert!((result.bin_edges[n_bins] - r_max).abs() < 1e-12);
        assert!((result.r_min - r_min).abs() < 1e-12);

        let dr = (r_max - r_min) / n_bins as F;
        for i in 0..n_bins {
            let expected = r_min + (i as F + 0.5) * dr;
            assert!((result.bin_centers[i] - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn zero_distance_pairs_are_skipped() {
        use molrs::store::block::Block;

        let mut block = Block::new();
        block
            .insert("x", A1::from_vec(vec![0.0 as F, 0.0]).into_dyn())
            .unwrap();
        block
            .insert("y", A1::from_vec(vec![0.0 as F, 0.0]).into_dyn())
            .unwrap();
        block
            .insert("z", A1::from_vec(vec![0.0 as F, 0.0]).into_dyn())
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", block);
        let simbox = SimBox::cube(10.0, array![0.0 as F, 0.0, 0.0], [true, true, true]).unwrap();
        frame.simbox = Some(simbox.clone());

        let nlist = nlist_from_frame(&frame, 2.0);

        let rdf = RDF::new(10, 2.0, 0.0).unwrap();
        let result = rdf.compute(&[&frame], &vec![nlist]).unwrap();

        for (i, &c) in result.n_r.iter().enumerate() {
            assert_eq!(c, 0.0, "bin {i} should be empty, got {c}");
        }
    }

    #[test]
    fn finalize_is_idempotent() {
        use crate::compute::result::ComputeResult;

        let frame = random_frame(200, 10.0, 42);
        let nlist = build_nlist(&frame, 4.0);
        let rdf = RDF::new(20, 4.0, 0.0).unwrap();
        let mut result = rdf.compute(&[&frame], &vec![nlist]).unwrap();
        let first = result.rdf.clone();
        result.finalize();
        result.finalize();
        assert_eq!(result.rdf, first);
    }

    #[test]
    fn new_validates_inputs() {
        assert!(RDF::new(0, 1.0, 0.0).is_err());
        assert!(RDF::new(10, 1.0, -0.1).is_err());
        assert!(RDF::new(10, 1.0, 1.0).is_err());
        assert!(RDF::new(10, 0.5, 1.0).is_err());
        assert!(RDF::new(10, 1.0, 0.0).is_ok());
    }

    /// Sanity check that 2D normalization (`2 π r dr` shells) reproduces the
    /// expected `g(r) → 1` plateau for a random 2-D ideal gas. Mirrors
    /// freud's `freud.density.RDF` behaviour when `box.is2D == True`.
    #[test]
    fn rdf_2d_orthorhombic_box_plateaus_to_one() {
        // Pack 600 points uniformly in a 10×10 plane (z fixed). Use Lz=1
        // so simbox.volume() returns the 2-D area (Lx · Ly).
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let n = 600;
        let lx: F = 10.0;
        let ly: F = 10.0;
        let lz: F = 1.0;
        let r_max: F = 3.5;
        let n_bins = 35;

        let mut rng = StdRng::seed_from_u64(123);
        let mut block = Block::new();
        let xs = A1::from_iter((0..n).map(|_| rng.random::<F>() * lx));
        let ys = A1::from_iter((0..n).map(|_| rng.random::<F>() * ly));
        let zs = A1::from_iter((0..n).map(|_| 0.0_f64));
        block.insert("x", xs.into_dyn()).unwrap();
        block.insert("y", ys.into_dyn()).unwrap();
        block.insert("z", zs.into_dyn()).unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", block);
        let simbox = SimBox::ortho(
            array![lx, ly, lz],
            array![0.0 as F, 0.0, 0.0],
            [true, true, false],
        )
        .unwrap();
        frame.simbox = Some(simbox.clone());

        // Build the neighbor list via the native SoA path (z is identical, so a
        // single-cell column).
        let nlist = nlist_from_frame(&frame, r_max);

        let rdf = RDF::new(n_bins, r_max, 0.0).unwrap().with_dimensionality(2);
        let result = rdf.compute(&[&frame], &vec![nlist]).unwrap();

        // Plateau check: bins outside the first-shell artifact should average
        // to ~1.0. Use a generous tolerance since this is a finite ideal gas.
        let plateau: F = result.rdf.iter().skip(5).copied().sum::<F>() / (n_bins - 5) as F;
        assert!(
            (plateau - 1.0).abs() < 0.15,
            "2-D ideal-gas RDF plateau = {plateau:.3}, expected ≈ 1.0"
        );
        assert_eq!(result.dimensionality, 2);
    }
}
