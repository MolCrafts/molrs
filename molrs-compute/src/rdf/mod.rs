//! Radial distribution function g(r) computation.
//!
//! Bins pair distances into a histogram between `r_min` and `r_max`, then
//! normalizes by the ideal-gas shell volume at the system number density.
//!
//! Normalization volume comes from the frame's `SimBox` (periodic) or from an
//! explicit caller-supplied value (non-periodic). The compute never
//! fabricates a bounding box to invent a volume — non-periodic frames must go
//! through [`RDF::compute_with_volume`] or supply a `SimBox`.
//!
//! Defaults follow freud (`r_min = 0`, normalize with the system density
//! `N/V`, exclude distances equal to zero to guard degenerate pair data).

mod result;

pub use result::RDFResult;

use molrs::frame_access::FrameAccess;
use molrs::neighbors::NeighborList;
use molrs::types::F;
use ndarray::Array1;

use super::accumulator::Accumulator;
use super::error::ComputeError;
use super::reducer::SumReducer;
use super::traits::Compute;

/// Radial distribution function g(r) calculator.
///
/// Bins neighbor-pair distances in `[r_min, r_max]` and normalizes by the
/// ideal-gas pair density. Pairs at `r = 0` are skipped.
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
}

impl RDF {
    /// Create an RDF analysis binning pair distances in `[r_min, r_max]`
    /// (angstrom) into `n_bins` bins.
    ///
    /// Returns `ComputeError` if `n_bins == 0`, `r_min < 0`, or `r_max <= r_min`.
    pub fn new(n_bins: usize, r_max: F, r_min: F) -> Result<Self, ComputeError> {
        if n_bins == 0 {
            return Err(ComputeError::Invalid("RDF: n_bins must be > 0".into()));
        }
        if !(r_min >= 0.0) {
            return Err(ComputeError::Invalid(format!(
                "RDF: r_min must be >= 0, got {r_min}"
            )));
        }
        if !(r_max > r_min) {
            return Err(ComputeError::Invalid(format!(
                "RDF: r_max must be > r_min, got r_max={r_max}, r_min={r_min}"
            )));
        }
        let bin_width = (r_max - r_min) / n_bins as F;
        let bin_edges =
            Array1::from_iter((0..=n_bins).map(|i| r_min + i as F * bin_width));
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
        })
    }

    /// Bin width in angstrom.
    pub fn bin_width(&self) -> F {
        self.bin_width
    }

    /// Number of histogram bins.
    pub fn n_bins(&self) -> usize {
        self.n_bins
    }

    /// Minimum radial distance (inclusive lower edge of bin 0), angstrom.
    pub fn r_min(&self) -> F {
        self.r_min
    }

    /// Maximum radial distance (upper edge of the last bin), angstrom.
    pub fn r_max(&self) -> F {
        self.r_max
    }

    /// Convenience: wrap this RDF in an `Accumulator<Self, SumReducer<RDFResult>>`.
    pub fn accumulate_sum(self) -> Accumulator<Self, SumReducer<RDFResult>> {
        Accumulator::new(self, SumReducer::new())
    }

    /// Compute g(r) using the neighbor list and an explicit normalization volume (A^3).
    ///
    /// Use this for non-periodic systems or to override the box volume.
    pub fn compute_with_volume(
        &self,
        nlist: &NeighborList,
        volume: F,
    ) -> Result<RDFResult, ComputeError> {
        if !(volume.is_finite() && volume > 0.0) {
            return Err(ComputeError::Invalid(format!(
                "RDF: volume must be a finite positive number, got {volume}"
            )));
        }
        Ok(self.build_result(nlist, volume))
    }

    fn build_result(&self, nlist: &NeighborList, volume: F) -> RDFResult {
        let mut n_r = Array1::<F>::zeros(self.n_bins);
        let dist_sq = nlist.dist_sq();

        for &d2 in dist_sq {
            // Degenerate zero-distance pairs have no physical meaning.
            if d2 <= 0.0 {
                continue;
            }
            if d2 < self.r_min_sq || d2 >= self.r_max_sq {
                continue;
            }
            let d = d2.sqrt();
            let bin = ((d - self.r_min) / self.bin_width) as usize;
            if bin < self.n_bins {
                n_r[bin] += 1.0;
            }
        }

        let n_points = nlist.num_points();
        let n_query_points = nlist.num_query_points();
        let mode = nlist.mode();

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
        };
        result.rdf = result.normalize(1);
        result
    }
}

impl Compute for RDF {
    type Args<'a> = &'a NeighborList;
    type Output = RDFResult;

    fn compute<FA: FrameAccess>(
        &self,
        frame: &FA,
        neighbors: &NeighborList,
    ) -> Result<RDFResult, ComputeError> {
        let simbox = frame.simbox_ref().ok_or(ComputeError::MissingSimBox)?;
        self.compute_with_volume(neighbors, simbox.volume())
    }
}

#[cfg(test)]
mod tests {
    use super::super::util::get_f_slice;
    use super::*;
    use molrs::Frame;
    use molrs::block::Block;
    use molrs::neighbors::{LinkCell, NbListAlgo};
    use molrs::region::simbox::SimBox;
    use ndarray::{Array1 as A1, array};
    use rand::Rng;

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

    fn positions(frame: &Frame) -> ndarray::Array2<F> {
        let atoms = frame.get("atoms").unwrap();
        let xs = get_f_slice(atoms, "atoms", "x").unwrap();
        let ys = get_f_slice(atoms, "atoms", "y").unwrap();
        let zs = get_f_slice(atoms, "atoms", "z").unwrap();
        let n = xs.len();
        let mut pos = ndarray::Array2::<F>::zeros((n, 3));
        for i in 0..n {
            pos[[i, 0]] = xs[i];
            pos[[i, 1]] = ys[i];
            pos[[i, 2]] = zs[i];
        }
        pos
    }

    #[test]
    fn ideal_gas_rdf_approaches_one() {
        let n = 500;
        let box_len: F = 10.0;
        let r_max: F = 4.0;
        let n_bins = 40;

        let frame = random_frame(n, box_len, 42);
        let pos = positions(&frame);
        let simbox = frame.simbox.as_ref().unwrap();

        let mut lc = LinkCell::new().cutoff(r_max);
        lc.build(pos.view(), simbox);
        let nbrs = lc.query();

        let rdf = RDF::new(n_bins, r_max, 0.0).unwrap();
        let result = rdf.compute(&frame, nbrs).unwrap();

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
    fn multi_frame_accumulation_smoother() {
        let n = 200;
        let box_len: F = 10.0;
        let r_max: F = 4.0;
        let n_bins = 20;

        let frame0 = random_frame(n, box_len, 100);
        let pos0 = positions(&frame0);
        let simbox0 = frame0.simbox.as_ref().unwrap();
        let mut lc = LinkCell::new().cutoff(r_max);
        lc.build(pos0.view(), simbox0);
        let single = RDF::new(n_bins, r_max, 0.0)
            .unwrap()
            .compute(&frame0, lc.query())
            .unwrap();

        let rdf = RDF::new(n_bins, r_max, 0.0).unwrap();
        let mut acc = rdf.accumulate_sum();

        for seed in 100..110u64 {
            let frame = random_frame(n, box_len, seed);
            let pos = positions(&frame);
            let sb = frame.simbox.as_ref().unwrap();
            let mut lc2 = LinkCell::new().cutoff(r_max);
            lc2.build(pos.view(), sb);
            acc.feed(&frame, lc2.query()).unwrap();
        }

        let accumulated = acc.result().unwrap();
        let gr_multi = accumulated.normalize(acc.count());

        let var_single: F = single
            .rdf
            .iter()
            .skip(3)
            .map(|g| (g - 1.0).powi(2))
            .sum::<F>()
            / (n_bins - 3) as F;

        let var_multi: F = gr_multi
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
    fn free_boundary_requires_explicit_volume() {
        use molrs::neighbors::NeighborQuery;

        let n = 200;
        let r_max: F = 4.0;
        let n_bins = 20;

        // Frame without simbox.
        let mut frame = random_frame(n, 10.0, 42);
        frame.simbox = None;

        let pos = positions(&frame);
        let nq = NeighborQuery::free(pos.view(), r_max);
        let nbrs = nq.query_self();

        let rdf = RDF::new(n_bins, r_max, 0.0).unwrap();

        // compute() must refuse: no simbox, no volume.
        let err = rdf.compute(&frame, &nbrs).unwrap_err();
        assert!(matches!(err, ComputeError::MissingSimBox));

        // compute_with_volume succeeds and carries the provided volume.
        let v: F = 10.0 * 10.0 * 10.0;
        let result = rdf.compute_with_volume(&nbrs, v).unwrap();
        assert!((result.volume - v).abs() < 1e-9);
        assert_eq!(result.bin_centers.len(), n_bins);
    }

    #[test]
    fn compute_matches_compute_with_volume() {
        // `compute(frame, nlist)` should give the same result as
        // `compute_with_volume(nlist, frame.simbox.volume())`.
        use molrs::neighbors::NeighborQuery;

        let frame = random_frame(500, 10.0, 7);
        let pos = positions(&frame);
        let simbox = frame.simbox.as_ref().unwrap().clone();
        let nlist = NeighborQuery::new(&simbox, pos.view(), 4.0).query_self();

        let rdf = RDF::new(40, 4.0, 0.0).unwrap();
        let g_via_frame = rdf.compute(&frame, &nlist).unwrap();
        let g_via_volume = rdf.compute_with_volume(&nlist, simbox.volume()).unwrap();

        for i in 0..g_via_frame.rdf.len() {
            assert!((g_via_frame.rdf[i] - g_via_volume.rdf[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn r_min_shifts_bins_and_filters_pairs() {
        use molrs::neighbors::NeighborQuery;

        let box_len: F = 10.0;
        let simbox =
            SimBox::cube(box_len, array![0.0 as F, 0.0, 0.0], [true, true, true]).unwrap();
        let frame = random_frame(200, box_len, 99);
        let pos = positions(&frame);
        let nq = NeighborQuery::new(&simbox, pos.view(), 4.0);
        let nlist = nq.query_self();

        let r_min: F = 1.5;
        let r_max: F = 4.0;
        let n_bins = 25;
        let rdf = RDF::new(n_bins, r_max, r_min).unwrap();
        let result = rdf.compute(&frame, &nlist).unwrap();

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
    fn new_validates_inputs() {
        assert!(RDF::new(0, 1.0, 0.0).is_err());
        assert!(RDF::new(10, 1.0, -0.1).is_err());
        assert!(RDF::new(10, 1.0, 1.0).is_err());
        assert!(RDF::new(10, 0.5, 1.0).is_err());
        assert!(RDF::new(10, 1.0, 0.0).is_ok());
    }

    #[test]
    fn zero_distance_pairs_are_skipped() {
        use molrs::block::Block;
        use molrs::neighbors::NeighborQuery;

        // Two atoms at the exact same position: would land in bin 0 otherwise.
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

        let pos = positions(&frame);
        let nq = NeighborQuery::new(&simbox, pos.view(), 2.0);
        let nlist = nq.query_self();

        let rdf = RDF::new(10, 2.0, 0.0).unwrap();
        let result = rdf.compute(&frame, &nlist).unwrap();

        for (i, &c) in result.n_r.iter().enumerate() {
            assert_eq!(c, 0.0, "bin {i} should be empty, got {c}");
        }
    }
}
