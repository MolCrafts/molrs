//! Radial distribution function g(r) computation.
//!
//! Computes the pair correlation function by binning neighbor-pair distances
//! and normalizing by the ideal-gas shell volume. Supports both periodic and
//! free-boundary (non-periodic) systems.
//!
//! For free-boundary systems (Frame without SimBox), the normalization volume
//! is derived from the axis-aligned bounding box of the atom positions.

mod result;

pub use result::RDFResult;

use molrs::frame_access::FrameAccess;
use molrs::neighbors::NeighborList;
use molrs::region::simbox::SimBox;
use molrs::types::F;
use ndarray::Array1;

use super::accumulator::Accumulator;
use super::error::ComputeError;
use super::reducer::SumReducer;
use super::traits::Compute;

/// Radial distribution function g(r).
///
/// Bins pair distances into a histogram and normalizes by the ideal-gas
/// pair density to produce g(r). Bin edges and centers are precomputed
/// at construction and shared across frames.
///
/// Supports both self-query and cross-query neighbor lists:
/// - **Self-query**: `g(r) = 2 * n_r / (N * rho * V_shell)`
/// - **Cross-query**: `g(r) = n_r * V / (N_A * N_B * V_shell)`
///
/// When the [`Frame`] has no `simbox`, a non-periodic bounding box is
/// auto-generated from atom positions for volume normalization.
#[derive(Debug, Clone)]
pub struct RDF {
    n_bins: usize,
    r_max_sq: F,
    bin_width: F,
    bin_edges: Array1<F>,
    bin_centers: Array1<F>,
}

impl RDF {
    /// Create an RDF analysis with `n_bins` histogram bins up to distance `r_max` (angstrom).
    pub fn new(n_bins: usize, r_max: F) -> Self {
        let bin_width = r_max / n_bins as F;
        let bin_edges = Array1::from_iter((0..=n_bins).map(|i| i as F * bin_width));
        let bin_centers = Array1::from_iter((0..n_bins).map(|i| (i as F + 0.5) * bin_width));
        Self {
            n_bins,
            r_max_sq: r_max * r_max,
            bin_width,
            bin_edges,
            bin_centers,
        }
    }

    /// Bin width in angstrom.
    pub fn bin_width(&self) -> F {
        self.bin_width
    }

    /// Number of histogram bins.
    pub fn n_bins(&self) -> usize {
        self.n_bins
    }

    /// Convenience: wrap this RDF in an `Accumulator<Self, SumReducer<RDFResult>>`.
    pub fn accumulate_sum(self) -> Accumulator<Self, SumReducer<RDFResult>> {
        Accumulator::new(self, SumReducer::new())
    }

    /// Compute g(r) directly from a `NeighborList` and `SimBox`, without
    /// needing a `Frame`. This is the freud-style API.
    pub fn compute_from_nlist(
        &self,
        nlist: &NeighborList,
        simbox: &SimBox,
    ) -> Result<RDFResult, ComputeError> {
        let volume = simbox.volume();

        // Histogram pair distances
        let mut n_r = Array1::<F>::zeros(self.n_bins);
        let dist_sq = nlist.dist_sq();

        for &d2 in dist_sq {
            if d2 < self.r_max_sq {
                let d = d2.sqrt();
                let bin = (d / self.bin_width) as usize;
                if bin < self.n_bins {
                    n_r[bin] += 1.0;
                }
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
        };
        result.rdf = result.normalize(1);

        Ok(result)
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
        let simbox = match frame.simbox_ref() {
            Some(sb) => std::borrow::Cow::Borrowed(sb),
            None => {
                // Free-boundary: auto-generate bounding box from positions
                let xs = frame
                    .get_float("atoms", "x")
                    .ok_or(ComputeError::MissingColumn {
                        block: "atoms",
                        col: "x",
                    })?;
                let ys = frame
                    .get_float("atoms", "y")
                    .ok_or(ComputeError::MissingColumn {
                        block: "atoms",
                        col: "y",
                    })?;
                let zs = frame
                    .get_float("atoms", "z")
                    .ok_or(ComputeError::MissingColumn {
                        block: "atoms",
                        col: "z",
                    })?;
                let n = xs.len();
                let mut pos = ndarray::Array2::<F>::zeros((n, 3));
                for i in 0..n {
                    pos[[i, 0]] = xs[i];
                    pos[[i, 1]] = ys[i];
                    pos[[i, 2]] = zs[i];
                }
                let r_max = self.r_max_sq.sqrt();
                let sb = molrs::region::simbox::SimBox::free(pos.view(), r_max).map_err(|e| {
                    ComputeError::MolRs(molrs::error::MolRsError::validation(format!(
                        "failed to create free-boundary box: {e:?}"
                    )))
                })?;
                std::borrow::Cow::Owned(sb)
            }
        };
        self.compute_from_nlist(neighbors, &simbox)
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

        let rdf = RDF::new(n_bins, r_max);
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
        let single = RDF::new(n_bins, r_max)
            .compute(&frame0, lc.query())
            .unwrap();

        let rdf = RDF::new(n_bins, r_max);
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
    fn compute_from_nlist_works() {
        use molrs::neighbors::NeighborQuery;

        let box_len: F = 10.0;
        let simbox = SimBox::cube(box_len, array![0.0 as F, 0.0, 0.0], [true, true, true]).unwrap();

        let frame = random_frame(100, box_len, 42);
        let pos = positions(&frame);

        let nq = NeighborQuery::new(&simbox, pos.view(), 4.0);
        let nlist = nq.query_self();

        let rdf = RDF::new(20, 4.0);
        let result = rdf.compute_from_nlist(&nlist, &simbox).unwrap();

        assert_eq!(result.bin_centers.len(), 20);
        assert!(result.rdf.iter().any(|&g| g > 0.0));
    }

    #[test]
    fn free_boundary_rdf_computes() {
        use molrs::neighbors::NeighborQuery;

        let n = 200;
        let box_len: F = 10.0;
        let r_max: F = 4.0;
        let n_bins = 20;

        // Create frame WITHOUT simbox
        let mut frame = random_frame(n, box_len, 42);
        frame.simbox = None; // remove the simbox

        let pos = positions(&frame);

        // Build neighbor list with free-boundary box
        let nq = NeighborQuery::free(pos.view(), r_max);
        let nbrs = nq.query_self();

        // RDF should still compute (auto-generates bounding box for volume)
        let rdf = RDF::new(n_bins, r_max);
        let result = rdf.compute(&frame, &nbrs).unwrap();

        assert_eq!(result.bin_centers.len(), n_bins);
        assert!(result.volume > 0.0, "volume should be positive");
        assert!(
            result.rdf.iter().any(|&g| g > 0.0),
            "should have non-zero g(r)"
        );
    }

    #[test]
    fn free_boundary_rdf_from_nlist() {
        use molrs::neighbors::NeighborQuery;

        // Small cluster: 5 points, some within cutoff, some not
        let pos = ndarray::array![
            [0.0 as F, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [10.0, 10.0, 10.0],
            [10.5, 10.0, 10.0],
        ];

        let r_max: F = 2.0;
        let nq = NeighborQuery::free(pos.view(), r_max);
        let nbrs = nq.query_self();

        // Expected pairs within r_max=2.0:
        // (0,1) dist=1.0, (0,2) dist=1.0, (1,2) dist=sqrt(2)~1.41, (3,4) dist=0.5
        assert_eq!(nbrs.n_pairs(), 4, "should find 4 pairs");

        // Build frame without simbox for PairCompute path
        let mut block = Block::new();
        block
            .insert(
                "x",
                A1::from_vec(vec![0.0 as F, 1.0, 0.0, 10.0, 10.5]).into_dyn(),
            )
            .unwrap();
        block
            .insert(
                "y",
                A1::from_vec(vec![0.0 as F, 0.0, 1.0, 10.0, 10.0]).into_dyn(),
            )
            .unwrap();
        block
            .insert(
                "z",
                A1::from_vec(vec![0.0 as F, 0.0, 0.0, 10.0, 10.0]).into_dyn(),
            )
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", block);
        // No simbox

        let rdf = RDF::new(10, r_max);
        let result = rdf.compute(&frame, &nbrs).unwrap();

        assert!(result.volume > 0.0, "volume should be positive");
        assert!(
            result.n_r.iter().any(|&c| c > 0.0),
            "should have non-zero pair counts"
        );
    }
}
