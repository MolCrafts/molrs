mod result;

pub use result::RDFResult;

use crate::Frame;
use crate::neighbors::NeighborList;
use crate::region::simbox::SimBox;
use crate::types::F;
use ndarray::Array1;

use super::accumulator::PairAccumulator;
use super::error::ComputeError;
use super::reducer::SumReducer;
use super::traits::PairCompute;

/// Radial distribution function g(r).
///
/// Bins pair distances into a histogram and normalizes by the ideal-gas
/// pair density to produce g(r). Bin edges and centers are precomputed
/// at construction and shared across frames.
///
/// Supports both self-query and cross-query neighbor lists:
/// - **Self-query**: `g(r) = 2 * n_r / (N * rho * V_shell)`
/// - **Cross-query**: `g(r) = n_r * V / (N_A * N_B * V_shell)`
#[derive(Debug, Clone)]
pub struct RDF {
    n_bins: usize,
    r_max_sq: F,
    bin_width: F,
    bin_edges: Array1<F>,
    bin_centers: Array1<F>,
}

impl RDF {
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

    /// Convenience: wrap this RDF in a `PairAccumulator<Self, SumReducer<RDFResult>>`.
    pub fn accumulate_sum(self) -> PairAccumulator<Self, SumReducer<RDFResult>> {
        PairAccumulator::new(self, SumReducer::new())
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

impl PairCompute for RDF {
    type Output = RDFResult;

    fn compute(&self, frame: &Frame, neighbors: &NeighborList) -> Result<RDFResult, ComputeError> {
        let simbox = frame.simbox.as_ref().ok_or(ComputeError::MissingSimBox)?;
        self.compute_from_nlist(neighbors, simbox)
    }
}

#[cfg(test)]
mod tests {
    use super::super::util::get_f_slice;
    use super::*;
    use crate::block::Block;
    use crate::neighbors::{LinkCell, NbListAlgo};
    use crate::region::simbox::SimBox;
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
        use crate::neighbors::AABBQuery;

        let box_len: F = 10.0;
        let simbox = SimBox::cube(box_len, array![0.0 as F, 0.0, 0.0], [true, true, true]).unwrap();

        let frame = random_frame(100, box_len, 42);
        let pos = positions(&frame);

        let nq = AABBQuery::new(&simbox, pos.view(), 4.0);
        let nlist = nq.query_self();

        let rdf = RDF::new(20, 4.0);
        let result = rdf.compute_from_nlist(&nlist, &simbox).unwrap();

        assert_eq!(result.bin_centers.len(), 20);
        assert!(result.rdf.iter().any(|&g| g > 0.0));
    }
}
