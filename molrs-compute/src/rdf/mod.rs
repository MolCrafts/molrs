//! Radial distribution function g(r) computation.
//!
//! Accumulates pair distances from neighbor lists (one per frame) into a
//! histogram between `r_min` and `r_max`, then normalizes by the ideal-gas
//! shell volume at the system number density during
//! [`ComputeResult::finalize`](crate::ComputeResult::finalize).
//!
//! Each frame contributes its own `SimBox` volume; non-periodic frames error
//! out — the compute never fabricates a bounding box.

mod result;

pub use result::RDFResult;

use molrs::frame_access::FrameAccess;
use molrs::neighbors::NeighborList;
use molrs::types::F;
use ndarray::Array1;

use crate::error::ComputeError;
use crate::traits::Compute;

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
        })
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

    fn accumulate_into(&self, nlist: &NeighborList, n_r: &mut Array1<F>) {
        for &d2 in nlist.dist_sq() {
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
    }
}

impl Compute for RDF {
    type Args<'a> = &'a Vec<NeighborList>;
    type Output = RDFResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        neighbors: &'a Vec<NeighborList>,
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

        let mode = neighbors[0].mode();
        let mut n_r = Array1::<F>::zeros(self.n_bins);
        let mut n_points: usize = 0;
        let mut n_query_points: usize = 0;
        let mut volume: F = 0.0;

        for (i, (frame, nlist)) in frames.iter().zip(neighbors.iter()).enumerate() {
            if nlist.mode() != mode {
                return Err(ComputeError::BadShape {
                    expected: format!("{mode:?} (frame 0)"),
                    got: format!("{:?} (frame {i})", nlist.mode()),
                });
            }
            let simbox = frame.simbox_ref().ok_or(ComputeError::MissingSimBox)?;
            let vol = simbox.volume();
            if !(vol.is_finite() && vol > 0.0) {
                return Err(ComputeError::OutOfRange {
                    field: "RDF::volume",
                    value: vol.to_string(),
                });
            }
            self.accumulate_into(nlist, &mut n_r);
            n_points += nlist.num_points();
            n_query_points += nlist.num_query_points();
            volume += vol;
        }

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
            n_frames: frames.len(),
            finalized: false,
        };
        // Normalize eagerly so direct callers (outside Graph) can read `rdf`
        // without having to call `finalize` themselves. `finalize` is idempotent.
        use crate::result::ComputeResult;
        result.finalize();
        Ok(result)
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

    fn build_nlist(frame: &Frame, r_max: F) -> NeighborList {
        let pos = positions(frame);
        let simbox = frame.simbox.as_ref().unwrap();
        let mut lc = LinkCell::new().cutoff(r_max);
        lc.build(pos.view(), simbox);
        lc.query().clone()
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
        let nlists: Vec<NeighborList> =
            frames_owned.iter().map(|f| build_nlist(f, r_max)).collect();
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
    fn empty_frames_is_error() {
        let rdf = RDF::new(10, 4.0, 0.0).unwrap();
        let frames: Vec<&Frame> = Vec::new();
        let nlists: Vec<NeighborList> = Vec::new();
        let err = rdf.compute(&frames, &nlists).unwrap_err();
        assert!(matches!(err, ComputeError::EmptyInput));
    }

    #[test]
    fn mismatched_nlist_count_is_error() {
        let frame = random_frame(50, 10.0, 1);
        let rdf = RDF::new(10, 4.0, 0.0).unwrap();
        let err = rdf
            .compute(&[&frame], &Vec::<NeighborList>::new())
            .unwrap_err();
        assert!(matches!(err, ComputeError::DimensionMismatch { .. }));
    }

    #[test]
    fn missing_simbox_is_error() {
        let mut frame = random_frame(50, 10.0, 1);
        frame.simbox = None;
        let nlist = {
            use molrs::neighbors::NeighborQuery;
            let pos = positions(&frame);
            NeighborQuery::free(pos.view(), 4.0).query_self()
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
        use molrs::block::Block;
        use molrs::neighbors::NeighborQuery;

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
        let result = rdf.compute(&[&frame], &vec![nlist]).unwrap();

        for (i, &c) in result.n_r.iter().enumerate() {
            assert_eq!(c, 0.0, "bin {i} should be empty, got {c}");
        }
    }

    #[test]
    fn finalize_is_idempotent() {
        use crate::result::ComputeResult;

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
}
