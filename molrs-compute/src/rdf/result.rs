use molrs::neighbors::QueryMode;
use molrs::types::F;
use ndarray::Array1;

use crate::result::{ComputeResult, DescriptorRow};

/// Result of an RDF computation across one or more frames.
///
/// Before [`finalize`](ComputeResult::finalize) is called the `rdf` array is
/// meaningless — only `n_r`, `volume`, `n_points`, and `n_query_points` carry
/// information. `Graph::run` calls `finalize` automatically; direct users of
/// `RDF::compute` must call it themselves before reading `rdf`.
#[derive(Debug, Clone)]
pub struct RDFResult {
    /// Bin edges in angstrom (n_bins + 1).
    pub bin_edges: Array1<F>,
    /// Bin centers in angstrom (n_bins).
    pub bin_centers: Array1<F>,
    /// Normalized g(r), dimensionless. Populated by [`finalize`](ComputeResult::finalize).
    pub rdf: Array1<F>,
    /// Raw pair count per bin (dimensionless), summed across frames.
    pub n_r: Array1<F>,
    /// Number of reference points, summed across frames.
    pub n_points: usize,
    /// Number of query points (cross-query mode), summed across frames.
    pub n_query_points: usize,
    /// Query mode (self-query or cross-query).
    pub mode: QueryMode,
    /// Total normalization volume in A^3, summed across frames.
    pub volume: F,
    /// Inner cutoff (lower edge of bin 0), angstrom.
    pub r_min: F,
    /// Number of frames fed into `n_r` / `volume` / `n_points`.
    pub n_frames: usize,
    /// Whether `finalize` has already been called.
    pub finalized: bool,
}

impl RDFResult {
    fn compute_normalized(&self) -> Array1<F> {
        let nf = self.n_frames.max(1) as F;
        let vol = self.volume / nf;
        let pi: F = std::f64::consts::PI as F;
        let n_bins = self.n_r.len();
        let mut gr = Array1::<F>::zeros(n_bins);

        match self.mode {
            QueryMode::SelfQuery => {
                let n = self.n_points as F / nf;
                if vol <= 0.0 || n <= 0.0 {
                    return gr;
                }
                let rho = n / vol;
                for i in 0..n_bins {
                    let r_inner = self.bin_edges[i];
                    let r_outer = self.bin_edges[i + 1];
                    let v_shell = (4.0 / 3.0) * pi * (r_outer.powi(3) - r_inner.powi(3));
                    let ideal_count = rho * v_shell * n * nf;
                    if ideal_count > 0.0 {
                        gr[i] = 2.0 * self.n_r[i] / ideal_count;
                    }
                }
            }
            QueryMode::CrossQuery => {
                let n_a = self.n_query_points as F / nf;
                let n_b = self.n_points as F / nf;
                if vol <= 0.0 {
                    return gr;
                }
                for i in 0..n_bins {
                    let r_inner = self.bin_edges[i];
                    let r_outer = self.bin_edges[i + 1];
                    let v_shell = (4.0 / 3.0) * pi * (r_outer.powi(3) - r_inner.powi(3));
                    let ideal_count = n_a * n_b * v_shell / vol * nf;
                    if ideal_count > 0.0 {
                        gr[i] = self.n_r[i] / ideal_count;
                    }
                }
            }
        }

        gr
    }
}

impl ComputeResult for RDFResult {
    fn finalize(&mut self) {
        if self.finalized {
            return;
        }
        self.rdf = self.compute_normalized();
        self.finalized = true;
    }
}

impl DescriptorRow for RDFResult {
    fn as_row(&self) -> &[F] {
        self.rdf
            .as_slice()
            .expect("RDFResult::rdf must be contiguous")
    }
}
