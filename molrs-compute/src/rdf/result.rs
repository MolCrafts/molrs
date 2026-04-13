use molrs::neighbors::QueryMode;
use molrs::types::F;
use ndarray::Array1;

/// Result of an RDF computation on a single frame.
#[derive(Debug, Clone)]
pub struct RDFResult {
    /// Bin edges in angstrom (n_bins + 1).
    pub bin_edges: Array1<F>,
    /// Bin centers in angstrom (n_bins).
    pub bin_centers: Array1<F>,
    /// Normalized g(r), dimensionless. Valid for single-frame results.
    /// After accumulation via `SumReducer`, call [`normalize`](Self::normalize)
    /// to recompute from accumulated `n_r`.
    pub rdf: Array1<F>,
    /// Raw pair count per bin (dimensionless).
    pub n_r: Array1<F>,
    /// Number of reference points (for normalization). Sum across frames when accumulated.
    pub n_points: usize,
    /// Number of query points (for cross-query normalization). Sum across frames when accumulated.
    pub n_query_points: usize,
    /// Query mode (self-query or cross-query).
    pub mode: QueryMode,
    /// Box volume in A^3 (for normalization). Sum across frames when accumulated.
    pub volume: F,
}

impl RDFResult {
    /// Normalize accumulated pair counts into g(r).
    ///
    /// For **self-query** (same point set, half-shell):
    ///   `g(r) = 2 * n_r / (N * rho * V_shell * n_frames)`
    ///   where V_shell = (4/3) pi (r_outer^3 - r_inner^3)
    ///
    /// For **cross-query** (different point sets, full-shell):
    ///   `g(r) = n_r * V / (N_A * N_B * V_shell * n_frames)`
    ///
    /// When accumulating across frames, `n_points`, `n_query_points`, and
    /// `volume` are sums, so we divide by `n_frames` to recover per-frame averages.
    pub fn normalize(&self, n_frames: usize) -> Array1<F> {
        let nf = n_frames.max(1) as F;
        let vol = self.volume / nf;
        let pi: F = std::f64::consts::PI as F;
        let n_bins = self.n_r.len();
        let mut gr = Array1::<F>::zeros(n_bins);

        match self.mode {
            QueryMode::SelfQuery => {
                let n = self.n_points as F / nf;
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

/// Supports element-wise accumulation via `SumReducer`.
/// Only `n_r`, `volume`, `n_points`, and `n_query_points` are accumulated.
/// `rdf` is zeroed (stale after accumulation — call `normalize(n_frames)`).
/// `bin_edges` / `bin_centers` are retained from the first frame.
impl std::ops::AddAssign for RDFResult {
    fn add_assign(&mut self, rhs: Self) {
        self.n_r += &rhs.n_r;
        self.volume += rhs.volume;
        self.n_points += rhs.n_points;
        self.n_query_points += rhs.n_query_points;
        self.rdf.fill(0.0);
    }
}
