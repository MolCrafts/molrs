//! Result type for the radial distribution function [`RDF`](super::RDF):
//! binned pair counts plus the normalized `g(r)` curve (bin edges/centers in
//! Å).
//!
//! `g(r)` is the ratio of the number of neighbor pairs found at separation `r`
//! to the number an ideal gas of the same number density would have there, so
//! it is dimensionless and tends to 1 at large `r`. Turning raw counts into
//! that ratio is what [`finalize`](ComputeResult::finalize) below does, and
//! [`RdfMode`] is the one input to it that depends on *how* the pairs were
//! searched rather than on where the particles are.

use molrs::spatial::neighbors::QueryMode;
use molrs::types::F;
use ndarray::Array1;

use crate::compute::result::{ComputeResult, DescriptorRow};

/// Which point sets a radial-distribution histogram was accumulated over — the
/// pairing alone, with no point counts attached.
///
/// The neighbor search that produced the pairs was one of two shapes, and the
/// two are normalized differently, so `g(r)` cannot be computed without knowing
/// which one it was. That, and nothing else about the search, is what this enum
/// records.
///
/// # Why the counts are not here
///
/// [`QueryMode`] — the neighbor layer's own answer to the same question —
/// carries the point counts of the one table it describes. An [`RDFResult`],
/// though, sums over every frame of a trajectory, so an embedded `QueryMode`
/// would report the counts of whichever single frame it was built from while
/// [`n_points`](RDFResult::n_points) / [`n_query_points`](RDFResult::n_query_points)
/// report the sum over all of them. That is two fields for one quantity,
/// disagreeing for every trajectory longer than one frame, with nothing in the
/// type to say which is authoritative. Normalization only ever read the
/// variant, so only the variant is kept here and the counts live in exactly one
/// place.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RdfMode {
    /// One point set searched against itself.
    ///
    /// Such a table is *half-shell*: each unordered pair `{i, j}` is stored
    /// exactly once, with `i < j`, rather than once as `(i, j)` and again as
    /// `(j, i)`. An ideal-gas reference count is a count of *ordered* pairs, so
    /// the normalization multiplies the histogram by 2 to restore the ordering
    /// that was never stored.
    SelfQuery,
    /// Query points searched against a distinct reference set.
    ///
    /// Directed pairs: every query point reports all of its reference
    /// neighbors, so both orderings of a pair are already present and the
    /// normalization carries no factor 2.
    CrossQuery,
}

/// The single conversion point from the neighbor layer's [`QueryMode`] to the
/// count-free [`RdfMode`] — dropping the point counts is what this `impl` is
/// for, and every caller goes through it so that no second, divergent mapping
/// can exist.
impl From<QueryMode> for RdfMode {
    fn from(mode: QueryMode) -> Self {
        match mode {
            QueryMode::SelfQuery { .. } => Self::SelfQuery,
            QueryMode::CrossQuery { .. } => Self::CrossQuery,
        }
    }
}

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
    /// Pairing the histogram was accumulated under (self-query or cross-query).
    /// Counts live in `n_points` / `n_query_points` only — see [`RdfMode`].
    pub mode: RdfMode,
    /// Total normalization volume in Å³, summed across frames (divided by
    /// `n_frames` before use, giving the mean box volume).
    pub volume: F,
    /// Inner cutoff (lower edge of bin 0), Å.
    pub r_min: F,
    /// Number of frames fed into `n_r` / `volume` / `n_points`.
    pub n_frames: usize,
    /// Dimensionality used for the shell-volume normalization (2 or 3).
    /// Defaults to 3.
    pub dimensionality: u8,
    /// Whether `finalize` has already been called.
    pub finalized: bool,
}

impl RDFResult {
    /// Volume of the spherical (2-D: annular) shell between radii `r_inner` and
    /// `r_outer`, both Å, for the configured dimensionality.
    /// 3-D: `(4/3) π (r_o³ − r_i³)`, Å³. 2-D: `π (r_o² − r_i²)`, Å².
    fn shell_volume(&self, r_inner: F, r_outer: F) -> F {
        let pi: F = std::f64::consts::PI;
        match self.dimensionality {
            2 => pi * (r_outer.powi(2) - r_inner.powi(2)),
            _ => (4.0 / 3.0) * pi * (r_outer.powi(3) - r_inner.powi(3)),
        }
    }

    fn compute_normalized(&self) -> Array1<F> {
        let nf = self.n_frames.max(1) as F;
        let vol = self.volume / nf;
        let n_bins = self.n_r.len();
        let mut gr = Array1::<F>::zeros(n_bins);

        match self.mode {
            RdfMode::SelfQuery => {
                let n = self.n_points as F / nf;
                if vol <= 0.0 || n <= 0.0 {
                    return gr;
                }
                let rho = n / vol;
                for i in 0..n_bins {
                    let r_inner = self.bin_edges[i];
                    let r_outer = self.bin_edges[i + 1];
                    let v_shell = self.shell_volume(r_inner, r_outer);
                    let ideal_count = rho * v_shell * n * nf;
                    if ideal_count > 0.0 {
                        gr[i] = 2.0 * self.n_r[i] / ideal_count;
                    }
                }
            }
            RdfMode::CrossQuery => {
                let n_a = self.n_query_points as F / nf;
                let n_b = self.n_points as F / nf;
                if vol <= 0.0 {
                    return gr;
                }
                for i in 0..n_bins {
                    let r_inner = self.bin_edges[i];
                    let r_outer = self.bin_edges[i + 1];
                    let v_shell = self.shell_volume(r_inner, r_outer);
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
