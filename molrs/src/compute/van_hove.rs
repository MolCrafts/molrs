//! Van Hove correlation function `G(r, t)` — self and distinct parts.
//!
//! The Van Hove function is the space–time generalization of the radial
//! distribution function (L. van Hove, *Phys. Rev.* **1954**, 95, 249; Hansen
//! & McDonald, *Theory of Simple Liquids*, 4th ed., §7):
//!
//! ```text
//!   G_s(r,t) = (1/N) Σ_i      ⟨ δ(r − |r_i(t) − r_i(0)|) ⟩          (self)
//!   G_d(r,t) = (1/N) Σ_i Σ_{j≠i} ⟨ δ(r − |r_j(t) − r_i(0)|) ⟩      (distinct)
//! ```
//!
//! with the structural anchor `G_d(r,0) = ρ g(r)` (ρ = N/V the number density,
//! `g(r)` the RDF) and the dynamical anchor `∫ r² G_s(r,t) dr = MSD(t)`. The
//! two thus bridge [`rdf`](crate::compute::rdf) (structure) and
//! [`msd`](crate::compute::msd) (dynamics).
//!
//! # Provenance
//!
//! TRAVIS (the reference implementation for this `travis-parity` chain) does
//! **not** ship a dedicated Van Hove analyzer — only its RDF and ACF machinery.
//! Accordingly the **definition** follows van Hove 1954 / Hansen-McDonald, the
//! **distinct-part binning + shell normalization** mirror the RDF pair-binning
//! convention molrs already ports (`CDF::AddToBin`, TRAVIS `src/df.cpp`, here
//! reused through [`Histogram1d`](crate::compute::distribution::Histogram1d) and
//! the `4π/3 (r_o³−r_i³)` shell volume of [`rdf`](crate::compute::rdf)), and the
//! **multi-time-origin averaging** mirrors the ACF origin accumulation in
//! TRAVIS `src/reordyn.cpp` / `src/acf.cpp`. Any deviation from a literal
//! TRAVIS port is therefore unavoidable (no source to port) and is documented
//! here.
//!
//! # Conventions
//!
//! - `g_self[t]` is the probability **density of the displacement magnitude**
//!   `|Δr|` at lag `t`: `∫ g_self dr = 1`, so its second moment
//!   `∫ r² g_self dr` is the MSD. (It is the radial-weighted self-part
//!   `4π r² G_s^{3D}`, the directly histogrammable quantity.)
//! - `g_distinct[t]` is the **number density** of other particles at distance
//!   `r` from a reference particle, `ρ g(r)` at `t = 0`.
//! - Self displacements use the raw (unwrapped) coordinate difference — exactly
//!   as [`msd`](crate::compute::msd) does — so the second-moment bridge holds.
//!   Distinct distances use the minimum image (matching `rdf`).

use molrs::spatial::neighbors::NeighborQuery;
use molrs::store::frame_access::FrameAccess;
use molrs::types::F;
use ndarray::{Array1, Array2};

use crate::compute::distribution::Histogram1d;
use crate::compute::error::ComputeError;
use crate::compute::result::ComputeResult;
use crate::compute::traits::Compute;
use crate::compute::util::get_positions_ref;

/// Van Hove correlation analyzer.
///
/// Stateless parameter bag: the r-grid (`n_rbins`, `r_max`), the set of integer
/// frame-lag times to evaluate, and the time-origin `stride`.
#[derive(Debug, Clone)]
pub struct VanHove {
    n_rbins: usize,
    r_max: F,
    lags: Vec<usize>,
    stride: usize,
}

impl VanHove {
    /// New analyzer binning distances in `[0, r_max]` into `n_rbins` bins,
    /// evaluated at the given integer frame `lags` (e.g. `[0, 1, 5, 10]`).
    ///
    /// Lags `≥` the trajectory length are silently dropped at compute time
    /// (you cannot form that displacement); lag `0` is always meaningful.
    pub fn new(n_rbins: usize, r_max: F, lags: Vec<usize>) -> Result<Self, ComputeError> {
        if n_rbins == 0 {
            return Err(ComputeError::OutOfRange {
                field: "VanHove::n_rbins",
                value: n_rbins.to_string(),
            });
        }
        if !(r_max.is_finite() && r_max > 0.0) {
            return Err(ComputeError::OutOfRange {
                field: "VanHove::r_max",
                value: r_max.to_string(),
            });
        }
        if lags.is_empty() {
            return Err(ComputeError::OutOfRange {
                field: "VanHove::lags",
                value: "empty".to_string(),
            });
        }
        Ok(Self {
            n_rbins,
            r_max,
            lags,
            stride: 1,
        })
    }

    /// Set the time-origin stride (default 1 = use every frame as an origin).
    pub fn with_stride(mut self, stride: usize) -> Self {
        self.stride = stride.max(1);
        self
    }

    fn shell_volume(&self, r_inner: F, r_outer: F) -> F {
        (4.0 / 3.0) * std::f64::consts::PI * (r_outer.powi(3) - r_inner.powi(3))
    }
}

impl Compute for VanHove {
    type Args<'a> = ();
    type Output = VanHoveResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        _args: (),
    ) -> Result<VanHoveResult, ComputeError> {
        if frames.is_empty() {
            return Err(ComputeError::EmptyInput);
        }
        let n_frames = frames.len();

        // Gather positions as one `N×3` array per frame (validating constant N).
        // `Array2` so the distinct part can hand views to `NeighborQuery`.
        let mut pos: Vec<Array2<F>> = Vec::with_capacity(n_frames);
        let mut n_particles: Option<usize> = None;
        for frame in frames {
            let (xp, yp, zp) = get_positions_ref(*frame)?;
            let (xs, ys, zs) = (xp.slice(), yp.slice(), zp.slice());
            let m = xs.len();
            match n_particles {
                None => n_particles = Some(m),
                Some(n0) if n0 != m => {
                    return Err(ComputeError::DimensionMismatch {
                        expected: n0,
                        got: m,
                        what: "VanHove particle count",
                    });
                }
                _ => {}
            }
            let mut arr = Array2::<F>::zeros((m, 3));
            for i in 0..m {
                arr[[i, 0]] = xs[i];
                arr[[i, 1]] = ys[i];
                arr[[i, 2]] = zs[i];
            }
            pos.push(arr);
        }
        let n = n_particles.unwrap_or(0);

        // Distinct part needs a box volume for the ρ = N/V normalization.
        let simbox = frames[0].simbox_ref();
        let volume = simbox.map(|b| b.volume());

        let dr = self.r_max / self.n_rbins as F;
        let edges = Array1::from_iter((0..=self.n_rbins).map(|i| i as F * dr));
        let centers = Array1::from_iter((0..self.n_rbins).map(|i| (i as F + 0.5) * dr));

        // Keep only realizable lags (< n_frames).
        let used_lags: Vec<usize> = self
            .lags
            .iter()
            .copied()
            .filter(|&t| t < n_frames)
            .collect();
        let n_lags = used_lags.len();

        let mut g_self = Array2::<F>::zeros((n_lags, self.n_rbins));
        let mut g_distinct = Array2::<F>::zeros((n_lags, self.n_rbins));

        for (li, &lag) in used_lags.iter().enumerate() {
            let mut hist_self = Histogram1d::new(self.n_rbins, 0.0, self.r_max);
            let mut hist_dist = Histogram1d::new(self.n_rbins, 0.0, self.r_max);
            let mut n_origins: usize = 0;

            let mut tau = 0usize;
            while tau + lag < n_frames {
                let a = &pos[tau];
                let b = &pos[tau + lag];
                // Self: raw (unwrapped) displacement magnitude — matches msd.
                for i in 0..n {
                    let dx = b[[i, 0]] - a[[i, 0]];
                    let dy = b[[i, 1]] - a[[i, 1]];
                    let dz = b[[i, 2]] - a[[i, 2]];
                    // Nearest-bin (RDF convention): Van Hove is the RDF's
                    // space-/time-resolved generalization, so its distinct part
                    // G_d(r,0) must equal `compute::rdf`'s nearest-bin g(r). Use
                    // add_nearest, not the geometric DF's TRAVIS cloud-in-cell.
                    hist_self.add_nearest((dx * dx + dy * dy + dz * dz).sqrt());
                }
                // Distinct: min-image distance from r_i(τ) to r_j(τ+lag), j≠i.
                // Use the same `NeighborQuery` spatial search as `compute::rdf`
                // (cutoff = r_max): every pair within r_max is found and binned,
                // every pair beyond it would be dropped by the histogram anyway,
                // so the result is identical to the old all-pairs loop but
                // O(N·neighbours) instead of O(N²). The self pair (i == j, the
                // self part) is excluded.
                if let Some(sb) = simbox {
                    let nlist = NeighborQuery::new(sb, a.view(), self.r_max).query(b.view());
                    let ref_i = nlist.point_indices(); // index into a (= r_i(τ))
                    let oth_j = nlist.query_point_indices(); // index into b (= r_j(τ+lag))
                    let d2 = nlist.dist_sq();
                    for k in 0..nlist.n_pairs() {
                        if ref_i[k] == oth_j[k] {
                            continue;
                        }
                        hist_dist.add_nearest(d2[k].sqrt());
                    }
                }
                n_origins += 1;
                tau += self.stride;
            }

            // g_self: probability density of |Δr| (∫ = 1).
            let self_density = hist_self.density();
            for k in 0..self.n_rbins {
                g_self[[li, k]] = self_density[k];
            }

            // g_distinct: number density of others at r → ρ g(r) at lag 0.
            if let Some(vol) = volume {
                let _ = vol; // ρ = N/V is folded per-bin below via shell volume.
                let counts = hist_dist.counts();
                let denom = (n as F) * (n_origins.max(1) as F);
                for k in 0..self.n_rbins {
                    let shell = self.shell_volume(edges[k], edges[k + 1]);
                    if shell > 0.0 {
                        g_distinct[[li, k]] = counts[k] / (denom * shell);
                    }
                }
            }
        }

        Ok(VanHoveResult {
            r_edges: edges,
            r_centers: centers,
            lags: used_lags,
            g_self,
            g_distinct,
            dr,
            has_distinct: volume.is_some(),
        })
    }
}

/// Result of a [`VanHove`] computation.
///
/// Rows are lag times (`lags`), columns are r-bins (`r_centers`).
#[derive(Debug, Clone)]
pub struct VanHoveResult {
    /// r-bin edges (`n_rbins + 1`), Å.
    pub r_edges: Array1<F>,
    /// r-bin centers (`n_rbins`), Å.
    pub r_centers: Array1<F>,
    /// Realizable lag times actually evaluated (frame units), aligned with rows.
    pub lags: Vec<usize>,
    /// Self part `g_self[t]`: probability density of `|Δr|`, `∫ g_self dr = 1`.
    pub g_self: Array2<F>,
    /// Distinct part `g_distinct[t]`: number density of others at `r`
    /// (`ρ g(r)` at `t = 0`). All zero when the frame had no `SimBox`.
    pub g_distinct: Array2<F>,
    /// Bin width, Å.
    pub dr: F,
    /// Whether the distinct part was computed (requires a `SimBox`).
    pub has_distinct: bool,
}

impl VanHoveResult {
    /// Second moment `∫ r² g_self(r,t) dr` at row `li` — equals the MSD at that
    /// lag (the dynamical bridge to [`msd`](crate::compute::msd)).
    pub fn self_second_moment(&self, li: usize) -> F {
        let mut m = 0.0;
        for k in 0..self.r_centers.len() {
            let r = self.r_centers[k];
            m += r * r * self.g_self[[li, k]] * self.dr;
        }
        m
    }
}

impl ComputeResult for VanHoveResult {}
