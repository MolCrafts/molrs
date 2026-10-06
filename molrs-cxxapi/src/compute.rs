//! Trajectory analysis for the C++ engine — the CXX face of `molrs::compute`.
//!
//! Two shapes per analysis: one-shot computes (`*Compute`) over a whole raw
//! trajectory buffer, and bounded-memory streaming accumulators
//! (`*Accumulator`) fed one frame per call. Each rebuilds transient molrs
//! frames from the caller's flat buffers and delegates the math to molrs; no
//! analysis math lives here. The engine measures; molrs analyzes.

use molrs::compute::RDFAccumulator as RDFAccumulatorCore;
use molrs::op::types::F;
use molrs::spatial::SimBox;
use molrs::store::{Block, Frame};
use ndarray::{Array1, Array2};

/// Build a bare frame carrying only `atoms.{x,y,z}` (+ optional simbox).
///
/// Used by the analysis kernels, which read positions via `FrameAccess`.
fn xyz_frame(
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    box_mat: Option<&[f64]>,
) -> Result<Frame, String> {
    let mut atoms = Block::new();
    atoms
        .insert("x", Array1::from_vec(x).into_dyn())
        .map_err(|e| format!("xyz_frame insert x: {e}"))?;
    atoms
        .insert("y", Array1::from_vec(y).into_dyn())
        .map_err(|e| format!("xyz_frame insert y: {e}"))?;
    atoms
        .insert("z", Array1::from_vec(z).into_dyn())
        .map_err(|e| format!("xyz_frame insert z: {e}"))?;
    let mut frame = Frame::new();
    frame.insert("atoms", atoms);
    if let Some(bm) = box_mat
        && bm.len() >= 9
    {
        let h = Array2::from_shape_vec((3, 3), bm[..9].to_vec())
            .map_err(|e| format!("xyz_frame box reshape: {e}"))?;
        if let Ok(sb) = SimBox::new(h, Array1::zeros(3), [true, true, true]) {
            frame.simbox = Some(sb);
        }
    }
    Ok(frame)
}

/// Mean squared displacement compute — see [`MsdCompute::compute`].
pub struct MsdCompute;

/// Construct an MSD compute (Direct mode: first frame = reference).
pub(crate) fn msd_compute_new() -> Box<MsdCompute> {
    Box::new(MsdCompute)
}

impl MsdCompute {
    /// `MSD(t)` over a raw `[n_frames, n_dof]` position buffer.
    ///
    /// `positions` is row-major with each frame blocked `x[0..na) y[..) z[..)`
    /// (`na = n_dof/3`). `MsdMode::Direct` — `MSD(t) = ⟨|r(t) − r(0)|²⟩_i` with
    /// the first frame as reference (LAMMPS `compute msd`). All math is molrs
    /// (`compute::MSD`). Returns one mean value per frame (index 0 = 0); empty on
    /// < 2 frames or a bad shape — never panics.
    pub fn compute(&self, positions: &[f64], n_frames: i64, n_dof: i64) -> Vec<f64> {
        use molrs::compute::{Compute, MSD};
        if n_frames < 2 || n_dof < 3 || n_dof % 3 != 0 {
            return Vec::new();
        }
        let (nf, nd) = (n_frames as usize, n_dof as usize);
        if positions.len() != nf * nd {
            return Vec::new();
        }
        let na = nd / 3;
        let frames: Vec<Frame> = match (0..nf)
            .map(|t| {
                let b = t * nd;
                xyz_frame(
                    positions[b..b + na].to_vec(),
                    positions[b + na..b + 2 * na].to_vec(),
                    positions[b + 2 * na..b + 3 * na].to_vec(),
                    None,
                )
            })
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(f) => f,
            Err(_) => return Vec::new(),
        };
        let refs: Vec<&Frame> = frames.iter().collect();
        match MSD::new().compute(&refs, ()) {
            Ok(ts) => ts.data.iter().map(|r| r.mean).collect(),
            Err(_) => Vec::new(),
        }
    }
}

/// Einstein self-diffusion compute — see [`DiffusionCompute::compute`].
pub struct DiffusionCompute {
    dt: f64,
    dims: i32,
    fit_lo: f64,
    fit_hi: f64,
}

/// Construct an Einstein-diffusion compute. `dt` = time between frames; `dims` =
/// spatial dimensionality; `[fit_lo, fit_hi]` = the MSD-slope fit window.
pub(crate) fn diffusion_compute_new(
    dt: f64,
    dims: i32,
    fit_lo: f64,
    fit_hi: f64,
) -> Box<DiffusionCompute> {
    Box::new(DiffusionCompute {
        dt,
        dims,
        fit_lo,
        fit_hi,
    })
}

impl DiffusionCompute {
    /// `D = slope / (2*dims)` from the windowed-MSD slope over `[fit_lo, fit_hi]`
    /// (fractions of the last lag), over a raw `[n_frames, n_dof]` position
    /// buffer (same layout as [`MsdCompute::compute`]). All math is molrs
    /// (`EinsteinDiffusion` + `LinearFit`). Returns `NaN` on < 2 frames, a bad
    /// shape/args, or a fit error — never panics.
    pub fn compute(&self, positions: &[f64], n_frames: i64, n_dof: i64) -> f64 {
        use molrs::compute::{Compute, EinsteinDiffusion, EinsteinDiffusionArgs, Fit, LinearFit};
        if n_frames < 2
            || n_dof < 3
            || n_dof % 3 != 0
            || self.dt <= 0.0
            || self.dt.is_nan()
            || self.dims <= 0
        {
            return f64::NAN;
        }
        let (nf, nd) = (n_frames as usize, n_dof as usize);
        if positions.len() != nf * nd {
            return f64::NAN;
        }
        let na = nd / 3;
        let frames: Vec<Frame> = match (0..nf)
            .map(|t| {
                let b = t * nd;
                xyz_frame(
                    positions[b..b + na].to_vec(),
                    positions[b + na..b + 2 * na].to_vec(),
                    positions[b + 2 * na..b + 3 * na].to_vec(),
                    None,
                )
            })
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(f) => f,
            Err(_) => return f64::NAN,
        };
        let refs: Vec<&Frame> = frames.iter().collect();
        let ed = match EinsteinDiffusion.compute(&refs, EinsteinDiffusionArgs { dt: self.dt }) {
            Ok(r) => r,
            Err(_) => return f64::NAN,
        };
        match (LinearFit {
            window: (self.fit_lo, self.fit_hi),
        })
        .fit((&ed.lag_times, &ed.msd))
        {
            Ok(fit) => fit.slope / (2.0 * self.dims as f64),
            Err(_) => f64::NAN,
        }
    }
}

/// Velocity autocorrelation compute — see [`VacfCompute::compute`].
pub struct VacfCompute {
    dt: f64,
    resolution: i64,
}

/// Construct a VACF compute. `dt` = time between frames; `resolution` caps the
/// max lag.
pub(crate) fn vacf_compute_new(dt: f64, resolution: i64) -> Box<VacfCompute> {
    Box::new(VacfCompute { dt, resolution })
}

impl VacfCompute {
    /// DOF-averaged VACF over a row-major `[n_frames, n_dof]` velocity buffer.
    /// All math is molrs (`compute::VACF`, FFT-ACF). Returns the VACF curve (one
    /// value per lag, index 0 = zero lag); empty on fewer than two frames, a
    /// shape/arg error, or a compute error.
    pub fn compute(&self, velocities: &[f64], n_frames: i64, n_dof: i64) -> Vec<f64> {
        use molrs::compute::{Compute, VACF};
        if n_frames < 2 || n_dof < 1 || self.dt <= 0.0 || self.dt.is_nan() || self.resolution < 1 {
            return Vec::new();
        }
        let (nf, nd) = (n_frames as usize, n_dof as usize);
        if velocities.len() != nf * nd {
            return Vec::new();
        }
        let vel = match Array2::from_shape_vec((nf, nd), velocities.to_vec()) {
            Ok(a) => a,
            Err(_) => return Vec::new(),
        };
        let frames: &[&Frame] = &[]; // VACF ignores the frame slice; velocities are in args
        match VACF.compute(frames, (&vel, self.dt, self.resolution as usize)) {
            Ok(r) => r.acf.to_vec(),
            Err(_) => Vec::new(),
        }
    }
}

/// Radial distribution function compute — see [`RdfCompute::compute`].
pub struct RdfCompute {
    n_bins: i64,
    r_max: f64,
    r_min: f64,
}

/// Construct an RDF compute. `n_bins`, `r_max`, `r_min` (Å) — argument order
/// mirrors molrs `RDF::new(n_bins, r_max, r_min)`.
pub(crate) fn rdf_compute_new(n_bins: i64, r_max: f64, r_min: f64) -> Box<RdfCompute> {
    Box::new(RdfCompute {
        n_bins,
        r_max,
        r_min,
    })
}

impl RdfCompute {
    /// g(r) over raw `[n_frames, 3*n_atoms]` positions (blocked `x|y|z` per
    /// frame) + `[n_frames, 9]` per-frame cell matrices.
    ///
    /// Per frame: rebuild a transient `Frame` (x/y/z + `SimBox` from the 9 cell
    /// values, periodic) and a `LinkCell` self-neighbor list (cutoff `r_max`),
    /// then `compute::RDF` batch-accumulates pair distances into `n_bins` bins
    /// over `[r_min, r_max]` (Å), normalized by the ideal-gas shell volume at
    /// each frame's number density. The per-frame `boxes` support NPT. The
    /// caller derives bin-center radii. Returns an empty vector on bad
    /// args/shape, a missing `SimBox`, or a compute error — never panics.
    pub fn compute(
        &self,
        positions: &[f64],
        boxes: &[f64],
        n_frames: i64,
        n_atoms: i64,
    ) -> Vec<f64> {
        use molrs::compute::RDF;
        if n_frames < 1
            || n_atoms < 1
            || self.n_bins < 1
            || self.r_max <= 0.0
            || self.r_max.is_nan()
            || self.r_min < 0.0
            || self.r_min >= self.r_max
        {
            return Vec::new();
        }
        let (nf, na) = (n_frames as usize, n_atoms as usize);
        let nd = na * 3;
        if positions.len() != nf * nd || boxes.len() != nf * 9 {
            return Vec::new();
        }
        // Stream frame-by-frame through the core accumulator (one transient
        // frame + LinkCell at a time — the batch buffer is never duplicated).
        let rdf = match RDF::new(self.n_bins as usize, self.r_max, self.r_min) {
            Ok(r) => r,
            Err(_) => return Vec::new(),
        };
        let mut acc = RDFAccumulatorCore::new(rdf);
        for t in 0..nf {
            let b = t * nd;
            let Some((frame, nlist)) =
                frame_and_self_nlist(&positions[b..b + nd], &boxes[t * 9..t * 9 + 9], self.r_max)
            else {
                return Vec::new();
            };
            if acc.accumulate(&frame, &nlist).is_err() {
                return Vec::new();
            }
        }
        match acc.finalize() {
            Ok(res) => res.rdf.to_vec(),
            Err(_) => Vec::new(),
        }
    }
}

/// Build one transient molrs frame + cell-list self-neighbor list from a flat
/// blocked `x|y|z` position frame and a row-major 3×3 cell.
///
/// Shared by the batch [`RdfCompute`] and the streaming [`RdfAccumulator`] —
/// one marshaling path for both. `None` on an empty / non-multiple-of-3
/// position slice, a malformed cell, or a rejected `SimBox`.
fn frame_and_self_nlist(
    positions: &[f64],
    box9: &[f64],
    r_max: f64,
) -> Option<(Frame, molrs::spatial::neighbors::Neighbors)> {
    use molrs::spatial::neighbors::{NeighborList, NeighborsStorage};
    if positions.is_empty() || !positions.len().is_multiple_of(3) || box9.len() != 9 {
        return None;
    }
    let na = positions.len() / 3;
    let frame = xyz_frame(
        positions[0..na].to_vec(),
        positions[na..2 * na].to_vec(),
        positions[2 * na..3 * na].to_vec(),
        Some(box9),
    )
    .ok()?;
    let mut pos = Array2::<F>::zeros((na, 3));
    for a in 0..na {
        pos[[a, 0]] = positions[a];
        pos[[a, 1]] = positions[na + a];
        pos[[a, 2]] = positions[2 * na + a];
    }
    let nlist = {
        let simbox = frame.simbox.as_ref()?;
        let mut nl = NeighborList::new(r_max);
        nl.build(pos.view(), simbox);
        nl.neighbors(NeighborsStorage::FULL)
    };
    Some((frame, nlist))
}

/// Streaming g(r) accumulator — wraps molrs `compute::RDFAccumulator`
/// (O(n_bins) state). Invalid construction parameters leave it inert: every
/// `accumulate` returns `false` and `finalize` returns empty.
pub struct RdfAccumulator {
    r_max: f64,
    inner: Option<RDFAccumulatorCore>,
}

/// Construct a streaming RDF accumulator. `n_bins`, `r_max`, `r_min` (Å) —
/// same argument order as [`rdf_compute_new`].
pub(crate) fn rdf_accumulator_new(n_bins: i64, r_max: f64, r_min: f64) -> Box<RdfAccumulator> {
    use molrs::compute::RDF;
    let inner = if n_bins >= 1 {
        RDF::new(n_bins as usize, r_max, r_min)
            .ok()
            .map(RDFAccumulatorCore::new)
    } else {
        None
    };
    Box::new(RdfAccumulator { r_max, inner })
}

impl RdfAccumulator {
    /// Fold one frame (blocked `x|y|z` positions + row-major 3×3 cell) into
    /// the running histogram. `false` when the frame is rejected (bad shape,
    /// missing/degenerate box) — state is unchanged in that case.
    pub fn accumulate(&mut self, positions: &[f64], box9: &[f64]) -> bool {
        let Some(acc) = self.inner.as_mut() else {
            return false;
        };
        let Some((frame, nlist)) = frame_and_self_nlist(positions, box9, self.r_max) else {
            return false;
        };
        acc.accumulate(&frame, &nlist).is_ok()
    }

    /// Number of accepted frames.
    pub fn n_frames(&self) -> i64 {
        self.inner.as_ref().map_or(0, |a| a.n_frames() as i64)
    }

    /// Normalized g(r) (one value per bin); empty before any accepted frame.
    pub fn finalize(&self) -> Vec<f64> {
        match self.inner.as_ref().map(|a| a.finalize()) {
            Some(Ok(res)) => res.rdf.to_vec(),
            _ => Vec::new(),
        }
    }
}

/// Streaming MSD accumulator — wraps molrs `compute::MSDAccumulator`
/// (Direct curve + windowed sums capped at `window` lags).
pub struct MsdAccumulator {
    inner: molrs::compute::MSDAccumulator,
}

/// Construct a streaming MSD accumulator resolving windowed lags up to
/// `window` frames (`<= 0` = Direct curve only; `diffusion` then returns NaN).
pub(crate) fn msd_accumulator_new(window: i64) -> Box<MsdAccumulator> {
    Box::new(MsdAccumulator {
        inner: molrs::compute::MSDAccumulator::new(window.max(0) as usize),
    })
}

impl MsdAccumulator {
    /// Fold one frame of blocked `x|y|z` positions (first frame = Direct-mode
    /// reference, latches the DOF count). `false` on a shape/DOF mismatch.
    pub fn accumulate(&mut self, positions: &[f64]) -> bool {
        self.inner.accumulate(positions).is_ok()
    }

    /// Number of accepted frames.
    pub fn n_frames(&self) -> i64 {
        self.inner.n_frames() as i64
    }

    /// Direct-mode MSD(t) curve (one value per accepted frame, index 0 = 0).
    pub fn direct_curve(&self) -> Vec<f64> {
        self.inner.direct_curve().to_vec()
    }

    /// Einstein D = LinearFit slope / (2·dims) over the windowed-MSD curve
    /// within `[fit_lo, fit_hi]` fractions of the max resolved lag. NaN on bad
    /// args, fewer than two frames, `window = 0`, or a fit error.
    pub fn diffusion(&self, dt: f64, dims: i32, fit_lo: f64, fit_hi: f64) -> f64 {
        use molrs::compute::{Fit, LinearFit};
        if dt <= 0.0 || dt.is_nan() || dims <= 0 {
            return f64::NAN;
        }
        let msd = self.inner.windowed_msd();
        if msd.len() < 2 {
            return f64::NAN;
        }
        let lag_times: Array1<f64> = (0..msd.len()).map(|i| i as f64 * dt).collect();
        let msd = Array1::from_vec(msd);
        match (LinearFit {
            window: (fit_lo, fit_hi),
        })
        .fit((&lag_times, &msd))
        {
            Ok(fit) => fit.slope / (2.0 * f64::from(dims)),
            Err(_) => f64::NAN,
        }
    }
}

/// Streaming DOF-averaged velocity-ACF accumulator — wraps molrs
/// `compute::VACFAccumulator`. `resolution < 1` leaves it inert.
pub struct VacfAccumulator {
    inner: Option<molrs::compute::VACFAccumulator>,
}

/// Construct a streaming VACF accumulator resolving lags `0..=resolution`.
pub(crate) fn vacf_accumulator_new(resolution: i64) -> Box<VacfAccumulator> {
    let inner = if resolution >= 1 {
        molrs::compute::VACFAccumulator::new(resolution as usize).ok()
    } else {
        None
    };
    Box::new(VacfAccumulator { inner })
}

impl VacfAccumulator {
    /// Fold one flat velocity frame (any fixed DOF count; Atomiverse feeds
    /// blocked `vx|vy|vz`). `false` on a DOF mismatch or empty frame.
    pub fn accumulate(&mut self, velocities: &[f64]) -> bool {
        self.inner
            .as_mut()
            .is_some_and(|a| a.accumulate(velocities).is_ok())
    }

    /// Number of accepted frames.
    pub fn n_frames(&self) -> i64 {
        self.inner.as_ref().map_or(0, |a| a.n_frames() as i64)
    }

    /// Mean-subtracted, DOF-averaged ACF for lags
    /// `0..=min(resolution, n_frames − 1)`; empty with fewer than two frames.
    pub fn finalize(&self) -> Vec<f64> {
        match self.inner.as_ref().map(|a| a.finalize()) {
            Some(Ok(curve)) => curve,
            _ => Vec::new(),
        }
    }
}
