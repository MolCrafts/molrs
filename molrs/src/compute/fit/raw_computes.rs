//! Raw-only [`Compute`] observables that return **only raw curves + scalar
//! metadata** — the separation half of the compute/fit split.
//!
//! Each returns a raw observable **without** any fitted scalar (no sigma, no
//! slope, no integrated D, no windowing). The fit step is then the analyst's
//! explicit choice of [`LinearFit`](super::LinearFit) /
//! [`RunningIntegral`](super::RunningIntegral) / [`Plateau`](super::Plateau) /
//! spectral transform.
//!
//! | struct | raw output | downstream fit |
//! |--------|-----------|----------------|
//! | [`VACF`] | unnormalized velocity ACF | [`PowerSpectrum`](super::PowerSpectrum) (VDOS) / [`RunningIntegral`](super::RunningIntegral) (D) |
//! | [`IRFlux`] | dipole-flux ACF | [`IRSpectrum`](super::IRSpectrum) |
//! | [`RamanTensor`] | polarizability iso/aniso ACFs | [`RamanSpectrum`](super::RamanSpectrum) |
//! | [`EinsteinDiffusion`] | self-MSD curve | [`LinearFit`](super::LinearFit) (D) |
//! | [`GreenKuboDiffusion`] | velocity ACF | same as [`VACF`] |
//! | [`EinsteinConductivity`] | collective charge-dipole MSD | [`LinearFit`](super::LinearFit) (σ) |
//! | [`GreenKuboConductivity`] | current ACF | [`RunningIntegral`](super::RunningIntegral) (σ) |
//! | [`DebyeRelaxation`] | dipole ACF + ⟨M(0)²⟩ + V/T/BC | [`DebyeFit`](super::DebyeFit) (invariants b, c) |
//!
//! Most take their numeric series through the `Args` GAT, so the `frames` slice
//! is unused (pass an empty slice); [`EinsteinDiffusion`] is the one that
//! consumes frames, delegating to [`MSD::windowed`](crate::compute::MSD).

use molrs::store::frame_access::FrameAccess;
use ndarray::{Array1, Array2};
use rustfft::FftPlanner;

use crate::compute::error::ComputeError;
use crate::compute::msd::MSD;
use crate::compute::result::ComputeResult;
use crate::compute::traits::Compute;
use molrs::signal as sig;

// ── VACF ───────────────────────────────────────────────────────────────────

/// Raw unnormalized velocity autocorrelation function.
#[derive(Debug, Clone)]
pub struct VacfResult {
    /// Lag times τ = i·dt, length `resolution + 1`. Units: `[dt]`.
    pub lag_times: Array1<f64>,
    /// Unnormalized velocity ACF `C(τ) = (1/n_dof) Σ_d ⟨δv_d(0)·δv_d(τ)⟩`,
    /// per-DOF mean-subtracted then DOF-averaged — identical to the `acf_sum`
    /// the VDOS [`PowerSpectrum`](super::PowerSpectrum) transform consumes. Units: `[v]²`.
    pub acf: Array1<f64>,
}

impl ComputeResult for VacfResult {}

/// Raw velocity ACF compute (the VDOS/Green–Kubo-diffusion input).
///
/// Lifts the per-DOF mean-subtract + FFT-ACF + DOF-average block from
/// the VDOS path (the part *before* windowing), returning only the raw ACF.
#[derive(Debug, Clone, Copy, Default)]
pub struct VACF;

/// `(velocities, dt, resolution)` argument bundle for [`VACF`] /
/// [`GreenKuboDiffusion`].
pub type VacfArgs<'a> = (&'a Array2<f64>, f64, usize);

fn velocity_acf(
    velocities: &Array2<f64>,
    dt: f64,
    resolution: usize,
) -> Result<VacfResult, ComputeError> {
    let n_frames = velocities.shape()[0];
    let n_dof = velocities.shape()[1];
    if n_frames < 2 {
        return Err(ComputeError::EmptyInput);
    }
    if dt <= 0.0 {
        return Err(ComputeError::OutOfRange {
            field: "dt",
            value: dt.to_string(),
        });
    }
    let max_lag = resolution.min(n_frames - 1);

    let inv_n_frames = 1.0 / n_frames as f64;
    let mut planner = FftPlanner::new();
    let mut acf_sum = Array1::<f64>::zeros(max_lag + 1);
    for d in 0..n_dof {
        let mut col: Array1<f64> = (0..n_frames).map(|t| velocities[[t, d]]).collect();
        let mean: f64 = col.iter().sum::<f64>() * inv_n_frames;
        for v in col.iter_mut() {
            *v -= mean;
        }
        let acf = sig::acf_fft_with_planner(&mut planner, &col, max_lag).map_err(|e| {
            ComputeError::OutOfRange {
                field: "acf_fft",
                value: e.to_string(),
            }
        })?;
        for k in 0..=max_lag {
            acf_sum[k] += acf[k];
        }
    }
    let inv_n_dof = 1.0 / n_dof as f64;
    for k in 0..=max_lag {
        acf_sum[k] *= inv_n_dof;
    }
    let lag_times = Array1::from_iter((0..=max_lag).map(|i| i as f64 * dt));
    Ok(VacfResult {
        lag_times,
        acf: acf_sum,
    })
}

impl Compute for VACF {
    /// `(velocities (n_frames, n_dof), dt, resolution)`. The `frames` slice is
    /// unused.
    type Args<'a> = VacfArgs<'a>;
    type Output = VacfResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        _frames: &[&'a FA],
        args: Self::Args<'a>,
    ) -> Result<Self::Output, ComputeError> {
        let (velocities, dt, resolution) = args;
        velocity_acf(velocities, dt, resolution)
    }
}

// ── IR dipole-flux ACF (IR-spectrum raw input) ───────────────────────────────

/// Raw dipole-flux autocorrelation function — the IR-spectrum raw input.
#[derive(Debug, Clone)]
pub struct IRFluxResult {
    /// Lag times τ = i·dt, length `max_lag + 1`. Units: `[dt]`.
    pub lag_times: Array1<f64>,
    /// Unnormalized dipole-flux ACF `C(τ) = Σ_d ⟨Ṁ_d(0)·Ṁ_d(τ)⟩`, summed over
    /// the 3 Cartesian components — the ACF the
    /// [`IRSpectrum`](super::IRSpectrum) transform consumes. Units: `[Ṁ]²`.
    pub acf: Array1<f64>,
}

impl ComputeResult for IRFluxResult {}

/// Raw dipole-flux-ACF compute (the IR-spectrum input).
///
/// Lifts the central-difference dipole flux + FFT-ACF + component-sum block (the
/// part *before* windowing), returning only the raw ACF. The window + FFT step
/// is then the [`IRSpectrum`](super::IRSpectrum) [`Fit`](crate::compute::traits::Fit).
#[derive(Debug, Clone, Copy, Default)]
pub struct IRFlux;

/// `(dipole_moments, dt, resolution)` argument bundle for [`IRFlux`].
///
/// `dipole_moments` is `(n_frames, 3)`; the central-difference flux loses the
/// first and last frame, so the effective flux length is `n_frames − 2`.
pub type IRFluxArgs<'a> = (&'a Array2<f64>, f64, usize);

impl Compute for IRFlux {
    type Args<'a> = IRFluxArgs<'a>;
    type Output = IRFluxResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        _frames: &[&'a FA],
        args: Self::Args<'a>,
    ) -> Result<Self::Output, ComputeError> {
        let (dipole_moments, dt, resolution) = args;
        let shape = dipole_moments.shape();
        let n_frames = shape[0];
        if shape[1] != 3 {
            return Err(ComputeError::DimensionMismatch {
                expected: 3,
                got: shape[1],
                what: "dipole_moments (expected (n_frames, 3))",
            });
        }
        if n_frames < 3 {
            return Err(ComputeError::EmptyInput);
        }
        if dt <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "dt",
                value: dt.to_string(),
            });
        }

        let flux_len = n_frames - 2;
        let max_lag = resolution.min(flux_len.saturating_sub(1));
        let inv_2dt = 0.5 / dt;

        let mut planner = FftPlanner::new();
        let mut acf_sum = Array1::<f64>::zeros(max_lag + 1);
        for d in 0..3 {
            let flux: Array1<f64> = (1..n_frames - 1)
                .map(|t| (dipole_moments[[t + 1, d]] - dipole_moments[[t - 1, d]]) * inv_2dt)
                .collect();
            let acf = sig::acf_fft_with_planner(&mut planner, &flux, max_lag).map_err(|e| {
                ComputeError::OutOfRange {
                    field: "acf_fft",
                    value: e.to_string(),
                }
            })?;
            for k in 0..=max_lag {
                acf_sum[k] += acf[k];
            }
        }
        let lag_times = Array1::from_iter((0..=max_lag).map(|i| i as f64 * dt));
        Ok(IRFluxResult {
            lag_times,
            acf: acf_sum,
        })
    }
}

// ── Raman polarizability iso/aniso ACFs (Raman-spectrum raw input) ────────────

/// Weight for diagonal anisotropy components in the Raman ACF.
const DIAG_ANISO_WEIGHT: f64 = 0.5;
/// Weight for off-diagonal anisotropy components in the Raman ACF.
const OFFDIAG_ANISO_WEIGHT: f64 = 3.0;
/// Number of anisotropy components (3 diagonal diffs + 3 off-diagonals).
const N_ANISO_COMPS: usize = 6;

/// Raw isotropic + (weighted) anisotropic polarizability-derivative ACFs — the
/// Raman-spectrum raw input.
#[derive(Debug, Clone)]
pub struct RamanTensorResult {
    /// Lag times τ = i·dt, length `max_lag + 1`. Units: `[dt]`.
    pub lag_times: Array1<f64>,
    /// Isotropic (trace) polarizability-derivative ACF — the `acf_iso` the
    /// [`RamanSpectrum`](super::RamanSpectrum) transform consumes.
    pub acf_iso: Array1<f64>,
    /// Weighted anisotropic (deviatoric) ACF — the `acf_aniso` the
    /// [`RamanSpectrum`](super::RamanSpectrum) transform consumes.
    pub acf_aniso: Array1<f64>,
}

impl ComputeResult for RamanTensorResult {}

/// Raw Raman-tensor-ACF compute (the Raman-spectrum input).
///
/// Lifts the central-difference polarizability derivative + iso/aniso
/// decomposition + FFT-ACF block (the part *before* windowing +
/// cross-section/Bose prefactors), returning only the raw iso/aniso ACFs. The
/// window + FFT + prefactor step is then the
/// [`RamanSpectrum`](super::RamanSpectrum) [`Fit`](crate::compute::traits::Fit).
#[derive(Debug, Clone, Copy, Default)]
pub struct RamanTensor;

/// `(polarizabilities, dt, resolution)` argument bundle for [`RamanTensor`].
///
/// `polarizabilities` is `(n_frames, 6)` in Voigt notation
/// `[α_xx, α_yy, α_zz, α_xy, α_xz, α_yz]`; the central-difference derivative
/// loses the first and last frame.
pub type RamanTensorArgs<'a> = (&'a Array2<f64>, f64, usize);

impl Compute for RamanTensor {
    type Args<'a> = RamanTensorArgs<'a>;
    type Output = RamanTensorResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        _frames: &[&'a FA],
        args: Self::Args<'a>,
    ) -> Result<Self::Output, ComputeError> {
        let (polarizabilities, dt, resolution) = args;
        let shape = polarizabilities.shape();
        let n_frames = shape[0];
        if shape[1] != 6 {
            return Err(ComputeError::DimensionMismatch {
                expected: 6,
                got: shape[1],
                what: "polarizabilities (expected (n_frames, 6) Voigt)",
            });
        }
        if n_frames < 3 {
            return Err(ComputeError::EmptyInput);
        }
        if dt <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "dt",
                value: dt.to_string(),
            });
        }

        let flux_len = n_frames - 2;
        let max_lag = resolution.min(flux_len.saturating_sub(1));
        let inv_2dt = 0.5 / dt;

        let mut iso = Vec::with_capacity(flux_len);
        let mut aniso_comps: [Vec<f64>; N_ANISO_COMPS] = [
            Vec::with_capacity(flux_len), // α_xx − α_yy
            Vec::with_capacity(flux_len), // α_yy − α_zz
            Vec::with_capacity(flux_len), // α_zz − α_xx
            Vec::with_capacity(flux_len), // α_xy
            Vec::with_capacity(flux_len), // α_xz
            Vec::with_capacity(flux_len), // α_yz
        ];

        for t in 1..n_frames - 1 {
            let a_prev = polarizabilities.row(t - 1);
            let a_next = polarizabilities.row(t + 1);

            let xx_dot = (a_next[0] - a_prev[0]) * inv_2dt;
            let yy_dot = (a_next[1] - a_prev[1]) * inv_2dt;
            let zz_dot = (a_next[2] - a_prev[2]) * inv_2dt;

            iso.push((xx_dot + yy_dot + zz_dot) / 3.0);
            aniso_comps[0].push(xx_dot - yy_dot);
            aniso_comps[1].push(yy_dot - zz_dot);
            aniso_comps[2].push(zz_dot - xx_dot);
            aniso_comps[3].push((a_next[3] - a_prev[3]) * inv_2dt); // xy
            aniso_comps[4].push((a_next[4] - a_prev[4]) * inv_2dt); // xz
            aniso_comps[5].push((a_next[5] - a_prev[5]) * inv_2dt); // yz
        }

        let mut planner = FftPlanner::new();

        let iso_series = Array1::from_vec(iso);
        let acf_iso =
            sig::acf_fft_with_planner(&mut planner, &iso_series, max_lag).map_err(|e| {
                ComputeError::OutOfRange {
                    field: "acf_fft",
                    value: e.to_string(),
                }
            })?;

        let mut acf_aniso = Array1::<f64>::zeros(max_lag + 1);
        for (c, comp) in aniso_comps.iter_mut().enumerate() {
            let weight = if c < 3 {
                DIAG_ANISO_WEIGHT
            } else {
                OFFDIAG_ANISO_WEIGHT
            };
            let col = Array1::from_vec(std::mem::take(comp));
            let acf = sig::acf_fft_with_planner(&mut planner, &col, max_lag).map_err(|e| {
                ComputeError::OutOfRange {
                    field: "acf_fft",
                    value: e.to_string(),
                }
            })?;
            for k in 0..=max_lag {
                acf_aniso[k] += weight * acf[k];
            }
        }

        let lag_times = Array1::from_iter((0..=max_lag).map(|i| i as f64 * dt));
        Ok(RamanTensorResult {
            lag_times,
            acf_iso,
            acf_aniso,
        })
    }
}

// ── Green–Kubo diffusion (reuses VACF) ───────────────────────────────────────

/// Raw velocity ACF for the Green–Kubo diffusion route — the same raw curve as
/// [`VACF`], named for the diffusion workflow. `D = (1/d)·∫ VACF dt` is then a
/// [`RunningIntegral`](super::RunningIntegral) + scale step.
#[derive(Debug, Clone, Copy, Default)]
pub struct GreenKuboDiffusion;

impl Compute for GreenKuboDiffusion {
    type Args<'a> = VacfArgs<'a>;
    type Output = VacfResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        _frames: &[&'a FA],
        args: Self::Args<'a>,
    ) -> Result<Self::Output, ComputeError> {
        let (velocities, dt, resolution) = args;
        velocity_acf(velocities, dt, resolution)
    }
}

// ── Einstein diffusion (delegates to MSD::windowed) ──────────────────────────

/// Raw self-MSD for the Einstein diffusion route.
#[derive(Debug, Clone)]
pub struct EinsteinDiffusionResult {
    /// Lag times τ = i·dt, length `n_frames`. Units: `[dt]`.
    pub lag_times: Array1<f64>,
    /// System-average MSD per lag, identical to
    /// [`MSD::windowed`](crate::compute::MSD)'s per-lag mean. Units: `[length]²`.
    pub msd: Array1<f64>,
}

impl ComputeResult for EinsteinDiffusionResult {}

/// Raw self-MSD compute. Delegates to
/// [`MSD::windowed`](crate::compute::MSD) — MSD math is **not** re-derived here.
/// `D = slope/(2d)` is then a [`LinearFit`](super::LinearFit) + scale step.
#[derive(Debug, Clone, Copy, Default)]
pub struct EinsteinDiffusion;

/// `EinsteinDiffusion` argument bundle: the frame spacing for the lag axis.
#[derive(Debug, Clone, Copy)]
pub struct EinsteinDiffusionArgs {
    /// Frame spacing for the lag-time axis. Units: time.
    pub dt: f64,
}

impl Compute for EinsteinDiffusion {
    /// Frame spacing only; positions come from `frames`.
    type Args<'a> = EinsteinDiffusionArgs;
    type Output = EinsteinDiffusionResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        args: Self::Args<'a>,
    ) -> Result<Self::Output, ComputeError> {
        if frames.is_empty() {
            return Err(ComputeError::EmptyInput);
        }
        if args.dt <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "dt",
                value: args.dt.to_string(),
            });
        }
        let series = MSD::windowed().compute(frames, ())?;
        let msd = Array1::from_iter(series.data.iter().map(|r| r.mean));
        let lag_times = Array1::from_iter((0..series.data.len()).map(|i| i as f64 * args.dt));
        Ok(EinsteinDiffusionResult { lag_times, msd })
    }
}

// ── Einstein–Helfand conductivity (raw collective-dipole MSD) ────────────────

/// Raw collective charge-dipole MSD — the raw portion of the legacy
/// `ConductivityResult`, with **no** fitted sigma/slope.
#[derive(Debug, Clone)]
pub struct EinsteinConductivityResult {
    /// Lag times τ = i·dt, length `max_lag + 1`. Units: `[dt]`.
    pub lag_times: Array1<f64>,
    /// Collective-dipole MSD ⟨|**M_J**(t+τ) − **M_J**(t)|²⟩ over time origins,
    /// identical to `ConductivityResult.msd`. Units: `(e·Å)²`.
    pub msd: Array1<f64>,
}

impl ComputeResult for EinsteinConductivityResult {}

/// Raw collective charge-dipole MSD compute. Lifts the time-origin MSD loop
/// from the Einstein–Helfand conductivity and stops there (no OLS, no σ). The
/// σ = slope/(6·V·k_B·T) step is a downstream [`LinearFit`](super::LinearFit).
#[derive(Debug, Clone, Copy, Default)]
pub struct EinsteinConductivity;

/// `(translational_dipole, dt, max_correlation_time)` bundle.
pub type EinsteinConductivityArgs<'a> = (&'a Array2<f64>, f64, usize);

impl Compute for EinsteinConductivity {
    type Args<'a> = EinsteinConductivityArgs<'a>;
    type Output = EinsteinConductivityResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        _frames: &[&'a FA],
        args: Self::Args<'a>,
    ) -> Result<Self::Output, ComputeError> {
        let (dipole, dt, max_correlation_time) = args;
        let shape = dipole.shape();
        if shape[1] != 3 {
            return Err(ComputeError::DimensionMismatch {
                expected: 3,
                got: shape[1],
                what: "translational_dipole (expected (n_frames, 3))",
            });
        }
        let n_frames = shape[0];
        if n_frames < 2 {
            return Err(ComputeError::EmptyInput);
        }
        if dt <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "dt",
                value: dt.to_string(),
            });
        }
        let max_lag = max_correlation_time.min(n_frames - 1);

        let mut msd = Array1::<f64>::zeros(max_lag + 1);
        for tau in 1..=max_lag {
            let count = n_frames - tau;
            let mut acc = 0.0;
            for t in 0..count {
                let mut s = 0.0;
                for d in 0..3 {
                    let dx = dipole[[t + tau, d]] - dipole[[t, d]];
                    s += dx * dx;
                }
                acc += s;
            }
            msd[tau] = acc / count as f64;
        }
        let lag_times = Array1::from_iter((0..=max_lag).map(|i| i as f64 * dt));
        Ok(EinsteinConductivityResult { lag_times, msd })
    }
}

// ── Green–Kubo conductivity (raw current ACF) ────────────────────────────────

/// Raw current autocorrelation function — the raw portion of the legacy
/// `JacfResult`, with **no** fitted sigma.
#[derive(Debug, Clone)]
pub struct GreenKuboConductivityResult {
    /// Lag times τ = i·dt, length `max_lag + 1`. Units: `[dt]`.
    pub lag_times: Array1<f64>,
    /// Current ACF `C(τ) = ⟨J(0)·J(τ)⟩` over time origins, identical to
    /// `JacfResult.jacf`. Units: `(e·Å·ps⁻¹)²`.
    pub jacf: Array1<f64>,
}

impl ComputeResult for GreenKuboConductivityResult {}

/// Raw current-ACF compute. Lifts the unbiased windowed-ACF loop from
/// the Green–Kubo conductivity and stops there (no trapezoid, no σ). The
/// σ = (1/(3·V·k_B·T))·∫⟨JJ⟩ step is a downstream
/// [`RunningIntegral`](super::RunningIntegral) + scale.
#[derive(Debug, Clone, Copy, Default)]
pub struct GreenKuboConductivity;

/// `(current, dt, max_correlation_time)` bundle.
pub type GreenKuboConductivityArgs<'a> = (&'a Array2<f64>, f64, usize);

impl Compute for GreenKuboConductivity {
    type Args<'a> = GreenKuboConductivityArgs<'a>;
    type Output = GreenKuboConductivityResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        _frames: &[&'a FA],
        args: Self::Args<'a>,
    ) -> Result<Self::Output, ComputeError> {
        let (current, dt, max_correlation_time) = args;
        let shape = current.shape();
        if shape[1] != 3 {
            return Err(ComputeError::DimensionMismatch {
                expected: 3,
                got: shape[1],
                what: "current (expected (n_frames, 3))",
            });
        }
        let n_frames = shape[0];
        if n_frames < 2 {
            return Err(ComputeError::EmptyInput);
        }
        if dt <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "dt",
                value: dt.to_string(),
            });
        }
        let max_lag = max_correlation_time.min(n_frames - 1);

        let mut jacf = Array1::<f64>::zeros(max_lag + 1);
        for tau in 0..=max_lag {
            let count = n_frames - tau;
            let mut acc = 0.0;
            for t in 0..count {
                let mut s = 0.0;
                for d in 0..3 {
                    s += current[[t, d]] * current[[t + tau, d]];
                }
                acc += s;
            }
            jacf[tau] = acc / count as f64;
        }
        let lag_times = Array1::from_iter((0..=max_lag).map(|i| i as f64 * dt));
        Ok(GreenKuboConductivityResult { lag_times, jacf })
    }
}

// ── Debye relaxation (raw dipole ACF + metadata, invariants b/c) ─────────────

/// Ewald boundary condition under which the dipole fluctuations were sampled.
///
/// The Debye / Neumann–Kirkwood amplitude `ε₀ − ε∞` depends on this boundary
/// condition, so it travels with the raw result (spec invariant (c)).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EwaldBoundary {
    /// Conducting / "tin-foil" boundary (ε' = ∞) — the common MD default.
    TinFoil,
    /// Vacuum boundary (ε' = 1).
    Vacuum,
}

/// Raw dipole ACF plus the scalar metadata the Debye amplitude needs.
///
/// Unlike the spectral/diffusion raw results, this carries an **unnormalized**
/// ACF and the zero-lag variance ⟨M(0)²⟩ explicitly (invariant (b)): the
/// normalized Φ(t) gives only the relaxation *shape*/τ via
/// [`DebyeFit`](super::DebyeFit); the amplitude `ε₀ − ε∞` comes from ⟨M²⟩
/// together with V, T, and the Ewald boundary condition (invariant (c)).
///
/// All four metadata fields are non-optional, so a `DebyeRelaxationResult`
/// **cannot be constructed without them**.
#[derive(Debug, Clone)]
pub struct DebyeRelaxationResult {
    /// Lag times τ = i·dt, length `max_lag + 1`. Units: `[dt]`.
    pub lag_times: Array1<f64>,
    /// **Unnormalized** dipole ACF `C(τ) = ⟨δM(0)·δM(τ)⟩` summed over the 3
    /// Cartesian components. Units: `(e·Å)²`.
    pub acf: Array1<f64>,
    /// Zero-lag variance ⟨M(0)²⟩ = `acf[0]`, the Debye amplitude scale
    /// (invariant b). Units: `(e·Å)²`.
    pub zero_lag_variance: f64,
    /// System volume V (invariant c). Units: `Å³`.
    pub volume: f64,
    /// Temperature T (invariant c). Units: `K`.
    pub temperature: f64,
    /// Ewald boundary condition the fluctuations were sampled under
    /// (invariant c).
    pub boundary: EwaldBoundary,
}

impl ComputeResult for DebyeRelaxationResult {}

/// Raw dipole-ACF compute for the Debye route. Computes the mean-subtracted,
/// per-component-summed dipole ACF (unbiased estimator) and carries the
/// zero-lag variance + V/T/Ewald-BC metadata the amplitude needs.
#[derive(Debug, Clone, Copy)]
pub struct DebyeRelaxation {
    /// System volume V. Units: `Å³`.
    pub volume: f64,
    /// Temperature T. Units: `K`.
    pub temperature: f64,
    /// Ewald boundary condition.
    pub boundary: EwaldBoundary,
}

/// `(dipole_moments, dt, max_correlation_time)` bundle for [`DebyeRelaxation`].
pub type DebyeRelaxationArgs<'a> = (&'a Array2<f64>, f64, usize);

impl Compute for DebyeRelaxation {
    type Args<'a> = DebyeRelaxationArgs<'a>;
    type Output = DebyeRelaxationResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        _frames: &[&'a FA],
        args: Self::Args<'a>,
    ) -> Result<Self::Output, ComputeError> {
        let (dipole, dt, max_correlation_time) = args;
        let shape = dipole.shape();
        if shape[1] != 3 {
            return Err(ComputeError::DimensionMismatch {
                expected: 3,
                got: shape[1],
                what: "dipole_moments (expected (n_frames, 3))",
            });
        }
        let n_frames = shape[0];
        if n_frames < 2 {
            return Err(ComputeError::EmptyInput);
        }
        if dt <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "dt",
                value: dt.to_string(),
            });
        }
        if self.volume <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "volume",
                value: self.volume.to_string(),
            });
        }
        if self.temperature <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "temperature",
                value: self.temperature.to_string(),
            });
        }
        let max_lag = max_correlation_time.min(n_frames - 1);

        // Per-component means → fluctuation ACF so acf[0] = ⟨|δM|²⟩.
        let mut mean = [0.0_f64; 3];
        for t in 0..n_frames {
            for d in 0..3 {
                mean[d] += dipole[[t, d]];
            }
        }
        for m in mean.iter_mut() {
            *m /= n_frames as f64;
        }

        let mut planner = FftPlanner::new();
        let mut acf = Array1::<f64>::zeros(max_lag + 1);
        for d in 0..3 {
            let col: Array1<f64> = (0..n_frames).map(|t| dipole[[t, d]] - mean[d]).collect();
            let component =
                sig::acf_fft_with_planner(&mut planner, &col, max_lag).map_err(|e| {
                    ComputeError::OutOfRange {
                        field: "acf_fft",
                        value: e.to_string(),
                    }
                })?;
            for k in 0..=max_lag {
                acf[k] += component[k];
            }
        }
        // Unbiased linear-ACF estimator C(k) = ⟨δM(0)·δM(k·dt)⟩.
        for k in 0..=max_lag {
            acf[k] /= (n_frames - k) as f64;
        }

        let zero_lag_variance = acf[0];
        let lag_times = Array1::from_iter((0..=max_lag).map(|i| i as f64 * dt));
        Ok(DebyeRelaxationResult {
            lag_times,
            acf,
            zero_lag_variance,
            volume: self.volume,
            temperature: self.temperature,
            boundary: self.boundary,
        })
    }
}

// ── VCD / ROA / resonance-Raman raw cross-correlations (chiral-spectra inputs) ─

/// Linear cross-correlation `C_ab[t] = Σ_τ a[τ]·b[τ+t]` for `t = 0..=max_lag`,
/// via the Wiener–Khinchin cross spectrum (`IFFT(conj(A)·B)`), using the same
/// zero-pad-to-`(2n).next_power_of_two()` and `1/n_pad` scaling as
/// [`acf_fft`](molrs::signal::acf_fft) — so `cross_correlate(a, a, …)` exactly
/// reproduces the autocorrelation. Both inputs must share the same length.
///
/// Ported from the cross-correlation step in `CROAEngine::ComputeACFPair`
/// (`src/roa.cpp`), which feeds each moment-component pair through
/// `m_pCrossCorr->CrossCorrelate(&in1, &in2, &out)`.
fn cross_correlate(
    planner: &mut FftPlanner<f64>,
    a: &Array1<f64>,
    b: &Array1<f64>,
    max_lag: usize,
) -> Array1<f64> {
    use rustfft::num_complex::Complex64;
    let n = a.len();
    let n_pad = (2 * n).next_power_of_two();
    let fwd = planner.plan_fft_forward(n_pad);
    let inv = planner.plan_fft_inverse(n_pad);
    let mut ca: Vec<Complex64> = a.iter().map(|&x| Complex64::new(x, 0.0)).collect();
    let mut cb: Vec<Complex64> = b.iter().map(|&x| Complex64::new(x, 0.0)).collect();
    ca.resize(n_pad, Complex64::new(0.0, 0.0));
    cb.resize(n_pad, Complex64::new(0.0, 0.0));
    fwd.process(&mut ca);
    fwd.process(&mut cb);
    let mut prod: Vec<Complex64> = (0..n_pad).map(|i| ca[i].conj() * cb[i]).collect();
    inv.process(&mut prod);
    let scale = 1.0 / n_pad as f64;
    prod[..=max_lag]
        .iter()
        .map(|c| c.re * scale)
        .collect::<Vec<_>>()
        .into()
}

/// Central-difference time derivative `ẋ[t] = (x[t+1] − x[t−1]) / (2·dt)` of one
/// column `col` of an `(n_frames, n_cols)` series, dropping the first and last
/// frame (length `n_frames − 2`) — the same flux convention as [`IRFlux`].
fn central_diff_col(series: &Array2<f64>, col: usize, dt: f64) -> Array1<f64> {
    let n = series.shape()[0];
    let inv_2dt = 0.5 / dt;
    (1..n - 1)
        .map(|t| (series[[t + 1, col]] - series[[t - 1, col]]) * inv_2dt)
        .collect()
}

/// Raw VCD cross-correlation — the VCD-spectrum raw input.
#[derive(Debug, Clone)]
pub struct VcdCrossResult {
    /// Lag times τ = i·dt, length `max_lag + 1`.
    pub lag_times: Array1<f64>,
    /// VCD cross-correlation `C(τ) = Σ_d ⟨μ̇_d(0)·ṁ_d(τ)⟩` summed over the 3
    /// Cartesian components — the (signed) ACF the
    /// [`VcdSpectrum`](super::VcdSpectrum) transform consumes.
    pub acf: Array1<f64>,
}

impl ComputeResult for VcdCrossResult {}

/// Raw VCD cross-flux compute: cross-correlation of the electric-dipole
/// derivative `μ̇` with the magnetic-dipole derivative `ṁ`.
///
/// Ported from `CROAEngine::ComputeACFPair`, `ROA_SPECTRUM_VCD` branch
/// (`src/roa.cpp`): for each Cartesian component it cross-correlates
/// `m_vaDElDip` (μ̇) with `m_vaDMagDip` (ṁ) and sums the three components.
///
/// **Deviation from the spec's literal `⟨μ̇·m⟩`:** TRAVIS correlates the two
/// *derivatives* (`μ̇` × `ṁ`); the two conventions differ only by a frequency-
/// domain factor and share the same peak positions and enantiomer sign law.
#[derive(Debug, Clone, Copy, Default)]
pub struct VcdCrossFlux;

/// `(electric_dipole (n,3), magnetic_dipole (n,3), dt, resolution)` for
/// [`VcdCrossFlux`]. Both series are `(n_frames, 3)`; the central-difference
/// derivatives drop the first and last frame.
pub type VcdCrossArgs<'a> = (&'a Array2<f64>, &'a Array2<f64>, f64, usize);

impl Compute for VcdCrossFlux {
    type Args<'a> = VcdCrossArgs<'a>;
    type Output = VcdCrossResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        _frames: &[&'a FA],
        args: Self::Args<'a>,
    ) -> Result<Self::Output, ComputeError> {
        let (electric, magnetic, dt, resolution) = args;
        let n_frames = electric.shape()[0];
        if electric.shape()[1] != 3 || magnetic.shape()[1] != 3 {
            return Err(ComputeError::DimensionMismatch {
                expected: 3,
                got: electric.shape()[1].max(magnetic.shape()[1]),
                what: "VCD (electric, magnetic) dipoles (expected (n_frames, 3))",
            });
        }
        if magnetic.shape()[0] != n_frames {
            return Err(ComputeError::DimensionMismatch {
                expected: n_frames,
                got: magnetic.shape()[0],
                what: "VCD electric/magnetic frame counts",
            });
        }
        if n_frames < 3 {
            return Err(ComputeError::EmptyInput);
        }
        if dt <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "dt",
                value: dt.to_string(),
            });
        }
        let flux_len = n_frames - 2;
        let max_lag = resolution.min(flux_len.saturating_sub(1));
        let mut planner = FftPlanner::new();
        let mut acf = Array1::<f64>::zeros(max_lag + 1);
        for d in 0..3 {
            let mu_dot = central_diff_col(electric, d, dt);
            let m_dot = central_diff_col(magnetic, d, dt);
            let c = cross_correlate(&mut planner, &mu_dot, &m_dot, max_lag);
            for k in 0..=max_lag {
                acf[k] += c[k];
            }
        }
        let lag_times = Array1::from_iter((0..=max_lag).map(|i| i as f64 * dt));
        Ok(VcdCrossResult { lag_times, acf })
    }
}

/// Raw ROA cross-correlation iso/aniso curves — the ROA-spectrum raw input.
#[derive(Debug, Clone)]
pub struct RoaCrossResult {
    /// Lag times τ = i·dt, length `max_lag + 1`.
    pub lag_times: Array1<f64>,
    /// Isotropic ROA cross-correlation of `α̇` (electric polarizability
    /// derivative) with `Ġ′` (magnetic-dipole polarizability / optical-activity
    /// tensor derivative).
    pub acf_iso: Array1<f64>,
    /// Weighted anisotropic ROA cross-correlation (same diagonal/off-diagonal
    /// weighting as the Raman anisotropy).
    pub acf_aniso: Array1<f64>,
}

impl ComputeResult for RoaCrossResult {}

/// Raw ROA cross-tensor compute: cross-correlation of the electric
/// polarizability derivative `α̇` with the optical-activity tensor derivative
/// `Ġ′` (and, in the full theory, the electric-quadrupole tensor `A`).
///
/// Ported from `CROAEngine::ComputeACFPair`, `ROA_SPECTRUM_ROA` branch
/// (`src/roa.cpp`): the iso part cross-correlates the polarizability trace with
/// the (negated) `G′` trace, and the anisotropic parts mirror the Raman
/// deviatoric decomposition. Both tensors are passed in Voigt form
/// `[xx, yy, zz, xy, xz, yz]`; the same diagonal (½) / off-diagonal (3) weights
/// as [`RamanTensor`] are used so ROA shares the Raman normal-mode frequencies.
#[derive(Debug, Clone, Copy, Default)]
pub struct RoaCrossTensor;

/// `(electric_pol (n,6), g_tensor (n,6), dt, resolution)` for [`RoaCrossTensor`],
/// both in Voigt notation `[xx, yy, zz, xy, xz, yz]`.
pub type RoaCrossArgs<'a> = (&'a Array2<f64>, &'a Array2<f64>, f64, usize);

impl Compute for RoaCrossTensor {
    type Args<'a> = RoaCrossArgs<'a>;
    type Output = RoaCrossResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        _frames: &[&'a FA],
        args: Self::Args<'a>,
    ) -> Result<Self::Output, ComputeError> {
        let (el_pol, g_tensor, dt, resolution) = args;
        let n_frames = el_pol.shape()[0];
        if el_pol.shape()[1] != 6 || g_tensor.shape()[1] != 6 {
            return Err(ComputeError::DimensionMismatch {
                expected: 6,
                got: el_pol.shape()[1].max(g_tensor.shape()[1]),
                what: "ROA (electric_pol, g_tensor) (expected (n_frames, 6) Voigt)",
            });
        }
        if g_tensor.shape()[0] != n_frames {
            return Err(ComputeError::DimensionMismatch {
                expected: n_frames,
                got: g_tensor.shape()[0],
                what: "ROA electric_pol/g_tensor frame counts",
            });
        }
        if n_frames < 3 {
            return Err(ComputeError::EmptyInput);
        }
        if dt <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "dt",
                value: dt.to_string(),
            });
        }
        let flux_len = n_frames - 2;
        let max_lag = resolution.min(flux_len.saturating_sub(1));
        let mut planner = FftPlanner::new();

        // Central-difference derivatives of all six Voigt components.
        let a: Vec<Array1<f64>> = (0..6).map(|c| central_diff_col(el_pol, c, dt)).collect();
        let g: Vec<Array1<f64>> = (0..6).map(|c| central_diff_col(g_tensor, c, dt)).collect();

        // Isotropic: trace × trace.
        let a_iso: Array1<f64> = (0..flux_len)
            .map(|t| (a[0][t] + a[1][t] + a[2][t]) / 3.0)
            .collect();
        let g_iso: Array1<f64> = (0..flux_len)
            .map(|t| (g[0][t] + g[1][t] + g[2][t]) / 3.0)
            .collect();
        let acf_iso = cross_correlate(&mut planner, &a_iso, &g_iso, max_lag);

        // Anisotropic: 3 diagonal differences (weight ½) + 3 off-diagonals
        // (weight 3) — the RamanTensor deviatoric decomposition, cross-correlated.
        let mut acf_aniso = Array1::<f64>::zeros(max_lag + 1);
        let diag_pairs = [(0usize, 1usize), (1, 2), (2, 0)];
        for (i, j) in diag_pairs {
            let av: Array1<f64> = (0..flux_len).map(|t| a[i][t] - a[j][t]).collect();
            let gv: Array1<f64> = (0..flux_len).map(|t| g[i][t] - g[j][t]).collect();
            let c = cross_correlate(&mut planner, &av, &gv, max_lag);
            for k in 0..=max_lag {
                acf_aniso[k] += DIAG_ANISO_WEIGHT * c[k];
            }
        }
        for off in 3..6 {
            let c = cross_correlate(&mut planner, &a[off], &g[off], max_lag);
            for k in 0..=max_lag {
                acf_aniso[k] += OFFDIAG_ANISO_WEIGHT * c[k];
            }
        }

        let lag_times = Array1::from_iter((0..=max_lag).map(|i| i as f64 * dt));
        Ok(RoaCrossResult {
            lag_times,
            acf_iso,
            acf_aniso,
        })
    }
}

/// Raw resonance-Raman iso/aniso ACFs — identical machinery to [`RamanTensor`]
/// but consuming a caller-supplied **resonant** (excitation-frequency-dependent)
/// polarizability series instead of the static one.
///
/// molrs does not compute the excited-state response; the caller supplies the
/// resonant polarizability time series and this compute produces the same raw
/// iso/aniso ACFs (Voigt `[xx, yy, zz, xy, xz, yz]`) that [`RamanTensor`] /
/// `raman.cpp` produce, so the resonance-Raman spectrum reuses the Raman fit.
#[derive(Debug, Clone, Copy, Default)]
pub struct ResonanceRamanTensor;

/// `(resonant_polarizabilities (n,6), dt, resolution)` for
/// [`ResonanceRamanTensor`] — same shape/convention as [`RamanTensorArgs`].
pub type ResonanceRamanArgs<'a> = (&'a Array2<f64>, f64, usize);

impl Compute for ResonanceRamanTensor {
    type Args<'a> = ResonanceRamanArgs<'a>;
    type Output = RamanTensorResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        args: Self::Args<'a>,
    ) -> Result<Self::Output, ComputeError> {
        // Structurally identical to the static Raman tensor; only the input
        // polarizability differs (resonant vs static). Reuse, do not duplicate.
        RamanTensor.compute(frames, args)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::Frame;
    use molrs::store::block::Block;
    use ndarray::{Array1 as A1, Array2};
    use rand::{RngExt, SeedableRng};

    /// MD→SI conductivity prefactor with the Einstein 1/6 factor, lifted from the
    /// (removed) Einstein–Helfand conductivity free fn so the tests fold in the exact
    /// same constant the legacy free function used.
    fn einstein_helfand_prefactor() -> f64 {
        use molrs::units::constants::{
            ANGSTROM_M, BOLTZMANN as K_B_SI, ELEMENTARY_CHARGE as E_C, PICOSECOND_S,
        };
        (E_C * E_C * ANGSTROM_M * ANGSTROM_M / PICOSECOND_S) / (6.0 * ANGSTROM_M.powi(3) * K_B_SI)
    }

    /// Empty frame slice for the series-based raw computes.
    fn no_frames() -> Vec<&'static Frame> {
        Vec::new()
    }

    fn rng_series(n: usize, cols: usize, seed: u64) -> Array2<f64> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut s = Array2::zeros((n, cols));
        for t in 0..n {
            for c in 0..cols {
                s[[t, c]] = rng.random_range(-1.0..1.0);
            }
        }
        s
    }

    #[test]
    fn vacf_equals_vdos_acf_sum() {
        // ac-011: VACF == the per-DOF mean-subtract + FFT-ACF + DOF-average curve
        // that the PowerSpectrum (VDOS) transform consumes.
        let n = 512;
        let dt = 0.5;
        let res = 100;
        let v = rng_series(n, 9, 7);

        // Rebuild the same raw ACF the PowerSpectrum compute path produces.
        let max_lag = res.min(n - 1);
        let inv_n_frames = 1.0 / n as f64;
        let mut planner = FftPlanner::new();
        let mut acf_sum = A1::<f64>::zeros(max_lag + 1);
        for d in 0..9 {
            let mut col: A1<f64> = (0..n).map(|t| v[[t, d]]).collect();
            let mean: f64 = col.iter().sum::<f64>() * inv_n_frames;
            for x in col.iter_mut() {
                *x -= mean;
            }
            let acf = sig::acf_fft_with_planner(&mut planner, &col, max_lag).unwrap();
            for k in 0..=max_lag {
                acf_sum[k] += acf[k];
            }
        }
        for k in 0..=max_lag {
            acf_sum[k] *= 1.0 / 9.0;
        }

        let raw = VACF.compute(&no_frames(), (&v, dt, res)).unwrap();
        assert_eq!(raw.acf.len(), acf_sum.len());
        for k in 0..raw.acf.len() {
            assert!((raw.acf[k] - acf_sum[k]).abs() < 1e-12, "k={k}");
        }
        assert_eq!(raw.acf.len(), max_lag + 1);
    }

    #[test]
    fn green_kubo_diffusion_equals_vacf() {
        let n = 64;
        let dt = 1.0;
        let v = rng_series(n, 3, 11);
        let a = VACF.compute(&no_frames(), (&v, dt, 20)).unwrap();
        let b = GreenKuboDiffusion
            .compute(&no_frames(), (&v, dt, 20))
            .unwrap();
        assert_eq!(a.acf, b.acf);
        assert_eq!(a.lag_times, b.lag_times);
    }

    #[test]
    fn einstein_conductivity_msd_matches_direct_time_origin_average() {
        // ac-009: EinsteinConductivity.msd == the direct time-origin collective-
        // dipole MSD (the raw observable the removed bundled result also carried).
        let n = 256;
        let dt = 0.5;
        let mct = 80;
        let dipole = rng_series(n, 3, 3);
        let max_lag = mct.min(n - 1);
        let mut expected = A1::<f64>::zeros(max_lag + 1);
        for tau in 1..=max_lag {
            let count = n - tau;
            let mut acc = 0.0;
            for t in 0..count {
                let mut s = 0.0;
                for d in 0..3 {
                    let dx = dipole[[t + tau, d]] - dipole[[t, d]];
                    s += dx * dx;
                }
                acc += s;
            }
            expected[tau] = acc / count as f64;
        }
        let raw = EinsteinConductivity
            .compute(&no_frames(), (&dipole, dt, mct))
            .unwrap();
        assert_eq!(raw.msd.len(), expected.len());
        for k in 0..raw.msd.len() {
            assert!((raw.msd[k] - expected[k]).abs() < 1e-12, "k={k}");
            assert!((raw.lag_times[k] - k as f64 * dt).abs() < 1e-12);
        }
    }

    #[test]
    fn einstein_conductivity_msd_exact_small() {
        // Hand-checkable collective MSD on a 3-frame, purely-x ramp (moved from
        // dielectric.rs): M_J = [0, 1, 3] along x. msd[1] = mean(1², 2²) = 2.5,
        // msd[2] = 3² = 9.
        let dipole = ndarray::arr2(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]);
        let raw = EinsteinConductivity
            .compute(&no_frames(), (&dipole, 1.0, 2))
            .unwrap();
        assert!((raw.msd[0] - 0.0).abs() < 1e-12);
        assert!((raw.msd[1] - 2.5).abs() < 1e-12);
        assert!((raw.msd[2] - 9.0).abs() < 1e-12);
        assert_eq!(raw.lag_times.len(), 3);

        // Folded MD→SI prefactor (Einstein 1/6) ≈ 3.0988e6 S·m⁻¹ per
        // [(e·Å)²·ps⁻¹·Å⁻³·K⁻¹]. Guards against conversion drift.
        let prefactor = einstein_helfand_prefactor();
        assert!((prefactor - 3.0988e6).abs() / 3.0988e6 < 1e-3);
    }

    #[test]
    fn green_kubo_raw_jacf_matches_direct_acf() {
        // ac-010: GreenKuboConductivity.jacf == the direct unbiased current ACF
        // (the raw observable the removed bundled result also carried).
        let n = 256;
        let dt = 0.5;
        let mct = 80;
        let current = rng_series(n, 3, 5);
        let max_lag = mct.min(n - 1);
        let mut expected = A1::<f64>::zeros(max_lag + 1);
        for tau in 0..=max_lag {
            let count = n - tau;
            let mut acc = 0.0;
            for t in 0..count {
                let mut s = 0.0;
                for d in 0..3 {
                    s += current[[t, d]] * current[[t + tau, d]];
                }
                acc += s;
            }
            expected[tau] = acc / count as f64;
        }
        let raw = GreenKuboConductivity
            .compute(&no_frames(), (&current, dt, mct))
            .unwrap();
        assert_eq!(raw.jacf.len(), expected.len());
        for k in 0..raw.jacf.len() {
            assert!((raw.jacf[k] - expected[k]).abs() < 1e-12, "k={k}");
            assert!((raw.lag_times[k] - k as f64 * dt).abs() < 1e-12);
        }
    }

    fn make_frame(x: &[f64], y: &[f64], z: &[f64]) -> Frame {
        let mut block = Block::new();
        block
            .insert("x", A1::from_vec(x.to_vec()).into_dyn())
            .unwrap();
        block
            .insert("y", A1::from_vec(y.to_vec()).into_dyn())
            .unwrap();
        block
            .insert("z", A1::from_vec(z.to_vec()).into_dyn())
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", block);
        frame
    }

    #[test]
    fn einstein_diffusion_delegates_to_msd_windowed() {
        // ac-012: EinsteinDiffusion.msd == MSD::windowed().compute means.
        let xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let ys = [0.1, -0.2, 0.3, -0.1, 0.05, 0.0];
        let frames_owned: Vec<Frame> = (0..6)
            .map(|t| make_frame(&[xs[t], xs[t] + 0.5], &[ys[t], ys[t] + 0.4], &[0.0, 0.0]))
            .collect();
        let frames: Vec<&Frame> = frames_owned.iter().collect();

        let series = MSD::windowed().compute(&frames, ()).unwrap();
        let raw = EinsteinDiffusion
            .compute(&frames, EinsteinDiffusionArgs { dt: 2.0 })
            .unwrap();
        assert_eq!(raw.msd.len(), series.data.len());
        for i in 0..raw.msd.len() {
            assert!((raw.msd[i] - series.data[i].mean).abs() < 1e-12, "i={i}");
            assert!((raw.lag_times[i] - i as f64 * 2.0).abs() < 1e-12);
        }
    }

    #[test]
    fn debye_relaxation_carries_unnormalized_acf_and_metadata() {
        // ac-013.
        let n = 128;
        let dt = 0.5;
        let dipole = rng_series(n, 3, 9);
        let res = DebyeRelaxation {
            volume: 1234.5,
            temperature: 298.0,
            boundary: EwaldBoundary::TinFoil,
        }
        .compute(&no_frames(), (&dipole, dt, 40))
        .unwrap();
        // zero-lag variance == acf[0] and is non-zero (unnormalized).
        assert!((res.zero_lag_variance - res.acf[0]).abs() < 1e-15);
        assert!(res.zero_lag_variance > 0.0);
        assert_eq!(res.volume, 1234.5);
        assert_eq!(res.temperature, 298.0);
        assert_eq!(res.boundary, EwaldBoundary::TinFoil);
    }

    #[test]
    fn raw_max_lag_exceeds_length_clamps_not_panics() {
        // ac-014 companion: clamping max_correlation_time is fine; over-long
        // *consumption* by a downstream Fit is the OutOfRange case tested in the
        // fit modules. Here we confirm raw computes clamp rather than panic.
        let v = rng_series(8, 3, 1);
        let raw = VACF.compute(&no_frames(), (&v, 1.0, 1000)).unwrap();
        assert_eq!(raw.acf.len(), 8); // clamped to n_frames - 1 + 1.
    }

    #[test]
    fn series_computes_reject_bad_shape() {
        // ac-016: non-(_,3) series -> DimensionMismatch.
        let bad = rng_series(10, 2, 1);
        assert!(matches!(
            EinsteinConductivity.compute(&no_frames(), (&bad, 1.0, 5)),
            Err(ComputeError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            GreenKuboConductivity.compute(&no_frames(), (&bad, 1.0, 5)),
            Err(ComputeError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn einstein_conductivity_plus_linear_fit_matches_manual_ols() {
        // ac-015: LinearFit slope on EinsteinConductivity.msd reproduces a manual
        // OLS over the same diffusive window, and the σ = slope/(6·V·k_B·T)·prefactor
        // composition is well-defined (replaces the removed bundled
        // Einstein–Helfand conductivity).
        use crate::compute::fit::LinearFit;
        use crate::compute::traits::Fit;

        let n = 256;
        let dt = 0.5;
        let mct = 80;
        let (volume, temperature) = (1000.0, 300.0);
        let (start_frac, end_frac) = (0.2, 0.8);
        let dipole = rng_series(n, 3, 17);

        let raw = EinsteinConductivity
            .compute(&no_frames(), (&dipole, dt, mct))
            .unwrap();
        let fit = LinearFit {
            window: (start_frac, end_frac),
        }
        .fit((&raw.lag_times, &raw.msd))
        .unwrap();

        // Manual OLS over [fit_start, fit_end] on (lag_times, msd).
        let (fs, fe) = (fit.fit_start, fit.fit_end);
        let np = (fe - fs + 1) as f64;
        let (mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0);
        for i in fs..=fe {
            let x = raw.lag_times[i];
            let y = raw.msd[i];
            sx += x;
            sy += y;
            sxx += x * x;
            sxy += x * y;
        }
        let manual_slope = (np * sxy - sx * sy) / (np * sxx - sx * sx);
        assert!((fit.slope - manual_slope).abs() < 1e-12);

        let prefactor = einstein_helfand_prefactor();
        let sigma = prefactor * fit.slope / (volume * temperature);
        assert!(sigma.is_finite());
    }

    #[test]
    fn einstein_conductivity_plus_fit_recovers_nernst_einstein() {
        // ac-008 (scientific regression, moved from dielectric.rs): N independent
        // ions of charge q on uncorrelated 3-D random walks. For independent
        // carriers EinsteinConductivity + LinearFit must reduce to the
        // Nernst–Einstein value σ = n·q²·D/(k_B·T) within the ≤0.13 ensemble
        // tolerance. M_J(t) is ONE stochastic trajectory, so we ENSEMBLE-AVERAGE
        // σ over many realisations. Seed is fixed → deterministic across CI.
        use crate::compute::fit::LinearFit;
        use crate::compute::traits::Fit;
        use molrs::units::constants::{
            ANGSTROM_M, BOLTZMANN as K_B_SI, ELEMENTARY_CHARGE as E_C, PICOSECOND_S,
        };

        let n_realisations = 48usize;
        let n_ions = 50usize;
        let n_frames = 1500usize;
        let dt = 1.0_f64; // ps
        let q = 1.0_f64; // e
        let volume = 1.0e5_f64; // Å³
        let temperature = 300.0_f64; // K
        let step = 0.5_f64; // Å, uniform per-axis displacement amplitude
        // Nernst–Einstein prefactor (no Einstein 1/6 here: D folds it in).
        let ne_prefactor =
            (E_C * E_C * ANGSTROM_M * ANGSTROM_M / PICOSECOND_S) / (ANGSTROM_M.powi(3) * K_B_SI);
        let eh_prefactor = einstein_helfand_prefactor();

        let mut rng = rand::rngs::StdRng::seed_from_u64(20260601);
        let mut sigma_eh_sum = 0.0_f64;
        let mut sigma_ne_sum = 0.0_f64;
        for _ in 0..n_realisations {
            let mut pos = vec![[0.0_f64; 3]; n_ions];
            let mut dipole = Array2::<f64>::zeros((n_frames, 3));
            let mut step_sq_sum = 0.0_f64;
            let mut step_count = 0.0_f64;
            for f in 0..n_frames {
                for ion in &mut pos {
                    if f > 0 {
                        for c in ion.iter_mut() {
                            let s = rng.random_range(-step..step);
                            *c += s;
                            step_sq_sum += s * s;
                            step_count += 1.0;
                        }
                    }
                }
                let mut m = [0.0_f64; 3];
                for p in &pos {
                    for d in 0..3 {
                        m[d] += q * p[d];
                    }
                }
                for d in 0..3 {
                    dipole[[f, d]] = m[d];
                }
            }

            let max_corr = n_frames / 5;
            let raw = EinsteinConductivity
                .compute(&no_frames(), (&dipole, dt, max_corr))
                .unwrap();
            let fit = LinearFit { window: (0.1, 0.5) }
                .fit((&raw.lag_times, &raw.msd))
                .unwrap();
            sigma_eh_sum += eh_prefactor * fit.slope / (volume * temperature);

            // Realised per-axis step variance → D = var/(2·dt); analytic
            // Nernst–Einstein σ = n·q²·D/(k_B·T).
            let var_axis = step_sq_sum / step_count; // Å²
            let d_diff = var_axis / (2.0 * dt); // Å²·ps⁻¹
            let number_density = n_ions as f64 / volume; // Å⁻³
            sigma_ne_sum += ne_prefactor * number_density * q * q * d_diff / temperature;
        }

        let sigma_eh = sigma_eh_sum / n_realisations as f64;
        let sigma_ne = sigma_ne_sum / n_realisations as f64;
        let rel_err = (sigma_eh - sigma_ne).abs() / sigma_ne.abs();
        assert!(
            rel_err < 0.13,
            "ensemble EH σ = {sigma_eh} S/m vs Nernst–Einstein {sigma_ne} S/m (rel err {rel_err:.3})"
        );
        assert!(sigma_eh > 0.0);
    }

    #[test]
    fn green_kubo_raw_plus_running_integral_matches_manual_trapezoid() {
        // ac-015: RunningIntegral on GreenKuboConductivity.jacf reproduces a manual
        // trapezoidal integral, and σ = prefactor·∫/(V·k_B·T) is well-defined
        // (replaces the removed bundled Green–Kubo conductivity).
        use crate::compute::fit::RunningIntegral;
        use crate::compute::traits::Fit;
        use molrs::units::constants::{
            ANGSTROM_M, BOLTZMANN as K_B_SI, ELEMENTARY_CHARGE as E_C, PICOSECOND_S,
        };

        let n = 256;
        let dt = 0.5;
        let mct = 80;
        let (volume, temperature) = (1000.0, 300.0);
        let current = rng_series(n, 3, 19);

        let raw = GreenKuboConductivity
            .compute(&no_frames(), (&current, dt, mct))
            .unwrap();
        let integ = RunningIntegral.fit((&raw.jacf, dt, None)).unwrap();

        // Manual cumulative trapezoid of the JACF.
        let mut manual = 0.0;
        for tau in 1..raw.jacf.len() {
            manual += 0.5 * (raw.jacf[tau - 1] + raw.jacf[tau]) * dt;
        }
        let last = integ.integral.len() - 1;
        assert!((integ.integral[last] - manual).abs() < 1e-12);

        // Green–Kubo 1/3 prefactor.
        let prefactor = (E_C * E_C * ANGSTROM_M * ANGSTROM_M / PICOSECOND_S)
            / (3.0 * ANGSTROM_M.powi(3) * K_B_SI);
        let sigma = prefactor * integ.integral[last] / (volume * temperature);
        assert!(sigma.is_finite());
    }
}
