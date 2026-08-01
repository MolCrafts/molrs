//! Python bindings for the phase-01 raw-compute + explicit-fit split
//! (`molrs::compute::fitting`).
//!
//! This module exposes, as top-level `molrs` classes:
//!
//! * **Raw computes** — return ONLY a raw curve (+ scalar metadata), never a
//!   fitted coefficient: [`PyVACF`], [`PyEinsteinDiffusion`],
//!   [`PyGreenKuboDiffusion`], [`PyEinsteinConductivity`],
//!   [`PyGreenKuboConductivity`], [`PyDebyeRelaxation`].
//! * **Fits / transforms** — consume a raw curve and produce the derived
//!   quantity (slope/integral/plateau/τ/spectrum): [`PyLinearFit`],
//!   [`PyCumulativeTrapezoid`], [`PyPlateau`], [`PyDebyeFit`], [`PyPowerSpectrum`],
//!   [`PyIRSpectrum`], [`PyRamanSpectrum`], [`PyEinsteinHelfandSpectrum`],
//!   [`PyGreenKuboSpectrum`].
//!
//! Each `compute(...)` / `fit(...)` returns a plain `dict` of NumPy arrays /
//! scalars, matching the dielectric / transport binding style. The raw computes
//! deliberately omit any fitted scalar (no `sigma`/`D`), so the fit step is the
//! analyst's explicit, parameterized choice — and a raw result can never
//! silently fabricate a transport coefficient.

use molrs::compute::fitting::{CumulativeTrapezoid, LinearFit, Plateau};
use molrs::compute::transport::DebyeFit;
use molrs::compute::spectroscopy::{
    EinsteinHelfandSpectrum, GreenKuboSpectrum, IRSpectrum, PowerSpectrum, RamanSpectrum,
    ResonanceRamanSpectrum, RoaSpectrum, VcdSpectrum,
};
use molrs::compute::traits::{Compute, Fit};
use molrs::compute::transport::{
    DebyeRelaxation, EinsteinConductivity, EinsteinDiffusion, EinsteinDiffusionArgs, EwaldBoundary,
    GreenKuboConductivity, GreenKuboDiffusion, VACF,
};
use molrs::store::frame::Frame as CoreFrame;
use ndarray::Array1;
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::helpers::{collect_frames, py_value_err};

/// Empty frame slice for the series-based raw computes (`frames` is unused).
const EMPTY_FRAMES: &[&CoreFrame] = &[];

// ═══════════════════════════════════════════════════════════════════════════
// Raw computes
// ═══════════════════════════════════════════════════════════════════════════

// ── VACF ─────────────────────────────────────────────────────────────────────

/// Raw unnormalized velocity autocorrelation function (the VDOS /
/// Green–Kubo-diffusion input). Returns only the raw ACF curve — compose with
/// [`PowerSpectrum`](PyPowerSpectrum) for VDOS or
/// [`CumulativeTrapezoid`](PyCumulativeTrapezoid) for D.
#[pyclass(module = "molrs", name = "VACF")]
pub struct PyVACF;

#[pymethods]
impl PyVACF {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Compute the raw velocity ACF.
    ///
    /// Returns a dict ``{"lag_times", "acf"}`` of float64 arrays. **No** fitted
    /// scalar.
    fn compute<'py>(
        &self,
        py: Python<'py>,
        velocities: PyReadonlyArray2<'py, f64>,
        dt: f64,
        resolution: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let v = velocities.as_array().to_owned();
        let r = VACF
            .compute(EMPTY_FRAMES, (&v, dt, resolution))
            .map_err(py_value_err)?;
        let d = PyDict::new(py);
        d.set_item("lag_times", r.lag_times.into_pyarray(py))?;
        d.set_item("acf", r.acf.into_pyarray(py))?;
        Ok(d)
    }
}

// ── GreenKuboDiffusion (raw velocity ACF, diffusion route) ────────────────────

/// Raw velocity ACF for the Green–Kubo diffusion route (same raw curve as
/// [`VACF`](PyVACF)). `D = (1/d)·∫ VACF dt` is then a
/// [`CumulativeTrapezoid`](PyCumulativeTrapezoid) + scale step.
#[pyclass(module = "molrs", name = "GreenKuboDiffusion")]
pub struct PyGreenKuboDiffusion;

#[pymethods]
impl PyGreenKuboDiffusion {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Returns ``{"lag_times", "acf"}``; no fitted D.
    fn compute<'py>(
        &self,
        py: Python<'py>,
        velocities: PyReadonlyArray2<'py, f64>,
        dt: f64,
        resolution: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let v = velocities.as_array().to_owned();
        let r = GreenKuboDiffusion
            .compute(EMPTY_FRAMES, (&v, dt, resolution))
            .map_err(py_value_err)?;
        let d = PyDict::new(py);
        d.set_item("lag_times", r.lag_times.into_pyarray(py))?;
        d.set_item("acf", r.acf.into_pyarray(py))?;
        Ok(d)
    }
}

// ── EinsteinDiffusion (raw self-MSD; consumes frames) ─────────────────────────

/// Raw self-MSD for the Einstein diffusion route. Delegates to
/// `MSD::windowed` — MSD math is NOT re-derived. `D = slope/(2d)` is then a
/// [`LinearFit`](PyLinearFit) + scale step.
#[pyclass(module = "molrs", name = "EinsteinDiffusion")]
pub struct PyEinsteinDiffusion;

#[pymethods]
impl PyEinsteinDiffusion {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Compute the raw self-MSD over ``frames`` (a `Frame` or list of `Frame`).
    ///
    /// Returns ``{"lag_times", "msd"}``; no fitted D.
    fn compute<'py>(
        &self,
        py: Python<'py>,
        frames: &Bound<'py, PyAny>,
        dt: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let owned = collect_frames(frames)?;
        let refs: Vec<&CoreFrame> = owned.iter().collect();
        let r = EinsteinDiffusion
            .compute(&refs, EinsteinDiffusionArgs { dt })
            .map_err(py_value_err)?;
        let d = PyDict::new(py);
        d.set_item("lag_times", r.lag_times.into_pyarray(py))?;
        d.set_item("msd", r.msd.into_pyarray(py))?;
        Ok(d)
    }
}

// ── EinsteinConductivity (raw collective charge-dipole MSD) ───────────────────

/// Raw collective charge-dipole MSD — the raw portion of the legacy
/// `dielectric_einstein_helfand_conductivity`, with **no** fitted sigma/slope.
/// `σ = slope/(6·V·k_B·T)·prefactor` is a downstream
/// [`LinearFit`](PyLinearFit) + scale step.
#[pyclass(module = "molrs", name = "EinsteinConductivity")]
pub struct PyEinsteinConductivity;

#[pymethods]
impl PyEinsteinConductivity {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Compute the raw collective-dipole MSD.
    ///
    /// ``translational_dipole`` is ``(n_frames, 3)``. Returns
    /// ``{"lag_times", "msd"}``; **no** ``sigma``/``slope``.
    fn compute<'py>(
        &self,
        py: Python<'py>,
        translational_dipole: PyReadonlyArray2<'py, f64>,
        dt: f64,
        max_correlation_time: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let td = translational_dipole.as_array().to_owned();
        let r = EinsteinConductivity
            .compute(EMPTY_FRAMES, (&td, dt, max_correlation_time))
            .map_err(py_value_err)?;
        let d = PyDict::new(py);
        d.set_item("lag_times", r.lag_times.into_pyarray(py))?;
        d.set_item("msd", r.msd.into_pyarray(py))?;
        Ok(d)
    }
}

// ── GreenKuboConductivity (raw current ACF) ───────────────────────────────────

/// Raw current autocorrelation function — the raw portion of the legacy
/// `transport_green_kubo_conductivity`, with **no** fitted sigma. The
/// σ = (1/(3·V·k_B·T))·∫⟨JJ⟩ step is a downstream
/// [`CumulativeTrapezoid`](PyCumulativeTrapezoid) + scale.
#[pyclass(module = "molrs", name = "GreenKuboConductivity")]
pub struct PyGreenKuboConductivity;

#[pymethods]
impl PyGreenKuboConductivity {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Compute the raw current ACF.
    ///
    /// ``current`` is ``(n_frames, 3)``. Returns ``{"lag_times", "jacf"}``;
    /// **no** ``sigma``/``sigma_running``.
    fn compute<'py>(
        &self,
        py: Python<'py>,
        current: PyReadonlyArray2<'py, f64>,
        dt: f64,
        max_correlation_time: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let j = current.as_array().to_owned();
        let r = GreenKuboConductivity
            .compute(EMPTY_FRAMES, (&j, dt, max_correlation_time))
            .map_err(py_value_err)?;
        let d = PyDict::new(py);
        d.set_item("lag_times", r.lag_times.into_pyarray(py))?;
        d.set_item("jacf", r.jacf.into_pyarray(py))?;
        Ok(d)
    }
}

// ── DebyeRelaxation (raw dipole ACF + V/T/BC metadata) ────────────────────────

/// Raw dipole-ACF compute for the Debye relaxation route. Carries the
/// unnormalized ACF, the zero-lag variance ⟨M(0)²⟩, and the V/T/Ewald-BC
/// metadata the Debye amplitude needs (invariants b, c). The relaxation *shape*
/// τ comes from [`DebyeFit`](PyDebyeFit) applied to the **normalized** ACF.
#[pyclass(module = "molrs", name = "DebyeRelaxation")]
pub struct PyDebyeRelaxation {
    inner: DebyeRelaxation,
}

#[pymethods]
impl PyDebyeRelaxation {
    /// ``DebyeRelaxation(volume, temperature, boundary="tinfoil")``.
    #[new]
    #[pyo3(signature = (volume, temperature, boundary="tinfoil"))]
    fn new(volume: f64, temperature: f64, boundary: &str) -> PyResult<Self> {
        Ok(Self {
            inner: DebyeRelaxation {
                volume,
                temperature,
                boundary: EwaldBoundary::from_name(boundary).map_err(py_value_err)?,
            },
        })
    }

    /// Compute the raw dipole ACF + metadata.
    ///
    /// ``dipole_moments`` is ``(n_frames, 3)``. Returns ``{"lag_times", "acf",
    /// "zero_lag_variance", "volume", "temperature", "boundary"}``; the ACF is
    /// **unnormalized** (``acf[0] == zero_lag_variance``).
    fn compute<'py>(
        &self,
        py: Python<'py>,
        dipole_moments: PyReadonlyArray2<'py, f64>,
        dt: f64,
        max_correlation_time: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let dm = dipole_moments.as_array().to_owned();
        let r = self
            .inner
            .compute(EMPTY_FRAMES, (&dm, dt, max_correlation_time))
            .map_err(py_value_err)?;
        let d = PyDict::new(py);
        d.set_item("lag_times", r.lag_times.into_pyarray(py))?;
        d.set_item("acf", r.acf.into_pyarray(py))?;
        d.set_item("zero_lag_variance", r.zero_lag_variance)?;
        d.set_item("volume", r.volume)?;
        d.set_item("temperature", r.temperature)?;
        d.set_item(
            "boundary",
            match r.boundary {
                EwaldBoundary::TinFoil => "tinfoil",
                EwaldBoundary::Vacuum => "vacuum",
            },
        )?;
        Ok(d)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Fits / transforms
// ═══════════════════════════════════════════════════════════════════════════

// ── LinearFit ─────────────────────────────────────────────────────────────────

/// Ordinary-least-squares line fit over a fractional ``(start, end)`` window of
/// an ``(x, y)`` curve. Reproduces the OLS slope of the legacy
/// `einstein_helfand_conductivity` bit-for-bit on the same curve + window.
#[pyclass(module = "molrs", name = "LinearFit")]
pub struct PyLinearFit {
    inner: LinearFit,
}

#[pymethods]
impl PyLinearFit {
    /// ``LinearFit(start_frac, end_frac)`` — window as fractions of the last
    /// index, ``0 <= start_frac < end_frac <= 1``.
    #[new]
    fn new(start_frac: f64, end_frac: f64) -> Self {
        Self {
            inner: LinearFit {
                window: (start_frac, end_frac),
            },
        }
    }

    /// Fit ``y = slope*x + intercept`` over the window.
    ///
    /// Returns ``{"slope", "intercept", "r2", "fit_start", "fit_end"}``.
    fn fit<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let xa = x.as_array().to_owned();
        let ya = y.as_array().to_owned();
        let r = self.inner.fit((&xa, &ya)).map_err(py_value_err)?;
        let d = PyDict::new(py);
        d.set_item("slope", r.slope)?;
        d.set_item("intercept", r.intercept)?;
        d.set_item("r2", r.r2)?;
        d.set_item("fit_start", r.fit_start)?;
        d.set_item("fit_end", r.fit_end)?;
        Ok(d)
    }
}

// ── CumulativeTrapezoid ──────────────────────────────────────────────────────────

/// Cumulative trapezoidal integral of a uniformly-sampled curve. Reproduces the
/// running integral inside the legacy `green_kubo_conductivity` bit-for-bit on
/// the same curve + dt (before the Green–Kubo prefactor).
#[pyclass(module = "molrs", name = "CumulativeTrapezoid")]
pub struct PyCumulativeTrapezoid;

#[pymethods]
impl PyCumulativeTrapezoid {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Integrate ``y`` cumulatively with the trapezoid rule on step ``dt``.
    ///
    /// ``n_lags`` (optional) integrates only the first ``n_lags`` samples and
    /// **errors** (never silently truncates) if it exceeds the curve length.
    /// Returns ``{"integral"}`` (float64 array; ``integral[0] == 0``).
    #[pyo3(signature = (y, dt, n_lags=None))]
    fn fit<'py>(
        &self,
        py: Python<'py>,
        y: PyReadonlyArray1<'py, f64>,
        dt: f64,
        n_lags: Option<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let ya = y.as_array().to_owned();
        let r = CumulativeTrapezoid
            .fit((&ya, dt, n_lags))
            .map_err(py_value_err)?;
        let d = PyDict::new(py);
        d.set_item("integral", r.integral.into_pyarray(py))?;
        Ok(d)
    }
}

// ── Plateau ──────────────────────────────────────────────────────────────────

/// Windowed-mean plateau reader over a fractional ``(a, b)`` window of a curve
/// (e.g. reading the converged tail of a Green–Kubo running integral).
#[pyclass(module = "molrs", name = "Plateau")]
pub struct PyPlateau {
    inner: Plateau,
}

#[pymethods]
impl PyPlateau {
    /// ``Plateau(a, b)`` — window as fractions of the last index,
    /// ``0 <= a < b <= 1``.
    #[new]
    fn new(a: f64, b: f64) -> Self {
        Self {
            inner: Plateau { window: (a, b) },
        }
    }

    /// Average the curve over the window.
    ///
    /// Returns ``{"value", "n_samples", "std"}``.
    fn fit<'py>(
        &self,
        py: Python<'py>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let ya = y.as_array().to_owned();
        let r = self.inner.fit(&ya).map_err(py_value_err)?;
        let d = PyDict::new(py);
        d.set_item("value", r.value)?;
        d.set_item("n_samples", r.n_samples)?;
        d.set_item("std", r.std)?;
        Ok(d)
    }
}

// ── DebyeFit (time-domain log-linear ACF fit) ─────────────────────────────────

/// Single-exponential (Debye) relaxation fit of a **normalized** dipole ACF
/// Φ(t) = A·exp(−t/τ), by log-linear least squares over the leading positive
/// run. This is the **time-domain** ACF fit consolidated from molpy; it returns
/// τ and the amplitude A.
#[pyclass(module = "molrs", name = "DebyeFit")]
pub struct PyDebyeFit;

#[pymethods]
impl PyDebyeFit {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Fit Φ(t) = A·exp(−t/τ) on the normalized ACF ``phi`` (sample step ``dt``).
    ///
    /// Returns ``{"tau", "amplitude", "n_samples"}``.
    fn fit<'py>(
        &self,
        py: Python<'py>,
        phi: PyReadonlyArray1<'py, f64>,
        dt: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let p = phi.as_array().to_owned();
        let r = DebyeFit.fit((&p, dt)).map_err(py_value_err)?;
        let d = PyDict::new(py);
        d.set_item("tau", r.tau)?;
        d.set_item("amplitude", r.amplitude)?;
        d.set_item("n_samples", r.n_samples)?;
        Ok(d)
    }
}

// ── PowerSpectrum (VDOS) ──────────────────────────────────────────────────────

fn spectrum_dict<'py>(
    py: Python<'py>,
    frequencies_cm1: Array1<f64>,
    intensities: Array1<f64>,
    resolution: usize,
    n_frames: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("frequencies_cm1", frequencies_cm1.into_pyarray(py))?;
    d.set_item("intensities", intensities.into_pyarray(py))?;
    d.set_item("resolution", resolution)?;
    d.set_item("n_frames", n_frames)?;
    Ok(d)
}

/// Velocity power spectrum (VDOS) transform of a **raw velocity ACF**
/// (CosineSq window + zero-padded forward FFT). Reproduces the legacy
/// `power_spectrum` bit-for-bit on the raw ACF that function builds internally.
#[pyclass(module = "molrs", name = "PowerSpectrum")]
pub struct PyPowerSpectrum;

#[pymethods]
impl PyPowerSpectrum {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Window + FFT the raw ACF (timestep ``dt_fs``) into a VDOS spectrum.
    ///
    /// Returns ``{"frequencies_cm1", "intensities", "resolution", "n_frames"}``.
    fn fit<'py>(
        &self,
        py: Python<'py>,
        acf: PyReadonlyArray1<'py, f64>,
        dt_fs: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let a = acf.as_array().to_owned();
        let r = PowerSpectrum.fit((&a, dt_fs)).map_err(py_value_err)?;
        spectrum_dict(
            py,
            r.frequencies_cm1,
            r.intensities,
            r.resolution,
            r.n_frames,
        )
    }
}

// ── IRSpectrum ───────────────────────────────────────────────────────────────

/// Infrared absorption spectrum transform of a **raw dipole-flux ACF**
/// (same window+FFT pipeline as [`PowerSpectrum`](PyPowerSpectrum); only the
/// supplied ACF differs). Reproduces the legacy `ir_spectrum` bit-for-bit.
#[pyclass(module = "molrs", name = "IRSpectrum")]
pub struct PyIRSpectrum;

#[pymethods]
impl PyIRSpectrum {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Window + FFT the raw flux ACF (timestep ``dt_fs``) into an IR spectrum.
    ///
    /// Returns ``{"frequencies_cm1", "intensities", "resolution", "n_frames"}``.
    fn fit<'py>(
        &self,
        py: Python<'py>,
        acf: PyReadonlyArray1<'py, f64>,
        dt_fs: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let a = acf.as_array().to_owned();
        let r = IRSpectrum.fit((&a, dt_fs)).map_err(py_value_err)?;
        spectrum_dict(
            py,
            r.frequencies_cm1,
            r.intensities,
            r.resolution,
            r.n_frames,
        )
    }
}

// ── RamanSpectrum ────────────────────────────────────────────────────────────

/// Raman spectrum transform of **raw isotropic + anisotropic ACFs**
/// (one CosineSq window per ACF, FFT both, then the cross-section + Bose
/// prefactors). Reproduces the legacy `raman_spectrum` bit-for-bit.
#[pyclass(module = "molrs", name = "RamanSpectrum")]
pub struct PyRamanSpectrum {
    inner: RamanSpectrum,
}

#[pymethods]
impl PyRamanSpectrum {
    /// ``RamanSpectrum(incident_frequency_cm1=0.0, temperature_k=0.0,
    /// averaged=False)`` — set ``incident_frequency_cm1``/``temperature_k`` to
    /// ``0.0`` to skip the cross-section / Bose correction.
    #[new]
    #[pyo3(signature = (incident_frequency_cm1=0.0, temperature_k=0.0, averaged=false))]
    fn new(incident_frequency_cm1: f64, temperature_k: f64, averaged: bool) -> Self {
        Self {
            inner: RamanSpectrum {
                incident_frequency_cm1,
                temperature_k,
                averaged,
            },
        }
    }

    /// Window + FFT + prefactors on the raw iso/aniso ACFs (timestep ``dt_fs``).
    ///
    /// Returns ``{"frequencies_cm1", "isotropic", "anisotropic", "parallel",
    /// "perpendicular", "resolution", "n_frames"}`` (``parallel`` /
    /// ``perpendicular`` are ``None`` unless ``averaged=True``).
    fn fit<'py>(
        &self,
        py: Python<'py>,
        acf_iso: PyReadonlyArray1<'py, f64>,
        acf_aniso: PyReadonlyArray1<'py, f64>,
        dt_fs: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let iso = acf_iso.as_array().to_owned();
        let aniso = acf_aniso.as_array().to_owned();
        let r = self
            .inner
            .fit((&iso, &aniso, dt_fs))
            .map_err(py_value_err)?;
        let d = PyDict::new(py);
        d.set_item("frequencies_cm1", r.frequencies_cm1.into_pyarray(py))?;
        d.set_item("isotropic", r.isotropic.into_pyarray(py))?;
        d.set_item("anisotropic", r.anisotropic.into_pyarray(py))?;
        match r.parallel {
            Some(p) => d.set_item("parallel", p.into_pyarray(py))?,
            None => d.set_item("parallel", py.None())?,
        }
        match r.perpendicular {
            Some(p) => d.set_item("perpendicular", p.into_pyarray(py))?,
            None => d.set_item("perpendicular", py.None())?,
        }
        d.set_item("resolution", r.resolution)?;
        d.set_item("n_frames", r.n_frames)?;
        Ok(d)
    }
}

// ── EinsteinHelfandSpectrum (dielectric ε(ω), dipole route) ───────────────────

fn dielectric_dict<'py>(
    py: Python<'py>,
    frequencies: Array1<f64>,
    eps_real: Array1<f64>,
    eps_imag: Array1<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("frequencies", frequencies.into_pyarray(py))?;
    d.set_item("eps_real", eps_real.into_pyarray(py))?;
    d.set_item("eps_imag", eps_imag.into_pyarray(py))?;
    Ok(d)
}

/// Einstein–Helfand ε(ω) transform of a **raw fluctuation dipole ACF** (the
/// [`DebyeRelaxation`](PyDebyeRelaxation) ACF): one-sided cos² taper +
/// derivative-FT + the `4π·KAPPA/(3·V·k_B·T)` prefactor. Reproduces the legacy
/// `einstein_helfand_spectrum` bit-for-bit on the raw ACF that function built
/// internally.
#[pyclass(module = "molrs", name = "EinsteinHelfandSpectrum")]
pub struct PyEinsteinHelfandSpectrum {
    inner: EinsteinHelfandSpectrum,
}

#[pymethods]
impl PyEinsteinHelfandSpectrum {
    /// ``EinsteinHelfandSpectrum(dt, volume, temperature, epsilon_inf,
    /// zero_lag_variance)`` — ``zero_lag_variance`` is ⟨|δM|²⟩ (the
    /// ``DebyeRelaxation`` ``zero_lag_variance``), which pins the exact DC bin.
    #[new]
    fn new(
        dt: f64,
        volume: f64,
        temperature: f64,
        epsilon_inf: f64,
        zero_lag_variance: f64,
    ) -> Self {
        Self {
            inner: EinsteinHelfandSpectrum {
                dt,
                volume,
                temperature,
                epsilon_inf,
                zero_lag_variance,
            },
        }
    }

    /// Transform the raw fluctuation dipole ``acf`` into ε(ω).
    ///
    /// Returns ``{"frequencies", "eps_real", "eps_imag"}`` (float64 arrays).
    fn fit<'py>(
        &self,
        py: Python<'py>,
        acf: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let a = acf.as_array().to_owned();
        let r = self.inner.fit(&a).map_err(py_value_err)?;
        dielectric_dict(py, r.frequencies, r.eps_real, r.eps_imag)
    }
}

// ── GreenKuboSpectrum (dielectric ε(ω), current route) ────────────────────────

/// Green–Kubo ε(ω) transform of a **raw current ACF** (the
/// [`GreenKuboConductivity`](PyGreenKuboConductivity) ACF over the post-NaN
/// series): window + FFT → σ(ω) → ε(ω). Reproduces the legacy
/// `green_kubo_spectrum` bit-for-bit on the raw ACF that function built
/// internally.
#[pyclass(module = "molrs", name = "GreenKuboSpectrum")]
pub struct PyGreenKuboSpectrum {
    inner: GreenKuboSpectrum,
}

#[pymethods]
impl PyGreenKuboSpectrum {
    /// ``GreenKuboSpectrum(dt, volume, temperature, epsilon_inf,
    /// window_type="hann")`` — ``window_type`` is ``"cosine_sq"``, ``"hann"``,
    /// or ``"blackman"``.
    #[new]
    #[pyo3(signature = (dt, volume, temperature, epsilon_inf, window_type="hann"))]
    fn new(dt: f64, volume: f64, temperature: f64, epsilon_inf: f64, window_type: &str) -> Self {
        Self {
            inner: GreenKuboSpectrum {
                dt,
                volume,
                temperature,
                epsilon_inf,
                window_type: window_type.to_string(),
            },
        }
    }

    /// Transform the raw current ``acf`` into ε(ω).
    ///
    /// Returns ``{"frequencies", "eps_real", "eps_imag"}`` (float64 arrays).
    fn fit<'py>(
        &self,
        py: Python<'py>,
        acf: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let a = acf.as_array().to_owned();
        let r = self.inner.fit(&a).map_err(py_value_err)?;
        dielectric_dict(py, r.frequencies, r.eps_real, r.eps_imag)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Registration
// ═══════════════════════════════════════════════════════════════════════════

// ── Chiral / resonance spectra (analysis parity) ───────────────────────────────

/// Build the Raman-family result dict (shared by ROA / resonance-Raman).
fn raman_dict<'py>(
    py: Python<'py>,
    r: molrs::compute::RamanSpectrumResult,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("frequencies_cm1", r.frequencies_cm1.into_pyarray(py))?;
    d.set_item("isotropic", r.isotropic.into_pyarray(py))?;
    d.set_item("anisotropic", r.anisotropic.into_pyarray(py))?;
    match r.parallel {
        Some(p) => d.set_item("parallel", p.into_pyarray(py))?,
        None => d.set_item("parallel", py.None())?,
    }
    match r.perpendicular {
        Some(p) => d.set_item("perpendicular", p.into_pyarray(py))?,
        None => d.set_item("perpendicular", py.None())?,
    }
    d.set_item("resolution", r.resolution)?;
    d.set_item("n_frames", r.n_frames)?;
    Ok(d)
}

/// Vibrational circular dichroism (VCD) transform of a **raw electric/magnetic
/// dipole cross-flux ACF** (same window+FFT pipeline as `IRSpectrum`; the signed
/// cross-flux ACF differs).
#[pyclass(module = "molrs", name = "VcdSpectrum")]
pub struct PyVcdSpectrum;

#[pymethods]
impl PyVcdSpectrum {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Window + FFT the raw VCD cross-flux ACF (timestep ``dt_fs``).
    ///
    /// Returns ``{"frequencies_cm1", "intensities", "resolution", "n_frames"}``.
    fn fit<'py>(
        &self,
        py: Python<'py>,
        acf: PyReadonlyArray1<'py, f64>,
        dt_fs: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let a = acf.as_array().to_owned();
        let r = VcdSpectrum.fit((&a, dt_fs)).map_err(py_value_err)?;
        spectrum_dict(
            py,
            r.frequencies_cm1,
            r.intensities,
            r.resolution,
            r.n_frames,
        )
    }
}

/// Raman optical activity (ROA) transform of **raw iso/aniso ROA
/// cross-correlations** (shares the Raman window+FFT+prefactor pipeline).
#[pyclass(module = "molrs", name = "RoaSpectrum")]
pub struct PyRoaSpectrum {
    inner: RoaSpectrum,
}

#[pymethods]
impl PyRoaSpectrum {
    /// ``RoaSpectrum(incident_frequency_cm1=0.0, temperature_k=0.0, averaged=False)``.
    #[new]
    #[pyo3(signature = (incident_frequency_cm1=0.0, temperature_k=0.0, averaged=false))]
    fn new(incident_frequency_cm1: f64, temperature_k: f64, averaged: bool) -> Self {
        Self {
            inner: RoaSpectrum {
                incident_frequency_cm1,
                temperature_k,
                averaged,
            },
        }
    }

    /// Window + FFT the raw iso/aniso ROA ACFs (timestep ``dt_fs``).
    ///
    /// Returns ``{"frequencies_cm1", "isotropic", "anisotropic", "parallel",
    /// "perpendicular", "resolution", "n_frames"}``.
    fn fit<'py>(
        &self,
        py: Python<'py>,
        acf_iso: PyReadonlyArray1<'py, f64>,
        acf_aniso: PyReadonlyArray1<'py, f64>,
        dt_fs: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let iso = acf_iso.as_array().to_owned();
        let aniso = acf_aniso.as_array().to_owned();
        let r = self
            .inner
            .fit((&iso, &aniso, dt_fs))
            .map_err(py_value_err)?;
        raman_dict(py, r)
    }
}

/// Resonance-Raman transform of **raw resonant iso/aniso ACFs** (identical
/// pipeline to `RamanSpectrum`; the ACFs come from a resonant polarizability
/// series upstream).
#[pyclass(module = "molrs", name = "ResonanceRamanSpectrum")]
pub struct PyResonanceRamanSpectrum {
    inner: ResonanceRamanSpectrum,
}

#[pymethods]
impl PyResonanceRamanSpectrum {
    /// ``ResonanceRamanSpectrum(incident_frequency_cm1=0.0, temperature_k=0.0, averaged=False)``.
    #[new]
    #[pyo3(signature = (incident_frequency_cm1=0.0, temperature_k=0.0, averaged=false))]
    fn new(incident_frequency_cm1: f64, temperature_k: f64, averaged: bool) -> Self {
        Self {
            inner: ResonanceRamanSpectrum {
                incident_frequency_cm1,
                temperature_k,
                averaged,
            },
        }
    }

    /// Window + FFT the raw resonant iso/aniso ACFs (timestep ``dt_fs``).
    ///
    /// Returns the same dict as ``RoaSpectrum.fit``.
    fn fit<'py>(
        &self,
        py: Python<'py>,
        acf_iso: PyReadonlyArray1<'py, f64>,
        acf_aniso: PyReadonlyArray1<'py, f64>,
        dt_fs: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let iso = acf_iso.as_array().to_owned();
        let aniso = acf_aniso.as_array().to_owned();
        let r = self
            .inner
            .fit((&iso, &aniso, dt_fs))
            .map_err(py_value_err)?;
        raman_dict(py, r)
    }
}

/// Register the raw-compute + fit classes at the top level of the `molrs`
/// module (`molrs.VACF`, `molrs.LinearFit`, …).
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Raw computes.
    m.add_class::<PyVACF>()?;
    m.add_class::<PyGreenKuboDiffusion>()?;
    m.add_class::<PyEinsteinDiffusion>()?;
    m.add_class::<PyEinsteinConductivity>()?;
    m.add_class::<PyGreenKuboConductivity>()?;
    m.add_class::<PyDebyeRelaxation>()?;
    // Fits / transforms.
    m.add_class::<PyLinearFit>()?;
    m.add_class::<PyCumulativeTrapezoid>()?;
    m.add_class::<PyPlateau>()?;
    m.add_class::<PyDebyeFit>()?;
    m.add_class::<PyPowerSpectrum>()?;
    m.add_class::<PyIRSpectrum>()?;
    m.add_class::<PyRamanSpectrum>()?;
    m.add_class::<PyEinsteinHelfandSpectrum>()?;
    m.add_class::<PyGreenKuboSpectrum>()?;
    // Chiral / resonance spectra (analysis parity).
    m.add_class::<PyVcdSpectrum>()?;
    m.add_class::<PyRoaSpectrum>()?;
    m.add_class::<PyResonanceRamanSpectrum>()?;
    Ok(())
}
