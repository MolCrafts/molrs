//! Generic curve fits (`molrs::compute::fitting`), in scipy's vocabulary:
//! `LinearFit`, `CumulativeTrapezoid`, `Plateau`.

use crate::error::py_value_err;
use molrs::compute::{CumulativeTrapezoid, Fit, LinearFit, Plateau};
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};

// ═══════════════════════════════════════════════════════════════════════════
// Fits / transforms
// ═══════════════════════════════════════════════════════════════════════════

// ── LinearFit ─────────────────────────────────────────────────────────────────

/// Ordinary-least-squares line fit over a fractional ``(start, end)`` window of
/// an ``(x, y)`` curve. Reproduces the OLS slope of the legacy
/// `einstein_helfand_conductivity` bit-for-bit on the same curve + window.
#[pyclass(module = "molrs.compute", name = "LinearFit")]
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
#[pyclass(module = "molrs.compute", name = "CumulativeTrapezoid")]
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
#[pyclass(module = "molrs.compute", name = "Plateau")]
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

/// Register this domain's classes and functions.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLinearFit>()?;
    m.add_class::<PyCumulativeTrapezoid>()?;
    m.add_class::<PyPlateau>()?;
    Ok(())
}
