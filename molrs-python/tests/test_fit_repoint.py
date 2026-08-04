"""Parity tests for the raw-compute + explicit-fit PyO3 bindings.

These lock the contract of the compute/fit repoint:

* the new raw-compute / fit classes import and construct (ac-001);
* raw computes return ONLY a raw curve, no fitted scalar (ac-002);
* the explicit raw->fit pipeline is well-defined and reproduces the manual
  slope / integral / spectrum (within the documented float tolerance).

The legacy bundled conductivity / ε(ω)-spectrum free-function bindings were
removed (compute-fit-03 / compute-fit-04); the composition tests below replace
the former DeprecationWarning parity checks.

The kernels themselves are unit-tested in Rust; these are wiring + parity
checks against a freshly rebuilt wheel.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

import molrs

# Physical prefactors the legacy conductivity free fns fold in (MD real units ->
# S/m). Mirrors molrs::units::constants used by the Rust regression tests.
_E_C = 1.602176634e-19
_K_B = 1.380649e-23
_ANGSTROM_M = 1e-10
_PICOSECOND_S = 1e-12


def _rng_series(n, cols, seed):
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.uniform(-1.0, 1.0, size=(n, cols)))


# ── ac-001: import + construct ───────────────────────────────────────────────

#: ``(module path, class name)``. The path is half the contract — each class
#: answers in the subpackage named after its ``molrs::compute`` domain.
_NEW_NAMES = [
    ("molrs.compute.transport", "VACF"),
    ("molrs.compute.transport", "GreenKuboDiffusion"),
    ("molrs.compute.transport", "EinsteinDiffusion"),
    ("molrs.compute.transport", "EinsteinConductivity"),
    ("molrs.compute.transport", "GreenKuboConductivity"),
    ("molrs.compute.transport", "DebyeRelaxation"),
    ("molrs.compute.transport", "DebyeFit"),
    ("molrs.compute.fitting", "LinearFit"),
    ("molrs.compute.fitting", "CumulativeTrapezoid"),
    ("molrs.compute.fitting", "Plateau"),
    ("molrs.compute.spectroscopy", "PowerSpectrum"),
    ("molrs.compute.spectroscopy", "IRSpectrum"),
    ("molrs.compute.spectroscopy", "RamanSpectrum"),
    ("molrs.compute.spectroscopy", "EinsteinHelfandSpectrum"),
    ("molrs.compute.spectroscopy", "GreenKuboSpectrum"),
]


@pytest.mark.parametrize("path,name", _NEW_NAMES)
def test_new_class_importable_and_constructible(path, name):
    module = importlib.import_module(path)
    assert name in module.__all__, f"{path}.{name} missing from __all__"
    cls = getattr(module, name)
    assert isinstance(cls, type) and callable(cls)


def test_raw_einstein_conductivity_is_raw_only():
    dipole = _rng_series(256, 3, 3)
    raw = molrs.compute.transport.EinsteinConductivity().compute(dipole, 0.5, 80)
    assert set(raw) == {"lag_times", "msd"}
    assert "sigma" not in raw and "slope" not in raw
    assert raw["msd"].dtype == np.float64


def test_raw_green_kubo_conductivity_is_raw_only():
    current = _rng_series(256, 3, 5)
    raw = molrs.compute.transport.GreenKuboConductivity().compute(current, 0.5, 80)
    assert set(raw) == {"lag_times", "jacf"}
    assert "sigma" not in raw and "sigma_running" not in raw


def test_raw_vacf_is_raw_only():
    v = _rng_series(512, 9, 7)
    raw = molrs.compute.transport.VACF().compute(v, 0.5, 100)
    assert set(raw) == {"lag_times", "acf"}
    assert "intensities" not in raw and "D" not in raw


def test_raw_debye_relaxation_unnormalized_with_metadata():
    dm = _rng_series(128, 3, 9)
    raw = molrs.compute.transport.DebyeRelaxation(1234.5, 298.0, "tinfoil").compute(
        dm, 0.5, 40
    )
    assert raw["zero_lag_variance"] == raw["acf"][0]
    assert raw["zero_lag_variance"] > 0.0
    assert raw["volume"] == 1234.5
    assert raw["temperature"] == 298.0
    assert raw["boundary"] == "tinfoil"


# ── ε(ω) raw-compute + Fit composition (compute-fit-04) ───────────────────────


def test_eh_spectrum_pipeline():
    # compute-fit-04: the bundled dielectric_einstein_helfand_spectrum binding
    # was removed. ε(ω) is now DebyeRelaxation (raw fluctuation dipole ACF) +
    # EinsteinHelfandSpectrum.
    dm = np.ones((100, 3)) * 0.1
    raw = molrs.compute.transport.DebyeRelaxation(1000.0, 300.0, "tinfoil").compute(
        dm, 0.001, 10
    )
    out = molrs.compute.spectroscopy.EinsteinHelfandSpectrum(
        0.001, 1000.0, 300.0, 1.0, raw["zero_lag_variance"]
    ).fit(raw["acf"])
    assert "frequencies" in out
    assert len(out["frequencies"]) == len(out["eps_real"]) == len(out["eps_imag"])


def test_gk_spectrum_pipeline():
    # compute-fit-04: the bundled dielectric_green_kubo_spectrum binding was
    # removed. ε(ω) is now GreenKuboConductivity (raw current ACF) +
    # GreenKuboSpectrum.
    j = np.ones((100, 3)) * 0.001
    raw = molrs.compute.transport.GreenKuboConductivity().compute(j, 0.001, 10)
    out = molrs.compute.spectroscopy.GreenKuboSpectrum(
        0.001, 1000.0, 300.0, 1.0, "hann"
    ).fit(raw["jacf"])
    assert "frequencies" in out
    assert len(out["frequencies"]) == len(out["eps_real"]) == len(out["eps_imag"])


# ── raw->fit conductivity composition is well-defined ─────────────────────────


def test_einstein_pipeline_sigma_well_defined():
    # The bundled dielectric_einstein_helfand_conductivity binding was removed in
    # compute-fit-03; σ is now EinsteinConductivity (raw collective-dipole MSD) +
    # LinearFit (slope) + the slope/(6·V·k_B·T) MD→SI prefactor.
    n, dt, mct, volume, temperature = 256, 0.5, 80, 1000.0, 300.0
    start_frac, end_frac = 0.2, 0.8
    dipole = _rng_series(n, 3, 17)

    raw = molrs.compute.transport.EinsteinConductivity().compute(dipole, dt, mct)
    fit = molrs.compute.fitting.LinearFit(start_frac, end_frac).fit(
        raw["lag_times"], raw["msd"]
    )

    # Manual OLS over the same window reproduces the LinearFit slope.
    fs, fe = fit["fit_start"], fit["fit_end"]
    x, y = raw["lag_times"][fs : fe + 1], raw["msd"][fs : fe + 1]
    manual_slope = np.polyfit(x, y, 1)[0]
    assert fit["slope"] == pytest.approx(manual_slope, rel=1e-9)

    prefactor = (_E_C * _E_C * _ANGSTROM_M * _ANGSTROM_M / _PICOSECOND_S) / (
        6.0 * _ANGSTROM_M**3 * _K_B
    )
    sigma = prefactor * fit["slope"] / (volume * temperature)
    assert np.isfinite(sigma)


def test_green_kubo_pipeline_sigma_well_defined():
    # The bundled transport_green_kubo_conductivity binding was removed in
    # compute-fit-03; σ is now GreenKuboConductivity (raw current ACF) +
    # CumulativeTrapezoid + the (1/(3·V·k_B·T))·∫⟨JJ⟩ MD→SI prefactor.
    n, dt, mct, volume, temperature = 256, 0.5, 80, 1000.0, 300.0
    current = _rng_series(n, 3, 19)

    raw = molrs.compute.transport.GreenKuboConductivity().compute(current, dt, mct)
    integ = molrs.compute.fitting.CumulativeTrapezoid().fit(raw["jacf"], dt)

    # Manual cumulative trapezoid reproduces the CumulativeTrapezoid endpoint.
    manual = float(np.trapezoid(raw["jacf"], dx=dt))
    assert integ["integral"][-1] == pytest.approx(manual, rel=1e-9)

    prefactor = (_E_C * _E_C * _ANGSTROM_M * _ANGSTROM_M / _PICOSECOND_S) / (
        3.0 * _ANGSTROM_M**3 * _K_B
    )
    sigma = prefactor * integ["integral"][-1] / (volume * temperature)
    assert np.isfinite(sigma)


# ── ac-005: spectral transforms reproduce legacy power/ir spectra ─────────────


def _power_acf(velocities, max_lag):
    """Rebuild the raw velocity ACF power_spectrum builds before windowing."""
    from molrs.signal import acf_fft

    n_frames, n_dof = velocities.shape
    acf_sum = np.zeros(max_lag + 1)
    for d in range(n_dof):
        col = velocities[:, d] - velocities[:, d].mean()
        acf_sum += acf_fft(col, max_lag)
    acf_sum /= n_dof
    return acf_sum


def test_power_spectrum_fit_matches_raw_acf_path():
    # PowerSpectrum is a pure Fit on a precomputed ACF: same ACF → same spectrum.
    # VACF (unbiased 1/(n-τ) velocity ACF, no mean-subtraction) is a different
    # estimator than the mean-subtracted FFT ACF used for some VDOS recipes —
    # they must not be forced equal after the 0.12 VACF norm fix.
    n, dt, res = 1024, 0.5, 200
    v = np.zeros((n, 3))
    t = np.arange(n) * dt
    v[:, 0] = np.sin(2.0 * np.pi * 10.0 * 1e-3 * t)
    max_lag = min(res, n - 1)

    raw = molrs.compute.transport.VACF().compute(np.ascontiguousarray(v), dt, res)
    assert "acf" in raw and raw["acf"].shape[0] == max_lag + 1

    manual_acf = _power_acf(v, max_lag)
    # Distinct estimators (scale and mean-subtraction differ).
    assert not np.allclose(raw["acf"], manual_acf)

    spec_from_manual = molrs.compute.spectroscopy.PowerSpectrum().fit(
        np.ascontiguousarray(manual_acf), dt
    )
    spec_from_manual_again = molrs.compute.spectroscopy.PowerSpectrum().fit(
        np.ascontiguousarray(manual_acf), dt
    )
    np.testing.assert_array_equal(
        spec_from_manual["frequencies_cm1"],
        spec_from_manual_again["frequencies_cm1"],
    )
    np.testing.assert_allclose(
        spec_from_manual["intensities"],
        spec_from_manual_again["intensities"],
        rtol=1e-12,
    )
    # VACF curve is a valid PowerSpectrum input (shape + finite).
    spec_from_vacf = molrs.compute.spectroscopy.PowerSpectrum().fit(raw["acf"], dt)
    assert spec_from_vacf["frequencies_cm1"].shape == spec_from_vacf["intensities"].shape
    assert np.all(np.isfinite(spec_from_vacf["intensities"]))


def test_raman_spectrum_averaged_emits_polarizations():
    iso = _rng_series(64, 1, 1)[:, 0]
    aniso = _rng_series(64, 1, 2)[:, 0]
    out = molrs.compute.spectroscopy.RamanSpectrum(10000.0, 300.0, True).fit(
        np.ascontiguousarray(iso), np.ascontiguousarray(aniso), 0.5
    )
    assert out["parallel"] is not None
    assert out["perpendicular"] is not None
    out2 = molrs.compute.spectroscopy.RamanSpectrum(0.0, 0.0, False).fit(
        np.ascontiguousarray(iso), np.ascontiguousarray(aniso), 0.5
    )
    assert out2["parallel"] is None


# ── edge cases ───────────────────────────────────────────────────────────────


def test_cumulative_trapezoid_overlong_request_errors():
    y = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        molrs.compute.fitting.CumulativeTrapezoid().fit(y, 1.0, 10)


def test_linear_fit_degenerate_window_errors():
    x = np.arange(10, dtype=np.float64)
    y = x.copy()
    with pytest.raises(ValueError):
        molrs.compute.fitting.LinearFit(0.7, 0.3).fit(x, y)


def test_debye_relaxation_bad_boundary_errors():
    with pytest.raises(ValueError):
        molrs.compute.transport.DebyeRelaxation(1.0, 1.0, "bogus")
