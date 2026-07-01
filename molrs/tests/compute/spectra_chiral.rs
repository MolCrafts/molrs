//! Chiral / advanced vibrational spectra (VCD, ROA, resonance Raman) —
//! travis-parity-08.
//!
//! Synthetic trajectories exercise the defining physical signatures: the
//! enantiomer sign law (VCD/ROA flip sign between mirror images, vanish for
//! achiral systems), peak-position consistency with the existing IR/Raman
//! frequency axis, and reuse of the shared `window_and_fft` spectral grid.

use molrs::Frame;
use molrs::compute::traits::{Compute, Fit};
use molrs::compute::{
    IRFlux, IRSpectrum, RamanSpectrum, RamanTensor, ResonanceRamanSpectrum, ResonanceRamanTensor,
    RoaCrossTensor, RoaSpectrum, VcdCrossFlux, VcdSpectrum,
};
use ndarray::{Array1, Array2};

const DT: f64 = 0.5;
const N: usize = 512;
const RES: usize = 120;
const FREQ_THZ: f64 = 12.0;

fn no_frames() -> Vec<&'static Frame> {
    Vec::new()
}

/// A single oscillating Cartesian/Voigt column: `x(t) = amp·sin(2π·f·t)`.
fn sine_col(n: usize, amp: f64) -> Vec<f64> {
    (0..n)
        .map(|t| {
            let tf = t as f64 * DT;
            amp * (2.0 * std::f64::consts::PI * FREQ_THZ * 1e-3 * tf).sin()
        })
        .collect()
}

/// `(n, k)` series with each column filled by `amp[c]·sin(ωt)`.
fn sine_series(n: usize, amps: &[f64]) -> Array2<f64> {
    let k = amps.len();
    let mut a = Array2::zeros((n, k));
    for (c, &amp) in amps.iter().enumerate() {
        let col = sine_col(n, amp);
        for t in 0..n {
            a[[t, c]] = col[t];
        }
    }
    a
}

/// Index of the dominant spectral peak (by |intensity|), skipping bin 0 and the
/// tail (where windowing artefacts live).
fn peak_bin(intensities: &Array1<f64>) -> usize {
    let end = intensities.len().saturating_sub(3);
    intensities
        .iter()
        .enumerate()
        .skip(1)
        .take(end.saturating_sub(1))
        .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

// ── ac-001: VCD enantiomer sign law + achiral zero ───────────────────────────

#[test]
fn vcd_enantiomers_are_equal_and_opposite() {
    let mu = sine_series(N, &[0.0, 0.0, 1.0]); // electric dipole along z
    let m_plus = sine_series(N, &[0.0, 0.0, 1.0]); // magnetic dipole (enantiomer +)
    let m_minus = sine_series(N, &[0.0, 0.0, -1.0]); // mirror image: m → −m

    let acf_p = VcdCrossFlux
        .compute(&no_frames(), (&mu, &m_plus, DT, RES))
        .unwrap();
    let acf_m = VcdCrossFlux
        .compute(&no_frames(), (&mu, &m_minus, DT, RES))
        .unwrap();
    let sp = VcdSpectrum.fit((&acf_p.acf, DT)).unwrap();
    let sm = VcdSpectrum.fit((&acf_m.acf, DT)).unwrap();

    // Enantiomers: equal-and-opposite, exactly (linear pipeline).
    let scale = sp.intensities.iter().map(|x| x.abs()).fold(0.0, f64::max);
    assert!(scale > 1e-6, "VCD signal should be non-trivial");
    for (a, b) in sp.intensities.iter().zip(sm.intensities.iter()) {
        assert!(
            (a + b).abs() < 1e-9 * scale,
            "VCD not sign-flipped: {a} vs {b}"
        );
    }
}

#[test]
fn vcd_achiral_system_is_near_zero() {
    let mu = sine_series(N, &[0.0, 0.0, 1.0]);
    let m_zero = Array2::zeros((N, 3)); // achiral: no magnetic moment
    let acf = VcdCrossFlux
        .compute(&no_frames(), (&mu, &m_zero, DT, RES))
        .unwrap();
    let spec = VcdSpectrum.fit((&acf.acf, DT)).unwrap();
    let peak = spec.intensities.iter().map(|x| x.abs()).fold(0.0, f64::max);
    assert!(peak < 1e-12, "achiral VCD should vanish, peak |I| = {peak}");
}

// ── ac-002: ROA enantiomer sign law + achiral zero ───────────────────────────

#[test]
fn roa_enantiomers_are_sign_flipped() {
    let alpha = sine_series(N, &[1.0, 0.7, 0.4, 0.3, 0.2, 0.1]); // electric polarizability
    let g_plus = sine_series(N, &[1.0, 0.7, 0.4, 0.3, 0.2, 0.1]); // optical-activity G′ (+)
    let g_minus = sine_series(N, &[-1.0, -0.7, -0.4, -0.3, -0.2, -0.1]); // mirror: G′ → −G′

    let p = RoaCrossTensor
        .compute(&no_frames(), (&alpha, &g_plus, DT, RES))
        .unwrap();
    let m = RoaCrossTensor
        .compute(&no_frames(), (&alpha, &g_minus, DT, RES))
        .unwrap();
    let sp = RoaSpectrum {
        incident_frequency_cm1: 0.0,
        temperature_k: 0.0,
        averaged: false,
    }
    .fit((&p.acf_iso, &p.acf_aniso, DT))
    .unwrap();
    let sm = RoaSpectrum {
        incident_frequency_cm1: 0.0,
        temperature_k: 0.0,
        averaged: false,
    }
    .fit((&m.acf_iso, &m.acf_aniso, DT))
    .unwrap();

    let scale = sp.isotropic.iter().map(|x| x.abs()).fold(0.0, f64::max);
    assert!(scale > 1e-6, "ROA signal should be non-trivial");
    for (a, b) in sp.isotropic.iter().zip(sm.isotropic.iter()) {
        assert!((a + b).abs() < 1e-9 * scale, "ROA iso not sign-flipped");
    }
}

#[test]
fn roa_achiral_system_is_near_zero() {
    let alpha = sine_series(N, &[1.0, 0.7, 0.4, 0.3, 0.2, 0.1]);
    let g_zero = Array2::zeros((N, 6)); // achiral: no optical activity
    let r = RoaCrossTensor
        .compute(&no_frames(), (&alpha, &g_zero, DT, RES))
        .unwrap();
    let spec = RoaSpectrum {
        incident_frequency_cm1: 0.0,
        temperature_k: 0.0,
        averaged: false,
    }
    .fit((&r.acf_iso, &r.acf_aniso, DT))
    .unwrap();
    let peak = spec.isotropic.iter().map(|x| x.abs()).fold(0.0, f64::max);
    assert!(peak < 1e-12, "achiral ROA should vanish, peak |I| = {peak}");
}

// ── ac-003: peak positions coincide with IR / Raman ──────────────────────────

#[test]
fn vcd_and_roa_peaks_coincide_with_ir_and_raman() {
    // Same vibrational frequency drives every moment → same cm⁻¹ peak.
    let mu = sine_series(N, &[0.0, 0.0, 1.0]);
    let mag = sine_series(N, &[0.0, 0.0, 1.0]);
    let alpha = sine_series(N, &[1.0, 0.7, 0.4, 0.3, 0.2, 0.1]);
    let g = sine_series(N, &[1.0, 0.7, 0.4, 0.3, 0.2, 0.1]);

    let ir = IRFlux.compute(&no_frames(), (&mu, DT, RES)).unwrap();
    let ir_spec = IRSpectrum.fit((&ir.acf, DT)).unwrap();
    let ir_peak = peak_bin(&ir_spec.intensities);

    let vcd = VcdCrossFlux
        .compute(&no_frames(), (&mu, &mag, DT, RES))
        .unwrap();
    let vcd_peak = peak_bin(&VcdSpectrum.fit((&vcd.acf, DT)).unwrap().intensities);

    let raman = RamanTensor
        .compute(&no_frames(), (&alpha, DT, RES))
        .unwrap();
    let raman_spec = RamanSpectrum {
        incident_frequency_cm1: 0.0,
        temperature_k: 0.0,
        averaged: false,
    }
    .fit((&raman.acf_iso, &raman.acf_aniso, DT))
    .unwrap();
    let raman_peak = peak_bin(&raman_spec.isotropic);

    let roa = RoaCrossTensor
        .compute(&no_frames(), (&alpha, &g, DT, RES))
        .unwrap();
    let roa_peak = peak_bin(
        &RoaSpectrum {
            incident_frequency_cm1: 0.0,
            temperature_k: 0.0,
            averaged: false,
        }
        .fit((&roa.acf_iso, &roa.acf_aniso, DT))
        .unwrap()
        .isotropic,
    );

    let rr = ResonanceRamanTensor
        .compute(&no_frames(), (&alpha, DT, RES))
        .unwrap();
    let rr_peak = peak_bin(
        &ResonanceRamanSpectrum {
            incident_frequency_cm1: 0.0,
            temperature_k: 0.0,
            averaged: false,
        }
        .fit((&rr.acf_iso, &rr.acf_aniso, DT))
        .unwrap()
        .isotropic,
    );

    // All within one frequency bin of the IR peak.
    for (name, p) in [
        ("vcd", vcd_peak),
        ("raman", raman_peak),
        ("roa", roa_peak),
        ("resonance-raman", rr_peak),
    ] {
        assert!(
            (p as i64 - ir_peak as i64).abs() <= 1,
            "{name} peak bin {p} != IR peak bin {ir_peak}"
        );
    }
}

// ── ac-004: windowing + frequency grid reuse the shared helper ───────────────

#[test]
fn vcd_reuses_the_ir_frequency_grid() {
    // Identical input length/dt ⇒ identical cm⁻¹ grid (both call window_and_fft).
    let mu = sine_series(N, &[0.0, 0.0, 1.0]);
    let mag = sine_series(N, &[0.0, 0.0, 1.0]);
    let ir = IRFlux.compute(&no_frames(), (&mu, DT, RES)).unwrap();
    let vcd = VcdCrossFlux
        .compute(&no_frames(), (&mu, &mag, DT, RES))
        .unwrap();
    assert_eq!(ir.acf.len(), vcd.acf.len());
    let ir_grid = IRSpectrum.fit((&ir.acf, DT)).unwrap().frequencies_cm1;
    let vcd_grid = VcdSpectrum.fit((&vcd.acf, DT)).unwrap().frequencies_cm1;
    assert_eq!(ir_grid, vcd_grid, "VCD and IR must share the cm⁻¹ grid");
}

#[test]
fn roa_reuses_the_raman_frequency_grid() {
    let alpha = sine_series(N, &[1.0, 0.7, 0.4, 0.3, 0.2, 0.1]);
    let g = sine_series(N, &[1.0, 0.7, 0.4, 0.3, 0.2, 0.1]);
    let raman = RamanTensor
        .compute(&no_frames(), (&alpha, DT, RES))
        .unwrap();
    let roa = RoaCrossTensor
        .compute(&no_frames(), (&alpha, &g, DT, RES))
        .unwrap();
    let raman_grid = RamanSpectrum {
        incident_frequency_cm1: 0.0,
        temperature_k: 0.0,
        averaged: false,
    }
    .fit((&raman.acf_iso, &raman.acf_aniso, DT))
    .unwrap()
    .frequencies_cm1;
    let roa_grid = RoaSpectrum {
        incident_frequency_cm1: 0.0,
        temperature_k: 0.0,
        averaged: false,
    }
    .fit((&roa.acf_iso, &roa.acf_aniso, DT))
    .unwrap()
    .frequencies_cm1;
    assert_eq!(
        raman_grid, roa_grid,
        "ROA and Raman must share the cm⁻¹ grid"
    );
}

// ── ac-005: edge cases ───────────────────────────────────────────────────────

#[test]
fn zero_length_series_is_typed_error() {
    use molrs::compute::error::ComputeError;
    let empty3 = Array2::<f64>::zeros((0, 3));
    assert!(matches!(
        VcdCrossFlux.compute(&no_frames(), (&empty3, &empty3, DT, RES)),
        Err(ComputeError::EmptyInput)
    ));
    let empty_acf: Array1<f64> = Array1::from_vec(vec![]);
    assert!(matches!(
        VcdSpectrum.fit((&empty_acf, DT)),
        Err(ComputeError::EmptyInput)
    ));
}

#[test]
fn single_molecule_input_is_well_defined() {
    // One molecule's moment series (no cross-molecule terms by construction).
    let mu = sine_series(N, &[0.0, 0.0, 1.0]);
    let mag = sine_series(N, &[0.0, 0.0, 0.5]);
    let acf = VcdCrossFlux
        .compute(&no_frames(), (&mu, &mag, DT, RES))
        .unwrap();
    let spec = VcdSpectrum.fit((&acf.acf, DT)).unwrap();
    assert!(spec.intensities.iter().all(|x| x.is_finite()));
    assert_eq!(spec.frequencies_cm1.len(), spec.intensities.len());
}
