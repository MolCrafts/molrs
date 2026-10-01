//! Ryckaert–Bellemans ↔ OPLS Fourier torsion conversions.
//!
//! GROMACS (dihedral funct 3) and OpenMM `<RBTorsionForce>` write a torsion in
//! the Ryckaert–Bellemans form, `V = Σₙ₌₀⁵ Cₙ cosⁿψ` with `ψ = φ − 180°`
//! (Ryckaert & Bellemans, *Faraday Discuss. Chem. Soc.* 66, 95 (1978),
//! DOI 10.1039/DC9786600095). The molrs `dihedral/opls` kernel evaluates the
//! OPLS Fourier form
//! `V = ½[F₁(1+cosφ) + F₂(1−cos2φ) + F₃(1+cos3φ) + F₄(1−cos4φ)]`
//! (Jorgensen, Maxwell, Tirado-Rives, *JACS* 118, 11225 (1996),
//! DOI 10.1021/ja9621760).
//!
//! The exact relations (GROMACS reference manual, Eqs. 200–201) are
//!
//! ```text
//! RB → Fourier:  F₁ = −2C₁ − 1.5C₃   F₂ = −C₂ − C₄   F₃ = −C₃/2   F₄ = −C₄/4
//! Fourier → RB:  C₀ = F₂ + ½(F₁+F₃)   C₁ = ½(−F₁+3F₃)   C₂ = −F₂+4F₄
//!                C₃ = −2F₃             C₄ = −4F₄          C₅ = 0
//! ```
//!
//! # Representability
//!
//! An RB row has a Fourier counterpart iff `C₅ = 0` and `ΣCₙ = 0`: the
//! Fourier form has no `cos⁵` term, and it vanishes at `φ = 180°`, where the
//! RB form equals `ΣCₙ`. Both are checked to within [`RB_TOL`].
//!
//! # Units
//!
//! Both functions are unit-agnostic linear maps: the output is in the input's
//! energy unit. The readers and writers pass kJ/mol and convert to or from
//! kcal/mol (÷ or × 4.184) at their own boundary. [`RB_TOL`] is in kJ/mol,
//! the unit RB rows are published in.
//!
//! These are free functions because the math has no owning type. They live in
//! `ff::forcefield`, not `ff::potential`, so that readers never import from a
//! kernel module (see `mixing.rs`).

/// Tolerance, in kJ/mol, on `|C₅|` and `|ΣCₙ|` for an RB row to count as
/// representable in the OPLS Fourier form.
///
/// GROMACS prints RB coefficients with 5 decimals, so six rounded
/// coefficients sum to at most 6 · 0.5e-5 = 3e-5 kJ/mol; 1e-4 kJ/mol admits
/// that rounding and nothing larger.
pub(crate) const RB_TOL: f64 = 1e-4;

/// Ryckaert–Bellemans `[C₀..C₅]` → OPLS Fourier `[F₁, F₂, F₃, F₄]`, same
/// energy unit in and out (kJ/mol at every current call site).
///
/// # Errors
///
/// Returns an error naming the six coefficients when `|C₅| > RB_TOL` or
/// `|ΣCₙ| > RB_TOL`: such a row has no OPLS Fourier counterpart.
pub(crate) fn rb_to_opls(c: [f64; 6]) -> Result<[f64; 4], String> {
    let sum: f64 = c.iter().sum();
    if c[5].abs() > RB_TOL || sum.abs() > RB_TOL {
        return Err(format!(
            "Ryckaert-Bellemans row {c:?} (C5 = {}, sum = {sum}) has no OPLS Fourier \
             form: it needs C5 = 0 and C0+...+C5 = 0 to within {RB_TOL} kJ/mol",
            c[5]
        ));
    }
    let [_, c1, c2, c3, c4, _] = c;
    Ok([-2.0 * c1 - 1.5 * c3, -c2 - c4, -0.5 * c3, -0.25 * c4])
}

/// OPLS Fourier `[F₁, F₂, F₃, F₄]` → Ryckaert–Bellemans `[C₀..C₅]`, same
/// energy unit in and out. The result always has `C₅ = 0` and `ΣCₙ = 0`.
pub(crate) fn opls_to_rb([f1, f2, f3, f4]: [f64; 4]) -> [f64; 6] {
    [
        f2 + 0.5 * (f1 + f3),
        0.5 * (-f1 + 3.0 * f3),
        -f2 + 4.0 * f4,
        -2.0 * f3,
        -4.0 * f4,
        0.0,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close<const N: usize>(got: [f64; N], want: [f64; N], tol: f64) {
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (g - w).abs() < tol,
                "[{i}]: got {g}, want {w} (all: {got:?})"
            );
        }
    }

    /// GROMACS OPLS-AA HC-CT-CT-HC, funct 3 (kJ/mol):
    /// `0.62760 1.88280 0.00000 -2.51040 0.00000 0.00000`. By hand:
    /// F1 = −2·1.8828 − 1.5·(−2.5104) = 0; F2 = −0 − 0 = 0;
    /// F3 = −(−2.5104)/2 = 1.2552; F4 = −0/4 = 0.
    #[test]
    fn rb_to_opls_inverts_the_hc_ct_ct_hc_row() {
        let f = rb_to_opls([0.62760, 1.88280, 0.0, -2.51040, 0.0, 0.0]).expect("representable");
        assert_close(f, [0.0, 0.0, 1.2552, 0.0], 1e-12);
    }

    /// The row formerly tested in `readers/opls.rs`, in kJ/mol throughout:
    /// ΣC = 0.75312 + 2.25936 − 3.01248 = 0; F1 = −4.51872 + 4.51872 = 0;
    /// F3 = 3.01248 / 2 = 1.50624.
    #[test]
    fn rb_to_opls_returns_kj_per_mol_without_unit_conversion() {
        let f = rb_to_opls([0.75312, 2.25936, 0.0, -3.01248, 0.0, 0.0]).expect("representable");
        assert_close(f, [0.0, 0.0, 1.50624, 0.0], 1e-12);
    }

    /// F = (0, 0, 1.2552, 0): C0 = 0 + ½(0 + 1.2552) = 0.6276;
    /// C1 = ½(−0 + 3·1.2552) = 1.8828; C2 = −0 + 0 = 0; C3 = −2·1.2552 = −2.5104;
    /// C4 = 0; C5 = 0.
    #[test]
    fn opls_to_rb_reproduces_the_hc_ct_ct_hc_row() {
        let c = opls_to_rb([0.0, 0.0, 1.2552, 0.0]);
        assert_close(c, [0.62760, 1.88280, 0.0, -2.51040, 0.0, 0.0], 1e-12);
    }

    /// F = (1, −2, 3, −4): C0 = −2 + ½(1 + 3) = 0; C1 = ½(−1 + 9) = 4;
    /// C2 = 2 − 16 = −14; C3 = −6; C4 = 16; C5 = 0.
    #[test]
    fn opls_to_rb_applies_every_fourier_term() {
        let c = opls_to_rb([1.0, -2.0, 3.0, -4.0]);
        assert_close(c, [0.0, 4.0, -14.0, -6.0, 16.0, 0.0], 1e-12);
    }

    /// Fourier → RB → Fourier is the identity on four-term inputs.
    #[test]
    fn round_trip_through_rb_is_the_identity_on_fourier_terms() {
        for f in [
            [1.0, -2.0, 3.0, -4.0],
            [5.4392, -0.2092, 0.8368, 0.4184],
            [0.0, 0.0, 1.2552, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ] {
            let back = rb_to_opls(opls_to_rb(f)).expect("opls_to_rb output is representable");
            assert_close(back, f, 1e-12);
        }
    }

    /// ΣCₙ = 1 is a constant offset the Fourier form (V(180°) = 0) cannot hold.
    #[test]
    fn rb_row_with_a_nonzero_sum_is_an_error() {
        let err = rb_to_opls([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]).expect_err("ΣC = 1");
        assert!(
            err.contains('1'),
            "error should name the coefficients: {err}"
        );
    }

    /// C5 = 0.1 with ΣC = 0 (C0 = −0.1): cos⁵ has no Fourier counterpart.
    #[test]
    fn rb_row_with_a_nonzero_c5_is_an_error() {
        let err = rb_to_opls([-0.1, 0.0, 0.0, 0.0, 0.0, 0.1]).expect_err("C5 = 0.1");
        assert!(
            err.contains("0.1"),
            "error should name the coefficients: {err}"
        );
    }

    /// GROMACS prints 5 decimals, so rounding leaves ΣC up to ~3e-5 kJ/mol:
    /// C3 = −2.51037 gives ΣC = 3e-5, inside RB_TOL = 1e-4.
    #[test]
    fn rb_row_off_by_printed_rounding_is_accepted() {
        let f = rb_to_opls([0.62760, 1.88280, 0.0, -2.51037, 0.0, 0.0]).expect("within RB_TOL");
        assert_close(
            f,
            [-2.0 * 1.8828 + 1.5 * 2.51037, 0.0, 2.51037 / 2.0, 0.0],
            1e-12,
        );
    }

    /// ΣC = 2e-4 kJ/mol is twice RB_TOL: refused.
    #[test]
    fn rb_row_off_by_more_than_rb_tol_is_an_error() {
        const { assert!(RB_TOL < 2e-4) }
        assert!(rb_to_opls([0.62780, 1.88280, 0.0, -2.51040, 0.0, 0.0]).is_err());
    }

    /// |C5| = 5e-5 kJ/mol (ΣC = 0) is below RB_TOL: accepted.
    #[test]
    fn rb_row_with_a_c5_below_rb_tol_is_accepted() {
        const { assert!(RB_TOL > 5e-5) }
        assert!(rb_to_opls([0.62755, 1.88280, 0.0, -2.51040, 0.0, 0.00005]).is_ok());
    }
}
