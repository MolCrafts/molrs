//! CL&Pol scaleLJ fragment parameters.
//!
//! Source: paduagroup/clandpol `fragment.ff` (`q`, `mu`) and `alpha.ff`
//! (per-atom polarizabilities summed per fragment). Units are elementary
//! charge, Debye, and Å³ respectively.

/// `(name, q, mu, alpha, polarizable)` rows used by CL&Pol scaleLJ.
pub const CLPOL_FRAGMENTS: &[(&str, f64, f64, f64, bool)] = &[
    ("c2c1im", 1.0, 1.1558, 12.383, false),
    ("bf4", -1.0, 0.0, 3.078, false),
    ("pf6", -1.0, 0.0, 4.987, false),
    ("ntf2", -1.0, 4.0070, 15.162, false),
    ("dca", -1.0, 0.8874, 8.268, false),
];
