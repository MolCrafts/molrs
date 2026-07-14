//! Numeric constants shared by the MMFF parameter resolver
//! ([`crate::ff::mmff::params`]) and the potential kernels
//! ([`crate::ff::potential`]). Values match RDKit's MMFF94 implementation exactly
//! — keeping them in one place prevents the rounding drift that creeps in when
//! the same literal is re-typed per module.

/// mdyne·Å → kcal/mol (RDKit `MDYNE_A_TO_KCAL_MOL`, `Params.h`).
pub(crate) const MDYNE_A_TO_KCAL: f64 = 143.9325;

// MMFF's Coulomb constant (Halgren's 332.0716) and its electrostatic buffering
// distance (0.05 Å) used to live here. They are gone — not deleted, *relocated*
// to `ff::params::mmff::MMFF_ELE_STYLE`, beside MMFF's other parameters.
//
// A constant in this file claims to be a property of the universe. Those two are
// properties of ONE force field: OPLS and LAMMPS evaluate the same Coulomb kernel
// with CODATA's 332.06371 (`molrs::units::constants::COULOMB_REAL`), and the 2.4e-5
// difference is above the RDKit parity tolerance on caffeine. Both are correct; the
// force field decides — so they reach the kernel through the STYLE, which is the
// only way a kernel is allowed to learn them.

/// Relative permittivity of vacuum, `D = 1` — the medium OPLS, GAFF/AMBER and MMFF
/// were each parameterised in.
///
/// This one genuinely *is* a property of the universe (it is 1 by definition), which
/// is why it may live here while Halgren's Coulomb constant may not. A force field
/// still has to **choose** vacuum: every force field that does states `dielectric` on
/// its `coul/cut` style, and the kernel has no default for it.
pub(crate) const VACUUM_DIELECTRIC: f64 = 1.0;

/// degrees → radians (RDKit `DEG2RAD`).
///
/// The MMFF tables store reference angles in degrees; molrs is radians
/// internally, so the resolver converts at that boundary. (`RAD2DEG` lived here
/// too, for the deleted bespoke energy kernels — which worked in degrees and
/// converted back on every gradient. The generic kernels never leave radians.)
pub(crate) const DEG2RAD: f64 = std::f64::consts::PI / 180.0;
