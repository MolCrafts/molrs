//! AMBER *file-format* constants that are neither `gaff.dat` rows nor
//! properties of the universe.
//!
//! Hand-maintained sibling of [`super::mmff`] / [`super::clpol`] / [`super::uff`]:
//! not emitted by `scripts/gen_param_tables.py`. Shared by every consumer of an AMBER-family
//! topology (GAFF/GAFF2 typifier force fields, ff14SB/GLYCAM prmtops).

/// AMBER electrostatic constant, kcal·Å·mol⁻¹·e⁻².
///
/// Equal to `18.2223² = 332.05221729`, where `18.2223` is the charge factor
/// Amber writes into the prmtop `CHARGE` section (Amber
/// [FileFormats](https://ambermd.org/FileFormats.php), ParmEd
/// `AMBER_ELECTROSTATIC`). The structure reader de-scales by that literal
/// (`molrs/src/io/data/prmtop.rs` `CHARGE_CONVERSION_FACTOR`); this is the
/// implied Coulomb prefactor sander/pmemd evaluate. AmberTools `sander`
/// single-points on acetate, methylammonium and imidazolium recover the same
/// value to printed precision. molrs's CODATA constant `COULOMB_REAL =
/// 332.06371` differs by a relative 3.46e-5 — a documented cross-engine
/// offset, not an error.
pub(crate) const AMBER_COULOMB: f64 = 332.052_217_29;

/// AMBER 1-4 Coulomb **divisor** (`SCEE`).
///
/// Two roles, one number: (i) the GAFF/GAFF2 typifier force field's 1-4
/// Coulomb parameter (`coul_14 = 1 / AMBER_SCEE`); (ii) the prmtop reader's
/// fallback when `SCEE_SCALE_FACTOR` is absent (the format's pre-Amber-11
/// default). A prmtop that *carries* the section never touches this fallback,
/// so a GLYCAM file (`SCEE = 1.0`) reads correctly. Changing the GAFF role
/// must keep the format-default role.
pub(crate) const AMBER_SCEE: f64 = 1.2;

/// AMBER 1-4 Lennard-Jones **divisor** (`SCNB`).
///
/// Two roles, one number: (i) the GAFF/GAFF2 typifier force field's 1-4 LJ
/// parameter (`lj_14 = 1 / AMBER_SCNB`); (ii) the prmtop reader's fallback
/// when `SCNB_SCALE_FACTOR` is absent (the format's pre-Amber-11 default).
pub(crate) const AMBER_SCNB: f64 = 2.0;
