//! AMBER force-field constants that are not `gaff.dat` rows: the 1-4 divisors.
//!
//! Hand-maintained sibling of [`super::mmff`] / `clpol` / [`super::uff`]:
//! not emitted by `scripts/gen_param_tables.py`. Shared by every consumer of an
//! AMBER-family force field (GAFF/GAFF2 typifier force fields, ff14SB/GLYCAM
//! prmtops). AMBER's Coulomb constant and prmtop charge factor are engine unit
//! facts and live in `crate::units::constants` (`AMBER_COULOMB`,
//! `AMBER_CHARGE_FACTOR`).

/// AMBER 1-4 Coulomb **divisor** (`SCEE`).
///
/// Two roles, one number: (i) the GAFF/GAFF2 typifier force field's 1-4
/// Coulomb parameter (`coul_14 = 1 / AMBER_SCEE`); (ii) the force-field prmtop
/// reader's value when `SCEE_SCALE_FACTOR` is absent (pre-Amber-11 files, whose
/// force field is AMBER's). A prmtop that *carries* the section never touches
/// it, so a GLYCAM file (`SCEE = 1.0`) reads correctly. The structure reader
/// (`io::data::prmtop`) applies no default: 1-4 weighting is force-field
/// knowledge.
pub const AMBER_SCEE: f64 = 1.2;

/// AMBER 1-4 Lennard-Jones **divisor** (`SCNB`).
///
/// Two roles, one number: (i) the GAFF/GAFF2 typifier force field's 1-4 LJ
/// parameter (`lj_14 = 1 / AMBER_SCNB`); (ii) the force-field prmtop
/// reader's value when `SCNB_SCALE_FACTOR` is absent.
pub const AMBER_SCNB: f64 = 2.0;
