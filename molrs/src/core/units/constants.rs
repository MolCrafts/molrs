//! Physical constants (CODATA 2018 / SI-2019 exact where applicable).
//!
//! Reference: SI Brochure, 9th edition (2019) for the exact defining
//! constants; CODATA 2018 recommended values,
//! <https://physics.nist.gov/cuu/Constants/>.
//!
//! # Examples
//!
//! ```
//! use molrs::units::constants::GAS_CONSTANT;
//!
//! // R = N_A · k_B = 8.314 462 618... J/(mol·K), exact under SI-2019.
//! assert!((GAS_CONSTANT - 8.314_462_618_153_24).abs() < 1e-12);
//! ```

use crate::op::types::F;

/// Avogadro constant `N_A` (exact, SI-2019), in mol⁻¹.
pub const AVOGADRO: F = 6.022_140_76e23;

/// Boltzmann constant `k_B` (exact, SI-2019), in J/K.
pub const BOLTZMANN: F = 1.380_649e-23;

/// Molar gas constant `R = N_A · k_B` (exact, SI-2019), in J/(mol·K).
pub const GAS_CONSTANT: F = AVOGADRO * BOLTZMANN;

/// Elementary charge `e` (exact, SI-2019), in coulombs.
pub const ELEMENTARY_CHARGE: F = 1.602_176_634e-19;

/// Coulomb constant `k_e = 1/(4π·ε₀)` in MD "real" units — LAMMPS `real`'s `qqr2e`
/// (kcal·Å·mol⁻¹·e⁻²), CODATA-derived. It is what the dielectric/conductivity
/// analyses use, and what the OPLS and LAMMPS force fields declare on their
/// `pair/coul/cut` style.
///
/// It is **not** a constant of the Coulomb kernel. MMFF rounds it differently
/// (Halgren's 332.0716, which lives in `ff::params::mmff::MMFF_ELE_STYLE`), and the
/// 2.4e-5 difference is above the RDKit parity tolerance on caffeine. Both values
/// are correct: the *force field* chooses one and states it on its `pair/coul/cut`
/// style, and the kernel has no default.
pub const COULOMB_REAL: F = 332.063_71;

/// Coulomb constant in LAMMPS `metal` units (eV·Å·e⁻²): LAMMPS's own `qqr2e`
/// for that unit style, so a `metal` force field read from LAMMPS prices its
/// electrostatics as LAMMPS does.
pub const COULOMB_METAL: F = 14.399_645;

/// Coulomb constant AMBER (sander/pmemd) evaluates, kcal·Å·mol⁻¹·e⁻².
///
/// Equal to `18.2223² = 332.05221729`, where `18.2223` is
/// [`AMBER_CHARGE_FACTOR`], the factor Amber writes into the prmtop `CHARGE`
/// section (Amber [FileFormats](https://ambermd.org/FileFormats.php), ParmEd
/// `AMBER_ELECTROSTATIC`). AmberTools `sander` single-points on acetate,
/// methylammonium and imidazolium recover it to printed precision.
/// [`COULOMB_REAL`] differs by a relative 3.46e-5 — a documented cross-engine
/// offset, not an error.
pub const AMBER_COULOMB: F = 332.052_217_29;

/// AMBER's prmtop charge factor: `CHARGE` stores `q · 18.2223`.
///
/// The literal is Amber's own (`18.2223² = `[`AMBER_COULOMB`]); do not re-derive
/// it from `√`[`COULOMB_REAL`], which would shift every charge by ~1.7e-5.
pub const AMBER_CHARGE_FACTOR: F = 18.2223;

/// CHARMM's `CCELEC`, kcal·Å·mol⁻¹·e⁻²: the Coulomb constant CHARMM (and a
/// chamber prmtop, which stores `CHARGE` as `q·√332.0716`, ParmEd's
/// `CHARMM_ELECTROSTATIC`) evaluates.
pub const CHARMM_COULOMB: F = 332.0716;

/// OpenMM's Coulomb constant `ONE_4PI_EPS0` = 138.93545764438198
/// kJ·nm·mol⁻¹·e⁻², here in kcal·Å·mol⁻¹·e⁻² — the constant every OpenMM
/// energy is computed with ([`COULOMB_REAL`]'s 332.06371 × (1 + 9.9·10⁻⁹)).
pub const OPENMM_COULOMB: F = 138.935_457_644_381_98 * ANGSTROM_PER_NM / KJ_PER_KCAL;

/// GROMACS's Coulomb constant `ONE_4PI_EPS0`, 1/(4π ε₀) from CODATA 2018 in
/// GROMACS's own expression (`units.h`), 138.93545764438196 kJ·nm·mol⁻¹·e⁻²
/// — one ulp below OpenMM's — here in kcal·Å·mol⁻¹·e⁻².
pub const GROMACS_COULOMB: F = 138.935_457_644_381_96 * ANGSTROM_PER_NM / KJ_PER_KCAL;

/// kJ per kcal (the thermochemical calorie, exact): kJ/mol = kcal/mol × this.
pub const KJ_PER_KCAL: F = 4.184;

/// Å per nm (exact): lengths in nm are multiplied by it, lengths in Å divided.
pub const ANGSTROM_PER_NM: F = 10.0;

/// Å per bohr (the Bohr radius, CODATA 2014): the Gaussian cube format's
/// length unit.
pub const ANGSTROM_PER_BOHR: F = 0.529_177_210_67;

/// Boltzmann constant in MD "real" units, kcal·mol⁻¹·K⁻¹.
pub const BOLTZMANN_REAL: F = 1.987_204_258_640_83e-3;

/// 1 ångström expressed in metres (SI length-unit conversion factor).
pub const ANGSTROM_M: F = 1e-10;

/// 1 femtosecond expressed in seconds (SI time-unit conversion factor).
///
/// Project analysis time unit (science.md / LAMMPS `real`).
pub const FEMTOSECOND_S: F = 1e-15;

#[cfg(test)]
mod tests {
    use super::*;

    /// AMBER's Coulomb constant is its prmtop charge factor squared, 3.46e-5
    /// below CODATA's `real` constant.
    #[test]
    fn amber_coulomb_is_its_charge_factor_squared() {
        assert!((AMBER_COULOMB - AMBER_CHARGE_FACTOR.powi(2)).abs() < 1e-9);
        let rel = (COULOMB_REAL - AMBER_COULOMB) / COULOMB_REAL;
        assert!((rel - 3.4610e-5).abs() < 1e-8, "relative offset {rel}");
    }
}
