//! Physical and engine constants (CODATA 2018 / SI-2019 exact where
//! applicable), and the conversion factors and scale factors molecular engines
//! and force fields define. Every numeric constant molrs uses lives here.
//!
//! Reference: SI Brochure, 9th edition (2019) for the exact defining
//! constants; CODATA 2018 recommended values,
//! <https://physics.nist.gov/cuu/Constants/>.
//!
//! # Examples
//!
//! ```
//! use molrs::core::constants::GAS_CONSTANT;
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

/// Speed of light in vacuum `c` (exact, SI-2019), in m/s.
pub const SPEED_OF_LIGHT: F = 299_792_458.0;

/// Second radiation constant `c₂ = h·c / k_B` (exact under SI-2019,
/// CODATA 2018), in cm·K: the `hcν̃ / k_BT` of a Boltzmann factor at
/// wavenumber `ν̃` (cm⁻¹).
pub const SECOND_RADIATION_CONSTANT: F = 1.438_776_877;

/// cm per m (exact).
pub const CENTIMETER_PER_METER: F = 100.0;

/// Å³ per cm³ (exact, `(10⁸)³`): a number density per cm³ is divided by it to
/// give one per Å³.
pub const ANGSTROM3_PER_CM3: F = 1e24;

/// kcal·mol⁻¹ per mdyne·Å (RDKit `MDYNE_A_TO_KCAL_MOL`, `Params.h`): MMFF's
/// force-constant unit conversion, to RDKit's digits.
pub const KCAL_MOL_PER_MDYNE_ANGSTROM: F = 143.9325;

/// Relative permittivity of vacuum, `ε_r = 1` — the medium OPLS, GAFF/AMBER
/// and MMFF were each parameterised in. A force field still has to choose
/// vacuum: every one that does states `dielectric` on its `coul/cut` style,
/// and the kernel has no default for it.
pub const VACUUM_DIELECTRIC: F = 1.0;

/// The Coulomb constant UFF's bond force-constant rule is written with,
/// kcal·Å·mol⁻¹·e⁻² (Rappé et al. 1992, Eq. 6, `k_ij = 664.12 Z_i Z_j / r³`
/// with `664.12 = 2 · 332.06`; RDKit `Params::G`).
pub const UFF_COULOMB: F = 332.06;

/// AMBER's 1-4 Coulomb **divisor** (`SCEE`): `coul_14 = 1 / AMBER_SCEE`.
///
/// The GAFF/GAFF2 typifier force field's 1-4 Coulomb scale, and the
/// force-field prmtop reader's value when `SCEE_SCALE_FACTOR` is absent
/// (pre-Amber-11 files, whose force field is AMBER's). A prmtop that carries
/// the section never touches it, so a GLYCAM file (`SCEE = 1.0`) reads
/// correctly.
pub const AMBER_SCEE: F = 1.2;

/// AMBER's 1-4 Lennard-Jones **divisor** (`SCNB`): `lj_14 = 1 / AMBER_SCNB`.
/// The GAFF/GAFF2 typifier force field's 1-4 LJ scale, and the force-field
/// prmtop reader's value when `SCNB_SCALE_FACTOR` is absent.
pub const AMBER_SCNB: F = 2.0;

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

    /// `c₂ = h·c / k_B` from the exact SI-2019 constants, in cm·K.
    #[test]
    fn second_radiation_constant_is_hc_over_kb() {
        const PLANCK: F = 6.626_070_15e-34;
        let c2 = PLANCK * SPEED_OF_LIGHT / BOLTZMANN * CENTIMETER_PER_METER;
        assert!((c2 - SECOND_RADIATION_CONSTANT).abs() < 1e-9, "{c2}");
    }
}
