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

use crate::types::F;

/// Avogadro constant `N_A` (exact, SI-2019), in mol⁻¹.
pub const AVOGADRO: F = 6.022_140_76e23;

/// Boltzmann constant `k_B` (exact, SI-2019), in J/K.
pub const BOLTZMANN: F = 1.380_649e-23;

/// Molar gas constant `R = N_A · k_B` (exact, SI-2019), in J/(mol·K).
pub const GAS_CONSTANT: F = AVOGADRO * BOLTZMANN;

/// Elementary charge `e` (exact, SI-2019), in coulombs.
pub const ELEMENTARY_CHARGE: F = 1.602_176_634e-19;

/// Coulomb constant `k_e = 1/(4π·ε₀)` in MD "real" units
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

/// Boltzmann constant in MD "real" units, kcal·mol⁻¹·K⁻¹.
pub const BOLTZMANN_REAL: F = 1.987_204_258_640_83e-3;

/// 1 ångström expressed in metres (SI length-unit conversion factor).
pub const ANGSTROM_M: F = 1e-10;

/// 1 picosecond expressed in seconds (SI time-unit conversion factor).
pub const PICOSECOND_S: F = 1e-12;
