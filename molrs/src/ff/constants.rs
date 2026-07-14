//! Numeric constants shared by the MMFF parameter resolver
//! ([`crate::ff::mmff::params`]) and the potential kernels
//! ([`crate::ff::potential`]). Values match RDKit's MMFF94 implementation exactly
//! — keeping them in one place prevents the rounding drift that creeps in when
//! the same literal is re-typed per module.

/// mdyne·Å → kcal/mol (RDKit `MDYNE_A_TO_KCAL_MOL`, `Params.h`).
pub(crate) const MDYNE_A_TO_KCAL: f64 = 143.9325;

/// Coulomb constant `e²/(4π·ε₀)` in kcal·Å·mol⁻¹·e⁻² (RDKit `Nonbonded.cpp`).
/// Distinct from the CODATA-derived [`molrs::units::constants::COULOMB_REAL`]
/// used by the generic Coulomb pair potential — RDKit rounds it differently and
/// the MMFF parity tests pin this exact value.
pub(crate) const COULOMB_MMFF: f64 = 332.0716;

/// Electrostatic buffering distance δ in Å (RDKit `calcEleEnergy`).
pub(crate) const ELE_BUFFER: f64 = 0.05;

/// degrees → radians (RDKit `DEG2RAD`).
///
/// The MMFF tables store reference angles in degrees; molrs is radians
/// internally, so the resolver converts at that boundary. (`RAD2DEG` lived here
/// too, for the deleted bespoke energy kernels — which worked in degrees and
/// converted back on every gradient. The generic kernels never leave radians.)
pub(crate) const DEG2RAD: f64 = std::f64::consts::PI / 180.0;
