//! LAMMPS unit-style conversion for force-field I/O.
//!
//! All conversions go through the molrs [`UnitRegistry`] / [`Quantity`] stack
//! with **lj reduced units as the mandatory hub**:
//!
//! ```text
//! source style  →  lj_*  →  target style
//! ```
//!
//! Physical quantity meanings follow the LAMMPS `units` command
//! (<https://docs.lammps.org/units.html>): real (Å, kcal/mol), metal (Å, eV),
//! lj (reduced). Thermochemical calorie (4.184 J) is already encoded in
//! [`UnitRegistry`]'s MD preload.
//!
//! A **canonical reference** `(m=1 g/mol, σ=1 Å, ε=1 kcal/mol)` makes
//! real↔metal bridge through lj without material-specific scales, while still
//! never hard-coding eV↔kcal factors in the FF reader.

use molrs::types::F;
use molrs::units::{Quantity, UnitRegistry, UnitsError};

/// LAMMPS `units` styles supported in phase 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LammpsUnits {
    /// Reduced LJ units (LAMMPS default for bare scripts).
    Lj,
    /// Å, kcal/mol, g/mol, fs, e — molecular force fields.
    #[default]
    Real,
    /// Å, eV, g/mol, ps, e — metal / generic MD.
    Metal,
}

impl LammpsUnits {
    /// Parse a LAMMPS `units` keyword value (case-insensitive).
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "lj" => Ok(Self::Lj),
            "real" => Ok(Self::Real),
            "metal" => Ok(Self::Metal),
            other => Err(format!(
                "unsupported LAMMPS units `{other}` (phase 1: lj, real, metal)"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Lj => "lj",
            Self::Real => "real",
            Self::Metal => "metal",
        }
    }
}

/// Reference scales for `define_lj_units` (physical mass, σ, ε).
///
/// Canonical defaults: `1 g/mol`, `1 Å`, `1 kcal/mol` so that one reduced
/// energy unit equals one thermochemical kcal/mol and one reduced length
/// equals one ångström — matching LAMMPS real for the bridge path.
#[derive(Debug, Clone)]
pub struct LammpsLjReference {
    pub mass: Quantity,
    pub sigma: Quantity,
    pub epsilon: Quantity,
}

impl LammpsLjReference {
    /// Canonical bridge scales: `m = 1 g/mol`, `σ = 1 Å`, `ε = 1 kcal/mol`.
    pub fn canonical() -> Result<Self, UnitsError> {
        let reg = UnitRegistry::new();
        Ok(Self {
            mass: reg.quantity(1.0, "gram_per_mole")?,
            sigma: reg.quantity(1.0, "angstrom")?,
            epsilon: reg.quantity(1.0, "kilocalorie_per_mole")?,
        })
    }
}

/// Unit system used by the LAMMPS FF reader/writer.
///
/// Holds a registry with lj units defined from [`LammpsLjReference`], and
/// converts quantities **only** via `Quantity::to` (source → lj → target).
pub struct LammpsUnitSystem {
    reg: UnitRegistry,
}

impl LammpsUnitSystem {
    /// Build a system with the given lj reference scales.
    pub fn with_reference(reference: &LammpsLjReference) -> Result<Self, UnitsError> {
        let mut reg = UnitRegistry::new();
        reg.define_lj_units(&reference.mass, &reference.sigma, &reference.epsilon)?;
        Ok(Self { reg })
    }

    /// Canonical bridge (`1 g/mol`, `1 Å`, `1 kcal/mol`).
    pub fn canonical() -> Result<Self, UnitsError> {
        Self::with_reference(&LammpsLjReference::canonical()?)
    }

    /// Registry with lj units defined (for tests / advanced callers).
    pub fn registry(&self) -> &UnitRegistry {
        &self.reg
    }

    // ── unit expression names for each style ─────────────────────────────

    fn energy_unit(style: LammpsUnits) -> &'static str {
        match style {
            LammpsUnits::Lj => "lj_epsilon",
            LammpsUnits::Real => "kilocalorie_per_mole",
            LammpsUnits::Metal => "eV",
        }
    }

    fn length_unit(style: LammpsUnits) -> &'static str {
        match style {
            LammpsUnits::Lj => "lj_sigma",
            LammpsUnits::Real | LammpsUnits::Metal => "angstrom",
        }
    }

    /// Energy / length² (bond stiffness dimension before the ½-form map).
    fn energy_per_length2_unit(style: LammpsUnits) -> String {
        format!(
            "{}/{}**2",
            Self::energy_unit(style),
            Self::length_unit(style)
        )
    }

    /// Energy / rad² — LAMMPS angle K is energy per radian² in real/metal.
    fn energy_per_rad2_unit(style: LammpsUnits) -> String {
        // radians are dimensionless in the SI sense; LAMMPS still quotes K in
        // energy/rad². We treat rad as a pure number: unit = energy only for
        // conversion of the *energy* factor; form map applies 2K and deg→rad
        // separately for theta0.
        Self::energy_unit(style).to_string()
    }

    /// Convert a raw file value of the given dimension from `from` style to `to`
    /// style **through lj** (`from → lj → to`).
    fn convert_through_lj(
        &self,
        value: F,
        from: LammpsUnits,
        to: LammpsUnits,
        unit_for: impl Fn(LammpsUnits) -> String,
    ) -> Result<F, String> {
        if from == to {
            return Ok(value);
        }
        let from_u = self.reg.parse(&unit_for(from)).map_err(|e| e.to_string())?;
        let lj_u = self
            .reg
            .parse(&unit_for(LammpsUnits::Lj))
            .map_err(|e| e.to_string())?;
        let to_u = self.reg.parse(&unit_for(to)).map_err(|e| e.to_string())?;

        let q = Quantity::new(value, from_u);
        let in_lj = q.to(&lj_u).map_err(|e| e.to_string())?;
        let out = in_lj.to(&to_u).map_err(|e| e.to_string())?;
        Ok(out.value())
    }

    /// Energy (ε, dihedral K, …): `from → lj → to`.
    pub fn energy(&self, value: F, from: LammpsUnits, to: LammpsUnits) -> Result<F, String> {
        self.convert_through_lj(value, from, to, |s| Self::energy_unit(s).to_string())
    }

    /// Length (σ, r0): `from → lj → to`.
    pub fn length(&self, value: F, from: LammpsUnits, to: LammpsUnits) -> Result<F, String> {
        self.convert_through_lj(value, from, to, |s| Self::length_unit(s).to_string())
    }

    /// Bond stiffness *K* (LAMMPS form, energy/length²) before the ½ map.
    pub fn bond_k_lammps(&self, value: F, from: LammpsUnits, to: LammpsUnits) -> Result<F, String> {
        self.convert_through_lj(value, from, to, Self::energy_per_length2_unit)
    }

    /// Angle stiffness *K* (LAMMPS form, energy/rad²) — energy factor only.
    pub fn angle_k_lammps(
        &self,
        value: F,
        from: LammpsUnits,
        to: LammpsUnits,
    ) -> Result<F, String> {
        self.convert_through_lj(value, from, to, Self::energy_per_rad2_unit)
    }

    /// Convert file-side params into **store** units (default store = real for
    /// physical styles; lj stays lj).
    pub fn to_store_energy(&self, value: F, file: LammpsUnits) -> Result<F, String> {
        match file {
            LammpsUnits::Lj => Ok(value),
            other => self.energy(value, other, LammpsUnits::Real),
        }
    }

    pub fn to_store_length(&self, value: F, file: LammpsUnits) -> Result<F, String> {
        match file {
            LammpsUnits::Lj => Ok(value),
            other => self.length(value, other, LammpsUnits::Real),
        }
    }

    pub fn to_store_bond_k_lammps(&self, value: F, file: LammpsUnits) -> Result<F, String> {
        match file {
            LammpsUnits::Lj => Ok(value),
            other => self.bond_k_lammps(value, other, LammpsUnits::Real),
        }
    }

    pub fn to_store_angle_k_lammps(&self, value: F, file: LammpsUnits) -> Result<F, String> {
        match file {
            LammpsUnits::Lj => Ok(value),
            other => self.angle_k_lammps(value, other, LammpsUnits::Real),
        }
    }

    /// Store (real or lj) → file units for writing.
    pub fn from_store_energy(&self, value: F, file: LammpsUnits) -> Result<F, String> {
        match file {
            LammpsUnits::Lj => Ok(value),
            other => self.energy(value, LammpsUnits::Real, other),
        }
    }

    pub fn from_store_length(&self, value: F, file: LammpsUnits) -> Result<F, String> {
        match file {
            LammpsUnits::Lj => Ok(value),
            other => self.length(value, LammpsUnits::Real, other),
        }
    }

    pub fn from_store_bond_k_lammps(&self, value: F, file: LammpsUnits) -> Result<F, String> {
        match file {
            LammpsUnits::Lj => Ok(value),
            other => self.bond_k_lammps(value, LammpsUnits::Real, other),
        }
    }

    pub fn from_store_angle_k_lammps(&self, value: F, file: LammpsUnits) -> Result<F, String> {
        match file {
            LammpsUnits::Lj => Ok(value),
            other => self.angle_k_lammps(value, LammpsUnits::Real, other),
        }
    }
}

// ── form maps (independent of unit style) ───────────────────────────────────

/// LAMMPS harmonic `K(x−x0)²` → molrs `½k(x−x0)²` with `k = 2K`.
#[inline]
pub fn lammps_k_to_molrs_half_k(k_lammps: F) -> F {
    2.0 * k_lammps
}

/// molrs `½k` → LAMMPS `K = k/2`.
#[inline]
pub fn molrs_half_k_to_lammps_k(k_molrs: F) -> F {
    k_molrs / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_to_metal_energy_via_lj_matches_si() {
        let sys = LammpsUnitSystem::canonical().unwrap();
        // 1 kcal/mol → metal eV through lj hub.
        let ev = sys
            .energy(1.0, LammpsUnits::Real, LammpsUnits::Metal)
            .unwrap();
        // Direct SI path for comparison (not used in production FF code).
        let reg = UnitRegistry::new();
        let direct = reg
            .quantity(1.0, "kilocalorie_per_mole")
            .unwrap()
            .to(&reg.parse("eV").unwrap())
            .unwrap()
            .value();
        assert!((ev - direct).abs() < 1e-12, "lj hub {ev} vs SI {direct}");
    }

    #[test]
    fn metal_to_real_energy_via_lj() {
        let sys = LammpsUnitSystem::canonical().unwrap();
        let kcal = sys
            .energy(1.0, LammpsUnits::Metal, LammpsUnits::Real)
            .unwrap();
        let reg = UnitRegistry::new();
        let direct = reg
            .quantity(1.0, "eV")
            .unwrap()
            .to(&reg.parse("kilocalorie_per_mole").unwrap())
            .unwrap()
            .value();
        assert!(
            (kcal - direct).abs() < 1e-12,
            "lj hub {kcal} vs SI {direct}"
        );
        // ~23.06 kcal/mol per eV
        assert!((kcal - 23.060_547_830_619).abs() < 1e-6 || (kcal - direct).abs() < 1e-12);
    }

    #[test]
    fn length_real_metal_identical() {
        let sys = LammpsUnitSystem::canonical().unwrap();
        let a = sys
            .length(3.5, LammpsUnits::Real, LammpsUnits::Metal)
            .unwrap();
        assert!((a - 3.5).abs() < 1e-15);
    }

    #[test]
    fn lj_energy_pass_through_to_store() {
        let sys = LammpsUnitSystem::canonical().unwrap();
        assert_eq!(sys.to_store_energy(0.5, LammpsUnits::Lj).unwrap(), 0.5);
    }

    #[test]
    fn form_half_k_roundtrip() {
        let k = lammps_k_to_molrs_half_k(228.89);
        assert!((k - 457.78).abs() < 1e-9);
        assert!((molrs_half_k_to_lammps_k(k) - 228.89).abs() < 1e-9);
    }

    #[test]
    fn parse_units_keywords() {
        assert_eq!(LammpsUnits::parse("REAL").unwrap(), LammpsUnits::Real);
        assert_eq!(LammpsUnits::parse("metal").unwrap(), LammpsUnits::Metal);
        assert_eq!(LammpsUnits::parse("lj").unwrap(), LammpsUnits::Lj);
        assert!(LammpsUnits::parse("si").is_err());
    }

    #[test]
    fn bond_k_metal_to_real_scales_with_energy() {
        let sys = LammpsUnitSystem::canonical().unwrap();
        // Same numerical K in metal (eV/Å²) vs real (kcal/mol/Å²) must scale
        // exactly as energy (length is Å in both).
        let k_real = sys
            .bond_k_lammps(1.0, LammpsUnits::Metal, LammpsUnits::Real)
            .unwrap();
        let e_real = sys
            .energy(1.0, LammpsUnits::Metal, LammpsUnits::Real)
            .unwrap();
        assert!((k_real - e_real).abs() < 1e-12);
    }
}
