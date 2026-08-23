//! Integrator energy unit and conversions from force-field presets.
//!
//! The arithmetic `v += Δt · F / m` needs `[F] = [m][length]/[time]²`, so
//! eV/Å + amu + fs is **not** consistent. The canonical MD system is
//! (amu, Å, fs) with energy in amu·Å²/fs². Force-field energies (LAMMPS
//! `real` = kcal/mol, `metal` = eV, …) convert at the adapter boundary
//! through [`molrs::units::UnitRegistry`].
//!
//! Analysis / force-field code keeps kcal/mol (see `.claude/notes/science.md`).
//! This module is the one place that names the MD-internal exception.

use molrs::types::F;
use molrs::units::UnitRegistry;
use molrs::units::constants::BOLTZMANN;

use super::error::MdError;

/// Integrator energy unit: `(amu, Å, fs)` so `v += Δt·F/m` is consistent.
pub const MD_ENERGY: &str = "amu * angstrom ** 2 / femtosecond ** 2";

/// Convert `value` expressed in `from` into amu·Å²/fs².
pub fn energy_to_md(value: F, from: &str) -> Result<F, MdError> {
    let q = UnitRegistry::global().quantity(value, from)?;
    Ok(q.to_parsed(MD_ENERGY)?.value())
}

/// Magnitude of one energy unit of a LAMMPS-style name in amu·Å²/fs².
///
/// Resolves the style the same way molpy `UnitSystem.preset` does
/// (`metal` → `electron_volt`, `real` → `kilocalorie_per_mole`) and converts
/// with the registry. Unknown styles are an error, not a silent 1.0.
pub fn preset_energy_to_md(style: &str) -> Result<F, MdError> {
    let energy = match style.to_ascii_lowercase().as_str() {
        "real" => "kilocalorie_per_mole",
        "metal" => "electron_volt",
        "si" => "joule",
        "cgs" => "erg",
        "electron" => "hartree",
        "lj" => {
            return Err(MdError::Invalid(
                "LAMMPS style `lj` is reduced units; convert ε to amu·Å²/fs² at the call site"
                    .into(),
            ));
        }
        other => {
            return Err(MdError::Invalid(format!(
                "unknown LAMMPS unit style `{other}` (real, metal, si, cgs, electron)"
            )));
        }
    };
    energy_to_md(1.0, energy)
}

/// Boltzmann constant in amu·Å²/fs²/K, so `T = 2K / (dof · k_B)` is kelvin.
pub fn kb_md() -> Result<F, MdError> {
    let q = UnitRegistry::global().quantity(BOLTZMANN, "J / K")?;
    Ok(
        q.to_parsed("amu * angstrom ** 2 / femtosecond ** 2 / kelvin")?
            .value(),
    )
}

#[cfg(test)]
mod tests {
    use molrs::units::UnitRegistry;

    use super::*;

    #[test]
    fn real_kcal_converts_to_a_finite_positive_md_energy() {
        let factor = preset_energy_to_md("real").unwrap();
        assert!(factor.is_finite() && factor > 0.0);
    }

    #[test]
    fn metal_is_larger_than_real_by_ev_to_kcal_ratio() {
        let real = preset_energy_to_md("real").unwrap();
        let metal = preset_energy_to_md("metal").unwrap();
        let ev_per_kcal = UnitRegistry::global()
            .quantity(1.0, "electron_volt")
            .unwrap()
            .to(&UnitRegistry::global()
                .parse("kilocalorie_per_mole")
                .unwrap())
            .unwrap()
            .value();
        assert!((metal / real - ev_per_kcal).abs() < 1e-12);
    }

    #[test]
    fn unknown_style_is_an_error() {
        let err = preset_energy_to_md("banana").unwrap_err();
        assert!(format!("{err}").contains("banana"));
    }

    #[test]
    fn lj_style_is_refused() {
        let err = preset_energy_to_md("lj").unwrap_err();
        assert!(format!("{err}").contains("reduced"));
    }

    #[test]
    fn kb_md_is_positive_and_finite() {
        let kb = kb_md().unwrap();
        assert!(kb.is_finite() && kb > 0.0);
    }
}
