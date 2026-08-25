//! Engine-neutral unit-system presets.
//!
//! A [`UnitPreset`] is a named view of the constants in [`super::constants`]
//! plus the ten base-unit names of a LAMMPS-style unit system. Preset **names**
//! keep the familiar `"real"` / `"metal"` / `"lj"` tokens; the type and module
//! names do not mention LAMMPS. Callers compose conversions themselves —
//! there is no `convert(value, from, to)` façade.
//!
//! Reference: LAMMPS `units` command,
//! <https://docs.lammps.org/units.html>; Thompson et al.,
//! *Comput. Phys. Commun.* **271** (2022) 108171.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use crate::types::F;

use super::constants::{BOLTZMANN, BOLTZMANN_REAL, COULOMB_REAL, ELEMENTARY_CHARGE};

/// Ten named dimensions a preset reports, in a stable order.
pub const PRESET_DIMENSIONS: [&str; 10] = [
    "mass",
    "length",
    "time",
    "energy",
    "temperature",
    "charge",
    "pressure",
    "velocity",
    "force",
    "density",
];

/// One unit-system view: ten unit names plus the Boltzmann and Coulomb
/// constants expressed in that system.
#[derive(Clone, Debug)]
pub struct UnitPreset {
    name: String,
    units: HashMap<&'static str, String>,
    boltzmann: F,
    coulomb: F,
}

impl UnitPreset {
    fn from_table(
        name: &str,
        units: [(&'static str, &'static str); 10],
        boltzmann: F,
        coulomb: F,
    ) -> Self {
        Self {
            name: name.to_owned(),
            units: units.into_iter().map(|(k, v)| (k, v.to_owned())).collect(),
            boltzmann,
            coulomb,
        }
    }

    /// LAMMPS `real`: Å, kcal/mol, fs, e.
    pub fn real() -> Self {
        Self::from_table(
            "real",
            [
                ("mass", "gram_per_mole"),
                ("length", "angstrom"),
                ("time", "femtosecond"),
                ("energy", "kilocalorie_per_mole"),
                ("temperature", "kelvin"),
                ("charge", "elementary_charge"),
                ("pressure", "atmosphere"),
                ("velocity", "angstrom / femtosecond"),
                ("force", "kilocalorie_per_mole / angstrom"),
                ("density", "gram / centimeter ** 3"),
            ],
            BOLTZMANN_REAL,
            COULOMB_REAL,
        )
    }

    /// LAMMPS `metal`: Å, eV, ps, e.
    pub fn metal() -> Self {
        Self::from_table(
            "metal",
            [
                ("mass", "gram_per_mole"),
                ("length", "angstrom"),
                ("time", "picosecond"),
                ("energy", "electron_volt"),
                ("temperature", "kelvin"),
                ("charge", "elementary_charge"),
                ("pressure", "bar"),
                ("velocity", "angstrom / picosecond"),
                ("force", "electron_volt / angstrom"),
                ("density", "gram / centimeter ** 3"),
            ],
            BOLTZMANN / ELEMENTARY_CHARGE,
            COULOMB_REAL * (BOLTZMANN / ELEMENTARY_CHARGE) / BOLTZMANN_REAL,
        )
    }

    /// SI: kg, m, s, J.
    pub fn si() -> Self {
        Self::from_table(
            "si",
            [
                ("mass", "kilogram"),
                ("length", "meter"),
                ("time", "second"),
                ("energy", "joule"),
                ("temperature", "kelvin"),
                ("charge", "coulomb"),
                ("pressure", "pascal"),
                ("velocity", "meter / second"),
                ("force", "newton"),
                ("density", "kilogram / meter ** 3"),
            ],
            BOLTZMANN,
            8.987_551_792_3e9,
        )
    }

    /// CGS.
    pub fn cgs() -> Self {
        Self::from_table(
            "cgs",
            [
                ("mass", "gram"),
                ("length", "centimeter"),
                ("time", "second"),
                ("energy", "erg"),
                ("temperature", "kelvin"),
                ("charge", "statcoulomb"),
                ("pressure", "dyne / centimeter ** 2"),
                ("velocity", "centimeter / second"),
                ("force", "dyne"),
                ("density", "gram / centimeter ** 3"),
            ],
            BOLTZMANN * 1e7,
            1.0,
        )
    }

    /// Atomic / `electron` units.
    pub fn electron() -> Self {
        Self::from_table(
            "electron",
            [
                ("mass", "amu"),
                ("length", "bohr"),
                ("time", "femtosecond"),
                ("energy", "hartree"),
                ("temperature", "kelvin"),
                ("charge", "elementary_charge"),
                ("pressure", "pascal"),
                ("velocity", "bohr / femtosecond"),
                ("force", "hartree / bohr"),
                ("density", "amu / bohr ** 3"),
            ],
            BOLTZMANN / 4.359_744_722_207_1e-18,
            1.0,
        )
    }

    /// Reduced LJ units. Numeric constants are 1; names follow the registry's
    /// `lj_*` definitions.
    pub fn lj() -> Self {
        Self::from_table(
            "lj",
            [
                ("mass", "lj_mass"),
                ("length", "lj_sigma"),
                ("time", "lj_tau"),
                ("energy", "lj_epsilon"),
                ("temperature", "lj_epsilon"),
                ("charge", "lj_charge"),
                ("pressure", "lj_epsilon / lj_sigma ** 3"),
                ("velocity", "lj_sigma / lj_tau"),
                ("force", "lj_epsilon / lj_sigma"),
                ("density", "lj_mass / lj_sigma ** 3"),
            ],
            1.0,
            1.0,
        )
    }

    /// Microscopic (`micro`) style.
    pub fn micro() -> Self {
        Self::from_table(
            "micro",
            [
                ("mass", "picogram"),
                ("length", "micrometer"),
                ("time", "microsecond"),
                ("energy", "picogram * micrometer ** 2 / microsecond ** 2"),
                ("temperature", "kelvin"),
                ("charge", "picocoulomb"),
                ("pressure", "picogram / (micrometer * microsecond ** 2)"),
                ("velocity", "micrometer / microsecond"),
                ("force", "picogram * micrometer / microsecond ** 2"),
                ("density", "picogram / micrometer ** 3"),
            ],
            BOLTZMANN,
            1.0,
        )
    }

    /// Nanoscopic (`nano`) style.
    pub fn nano() -> Self {
        Self::from_table(
            "nano",
            [
                ("mass", "attogram"),
                ("length", "nanometer"),
                ("time", "nanosecond"),
                ("energy", "attogram * nanometer ** 2 / nanosecond ** 2"),
                ("temperature", "kelvin"),
                ("charge", "elementary_charge"),
                ("pressure", "attogram / (nanometer * nanosecond ** 2)"),
                ("velocity", "nanometer / nanosecond"),
                ("force", "attogram * nanometer / nanosecond ** 2"),
                ("density", "attogram / nanometer ** 3"),
            ],
            BOLTZMANN,
            1.0,
        )
    }

    /// Preset name (`"real"`, `"metal"`, …).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Boltzmann constant in this system's energy / temperature units.
    pub fn boltzmann(&self) -> F {
        self.boltzmann
    }

    /// Coulomb constant in this system's energy · length / charge² units.
    pub fn coulomb(&self) -> F {
        self.coulomb
    }

    /// Unit name for `dimension`, or `None` if the preset does not define it.
    pub fn unit(&self, dimension: &str) -> Option<&str> {
        self.units.get(dimension).map(String::as_str)
    }

    pub fn mass(&self) -> &str {
        self.unit("mass").expect("preset defines mass")
    }
    pub fn length(&self) -> &str {
        self.unit("length").expect("preset defines length")
    }
    pub fn time(&self) -> &str {
        self.unit("time").expect("preset defines time")
    }
    pub fn energy(&self) -> &str {
        self.unit("energy").expect("preset defines energy")
    }
    pub fn temperature(&self) -> &str {
        self.unit("temperature")
            .expect("preset defines temperature")
    }
    pub fn charge(&self) -> &str {
        self.unit("charge").expect("preset defines charge")
    }
    pub fn pressure(&self) -> &str {
        self.unit("pressure").expect("preset defines pressure")
    }
    pub fn velocity(&self) -> &str {
        self.unit("velocity").expect("preset defines velocity")
    }
    pub fn force(&self) -> &str {
        self.unit("force").expect("preset defines force")
    }
    pub fn density(&self) -> &str {
        self.unit("density").expect("preset defines density")
    }
}

/// Named registry of [`UnitPreset`]s. Built-ins are pre-registered; callers
/// extend it with [`register`](Self::register).
pub struct UnitPresetRegistry {
    inner: HashMap<String, UnitPreset>,
}

impl UnitPresetRegistry {
    /// Empty registry (no built-ins).
    pub fn empty() -> Self {
        Self {
            inner: HashMap::new(),
        }
    }

    /// Built-in presets: real, metal, si, cgs, electron, lj, micro, nano.
    pub fn new() -> Self {
        let mut reg = Self::empty();
        for p in [
            UnitPreset::real(),
            UnitPreset::metal(),
            UnitPreset::si(),
            UnitPreset::cgs(),
            UnitPreset::electron(),
            UnitPreset::lj(),
            UnitPreset::micro(),
            UnitPreset::nano(),
        ] {
            let name = p.name().to_owned();
            reg.inner.insert(name, p);
        }
        reg
    }

    /// Insert `data` under `name`. Errors if the name is already taken.
    pub fn register(&mut self, name: impl Into<String>, data: UnitPreset) -> Result<(), String> {
        let name = name.into();
        if self.inner.contains_key(&name) {
            return Err(format!("unit preset `{name}` is already registered"));
        }
        self.inner.insert(name, data);
        Ok(())
    }

    /// Look up a preset by name.
    pub fn get(&self, name: &str) -> Option<&UnitPreset> {
        self.inner.get(name)
    }

    /// Iterate registered `(name, preset)` pairs. Order is the map's.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &UnitPreset)> {
        self.inner.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Number of registered presets.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl Default for UnitPresetRegistry {
    fn default() -> Self {
        Self::new()
    }
}

fn global() -> &'static Mutex<UnitPresetRegistry> {
    static REGISTRY: OnceLock<Mutex<UnitPresetRegistry>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(UnitPresetRegistry::new()))
}

/// Process-wide preset lookup (built-ins plus anything [`register_preset`] added).
pub fn lookup_preset(name: &str) -> Option<UnitPreset> {
    global().lock().ok()?.get(name).cloned()
}

/// Register an extra preset on the process-wide registry.
pub fn register_preset(name: impl Into<String>, data: UnitPreset) -> Result<(), String> {
    global()
        .lock()
        .map_err(|e| e.to_string())?
        .register(name, data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::units::constants::BOLTZMANN_REAL;

    #[test]
    fn real_boltzmann_is_bit_identical_to_the_constant() {
        assert_eq!(UnitPreset::real().boltzmann(), BOLTZMANN_REAL);
    }

    #[test]
    fn real_energy_is_kcal_and_time_is_fs() {
        assert_eq!(UnitPreset::real().energy(), "kilocalorie_per_mole");
        assert_eq!(UnitPreset::real().time(), "femtosecond");
    }

    #[test]
    fn every_builtin_preset_reports_ten_dimensions() {
        let reg = UnitPresetRegistry::new();
        assert!(reg.len() >= 7);
        for (_name, preset) in reg.iter() {
            for dim in PRESET_DIMENSIONS {
                assert!(
                    preset.unit(dim).is_some(),
                    "preset `{}` missing dimension `{dim}`",
                    preset.name()
                );
            }
        }
    }

    #[test]
    fn register_rejects_a_duplicate_name() {
        let mut reg = UnitPresetRegistry::empty();
        reg.register("real", UnitPreset::real()).unwrap();
        let err = reg.register("real", UnitPreset::metal()).unwrap_err();
        assert!(err.contains("real"));
    }

    #[test]
    fn lookup_by_name_returns_the_real_preset() {
        assert_eq!(
            lookup_preset("real").unwrap().boltzmann(),
            UnitPreset::real().boltzmann()
        );
    }
}
