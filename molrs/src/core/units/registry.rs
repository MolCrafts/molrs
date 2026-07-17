//! The unit registry: definitions + prefix table + parse entry points.

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::types::F;

use super::dimension::Dimension;
use super::error::UnitsError;
use super::quantity::Quantity;
use super::unit::Unit;

/// A single unit definition added to a [`UnitRegistry`].
///
/// Conversion to SI base is `si = value * factor + offset`, where `factor`
/// and `offset` are expressed in the SI base units of `dimension` (e.g.
/// `angstrom` has `factor = 1e-10` metres, `degC` has `offset = 273.15`
/// kelvin).
#[derive(Clone, Debug, PartialEq)]
pub struct UnitDef {
    /// Canonical name, e.g. `"calorie"`.
    pub name: String,
    /// Alternative names, e.g. `["cal"]`.
    pub aliases: Vec<String>,
    /// Preferred display symbol.
    pub symbol: String,
    /// Multiplicative factor to SI base (SI base units per one of this unit).
    pub factor: F,
    /// Additive offset to SI base, in SI base units (non-zero only for
    /// affine units such as `degC`).
    pub offset: F,
    /// Dimension of the unit.
    pub dimension: Dimension,
    /// Whether the unit accepts SI prefixes (kcal, nm, fs).
    pub prefixable: bool,
}

/// SI prefix table, longest prefix first so `da` wins over `d`.
///
/// `u` and `µ` (U+00B5) are both accepted for micro.
const PREFIXES: &[(&str, F)] = &[
    ("da", 1e1),
    ("y", 1e-24),
    ("z", 1e-21),
    ("a", 1e-18),
    ("f", 1e-15),
    ("p", 1e-12),
    ("n", 1e-9),
    ("u", 1e-6),
    ("µ", 1e-6),
    ("m", 1e-3),
    ("c", 1e-2),
    ("d", 1e-1),
    ("h", 1e2),
    ("k", 1e3),
    ("M", 1e6),
    ("G", 1e9),
    ("T", 1e12),
    ("P", 1e15),
    ("E", 1e18),
    ("Z", 1e21),
    ("Y", 1e24),
];

/// Spelled-out SI prefixes accepted alongside their symbols.
///
/// This keeps Python attribute-style unit lookup intuitive (``nanometer``,
/// ``femtosecond``) without duplicating prefixed definitions in the registry.
const LONG_PREFIXES: &[(&str, F)] = &[
    ("quetta", 1e30),
    ("ronna", 1e27),
    ("yotta", 1e24),
    ("zetta", 1e21),
    ("exa", 1e18),
    ("peta", 1e15),
    ("tera", 1e12),
    ("giga", 1e9),
    ("mega", 1e6),
    ("kilo", 1e3),
    ("hecto", 1e2),
    ("deca", 1e1),
    ("deci", 1e-1),
    ("centi", 1e-2),
    ("milli", 1e-3),
    ("micro", 1e-6),
    ("nano", 1e-9),
    ("pico", 1e-12),
    ("femto", 1e-15),
    ("atto", 1e-18),
    ("zepto", 1e-21),
    ("yocto", 1e-24),
    ("ronto", 1e-27),
    ("quecto", 1e-30),
];

/// A registry of unit definitions and the SI prefix table.
///
/// The registry resolves unit names (canonical name, symbol, or alias,
/// optionally with one SI prefix) and parses compound expressions into
/// self-contained [`Unit`] values. Preload factors are SI-2019 exact or
/// CODATA 2018 recommended values; each entry in the preload tables below
/// cites its source.
///
/// # Examples
///
/// End-to-end: build a registry, parse units, convert a quantity.
///
/// ```
/// use molrs::units::{UnitRegistry, UnitsError};
///
/// let reg = UnitRegistry::new();
///
/// // A force gradient of 2 kcal/(mol·Å), converted to kJ/(mol·nm).
/// let g = reg.quantity(2.0, "kcal/mol/angstrom")?;
/// let target = reg.parse("kJ/mol/nm")?;
/// let converted = g.to(&target)?;
/// assert!((converted.value() - 83.68).abs() < 1e-9);
/// # Ok::<(), UnitsError>(())
/// ```
pub struct UnitRegistry {
    defs: Vec<UnitDef>,
    /// Canonical name, symbol, and every alias → index into `defs`.
    index: HashMap<String, usize>,
}

static GLOBAL_REGISTRY: OnceLock<UnitRegistry> = OnceLock::new();

impl UnitRegistry {
    /// Preloaded with SI + molecular-simulation units.
    ///
    /// Covers the SI base set plus the MD working set: `angstrom`, `bohr`,
    /// `calorie`/`kcal`, `electron_volt`, `hartree`, `dalton`/`amu`, `bar`,
    /// `atmosphere`, `degC`, `radian`/`degree`, `elementary_charge`, `debye`,
    /// and all prefixable SI derivatives (`nm`, `fs`, `kJ`, ...).
    pub fn new() -> UnitRegistry {
        let mut r = UnitRegistry::empty();
        for def in md_defs() {
            r.define(def)
                .expect("preloaded MD unit table must be collision-free");
        }
        r
    }

    /// SI base units only — for building a custom system from scratch.
    ///
    /// Includes `gram` as the prefixable mass atom (the SI base unit is
    /// `kilogram`, which itself takes no further prefix).
    pub fn empty() -> UnitRegistry {
        let mut r = UnitRegistry {
            defs: Vec::new(),
            index: HashMap::new(),
        };
        for def in base_defs() {
            r.define(def)
                .expect("SI base unit table must be collision-free");
        }
        r
    }

    /// Shared immutable preloaded registry (OnceLock singleton).
    pub fn global() -> &'static UnitRegistry {
        GLOBAL_REGISTRY.get_or_init(UnitRegistry::new)
    }

    /// Define a unit, registering its name, symbol, and all aliases.
    ///
    /// # Errors
    ///
    /// [`UnitsError::Redefinition`] if the name, symbol, or any alias is
    /// already registered.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::units::{Dimension, UnitDef, UnitRegistry, UnitsError};
    ///
    /// let mut reg = UnitRegistry::empty();
    /// reg.define(UnitDef {
    ///     name: "smoot".to_string(),
    ///     aliases: vec![],
    ///     symbol: "smoot".to_string(),
    ///     factor: 1.7018, // metres per smoot
    ///     offset: 0.0,
    ///     dimension: Dimension::LENGTH,
    ///     prefixable: false,
    /// })?;
    /// assert!(reg.parse("smoot").is_ok());
    /// # Ok::<(), UnitsError>(())
    /// ```
    pub fn define(&mut self, def: UnitDef) -> Result<(), UnitsError> {
        let mut keys: Vec<&str> = Vec::with_capacity(2 + def.aliases.len());
        keys.push(&def.name);
        if def.symbol != def.name {
            keys.push(&def.symbol);
        }
        for alias in &def.aliases {
            if !keys.contains(&alias.as_str()) {
                keys.push(alias);
            }
        }
        for key in &keys {
            if self.index.contains_key(*key) {
                return Err(UnitsError::Redefinition {
                    name: (*key).to_string(),
                });
            }
        }
        let idx = self.defs.len();
        let keys: Vec<String> = keys.into_iter().map(str::to_string).collect();
        self.defs.push(def);
        for key in keys {
            self.index.insert(key, idx);
        }
        Ok(())
    }

    /// Parse a compound unit expression (`"kcal/mol/angstrom"`, `"m s^-2"`).
    ///
    /// Supports `*`, `/`, `·`, whitespace (implicit multiplication),
    /// parentheses, `^`/`**` integer exponents, numeric factors, and one SI
    /// prefix per atom. Affine units (`degC`) are only legal as a single
    /// bare atom, never inside a compound expression.
    ///
    /// # Errors
    ///
    /// - [`UnitsError::UnknownUnit`] — an atom resolves to no definition.
    /// - [`UnitsError::Parse`] — malformed expression (empty input, bad
    ///   exponent, unbalanced parentheses, trailing tokens).
    /// - [`UnitsError::AffineUnit`] — an affine unit appears in a compound
    ///   expression.
    pub fn parse(&self, expr: &str) -> Result<Unit, UnitsError> {
        super::parse::parse_expr(self, expr)
    }

    /// Convenience: `registry.quantity(1.5, "kcal/mol")`.
    ///
    /// # Errors
    ///
    /// Same as [`UnitRegistry::parse`].
    pub fn quantity(&self, value: F, expr: &str) -> Result<Quantity, UnitsError> {
        Ok(Quantity::new(value, self.parse(expr)?))
    }

    /// Define Lennard-Jones reduced units from physical mass, length, and energy scales.
    ///
    /// The derived time and temperature scales are
    /// `tau = sqrt(mass * sigma^2 / epsilon)` and `epsilon / k_B`.
    /// The definitions retain their physical dimensions, so normal checked
    /// conversion works without a separate context engine.
    pub fn define_lj_units(
        &mut self,
        mass: &Quantity,
        sigma: &Quantity,
        epsilon: &Quantity,
    ) -> Result<(), UnitsError> {
        let mass_kg = mass.to(&self.parse("kg")?)?.value();
        let sigma_m = sigma.to(&self.parse("m")?)?.value();
        let epsilon_j = epsilon.to(&self.parse("J")?)?.value();
        if !mass_kg.is_finite()
            || !sigma_m.is_finite()
            || !epsilon_j.is_finite()
            || mass_kg <= 0.0
            || sigma_m <= 0.0
            || epsilon_j <= 0.0
        {
            return Err(UnitsError::Parse {
                expr: "Lennard-Jones scales".to_string(),
                message: "mass, sigma, and epsilon must be finite and positive".to_string(),
            });
        }

        let tau_s = (mass_kg * sigma_m * sigma_m / epsilon_j).sqrt();
        let temperature_k = epsilon_j / super::constants::BOLTZMANN;
        for definition in [
            def(
                "lj_sigma",
                "lj_sigma",
                &[],
                sigma_m,
                0.0,
                Dimension::LENGTH,
                false,
            ),
            def(
                "lj_epsilon",
                "lj_epsilon",
                &[],
                epsilon_j,
                0.0,
                Dimension::ENERGY,
                false,
            ),
            def("lj_tau", "lj_tau", &[], tau_s, 0.0, Dimension::TIME, false),
            def(
                "lj_epsilon_over_kB",
                "lj_epsilon_over_kB",
                &[],
                temperature_k,
                0.0,
                Dimension::TEMPERATURE,
                false,
            ),
        ] {
            self.define(definition)?;
        }
        Ok(())
    }

    /// Iterate all registered definitions (test/introspection support).
    pub fn definitions(&self) -> impl Iterator<Item = &UnitDef> {
        self.defs.iter()
    }

    /// Resolve a single unit atom: exact name/symbol/alias first, then one
    /// SI prefix + prefixable atom. Returns `(factor, offset, dimension)`.
    pub(crate) fn resolve_atom(&self, atom: &str) -> Option<(F, F, Dimension)> {
        if let Some(&idx) = self.index.get(atom) {
            let def = &self.defs[idx];
            return Some((def.factor, def.offset, def.dimension));
        }
        for (prefix, scale) in PREFIXES {
            let Some(rest) = atom.strip_prefix(prefix) else {
                continue;
            };
            if rest.is_empty() {
                continue;
            }
            if let Some(&idx) = self.index.get(rest) {
                let def = &self.defs[idx];
                if def.prefixable && def.offset == 0.0 {
                    return Some((scale * def.factor, 0.0, def.dimension));
                }
            }
        }
        for (prefix, scale) in LONG_PREFIXES {
            let Some(rest) = atom.strip_prefix(prefix) else {
                continue;
            };
            if rest.is_empty() {
                continue;
            }
            if let Some(&idx) = self.index.get(rest) {
                let def = &self.defs[idx];
                if def.prefixable && def.offset == 0.0 {
                    return Some((scale * def.factor, 0.0, def.dimension));
                }
            }
        }
        None
    }
}

/// Shorthand constructor for the preload tables.
fn def(
    name: &str,
    symbol: &str,
    aliases: &[&str],
    factor: F,
    offset: F,
    dimension: Dimension,
    prefixable: bool,
) -> UnitDef {
    UnitDef {
        name: name.to_string(),
        aliases: aliases.iter().map(|s| s.to_string()).collect(),
        symbol: symbol.to_string(),
        factor,
        offset,
        dimension,
        prefixable,
    }
}

/// The 7 SI base units plus `gram` (the prefixable mass atom).
fn base_defs() -> Vec<UnitDef> {
    const LUMINOSITY: Dimension = Dimension::from_exponents([0, 0, 0, 0, 0, 0, 1]);
    vec![
        def("meter", "m", &[], 1.0, 0.0, Dimension::LENGTH, true),
        def("kilogram", "kg", &[], 1.0, 0.0, Dimension::MASS, false),
        def("gram", "g", &[], 1e-3, 0.0, Dimension::MASS, true),
        def("second", "s", &["sec"], 1.0, 0.0, Dimension::TIME, true),
        def("ampere", "A", &[], 1.0, 0.0, Dimension::CURRENT, true),
        def("kelvin", "K", &[], 1.0, 0.0, Dimension::TEMPERATURE, true),
        def("mole", "mol", &[], 1.0, 0.0, Dimension::AMOUNT, true),
        def("candela", "cd", &[], 1.0, 0.0, LUMINOSITY, true),
    ]
}

/// Derived + molecular-simulation unit set.
///
/// Factors are SI-2019 exact or CODATA 2018 recommended values; each entry
/// cites its source.
fn md_defs() -> Vec<UnitDef> {
    const CHARGE_LENGTH: Dimension = Dimension::from_exponents([1, 0, 1, 1, 0, 0, 0]);
    let l = Dimension::LENGTH;
    let e = Dimension::ENERGY;
    vec![
        // Length. angstrom: exact; bohr: CODATA 2018 a0.
        def("angstrom", "Å", &["ang"], 1e-10, 0.0, l, false),
        def("bohr", "bohr", &["a0"], 5.291_772_109_03e-11, 0.0, l, false),
        // Energy. joule: SI derived; calorie: thermochemical, exact 4.184 J;
        // eV: SI-2019 exact; hartree: CODATA 2018.
        def("joule", "J", &[], 1.0, 0.0, e, true),
        def("calorie", "cal", &[], 4.184, 0.0, e, true),
        def(
            "kilocalorie_per_mole",
            "kcal_per_mol",
            &[],
            4184.0 / super::constants::AVOGADRO,
            0.0,
            e,
            false,
        ),
        def(
            "kilojoule_per_mole",
            "kJ_per_mol",
            &[],
            1000.0 / super::constants::AVOGADRO,
            0.0,
            e,
            false,
        ),
        def("erg", "erg", &[], 1e-7, 0.0, e, false),
        def("electron_volt", "eV", &[], 1.602_176_634e-19, 0.0, e, true),
        def("hartree", "Eh", &[], 4.359_744_722_207_1e-18, 0.0, e, false),
        // Force / pressure (SI derived, exact).
        def("newton", "N", &[], 1.0, 0.0, Dimension::FORCE, true),
        def("dyne", "dyn", &[], 1e-5, 0.0, Dimension::FORCE, false),
        def("pascal", "Pa", &[], 1.0, 0.0, Dimension::PRESSURE, true),
        def("bar", "bar", &[], 1e5, 0.0, Dimension::PRESSURE, false),
        def(
            "atmosphere",
            "atm",
            &[],
            101_325.0,
            0.0,
            Dimension::PRESSURE,
            false,
        ),
        // Time (exact).
        def("minute", "min", &[], 60.0, 0.0, Dimension::TIME, false),
        def("hour", "h", &[], 3600.0, 0.0, Dimension::TIME, false),
        // Mass. dalton: CODATA 2018; prefixable for kDa.
        def(
            "dalton",
            "Da",
            &["amu"],
            1.660_539_066_60e-27,
            0.0,
            Dimension::MASS,
            true,
        ),
        def(
            "gram_per_mole",
            "g_per_mol",
            &[],
            1e-3 / super::constants::AVOGADRO,
            0.0,
            Dimension::MASS,
            false,
        ),
        // Temperature: affine, offset 273.15 K (exact).
        def(
            "degC",
            "degC",
            &["celsius", "°C"],
            1.0,
            273.15,
            Dimension::TEMPERATURE,
            false,
        ),
        // Angle (dimensionless). degree: exact π/180.
        def(
            "radian",
            "rad",
            &[],
            1.0,
            0.0,
            Dimension::DIMENSIONLESS,
            true,
        ),
        def(
            "degree",
            "deg",
            &[],
            std::f64::consts::PI / 180.0,
            0.0,
            Dimension::DIMENSIONLESS,
            false,
        ),
        // Charge. coulomb: SI derived; elementary charge: SI-2019 exact;
        // debye: 1e-21/c C·m, c exact.
        def("coulomb", "C", &[], 1.0, 0.0, Dimension::CHARGE, true),
        def(
            "statcoulomb",
            "statC",
            &[],
            3.335_640_951_981_52e-10,
            0.0,
            Dimension::CHARGE,
            false,
        ),
        def(
            "elementary_charge",
            "e",
            &[],
            1.602_176_634e-19,
            0.0,
            Dimension::CHARGE,
            false,
        ),
        def(
            "debye",
            "D",
            &[],
            3.335_640_951_98e-30,
            0.0,
            CHARGE_LENGTH,
            false,
        ),
    ]
}

impl Default for UnitRegistry {
    fn default() -> UnitRegistry {
        UnitRegistry::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn custom_def(name: &str, aliases: &[&str]) -> UnitDef {
        UnitDef {
            name: name.to_string(),
            aliases: aliases.iter().map(|s| s.to_string()).collect(),
            symbol: name.to_string(),
            factor: 1.0,
            offset: 0.0,
            dimension: Dimension::LENGTH,
            prefixable: false,
        }
    }

    #[test]
    fn define_new_unit_ok() {
        let mut r = UnitRegistry::empty();
        assert!(r.define(custom_def("smoot", &["sm"])).is_ok());
        // newly defined unit is parseable
        assert!(r.parse("smoot").is_ok());
    }

    #[test]
    fn redefinition_is_error() {
        let mut r = UnitRegistry::empty();
        r.define(custom_def("smoot", &[])).unwrap();
        let err = r.define(custom_def("smoot", &[])).unwrap_err();
        assert!(matches!(err, UnitsError::Redefinition { .. }));
    }

    #[test]
    fn alias_collision_is_error() {
        let mut r = UnitRegistry::empty();
        r.define(custom_def("smoot", &["sm"])).unwrap();
        // second unit reuses an existing alias
        let err = r.define(custom_def("widget", &["sm"])).unwrap_err();
        assert!(matches!(err, UnitsError::Redefinition { .. }));
    }

    #[test]
    fn custom_registry_independent_of_global() {
        let mut r = UnitRegistry::empty();
        r.define(custom_def("smoot", &[])).unwrap();
        // global / default registry must not know the custom unit
        assert!(UnitRegistry::global().parse("smoot").is_err());
        assert!(UnitRegistry::new().parse("smoot").is_err());
    }

    #[test]
    fn spelled_out_prefixes_resolve_without_duplicate_definitions() {
        let r = UnitRegistry::new();
        for (long, short) in [
            ("nanometer", "nm"),
            ("femtosecond", "fs"),
            ("kilocalorie", "kcal"),
        ] {
            let long = r.parse(long).unwrap();
            let short = r.parse(short).unwrap();
            assert_eq!(long.dimension(), short.dimension());
            assert_eq!(long.factor_to(&short).unwrap(), 1.0);
        }
    }

    #[test]
    fn per_particle_molar_named_units_convert_to_energy_and_mass() {
        let r = UnitRegistry::new();
        let kcal = r.quantity(1.0, "kilocalorie_per_mole").unwrap();
        let kj = kcal.to(&r.parse("kilojoule_per_mole").unwrap()).unwrap();
        assert!((kj.value() - 4.184).abs() < 1e-12);
        assert!(kcal.to(&r.parse("eV").unwrap()).is_ok());
        assert!(
            r.quantity(1.0, "gram_per_mole")
                .unwrap()
                .to(&r.parse("Da").unwrap())
                .is_ok()
        );
    }

    #[test]
    fn lennard_jones_scales_are_native_convertible_units() {
        let source = UnitRegistry::new();
        let mass = source.quantity(39.948, "amu").unwrap();
        let sigma = source.quantity(3.405, "angstrom").unwrap();
        let epsilon = source.quantity(0.2381, "kilocalorie_per_mole").unwrap();
        let mut reduced = UnitRegistry::new();
        reduced.define_lj_units(&mass, &sigma, &epsilon).unwrap();

        let sigma_star = reduced
            .quantity(3.405, "angstrom")
            .unwrap()
            .to(&reduced.parse("lj_sigma").unwrap())
            .unwrap();
        assert!((sigma_star.value() - 1.0).abs() < 1e-12);
        let tau_ps = reduced
            .quantity(1.0, "lj_tau")
            .unwrap()
            .to(&reduced.parse("ps").unwrap())
            .unwrap();
        assert!((tau_ps.value() - 2.16).abs() < 0.01);
        let temperature = reduced
            .quantity(1.0, "lj_epsilon_over_kB")
            .unwrap()
            .to(&reduced.parse("K").unwrap())
            .unwrap();
        assert!((temperature.value() - 119.8).abs() < 0.5);
    }

    #[test]
    fn empty_has_si_base_only() {
        let r = UnitRegistry::empty();
        // SI base present
        assert!(r.parse("meter").is_ok());
        assert!(r.parse("kg").is_ok());
        assert!(r.parse("s").is_ok());
        assert!(r.parse("mol").is_ok());
        assert!(r.parse("K").is_ok());
        // MD-set units absent in empty registry
        assert!(r.parse("kcal").is_err());
        assert!(r.parse("angstrom").is_err());
        assert!(r.parse("eV").is_err());
    }

    #[test]
    fn new_has_md_units() {
        let r = UnitRegistry::new();
        assert!(r.parse("kcal").is_ok());
        assert!(r.parse("angstrom").is_ok());
        assert!(r.parse("eV").is_ok());
        assert!(r.parse("hartree").is_ok());
        assert!(r.parse("bohr").is_ok());
        assert!(r.parse("atm").is_ok());
    }

    #[test]
    fn quantity_convenience() {
        let r = UnitRegistry::new();
        let q = r.quantity(1.5, "kcal").unwrap();
        assert_eq!(q.value(), 1.5);
    }

    #[test]
    fn defined_unit_converts_correctly() {
        // define() must yield correct conversions, not merely parse success.
        // 1 smoot = 1.7018 m (MIT bridge convention).
        let mut r = UnitRegistry::empty();
        let mut def = custom_def("smoot", &[]);
        def.factor = 1.7018;
        r.define(def).unwrap();
        let smoot = r.parse("smoot").unwrap();
        let meter = r.parse("m").unwrap();
        let factor = smoot.factor_to(&meter).unwrap();
        assert!(
            ((factor - 1.7018) / 1.7018).abs() <= 1e-15,
            "smoot->m factor = {factor}"
        );
        // and through the Quantity path
        let q = r.quantity(2.0, "smoot").unwrap().to(&meter).unwrap();
        assert!(
            ((q.value() - 3.4036) / 3.4036).abs() <= 1e-15,
            "2 smoot = {} m",
            q.value()
        );
    }

    #[test]
    fn defined_prefixable_unit_accepts_prefix() {
        let mut r = UnitRegistry::empty();
        let mut def = custom_def("blip", &[]);
        def.factor = 2.0;
        def.dimension = Dimension::TIME;
        def.prefixable = true;
        r.define(def).unwrap();
        let kblip = r.parse("kblip").unwrap();
        let blip = r.parse("blip").unwrap();
        assert_eq!(kblip.factor_to(&blip).unwrap(), 1000.0);
        assert_eq!(kblip.dimension(), Dimension::TIME);
    }

    #[test]
    fn definitions_iterates_all_defs() {
        let r = UnitRegistry::empty();
        let base_count = r.definitions().count();
        assert!(base_count >= 8, "SI base set missing: {base_count}");
        let mut r = UnitRegistry::empty();
        r.define(custom_def("smoot", &[])).unwrap();
        assert!(r.definitions().any(|d| d.name == "smoot"));
        // preloaded registry strictly larger than base set
        assert!(UnitRegistry::new().definitions().count() > base_count);
    }
}
