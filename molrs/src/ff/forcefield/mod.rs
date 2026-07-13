//! Force field definition types.
//!
//! Provides a declarative layer for defining atom types, bond types, pair types,
//! etc. with their parameters. A [`ForceField`] holds [`Style`]s, each of which
//! holds typed parameter sets via [`StyleDefs`]. The forcefield can be compiled
//! into computational [`Potential`](super::potential::Potential) objects via
//! [`ForceField::to_potentials`].

pub mod gaff;
pub mod readers;
pub mod xml;

use std::collections::{HashMap, HashSet};

use molrs::store::frame::Frame;

// ---------------------------------------------------------------------------
// Params
// ---------------------------------------------------------------------------

/// Key-value parameter bag for type definitions.
///
/// Holds numeric params (`k`, `r0`, the numeric type `id`, …) and, separately,
/// string params (`element`, or any string metadata carried by convention as a
/// keyword param). Energy kernels read only the numeric side; the string side
/// preserves I/O metadata across the boundary.
#[derive(Debug, Clone, Default)]
pub struct Params {
    inner: HashMap<String, f64>,
    strings: HashMap<String, String>,
}

impl Params {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_pairs(pairs: &[(&str, f64)]) -> Self {
        let mut inner = HashMap::new();
        for &(k, v) in pairs {
            inner.insert(k.to_owned(), v);
        }
        Self {
            inner,
            strings: HashMap::new(),
        }
    }

    pub fn get(&self, key: &str) -> Option<f64> {
        self.inner.get(key).copied()
    }

    pub fn set(&mut self, key: &str, value: f64) {
        self.inner.insert(key.to_owned(), value);
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, f64)> + '_ {
        self.inner.iter().map(|(k, v)| (k.as_str(), *v))
    }

    // -- string params (element, and other string metadata by convention) --

    pub fn set_str(&mut self, key: &str, value: &str) {
        self.strings.insert(key.to_owned(), value.to_owned());
    }

    pub fn iter_strings(&self) -> impl Iterator<Item = (&str, &str)> + '_ {
        self.strings.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }
}

// ---------------------------------------------------------------------------
// Type definitions
// ---------------------------------------------------------------------------

/// Atom type definition.
#[derive(Debug, Clone)]
pub struct AtomType {
    pub name: String,
    pub params: Params,
}

/// Bond type definition (references two atom type names).
#[derive(Debug, Clone)]
pub struct BondType {
    pub name: String,
    pub itom: String,
    pub jtom: String,
    pub params: Params,
}

/// Angle type definition (references three atom type names).
#[derive(Debug, Clone)]
pub struct AngleType {
    pub name: String,
    pub itom: String,
    pub jtom: String,
    pub ktom: String,
    pub params: Params,
}

/// Dihedral type definition (references four atom type names).
#[derive(Debug, Clone)]
pub struct DihedralType {
    pub name: String,
    pub itom: String,
    pub jtom: String,
    pub ktom: String,
    pub ltom: String,
    pub params: Params,
}

/// Improper type definition (references four atom type names).
#[derive(Debug, Clone)]
pub struct ImproperType {
    pub name: String,
    pub itom: String,
    pub jtom: String,
    pub ktom: String,
    pub ltom: String,
    pub params: Params,
}

/// Pair type definition (one or two atom type names).
#[derive(Debug, Clone)]
pub struct PairType {
    pub name: String,
    pub itom: String,
    pub jtom: String,
    pub params: Params,
}

/// Each variant IS the category and holds only the relevant type definitions.
#[derive(Debug, Clone)]
pub enum StyleDefs {
    Atom(Vec<AtomType>),
    Bond(Vec<BondType>),
    Angle(Vec<AngleType>),
    Dihedral(Vec<DihedralType>),
    Improper(Vec<ImproperType>),
    Pair(Vec<PairType>),
    /// K-space styles (e.g. PME) have no per-type defs; all params at style level.
    KSpace,
}

impl StyleDefs {
    /// Category string for registry lookups.
    pub fn category(&self) -> &'static str {
        match self {
            Self::Atom(_) => "atom",
            Self::Bond(_) => "bond",
            Self::Angle(_) => "angle",
            Self::Dihedral(_) => "dihedral",
            Self::Improper(_) => "improper",
            Self::Pair(_) => "pair",
            Self::KSpace => "kspace",
        }
    }

    /// Collect `(type_name, params)` pairs for kernel construction.
    pub fn collect_type_params(&self) -> Vec<(String, Params)> {
        match self {
            Self::Atom(types) => types
                .iter()
                .map(|t| (t.name.clone(), t.params.clone()))
                .collect(),
            Self::Bond(types) => types
                .iter()
                .map(|t| (t.name.clone(), t.params.clone()))
                .collect(),
            Self::Angle(types) => types
                .iter()
                .map(|t| (t.name.clone(), t.params.clone()))
                .collect(),
            Self::Dihedral(types) => types
                .iter()
                .map(|t| (t.name.clone(), t.params.clone()))
                .collect(),
            Self::Improper(types) => types
                .iter()
                .map(|t| (t.name.clone(), t.params.clone()))
                .collect(),
            Self::Pair(types) => types
                .iter()
                .map(|t| (t.name.clone(), t.params.clone()))
                .collect(),
            // KSpace has no per-type defs; return a dummy entry so the compile loop works.
            Self::KSpace => vec![("*".into(), Params::new())],
        }
    }
}

// ---------------------------------------------------------------------------
// Style
// ---------------------------------------------------------------------------

/// A style groups a named interaction method with its type definitions.
#[derive(Debug, Clone)]
pub struct Style {
    pub name: String,
    pub params: Params,
    pub defs: StyleDefs,
}

impl Style {
    fn new(defs: StyleDefs, name: &str, params: Params) -> Self {
        Self {
            name: name.to_owned(),
            params,
            defs,
        }
    }

    /// Category string derived from the `StyleDefs` variant.
    pub fn category(&self) -> &'static str {
        self.defs.category()
    }

    // -- atom --

    pub fn def_atomtype(&mut self, name: &str, params: &[(&str, f64)]) -> &AtomType {
        let StyleDefs::Atom(types) = &mut self.defs else {
            panic!("def_atomtype called on non-atom style");
        };
        types.push(AtomType {
            name: name.to_owned(),
            params: Params::from_pairs(params),
        });
        types.last().unwrap()
    }

    pub fn get_atomtype(&self, name: &str) -> Option<&AtomType> {
        let StyleDefs::Atom(types) = &self.defs else {
            return None;
        };
        types.iter().find(|t| t.name == name)
    }

    // -- bond --

    pub fn def_bondtype(&mut self, itom: &str, jtom: &str, params: &[(&str, f64)]) -> &BondType {
        let StyleDefs::Bond(types) = &mut self.defs else {
            panic!("def_bondtype called on non-bond style");
        };
        let name = format!("{}-{}", itom, jtom);
        types.push(BondType {
            name,
            itom: itom.to_owned(),
            jtom: jtom.to_owned(),
            params: Params::from_pairs(params),
        });
        types.last().unwrap()
    }

    pub fn get_bondtype(&self, itom: &str, jtom: &str) -> Option<&BondType> {
        let StyleDefs::Bond(types) = &self.defs else {
            return None;
        };
        types
            .iter()
            .find(|t| (t.itom == itom && t.jtom == jtom) || (t.itom == jtom && t.jtom == itom))
    }

    // -- angle --

    pub fn def_angletype(
        &mut self,
        itom: &str,
        jtom: &str,
        ktom: &str,
        params: &[(&str, f64)],
    ) -> &AngleType {
        let StyleDefs::Angle(types) = &mut self.defs else {
            panic!("def_angletype called on non-angle style");
        };
        let name = format!("{}-{}-{}", itom, jtom, ktom);
        types.push(AngleType {
            name,
            itom: itom.to_owned(),
            jtom: jtom.to_owned(),
            ktom: ktom.to_owned(),
            params: Params::from_pairs(params),
        });
        types.last().unwrap()
    }

    // -- dihedral --

    pub fn def_dihedraltype(
        &mut self,
        itom: &str,
        jtom: &str,
        ktom: &str,
        ltom: &str,
        params: &[(&str, f64)],
    ) -> &DihedralType {
        let StyleDefs::Dihedral(types) = &mut self.defs else {
            panic!("def_dihedraltype called on non-dihedral style");
        };
        let name = format!("{}-{}-{}-{}", itom, jtom, ktom, ltom);
        types.push(DihedralType {
            name,
            itom: itom.to_owned(),
            jtom: jtom.to_owned(),
            ktom: ktom.to_owned(),
            ltom: ltom.to_owned(),
            params: Params::from_pairs(params),
        });
        types.last().unwrap()
    }

    // -- improper --

    pub fn def_impropertype(
        &mut self,
        itom: &str,
        jtom: &str,
        ktom: &str,
        ltom: &str,
        params: &[(&str, f64)],
    ) -> &ImproperType {
        let StyleDefs::Improper(types) = &mut self.defs else {
            panic!("def_impropertype called on non-improper style");
        };
        let name = format!("{}-{}-{}-{}", itom, jtom, ktom, ltom);
        types.push(ImproperType {
            name,
            itom: itom.to_owned(),
            jtom: jtom.to_owned(),
            ktom: ktom.to_owned(),
            ltom: ltom.to_owned(),
            params: Params::from_pairs(params),
        });
        types.last().unwrap()
    }

    // -- pair --

    pub fn def_pairtype(
        &mut self,
        itom: &str,
        jtom: Option<&str>,
        params: &[(&str, f64)],
    ) -> &PairType {
        let StyleDefs::Pair(types) = &mut self.defs else {
            panic!("def_pairtype called on non-pair style");
        };
        let jtom_str = jtom.unwrap_or(itom);
        let name = if itom == jtom_str {
            itom.to_owned()
        } else {
            format!("{}-{}", itom, jtom_str)
        };
        types.push(PairType {
            name,
            itom: itom.to_owned(),
            jtom: jtom_str.to_owned(),
            params: Params::from_pairs(params),
        });
        types.last().unwrap()
    }

    pub fn get_pairtype(&self, itom: &str, jtom: Option<&str>) -> Option<&PairType> {
        let StyleDefs::Pair(types) = &self.defs else {
            return None;
        };
        let jtom_str = jtom.unwrap_or(itom);
        types.iter().find(|t| {
            (t.itom == itom && t.jtom == jtom_str) || (t.itom == jtom_str && t.jtom == itom)
        })
    }

    // -- unified def_type --

    /// Define a type using the unified name format.
    ///
    /// The `name` encodes atom types based on the style's category:
    /// - **Atom**: `"A"` -> atom type name
    /// - **Bond**: `"A-B"` -> itom=A, jtom=B
    /// - **Angle**: `"A-B-C"` -> itom=A, jtom=B, ktom=C
    /// - **Pair**: `"A"` -> self-pair (itom=A, jtom=A); `"A-B"` -> cross-pair
    /// - **Dihedral/Improper**: `"A-B-C-D"` -> itom=A, jtom=B, ktom=C, ltom=D
    pub fn def_type(&mut self, name: &str, params: &[(&str, f64)]) -> &mut Self {
        // Infallible chaining form: a malformed name is a programmer error here
        // (literal call sites). The Python binding uses the fallible
        // [`Style::try_def_type`] so user input raises instead of aborting.
        if let Err(e) = self.try_def_type(name, params) {
            panic!("{e}");
        }
        self
    }

    /// Add a type to this style from its dash-form `name`, validating the part
    /// count against the style's category. This is the single source of truth
    /// for the type-name grammar; [`Self::def_type`] and
    /// [`ForceField::def_type`] both go through it.
    pub fn try_def_type(&mut self, name: &str, params: &[(&str, f64)]) -> Result<(), DefTypeError> {
        let parts: Vec<&str> = name.split('-').collect();
        let category = self.category();
        let arity = |expected: &'static str| DefTypeError::Arity {
            category,
            expected,
            name: name.to_string(),
            got: parts.len(),
        };
        match category {
            "atom" => {
                self.def_atomtype(name, params);
            }
            "bond" => {
                if parts.len() != 2 {
                    return Err(arity("A-B"));
                }
                self.def_bondtype(parts[0], parts[1], params);
            }
            "angle" => {
                if parts.len() != 3 {
                    return Err(arity("A-B-C"));
                }
                self.def_angletype(parts[0], parts[1], parts[2], params);
            }
            "dihedral" => {
                if parts.len() != 4 {
                    return Err(arity("A-B-C-D"));
                }
                self.def_dihedraltype(parts[0], parts[1], parts[2], parts[3], params);
            }
            "improper" => {
                if parts.len() != 4 {
                    return Err(arity("A-B-C-D"));
                }
                self.def_impropertype(parts[0], parts[1], parts[2], parts[3], params);
            }
            "pair" => match parts.len() {
                1 => {
                    self.def_pairtype(parts[0], None, params);
                }
                2 => {
                    self.def_pairtype(parts[0], Some(parts[1]), params);
                }
                _ => return Err(arity("A\" or \"A-B")),
            },
            "kspace" => return Err(DefTypeError::Unsupported(category)),
            other => return Err(DefTypeError::UnknownCategory(other.to_string())),
        }
        Ok(())
    }
}

/// Error from the dash-form type-name grammar used by [`Style::try_def_type`]
/// and [`ForceField::def_type`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DefTypeError {
    /// `name` has the wrong number of dash-separated parts for `category`.
    Arity {
        category: &'static str,
        expected: &'static str,
        name: String,
        got: usize,
    },
    /// The category accepts no per-type definitions (e.g. `kspace`).
    Unsupported(&'static str),
    /// Unknown style category string.
    UnknownCategory(String),
}

impl std::fmt::Display for DefTypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DefTypeError::Arity {
                category,
                expected,
                name,
                got,
            } => write!(
                f,
                "{category} type name must be \"{expected}\", got \"{name}\" ({got} parts)"
            ),
            DefTypeError::Unsupported(category) => {
                write!(f, "{category} styles do not support per-type definitions")
            }
            DefTypeError::UnknownCategory(category) => {
                write!(f, "unknown style category '{category}'")
            }
        }
    }
}

impl std::error::Error for DefTypeError {}

/// In-place mutators backing the Python handle-view layer (Style/Type views read
/// through [`collect_type_params`](StyleDefs::collect_type_params) and write
/// through these). Each operates on the type identified by its dash-form name.
impl Style {
    /// Endpoint atom-type names of the type named `name` (e.g. `["CT","CT"]`),
    /// or `None` if no such type. Atom/kspace styles return an empty vec.
    pub fn type_endpoints(&self, name: &str) -> Option<Vec<String>> {
        match &self.defs {
            StyleDefs::Atom(v) => v.iter().find(|t| t.name == name).map(|_| Vec::new()),
            StyleDefs::Bond(v) => v
                .iter()
                .find(|t| t.name == name)
                .map(|t| vec![t.itom.clone(), t.jtom.clone()]),
            StyleDefs::Angle(v) => v
                .iter()
                .find(|t| t.name == name)
                .map(|t| vec![t.itom.clone(), t.jtom.clone(), t.ktom.clone()]),
            StyleDefs::Dihedral(v) => v.iter().find(|t| t.name == name).map(|t| {
                vec![
                    t.itom.clone(),
                    t.jtom.clone(),
                    t.ktom.clone(),
                    t.ltom.clone(),
                ]
            }),
            StyleDefs::Improper(v) => v.iter().find(|t| t.name == name).map(|t| {
                vec![
                    t.itom.clone(),
                    t.jtom.clone(),
                    t.ktom.clone(),
                    t.ltom.clone(),
                ]
            }),
            StyleDefs::Pair(v) => v
                .iter()
                .find(|t| t.name == name)
                .map(|t| vec![t.itom.clone(), t.jtom.clone()]),
            StyleDefs::KSpace => None,
        }
    }

    /// Set (or add) a single param on the type named `name`. Returns `false` if
    /// no such type exists.
    pub fn set_type_param(&mut self, name: &str, key: &str, value: f64) -> bool {
        macro_rules! set_on {
            ($v:expr) => {{
                if let Some(t) = $v.iter_mut().find(|t| t.name == name) {
                    t.params.set(key, value);
                    return true;
                }
            }};
        }
        match &mut self.defs {
            StyleDefs::Atom(v) => set_on!(v),
            StyleDefs::Bond(v) => set_on!(v),
            StyleDefs::Angle(v) => set_on!(v),
            StyleDefs::Dihedral(v) => set_on!(v),
            StyleDefs::Improper(v) => set_on!(v),
            StyleDefs::Pair(v) => set_on!(v),
            StyleDefs::KSpace => {}
        }
        false
    }

    /// Set (or add) a single string param on the type named `name`. Returns
    /// `false` if no such type exists.
    pub fn set_type_str_param(&mut self, name: &str, key: &str, value: &str) -> bool {
        macro_rules! set_on {
            ($v:expr) => {{
                if let Some(t) = $v.iter_mut().find(|t| t.name == name) {
                    t.params.set_str(key, value);
                    return true;
                }
            }};
        }
        match &mut self.defs {
            StyleDefs::Atom(v) => set_on!(v),
            StyleDefs::Bond(v) => set_on!(v),
            StyleDefs::Angle(v) => set_on!(v),
            StyleDefs::Dihedral(v) => set_on!(v),
            StyleDefs::Improper(v) => set_on!(v),
            StyleDefs::Pair(v) => set_on!(v),
            StyleDefs::KSpace => {}
        }
        false
    }

    /// Rename every type named `old` to `new`. Returns the count renamed.
    pub fn rename_type(&mut self, old: &str, new: &str) -> usize {
        macro_rules! rename_in {
            ($v:expr) => {{
                let mut n = 0;
                for t in $v.iter_mut().filter(|t| t.name == old) {
                    t.name = new.to_owned();
                    n += 1;
                }
                n
            }};
        }
        match &mut self.defs {
            StyleDefs::Atom(v) => rename_in!(v),
            StyleDefs::Bond(v) => rename_in!(v),
            StyleDefs::Angle(v) => rename_in!(v),
            StyleDefs::Dihedral(v) => rename_in!(v),
            StyleDefs::Improper(v) => rename_in!(v),
            StyleDefs::Pair(v) => rename_in!(v),
            StyleDefs::KSpace => 0,
        }
    }

    /// Remove every type named `name`. Returns the count removed.
    pub fn remove_type(&mut self, name: &str) -> usize {
        macro_rules! remove_in {
            ($v:expr) => {{
                let before = $v.len();
                $v.retain(|t| t.name != name);
                before - $v.len()
            }};
        }
        match &mut self.defs {
            StyleDefs::Atom(v) => remove_in!(v),
            StyleDefs::Bond(v) => remove_in!(v),
            StyleDefs::Angle(v) => remove_in!(v),
            StyleDefs::Dihedral(v) => remove_in!(v),
            StyleDefs::Improper(v) => remove_in!(v),
            StyleDefs::Pair(v) => remove_in!(v),
            StyleDefs::KSpace => 0,
        }
    }
}

// ---------------------------------------------------------------------------
// ForceField
// ---------------------------------------------------------------------------

/// Top-level forcefield container holding styles and their type definitions.
///
/// # Example
///
/// ```
/// use molrs::ff::forcefield::ForceField;
///
/// let mut ff = ForceField::new("example");
/// ff.def_bondstyle("harmonic")
///     .def_type("A-B", &[("k0", 300.0), ("r0", 1.5)]);
/// ff.def_pairstyle("lj/cut", &[("cutoff", 10.0)])
///     .def_type("A", &[("epsilon", 0.5), ("sigma", 1.0)]);
///
/// // Compile into Potentials with a Frame containing topology
/// // let potentials = ff.to_potentials(&frame).unwrap();
/// ```
/// Per-nonbonded-kind 1-2 / 1-3 / 1-4 interaction scale weights — LAMMPS
/// `special_bonds` semantics, owned by the [`ForceField`].
///
/// A weight of `0.0` fully excludes that neighbour class; `1.0` leaves it at
/// full strength. molrs realises 1-2 / 1-3 *exclusion* by **omitting** those
/// pairs from the neighbour list (`intramolecular_pairs`), so the pair kernels
/// consume only the 1-4 weight (`[2]`) today; the 1-2 / 1-3 entries are stored
/// for completeness and for force fields that *scale* (rather than exclude)
/// close neighbours.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpecialBonds {
    /// LJ / van-der-Waals `[1-2, 1-3, 1-4]` scale weights.
    pub lj: [f64; 3],
    /// Coulomb `[1-2, 1-3, 1-4]` scale weights.
    pub coul: [f64; 3],
}

impl Default for SpecialBonds {
    /// Exclude 1-2 and 1-3 neighbours; leave 1-4 unscaled. Force-field readers
    /// override the 1-4 weights (Amber: lj `0.5`, coul `0.8333`).
    fn default() -> Self {
        Self {
            lj: [0.0, 0.0, 1.0],
            coul: [0.0, 0.0, 1.0],
        }
    }
}

impl SpecialBonds {
    /// The LJ 1-4 scale weight (the `[2]` entry of [`Self::lj`]).
    pub fn lj_14(&self) -> f64 {
        self.lj[2]
    }

    /// The Coulomb 1-4 scale weight (the `[2]` entry of [`Self::coul`]).
    pub fn coul_14(&self) -> f64 {
        self.coul[2]
    }
}

#[derive(Debug, Clone)]
pub struct ForceField {
    pub name: String,
    styles: Vec<Style>,
    special_bonds: SpecialBonds,
}

impl ForceField {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_owned(),
            styles: Vec::new(),
            special_bonds: SpecialBonds::default(),
        }
    }

    /// The force field's [`SpecialBonds`] 1-2 / 1-3 / 1-4 nonbonded scale
    /// weights. Pair kernels apply the 1-4 weight to flagged (`is_14`) pairs.
    pub fn special_bonds(&self) -> &SpecialBonds {
        &self.special_bonds
    }

    /// Replace the [`SpecialBonds`] weights. Force-field readers set these per
    /// force field (Amber/GAFF, OPLS, …); the default excludes 1-2/1-3 and
    /// leaves 1-4 unscaled.
    pub fn set_special_bonds(&mut self, special_bonds: SpecialBonds) {
        self.special_bonds = special_bonds;
    }

    // -- def_*style: register or retrieve existing style --

    fn def_style(&mut self, defs: StyleDefs, name: &str, params: Params) -> &mut Style {
        let category = defs.category();
        if let Some(idx) = self
            .styles
            .iter()
            .position(|s| s.category() == category && s.name == name)
        {
            return &mut self.styles[idx];
        }
        self.styles.push(Style::new(defs, name, params));
        self.styles.last_mut().unwrap()
    }

    pub fn def_atomstyle(&mut self, name: &str) -> &mut Style {
        self.def_style(StyleDefs::Atom(Vec::new()), name, Params::new())
    }

    pub fn def_bondstyle(&mut self, name: &str) -> &mut Style {
        self.def_style(StyleDefs::Bond(Vec::new()), name, Params::new())
    }

    pub fn def_anglestyle(&mut self, name: &str) -> &mut Style {
        self.def_style(StyleDefs::Angle(Vec::new()), name, Params::new())
    }

    pub fn def_dihedralstyle(&mut self, name: &str) -> &mut Style {
        self.def_style(StyleDefs::Dihedral(Vec::new()), name, Params::new())
    }

    pub fn def_improperstyle(&mut self, name: &str) -> &mut Style {
        self.def_style(StyleDefs::Improper(Vec::new()), name, Params::new())
    }

    pub fn def_pairstyle(&mut self, name: &str, params: &[(&str, f64)]) -> &mut Style {
        self.def_style(
            StyleDefs::Pair(Vec::new()),
            name,
            Params::from_pairs(params),
        )
    }

    pub fn def_kspacestyle(&mut self, name: &str, params: &[(&str, f64)]) -> &mut Style {
        self.def_style(StyleDefs::KSpace, name, Params::from_pairs(params))
    }

    /// Define a type in one call: ensure the `category` style named `style`
    /// exists, then add the type whose dash-form `name` is validated against
    /// the category's arity. Owns the type-name grammar so bindings forward the
    /// raw `name` instead of re-parsing it.
    ///
    /// `category` is one of `atom`/`bond`/`angle`/`dihedral`/`improper`/`pair`.
    pub fn def_type(
        &mut self,
        category: &str,
        style: &str,
        name: &str,
        params: &[(&str, f64)],
    ) -> Result<(), DefTypeError> {
        let target = match category {
            "atom" => self.def_atomstyle(style),
            "bond" => self.def_bondstyle(style),
            "angle" => self.def_anglestyle(style),
            "dihedral" => self.def_dihedralstyle(style),
            "improper" => self.def_improperstyle(style),
            "pair" => self.def_pairstyle(style, &[]),
            "kspace" => return Err(DefTypeError::Unsupported("kspace")),
            other => return Err(DefTypeError::UnknownCategory(other.to_string())),
        };
        target.try_def_type(name, params)
    }

    // -- with_* builder pattern (consumes and returns self) --

    pub fn with_atomstyle(mut self, name: &str) -> Self {
        self.def_atomstyle(name);
        self
    }

    pub fn with_bondstyle(mut self, name: &str) -> Self {
        self.def_bondstyle(name);
        self
    }

    pub fn with_anglestyle(mut self, name: &str) -> Self {
        self.def_anglestyle(name);
        self
    }

    pub fn with_pairstyle(mut self, name: &str, params: &[(&str, f64)]) -> Self {
        self.def_pairstyle(name, params);
        self
    }

    // -- queries --

    pub fn styles(&self) -> &[Style] {
        &self.styles
    }

    pub fn get_style(&self, category: &str, name: &str) -> Option<&Style> {
        self.styles
            .iter()
            .find(|s| s.category() == category && s.name == name)
    }

    /// Mutable style lookup, backing the Python handle-view writes.
    pub fn get_style_mut(&mut self, category: &str, name: &str) -> Option<&mut Style> {
        self.styles
            .iter_mut()
            .find(|s| s.category() == category && s.name == name)
    }

    /// Remove the style identified by `(category, name)`. Returns whether one was
    /// removed.
    pub fn remove_style(&mut self, category: &str, name: &str) -> bool {
        let before = self.styles.len();
        self.styles
            .retain(|s| !(s.category() == category && s.name == name));
        before != self.styles.len()
    }

    pub fn get_styles(&self, category: &str) -> Vec<&Style> {
        self.styles
            .iter()
            .filter(|s| s.category() == category)
            .collect()
    }

    pub fn get_atomtypes(&self) -> Vec<&AtomType> {
        self.styles
            .iter()
            .filter_map(|s| match &s.defs {
                StyleDefs::Atom(types) => Some(types.iter()),
                _ => None,
            })
            .flatten()
            .collect()
    }

    pub fn get_bondtypes(&self) -> Vec<&BondType> {
        self.styles
            .iter()
            .filter_map(|s| match &s.defs {
                StyleDefs::Bond(types) => Some(types.iter()),
                _ => None,
            })
            .flatten()
            .collect()
    }

    pub fn get_angletypes(&self) -> Vec<&AngleType> {
        self.styles
            .iter()
            .filter_map(|s| match &s.defs {
                StyleDefs::Angle(types) => Some(types.iter()),
                _ => None,
            })
            .flatten()
            .collect()
    }

    pub fn get_pairtypes(&self) -> Vec<&PairType> {
        self.styles
            .iter()
            .filter_map(|s| match &s.defs {
                StyleDefs::Pair(types) => Some(types.iter()),
                _ => None,
            })
            .flatten()
            .collect()
    }

    pub fn get_dihedraltypes(&self) -> Vec<&DihedralType> {
        self.styles
            .iter()
            .filter_map(|s| match &s.defs {
                StyleDefs::Dihedral(types) => Some(types.iter()),
                _ => None,
            })
            .flatten()
            .collect()
    }

    pub fn get_impropertypes(&self) -> Vec<&ImproperType> {
        self.styles
            .iter()
            .filter_map(|s| match &s.defs {
                StyleDefs::Improper(types) => Some(types.iter()),
                _ => None,
            })
            .flatten()
            .collect()
    }

    /// Project this force field onto the types a typed [`Frame`] actually uses.
    ///
    /// Reading a full force field (e.g. OPLS with ~900 atom types) yields a
    /// large `ForceField`, but a concrete typed structure references only a
    /// small fraction of those types. `subset` returns a fresh `ForceField`
    /// restricted to exactly the types named in `frame`'s per-block `type`
    /// columns, leaving `self` unmodified.
    ///
    /// The projection is a pure set operation. For each topology category, the
    /// used type-name set is read from the matching block's `type` column
    /// (`atoms`/`bonds`/`angles`/`dihedrals`/`impropers`); a category whose
    /// block or `type` column is absent contributes an empty set. Each `Style`
    /// keeps only the `*Type` entries whose `name` is in that set.
    ///
    /// Pair types are not keyed by a topology block: a [`PairType`] is kept iff
    /// **both** of its endpoint atom-type names (`itom` and `jtom`) are in the
    /// used atom-type set. This one predicate covers self-interaction pairs
    /// (`itom == jtom`) and explicit cross pairs uniformly.
    ///
    /// Styles left with no surviving types are dropped (a `KSpace` style, which
    /// legitimately carries no per-type defs, is preserved verbatim). Type
    /// names are copied unchanged — no renumbering.
    pub fn subset(&self, frame: &Frame) -> ForceField {
        let used = |block: &str| -> HashSet<String> {
            frame
                .get(block)
                .and_then(|b| b.get_string("type"))
                .map(|arr| arr.iter().cloned().collect())
                .unwrap_or_default()
        };
        let used_atoms = used("atoms");
        let used_bonds = used("bonds");
        let used_angles = used("angles");
        let used_dihedrals = used("dihedrals");
        let used_impropers = used("impropers");

        let mut out = ForceField::new(&self.name);
        out.special_bonds = self.special_bonds;
        for style in &self.styles {
            let defs = match &style.defs {
                StyleDefs::Atom(types) => StyleDefs::Atom(
                    types
                        .iter()
                        .filter(|t| used_atoms.contains(&t.name))
                        .cloned()
                        .collect(),
                ),
                StyleDefs::Bond(types) => StyleDefs::Bond(
                    types
                        .iter()
                        .filter(|t| used_bonds.contains(&t.name))
                        .cloned()
                        .collect(),
                ),
                StyleDefs::Angle(types) => StyleDefs::Angle(
                    types
                        .iter()
                        .filter(|t| used_angles.contains(&t.name))
                        .cloned()
                        .collect(),
                ),
                StyleDefs::Dihedral(types) => StyleDefs::Dihedral(
                    types
                        .iter()
                        .filter(|t| used_dihedrals.contains(&t.name))
                        .cloned()
                        .collect(),
                ),
                StyleDefs::Improper(types) => StyleDefs::Improper(
                    types
                        .iter()
                        .filter(|t| used_impropers.contains(&t.name))
                        .cloned()
                        .collect(),
                ),
                StyleDefs::Pair(types) => StyleDefs::Pair(
                    types
                        .iter()
                        .filter(|t| used_atoms.contains(&t.itom) && used_atoms.contains(&t.jtom))
                        .cloned()
                        .collect(),
                ),
                StyleDefs::KSpace => StyleDefs::KSpace,
            };

            let keep = match &defs {
                StyleDefs::Atom(t) => !t.is_empty(),
                StyleDefs::Bond(t) => !t.is_empty(),
                StyleDefs::Angle(t) => !t.is_empty(),
                StyleDefs::Dihedral(t) => !t.is_empty(),
                StyleDefs::Improper(t) => !t.is_empty(),
                StyleDefs::Pair(t) => !t.is_empty(),
                StyleDefs::KSpace => true,
            };
            if keep {
                out.styles
                    .push(Style::new(defs, &style.name, style.params.clone()));
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::store::block::Block;

    #[test]
    fn test_params() {
        let p = Params::from_pairs(&[("k0", 300.0), ("r0", 1.5)]);
        assert_eq!(p.get("k0"), Some(300.0));
        assert_eq!(p.get("r0"), Some(1.5));
        assert_eq!(p.get("missing"), None);
    }

    #[test]
    fn test_def_atomstyle_and_types() {
        let mut ff = ForceField::new("test");
        let style = ff.def_atomstyle("full");
        style.def_atomtype("CT", &[("mass", 12.011), ("charge", -0.12)]);
        style.def_atomtype("HC", &[("mass", 1.008), ("charge", 0.06)]);

        let style = ff.get_style("atom", "full").unwrap();
        let StyleDefs::Atom(types) = &style.defs else {
            panic!("expected Atom defs");
        };
        assert_eq!(types.len(), 2);

        let ct = style.get_atomtype("CT").unwrap();
        assert_eq!(ct.params.get("mass"), Some(12.011));
        assert_eq!(ct.params.get("charge"), Some(-0.12));
    }

    #[test]
    fn test_def_bondstyle_and_types() {
        let mut ff = ForceField::new("test");
        let style = ff.def_bondstyle("harmonic");
        style.def_bondtype("CT", "CT", &[("k0", 268.0), ("r0", 1.529)]);
        style.def_bondtype("CT", "HC", &[("k0", 340.0), ("r0", 1.09)]);

        let style = ff.get_style("bond", "harmonic").unwrap();
        let StyleDefs::Bond(types) = &style.defs else {
            panic!("expected Bond defs");
        };
        assert_eq!(types.len(), 2);

        let bt = style.get_bondtype("CT", "CT").unwrap();
        assert_eq!(bt.params.get("k0"), Some(268.0));

        // Order-independent lookup
        let bt2 = style.get_bondtype("HC", "CT").unwrap();
        assert_eq!(bt2.params.get("r0"), Some(1.09));
    }

    #[test]
    fn test_def_anglestyle_and_types() {
        let mut ff = ForceField::new("test");
        let style = ff.def_anglestyle("harmonic");
        style.def_angletype("HC", "CT", "HC", &[("k0", 33.0), ("theta0", 107.8)]);

        let types = ff.get_angletypes();
        assert_eq!(types.len(), 1);
        assert_eq!(types[0].params.get("theta0"), Some(107.8));
    }

    #[test]
    fn test_def_pairstyle_and_types() {
        let mut ff = ForceField::new("test");
        let style = ff.def_pairstyle("lj/cut", &[("cutoff", 10.0)]);
        style.def_pairtype("CT", None, &[("epsilon", 0.066), ("sigma", 3.5)]);
        style.def_pairtype("CT", Some("OH"), &[("epsilon", 0.1), ("sigma", 3.3)]);

        let style = ff.get_style("pair", "lj/cut").unwrap();
        assert_eq!(style.params.get("cutoff"), Some(10.0));
        let StyleDefs::Pair(types) = &style.defs else {
            panic!("expected Pair defs");
        };
        assert_eq!(types.len(), 2);

        // self-interaction
        let pt = style.get_pairtype("CT", None).unwrap();
        assert_eq!(pt.itom, "CT");
        assert_eq!(pt.jtom, "CT");

        // cross-interaction (order-independent)
        let pt2 = style.get_pairtype("OH", Some("CT")).unwrap();
        assert_eq!(pt2.params.get("epsilon"), Some(0.1));
    }

    #[test]
    fn test_duplicate_style_returns_existing() {
        let mut ff = ForceField::new("test");
        ff.def_bondstyle("harmonic")
            .def_bondtype("A", "B", &[("k0", 1.0), ("r0", 1.0)]);

        // Second call returns the same style, not a new one
        ff.def_bondstyle("harmonic")
            .def_bondtype("C", "D", &[("k0", 2.0), ("r0", 2.0)]);

        let styles = ff.get_styles("bond");
        assert_eq!(styles.len(), 1);
        let StyleDefs::Bond(types) = &styles[0].defs else {
            panic!("expected Bond defs");
        };
        assert_eq!(types.len(), 2);
    }

    #[test]
    fn test_builder_pattern() {
        let ff = ForceField::new("TIP3P")
            .with_atomstyle("full")
            .with_bondstyle("harmonic")
            .with_pairstyle("lj/cut", &[("cutoff", 10.0)]);

        assert_eq!(ff.styles().len(), 3);
        assert!(ff.get_style("atom", "full").is_some());
        assert!(ff.get_style("bond", "harmonic").is_some());
        assert!(ff.get_style("pair", "lj/cut").is_some());
    }

    // --- def_type unified tests ---

    #[test]
    fn test_def_type_bond() {
        let mut ff = ForceField::new("test");
        let style = ff.def_bondstyle("harmonic");
        style.def_type("CT-OH", &[("k0", 300.0), ("r0", 1.4)]);

        let bt = style.get_bondtype("CT", "OH").unwrap();
        assert_eq!(bt.itom, "CT");
        assert_eq!(bt.jtom, "OH");
        assert_eq!(bt.params.get("k0"), Some(300.0));
        assert_eq!(bt.params.get("r0"), Some(1.4));
    }

    #[test]
    fn test_def_type_angle() {
        let mut ff = ForceField::new("test");
        let style = ff.def_anglestyle("harmonic");
        style.def_type("HC-CT-HC", &[("k0", 33.0), ("theta0", 107.8)]);

        let StyleDefs::Angle(types) = &style.defs else {
            panic!("expected Angle defs");
        };
        assert_eq!(types.len(), 1);
        assert_eq!(types[0].itom, "HC");
        assert_eq!(types[0].jtom, "CT");
        assert_eq!(types[0].ktom, "HC");
        assert_eq!(types[0].params.get("theta0"), Some(107.8));
    }

    #[test]
    fn test_def_type_pair_self() {
        let mut ff = ForceField::new("test");
        let style = ff.def_pairstyle("lj/cut", &[("cutoff", 10.0)]);
        style.def_type("Ar", &[("epsilon", 1.0), ("sigma", 3.4)]);

        let pt = style.get_pairtype("Ar", None).unwrap();
        assert_eq!(pt.itom, "Ar");
        assert_eq!(pt.jtom, "Ar");
        assert_eq!(pt.params.get("epsilon"), Some(1.0));
    }

    #[test]
    fn test_def_type_pair_cross() {
        let mut ff = ForceField::new("test");
        let style = ff.def_pairstyle("lj/cut", &[("cutoff", 10.0)]);
        style.def_type("CT-OH", &[("epsilon", 0.1), ("sigma", 3.3)]);

        let pt = style.get_pairtype("CT", Some("OH")).unwrap();
        assert_eq!(pt.itom, "CT");
        assert_eq!(pt.jtom, "OH");
    }

    #[test]
    fn test_def_type_dihedral() {
        let mut ff = ForceField::new("test");
        let style = ff.def_dihedralstyle("opls");
        style.def_type("HC-CT-CT-HC", &[("k1", 0.0), ("k2", 0.0), ("k3", 0.3)]);

        let StyleDefs::Dihedral(types) = &style.defs else {
            panic!("expected Dihedral defs");
        };
        assert_eq!(types.len(), 1);
        assert_eq!(types[0].itom, "HC");
        assert_eq!(types[0].ltom, "HC");
    }

    #[test]
    fn test_def_type_chaining() {
        let mut ff = ForceField::new("test");
        let style = ff.def_bondstyle("harmonic");
        style
            .def_type("A-B", &[("k0", 1.0), ("r0", 1.0)])
            .def_type("C-D", &[("k0", 2.0), ("r0", 2.0)]);

        let StyleDefs::Bond(types) = &style.defs else {
            panic!("expected Bond defs");
        };
        assert_eq!(types.len(), 2);
    }

    #[test]
    #[should_panic(expected = "bond type name must be")]
    fn test_def_type_bond_invalid_format() {
        let mut ff = ForceField::new("test");
        let style = ff.def_bondstyle("harmonic");
        style.def_type("CT", &[("k0", 300.0), ("r0", 1.4)]);
    }

    #[test]
    fn test_get_all_types() {
        let mut ff = ForceField::new("test");

        let style = ff.def_atomstyle("full");
        style.def_atomtype("CT", &[("mass", 12.0)]);
        style.def_atomtype("OH", &[("mass", 16.0)]);

        let style = ff.def_bondstyle("harmonic");
        style.def_bondtype("CT", "OH", &[("k0", 300.0), ("r0", 1.4)]);

        assert_eq!(ff.get_atomtypes().len(), 2);
        assert_eq!(ff.get_bondtypes().len(), 1);
        assert_eq!(ff.get_pairtypes().len(), 0);
    }

    // -- subset projection ---------------------------------------------------

    /// A force field spanning more types than any single fixture frame uses:
    /// atom {CT, HC, OH}, bond {CT-HC, CT-OH}, angle {HC-CT-HC, HC-CT-OH},
    /// dihedral {HC-CT-CT-HC}, improper {CT-CT-CT-OH}, pair {CT self, HC self,
    /// OH self, CT-OH cross}.
    fn full_ff() -> ForceField {
        let mut ff = ForceField::new("fixture");
        let a = ff.def_atomstyle("full");
        a.def_atomtype("CT", &[("mass", 12.011)]);
        a.def_atomtype("HC", &[("mass", 1.008)]);
        a.def_atomtype("OH", &[("mass", 15.999)]);
        let b = ff.def_bondstyle("harmonic");
        b.def_bondtype("CT", "HC", &[("k0", 340.0), ("r0", 1.09)]);
        b.def_bondtype("CT", "OH", &[("k0", 320.0), ("r0", 1.41)]);
        let ang = ff.def_anglestyle("harmonic");
        ang.def_angletype("HC", "CT", "HC", &[("k0", 33.0), ("theta0", 107.8)]);
        ang.def_angletype("HC", "CT", "OH", &[("k0", 35.0), ("theta0", 109.5)]);
        let dih = ff.def_dihedralstyle("opls");
        dih.def_dihedraltype("HC", "CT", "CT", "HC", &[("k1", 0.0)]);
        let imp = ff.def_improperstyle("cvff");
        imp.def_impropertype("CT", "CT", "CT", "OH", &[("k", 1.0)]);
        let p = ff.def_pairstyle("lj/cut", &[("cutoff", 10.0)]);
        p.def_pairtype("CT", None, &[("epsilon", 0.066), ("sigma", 3.5)]);
        p.def_pairtype("HC", None, &[("epsilon", 0.03), ("sigma", 2.5)]);
        p.def_pairtype("OH", None, &[("epsilon", 0.17), ("sigma", 3.12)]);
        p.def_pairtype("CT", Some("OH"), &[("epsilon", 0.1), ("sigma", 3.3)]);
        ff
    }

    fn type_block(names: &[&str]) -> Block {
        use ndarray::Array1;
        let mut block = Block::new();
        let col: Vec<String> = names.iter().map(|s| s.to_string()).collect();
        block
            .insert("type", Array1::from_vec(col).into_dyn())
            .unwrap();
        block
    }

    /// Typed frame using only: atoms {CT, HC}, bonds {CT-HC}, angles {HC-CT-HC},
    /// no dihedrals/impropers blocks. OH is never referenced.
    fn partial_frame() -> Frame {
        let mut frame = Frame::new();
        frame.insert("atoms", type_block(&["CT", "HC", "CT", "HC"]));
        frame.insert("bonds", type_block(&["CT-HC"]));
        frame.insert("angles", type_block(&["HC-CT-HC"]));
        frame
    }

    #[test]
    fn test_subset_does_not_mutate_self() {
        let ff = full_ff();
        let n_atoms = ff.get_atomtypes().len();
        let n_bonds = ff.get_bondtypes().len();
        let _ = ff.subset(&partial_frame());
        assert_eq!(ff.get_atomtypes().len(), n_atoms);
        assert_eq!(ff.get_bondtypes().len(), n_bonds);
    }

    #[test]
    fn test_subset_per_category_exact_match() {
        let mini = full_ff().subset(&partial_frame());

        let atoms: HashSet<&str> = mini
            .get_atomtypes()
            .iter()
            .map(|t| t.name.as_str())
            .collect();
        assert_eq!(atoms, HashSet::from(["CT", "HC"]));

        let bonds: HashSet<&str> = mini
            .get_bondtypes()
            .iter()
            .map(|t| t.name.as_str())
            .collect();
        assert_eq!(bonds, HashSet::from(["CT-HC"]));

        let angles: HashSet<&str> = mini
            .get_angletypes()
            .iter()
            .map(|t| t.name.as_str())
            .collect();
        assert_eq!(angles, HashSet::from(["HC-CT-HC"]));

        // dihedrals/impropers blocks absent -> empty
        assert!(mini.get_dihedraltypes().is_empty());
        assert!(mini.get_impropertypes().is_empty());
    }

    #[test]
    fn test_subset_pairtype_both_endpoints_predicate() {
        let mini = full_ff().subset(&partial_frame());
        let pairs: HashSet<&str> = mini
            .get_pairtypes()
            .iter()
            .map(|t| t.name.as_str())
            .collect();
        // CT, HC self-pairs survive (both endpoints used); OH self-pair dropped
        // (OH unused); CT-OH cross dropped (one endpoint unused).
        assert_eq!(pairs, HashSet::from(["CT", "HC"]));
    }

    #[test]
    fn test_subset_drops_empty_styles() {
        let mini = full_ff().subset(&partial_frame());
        // dihedral/improper styles end up empty and are dropped entirely.
        assert!(mini.get_style("dihedral", "opls").is_none());
        assert!(mini.get_style("improper", "cvff").is_none());
        // a referenced style survives.
        assert!(mini.get_style("bond", "harmonic").is_some());
    }

    #[test]
    fn test_subset_preserves_names_verbatim() {
        let mini = full_ff().subset(&partial_frame());
        assert!(mini.get_atomtypes().iter().any(|t| t.name == "CT"));
        assert!(mini.get_bondtypes().iter().any(|t| t.name == "CT-HC"));
    }

    #[test]
    fn test_subset_zero_overlap_yields_empty() {
        let mut frame = Frame::new();
        frame.insert("atoms", type_block(&["XX", "ZZ"]));
        let mini = full_ff().subset(&frame);
        assert!(mini.get_atomtypes().is_empty());
        assert!(mini.get_pairtypes().is_empty());
        // every style had zero surviving types -> no styles remain.
        assert!(mini.styles().is_empty());
    }

    #[test]
    fn test_subset_missing_block_treated_as_empty() {
        // frame with atoms only -> bond/angle/etc categories empty, no panic.
        let mut frame = Frame::new();
        frame.insert("atoms", type_block(&["CT", "OH"]));
        let mini = full_ff().subset(&frame);
        let atoms: HashSet<&str> = mini
            .get_atomtypes()
            .iter()
            .map(|t| t.name.as_str())
            .collect();
        assert_eq!(atoms, HashSet::from(["CT", "OH"]));
        assert!(mini.get_bondtypes().is_empty());
        // CT, OH self-pairs and CT-OH cross all survive (all endpoints used).
        let pairs: HashSet<&str> = mini
            .get_pairtypes()
            .iter()
            .map(|t| t.name.as_str())
            .collect();
        assert_eq!(pairs, HashSet::from(["CT", "OH", "CT-OH"]));
    }
}
