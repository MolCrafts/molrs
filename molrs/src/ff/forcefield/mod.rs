//! Force field definition types.
//!
//! Provides a declarative layer for defining atom types, bond types, pair types,
//! etc. with their parameters. A [`ForceField`] holds [`Style`]s, each of which
//! holds typed parameter sets via [`StyleDefs`]. The forcefield is compiled
//! into computational [`Potential`](super::potential::Potential) objects by
//! [`PotentialCompiler`](super::potential::PotentialCompiler).

pub mod lammps_units;
pub mod mixing;
pub mod readers;
pub(crate) mod torsion;
pub mod writers;
pub mod xml;

use std::collections::HashMap;

use molrs::system::bond_weights::BondDistanceWeights;

// ---------------------------------------------------------------------------
// Params
// ---------------------------------------------------------------------------

/// Key-value parameter bag for type definitions.
///
/// Holds numeric params (`k`, `r0`, the numeric type `id`, …) and, separately,
/// string params (`element`, or any string metadata carried by convention as a
/// keyword param). Energy kernels read only the numeric side; the string side
/// preserves I/O metadata across the boundary.
///
/// Equality is exact on both sides (the same keys, `f64` values equal under
/// `==` with no tolerance, equal strings): it decides whether a re-definition
/// is the same definition, which is a question of identity, not closeness.
#[derive(Debug, Clone, Default, PartialEq)]
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

    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.strings.get(key).map(String::as_str)
    }

    pub fn iter_strings(&self) -> impl Iterator<Item = (&str, &str)> + '_ {
        self.strings.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }
}

// ---------------------------------------------------------------------------
// Type definitions
// ---------------------------------------------------------------------------

/// Atom type definition.
#[derive(Debug, Clone, PartialEq)]
pub struct AtomType {
    pub name: String,
    pub params: Params,
}

/// Bond type definition (references two atom type names).
#[derive(Debug, Clone, PartialEq)]
pub struct BondType {
    pub name: String,
    pub itom: String,
    pub jtom: String,
    pub params: Params,
}

/// Angle type definition (references three atom type names).
#[derive(Debug, Clone, PartialEq)]
pub struct AngleType {
    pub name: String,
    pub itom: String,
    pub jtom: String,
    pub ktom: String,
    pub params: Params,
}

/// Dihedral type definition (references four atom type names).
#[derive(Debug, Clone, PartialEq)]
pub struct DihedralType {
    pub name: String,
    pub itom: String,
    pub jtom: String,
    pub ktom: String,
    pub ltom: String,
    pub params: Params,
}

/// Improper type definition (references four atom type names).
#[derive(Debug, Clone, PartialEq)]
pub struct ImproperType {
    pub name: String,
    pub itom: String,
    pub jtom: String,
    pub ktom: String,
    pub ltom: String,
    pub params: Params,
}

/// Pair type definition (one or two atom type names).
#[derive(Debug, Clone, PartialEq)]
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
}

impl StyleDefs {
    /// No definitions, under `category` (`atom`/`bond`/`angle`/`dihedral`/
    /// `improper`/`pair`); anything else is `Err(DefError::UnknownCategory)`.
    fn empty(category: &str) -> Result<Self, DefError> {
        Ok(match category {
            "atom" => Self::Atom(Vec::new()),
            "bond" => Self::Bond(Vec::new()),
            "angle" => Self::Angle(Vec::new()),
            "dihedral" => Self::Dihedral(Vec::new()),
            "improper" => Self::Improper(Vec::new()),
            "pair" => Self::Pair(Vec::new()),
            other => return Err(DefError::UnknownCategory(other.to_owned())),
        })
    }

    /// Category string for registry lookups.
    pub fn category(&self) -> &'static str {
        match self {
            Self::Atom(_) => "atom",
            Self::Bond(_) => "bond",
            Self::Angle(_) => "angle",
            Self::Dihedral(_) => "dihedral",
            Self::Improper(_) => "improper",
            Self::Pair(_) => "pair",
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
        }
    }
}

impl StyleDefs {
    /// The rows a kernel resolves parameters from: a bonded or atom type by its
    /// name (the label a frame row carries), a pair type by the atom types it
    /// was defined between. A pair kernel meets an atom-type pair, never a
    /// pair-type name, so the key is built from the endpoints
    /// ([`TypeName::pair`](molrs::store::type_labels::TypeName::pair)) — a type
    /// named anything is found through them.
    pub fn kernel_type_params(&self) -> Result<Vec<(String, Params)>, String> {
        match self {
            Self::Pair(types) => types
                .iter()
                .map(|t| {
                    let key = molrs::store::type_labels::TypeName::pair(&t.itom, &t.jtom)?;
                    Ok((key.as_str().to_owned(), t.params.clone()))
                })
                .collect(),
            _ => Ok(self.collect_type_params()),
        }
    }
}

// ---------------------------------------------------------------------------
// Style
// ---------------------------------------------------------------------------

/// A style groups a named interaction method with its type definitions.
///
/// A style is identified by its `(category, name)` pair. It is defined through
/// [`ForceField::def_style`], and its types through [`Style::def_type`] (name
/// with its endpoints given). The fields are private: a writable name could collide with another style, and a
/// direct push onto the definitions would bypass the arity check and the
/// conflict rule.
///
/// # The conflict rule
///
/// A type is identified by `(category, style, name)`. Defining a name this
/// style already holds with the same endpoints and exactly equal [`Params`] is
/// a no-op; anything else is `Err(DefError::TypeConflict)` and leaves the first
/// definition in place. The edits `set_type_param`, `set_type_str_param` and
/// `remove_type` change an existing definition and are outside the rule;
/// `rename_type` can land on an existing name and therefore carries it.
#[derive(Debug, Clone)]
pub struct Style {
    name: String,
    params: Params,
    defs: StyleDefs,
}

impl Style {
    fn new(defs: StyleDefs, name: &str, params: Params) -> Self {
        Self {
            name: name.to_owned(),
            params,
            defs,
        }
    }

    /// The style name (e.g. `harmonic`, `lj/cut`).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Style-level params, numeric and string (`cutoff`, `mixing`, …).
    pub fn params(&self) -> &Params {
        &self.params
    }

    /// The type definitions this style holds.
    pub fn defs(&self) -> &StyleDefs {
        &self.defs
    }

    /// Category string derived from the `StyleDefs` variant.
    pub fn category(&self) -> &'static str {
        self.defs.category()
    }

    pub fn get_atomtype(&self, name: &str) -> Option<&AtomType> {
        let StyleDefs::Atom(types) = &self.defs else {
            return None;
        };
        types.iter().find(|t| t.name == name)
    }

    pub fn get_bondtype(&self, itom: &str, jtom: &str) -> Option<&BondType> {
        let StyleDefs::Bond(types) = &self.defs else {
            return None;
        };
        types
            .iter()
            .find(|t| (t.itom == itom && t.jtom == jtom) || (t.itom == jtom && t.jtom == itom))
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

    /// Define a type named `name` on the given `endpoints` (atom-type names).
    ///
    /// The name is an opaque identifier, stored verbatim: it may be built from
    /// the endpoint labels ([`TypeName::join`](molrs::store::type_labels::TypeName::join)
    /// is the convention) but it is never read back into endpoints — a `-` in
    /// a name is just a character. The endpoint count follows the category:
    /// an atom style takes none; a pair style one (a self pair) or two; bond
    /// two; angle three; dihedral and improper four. Any other count is
    /// `Err(DefError::Arity)` and never panics. Re-defining a stored name
    /// follows the conflict rule (see [`Style`]): identical is a no-op,
    /// different is `Err(DefError::TypeConflict)`.
    pub fn def_type(
        &mut self,
        name: &str,
        endpoints: &[&str],
        params: Params,
    ) -> Result<&mut Self, DefError> {
        if !self.accepts_endpoint_count(endpoints.len()) {
            return Err(DefError::Arity {
                category: self.category(),
                expected: self.endpoint_arity(),
                name: name.to_owned(),
                got: endpoints.len(),
            });
        }
        // An endpoint label — and an atom type's name, the label every other
        // type is defined on — is joined into type names and pair keys by
        // `TypeName::join`, so it may not contain what `join` refuses (`@`).
        // Checked here, where the label enters, not at the kernel that reads
        // it back: a force field that accepts a label it cannot compile is a
        // trap sprung later.
        let labels: &[&str] = match self.defs {
            StyleDefs::Atom(_) => &[name],
            _ => endpoints,
        };
        molrs::store::type_labels::TypeName::join(labels).map_err(DefError::Name)?;
        self.insert_type(name.to_owned(), endpoints, params)?;
        Ok(self)
    }

    fn accepts_endpoint_count(&self, n: usize) -> bool {
        match self.defs {
            StyleDefs::Atom(_) => n == 0,
            StyleDefs::Bond(_) => n == 2,
            StyleDefs::Angle(_) => n == 3,
            StyleDefs::Dihedral(_) | StyleDefs::Improper(_) => n == 4,
            StyleDefs::Pair(_) => n == 1 || n == 2,
        }
    }

    /// The `def_type` endpoint count this category expects.
    fn endpoint_arity(&self) -> &'static str {
        match self.defs {
            StyleDefs::Atom(_) => "no endpoints",
            StyleDefs::Bond(_) => "2 endpoints",
            StyleDefs::Angle(_) => "3 endpoints",
            StyleDefs::Dihedral(_) | StyleDefs::Improper(_) => "4 endpoints",
            StyleDefs::Pair(_) => "1 or 2 endpoints",
        }
    }

    /// The conflict rule, and the one place it lives: whether defining `name`
    /// with `endpoints` and `params` here would be accepted, without defining
    /// it.
    ///
    /// `Ok(false)` when `name` is not defined here (the definition would be
    /// appended); `Ok(true)` when it is defined with the same endpoints and
    /// equal params (the re-definition is a no-op);
    /// `Err(DefError::TypeConflict)` when it is defined with anything else.
    /// `endpoints` are the stored form: the endpoint count has been checked
    /// against the category, and a one-endpoint pair is a self-pair.
    pub(crate) fn check_type(
        &self,
        name: &str,
        endpoints: &[&str],
        params: &Params,
    ) -> Result<bool, DefError> {
        let Some((stored, stored_params)) = self
            .type_rows()
            .into_iter()
            .find(|(n, _, _)| *n == name)
            .map(|(_, e, p)| (e, p))
        else {
            return Ok(false);
        };
        let given: Vec<&str> = match (&self.defs, endpoints) {
            (StyleDefs::Pair(_), [only]) => vec![*only, *only],
            _ => endpoints.to_vec(),
        };
        if stored == given && stored_params == params {
            Ok(true)
        } else {
            Err(DefError::TypeConflict {
                category: self.category(),
                style: self.name.clone(),
                name: name.to_owned(),
            })
        }
    }

    /// Every type this style holds as `(name, endpoints, params)`, in
    /// definition order, with endpoints in the form [`Style::def_type`]
    /// accepts (a pair row always has two). Replaying the rows through
    /// `def_type` reproduces the definitions. The one-pass read of a whole
    /// style; [`type_params`](Self::type_params) and
    /// [`type_endpoints`](Self::type_endpoints) answer for one type.
    pub fn type_rows(&self) -> Vec<(&str, Vec<&str>, &Params)> {
        match &self.defs {
            StyleDefs::Atom(v) => v
                .iter()
                .map(|t| (t.name.as_str(), Vec::new(), &t.params))
                .collect(),
            StyleDefs::Bond(v) => v
                .iter()
                .map(|t| (t.name.as_str(), vec![&*t.itom, &*t.jtom], &t.params))
                .collect(),
            StyleDefs::Angle(v) => v
                .iter()
                .map(|t| {
                    (
                        t.name.as_str(),
                        vec![&*t.itom, &*t.jtom, &*t.ktom],
                        &t.params,
                    )
                })
                .collect(),
            StyleDefs::Dihedral(v) => v
                .iter()
                .map(|t| {
                    let e = vec![&*t.itom, &*t.jtom, &*t.ktom, &*t.ltom];
                    (t.name.as_str(), e, &t.params)
                })
                .collect(),
            StyleDefs::Improper(v) => v
                .iter()
                .map(|t| {
                    let e = vec![&*t.itom, &*t.jtom, &*t.ktom, &*t.ltom];
                    (t.name.as_str(), e, &t.params)
                })
                .collect(),
            StyleDefs::Pair(v) => v
                .iter()
                .map(|t| (t.name.as_str(), vec![&*t.itom, &*t.jtom], &t.params))
                .collect(),
        }
    }

    /// The one insert path: [`Style::check_type`] decides, then a name not
    /// yet defined is appended.
    ///
    /// The endpoint count has been checked against the category by the
    /// caller; a one-endpoint pair is a self-pair. On
    /// `Err(DefError::TypeConflict)` the stored definition is unchanged.
    fn insert_type(&mut self, name: String, e: &[&str], params: Params) -> Result<(), DefError> {
        if self.check_type(&name, e, &params)? {
            return Ok(());
        }
        let own = |i: usize| e[i].to_owned();
        match &mut self.defs {
            StyleDefs::Atom(types) => types.push(AtomType { name, params }),
            StyleDefs::Bond(types) => types.push(BondType {
                name,
                itom: own(0),
                jtom: own(1),
                params,
            }),
            StyleDefs::Angle(types) => types.push(AngleType {
                name,
                itom: own(0),
                jtom: own(1),
                ktom: own(2),
                params,
            }),
            StyleDefs::Dihedral(types) => types.push(DihedralType {
                name,
                itom: own(0),
                jtom: own(1),
                ktom: own(2),
                ltom: own(3),
                params,
            }),
            StyleDefs::Improper(types) => types.push(ImproperType {
                name,
                itom: own(0),
                jtom: own(1),
                ktom: own(2),
                ltom: own(3),
                params,
            }),
            StyleDefs::Pair(types) => types.push(PairType {
                name,
                itom: own(0),
                jtom: own(e.len() - 1),
                params,
            }),
        }
        Ok(())
    }
}

/// Error from defining a style or a type: [`ForceField::def_style`] and
/// [`Style::def_type`] (and [`Style::rename_type`],
/// the one edit that can land on an existing name), and from
/// [`ForceField::merge`].
#[derive(Debug, Clone, PartialEq)]
pub enum DefError {
    /// The type `name` was given the wrong number of endpoints for
    /// `category`.
    Arity {
        category: &'static str,
        expected: &'static str,
        name: String,
        got: usize,
    },
    /// The category accepts no per-type definitions.
    Unsupported(&'static str),
    /// Unknown style category string.
    UnknownCategory(String),
    /// No style `name` is defined under `category`.
    UnknownStyle { category: String, name: String },
    /// The `category` style `style` already defines a type `name` with other
    /// endpoints or other params. The first definition is kept.
    TypeConflict {
        category: &'static str,
        style: String,
        name: String,
    },
    /// The `category` style `name` is already defined with other style params.
    /// The first definition is kept.
    StyleConflict { category: String, name: String },
    /// [`ForceField::merge`]: both force fields declare units, and they differ.
    UnitsConflict { ours: String, theirs: String },
    /// [`ForceField::merge`]: both force fields declare [`SpecialBonds`], and
    /// they differ.
    SpecialBondsConflict {
        ours: SpecialBonds,
        theirs: SpecialBonds,
    },
    /// A type name could not be built from its endpoint labels (a label
    /// containing `@`, or a malformed qualifier; see
    /// [`TypeName`](molrs::store::type_labels::TypeName)).
    Name(String),
}

impl std::fmt::Display for DefError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DefError::Arity {
                category,
                expected,
                name,
                got,
            } => write!(
                f,
                "{category} type \"{name}\": expected {expected}, got {got} endpoint(s)"
            ),
            DefError::Unsupported(category) => {
                write!(f, "{category} styles do not support per-type definitions")
            }
            DefError::UnknownCategory(category) => {
                write!(f, "unknown style category '{category}'")
            }
            DefError::UnknownStyle { category, name } => {
                write!(f, "no {category} style named '{name}'")
            }
            DefError::TypeConflict {
                category,
                style,
                name,
            } => write!(
                f,
                "{category} style '{style}' already defines type \"{name}\" with \
                 different endpoints or params"
            ),
            DefError::StyleConflict { category, name } => write!(
                f,
                "{category} style '{name}' is already defined with different style params"
            ),
            DefError::UnitsConflict { ours, theirs } => write!(
                f,
                "force field declares units '{ours}', the merged one declares '{theirs}'"
            ),
            DefError::SpecialBondsConflict { ours, theirs } => write!(
                f,
                "force field declares special_bonds lj {:?} coul {:?}, the merged one \
                 declares lj {:?} coul {:?}",
                ours.lj, ours.coul, theirs.lj, theirs.coul
            ),
            DefError::Name(reason) => write!(f, "invalid type name: {reason}"),
        }
    }
}

impl std::error::Error for DefError {}

/// In-place mutators backing the Python handle-view layer (Style/Type views read
/// through [`collect_type_params`](StyleDefs::collect_type_params) and write
/// through these). Each operates on the type identified by its dash-form name.
impl Style {
    /// The params of the type named `name`, or `None` if no such type.
    pub fn type_params(&self, name: &str) -> Option<&Params> {
        macro_rules! find_in {
            ($v:expr) => {
                $v.iter().find(|t| t.name == name).map(|t| &t.params)
            };
        }
        match &self.defs {
            StyleDefs::Atom(v) => find_in!(v),
            StyleDefs::Bond(v) => find_in!(v),
            StyleDefs::Angle(v) => find_in!(v),
            StyleDefs::Dihedral(v) => find_in!(v),
            StyleDefs::Improper(v) => find_in!(v),
            StyleDefs::Pair(v) => find_in!(v),
        }
    }

    /// Endpoint atom-type names of the type named `name` (e.g. `["CT","CT"]`),
    /// or `None` if no such type. Atom styles return an empty vec.
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
        }
    }

    /// Set (or add) a single param on the type named `name`. Returns `false` if
    /// no such type exists.
    ///
    /// An edit of an existing definition, not a definition: it changes the
    /// stored value in place and is outside the conflict rule (see [`Style`]).
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
        }
        false
    }

    /// Set (or add) one numeric style-level param, e.g. a pair style's
    /// `cutoff`.
    ///
    /// An edit of an existing definition, not a definition (see [`Style`]):
    /// the way a caller declares a run setting — the cutoff no reader or
    /// typifier invents — on a style it already has.
    pub fn set_param(&mut self, key: &str, value: f64) {
        self.params.set(key, value);
    }

    /// Set (or add) one string style-level param, e.g. a pair style's
    /// `mixing`: the string counterpart of [`set_param`](Self::set_param).
    pub fn set_str_param(&mut self, key: &str, value: &str) {
        self.params.set_str(key, value);
    }

    /// Set (or add) a single string param on the type named `name`. Returns
    /// `false` if no such type exists.
    ///
    /// An edit of an existing definition, not a definition: it changes the
    /// stored value in place and is outside the conflict rule (see [`Style`]).
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
        }
        false
    }

    /// Rename the type named `old` to `new`. Returns `Ok(false)` if no type
    /// is named `old`.
    ///
    /// The one edit that can land on an existing name, so it carries the
    /// conflict rule (see [`Style`]): if `new` is already defined with the same
    /// endpoints and equal params, the two are one definition and `old` is
    /// dropped (`Ok(true)`); if it is defined with anything else, the rename is
    /// `Err(DefError::TypeConflict)` and nothing changes.
    pub fn rename_type(&mut self, old: &str, new: &str) -> Result<bool, DefError> {
        let category = self.defs.category();
        let style = &self.name;
        macro_rules! rename_in {
            ($v:expr) => {{
                let Some(from) = $v.iter().position(|t| t.name == old) else {
                    return Ok(false);
                };
                if old == new {
                    return Ok(true);
                }
                let mut renamed = $v[from].clone();
                renamed.name = new.to_owned();
                match $v.iter().find(|t| t.name == new) {
                    Some(existing) if *existing == renamed => {
                        $v.remove(from);
                    }
                    Some(_) => {
                        return Err(DefError::TypeConflict {
                            category,
                            style: style.clone(),
                            name: new.to_owned(),
                        });
                    }
                    None => $v[from] = renamed,
                }
                Ok(true)
            }};
        }
        match &mut self.defs {
            StyleDefs::Atom(v) => rename_in!(v),
            StyleDefs::Bond(v) => rename_in!(v),
            StyleDefs::Angle(v) => rename_in!(v),
            StyleDefs::Dihedral(v) => rename_in!(v),
            StyleDefs::Improper(v) => rename_in!(v),
            StyleDefs::Pair(v) => rename_in!(v),
        }
    }

    /// Remove the type named `name`. Returns the count removed (`0` or `1`:
    /// a name is defined at most once per style).
    ///
    /// An edit of an existing definition, not a definition: it is outside the
    /// conflict rule (see [`Style`]).
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
        }
    }
}

// ---------------------------------------------------------------------------
// ForceField
// ---------------------------------------------------------------------------

/// Per-nonbonded-kind 1-2 / 1-3 / 1-4 interaction scale weights — LAMMPS
/// `special_bonds` semantics, owned by the [`ForceField`].
///
/// The always-on geometric table is [`crate::BondDistanceWeights`]: one
/// arbitrary-length vector with an explicit 1-N tail. A LAMMPS triple is not
/// a transcription (`charmm 0 0 0` is `[0, 0, 0, 1]` there). There is no
/// `From` / `Into` between the two types.
///
/// A weight of `0.0` fully excludes that neighbour class; `1.0` leaves it at
/// full strength.
///
/// # Two doors, two expressive powers
///
/// A **compiled** pair list (`intramolecular_pairs` → `PotentialCompiler::compile`) carries
/// the 1-2 / 1-3 weights by *presence*: the row is there or it is not. That is
/// one bit, so it expresses `0.0` and `1.0` and nothing between, and it
/// expresses only weights the van-der-Waals and Coulomb kernels **share** —
/// one list feeds both. [`compiled_inclusion`](Self::compiled_inclusion) is
/// that judgement, and both doors on that path call it rather than assume.
///
/// A **neighbour-driven** evaluation (`PotentialCompiler::compile_typed`) carries them as a
/// per-pair factor ([`lj_weights`](Self::lj_weights) /
/// [`coul_weights`](Self::coul_weights)), so it expresses every weight, and
/// the two kernels independently.
///
/// The 1-4 weight `[2]` is not part of this: both doors scale it inside the
/// kernel, so a fraction is fine there.
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
        DEFAULT_SPECIAL_BONDS
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

    /// The LJ weights as a bond-distance table, full strength past 1-4.
    ///
    /// What a neighbour-driven evaluation needs. A compiled intramolecular
    /// list carried these by *omitting* the excluded rows and baking the 1-4
    /// factor into the parameters, so only `[2]` was ever read; a neighbour
    /// table finds every pair inside the cutoff and needs all three.
    pub fn lj_weights(&self) -> BondDistanceWeights {
        Self::table(self.lj)
    }

    /// The Coulomb weights as a bond-distance table, full strength past 1-4.
    ///
    /// Separate from [`lj_weights`](Self::lj_weights) because a force field may
    /// scale the two differently — Amber uses `1/2` for van der Waals and
    /// `1/1.2` for electrostatics — and in molrs they are separate kernels.
    pub fn coul_weights(&self) -> BondDistanceWeights {
        Self::table(self.coul)
    }

    /// Whether a compiled `pairs` list can carry the 1-2 and 1-3 weights, and
    /// if so whether each class belongs *in* the list.
    ///
    /// `Ok([keep_12, keep_13])` — `false` means omit those rows (the class is
    /// excluded), `true` means emit them unflagged (full strength). `Err` means
    /// the weights are outside what a presence/absence list can say, and the
    /// caller must use the neighbour-driven door instead of quietly rounding.
    ///
    /// Two ways to fall outside:
    ///
    /// * a **fraction** — `lj[1] == 0.5` scales 1-3 pairs to half strength, and
    ///   a row that is merely present cannot say "half";
    /// * a **split** — `lj[1] == 1.0` with `coul[1] == 0.0` wants the row for
    ///   one kernel and not for the other, and there is one list for both.
    ///
    /// LAMMPS's own presets exercise both the accepted values: `amber`,
    /// `charmm` and `dreiding` exclude 1-3 (`false`), `fene` keeps it
    /// (`[0, 1, 1]` → `true`).
    pub fn compiled_inclusion(&self) -> Result<[bool; 2], String> {
        let mut keep = [false; 2];
        for (k, slot) in keep.iter_mut().enumerate() {
            let class = if k == 0 { "1-2" } else { "1-3" };
            let (lj, coul) = (self.lj[k], self.coul[k]);
            if lj != coul {
                return Err(format!(
                    "special_bonds {class}: lj {lj} and coul {coul} differ, and a \
                     compiled pairs list is shared by both kernels — it can include \
                     the row or omit it, not do one for van der Waals and the other \
                     for Coulomb. Use PotentialCompiler::compile_typed, which carries \
                     a per-pair weight per kernel."
                ));
            }
            *slot = if lj == 0.0 {
                false
            } else if lj == 1.0 {
                true
            } else {
                return Err(format!(
                    "special_bonds {class} weight {lj}: a compiled pairs list carries \
                     this class by whether the row is present, so it expresses 0 or 1 \
                     and nothing between. Use PotentialCompiler::compile_typed, which \
                     carries a per-pair weight."
                ));
            };
        }
        Ok(keep)
    }

    fn table(w: [f64; 3]) -> BondDistanceWeights {
        BondDistanceWeights::new(vec![w[0], w[1], w[2], 1.0])
            .expect("a four-entry weight table is always well formed")
    }
}

/// Top-level forcefield container holding styles and their type definitions.
///
/// A force field is built through exactly two fallible primitives:
/// [`ForceField::def_style`] and [`Style::def_type`] (a type's name and its
/// endpoints are both given).
///
/// # Example
///
/// ```
/// use molrs::ff::forcefield::{ForceField, Params};
///
/// let mut ff = ForceField::new("example");
/// ff.def_style("bond", "harmonic", Params::new())
///     .unwrap()
///     .def_type("A-B", &["A", "B"], Params::from_pairs(&[("k", 300.0), ("r0", 1.5)]))
///     .unwrap();
/// ff.def_style("pair", "lj/cut", Params::from_pairs(&[("cutoff", 10.0)]))
///     .unwrap()
///     .def_type("A", &["A"], Params::from_pairs(&[("epsilon", 0.5), ("sigma", 1.0)]))
///     .unwrap();
///
/// // Compile into Potentials against a typed topology
/// // let potentials = PotentialCompiler::new(&ff).compile(&frame).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct ForceField {
    pub name: String,
    styles: Vec<Style>,
    /// `None` until declared; [`Self::units`] reads the default then.
    units: Option<String>,
    /// `None` until declared; [`Self::special_bonds`] reads the default then.
    special_bonds: Option<SpecialBonds>,
}

/// The unit system of a force field that declares none (LAMMPS `real`).
const DEFAULT_UNITS: &str = "real";

/// The weights of a force field that declares none ([`SpecialBonds::default`]).
const DEFAULT_SPECIAL_BONDS: SpecialBonds = SpecialBonds {
    lj: [0.0, 0.0, 1.0],
    coul: [0.0, 0.0, 1.0],
};

impl ForceField {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_owned(),
            styles: Vec::new(),
            units: None,
            special_bonds: None,
        }
    }

    /// The unit system the parameters are expressed in (a LAMMPS `units`
    /// name: `real`, `metal`, `lj`, …). `"real"` when none is declared.
    pub fn units(&self) -> &str {
        self.units.as_deref().unwrap_or(DEFAULT_UNITS)
    }

    /// The declared unit system, or `None` when the force field declares none.
    pub fn declared_units(&self) -> Option<&str> {
        self.units.as_deref()
    }

    /// Declare the unit system the parameters are expressed in.
    pub fn set_units(&mut self, units: &str) {
        self.units = Some(units.to_owned());
    }

    /// The force field's [`SpecialBonds`] 1-2 / 1-3 / 1-4 nonbonded scale
    /// weights. Pair kernels apply the 1-4 weight to flagged (`is_14`) pairs.
    /// [`SpecialBonds::default`] when none are declared.
    pub fn special_bonds(&self) -> &SpecialBonds {
        self.special_bonds
            .as_ref()
            .unwrap_or(&DEFAULT_SPECIAL_BONDS)
    }

    /// The declared [`SpecialBonds`], or `None` when the force field declares
    /// none.
    pub fn declared_special_bonds(&self) -> Option<&SpecialBonds> {
        self.special_bonds.as_ref()
    }

    /// Declare the [`SpecialBonds`] weights. Force-field readers set these per
    /// force field (Amber/GAFF, OPLS, …). Declaring the default value is still
    /// a declaration.
    pub fn set_special_bonds(&mut self, special_bonds: SpecialBonds) {
        self.special_bonds = Some(special_bonds);
    }

    /// Merge `other` into `self`: the union of both definitions.
    ///
    /// `other` is replayed through the definition primitives
    /// ([`Self::def_style`], [`Style::def_type`]), so the conflict rule
    /// holds unchanged: an identical re-definition is a no-op, a different one
    /// is an error. Style params (pair `cutoff`, `coulomb`, `mixing`, …) take
    /// part in the style identity check. `self`'s styles come first, then
    /// `other`'s new styles in `other`'s order; types keep their order.
    ///
    /// Declared `units` and `special_bonds` are adopted when `self` declares
    /// none; two declared values that differ are
    /// `Err(DefError::UnitsConflict)` / `Err(DefError::SpecialBondsConflict)`.
    ///
    /// All-or-nothing: on `Err`, `self` is unchanged. `self.name` is kept.
    pub fn merge(&mut self, other: &ForceField) -> Result<(), DefError> {
        let mut out = self.clone();
        match (&out.units, &other.units) {
            (Some(ours), Some(theirs)) if ours != theirs => {
                return Err(DefError::UnitsConflict {
                    ours: ours.clone(),
                    theirs: theirs.clone(),
                });
            }
            (None, Some(theirs)) => out.units = Some(theirs.clone()),
            _ => {}
        }
        match (out.special_bonds, other.special_bonds) {
            (Some(ours), Some(theirs)) if ours != theirs => {
                return Err(DefError::SpecialBondsConflict { ours, theirs });
            }
            (None, Some(theirs)) => out.special_bonds = Some(theirs),
            _ => {}
        }
        for style in &other.styles {
            let target = out.def_style(style.category(), &style.name, style.params.clone())?;
            for (name, endpoints, params) in style.type_rows() {
                target.def_type(name, &endpoints, params.clone())?;
            }
        }
        *self = out;
        Ok(())
    }

    /// Define the `category` style named `name`, or return the existing one.
    ///
    /// `category` is one of `atom`/`bond`/`angle`/`dihedral`/`improper`/`pair`;
    /// anything else is `Err(DefError::UnknownCategory)`. A style is identified
    /// by `(category, name)`: a repeated definition with exactly equal `params`
    /// returns the style already defined; with different `params` it is
    /// `Err(DefError::StyleConflict)` and the first definition is kept.
    pub fn def_style(
        &mut self,
        category: &str,
        name: &str,
        params: Params,
    ) -> Result<&mut Style, DefError> {
        self.check_style(category, name, &params)?;
        if let Some(idx) = self
            .styles
            .iter()
            .position(|s| s.category() == category && s.name == name)
        {
            return Ok(&mut self.styles[idx]);
        }
        self.styles
            .push(Style::new(StyleDefs::empty(category)?, name, params));
        Ok(self.styles.last_mut().expect("a style was just pushed"))
    }

    /// The rule [`Self::def_style`] applies, checked without defining: an
    /// unknown `category` is `Err(DefError::UnknownCategory)`, and a style
    /// `(category, name)` already defined with other `params` is
    /// `Err(DefError::StyleConflict)`. `Ok(())` means `def_style` would
    /// either define the style or return the identical one.
    pub(crate) fn check_style(
        &self,
        category: &str,
        name: &str,
        params: &Params,
    ) -> Result<(), DefError> {
        StyleDefs::empty(category)?;
        match self.get_style(category, name) {
            Some(existing) if existing.params != *params => Err(DefError::StyleConflict {
                category: category.to_owned(),
                name: name.to_owned(),
            }),
            _ => Ok(()),
        }
    }

    /// An empty force field with this one's name and its **declared** units
    /// and special_bonds, and no styles or types.
    ///
    /// The seed of a typing output: a typifier's output starts as
    /// `library.empty_like()` and gains only the definitions typing assigns.
    /// Undeclared state stays undeclared — the defaults are not declared.
    pub fn empty_like(&self) -> ForceField {
        ForceField {
            name: self.name.clone(),
            styles: Vec::new(),
            units: self.units.clone(),
            special_bonds: self.special_bonds,
        }
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

    /// Mutable style lookup, for the explicit edits (`set_type_param`,
    /// `set_type_str_param`, `rename_type`, `remove_type`) of an existing
    /// definition.
    pub fn get_style_mut(&mut self, category: &str, name: &str) -> Option<&mut Style> {
        self.styles
            .iter_mut()
            .find(|s| s.category() == category && s.name == name)
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
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn test_params() {
        let p = Params::from_pairs(&[("k", 300.0), ("r0", 1.5)]);
        assert_eq!(p.get("k"), Some(300.0));
        assert_eq!(p.get("r0"), Some(1.5));
        assert_eq!(p.get("missing"), None);
    }

    // -- ForceField::def_style ---------------------------------------------

    #[test]
    fn def_style_returns_the_existing_style_for_a_repeated_category_and_name() {
        let mut ff = ForceField::new("test");
        ff.def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "A-B",
                &["A", "B"],
                Params::from_pairs(&[("k", 1.0), ("r0", 1.0)]),
            )
            .unwrap();
        ff.def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "C-D",
                &["C", "D"],
                Params::from_pairs(&[("k", 2.0), ("r0", 2.0)]),
            )
            .unwrap();

        let styles = ff.get_styles("bond");
        assert_eq!(styles.len(), 1);
        let StyleDefs::Bond(types) = styles[0].defs() else {
            panic!("expected Bond defs");
        };
        assert_eq!(types.len(), 2);
    }

    #[test]
    fn def_style_keys_a_style_by_category_and_name() {
        let mut ff = ForceField::new("test");
        ff.def_style("bond", "harmonic", Params::new()).unwrap();
        ff.def_style("angle", "harmonic", Params::new()).unwrap();

        assert_eq!(ff.styles().len(), 2);
        assert!(ff.get_style("bond", "harmonic").is_some());
        assert!(ff.get_style("angle", "harmonic").is_some());
    }

    #[test]
    fn def_style_rejects_an_unknown_category() {
        let mut ff = ForceField::new("test");
        assert!(matches!(
            ff.def_style("bogus", "x", Params::new()),
            Err(DefError::UnknownCategory(_))
        ));
        assert!(ff.styles().is_empty());
    }

    /// `kspace` is not a category a force field can declare a style under.
    ///
    /// Where PME actually *is* registered — `pair/coul/long/pme`, and nothing
    /// under `kspace` — is a registry claim, and `registry.rs` asserts it.
    #[test]
    fn kspace_is_not_a_style_category() {
        let mut ff = ForceField::new("test");
        assert!(matches!(
            ff.def_style("kspace", "pme", Params::new()),
            Err(DefError::UnknownCategory(_))
        ));
    }

    // -- Style accessors ---------------------------------------------------

    #[test]
    fn style_name_is_the_defined_name() {
        let mut ff = ForceField::new("test");
        let style = ff.def_style("pair", "lj/cut", Params::new()).unwrap();
        assert_eq!(style.name(), "lj/cut");
    }

    #[test]
    fn style_params_carry_numeric_and_string_values_of_the_definition() {
        let mut params = Params::from_pairs(&[("cutoff", 10.0)]);
        params.set_str("mixing", "geometric");
        let mut ff = ForceField::new("test");
        ff.def_style("pair", "lj/cut", params).unwrap();

        let style = ff.get_style("pair", "lj/cut").unwrap();
        assert_eq!(style.params().get("cutoff"), Some(10.0));
        assert_eq!(style.params().get_str("mixing"), Some("geometric"));
    }

    #[test]
    fn style_defs_hold_the_defined_types() {
        let mut ff = ForceField::new("test");
        ff.def_style("atom", "full", Params::new())
            .unwrap()
            .def_type("CT", &[], Params::from_pairs(&[("mass", 12.011)]))
            .unwrap();

        let style = ff.get_style("atom", "full").unwrap();
        let StyleDefs::Atom(types) = style.defs() else {
            panic!("expected Atom defs");
        };
        assert_eq!(types.len(), 1);
        assert_eq!(types[0].name, "CT");
        assert_eq!(types[0].params.get("mass"), Some(12.011));
    }

    // -- Style::def_type -----------------------------------------------------

    #[test]
    fn set_param_declares_a_cutoff_on_an_existing_style() {
        let mut ff = ForceField::new("t");
        ff.def_style("pair", "lj/cut", Params::new()).unwrap();
        ff.get_style_mut("pair", "lj/cut")
            .unwrap()
            .set_param("cutoff", 12.0);
        assert_eq!(
            ff.get_style("pair", "lj/cut")
                .unwrap()
                .params()
                .get("cutoff"),
            Some(12.0)
        );
    }

    #[test]
    fn def_type_atom() {
        let mut ff = ForceField::new("test");
        ff.def_style("atom", "full", Params::new())
            .unwrap()
            .def_type(
                "CT",
                &[],
                Params::from_pairs(&[("mass", 12.011), ("charge", -0.12)]),
            )
            .unwrap()
            .def_type(
                "HC",
                &[],
                Params::from_pairs(&[("mass", 1.008), ("charge", 0.06)]),
            )
            .unwrap();

        let style = ff.get_style("atom", "full").unwrap();
        let StyleDefs::Atom(types) = style.defs() else {
            panic!("expected Atom defs");
        };
        assert_eq!(types.len(), 2);

        let ct = style.get_atomtype("CT").unwrap();
        assert_eq!(ct.params.get("mass"), Some(12.011));
        assert_eq!(ct.params.get("charge"), Some(-0.12));
    }

    #[test]
    fn def_type_bond() {
        let mut ff = ForceField::new("test");
        let style = ff.def_style("bond", "harmonic", Params::new()).unwrap();
        style
            .def_type(
                "CT-OH",
                &["CT", "OH"],
                Params::from_pairs(&[("k", 300.0), ("r0", 1.4)]),
            )
            .unwrap();

        let bt = style.get_bondtype("CT", "OH").unwrap();
        assert_eq!(bt.itom, "CT");
        assert_eq!(bt.jtom, "OH");
        assert_eq!(bt.params.get("k"), Some(300.0));
        assert_eq!(bt.params.get("r0"), Some(1.4));
    }

    #[test]
    fn get_bondtype_is_order_independent() {
        let mut ff = ForceField::new("test");
        let style = ff.def_style("bond", "harmonic", Params::new()).unwrap();
        style
            .def_type(
                "CT-HC",
                &["CT", "HC"],
                Params::from_pairs(&[("k", 340.0), ("r0", 1.09)]),
            )
            .unwrap();

        let bt = style.get_bondtype("HC", "CT").unwrap();
        assert_eq!(bt.params.get("r0"), Some(1.09));
    }

    #[test]
    fn def_type_angle() {
        let mut ff = ForceField::new("test");
        let style = ff.def_style("angle", "harmonic", Params::new()).unwrap();
        style
            .def_type(
                "HC-CT-HC",
                &["HC", "CT", "HC"],
                Params::from_pairs(&[("k", 33.0), ("theta0", 107.8)]),
            )
            .unwrap();

        let StyleDefs::Angle(types) = style.defs() else {
            panic!("expected Angle defs");
        };
        assert_eq!(types.len(), 1);
        assert_eq!(types[0].itom, "HC");
        assert_eq!(types[0].jtom, "CT");
        assert_eq!(types[0].ktom, "HC");
        assert_eq!(types[0].params.get("theta0"), Some(107.8));
    }

    #[test]
    fn def_type_dihedral() {
        let mut ff = ForceField::new("test");
        let style = ff.def_style("dihedral", "opls", Params::new()).unwrap();
        style
            .def_type(
                "HC-CT-CT-HC",
                &["HC", "CT", "CT", "HC"],
                Params::from_pairs(&[("k1", 0.0), ("k2", 0.0), ("k3", 0.3)]),
            )
            .unwrap();

        let StyleDefs::Dihedral(types) = style.defs() else {
            panic!("expected Dihedral defs");
        };
        assert_eq!(types.len(), 1);
        assert_eq!(types[0].itom, "HC");
        assert_eq!(types[0].ltom, "HC");
    }

    #[test]
    fn def_type_pair_self() {
        let mut ff = ForceField::new("test");
        let style = ff
            .def_style("pair", "lj/cut", Params::from_pairs(&[("cutoff", 10.0)]))
            .unwrap();
        style
            .def_type(
                "Ar",
                &["Ar"],
                Params::from_pairs(&[("epsilon", 1.0), ("sigma", 3.4)]),
            )
            .unwrap();

        let pt = style.get_pairtype("Ar", None).unwrap();
        assert_eq!(pt.itom, "Ar");
        assert_eq!(pt.jtom, "Ar");
        assert_eq!(pt.params.get("epsilon"), Some(1.0));
    }

    #[test]
    fn def_type_pair_cross() {
        let mut ff = ForceField::new("test");
        let style = ff
            .def_style("pair", "lj/cut", Params::from_pairs(&[("cutoff", 10.0)]))
            .unwrap();
        style
            .def_type(
                "CT-OH",
                &["CT", "OH"],
                Params::from_pairs(&[("epsilon", 0.1), ("sigma", 3.3)]),
            )
            .unwrap();

        let pt = style.get_pairtype("OH", Some("CT")).unwrap();
        assert_eq!(pt.itom, "CT");
        assert_eq!(pt.jtom, "OH");
        assert_eq!(pt.params.get("epsilon"), Some(0.1));
    }

    #[test]
    fn def_type_carries_string_params_with_the_definition() {
        let mut params = Params::from_pairs(&[("mass", 12.011)]);
        params.set_str("element", "C");
        let mut ff = ForceField::new("test");
        let style = ff.def_style("atom", "full", Params::new()).unwrap();
        style.def_type("CT", &[], params).unwrap();

        let ct = style.get_atomtype("CT").unwrap();
        assert_eq!(ct.params.get_str("element"), Some("C"));
    }

    #[test]
    fn def_type_chains_on_the_returned_style() {
        let mut ff = ForceField::new("test");
        let style = ff.def_style("bond", "harmonic", Params::new()).unwrap();
        style
            .def_type(
                "A-B",
                &["A", "B"],
                Params::from_pairs(&[("k", 1.0), ("r0", 1.0)]),
            )
            .unwrap()
            .def_type(
                "C-D",
                &["C", "D"],
                Params::from_pairs(&[("k", 2.0), ("r0", 2.0)]),
            )
            .unwrap();

        let StyleDefs::Bond(types) = style.defs() else {
            panic!("expected Bond defs");
        };
        assert_eq!(types.len(), 2);
    }

    // -- Style::def_type: endpoints given, name stored verbatim ----------------

    /// The name is an opaque identifier: a bond named `anything` holds exactly
    /// the endpoints it was given, under exactly that name.
    #[test]
    fn def_type_stores_the_given_endpoints_under_the_verbatim_name() {
        let mut ff = ForceField::new("test");
        let style = ff.def_style("bond", "harmonic", Params::new()).unwrap();
        style
            .def_type(
                "anything",
                &["CT", "OH"],
                Params::from_pairs(&[("k", 300.0), ("r0", 1.4)]),
            )
            .unwrap();

        let StyleDefs::Bond(types) = style.defs() else {
            panic!("expected Bond defs");
        };
        assert_eq!(types.len(), 1);
        assert_eq!(types[0].name, "anything");
        assert_eq!(types[0].itom, "CT");
        assert_eq!(types[0].jtom, "OH");
        assert_eq!(
            style.type_endpoints("anything"),
            Some(vec!["CT".to_string(), "OH".to_string()])
        );
    }

    /// A `-` in a name is a character, never a separator: the bond named
    /// `CT-OH` defined on `HC`, `OS` holds `HC`, `OS`; the atom type `C-1`
    /// takes no endpoints; the self pair `tip3p-O` holds its one label twice.
    #[test]
    fn def_type_never_splits_a_dashed_name() {
        let mut ff = ForceField::new("test");
        ff.def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type("CT-OH", &["HC", "OS"], Params::new())
            .unwrap();
        ff.def_style("atom", "full", Params::new())
            .unwrap()
            .def_type("C-1", &[], Params::new())
            .unwrap();
        ff.def_style("pair", "lj/cut", Params::new())
            .unwrap()
            .def_type("tip3p-O", &["tip3p-O"], Params::new())
            .unwrap();

        assert_eq!(
            ff.get_style("bond", "harmonic")
                .unwrap()
                .type_endpoints("CT-OH"),
            Some(vec!["HC".to_string(), "OS".to_string()])
        );
        assert_eq!(
            ff.get_style("atom", "full").unwrap().type_endpoints("C-1"),
            Some(Vec::new())
        );
        assert_eq!(
            ff.get_style("pair", "lj/cut")
                .unwrap()
                .type_endpoints("tip3p-O"),
            Some(vec!["tip3p-O".to_string(), "tip3p-O".to_string()])
        );
    }

    /// A reversed name is its own name: `OH-CT` is stored as written, not
    /// reoriented.
    #[test]
    fn def_type_keeps_a_reversed_name_as_written() {
        let mut ff = ForceField::new("test");
        let style = ff.def_style("bond", "harmonic", Params::new()).unwrap();
        style
            .def_type("OH-CT", &["OH", "CT"], Params::new())
            .unwrap();

        let StyleDefs::Bond(types) = style.defs() else {
            panic!("expected Bond defs");
        };
        assert_eq!(types[0].name, "OH-CT");
        assert_eq!(
            (types[0].itom.as_str(), types[0].jtom.as_str()),
            ("OH", "CT")
        );
    }

    /// An endpoint count off the category's arity is `Err(Arity)` (never a
    /// panic), whatever the name looks like; nothing is stored.
    /// `@` starts a [`TypeName`](molrs::store::type_labels::TypeName)
    /// qualifier, so no endpoint label may carry it — nor an atom type's name,
    /// which is the endpoint every other type is defined on. A qualified
    /// *type name* (`C_3-C_R@1.5`) is fine: names are never joined.
    #[test]
    fn def_type_refuses_an_endpoint_label_containing_at() {
        let mut ff = ForceField::new("t");
        let err = ff
            .def_style("pair", "lj/cut", Params::new())
            .unwrap()
            .def_type("x", &["a@b"], Params::new())
            .unwrap_err();
        assert!(matches!(err, DefError::Name(_)), "{err}");
        let err = ff
            .def_style("atom", "full", Params::new())
            .unwrap()
            .def_type("a@b", &[], Params::new())
            .unwrap_err();
        assert!(matches!(err, DefError::Name(_)), "{err}");
        ff.def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type("C_3-C_R@1.5", &["C_3", "C_R"], Params::new())
            .unwrap();
    }

    #[test]
    fn type_params_finds_one_type_by_name() {
        let mut ff = ForceField::new("t");
        let style = ff.def_style("bond", "harmonic", Params::new()).unwrap();
        style
            .def_type("A-B", &["A", "B"], Params::from_pairs(&[("k", 1.0)]))
            .unwrap();
        assert_eq!(style.type_params("A-B").and_then(|p| p.get("k")), Some(1.0));
        assert!(style.type_params("B-A").is_none());
    }

    #[test]
    fn set_str_param_writes_a_string_style_param() {
        let mut ff = ForceField::new("t");
        let style = ff.def_style("pair", "lj/cut", Params::new()).unwrap();
        style.set_str_param("mixing", "geometric");
        assert_eq!(style.params().get_str("mixing"), Some("geometric"));
    }

    #[test]
    fn def_type_endpoint_count_off_the_category_arity_is_an_arity_error() {
        let cases: [(&str, &str, &[&str]); 8] = [
            ("atom", "full", &["CT"]),
            ("bond", "harmonic", &["CT"]),
            ("bond", "harmonic", &["CT", "CT", "CT"]),
            ("angle", "harmonic", &["CT", "CT"]),
            ("dihedral", "opls", &["CT", "CT", "CT"]),
            ("improper", "harmonic", &["CT", "CT", "CT", "CT", "CT"]),
            ("pair", "lj/cut", &[]),
            ("pair", "lj/cut", &["CT", "CT", "CT"]),
        ];
        for (category, style_name, endpoints) in cases {
            let mut ff = ForceField::new("test");
            let style = ff.def_style(category, style_name, Params::new()).unwrap();
            let got = style
                .def_type("CT-CT", endpoints, Params::new())
                .map(|_| ());
            assert!(
                matches!(
                    got,
                    Err(DefError::Arity { got: n, .. }) if n == endpoints.len()
                ),
                "{category} with {} endpoints: {got:?}",
                endpoints.len()
            );
            assert!(style.defs().collect_type_params().is_empty(), "{category}");
        }
    }

    #[test]
    fn def_type_atom_accepts_empty_endpoints() {
        let mut ff = ForceField::new("test");
        let style = ff.def_style("atom", "full", Params::new()).unwrap();
        style
            .def_type("CT", &[], Params::from_pairs(&[("mass", 12.011)]))
            .unwrap();

        assert_eq!(style.type_endpoints("CT"), Some(Vec::new()));
    }

    // -- Params equality -------------------------------------------------------

    /// Identity, not closeness: one ulp apart is a different parameter set, and
    /// the string side counts as much as the numeric side.
    #[test]
    fn params_equality_is_exact_on_both_sides() {
        let mut a = Params::from_pairs(&[("k", 300.0)]);
        a.set_str("element", "C");
        let mut same = Params::from_pairs(&[("k", 300.0)]);
        same.set_str("element", "C");
        let mut next_ulp = Params::from_pairs(&[("k", f64::from_bits(300.0_f64.to_bits() + 1))]);
        next_ulp.set_str("element", "C");
        let mut other_string = Params::from_pairs(&[("k", 300.0)]);
        other_string.set_str("element", "N");

        assert_eq!(a, same);
        assert_ne!(a, next_ulp);
        assert_ne!(a, other_string);
    }

    // -- the conflict rule: Style::def_type ---------------------------------------

    #[test]
    fn def_type_identical_redefinition_leaves_one_type() {
        let mut ff = ForceField::new("test");
        let style = ff.def_style("bond", "harmonic", Params::new()).unwrap();
        style
            .def_type(
                "CT-CT",
                &["CT", "CT"],
                Params::from_pairs(&[("k", 268.0), ("r0", 1.529)]),
            )
            .unwrap();
        style
            .def_type(
                "CT-CT",
                &["CT", "CT"],
                Params::from_pairs(&[("k", 268.0), ("r0", 1.529)]),
            )
            .unwrap();

        let StyleDefs::Bond(types) = style.defs() else {
            panic!("expected Bond defs");
        };
        assert_eq!(types.len(), 1);
    }

    /// A different numeric param under one name is rejected, and the first
    /// definition is the one left standing.
    #[test]
    fn def_type_with_a_different_numeric_param_is_a_type_conflict() {
        let mut ff = ForceField::new("test");
        let style = ff.def_style("bond", "harmonic", Params::new()).unwrap();
        style
            .def_type(
                "CT-CT",
                &["CT", "CT"],
                Params::from_pairs(&[("k", 268.0), ("r0", 1.529)]),
            )
            .unwrap();

        let second = style.def_type(
            "CT-CT",
            &["CT", "CT"],
            Params::from_pairs(&[("k", 310.0), ("r0", 1.529)]),
        );

        assert!(
            matches!(second, Err(DefError::TypeConflict { .. })),
            "{second:?}"
        );
        let StyleDefs::Bond(types) = style.defs() else {
            panic!("expected Bond defs");
        };
        assert_eq!(types.len(), 1);
        assert_eq!(types[0].params.get("k"), Some(268.0));
    }

    /// A different string param under one name is rejected, and the first
    /// definition is the one left standing.
    #[test]
    fn def_type_with_a_different_string_param_is_a_type_conflict() {
        let mut carbon = Params::from_pairs(&[("mass", 12.011)]);
        carbon.set_str("element", "C");
        let mut nitrogen = Params::from_pairs(&[("mass", 12.011)]);
        nitrogen.set_str("element", "N");
        let mut ff = ForceField::new("test");
        let style = ff.def_style("atom", "full", Params::new()).unwrap();
        style.def_type("CT", &[], carbon).unwrap();

        let second = style.def_type("CT", &[], nitrogen);

        assert!(
            matches!(second, Err(DefError::TypeConflict { .. })),
            "{second:?}"
        );
        let StyleDefs::Atom(types) = style.defs() else {
            panic!("expected Atom defs");
        };
        assert_eq!(types.len(), 1);
        assert_eq!(types[0].params.get_str("element"), Some("C"));
    }

    /// One name, equal params, different endpoints: still a different
    /// definition, and the first endpoints are the ones left standing.
    #[test]
    fn def_type_same_name_with_different_endpoints_is_a_type_conflict() {
        let mut ff = ForceField::new("test");
        let style = ff.def_style("bond", "mmff", Params::new()).unwrap();
        style
            .def_type("0_1_5", &["1", "5"], Params::from_pairs(&[("kb", 4.258)]))
            .unwrap();

        let second = style.def_type("0_1_5", &["1", "6"], Params::from_pairs(&[("kb", 4.258)]));

        assert!(
            matches!(second, Err(DefError::TypeConflict { .. })),
            "{second:?}"
        );
        assert_eq!(
            style.type_endpoints("0_1_5"),
            Some(vec!["1".to_string(), "5".to_string()])
        );
    }

    // -- the conflict rule: ForceField::def_style --------------------------------

    /// Different style params under one `(category, name)` are rejected, and the
    /// first style params are the ones left standing.
    #[test]
    fn def_style_with_different_params_is_a_style_conflict() {
        let mut ff = ForceField::new("test");
        ff.def_style("pair", "lj/cut", Params::from_pairs(&[("cutoff", 10.0)]))
            .unwrap();

        let second = ff
            .def_style("pair", "lj/cut", Params::from_pairs(&[("cutoff", 12.0)]))
            .map(|_| ());

        assert!(
            matches!(second, Err(DefError::StyleConflict { .. })),
            "{second:?}"
        );
        let style = ff.get_style("pair", "lj/cut").unwrap();
        assert_eq!(style.params().get("cutoff"), Some(10.0));
    }

    #[test]
    fn def_style_with_identical_params_returns_the_existing_style() {
        let mut ff = ForceField::new("test");
        ff.def_style("pair", "lj/cut", Params::from_pairs(&[("cutoff", 10.0)]))
            .unwrap()
            .def_type(
                "Ar",
                &["Ar"],
                Params::from_pairs(&[("epsilon", 0.238), ("sigma", 3.4)]),
            )
            .unwrap();

        let again = ff
            .def_style("pair", "lj/cut", Params::from_pairs(&[("cutoff", 10.0)]))
            .unwrap();

        assert!(again.get_pairtype("Ar", None).is_some());
        assert_eq!(ff.styles().len(), 1);
    }

    // -- edits: rename_type carries the collision rule, set_type_param does not --

    #[test]
    fn rename_type_to_an_unused_name_renames_it() {
        let mut ff = ForceField::new("test");
        ff.def_style("atom", "full", Params::new())
            .unwrap()
            .def_type("CX", &[], Params::from_pairs(&[("mass", 13.0)]))
            .unwrap();
        let style = ff.get_style_mut("atom", "full").unwrap();

        assert_eq!(style.rename_type("CX", "CY"), Ok(true));
        assert!(style.get_atomtype("CY").is_some());
        assert!(style.get_atomtype("CX").is_none());
    }

    /// Renaming onto a name already defined with different params would leave
    /// two definitions under one name: a conflict, and nothing is renamed.
    #[test]
    fn rename_type_onto_an_existing_name_with_different_params_is_an_error() {
        let mut ff = ForceField::new("test");
        ff.def_style("atom", "full", Params::new())
            .unwrap()
            .def_type("CT", &[], Params::from_pairs(&[("mass", 12.011)]))
            .unwrap()
            .def_type("CX", &[], Params::from_pairs(&[("mass", 13.0)]))
            .unwrap();
        let style = ff.get_style_mut("atom", "full").unwrap();

        let renamed = style.rename_type("CX", "CT");

        assert!(
            matches!(renamed, Err(DefError::TypeConflict { .. })),
            "{renamed:?}"
        );
        assert_eq!(
            style.get_atomtype("CT").unwrap().params.get("mass"),
            Some(12.011)
        );
        assert!(style.get_atomtype("CX").is_some());
    }

    /// An edit of an existing definition is not a definition: it changes the
    /// value in place and never meets the conflict rule.
    #[test]
    fn set_type_param_changes_an_existing_type_in_place() {
        let mut ff = ForceField::new("test");
        ff.def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "CT-CT",
                &["CT", "CT"],
                Params::from_pairs(&[("k", 268.0), ("r0", 1.529)]),
            )
            .unwrap();
        let style = ff.get_style_mut("bond", "harmonic").unwrap();

        let changed: bool = style.set_type_param("CT-CT", "k", 310.0);

        assert!(changed);
        let StyleDefs::Bond(types) = style.defs() else {
            panic!("expected Bond defs");
        };
        assert_eq!(types.len(), 1);
        assert_eq!(types[0].params.get("k"), Some(310.0));
    }

    // -- ForceField queries ----------------------------------------------------

    #[test]
    fn test_get_all_types() {
        let mut ff = ForceField::new("test");
        ff.def_style("atom", "full", Params::new())
            .unwrap()
            .def_type("CT", &[], Params::from_pairs(&[("mass", 12.0)]))
            .unwrap()
            .def_type("OH", &[], Params::from_pairs(&[("mass", 16.0)]))
            .unwrap();
        ff.def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "CT-OH",
                &["CT", "OH"],
                Params::from_pairs(&[("k", 300.0), ("r0", 1.4)]),
            )
            .unwrap();

        assert_eq!(ff.get_atomtypes().len(), 2);
        assert_eq!(ff.get_bondtypes().len(), 1);
        assert_eq!(ff.get_pairtypes().len(), 0);
    }

    #[test]
    fn get_angletypes_returns_the_defined_angle_params() {
        let mut ff = ForceField::new("test");
        ff.def_style("angle", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "HC-CT-HC",
                &["HC", "CT", "HC"],
                Params::from_pairs(&[("k", 33.0), ("theta0", 107.8)]),
            )
            .unwrap();

        let types = ff.get_angletypes();
        assert_eq!(types.len(), 1);
        assert_eq!(types[0].params.get("theta0"), Some(107.8));
    }

    /// The two values a presence/absence list can say, and the two ways to
    /// fall outside them.
    #[test]
    fn a_compiled_pair_list_says_only_in_or_out() {
        // The default and every Amber-family reader: both classes excluded.
        assert_eq!(
            SpecialBonds::default().compiled_inclusion(),
            Ok([false, false])
        );
        // LAMMPS `special_bonds fene`: 1-3 stays, at full strength.
        let fene = SpecialBonds {
            lj: [0.0, 1.0, 1.0],
            coul: [0.0, 1.0, 1.0],
        };
        assert_eq!(fene.compiled_inclusion(), Ok([false, true]));

        // A fraction is not expressible by a row that is merely there.
        let half = SpecialBonds {
            lj: [0.0, 0.5, 0.5],
            coul: [0.0, 0.5, 0.5],
        };
        let err = half.compiled_inclusion().unwrap_err();
        assert!(err.contains("1-3"), "{err}");
        assert!(err.contains("PotentialCompiler::compile_typed"), "{err}");

        // Neither is a class one kernel wants and the other does not.
        let split = SpecialBonds {
            lj: [0.0, 1.0, 1.0],
            coul: [0.0, 0.0, 1.0],
        };
        let err = split.compiled_inclusion().unwrap_err();
        assert!(err.contains("lj 1 and coul 0 differ"), "{err}");

        // The 1-4 weight is not part of this judgement: both doors scale it
        // inside the kernel, so a fraction there is ordinary.
        let amber = SpecialBonds {
            lj: [0.0, 0.0, 0.5],
            coul: [0.0, 0.0, 1.0 / 1.2],
        };
        assert_eq!(amber.compiled_inclusion(), Ok([false, false]));
    }

    // -- declared state: units / special_bonds ---------------------------------

    const AMBER_SB: SpecialBonds = SpecialBonds {
        lj: [0.0, 0.0, 0.5],
        coul: [0.0, 0.0, 0.8333],
    };

    const OPLS_SB: SpecialBonds = SpecialBonds {
        lj: [0.0, 0.0, 0.5],
        coul: [0.0, 0.0, 0.5],
    };

    #[test]
    fn declared_units_is_none_until_set_units() {
        let mut ff = ForceField::new("test");
        assert_eq!(ff.declared_units(), None);
        assert_eq!(ff.units(), "real");

        ff.set_units("lj");

        assert_eq!(ff.declared_units(), Some("lj"));
        assert_eq!(ff.units(), "lj");
    }

    #[test]
    fn declared_special_bonds_is_none_until_set_special_bonds() {
        let mut ff = ForceField::new("test");
        assert_eq!(ff.declared_special_bonds(), None);
        assert_eq!(*ff.special_bonds(), SpecialBonds::default());

        ff.set_special_bonds(AMBER_SB);

        assert_eq!(ff.declared_special_bonds(), Some(&AMBER_SB));
        assert_eq!(*ff.special_bonds(), AMBER_SB);
    }

    /// Declaring the default value is still a declaration.
    #[test]
    fn set_special_bonds_to_the_default_declares_it() {
        let mut ff = ForceField::new("test");
        ff.set_special_bonds(SpecialBonds::default());
        assert_eq!(ff.declared_special_bonds(), Some(&SpecialBonds::default()));
    }

    // -- ForceField::empty_like ------------------------------------------------

    /// The seed of a typing output: same name and declared state, no styles.
    #[test]
    fn empty_like_keeps_name_and_declarations_and_drops_every_style() {
        let mut library = ForceField::new("library");
        library.set_units("metal");
        library.set_special_bonds(AMBER_SB);
        library
            .def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "A-B",
                &["A", "B"],
                Params::from_pairs(&[("k", 1.0), ("r0", 1.0)]),
            )
            .unwrap();
        library
            .def_style("pair", "lj/cut", Params::from_pairs(&[("cutoff", 10.0)]))
            .unwrap();

        let empty = library.empty_like();

        assert_eq!(empty.name, "library");
        assert_eq!(empty.declared_units(), Some("metal"));
        assert_eq!(empty.declared_special_bonds(), Some(&AMBER_SB));
        assert!(empty.styles().is_empty(), "{:?}", empty.styles());
    }

    /// Undeclared stays undeclared: `empty_like` does not declare the defaults.
    #[test]
    fn empty_like_of_an_undeclared_force_field_declares_nothing() {
        let library = ForceField::new("bare");

        let empty = library.empty_like();

        assert_eq!(empty.name, "bare");
        assert_eq!(empty.declared_units(), None);
        assert_eq!(empty.declared_special_bonds(), None);
        assert!(empty.styles().is_empty());
    }

    /// `empty_like` reads `self`; the library keeps its definitions.
    #[test]
    fn empty_like_leaves_the_source_unchanged() {
        let mut library = ForceField::new("library");
        library.set_special_bonds(OPLS_SB);
        library
            .def_style("atom", "full", Params::new())
            .unwrap()
            .def_type("X", &[], Params::from_pairs(&[("mass", 1.0)]))
            .unwrap();
        let before = library.clone();

        let _ = library.empty_like();

        assert_same_definitions(&library, &before);
    }

    // -- ForceField::merge -----------------------------------------------------

    fn same_defs(a: &StyleDefs, b: &StyleDefs) -> bool {
        match (a, b) {
            (StyleDefs::Atom(x), StyleDefs::Atom(y)) => x == y,
            (StyleDefs::Bond(x), StyleDefs::Bond(y)) => x == y,
            (StyleDefs::Angle(x), StyleDefs::Angle(y)) => x == y,
            (StyleDefs::Dihedral(x), StyleDefs::Dihedral(y)) => x == y,
            (StyleDefs::Improper(x), StyleDefs::Improper(y)) => x == y,
            (StyleDefs::Pair(x), StyleDefs::Pair(y)) => x == y,
            _ => false,
        }
    }

    /// Every definition of `a` equals `b`'s, in order: name, declared state,
    /// and per style its category, name, params and types (endpoints and
    /// params included).
    pub(crate) fn assert_same_definitions(a: &ForceField, b: &ForceField) {
        assert_eq!(a.name, b.name);
        assert_eq!(a.declared_units(), b.declared_units());
        assert_eq!(a.declared_special_bonds(), b.declared_special_bonds());
        assert_eq!(a.styles().len(), b.styles().len(), "style count");
        for (x, y) in a.styles().iter().zip(b.styles()) {
            assert_eq!(
                (x.category(), x.name(), x.params()),
                (y.category(), y.name(), y.params())
            );
            assert!(
                same_defs(x.defs(), y.defs()),
                "{}:{} types differ: {:?} vs {:?}",
                x.category(),
                x.name(),
                x.defs(),
                y.defs()
            );
        }
    }

    fn style_keys(ff: &ForceField) -> Vec<(&'static str, &str)> {
        ff.styles()
            .iter()
            .map(|s| (s.category(), s.name()))
            .collect()
    }

    fn bond_names(style: &Style) -> Vec<&str> {
        let StyleDefs::Bond(types) = style.defs() else {
            panic!("expected Bond defs");
        };
        types.iter().map(|t| t.name.as_str()).collect()
    }

    /// `self`'s styles first, then `other`'s new styles in `other`'s order; a
    /// shared style gains `other`'s new types after its own.
    #[test]
    fn merge_is_the_union_with_self_styles_first_then_new_styles_in_order() {
        let mut ff = ForceField::new("target");
        ff.def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "A-B",
                &["A", "B"],
                Params::from_pairs(&[("k", 1.0), ("r0", 1.0)]),
            )
            .unwrap();
        ff.def_style("pair", "lj/cut", Params::new())
            .unwrap()
            .def_type(
                "A",
                &["A"],
                Params::from_pairs(&[("epsilon", 0.1), ("sigma", 3.0)]),
            )
            .unwrap();

        let mut other = ForceField::new("source");
        other
            .def_style("angle", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "A-B-C",
                &["A", "B", "C"],
                Params::from_pairs(&[("k", 50.0), ("theta0", 1.9)]),
            )
            .unwrap();
        other
            .def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "C-D",
                &["C", "D"],
                Params::from_pairs(&[("k", 2.0), ("r0", 2.0)]),
            )
            .unwrap()
            .def_type(
                "B-C",
                &["B", "C"],
                Params::from_pairs(&[("k", 3.0), ("r0", 3.0)]),
            )
            .unwrap();
        other
            .def_style("atom", "full", Params::new())
            .unwrap()
            .def_type("X", &[], Params::from_pairs(&[("mass", 1.0)]))
            .unwrap();

        ff.merge(&other).unwrap();

        assert_eq!(ff.name, "target");
        assert_eq!(
            style_keys(&ff),
            vec![
                ("bond", "harmonic"),
                ("pair", "lj/cut"),
                ("angle", "harmonic"),
                ("atom", "full"),
            ]
        );
        assert_eq!(
            bond_names(ff.get_style("bond", "harmonic").unwrap()),
            vec!["A-B", "C-D", "B-C"]
        );
        assert!(
            ff.get_style("angle", "harmonic")
                .unwrap()
                .type_endpoints("A-B-C")
                .is_some()
        );
        assert!(
            ff.get_style("atom", "full")
                .unwrap()
                .get_atomtype("X")
                .is_some()
        );
    }

    /// A type keeps its given endpoints through the merge.
    #[test]
    fn merge_keeps_the_given_endpoints() {
        let mut other = ForceField::new("mmff");
        other
            .def_style("bond", "mmff", Params::new())
            .unwrap()
            .def_type("0_1_5", &["1", "5"], Params::from_pairs(&[("kb", 4.258)]))
            .unwrap();
        let mut ff = ForceField::new("target");

        ff.merge(&other).unwrap();

        assert_eq!(
            ff.get_style("bond", "mmff")
                .unwrap()
                .type_endpoints("0_1_5"),
            Some(vec!["1".to_string(), "5".to_string()])
        );
    }

    #[test]
    fn merge_of_an_identical_overlap_is_a_no_op() {
        let mut ff = ForceField::new("same");
        ff.set_units("real");
        ff.set_special_bonds(AMBER_SB);
        ff.def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "CT-CT",
                &["CT", "CT"],
                Params::from_pairs(&[("k", 268.0), ("r0", 1.529)]),
            )
            .unwrap();
        let mut lj = Params::from_pairs(&[("cutoff", 10.0)]);
        lj.set_str("mixing", "geometric");
        ff.def_style("pair", "lj/cut", lj)
            .unwrap()
            .def_type(
                "CT",
                &["CT"],
                Params::from_pairs(&[("epsilon", 0.066), ("sigma", 3.5)]),
            )
            .unwrap();
        let before = ff.clone();

        ff.merge(&before).unwrap();

        assert_same_definitions(&ff, &before);
    }

    /// All-or-nothing: `other` defines a new style and a new type *before* the
    /// conflicting type and declares units and special_bonds; none of it may
    /// land.
    #[test]
    fn merge_type_conflict_leaves_self_equal_to_its_pre_merge_clone() {
        let mut ff = ForceField::new("target");
        ff.def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "CT-CT",
                &["CT", "CT"],
                Params::from_pairs(&[("k", 268.0), ("r0", 1.529)]),
            )
            .unwrap();
        let before = ff.clone();

        let mut other = ForceField::new("source");
        other.set_units("real");
        other.set_special_bonds(AMBER_SB);
        other
            .def_style("angle", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "CT-CT-CT",
                &["CT", "CT", "CT"],
                Params::from_pairs(&[("k", 58.35), ("theta0", 1.95)]),
            )
            .unwrap();
        other
            .def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "CT-HC",
                &["CT", "HC"],
                Params::from_pairs(&[("k", 340.0), ("r0", 1.09)]),
            )
            .unwrap()
            .def_type(
                "CT-CT",
                &["CT", "CT"],
                Params::from_pairs(&[("k", 310.0), ("r0", 1.529)]),
            )
            .unwrap();

        let merged = ff.merge(&other);

        assert!(
            matches!(merged, Err(DefError::TypeConflict { .. })),
            "{merged:?}"
        );
        assert_same_definitions(&ff, &before);
    }

    #[test]
    fn merge_keeps_lj_cut_cutoff_and_mixing() {
        let mut lj = Params::from_pairs(&[("cutoff", 12.0)]);
        lj.set_str("mixing", "geometric");
        let mut other = ForceField::new("source");
        other
            .def_style("pair", "lj/cut", lj)
            .unwrap()
            .def_type(
                "OW",
                &["OW"],
                Params::from_pairs(&[("epsilon", 0.1553), ("sigma", 3.166)]),
            )
            .unwrap();
        let mut ff = ForceField::new("target");

        ff.merge(&other).unwrap();

        let style = ff.get_style("pair", "lj/cut").unwrap();
        assert_eq!(style.params().get("cutoff"), Some(12.0));
        assert_eq!(style.params().get_str("mixing"), Some("geometric"));
    }

    #[test]
    fn merge_with_a_different_coul_cut_coulomb_is_a_style_conflict() {
        let mut ff = ForceField::new("target");
        ff.def_style(
            "pair",
            "coul/cut",
            Params::from_pairs(&[("coulomb", 332.06371), ("dielectric", 1.0)]),
        )
        .unwrap();
        let before = ff.clone();
        let mut other = ForceField::new("source");
        other
            .def_style(
                "pair",
                "coul/cut",
                Params::from_pairs(&[("coulomb", 332.0716), ("dielectric", 1.0)]),
            )
            .unwrap();

        let merged = ff.merge(&other);

        assert!(
            matches!(merged, Err(DefError::StyleConflict { .. })),
            "{merged:?}"
        );
        assert_same_definitions(&ff, &before);
    }

    #[test]
    fn merge_with_a_different_pair_cutoff_is_a_style_conflict() {
        let mut ff = ForceField::new("target");
        ff.def_style("pair", "lj/cut", Params::from_pairs(&[("cutoff", 10.0)]))
            .unwrap();
        let before = ff.clone();
        let mut other = ForceField::new("source");
        other
            .def_style("pair", "lj/cut", Params::from_pairs(&[("cutoff", 12.0)]))
            .unwrap();

        let merged = ff.merge(&other);

        assert!(
            matches!(merged, Err(DefError::StyleConflict { .. })),
            "{merged:?}"
        );
        assert_same_definitions(&ff, &before);
    }

    #[test]
    fn merge_into_undeclared_special_bonds_adopts_the_others() {
        let mut ff = ForceField::new("target");
        let mut other = ForceField::new("source");
        other.set_special_bonds(AMBER_SB);

        ff.merge(&other).unwrap();

        assert_eq!(ff.declared_special_bonds(), Some(&AMBER_SB));
    }

    #[test]
    fn merge_into_undeclared_units_adopts_the_others() {
        let mut ff = ForceField::new("target");
        let mut other = ForceField::new("source");
        other.set_units("lj");

        ff.merge(&other).unwrap();

        assert_eq!(ff.declared_units(), Some("lj"));
        assert_eq!(ff.units(), "lj");
    }

    /// An undeclared `other` leaves `self`'s declarations alone, and an
    /// undeclared pair stays undeclared.
    #[test]
    fn merge_of_undeclared_state_keeps_self_declarations() {
        let mut declared = ForceField::new("target");
        declared.set_units("metal");
        declared.set_special_bonds(OPLS_SB);
        declared.merge(&ForceField::new("source")).unwrap();
        assert_eq!(declared.declared_units(), Some("metal"));
        assert_eq!(declared.declared_special_bonds(), Some(&OPLS_SB));

        let mut undeclared = ForceField::new("target");
        undeclared.merge(&ForceField::new("source")).unwrap();
        assert_eq!(undeclared.declared_units(), None);
        assert_eq!(undeclared.declared_special_bonds(), None);
    }

    /// MMFF-style and OPLS-style 1-4 rules must refuse to combine.
    #[test]
    fn merge_with_a_different_declared_special_bonds_is_a_special_bonds_conflict() {
        let mut ff = ForceField::new("target");
        ff.set_special_bonds(OPLS_SB);
        let before = ff.clone();
        let mut other = ForceField::new("source");
        other.set_special_bonds(AMBER_SB);

        let merged = ff.merge(&other);

        assert!(
            matches!(merged, Err(DefError::SpecialBondsConflict { .. })),
            "{merged:?}"
        );
        assert_same_definitions(&ff, &before);
    }

    #[test]
    fn merge_with_a_different_declared_units_is_a_units_conflict() {
        let mut ff = ForceField::new("target");
        ff.set_units("real");
        let before = ff.clone();
        let mut other = ForceField::new("source");
        other.set_units("lj");

        let merged = ff.merge(&other);

        assert!(
            matches!(merged, Err(DefError::UnitsConflict { .. })),
            "{merged:?}"
        );
        assert_same_definitions(&ff, &before);
    }
}
