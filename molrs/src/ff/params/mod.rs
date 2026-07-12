//! Antechamber parameter tables as typed, compile-time Rust data.
//!
//! molrs parses **no** parameter text at runtime. The upstream `.DAT` / `.DEF`
//! tables are translated into the `const`s under [`generated`] by
//! `scripts/gen_param_tables.py`, which reads them from `$AMBERHOME`. The
//! committed `.rs` is the single in-repo source of truth; a malformed table is
//! therefore a **compile** error, not a runtime one, and the tables can be
//! grepped, diffed and stepped through like any other code.
//!
//! This module owns the row types; [`generated`] owns only data.
//!
//! # The ATD / WILDATOM grammar
//!
//! All seven `ATOMTYPE_*.DEF` files share one grammar. An [`AtdRule`] is a
//! conjunction of constraints on a candidate atom — atomic number, degree,
//! attached-hydrogen count, an [`AtomProp`] expression, and a neighbourhood
//! [`AtomPattern`] tree. Rules are tried **in file order** and the first match
//! wins.
//!
//! The environment and property mini-languages are **pre-parsed** into the
//! static AST below, so the typifier walks a tree instead of re-parsing a
//! string for every atom and every candidate rule.

pub mod generated;

pub use generated::atomtype_abcg2::ATOMTYPE_ABCG2;
pub use generated::atomtype_amber::ATOMTYPE_AMBER;
pub use generated::atomtype_bcc::ATOMTYPE_BCC;
pub use generated::atomtype_gas::ATOMTYPE_GAS;
pub use generated::atomtype_gff::ATOMTYPE_GFF;
pub use generated::atomtype_gff2::ATOMTYPE_GFF2;
pub use generated::atomtype_sybyl::ATOMTYPE_SYBYL;
pub use generated::bccparm::{BCC_ALIASES, BCC_CORRECTIONS};
pub use generated::bccparm_abcg2::{ABCG2_ALIASES, ABCG2_CORRECTIONS};
pub use generated::gasparm::GASTEIGER_PARAMS;

/// One oriented bond charge correction from a `BCCPARM*.DAT` table.
///
/// The bond `left|right|bond_type` adds `+delta` to `left` and `-delta` to
/// `right`; the same bond seen in the reverse orientation applies `-delta` /
/// `+delta`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BccCorrectionRow {
    /// BCC atom type on the positive side of the correction.
    pub left: &'static str,
    /// BCC atom type on the negative side of the correction.
    pub right: &'static str,
    /// BCC bond type (1 single, 2 double, 3 triple, 7/8/10 aromatic, 9 delocalized).
    pub bond_type: i32,
    /// Charge transferred from `right` to `left`, in elementary charge units.
    pub delta: f64,
}

/// A `CORR` row: `atom_type` borrows the corrections of `reference`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BccAlias {
    /// The atom type that carries no corrections of its own.
    pub atom_type: &'static str,
    /// The atom type whose corrections it borrows.
    pub reference: &'static str,
}

/// One row of `GASPARM.DAT` — the Gasteiger–Marsili PEOE parameters.
///
/// `a`, `b` and `c` are the electronegativity polynomial coefficients:
/// `chi(q) = a + b*q + c*q^2`. `chi_plus` is the normalisation denominator
/// (the cation electronegativity), **not** a quartic coefficient — the
/// upstream file calls that column `d`. `seed_charge` is the `formal_charge`
/// column, the initial charge q0 the iteration starts from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GasteigerRow {
    /// Gasteiger atom type (`h`, `c3`, `o-1`, `na+`, …).
    pub atom_type: &'static str,
    /// Constant term of the electronegativity polynomial.
    pub a: f64,
    /// Linear coefficient of the electronegativity polynomial.
    pub b: f64,
    /// Quadratic coefficient of the electronegativity polynomial.
    pub c: f64,
    /// Cation electronegativity chi+, the damping denominator (column `d`).
    pub chi_plus: f64,
    /// Seed charge q0 (column `formal_charge`).
    pub seed_charge: f64,
}

/// One alternative of a `WILDATOM` alias: an element, optionally at a fixed degree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WildAtomSpec {
    /// Atomic number.
    pub z: u8,
    /// Required connectivity, or `None` when the alias accepts any degree.
    pub degree: Option<usize>,
}

/// A `WILDATOM` alias — a named set of [`WildAtomSpec`] alternatives.
///
/// The set is file-local: `XB` is `C3 N2 N3 O2 S2 P2` in `ATOMTYPE_BCC.DEF`
/// but `N P` in `ATOMTYPE_GFF.DEF`. Pattern names are resolved against the
/// owning file at generation time, so this table is documentation rather than
/// a runtime lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WildAtom {
    /// Alias name as it appears in the `.DEF` (`XX`, `XA`, …).
    pub name: &'static str,
    /// The element alternatives it stands for.
    pub specs: &'static [WildAtomSpec],
}

/// A topological property an atom can be counted against.
///
/// The lowercase source tokens (`sb` / `db` / `tb`) count aromatic and
/// delocalized bonds as well; the uppercase ones (`SB` / `DB` / `TB`) are
/// strict. That distinction is load-bearing, so the two spellings map to
/// distinct variants rather than to one case-folded name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomProp {
    /// `RG` — member of any ring.
    Rg,
    /// `RG3` — member of a 3-membered ring.
    Rg3,
    /// `RG4` — member of a 4-membered ring.
    Rg4,
    /// `RG5` — member of a 5-membered ring.
    Rg5,
    /// `RG6` — member of a 6-membered ring.
    Rg6,
    /// `RG7` — member of a 7-membered ring.
    Rg7,
    /// `RG8` — member of an 8-membered ring.
    Rg8,
    /// `RG9` — member of a 9-membered ring.
    Rg9,
    /// `RG10` — member of a 10-membered ring.
    Rg10,
    /// `NR` — in no ring at all.
    Nr,
    /// `AR1` — pure aromatic ring (benzene-like).
    Ar1,
    /// `AR2` — planar ring fused to a pure aromatic ring.
    Ar2,
    /// `AR3` — planar ring not otherwise classified.
    Ar3,
    /// `AR4` — ring with one or more sp3 atoms.
    Ar4,
    /// `AR5` — non-planar ring.
    Ar5,
    /// `SB` — single bonds, strictly.
    SbStrict,
    /// `sb` — single bonds, counting aromatic and delocalized ones.
    SbAny,
    /// `DB` — double bonds, strictly.
    DbStrict,
    /// `db` — double bonds, counting aromatic and delocalized ones.
    DbAny,
    /// `TB` — triple bonds, strictly.
    TbStrict,
    /// `tb` — triple bonds, counting aromatic ones.
    TbAny,
    /// `AB` — aromatic bonds.
    Ab,
    /// `DL` — delocalized bonds.
    Dl,
}

/// The `'` / `''` suffix on a property unit.
///
/// It constrains the bond back to the atom the pattern arrived from, so it is
/// only meaningful inside an [`AtomPattern`], never on a rule's top-level
/// property.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropRelation {
    /// `'` — the bond to the previous atom must itself be of this type.
    BondedToPrev,
    /// `''` — the bond to the previous atom must NOT be of this type.
    NotBondedToPrev,
}

/// One `[count]PROP['|'']` unit, e.g. `2DL`, `AR1`, `db'`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PropUnit {
    /// Exact count required; `None` means "at least one".
    pub count: Option<usize>,
    /// The property being counted.
    pub prop: AtomProp,
    /// Optional constraint on the bond back to the previous atom.
    pub relation: Option<PropRelation>,
}

/// A comma-group of a property expression: satisfied when **any** unit matches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PropConstraint {
    /// Dot-separated alternatives (`AR1.AR2` = AR1 **or** AR2).
    pub units: &'static [PropUnit],
}

/// A pre-parsed `[...]` atom-property expression: an AND of ORs.
///
/// `[RG5.RG6,AR1.AR2.AR3]` is "(5- or 6-ring) **and** (AR1, AR2 or AR3)".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PropExpr {
    /// Comma-separated constraints, **all** of which must hold.
    pub constraints: &'static [PropConstraint],
}

/// What an [`AtomPattern`] matches, resolved at generation time.
///
/// The `.DEF` files name pattern atoms with a bare symbol; that name is looked
/// up as `EW`, then as a `WILDATOM` of the owning file, then as an element.
/// Doing that here rather than at match time is what lets the typifier run
/// without a name table — and makes an unknown name a generator error instead
/// of a rule that silently never fires.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatternAtom {
    /// `EW` — any electron-withdrawing atom (N, O, F, S, Cl, Br, I).
    ElectronWithdrawing,
    /// A `WILDATOM` alias, already expanded to its alternatives.
    Wild(&'static [WildAtomSpec]),
    /// A concrete element, by atomic number.
    Element(u8),
}

/// One node of a pre-parsed environment expression, e.g. `C3[AR1](O1,O1)`.
///
/// `children` are neighbours of *this* atom, excluding the atom the match
/// arrived from — so the tree is walked outward, never back on itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AtomPattern {
    /// Which atoms this node can match.
    pub atom: PatternAtom,
    /// Required connectivity, or `None` for any.
    pub degree: Option<usize>,
    /// Optional `[...]` property expression.
    pub property: Option<PropExpr>,
    /// Optional `<label>`, referenced by an [`EnvBond`].
    pub label: Option<&'static str>,
    /// Neighbour patterns that must match, one atom each.
    pub children: &'static [AtomPattern],
}

/// A bond type an [`EnvBond`] can require.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnvBondType {
    /// `any` — the two atoms need only be bonded.
    Any,
    /// A single bond (aromatic and delocalized single bonds count).
    Single,
    /// A double bond (aromatic and delocalized double bonds count).
    Double,
    /// A triple bond.
    Triple,
    /// An aromatic bond.
    Aromatic,
}

/// An `a:b:TYPE` constraint between two `<label>`ed atoms of an environment.
///
/// This is how the `.DEF` closes a ring: it demands that two atoms the pattern
/// reached by different branches are themselves bonded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EnvBond {
    /// Label of the first atom.
    pub a: &'static str,
    /// Label of the second atom.
    pub b: &'static str,
    /// The bond type required between them.
    pub bond: EnvBondType,
}

/// One `ATD` row: a conjunction of constraints on a candidate atom.
///
/// Every `Option` field is an unconstrained `*` / `&` column in the source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AtdRule {
    /// The atom type assigned when the rule matches.
    pub atom_type: &'static str,
    /// Residue name, or `*` for any.
    pub residue: &'static str,
    /// Required atomic number.
    pub atomic_number: Option<u8>,
    /// Required connectivity.
    pub degree: Option<usize>,
    /// Required number of attached hydrogens.
    pub hydrogen_count: Option<usize>,
    /// Required number of electron-withdrawing atoms around the attachment point.
    pub ewd_count: Option<usize>,
    /// Required `[...]` properties of the atom itself.
    pub atom_property: Option<PropExpr>,
    /// Required neighbourhood, as a pre-parsed pattern forest.
    pub environment: Option<&'static [AtomPattern]>,
    /// Required bonds between `<label>`ed environment atoms.
    pub environment_bonds: Option<&'static [EnvBond]>,
}

/// One `ATOMTYPE_*.DEF` file as a typed table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AtdTable {
    /// Upstream file name, e.g. `ATOMTYPE_BCC.DEF`.
    pub name: &'static str,
    /// The `WILDATOM` aliases the file declares.
    pub wildatoms: &'static [WildAtom],
    /// The `ATD` rules, in file order. The first match wins.
    pub rules: &'static [AtdRule],
}
