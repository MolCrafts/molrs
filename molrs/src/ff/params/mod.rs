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
//!
//! # The `parm` force fields
//!
//! [`GAFF`] and [`GAFF2`] are the same idea applied to AMBER's `parm` format:
//! `gaff.dat` and `gaff2.dat` become [`ParmTable`]s of MASS / BOND / ANGLE /
//! DIHE / IMPROPER / NONBON rows. Values are kept in the **upstream's own units
//! and conventions** — degrees, and AMBER's un-halved force constants — because
//! the table is a transcription of the file, not a force field: converting to
//! molrs's radians-and-half-k kernel convention is the job of the reader that
//! populates a [`ForceField`](crate::ff::forcefield::ForceField) from it (see
//! [`crate::ff::forcefield::gaff`]).

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
pub use generated::gaff::GAFF;
pub use generated::gaff2::GAFF2;
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
    /// `AR1` — pure aromatic ring (benzene, pyridine).
    Ar1,
    /// `AR2` — planar ring with two continuous single bonds and at least two
    /// double bonds (imidazole, thiophene, pyrrole).
    Ar2,
    /// `AR3` — planar ring whose double bonds are formed between ring atoms and
    /// non-ring atoms (a quinone, a pyridone).
    Ar3,
    /// `AR4` — a ring that is none of `AR1`, `AR2`, `AR3` or `AR5`.
    Ar4,
    /// `AR5` — pure aliphatic ring, made of sp3 carbon.
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
    /// The conjugated partner of [`atom_type`](Self::atom_type), when it has one.
    ///
    /// An `ATOMTYPE_GFF*.DEF` row only ever emits the **phase-1** name of a
    /// conjugated system — `cc`, `ce`, `cg`, `nc`, `ne`, `pc`, `pe`. antechamber
    /// then 2-colours each conjugated system and renames one colour to its
    /// partner (`cd`, `cf`, `ch`, `nd`, `nf`, `pd`, `pf`), names that appear in
    /// no `.DEF` row at all. That pairing is upstream **data**, not an engine
    /// invention: `PARMCHK.DAT` declares it in a per-type `equivalent_flag`
    /// column (`1` for the phase-1 names, `2` for the phase-2 names, `0` for
    /// everything else). Hence a column of the rule rather than a `match` in the
    /// typifier.
    ///
    /// `None` whenever that column says `0` — including `ATOMTYPE_AMBER.DEF`'s
    /// `CC` / `CD`, which are parm94's histidine carbons and **not** a conjugated
    /// pair despite the spelling.
    pub alternate: Option<&'static str>,
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

// ---------------------------------------------------------------------------
// AMBER `parm` force fields — gaff.dat / gaff2.dat
// ---------------------------------------------------------------------------

/// An atom type of a [`ParmTable`], as an index into its [`masses`](ParmTable::masses).
///
/// **Why an index and not a `&'static str`.** The two `parm` tables hold about
/// 60,000 atom-type slots between them, and a `&str` is a 16-byte fat pointer
/// that also needs a load-time relocation. Spelling every slot out cost **1416
/// KB** of stripped release binary, measured; interning to one byte costs
/// [see the module docs of `generated`] a fraction of that, and the table stays
/// just as readable because the generator emits a named `const` per type
/// (`T_C3`), never a bare number.
///
/// The index is the type's row in the `MASS` section — the section that
/// *declares* the atom types — so [`ParmTable::name_of`] is an array read, and a
/// type with no `MASS` row is a generator error rather than a dangling index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParmType(pub u8);

/// One `MASS` row of an AMBER `parm` table.
///
/// The row order of this section **is** the [`ParmType`] numbering.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParmMassRow {
    /// GAFF atom type (`c3`, `ha`, `os`, …).
    pub atom_type: &'static str,
    /// Atomic mass, in amu.
    pub mass: f64,
    /// Atomic polarizability, in Å³. Read by the polarizable force fields only;
    /// a fixed-charge GAFF run never touches it.
    pub polarizability: f64,
}

/// One `BOND` row: `E = force_constant · (r − length)²`.
///
/// That is **AMBER's** convention and it carries no ½ — unlike molrs's
/// [`BondHarmonic`](crate::ff::potential::bond::harmonic::BondHarmonic), whose
/// `k0` is `2 · force_constant`. The factor is applied where the units are
/// normalised (the [`gaff`](crate::ff::forcefield::gaff) reader), never here:
/// this row is what the file says.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParmBondRow {
    /// Atom type of the first endpoint.
    pub i: ParmType,
    /// Atom type of the second endpoint.
    pub j: ParmType,
    /// Force constant, in kcal/mol/Å².
    pub force_constant: f64,
    /// Equilibrium bond length, in Å.
    pub length: f64,
}

/// One `ANGLE` row: `E = force_constant · (θ − angle_deg)²`, θ in radians.
///
/// Again AMBER's un-halved convention, and again the equilibrium angle is left
/// in the **degrees** the file writes it in — molrs consumes radians, and that
/// conversion belongs to the reader boundary, not to a transcription.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParmAngleRow {
    /// Atom type of the first leg.
    pub i: ParmType,
    /// Atom type of the vertex.
    pub j: ParmType,
    /// Atom type of the second leg.
    pub k: ParmType,
    /// Force constant, in kcal/mol/rad².
    pub force_constant: f64,
    /// Equilibrium angle, in **degrees**.
    pub angle_deg: f64,
}

/// One `DIHE` row — a single cosine term of a proper torsion:
///
/// ```text
/// E = (barrier / divisor) · [1 + cos(periodicity · φ − phase_deg)]
/// ```
///
/// A slot is `None` where the file writes the `X` wildcard, which matches any
/// atom type. Modelling that as `Option` rather than as a distinguished index is
/// what makes a wildcard impossible to read as a concrete type by accident:
/// there are 607 such rows per table, and the exact-match population path has to
/// skip every one of them while the parmchk2-style fallback lives on them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParmDihedralRow {
    /// Atom type of the first atom, or `None` for the `X` wildcard.
    pub i: Option<ParmType>,
    /// Atom type of the second (bonded-pair) atom, or `None`.
    pub j: Option<ParmType>,
    /// Atom type of the third (bonded-pair) atom, or `None`.
    pub k: Option<ParmType>,
    /// Atom type of the fourth atom, or `None`.
    pub l: Option<ParmType>,
    /// `IDIVF` — the number of torsions sharing this barrier. The per-torsion
    /// force constant is `barrier / divisor`.
    pub divisor: u32,
    /// `PK` — the **total** barrier height, in kcal/mol, before division.
    pub barrier: f64,
    /// Phase, in **degrees**.
    pub phase_deg: f64,
    /// `|PN|` — the multiplicity.
    pub periodicity: u32,
    /// Upstream wrote `PN` negative: another cosine term for the *same* quartet
    /// follows on the next row. The sign is a structural fact about the file
    /// rather than a number, so it becomes a flag of its own and `periodicity`
    /// stays a magnitude no kernel can be handed a negative multiplicity from.
    pub more_terms: bool,
}

/// One `IMPROPER` row: `E = barrier · [1 + cos(periodicity · φ − phase_deg)]`.
///
/// **The central atom is [`k`](Self::k) — the third** — which is AMBER's
/// convention, not molrs's (`Atomistic::add_improper` documents `i` as
/// conventionally central, and MMFF's Wilson out-of-plane puts it second). φ is
/// the I-J-K-L dihedral, so the slot order is load-bearing: an improper relation
/// built from this row must be written in it.
///
/// There is no `divisor`: an improper's barrier is never divided.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParmImproperRow {
    /// Atom type of the first peripheral atom, or `None` for the `X` wildcard.
    pub i: Option<ParmType>,
    /// Atom type of the second peripheral atom, or `None`.
    pub j: Option<ParmType>,
    /// Atom type of the **central** atom, or `None`.
    pub k: Option<ParmType>,
    /// Atom type of the third peripheral atom, or `None`.
    pub l: Option<ParmType>,
    /// Barrier height, in kcal/mol.
    pub barrier: f64,
    /// Phase, in **degrees** (180 for every GAFF improper).
    pub phase_deg: f64,
    /// Multiplicity (2 for every GAFF improper).
    pub periodicity: u32,
}

/// One `NONBON` row, under the `MOD4  RE` label: R\* and ε.
///
/// `RE` is what declares these two columns to be a radius and a well depth;
/// under an `AC` label they would be the A/C coefficients of the same 12-6 form
/// instead. The generator refuses any other label rather than reinterpret them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParmNonbondedRow {
    /// The atom type these parameters belong to.
    pub atom_type: ParmType,
    /// R\* — **half** the LJ minimum-energy separation, in Å. Not σ:
    /// `σ = 2·R* / 2^(1/6)`.
    pub r_min_half: f64,
    /// Well depth ε, in kcal/mol.
    pub epsilon: f64,
}

/// One AMBER `parm` force field (`gaff.dat`, `gaff2.dat`) as a typed table.
///
/// Rows keep file order in every section, which is load-bearing twice over: for
/// `dihedrals`, consecutive rows sharing a quartet are the cosine terms of one
/// torsion; for `impropers`, the first matching row wins.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParmTable {
    /// Upstream file name, e.g. `gaff.dat`.
    pub name: &'static str,
    /// The `MASS` section. Its row order is the [`ParmType`] numbering.
    pub masses: &'static [ParmMassRow],
    /// The `BOND` section.
    pub bonds: &'static [ParmBondRow],
    /// The `ANGLE` section.
    pub angles: &'static [ParmAngleRow],
    /// The `DIHE` section.
    pub dihedrals: &'static [ParmDihedralRow],
    /// The `IMPROPER` section.
    pub impropers: &'static [ParmImproperRow],
    /// The `NONBON` section.
    pub nonbonded: &'static [ParmNonbondedRow],
}

impl ParmTable {
    /// The name of an interned atom type, e.g. `c3`.
    ///
    /// # Panics
    ///
    /// If `ty` is not a type of this table. Every [`ParmType`] in a table's rows
    /// indexes that table's own `MASS` section by construction, so this can only
    /// fire if one table's type is used against another's.
    pub fn name_of(&self, ty: ParmType) -> &'static str {
        self.masses[usize::from(ty.0)].atom_type
    }

    /// The interned form of a type name, or `None` if this table has no such type.
    ///
    /// Linear over the `MASS` section (83 / 99 rows). Callers that resolve many
    /// labels should build a map once instead of calling this per atom.
    pub fn type_of(&self, name: &str) -> Option<ParmType> {
        self.masses
            .iter()
            .position(|row| row.atom_type == name)
            .map(|idx| ParmType(idx as u8))
    }
}
