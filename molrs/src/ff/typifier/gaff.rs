//! GAFF / GAFF2 — a [`Typifier`] over the compiled `parm` tables.
//!
//! [`GaffTypifier`] matches a molecule whose atoms already carry GAFF atom types
//! (what [`AtdTypifier`](crate::ff::typifier::AtdTypifier) with
//! [`AtdParameterSet::Gff`](crate::ff::typifier::AtdParameterSet::Gff) stamps, or
//! labels written by any other means): it enumerates the molecule's bonded terms
//! and looks every one of them up in the [`GAFF`] / [`GAFF2`] static table.
//! Nothing is parsed: the tables are `&'static` Rust data (see
//! [`crate::ff::params`]). Typing atoms is not this typifier's job; the caller
//! composes the two:
//!
//! ```ignore
//! let labelled = Typing::new(AtdTypifier::new(AtdParameterSet::Gff)).typify(&mol)?;
//! let mut gaff = Typing::new(GaffTypifier::new(GaffParameterSet::Gaff));
//! let typed = gaff.typify(&labelled)?;
//! ```
//!
//! # Exact rows first, then the estimator
//!
//! A term is looked up first **only** against rows whose every slot is a
//! concrete atom type. A term no such row covers goes to the table's
//! [`Parmchk2Estimator`] ([`gaff_estimator`]): a wildcard (`X`) row that covers
//! it is a parameter, anything reached by analogy or formula is an estimate and
//! says so (see `Typifier::r#match` on [`GaffTypifier`]). A term neither
//! covers is missing, and every missing term is reported at once.
//!
//! Matching a term in either orientation (`c3-c3-oh` covers `oh-c3-c3`) is not a
//! fallback but the undirected nature of a bonded term, and the generator
//! guarantees no table holds both a term and its reverse as separate rows.
//!
//! Impropers are the one exception to "missing is an error": AMBER adds an
//! improper only at a centre `PARMCHK.DAT` flags as planar, and the estimator
//! always answers one.
//!
//! # Units
//!
//! The table holds what `gaff.dat` says; the kernels want molrs's conventions.
//! The candidate library ([`Typifier::library`]) is the boundary between the
//! two, and the only place the conversions happen:
//!
//! | upstream | molrs |
//! |---|---|
//! | `E = K(r−r₀)²` | `E = ½k(r−r₀)²`, so `k = 2·K` |
//! | `E = K(θ−θ₀)²` | `E = ½k(θ−θ₀)²`, so `k = 2·K` |
//! | θ₀ and phases in degrees | radians |
//! | one `PK` shared by `IDIVF` torsions | one `k` per torsion: `k = PK/IDIVF` |
//! | R\*, half the LJ minimum separation | σ = 2·R\*/2^(1/6) |

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::OnceLock;

use molrs::store::keys;
use molrs::store::type_labels::TypeName;
use molrs::system::atomistic::ImproperId;
use molrs::{AtomId, Atomistic};

use crate::ff::constants::VACUUM_DIELECTRIC;
use crate::ff::forcefield::{DefError, ForceField, Params, SpecialBonds};
use crate::ff::params::amber::{AMBER_COULOMB, AMBER_SCEE, AMBER_SCNB};
use crate::ff::params::{
    GAFF, GAFF2, ParmAngleRow, ParmBondRow, ParmDihedralRow, ParmImproperRow, ParmNonbondedRow,
    ParmTable, ParmType,
};
use crate::ff::typifier::estimate::{
    BondedTerm, EmpiricalSet, Estimate, Parmchk2Estimator, Provenance, TypifierParameterContext,
};
use crate::ff::typifier::{Annotation, Match, Typifier};

/// Which AMBER `parm` force field to match against.
///
/// Pairs one-to-one with the atom-type table that produces the labels:
/// [`AtdParameterSet::Gff`](crate::ff::typifier::AtdParameterSet::Gff) types a
/// molecule for [`Gaff`](Self::Gaff), `Gff2` for [`Gaff2`](Self::Gaff2). Typing
/// with one and parameterising with the other is a category error no signature
/// can catch — `gaff2.dat` declares atom types (`cs`, `sq`, …) that `gaff.dat`
/// has no row for at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GaffParameterSet {
    /// `gaff.dat` — GAFF 1.81.
    Gaff,
    /// `gaff2.dat` — GAFF 2.
    Gaff2,
}

impl GaffParameterSet {
    /// The compile-time table this set names.
    pub fn table(self) -> ParmTable {
        match self {
            Self::Gaff => GAFF,
            Self::Gaff2 => GAFF2,
        }
    }

    /// The name of this set's library, and so of every typing output seeded
    /// from it.
    pub fn name(self) -> &'static str {
        match self {
            Self::Gaff => "gaff",
            Self::Gaff2 => "gaff2",
        }
    }

    /// The empirical bond / angle constants that pair with this table.
    fn empirical(self) -> EmpiricalSet {
        match self {
            Self::Gaff => EmpiricalSet::Gaff,
            Self::Gaff2 => EmpiricalSet::Gaff2,
        }
    }

    /// This set's candidate library, built on first use and shared after.
    ///
    /// `gaff.dat` transcribes to some 5,900 candidate rows, and it is the *same*
    /// 5,900 rows for every molecule ever parameterised; rebuilding them per call
    /// cost more than the whole rest of the typing put together. The library is
    /// named after the set ([`name`](Self::name)).
    fn library(self) -> &'static ForceField {
        static GAFF_LIBRARY: OnceLock<ForceField> = OnceLock::new();
        static GAFF2_LIBRARY: OnceLock<ForceField> = OnceLock::new();

        let cell = match self {
            Self::Gaff => &GAFF_LIBRARY,
            Self::Gaff2 => &GAFF2_LIBRARY,
        };
        cell.get_or_init(|| {
            let mut ff = candidate_forcefield(self.table());
            ff.name = self.name().to_owned();
            ff
        })
    }
}

/// The missing-parameter estimator of one `parm` table.
///
/// The estimator is the general one — [`Parmchk2Estimator`], of which molrs has
/// exactly one — and this is how GAFF reaches it: over the set's candidate
/// library, which the estimator flattens by style kind. Wildcard rows are
/// included, because they are most of what the estimator is for. The library's
/// rows are already in molrs's conventions (`k = 2·K`, radians), and so is every
/// estimate drawn from them (`empirical`'s formulas are doubled where the cascade
/// applies them), so a looked-up row and an estimate reach the output in one
/// convention.
///
/// Memoised per parameter set, like the library it reads: the table is
/// `&'static` data, and rebuilding the estimator per call doubled the test
/// suite's wall time when it was written that way.
pub fn gaff_estimator(set: GaffParameterSet) -> &'static Parmchk2Estimator {
    static GAFF_ESTIMATOR: OnceLock<Parmchk2Estimator> = OnceLock::new();
    static GAFF2_ESTIMATOR: OnceLock<Parmchk2Estimator> = OnceLock::new();

    let cell = match set {
        GaffParameterSet::Gaff => &GAFF_ESTIMATOR,
        GaffParameterSet::Gaff2 => &GAFF2_ESTIMATOR,
    };
    cell.get_or_init(|| {
        let candidates = set.library();
        let context = TypifierParameterContext::new().with_forcefield_elements(candidates);
        Parmchk2Estimator::with_context(candidates, context).with_empirical(set.empirical())
    })
}

/// Transcribe a whole `parm` table into a [`ForceField`] of candidate rows.
///
/// An input-free constructor over a compiled `ff/params` table: the one
/// definition result it can meet is the table's own, and
/// `tests::candidate_forcefield_defines_without_conflict` proves it `Ok` for
/// GAFF and GAFF2.
fn candidate_forcefield(table: ParmTable) -> ForceField {
    try_candidate_forcefield(table).expect(
        "GAFF/GAFF2 parm table defines without conflict — proved by \
         ff::typifier::gaff::tests::candidate_forcefield_defines_without_conflict",
    )
}

/// The fallible body of [`candidate_forcefield`].
///
/// Every row, wildcards and all — the estimator's whole job is the rows the exact
/// matcher skips. `X` fills a wildcard slot, which is the spelling
/// [`Candidate`](crate::ff::typifier::estimate::Candidate) reads.
///
/// The library declares AMBER's 1-4 handling — 1-2 / 1-3 excluded outright, 1-4
/// scaled by 1/SCNB = 1/2 (LJ) and 1/SCEE = 1/1.2 (Coulomb) — so every typing
/// output seeded from it carries the weights.
fn try_candidate_forcefield(table: ParmTable) -> Result<ForceField, DefError> {
    let mut ff = ForceField::new(table.name);
    ff.set_special_bonds(SpecialBonds {
        lj: [0.0, 0.0, 1.0 / AMBER_SCNB],
        coul: [0.0, 0.0, 1.0 / AMBER_SCEE],
    });

    let atoms = ff.def_style("atom", "full", Params::new())?;
    for row in table.masses {
        atoms.def_type(
            row.atom_type,
            &[],
            Params::from_pairs(&[("mass", row.mass)]),
        )?;
    }

    let lj = ff.def_style("pair", "lj/cut", Params::new())?;
    for row in table.nonbonded {
        let name = table.name_of(row.atom_type);
        lj.def_type(name, &[name], Params::from_pairs(&lj_params(row)))?;
    }
    // GAFF/AMBER uses the unbuffered Coulomb form. The constant is force-field
    // data (and differs measurably from CODATA and MMFF), while `delta = 0`
    // explicitly selects the unbuffered branch of the shared `coul/cut` kernel.
    ff.def_style(
        "pair",
        "coul/cut",
        Params::from_pairs(&[
            ("coulomb", AMBER_COULOMB),
            ("dielectric", VACUUM_DIELECTRIC),
            ("delta", 0.0),
        ]),
    )?;

    let bonds = ff.def_style("bond", "harmonic", Params::new())?;
    for row in table.bonds {
        let ends = [row.i, row.j].map(|ty| table.name_of(ty));
        bonds.def_type(
            TypeName::join(&ends).map_err(DefError::Name)?.as_str(),
            &ends,
            Params::from_pairs(&bond_params(row)),
        )?;
    }

    let angles = ff.def_style("angle", "harmonic", Params::new())?;
    for row in table.angles {
        let ends = [row.i, row.j, row.k].map(|ty| table.name_of(ty));
        angles.def_type(
            TypeName::join(&ends).map_err(DefError::Name)?.as_str(),
            &ends,
            Params::from_pairs(&angle_params(row)),
        )?;
    }

    // The DIHE section writes one row per cosine term; a torsion is the whole group
    // of rows sharing a quartet, so the group is ONE candidate.
    let dihedrals = ff.def_style("dihedral", "periodic", Params::new())?;
    let mut groups: Vec<([Option<ParmType>; 4], Vec<&ParmDihedralRow>)> = Vec::new();
    for row in table.dihedrals {
        let slots = [row.i, row.j, row.k, row.l];
        match groups.iter_mut().find(|(seen, _)| *seen == slots) {
            Some((_, rows)) => rows.push(row),
            None => groups.push((slots, vec![row])),
        }
    }
    for (slots, rows) in &groups {
        let ends = slots.map(|slot| slot_name(&table, &slot));
        let params = dihedral_params(rows);
        dihedrals.def_type(
            TypeName::join(&ends).map_err(DefError::Name)?.as_str(),
            &ends,
            Params::from_pairs(&borrowed(&params)),
        )?;
    }

    let impropers = ff.def_style("improper", "periodic", Params::new())?;
    for row in table.impropers {
        let ends = [row.i, row.j, row.k, row.l].map(|slot| slot_name(&table, &slot));
        impropers.def_type(
            TypeName::join(&ends).map_err(DefError::Name)?.as_str(),
            &ends,
            Params::from_pairs(&improper_params(row)),
        )?;
    }

    Ok(ff)
}

/// A row slot's atom-type name, or `X` where the file wrote a wildcard.
fn slot_name(table: &ParmTable, slot: &Option<ParmType>) -> &'static str {
    slot.map_or("X", |ty| table.name_of(ty))
}

/// One `NONBON` row as `lj/cut` params: ε as written, σ = 2·R\*/2^(1/6).
///
/// R\* is **half** the separation at the Lennard-Jones minimum, so the minimum
/// itself is `2·R*` and σ — where the potential crosses zero — is that over the
/// sixth root of two. Reading R\* as σ would inflate every radius by 78%.
fn lj_params(row: &ParmNonbondedRow) -> [(&'static str, f64); 2] {
    [
        ("epsilon", row.epsilon),
        ("sigma", 2.0 * row.r_min_half / 2f64.powf(1.0 / 6.0)),
    ]
}

/// One `BOND` row as candidate params, in molrs's convention.
///
/// AMBER writes `E = K(r−r₀)²` and molrs's kernel writes `E = ½k(r−r₀)²`, so
/// `k = 2·K`, and the doubling happens **here**, at the table boundary. It used
/// to happen where the output force field was assembled instead, which left the
/// candidate rows and every estimate drawn from them in AMBER's convention
/// under the same name the kernels read (spec ff-params-01).
fn bond_params(row: &ParmBondRow) -> [(&'static str, f64); 2] {
    [("k", 2.0 * row.force_constant), ("r0", row.length)]
}

/// One `ANGLE` row as candidate params: `k = 2·K`, θ₀ in **radians**.
fn angle_params(row: &ParmAngleRow) -> [(&'static str, f64); 2] {
    [
        ("k", 2.0 * row.force_constant),
        ("theta0", row.angle_deg.to_radians()),
    ]
}

/// The cosine terms of one torsion, in the `k{m}` / `periodicity{m}` / `phase{m}` encoding
/// [`DihedralPeriodic`](crate::ff::potential::dihedral::periodic::DihedralPeriodic)
/// scans upward from `m = 1`. `k = PK / IDIVF`; phases in radians.
fn dihedral_params(rows: &[&ParmDihedralRow]) -> Vec<(String, f64)> {
    let mut out = Vec::with_capacity(rows.len() * 3);
    for (m, row) in rows.iter().enumerate() {
        let m = m + 1;
        out.push((format!("k{m}"), row.barrier / f64::from(row.divisor)));
        out.push((format!("periodicity{m}"), f64::from(row.periodicity)));
        out.push((format!("phase{m}"), row.phase_deg.to_radians()));
    }
    out
}

/// One `IMPROPER` row as candidate params. An improper's barrier is never divided.
fn improper_params(row: &ParmImproperRow) -> [(&'static str, f64); 3] {
    [
        ("k", row.barrier),
        ("periodicity", f64::from(row.periodicity)),
        ("phase", row.phase_deg.to_radians()),
    ]
}

/// `&[(String, f64)]` as the `&[(&str, f64)]` [`Params::from_pairs`] takes.
fn borrowed(params: &[(String, f64)]) -> Vec<(&str, f64)> {
    params.iter().map(|(k, v)| (k.as_str(), *v)).collect()
}

/// A term of the molecule that neither an exact row nor the estimator covers.
///
/// Atom types are the GAFF labels of the term's atoms, in the molecule's own
/// order. Every miss is collected before returning, so one call names every
/// parameter the table is short of, rather than only the first.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum MissingTerm {
    /// The table declares no such atom type: it has no `MASS` row.
    Mass(String),
    /// No `NONBON` row for this atom type. `gaff2.dat` has none for `ow` / `hw`:
    /// the water types take their Lennard-Jones parameters from a water model,
    /// not from the force field.
    Nonbonded(String),
    /// No `BOND` row for this pair of atom types.
    Bond([String; 2]),
    /// No `ANGLE` row for this triple (vertex in the middle).
    Angle([String; 3]),
    /// No `DIHE` row for this quartet.
    Dihedral([String; 4]),
}

impl std::fmt::Display for MissingTerm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Mass(t) => write!(f, "MASS {t}"),
            Self::Nonbonded(t) => write!(f, "NONBON {t}"),
            Self::Bond(t) => write!(f, "BOND {}", t.join("-")),
            Self::Angle(t) => write!(f, "ANGLE {}", t.join("-")),
            Self::Dihedral(t) => write!(f, "DIHE {}", t.join("-")),
        }
    }
}

/// Why a molecule could not be matched against a GAFF table. Reaches callers
/// as its `Display` text, through [`Typifier::r#match`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum GaffError {
    /// An atom carries no [`keys::TYPE`] label: the molecule was never typed.
    Untyped {
        /// The atom's 0-based index in graph atom order.
        atom: usize,
    },
    /// Neither the table nor the estimator covers one or more of the
    /// molecule's terms.
    ///
    /// An atom type the table does not declare at all ([`MissingTerm::Mass`])
    /// short-circuits: every term touching it would be missing too, and listing
    /// them would bury the one fact that matters.
    Missing {
        /// The upstream table that was searched, e.g. `gaff.dat`.
        table: &'static str,
        /// Every uncovered term, deduplicated, in a stable order.
        terms: Vec<MissingTerm>,
    },
    /// The graph could not be walked — a defect in the input.
    Malformed {
        /// What the graph layer said.
        detail: String,
    },
}

impl std::fmt::Display for GaffError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Untyped { atom } => {
                write!(f, "atom {atom} carries no `{}` label", keys::TYPE)
            }
            Self::Missing { table, terms } => {
                let listed: Vec<String> = terms.iter().map(ToString::to_string).collect();
                write!(
                    f,
                    "{table} has no exact parameter for {} term(s): {}",
                    terms.len(),
                    listed.join(", ")
                )
            }
            Self::Malformed { detail } => write!(f, "{detail}"),
        }
    }
}

impl std::error::Error for GaffError {}

/// The GAFF / GAFF2 typifier: matches a molecule whose atoms already carry GAFF
/// atom types against one `parm` table.
///
/// Run it through [`Typing`](crate::ff::typifier::Typing); its output holds the
/// types the molecules typed so far assigned, under the AMBER 1-4 weights its
/// library declares.
///
/// # Example
///
/// Methane, end to end — perceive + type atoms, match terms, evaluate:
///
/// ```
/// use molrs::Atomistic;
/// use molrs::ff::potential::{PotentialCompiler, intramolecular_pairs};
/// use molrs::ff::typifier::Typing;
/// use molrs::ff::typifier::atd::{AtdParameterSet, AtdTypifier};
/// use molrs::ff::typifier::gaff::{GaffParameterSet, GaffTypifier};
/// use molrs::store::keys;
/// use molrs::system::molgraph::PropValue;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut mol = Atomistic::new();
/// let c = mol.add_atom_xyz("C", 0.000, 0.000, 0.000);
/// for (x, y, z) in [
///     (0.629, 0.629, 0.629),
///     (-0.629, -0.629, 0.629),
///     (-0.629, 0.629, -0.629),
///     (0.629, -0.629, -0.629),
/// ] {
///     let h = mol.add_atom_xyz("H", x, y, z);
///     mol.add_bond(c, h)?;
/// }
///
/// // ATD types the atoms (perceiving the BCC bond types on the way through) …
/// let labelled = Typing::new(AtdTypifier::new(AtdParameterSet::Gff)).typify(&mol)?;
/// // … and GAFF matches the bonded terms of the labelled molecule.
/// let mut gaff = Typing::new(GaffTypifier::new(GaffParameterSet::Gaff));
/// let typed = gaff.typify(&labelled)?;
/// let ff = gaff.forcefield();
///
/// // Every bond's `type` is now the force field's NAME, not a perceived integer.
/// let name = PropValue::Str("c3-hc".to_owned());
/// for (_, bond) in typed.bonds() {
///     assert_eq!(bond.props.get(keys::TYPE), Some(&name));
/// }
///
/// let mut frame = typed.to_frame()?;
/// let pairs = intramolecular_pairs(&frame, ff.special_bonds())?;
/// frame.insert("pairs", pairs);
/// let potentials = PotentialCompiler::new(ff).compile(&frame)?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GaffTypifier {
    set: GaffParameterSet,
}

impl GaffTypifier {
    /// A typifier over `set`'s table. The library and estimator it reads are
    /// shared per set, so construction is free.
    pub fn new(set: GaffParameterSet) -> Self {
        Self { set }
    }

    /// The body of [`Typifier::r#match`], with the typed error.
    fn match_terms(&self, graph: &mut Atomistic) -> Result<Match, GaffError> {
        let index = TableIndex::new(self.set.table());
        let estimator = gaff_estimator(self.set);
        let type_of = index.intern_atoms(graph)?;
        let mut missed = Misses::default();

        // Topology first, so every positional vector below is read off the
        // graph `Match::write_onto` stamps: angles and dihedrals regenerated
        // from the bond graph, impropers rebuilt from the table.
        graph
            .generate_topology(true, true, false, true)
            .map_err(malformed)?;
        let mut improper_terms = add_impropers(graph, &index, estimator, &type_of)?;

        let mut m = Match::default();

        // --- atoms: mass (atom/full) + Lennard-Jones (pair rows) ---
        let used: BTreeSet<ParmType> = type_of.values().copied().collect();
        for &ty in &used {
            // Interning proved the MASS row exists. The NONBON row is a separate
            // section and can be absent (gaff2's `ow` / `hw`).
            if index.nonbonded(ty).is_none() {
                missed.note(MissingTerm::Nonbonded(index.name_of(ty).to_owned()));
            }
        }
        let library = self.set.library();
        m.nodes = graph
            .atoms()
            .map(|(id, _)| {
                let name = index.name_of(type_of[&id]);
                let mass = index.table.masses[usize::from(type_of[&id].0)].mass;
                vec![(
                    keys::TYPE.to_owned(),
                    Annotation::Type {
                        style: "full".to_owned(),
                        name: name.to_owned(),
                        endpoints: Vec::new(),
                        params: Params::from_pairs(&[("mass", mass)]),
                    },
                )]
            })
            .collect();

        // --- bonds ---
        let bonds: Vec<(AtomId, AtomId)> = graph
            .bonds()
            .map(|(_, b)| (b.nodes[0], b.nodes[1]))
            .collect();
        for (i, j) in bonds {
            let (ti, tj) = (type_of[&i], type_of[&j]);
            let resolved = match index.bond(ti, tj) {
                Some(row) => Some((
                    index.names([row.i, row.j]).to_vec(),
                    Bonded::matched(Params::from_pairs(&bond_params(row))),
                )),
                None => {
                    let names = index.names([ti, tj]);
                    match Bonded::estimate(estimator, &BondedTerm::Bond(names.clone()))? {
                        Some(resolved) => Some(resolved),
                        None => {
                            missed.note(MissingTerm::Bond(names));
                            None
                        }
                    }
                }
            };
            m.bonds.push(Bonded::annotation("harmonic", resolved)?);
        }

        // --- angles ---
        let angles: Vec<[AtomId; 3]> = graph
            .angles()
            .map(|(_, a)| [a.nodes[0], a.nodes[1], a.nodes[2]])
            .collect();
        for [i, j, k] in angles {
            let (ti, tj, tk) = (type_of[&i], type_of[&j], type_of[&k]);
            let resolved = match index.angle(ti, tj, tk) {
                Some(row) => Some((
                    index.names([row.i, row.j, row.k]).to_vec(),
                    Bonded::matched(Params::from_pairs(&angle_params(row))),
                )),
                None => {
                    let names = index.names([ti, tj, tk]);
                    match Bonded::estimate(estimator, &BondedTerm::Angle(names.clone()))? {
                        Some(resolved) => Some(resolved),
                        None => {
                            missed.note(MissingTerm::Angle(names));
                            None
                        }
                    }
                }
            };
            m.angles.push(Bonded::annotation("harmonic", resolved)?);
        }

        // --- dihedrals ---
        // A torsion the exact index misses is not yet missing: the wildcard rows
        // are most of the DIHE section, and parmchk2 estimates only what they too
        // fail to cover.
        let dihedrals: Vec<[AtomId; 4]> = graph
            .dihedrals()
            .map(|(_, d)| [d.nodes[0], d.nodes[1], d.nodes[2], d.nodes[3]])
            .collect();
        for atoms in dihedrals {
            let quartet = atoms.map(|a| type_of[&a]);
            let resolved = match index.dihedral(quartet) {
                Some(rows) => {
                    let first = rows[0];
                    Some((
                        index
                            .names(concrete(first.i, first.j, first.k, first.l))
                            .to_vec(),
                        Bonded::matched(Params::from_pairs(&borrowed(&dihedral_params(rows)))),
                    ))
                }
                None => {
                    let names = index.names(quartet);
                    match Bonded::estimate(estimator, &BondedTerm::Dihedral(names.clone()))? {
                        Some(resolved) => Some(resolved),
                        None => {
                            missed.note(MissingTerm::Dihedral(names));
                            None
                        }
                    }
                }
            };
            m.dihedrals.push(Bonded::annotation("periodic", resolved)?);
        }

        // --- impropers: positional against the rows `add_impropers` left ---
        let improper_ids: Vec<ImproperId> = graph.impropers().map(|(id, _)| id).collect();
        for id in improper_ids {
            m.impropers
                .push(Bonded::annotation("periodic", improper_terms.remove(&id))?);
        }

        let terms = missed.into_terms();
        if !terms.is_empty() {
            return Err(GaffError::Missing {
                table: index.table.name,
                terms,
            });
        }

        let used: HashSet<&str> = used.iter().map(|&ty| index.name_of(ty)).collect();
        m.declare_styles_of(library);
        m.add_pairs_among(library, &used);
        Ok(m)
    }
}

impl Typifier for GaffTypifier {
    /// Match the bonded terms of a molecule whose atoms carry GAFF types.
    ///
    /// Angles and dihedrals are regenerated from the bond graph onto `graph`;
    /// impropers are rebuilt — in AMBER's central-atom-third order — at every
    /// 3-coordinate centre `PARMCHK.DAT` flags as planar. Atoms get `type` as a
    /// type under `atom/full` with the table's `mass`; every bond, angle,
    /// dihedral and improper gets `type` as a type under `bond/harmonic`,
    /// `angle/harmonic`, `dihedral/periodic` or `improper/periodic`. A term an
    /// exact row covers is named by that row; any other is named by
    /// [`BondedTerm::type_name`], and when it was estimated rather than covered
    /// by a wildcard row its params carry the provenance keys `estimated`,
    /// `estimate_penalty`, `estimate_method` and `estimate_analog`. Styles:
    /// every library style, in library order. Pairs: the library's `lj/cut`
    /// rows of the atom types used.
    ///
    /// # The bond `type` is the force field's
    ///
    /// Perception keeps its answer in its own
    /// [`BCC_BOND_TYPE`](molrs::perceive::bond_type::BCC_BOND_TYPE) prop, so the
    /// bond's [`keys::TYPE`] is free for the force-field type *name* (`c3-hc`),
    /// which `to_frame` writes to the `bonds` block's `type` column and every
    /// bonded kernel resolves its parameters by.
    ///
    /// # Errors
    ///
    /// An atom that carries no [`keys::TYPE`]; a table that declares none of an
    /// atom's type, or covers — exactly, by wildcard or by estimate — none of a
    /// term, listing **every** such term; a graph that cannot be walked.
    fn r#match(&self, graph: &mut Atomistic) -> Result<Match, String> {
        self.match_terms(graph).map_err(|e| e.to_string())
    }

    /// The set's candidate library: every row of the table (wildcards
    /// included) in molrs's units, the `atom/full`, `pair/lj/cut`,
    /// `pair/coul/cut`, `bond/harmonic`, `angle/harmonic`, `dihedral/periodic`
    /// and `improper/periodic` styles, and AMBER's special_bonds. Built once per
    /// set.
    fn library(&self) -> &ForceField {
        self.set.library()
    }
}

/// One bonded term of the molecule: its parameters, and — when they were estimated
/// rather than looked up — what that cost.
///
/// The parameters are in the library's convention whether they came from a row
/// of the table or from the estimator: that is the point of the convention, and
/// it is why one struct serves all four arities. The provenance rides with them
/// rather than being reconstructed afterwards — an estimate that does not say so
/// is not auditable.
struct Bonded {
    params: Params,
    estimate: Option<Provenance>,
}

impl Bonded {
    fn matched(params: Params) -> Self {
        Self {
            params,
            estimate: None,
        }
    }

    /// Ask the estimator for `term`: its [`BondedTerm::endpoints`] and params,
    /// or `None` when nothing can produce it.
    fn estimate(
        estimator: &Parmchk2Estimator,
        term: &BondedTerm,
    ) -> Result<Option<(Vec<String>, Self)>, GaffError> {
        let Some(estimate) = estimator.estimate(term) else {
            return Ok(None);
        };
        Ok(Some((
            term.endpoints().into_iter().map(str::to_owned).collect(),
            Self::from(estimate),
        )))
    }

    /// The annotations of one term: `type` → the term's type under `style`,
    /// on its endpoints and named by their [`TypeName::join`], with the
    /// provenance keys written into its params when it was estimated. Nothing
    /// for an unresolved term (a miss, which fails the match).
    fn annotation(
        style: &str,
        resolved: Option<(Vec<String>, Self)>,
    ) -> Result<Vec<(String, Annotation)>, GaffError> {
        let Some((endpoints, term)) = resolved else {
            return Ok(Vec::new());
        };
        let ends: Vec<&str> = endpoints.iter().map(String::as_str).collect();
        let name = TypeName::join(&ends).map_err(malformed)?;
        let Bonded {
            mut params,
            estimate,
        } = term;
        if let Some(provenance) = &estimate {
            provenance.write_onto(&mut params);
        }
        Ok(vec![(
            keys::TYPE.to_owned(),
            Annotation::Type {
                style: style.to_owned(),
                name: name.to_string(),
                endpoints,
                params,
            },
        )])
    }
}

impl From<Estimate> for Bonded {
    /// A term the estimator answered. A **wildcard row** answer is
    /// [`Estimate::Covered`]: the table covered the term, so it is a parameter and
    /// carries no provenance — which is exactly what parmchk2 does.
    fn from(estimate: Estimate) -> Self {
        match estimate {
            Estimate::Covered { params, .. } => Self::matched(params),
            Estimate::Estimated { params, provenance } => Self {
                params,
                estimate: Some(provenance),
            },
        }
    }
}

/// Enumerate AMBER impropers onto `graph` and resolve each one.
///
/// A centre with exactly three neighbours is a candidate; it becomes an improper
/// only if `PARMCHK.DAT` flags its type as an improper centre. An exact row
/// whose central slot is the centre's type and whose three peripheral slots are,
/// as a multiset, the neighbours' types fixes the atom order — the relation is
/// added as I-J-K-L with the centre at K, which is where
/// [`ImproperPeriodic`](crate::ff::potential::improper::periodic::ImproperPeriodic)
/// expects it — so no peripheral ordering convention has to be invented here.
/// Returns the resolved term of every improper added, by id.
fn add_impropers(
    graph: &mut Atomistic,
    index: &TableIndex,
    estimator: &Parmchk2Estimator,
    type_of: &HashMap<AtomId, ParmType>,
) -> Result<HashMap<ImproperId, (Vec<String>, Bonded)>, GaffError> {
    // Rebuild from scratch: a pre-existing improper would survive with a label
    // this force field never defines.
    let existing: Vec<_> = graph.impropers().map(|(id, _)| id).collect();
    for id in existing {
        graph.remove_improper(id).map_err(malformed)?;
    }

    let centres: Vec<(AtomId, Vec<AtomId>)> = graph
        .atoms()
        .map(|(id, _)| {
            let neighbours: Vec<AtomId> = graph.neighbor_bonds(id).map(|(n, _)| n).collect();
            (id, neighbours)
        })
        .filter(|(id, neighbours)| {
            // The `improper_flag` column of PARMCHK.DAT — a planar centre carries
            // an improper, an sp3 one does not, and that is upstream DATA rather
            // than a hybridisation this crate re-derives.
            neighbours.len() == 3 && estimator.is_improper_centre(index.name_of(type_of[id]))
        })
        .collect();

    let mut terms = HashMap::new();
    for (centre, peripherals) in centres {
        let peripheral_types: Vec<ParmType> = peripherals.iter().map(|n| type_of[n]).collect();

        // An exact row fixes the atom order itself; anything else is an estimate,
        // and its peripherals are ordered by type name so one improper has one name.
        let (order, endpoints, term) = match index.improper(type_of[&centre], &peripheral_types) {
            Some((row, order)) => (
                order,
                index.names(concrete(row.i, row.j, row.k, row.l)).to_vec(),
                Bonded::matched(Params::from_pairs(&improper_params(row))),
            ),
            None => {
                let mut order = [0usize, 1, 2];
                order.sort_by_key(|&slot| index.name_of(peripheral_types[slot]));
                let names: Vec<&'static str> = order
                    .iter()
                    .map(|&slot| index.name_of(peripheral_types[slot]))
                    .collect();
                let centre_name = index.name_of(type_of[&centre]);
                // AMBER slot order: the centre is THIRD, the peripherals a set.
                let term = BondedTerm::Improper([
                    names[0].to_owned(),
                    names[1].to_owned(),
                    centre_name.to_owned(),
                    names[2].to_owned(),
                ]);
                let estimate = estimator
                    .estimate(&term)
                    .expect("an improper always resolves — a row, or parmchk2's default");
                (
                    order,
                    term.endpoints().into_iter().map(str::to_owned).collect(),
                    Bonded::from(estimate),
                )
            }
        };

        let id = graph
            .add_improper(
                peripherals[order[0]],
                peripherals[order[1]],
                centre,
                peripherals[order[2]],
            )
            .map_err(malformed)?;
        terms.insert(id, (endpoints, term));
    }
    Ok(terms)
}

/// The four slots of a row this module has already established is wildcard-free.
///
/// Only the exact-match index feeds this, and it holds no row with a `None` slot,
/// so the panic is unreachable by construction rather than by hope.
fn concrete(
    i: Option<ParmType>,
    j: Option<ParmType>,
    k: Option<ParmType>,
    l: Option<ParmType>,
) -> [ParmType; 4] {
    [i, j, k, l].map(|slot| slot.expect("the exact-match index holds no wildcard row"))
}

fn malformed(e: impl std::fmt::Display) -> GaffError {
    GaffError::Malformed {
        detail: e.to_string(),
    }
}

/// The missing terms of one molecule, deduplicated and kept in first-seen order.
#[derive(Default)]
struct Misses {
    seen: BTreeSet<MissingTerm>,
    terms: Vec<MissingTerm>,
}

impl Misses {
    fn note(&mut self, term: MissingTerm) {
        if self.seen.insert(term.clone()) {
            self.terms.push(term);
        }
    }

    fn into_terms(self) -> Vec<MissingTerm> {
        self.terms
    }
}

// ---------------------------------------------------------------------------
// The exact-match index
// ---------------------------------------------------------------------------

/// Exact-match lookups over one [`ParmTable`], built once per call.
///
/// Only rows with no wildcard slot are indexed. Keys are the table's own
/// [`ParmType`] indices — one byte each, so a bond key is two bytes and a
/// dihedral key four. They are stored in the row's own orientation and looked up
/// in both, which is sound precisely because the generator rejects a table that
/// holds both a term and its reverse.
struct TableIndex {
    table: ParmTable,
    /// Atom-type name -> its index. The only string keying in the whole path.
    names: HashMap<&'static str, ParmType>,
    nonbonded: HashMap<ParmType, &'static ParmNonbondedRow>,
    bonds: HashMap<[ParmType; 2], &'static ParmBondRow>,
    angles: HashMap<[ParmType; 3], &'static ParmAngleRow>,
    /// Every cosine term of a quartet, in file order.
    dihedrals: HashMap<[ParmType; 4], Vec<&'static ParmDihedralRow>>,
    /// Wildcard-free improper rows, in file order — the first match wins.
    impropers: Vec<&'static ParmImproperRow>,
}

impl TableIndex {
    fn new(table: ParmTable) -> Self {
        let mut dihedrals: HashMap<[ParmType; 4], Vec<&'static ParmDihedralRow>> = HashMap::new();
        for row in table.dihedrals {
            // A wildcard row is the fallback matcher's business, not this one's.
            if let (Some(i), Some(j), Some(k), Some(l)) = (row.i, row.j, row.k, row.l) {
                dihedrals.entry([i, j, k, l]).or_default().push(row);
            }
        }

        Self {
            table,
            names: table
                .masses
                .iter()
                .enumerate()
                .map(|(idx, row)| (row.atom_type, ParmType(idx as u8)))
                .collect(),
            nonbonded: table.nonbonded.iter().map(|r| (r.atom_type, r)).collect(),
            bonds: table.bonds.iter().map(|r| ([r.i, r.j], r)).collect(),
            angles: table.angles.iter().map(|r| ([r.i, r.j, r.k], r)).collect(),
            dihedrals,
            impropers: table
                .impropers
                .iter()
                .filter(|r| [r.i, r.j, r.k, r.l].iter().all(Option::is_some))
                .collect(),
        }
    }

    /// Resolve every atom's label of `typed` to the table's [`ParmType`].
    ///
    /// Interning up front is what lets every later lookup key on a one-byte index
    /// rather than on a string. It is also the one check that short-circuits: an
    /// atom type the table never declares makes every term touching it missing,
    /// so reporting those terms as well would bury the actual fault.
    fn intern_atoms(&self, typed: &Atomistic) -> Result<HashMap<AtomId, ParmType>, GaffError> {
        let mut type_of = HashMap::new();
        let mut undeclared: BTreeSet<String> = BTreeSet::new();

        for (position, (id, atom)) in typed.atoms().enumerate() {
            let label = atom
                .get_str(keys::TYPE)
                .ok_or(GaffError::Untyped { atom: position })?;
            match self.names.get(label) {
                Some(&interned) => {
                    type_of.insert(id, interned);
                }
                None => {
                    undeclared.insert(label.to_owned());
                }
            }
        }

        if !undeclared.is_empty() {
            return Err(GaffError::Missing {
                table: self.table.name,
                terms: undeclared.into_iter().map(MissingTerm::Mass).collect(),
            });
        }
        Ok(type_of)
    }

    fn name_of(&self, ty: ParmType) -> &'static str {
        self.table.name_of(ty)
    }

    /// The type names of a term, for a [`MissingTerm`] or a [`BondedTerm`].
    fn names<const N: usize>(&self, types: [ParmType; N]) -> [String; N] {
        types.map(|ty| self.name_of(ty).to_owned())
    }

    fn nonbonded(&self, ty: ParmType) -> Option<&'static ParmNonbondedRow> {
        self.nonbonded.get(&ty).copied()
    }

    fn bond(&self, i: ParmType, j: ParmType) -> Option<&'static ParmBondRow> {
        self.bonds
            .get(&[i, j])
            .or_else(|| self.bonds.get(&[j, i]))
            .copied()
    }

    fn angle(&self, i: ParmType, j: ParmType, k: ParmType) -> Option<&'static ParmAngleRow> {
        self.angles
            .get(&[i, j, k])
            .or_else(|| self.angles.get(&[k, j, i]))
            .copied()
    }

    fn dihedral(&self, [i, j, k, l]: [ParmType; 4]) -> Option<&[&'static ParmDihedralRow]> {
        self.dihedrals
            .get(&[i, j, k, l])
            .or_else(|| self.dihedrals.get(&[l, k, j, i]))
            .map(Vec::as_slice)
    }

    /// The first wildcard-free improper row for a `centre`-typed atom whose three
    /// neighbours carry `peripherals` (in any order).
    ///
    /// Returns the row together with which neighbour fills each of the row's
    /// I / J / L slots, so the caller adds the relation in the row's order rather
    /// than inventing one.
    fn improper(
        &self,
        centre: ParmType,
        peripherals: &[ParmType],
    ) -> Option<(&'static ParmImproperRow, [usize; 3])> {
        self.impropers.iter().find_map(|row| {
            if row.k? != centre {
                return None;
            }
            let wanted = [row.i?, row.j?, row.l?];
            let mut taken = [false; 3];
            let mut order = [0usize; 3];
            for (slot, want) in wanted.iter().enumerate() {
                let found =
                    (0..peripherals.len()).find(|&idx| !taken[idx] && peripherals[idx] == *want)?;
                taken[found] = true;
                order[slot] = found;
            }
            Some((*row, order))
        })
    }
}

#[cfg(test)]
mod tests {
    //! `Typing::typify` with a `GaffTypifier`, on hand-built molecules whose atom
    //! `type`s are stamped by hand. No ATD runs here: typing atoms is ATD's job,
    //! and this typifier starts from labels however they were written. Every
    //! expectation is read off the table by hand (`ff/params/gaff.rs`,
    //! `ff/params/gaff2.rs`), never captured from another program.

    use std::collections::{BTreeMap, BTreeSet};

    use molrs::store::keys;
    use molrs::system::atomistic::Atomistic;
    use molrs::system::molgraph::PropValue;

    use super::{
        GaffParameterSet, GaffTypifier, angle_params, bond_params, gaff_estimator,
        try_candidate_forcefield,
    };
    use crate::ff::forcefield::{ForceField, SpecialBonds};
    use crate::ff::typifier::Typing;

    // -- fixtures ----------------------------------------------------------------

    /// Methane, `C` + 4 `H`, four C-H bonds and nothing else: no angle is
    /// written, so every angle the output holds was enumerated by the typifier.
    /// `h_types` are stamped on the hydrogens in order; `None` leaves that
    /// hydrogen without a `type`.
    fn methane(h_types: [Option<&str>; 4]) -> Atomistic {
        let mut mol = Atomistic::new();
        let c = mol.add_atom_bare("C");
        mol.set_atom(c, keys::TYPE, "c3").expect("stamp c3");
        for h_type in h_types {
            let h = mol.add_atom_bare("H");
            if let Some(label) = h_type {
                mol.set_atom(h, keys::TYPE, label).expect("stamp hc");
            }
            mol.add_bond(c, h).expect("C-H bond");
        }
        mol
    }

    /// A two-atom molecule `a-b`, each atom stamped with its given type.
    fn diatomic((sym_a, type_a): (&str, &str), (sym_b, type_b): (&str, &str)) -> Atomistic {
        let mut mol = Atomistic::new();
        let a = mol.add_atom_bare(sym_a);
        mol.set_atom(a, keys::TYPE, type_a).expect("stamp a");
        let b = mol.add_atom_bare(sym_b);
        mol.set_atom(b, keys::TYPE, type_b).expect("stamp b");
        mol.add_bond(a, b).expect("a-b bond");
        mol
    }

    /// Water `hw-ow-hw`: two O-H bonds, no angle written.
    fn water() -> Atomistic {
        let mut mol = Atomistic::new();
        let o = mol.add_atom_bare("O");
        mol.set_atom(o, keys::TYPE, "ow").expect("stamp ow");
        for _ in 0..2 {
            let h = mol.add_atom_bare("H");
            mol.set_atom(h, keys::TYPE, "hw").expect("stamp hw");
            mol.add_bond(o, h).expect("O-H bond");
        }
        mol
    }

    /// Per `(category, style)` of `ff`, the set of type names it defines;
    /// styles holding no type are left out.
    fn output_names(ff: &ForceField) -> BTreeMap<(String, String), BTreeSet<String>> {
        ff.styles()
            .iter()
            .filter_map(|s| {
                let names: BTreeSet<String> = s
                    .defs()
                    .collect_type_params()
                    .into_iter()
                    .map(|(name, _)| name)
                    .collect();
                (!names.is_empty()).then(|| ((s.category().to_owned(), s.name().to_owned()), names))
            })
            .collect()
    }

    fn names(list: &[&str]) -> BTreeSet<String> {
        list.iter().map(|s| (*s).to_owned()).collect()
    }

    // -- tests -------------------------------------------------------------------

    /// Hand-stamped methane (`c3`, 4 x `hc`) is covered by exact `gaff.dat`
    /// rows: MASS/NONBON `c3` and `hc`, BOND `c3-hc`, ANGLE `hc-c3-hc`. It has
    /// no torsion, and `c3` is sp3 with four neighbours, so no improper. The
    /// output holds exactly those types, each in the style GAFF defines it
    /// under, and nothing else.
    #[test]
    fn gaff_typing_hand_stamped_methane_outputs_exactly_its_types() {
        let mut typing = Typing::new(GaffTypifier::new(GaffParameterSet::Gaff));
        typing
            .typify(&methane([Some("hc"); 4]))
            .expect("hand-stamped methane is fully covered by gaff.dat");

        let key = |c: &str, s: &str| (c.to_owned(), s.to_owned());
        let expected = BTreeMap::from([
            (key("atom", "full"), names(&["c3", "hc"])),
            (key("pair", "lj/cut"), names(&["c3", "hc"])),
            (key("bond", "harmonic"), names(&["c3-hc"])),
            (key("angle", "harmonic"), names(&["hc-c3-hc"])),
        ]);
        assert_eq!(output_names(typing.forcefield()), expected);
    }

    /// The typifier types terms from atom labels it is handed; an atom that
    /// carries none cannot be matched and is an error, not a guess.
    #[test]
    fn gaff_typing_an_atom_without_a_type_is_an_error() {
        let mut typing = Typing::new(GaffTypifier::new(GaffParameterSet::Gaff));
        let result = typing.typify(&methane([Some("hc"), Some("hc"), Some("hc"), None]));
        assert!(
            result.is_err(),
            "an atom with no `type` must fail, got {result:?}"
        );
    }

    /// AMBER excludes 1-2 / 1-3 and scales 1-4 by 1/SCNB = 1/2 (LJ) and
    /// 1/SCEE = 1/1.2 (Coulomb). The library declares them, so the output a
    /// fresh `Typing` seeds from it carries them before any molecule is typed.
    #[test]
    fn gaff_typing_output_declares_the_amber_special_bonds() {
        let typing = Typing::new(GaffTypifier::new(GaffParameterSet::Gaff));
        let amber = SpecialBonds {
            lj: [0.0, 0.0, 0.5],
            coul: [0.0, 0.0, 1.0 / 1.2],
        };
        assert_eq!(typing.forcefield().declared_special_bonds(), Some(&amber));
    }

    /// `gaff2.dat` declares `ow` and `hw` (MASS rows) but gives neither a NONBON
    /// row — the water types take their Lennard-Jones terms from a water model.
    /// Its BOND `ow-hw` and ANGLE `hw-ow-hw` rows exist. Both nonbonded misses
    /// are reported together, not just the first.
    #[test]
    fn gaff_typing_a_table_miss_lists_every_missing_term() {
        let mut typing = Typing::new(GaffTypifier::new(GaffParameterSet::Gaff2));
        let err = typing
            .typify(&water())
            .expect_err("gaff2.dat has no NONBON row for ow or hw");
        for term in ["NONBON ow", "NONBON hw"] {
            assert!(
                err.contains(term),
                "the error must list `{term}`, got: {err}"
            );
        }
    }

    /// `gaff.dat` has no BOND row for `hc-br` (no GAFF-typed molecule bonds H to
    /// Br), so the bond is estimated. Its output definition says so: it carries
    /// `estimated = 1`, a numeric `estimate_penalty`, and the string params
    /// `estimate_method` and `estimate_analog` (empty for a formula).
    #[test]
    fn gaff_typing_an_estimated_term_carries_its_provenance() {
        let mut typing = Typing::new(GaffTypifier::new(GaffParameterSet::Gaff));
        let typed = typing
            .typify(&diatomic(("H", "hc"), ("Br", "br")))
            .expect("an H-Br bond is estimated, not missing");

        let labels: Vec<String> = typed
            .bonds()
            .map(|(_, bond)| match bond.props.get(keys::TYPE) {
                Some(PropValue::Str(label)) => label.clone(),
                other => panic!("the bond carries no string `type`: {other:?}"),
            })
            .collect();
        assert_eq!(labels.len(), 1, "one bond");
        let label = &labels[0];

        let definitions: Vec<_> = typing
            .forcefield()
            .get_styles("bond")
            .into_iter()
            .flat_map(|s| s.defs().collect_type_params())
            .filter(|(name, _)| name == label)
            .collect();
        assert_eq!(
            definitions.len(),
            1,
            "exactly one bond definition named `{label}`"
        );
        let params = &definitions[0].1;

        assert_eq!(params.get("estimated"), Some(1.0), "`{label}` is flagged");
        assert!(
            params.get("estimate_penalty").is_some(),
            "`{label}` carries its penalty"
        );
        assert!(
            params.get_str("estimate_method").is_some(),
            "`{label}` carries its method"
        );
        assert!(
            params.get_str("estimate_analog").is_some(),
            "`{label}` carries its analog"
        );
    }

    // -- the candidate library and its estimator ---------------------------------

    /// AMBER writes `E = K(r−r₀)²`; molrs's kernel writes `E = ½k(r−r₀)²`. The
    /// doubling belongs at this boundary, so a candidate row and an estimate
    /// drawn from it reach every consumer in one convention.
    #[test]
    fn bond_params_double_ambers_force_constant() {
        let row = &GaffParameterSet::Gaff.table().bonds[0];
        let params = bond_params(row);
        assert_eq!(params[0].0, "k", "the canonical key is `k`, not `k0`");
        assert!((params[0].1 - 2.0 * row.force_constant).abs() < 1e-12);
        assert!((params[1].1 - row.length).abs() < 1e-12);
    }

    /// Both shipped `parm` tables (GAFF, GAFF2) transcribe into a candidate
    /// force field without a conflict. `candidate_forcefield` `expect`s this
    /// result and names this test.
    #[test]
    fn candidate_forcefield_defines_without_conflict() {
        for set in [GaffParameterSet::Gaff, GaffParameterSet::Gaff2] {
            assert_eq!(
                try_candidate_forcefield(set.table()).err(),
                None,
                "{}",
                set.name()
            );
        }
    }

    #[test]
    fn angle_params_double_the_constant_and_convert_to_radians() {
        let row = &GaffParameterSet::Gaff.table().angles[0];
        let params = angle_params(row);
        assert_eq!(params[0].0, "k");
        assert!((params[0].1 - 2.0 * row.force_constant).abs() < 1e-12);
        assert!((params[1].1 - row.angle_deg.to_radians()).abs() < 1e-12);
    }

    /// The estimator draws from the same candidate tables, so a formula-derived
    /// constant must be in the same convention as a looked-up row. Half strength
    /// here is invisible in an energy — it just makes a bond too soft.
    #[test]
    fn an_empirical_estimate_shares_the_tables_convention() {
        let estimator = gaff_estimator(GaffParameterSet::Gaff);
        let table = GaffParameterSet::Gaff.table();
        let row = &table.bonds[0];
        let names = [
            table.name_of(row.i).to_owned(),
            table.name_of(row.j).to_owned(),
        ];
        let looked_up = estimator.estimate_bond(&names).expect("a row exists");
        let k = looked_up.get("k").expect("canonical `k`");
        assert!(
            (k - 2.0 * row.force_constant).abs() < 1e-9,
            "a table hit arrives doubled, like every estimate"
        );
    }
}
