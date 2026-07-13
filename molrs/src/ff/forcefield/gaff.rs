//! GAFF / GAFF2 — a [`ForceField`] populated from the compiled `parm` tables.
//!
//! [`gaff_forcefield`] takes a molecule already typed with GAFF atom types (what
//! [`AtdTypifier`](crate::ff::typifier::AtdTypifier) with
//! [`AtdParameterSet::Gff`](crate::ff::typifier::AtdParameterSet::Gff) produces),
//! enumerates its bonded terms, and looks every one of them up in the
//! [`GAFF`] / [`GAFF2`] static table. Nothing is parsed: the tables are
//! `&'static` Rust data (see [`crate::ff::params`]).
//!
//! # Exact matching only
//!
//! A term is matched **only** against rows whose every slot is a concrete atom
//! type. The 615 wildcard (`X`) rows of each table — which is to say most of the
//! torsion section — are *not* consulted here, and a bond / angle / dihedral one
//! of them would have covered is reported as a [`MissingTerm`] rather than
//! silently parameterised. Wildcard matching, atom-type equivalence and the
//! empirical parmchk2 estimates are a layer above this one; adding them can only
//! ever *shrink* the set of missing terms reported here.
//!
//! Matching a term in either orientation (`c3-c3-oh` covers `oh-c3-c3`) is not a
//! fallback but the undirected nature of a bonded term, and the generator
//! guarantees no table holds both a term and its reverse as separate rows.
//!
//! Impropers are the one exception to "missing is an error": AMBER adds an
//! improper only where its table has a row for one, so a 3-coordinate centre no
//! row covers simply gets no improper term.
//!
//! # Units
//!
//! The table holds what `gaff.dat` says; the kernels want molrs's conventions.
//! This module is the boundary between the two, and the only place the
//! conversions happen:
//!
//! | upstream | molrs |
//! |---|---|
//! | `E = K(r−r₀)²` | `E = ½k₀(r−r₀)²`, so `k0 = 2·K` |
//! | `E = K(θ−θ₀)²` | `E = ½k₀(θ−θ₀)²`, so `k0 = 2·K` |
//! | θ₀ and phases in degrees | radians |
//! | one `PK` shared by `IDIVF` torsions | one `k` per torsion: `k = PK/IDIVF` |
//! | R\*, half the LJ minimum separation | σ = 2·R\*/2^(1/6) |

use std::collections::{BTreeMap, BTreeSet, HashMap};

use molrs::store::keys;
use molrs::{AtomId, Atomistic};

use crate::ff::forcefield::{ForceField, SpecialBonds};
use crate::ff::params::{
    GAFF, GAFF2, ParmAngleRow, ParmBondRow, ParmDihedralRow, ParmImproperRow, ParmMassRow,
    ParmNonbondedRow, ParmTable, ParmType,
};

/// AMBER's 1-4 Lennard-Jones scale factor (`SCNB = 2.0`).
const AMBER_LJ_14: f64 = 0.5;

/// AMBER's 1-4 Coulomb scale factor (`SCEE = 1.2`).
const AMBER_COUL_14: f64 = 1.0 / 1.2;

/// Which AMBER `parm` force field to populate from.
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

    /// The name given to the populated [`ForceField`].
    pub fn name(self) -> &'static str {
        match self {
            Self::Gaff => "gaff",
            Self::Gaff2 => "gaff2",
        }
    }
}

/// A term of the molecule that no **exact** row of the table covers.
///
/// Atom types are the GAFF labels of the term's atoms, in the molecule's own
/// order. Every miss is collected before returning, so one call names every
/// parameter the table is short of — which is what a parmchk2-style estimator
/// has to be handed — rather than only the first.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum MissingTerm {
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
    /// No wildcard-free `DIHE` row for this quartet.
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

/// Why a molecule could not be given a GAFF force field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GaffError {
    /// An atom carries no [`keys::TYPE`] label: the molecule was never typed.
    Untyped {
        /// The atom's 0-based index in graph atom order.
        atom: usize,
    },
    /// The table has no exact row for one or more of the molecule's terms.
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

/// Build the GAFF force field of a GAFF-typed molecule.
///
/// Returns the molecule with its bonded topology enumerated and every term
/// labelled with the force-field type it matched, together with the
/// [`ForceField`] holding those types' parameters. The two are a pair:
/// `mol.to_frame()` carries the labels that `ff.to_potentials` resolves.
///
/// Angles and dihedrals are regenerated from the bond graph; impropers are added
/// — in AMBER's central-atom-third order — wherever the table has an exact row
/// for a 3-coordinate centre.
///
/// # The bond `type` column is the force field's
///
/// The molecule arrives *perceived* — the ATD engine types an atom by counting its
/// `sb`/`db`/`ab`/`DL` bonds — but perception keeps its answer in its own
/// [`BCC_BOND_TYPE`](molrs::perceive::bond_type::BCC_BOND_TYPE) prop, so the bond's
/// [`keys::TYPE`] is free for what this function puts there: the bond's
/// force-field type *name* (`c3-hc`), which is what `to_frame` writes to the
/// `bonds` block's `type` column and what every bonded kernel resolves its
/// parameters by. A perceived molecule is therefore consumable **directly**; no
/// stripping copy stands in between.
///
/// # Errors
///
/// [`GaffError::Untyped`] if an atom carries no [`keys::TYPE`];
/// [`GaffError::Missing`] listing **every** term the table has no exact row for;
/// [`GaffError::Malformed`] if the graph cannot be walked.
///
/// # Example
///
/// Methane, end to end — perceive + type, parameterise, evaluate:
///
/// ```
/// use molrs::Atomistic;
/// use molrs::ff::forcefield::gaff::{GaffParameterSet, gaff_forcefield};
/// use molrs::ff::potential::intramolecular_pairs;
/// use molrs::ff::typifier::Typifier;
/// use molrs::ff::typifier::atd::{AtdParameterSet, AtdTypifier};
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
/// // Typing perceives the BCC bond types on the way through …
/// let typed = AtdTypifier::new(AtdParameterSet::Gff).typify(&mol)?;
/// // … and the perceived molecule builds a force field directly.
/// let (labelled, ff) = gaff_forcefield(GaffParameterSet::Gaff, &typed)?;
///
/// // Every bond's `type` is now the force field's NAME, not a perceived integer.
/// let name = PropValue::Str("c3-hc".to_owned());
/// for (_, bond) in labelled.bonds() {
///     assert_eq!(bond.props.get(keys::TYPE), Some(&name));
/// }
///
/// let mut frame = labelled.to_frame();
/// let pairs = intramolecular_pairs(&frame);
/// frame.insert("pairs", pairs);
/// let potentials = ff.to_potentials(&frame)?;
/// # Ok(())
/// # }
/// ```
pub fn gaff_forcefield(
    set: GaffParameterSet,
    typed: &Atomistic,
) -> Result<(Atomistic, ForceField), GaffError> {
    let index = TableIndex::new(set.table());
    let mut out = typed.clone();
    let type_of = intern_atom_types(&index, &out)?;
    let mut missed = Misses::default();

    // --- atoms: mass + Lennard-Jones ---
    let used: BTreeSet<ParmType> = type_of.values().copied().collect();
    let mut atom_types: BTreeMap<&'static str, (&ParmMassRow, &ParmNonbondedRow)> = BTreeMap::new();
    for ty in used {
        // Interning proved the MASS row exists — it IS the MASS row. The NONBON
        // row is a separate section and can be absent (gaff2's `ow` / `hw`).
        let name = index.name_of(ty);
        let mass = &index.table.masses[usize::from(ty.0)];
        match index.nonbonded(ty) {
            Some(nonbonded) => {
                atom_types.insert(name, (mass, nonbonded));
            }
            None => missed.note(MissingTerm::Nonbonded(name.to_owned())),
        }
    }

    // --- bonds (from the input topology) ---
    let mut bond_types: BTreeMap<String, &'static ParmBondRow> = BTreeMap::new();
    let bonds: Vec<_> = out
        .bonds()
        .map(|(id, b)| (id, b.nodes[0], b.nodes[1]))
        .collect();
    for (id, i, j) in bonds {
        let (ti, tj) = (type_of[&i], type_of[&j]);
        match index.bond(ti, tj) {
            Some(row) => {
                let name = index.dash([row.i, row.j]);
                out.set_bond_prop(id, keys::TYPE, name.clone())
                    .map_err(malformed)?;
                bond_types.insert(name, row);
            }
            None => missed.note(MissingTerm::Bond(index.names([ti, tj]))),
        }
    }

    // --- angles + dihedrals, enumerated from the bond graph ---
    out.generate_topology(true, true, true).map_err(malformed)?;

    let mut angle_types: BTreeMap<String, &'static ParmAngleRow> = BTreeMap::new();
    let angles: Vec<_> = out
        .angles()
        .map(|(id, a)| (id, a.nodes[0], a.nodes[1], a.nodes[2]))
        .collect();
    for (id, i, j, k) in angles {
        let (ti, tj, tk) = (type_of[&i], type_of[&j], type_of[&k]);
        match index.angle(ti, tj, tk) {
            Some(row) => {
                let name = index.dash([row.i, row.j, row.k]);
                out.set_angle_prop(id, keys::TYPE, name.clone())
                    .map_err(malformed)?;
                angle_types.insert(name, row);
            }
            None => missed.note(MissingTerm::Angle(index.names([ti, tj, tk]))),
        }
    }

    let mut dihedral_types: BTreeMap<String, Vec<&'static ParmDihedralRow>> = BTreeMap::new();
    let dihedrals: Vec<_> = out
        .dihedrals()
        .map(|(id, d)| (id, d.nodes[0], d.nodes[1], d.nodes[2], d.nodes[3]))
        .collect();
    for (id, i, j, k, l) in dihedrals {
        let (ti, tj, tk, tl) = (type_of[&i], type_of[&j], type_of[&k], type_of[&l]);
        match index.dihedral(ti, tj, tk, tl) {
            Some(terms) => {
                let first = terms[0];
                let name = index.dash(concrete(first.i, first.j, first.k, first.l));
                out.set_dihedral_prop(id, keys::TYPE, name.clone())
                    .map_err(malformed)?;
                dihedral_types.insert(name, terms.to_vec());
            }
            None => missed.note(MissingTerm::Dihedral(index.names([ti, tj, tk, tl]))),
        }
    }

    // --- impropers: table-driven, and never an error (see the module docs) ---
    let improper_types = add_impropers(&mut out, &index, &type_of)?;

    let terms = missed.into_terms();
    if !terms.is_empty() {
        return Err(GaffError::Missing {
            table: index.table.name,
            terms,
        });
    }

    let ff = build_forcefield(
        set,
        &index.table,
        &atom_types,
        &bond_types,
        &angle_types,
        &dihedral_types,
        &improper_types,
    );
    Ok((out, ff))
}

/// Resolve every atom's label to the table's [`ParmType`].
///
/// Interning up front is what lets every later lookup key on a one-byte index
/// rather than on a string. It is also the one check that short-circuits: an atom
/// type the table never declares makes every term touching it missing, so
/// reporting those terms as well would bury the actual fault.
fn intern_atom_types(
    index: &TableIndex,
    typed: &Atomistic,
) -> Result<HashMap<AtomId, ParmType>, GaffError> {
    let mut type_of = HashMap::new();
    let mut undeclared: BTreeSet<String> = BTreeSet::new();

    for (position, (id, atom)) in typed.atoms().enumerate() {
        let label = atom
            .get_str(keys::TYPE)
            .ok_or(GaffError::Untyped { atom: position })?;
        match index.intern(label) {
            Some(interned) => {
                type_of.insert(id, interned);
            }
            None => {
                undeclared.insert(label.to_owned());
            }
        }
    }

    if !undeclared.is_empty() {
        return Err(GaffError::Missing {
            table: index.table.name,
            terms: undeclared.into_iter().map(MissingTerm::Mass).collect(),
        });
    }
    Ok(type_of)
}

/// Enumerate AMBER impropers and label them.
///
/// A centre with exactly three neighbours is a candidate; it becomes an improper
/// only if the table has an exact row whose central slot is the centre's type and
/// whose three peripheral slots are, as a multiset, the neighbours' types. The
/// **row** then fixes the atom order — the relation is added as I-J-K-L with the
/// centre at K, which is where
/// [`ImproperPeriodic`](crate::ff::potential::improper::periodic::ImproperPeriodic)
/// expects it — so no peripheral ordering convention has to be invented here.
fn add_impropers(
    out: &mut Atomistic,
    index: &TableIndex,
    type_of: &HashMap<AtomId, ParmType>,
) -> Result<BTreeMap<String, &'static ParmImproperRow>, GaffError> {
    // Rebuild from scratch: a pre-existing improper would survive with a label
    // this force field never defines.
    let existing: Vec<_> = out.impropers().map(|(id, _)| id).collect();
    for id in existing {
        out.remove_improper(id).map_err(malformed)?;
    }

    let centres: Vec<(AtomId, Vec<AtomId>)> = out
        .atoms()
        .map(|(id, _)| {
            let neighbours: Vec<AtomId> = out.neighbor_bonds(id).map(|(n, _)| n).collect();
            (id, neighbours)
        })
        .filter(|(_, neighbours)| neighbours.len() == 3)
        .collect();

    let mut types: BTreeMap<String, &'static ParmImproperRow> = BTreeMap::new();
    for (centre, peripherals) in centres {
        let peripheral_types: Vec<ParmType> = peripherals.iter().map(|n| type_of[n]).collect();
        let Some((row, order)) = index.improper(type_of[&centre], &peripheral_types) else {
            continue;
        };
        let id = out
            .add_improper(
                peripherals[order[0]],
                peripherals[order[1]],
                centre,
                peripherals[order[2]],
            )
            .map_err(malformed)?;
        let name = index.dash(concrete(row.i, row.j, row.k, row.l));
        out.set_improper_prop(id, keys::TYPE, name.clone())
            .map_err(malformed)?;
        types.insert(name, row);
    }
    Ok(types)
}

/// Assemble the [`ForceField`] from the rows the molecule's terms matched.
///
/// Every unit conversion of this module happens here; the tables hold upstream's
/// own numbers, untouched.
fn build_forcefield(
    set: GaffParameterSet,
    table: &ParmTable,
    atom_types: &BTreeMap<&'static str, (&ParmMassRow, &ParmNonbondedRow)>,
    bond_types: &BTreeMap<String, &'static ParmBondRow>,
    angle_types: &BTreeMap<String, &'static ParmAngleRow>,
    dihedral_types: &BTreeMap<String, Vec<&'static ParmDihedralRow>>,
    improper_types: &BTreeMap<String, &'static ParmImproperRow>,
) -> ForceField {
    let mut ff = ForceField::new(set.name());
    // AMBER excludes 1-2 / 1-3 outright and scales 1-4 by SCNB = 2 (LJ) and
    // SCEE = 1.2 (Coulomb).
    ff.set_special_bonds(SpecialBonds {
        lj: [0.0, 0.0, AMBER_LJ_14],
        coul: [0.0, 0.0, AMBER_COUL_14],
    });

    if !atom_types.is_empty() {
        let atom = ff.def_atomstyle("full");
        for (name, (mass, _)) in atom_types {
            atom.def_atomtype(name, &[("mass", mass.mass)]);
        }
        let lj = ff.def_pairstyle("lj/cut", &[]);
        for (name, (_, nonbonded)) in atom_types {
            lj.def_pairtype(
                name,
                None,
                &[
                    ("epsilon", nonbonded.epsilon),
                    ("sigma", sigma_of(nonbonded.r_min_half)),
                ],
            );
        }
    }

    if !bond_types.is_empty() {
        let style = ff.def_bondstyle("harmonic");
        for row in bond_types.values() {
            // AMBER's K carries no ½; molrs's BondHarmonic does. Hence the 2.
            style.def_bondtype(
                table.name_of(row.i),
                table.name_of(row.j),
                &[("k0", 2.0 * row.force_constant), ("r0", row.length)],
            );
        }
    }

    if !angle_types.is_empty() {
        let style = ff.def_anglestyle("harmonic");
        for row in angle_types.values() {
            style.def_angletype(
                table.name_of(row.i),
                table.name_of(row.j),
                table.name_of(row.k),
                &[
                    ("k0", 2.0 * row.force_constant),
                    ("theta0", row.angle_deg.to_radians()),
                ],
            );
        }
    }

    if !dihedral_types.is_empty() {
        let style = ff.def_dihedralstyle("periodic");
        for terms in dihedral_types.values() {
            // The cosine terms of one torsion, in the `k{m}`/`n{m}`/`d{m}`
            // encoding `DihedralPeriodic` scans upward from m = 1.
            let mut keys: Vec<(String, f64)> = Vec::with_capacity(terms.len() * 3);
            for (m, row) in terms.iter().enumerate() {
                let m = m + 1;
                keys.push((format!("k{m}"), row.barrier / f64::from(row.divisor)));
                keys.push((format!("n{m}"), f64::from(row.periodicity)));
                keys.push((format!("d{m}"), row.phase_deg.to_radians()));
            }
            let params: Vec<(&str, f64)> = keys.iter().map(|(k, v)| (k.as_str(), *v)).collect();
            let slots = concrete(terms[0].i, terms[0].j, terms[0].k, terms[0].l);
            let [i, j, k, l] = slots.map(|ty| table.name_of(ty));
            style.def_dihedraltype(i, j, k, l, &params);
        }
    }

    if !improper_types.is_empty() {
        let style = ff.def_improperstyle("periodic");
        for row in improper_types.values() {
            let [i, j, k, l] = concrete(row.i, row.j, row.k, row.l).map(|ty| table.name_of(ty));
            style.def_impropertype(
                i,
                j,
                k,
                l,
                &[
                    ("k", row.barrier),
                    ("n", f64::from(row.periodicity)),
                    ("d", row.phase_deg.to_radians()),
                ],
            );
        }
    }

    ff
}

/// σ from R\*: `σ = 2·R* / 2^(1/6)`.
///
/// R\* is **half** the separation at the Lennard-Jones minimum, so the minimum
/// itself is `2·R*` and σ — where the potential crosses zero — is that over the
/// sixth root of two. Reading R\* as σ would inflate every radius by 78%.
fn sigma_of(r_min_half: f64) -> f64 {
    2.0 * r_min_half / 2f64.powf(1.0 / 6.0)
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

    /// The table's index for a type name, or `None` if it declares no such type.
    fn intern(&self, label: &str) -> Option<ParmType> {
        self.names.get(label).copied()
    }

    fn name_of(&self, ty: ParmType) -> &'static str {
        self.table.name_of(ty)
    }

    /// The dash-form force-field type name of a row, e.g. `c3-c3-oh`.
    fn dash<const N: usize>(&self, types: [ParmType; N]) -> String {
        types.map(|ty| self.name_of(ty)).join("-")
    }

    /// The type names of a term, for a [`MissingTerm`].
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

    fn dihedral(
        &self,
        i: ParmType,
        j: ParmType,
        k: ParmType,
        l: ParmType,
    ) -> Option<&[&'static ParmDihedralRow]> {
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
