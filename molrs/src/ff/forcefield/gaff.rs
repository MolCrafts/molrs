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
use std::sync::OnceLock;

use molrs::store::keys;
use molrs::{AtomId, Atomistic};

use crate::ff::constants::VACUUM_DIELECTRIC;
use crate::ff::forcefield::{ForceField, Params, SpecialBonds, Style};
use crate::ff::params::{
    GAFF, GAFF2, ParmAngleRow, ParmBondRow, ParmDihedralRow, ParmImproperRow, ParmMassRow,
    ParmNonbondedRow, ParmTable, ParmType,
};
use crate::ff::typifier::estimate::{
    BondedTerm, EmpiricalSet, Estimate, Parmchk2Estimator, Provenance, TypifierParameterContext,
};

/// AMBER's 1-4 Lennard-Jones scale factor (`SCNB = 2.0`).
const AMBER_LJ_14: f64 = 0.5;

/// AMBER's 1-4 Coulomb scale factor (`SCEE = 1.2`).
const AMBER_COUL_14: f64 = 1.0 / 1.2;

/// AMBER's electrostatic conversion factor (kcal·Å·mol⁻¹·e⁻²).
///
/// This is measured, not copied from a constants table: AmberTools25 `sander`
/// single-points on acetate, methylammonium and imidazolium were divided by
/// `Σ scale(i,j)·qᵢqⱼ/rᵢⱼ`, using the topology's 1-2/1-3 exclusions and SCEE=1.2.
/// All three recover this value to the precision printed by `sander`; regenerate
/// the evidence with `scripts/gen_gaff_energy_oracle.py`.
const AMBER_COULOMB: f64 = 332.052_217_29;

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

    /// The empirical bond / angle constants that pair with this table.
    fn empirical(self) -> EmpiricalSet {
        match self {
            Self::Gaff => EmpiricalSet::Gaff,
            Self::Gaff2 => EmpiricalSet::Gaff2,
        }
    }
}

/// The missing-parameter estimator of one `parm` table.
///
/// The estimator is the general one — [`Parmchk2Estimator`], of which molrs has
/// exactly one — and this is how GAFF reaches it: the table is
/// transcribed into a [`ForceField`] of **candidate rows**, which the estimator
/// flattens by style kind. Wildcard rows are included, because they are most of
/// what the estimator is for.
///
/// # The candidate force field is a TABLE, not a potential
///
/// Its harmonic force constants keep `gaff.dat`'s own un-halved convention
/// (`E = K(x − x₀)²`), because that is the convention the estimator's empirical
/// formulas are calibrated in — so an analogy copied off a row and a `K` computed
/// from Badger's rule come back in the *same* convention, and neither is silently a
/// factor of two out. The ½ that molrs's kernels want is applied to both, once,
/// where this module builds the real force field. Handing *this* one to
/// `to_potentials` would be a category error; it never leaves this module.
///
/// Angles and phases, by contrast, are radians here — that is molrs's convention
/// everywhere, including inside the estimator, and the table's degrees are
/// converted at this boundary.
///
/// # Built once per table, not once per molecule
///
/// `gaff.dat` transcribes to some 5,900 candidate rows, and it is the *same* 5,900
/// rows for every molecule ever parameterised — so the estimator is memoised per
/// parameter set. Rebuilding it per call cost more than the whole rest of the
/// force-field build put together (it doubled the test suite's wall time when it
/// was written that way), and it bought nothing: the table is `&'static` data.
pub fn gaff_estimator(set: GaffParameterSet) -> &'static Parmchk2Estimator {
    static GAFF_ESTIMATOR: OnceLock<Parmchk2Estimator> = OnceLock::new();
    static GAFF2_ESTIMATOR: OnceLock<Parmchk2Estimator> = OnceLock::new();

    let cell = match set {
        GaffParameterSet::Gaff => &GAFF_ESTIMATOR,
        GaffParameterSet::Gaff2 => &GAFF2_ESTIMATOR,
    };
    cell.get_or_init(|| {
        let candidates = candidate_forcefield(set.table());
        let context = TypifierParameterContext::new().with_forcefield_elements(&candidates);
        Parmchk2Estimator::with_context(&candidates, context).with_empirical(set.empirical())
    })
}

/// Transcribe a whole `parm` table into a [`ForceField`] of candidate rows.
///
/// Every row, wildcards and all — the estimator's whole job is the rows the exact
/// matcher skips. `X` fills a wildcard slot, which is the spelling
/// [`Candidate`](crate::ff::typifier::estimate::Candidate) reads.
fn candidate_forcefield(table: ParmTable) -> ForceField {
    let mut ff = ForceField::new(table.name);

    let atoms = ff.def_atomstyle("full");
    for row in table.masses {
        atoms.def_atomtype(row.atom_type, &[("mass", row.mass)]);
    }

    let bonds = ff.def_bondstyle("harmonic");
    for row in table.bonds {
        let [i, j] = [row.i, row.j].map(|ty| table.name_of(ty));
        bonds.def_bondtype(i, j, &bond_params(row));
    }

    let angles = ff.def_anglestyle("harmonic");
    for row in table.angles {
        let [i, j, k] = [row.i, row.j, row.k].map(|ty| table.name_of(ty));
        angles.def_angletype(i, j, k, &angle_params(row));
    }

    // The DIHE section writes one row per cosine term; a torsion is the whole group
    // of rows sharing a quartet, so the group is ONE candidate.
    let dihedrals = ff.def_dihedralstyle("periodic");
    let mut groups: Vec<([Option<ParmType>; 4], Vec<&ParmDihedralRow>)> = Vec::new();
    for row in table.dihedrals {
        let slots = [row.i, row.j, row.k, row.l];
        match groups.iter_mut().find(|(seen, _)| *seen == slots) {
            Some((_, rows)) => rows.push(row),
            None => groups.push((slots, vec![row])),
        }
    }
    for (slots, rows) in &groups {
        let [i, j, k, l] = slots.map(|slot| slot_name(&table, &slot));
        let params = dihedral_params(rows);
        dihedrals.def_dihedraltype(i, j, k, l, &borrowed(&params));
    }

    let impropers = ff.def_improperstyle("periodic");
    for row in table.impropers {
        let [i, j, k, l] = [row.i, row.j, row.k, row.l].map(|slot| slot_name(&table, &slot));
        impropers.def_impropertype(i, j, k, l, &improper_params(row));
    }

    ff
}

/// A row slot's atom-type name, or `X` where the file wrote a wildcard.
fn slot_name(table: &ParmTable, slot: &Option<ParmType>) -> &'static str {
    slot.map_or("X", |ty| table.name_of(ty))
}

/// One `BOND` row as candidate params. AMBER's un-halved `K` (see [`gaff_estimator`]).
fn bond_params(row: &ParmBondRow) -> [(&'static str, f64); 2] {
    [("k0", row.force_constant), ("r0", row.length)]
}

/// One `ANGLE` row as candidate params: `K` un-halved, θ₀ in **radians**.
fn angle_params(row: &ParmAngleRow) -> [(&'static str, f64); 2] {
    [
        ("k0", row.force_constant),
        ("theta0", row.angle_deg.to_radians()),
    ]
}

/// The cosine terms of one torsion, in the `k{m}` / `n{m}` / `d{m}` encoding
/// [`DihedralPeriodic`](crate::ff::potential::dihedral::periodic::DihedralPeriodic)
/// scans upward from `m = 1`. `k = PK / IDIVF`; phases in radians.
fn dihedral_params(rows: &[&ParmDihedralRow]) -> Vec<(String, f64)> {
    let mut out = Vec::with_capacity(rows.len() * 3);
    for (m, row) in rows.iter().enumerate() {
        let m = m + 1;
        out.push((format!("k{m}"), row.barrier / f64::from(row.divisor)));
        out.push((format!("n{m}"), f64::from(row.periodicity)));
        out.push((format!("d{m}"), row.phase_deg.to_radians()));
    }
    out
}

/// One `IMPROPER` row as candidate params. An improper's barrier is never divided.
fn improper_params(row: &ParmImproperRow) -> [(&'static str, f64); 3] {
    [
        ("k", row.barrier),
        ("n", f64::from(row.periodicity)),
        ("d", row.phase_deg.to_radians()),
    ]
}

/// `&[(String, f64)]` as the `&[(&str, f64)]` the `def_*type` builders take.
fn borrowed(params: &[(String, f64)]) -> Vec<(&str, f64)> {
    params.iter().map(|(k, v)| (k.as_str(), *v)).collect()
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
    let estimator = gaff_estimator(set);
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
    let mut bond_types: BTreeMap<String, Bonded> = BTreeMap::new();
    let bonds: Vec<_> = out
        .bonds()
        .map(|(id, b)| (id, b.nodes[0], b.nodes[1]))
        .collect();
    for (id, i, j) in bonds {
        let (ti, tj) = (type_of[&i], type_of[&j]);
        let (name, term) = match index.bond(ti, tj) {
            Some(row) => (
                index.dash([row.i, row.j]),
                Bonded::matched(Params::from_pairs(&bond_params(row))),
            ),
            None => {
                let names = index.names([ti, tj]);
                let term = BondedTerm::Bond([names[0].clone(), names[1].clone()]);
                let Some(estimate) = estimator.estimate(&term) else {
                    missed.note(MissingTerm::Bond(names));
                    continue;
                };
                (names.join("-"), Bonded::from(estimate))
            }
        };
        out.set_bond_prop(id, keys::TYPE, name.clone())
            .map_err(malformed)?;
        bond_types.insert(name, term);
    }

    // --- angles + dihedrals, enumerated from the bond graph ---
    out.generate_topology(true, true, true).map_err(malformed)?;

    let mut angle_types: BTreeMap<String, Bonded> = BTreeMap::new();
    let angles: Vec<_> = out
        .angles()
        .map(|(id, a)| (id, a.nodes[0], a.nodes[1], a.nodes[2]))
        .collect();
    for (id, i, j, k) in angles {
        let (ti, tj, tk) = (type_of[&i], type_of[&j], type_of[&k]);
        let (name, term) = match index.angle(ti, tj, tk) {
            Some(row) => (
                index.dash([row.i, row.j, row.k]),
                Bonded::matched(Params::from_pairs(&angle_params(row))),
            ),
            None => {
                let names = index.names([ti, tj, tk]);
                let term =
                    BondedTerm::Angle([names[0].clone(), names[1].clone(), names[2].clone()]);
                let Some(estimate) = estimator.estimate(&term) else {
                    missed.note(MissingTerm::Angle(names));
                    continue;
                };
                (names.join("-"), Bonded::from(estimate))
            }
        };
        out.set_angle_prop(id, keys::TYPE, name.clone())
            .map_err(malformed)?;
        angle_types.insert(name, term);
    }

    // A torsion the exact index misses is not yet missing: the wildcard rows are
    // most of the DIHE section, and parmchk2 estimates only what they too fail to
    // cover.
    let mut dihedral_types: BTreeMap<String, Bonded> = BTreeMap::new();
    let dihedrals: Vec<_> = out
        .dihedrals()
        .map(|(id, d)| (id, d.nodes[0], d.nodes[1], d.nodes[2], d.nodes[3]))
        .collect();
    for (id, i, j, k, l) in dihedrals {
        let quartet = [type_of[&i], type_of[&j], type_of[&k], type_of[&l]];
        let (name, term) = match index.dihedral(quartet) {
            Some(terms) => {
                let first = terms[0];
                (
                    index.dash(concrete(first.i, first.j, first.k, first.l)),
                    Bonded::matched(Params::from_pairs(&borrowed(&dihedral_params(terms)))),
                )
            }
            None => {
                let names = index.names(quartet);
                let term = BondedTerm::Dihedral([
                    names[0].clone(),
                    names[1].clone(),
                    names[2].clone(),
                    names[3].clone(),
                ]);
                let Some(estimate) = estimator.estimate(&term) else {
                    missed.note(MissingTerm::Dihedral(names));
                    continue;
                };
                (canonical_name(&names), Bonded::from(estimate))
            }
        };
        out.set_dihedral_prop(id, keys::TYPE, name.clone())
            .map_err(malformed)?;
        dihedral_types.insert(name, term);
    }

    // --- impropers: table-driven, and never an error (see the module docs) ---
    let improper_types = add_impropers(&mut out, &index, estimator, &type_of)?;

    let terms = missed.into_terms();
    if !terms.is_empty() {
        return Err(GaffError::Missing {
            table: index.table.name,
            terms,
        });
    }

    let ff = build_forcefield(
        set,
        &atom_types,
        &bond_types,
        &angle_types,
        &dihedral_types,
        &improper_types,
    );
    Ok((out, ff))
}

/// One bonded term of the molecule: its parameters, and — when they were estimated
/// rather than looked up — what that cost.
///
/// The parameters are in the candidate table's convention (see [`gaff_estimator`]),
/// whether they came from a row of the table or from the estimator: that is the
/// point of the convention, and it is why one struct now serves all four arities.
/// The provenance rides with them rather than being reconstructed afterwards — an
/// estimate that does not say so is not auditable.
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
}

impl From<Estimate> for Bonded {
    /// A term the estimator answered. A **wildcard row** answer is
    /// [`Estimate::Covered`]: the table covered the term, so it is a parameter and
    /// carries no provenance — which is exactly what parmchk2 does, and what the
    /// oracle tests hold molrs to.
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

/// A bonded term and its reverse are the same term, so the two spellings have to
/// name one force-field type.
fn canonical_name(types: &[String]) -> String {
    let forward = types.join("-");
    let backward: Vec<&str> = types.iter().rev().map(String::as_str).collect();
    let backward = backward.join("-");
    forward.min(backward)
}

/// Write the provenance convention onto an estimated force-field type.
///
/// `def_*type` takes numbers only, so this runs a step behind it — an estimate
/// that does not record that it IS one cannot be audited, and the two provenance
/// keys `ff::typifier::estimate` documents are what a consumer tiers it by.
fn write_provenance(style: &mut Style, name: &str, provenance: &Provenance) {
    style.set_type_param(name, "estimated", 1.0);
    style.set_type_param(name, "estimate_penalty", provenance.penalty);
    style.set_type_str_param(name, "estimate_method", provenance.method.as_str());
    style.set_type_str_param(name, "estimate_analog", &provenance.analog);
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
    estimator: &Parmchk2Estimator,
    type_of: &HashMap<AtomId, ParmType>,
) -> Result<BTreeMap<String, Bonded>, GaffError> {
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
        .filter(|(id, neighbours)| {
            // The `improper_flag` column of PARMCHK.DAT — a planar centre carries
            // an improper, an sp3 one does not, and that is upstream DATA rather
            // than a hybridisation this crate re-derives.
            neighbours.len() == 3 && estimator.is_improper_centre(index.name_of(type_of[id]))
        })
        .collect();

    let mut types: BTreeMap<String, Bonded> = BTreeMap::new();
    for (centre, peripherals) in centres {
        let peripheral_types: Vec<ParmType> = peripherals.iter().map(|n| type_of[n]).collect();

        // An exact row fixes the atom order itself; anything else is an estimate,
        // and its peripherals are ordered by type name so one improper has one name.
        let (order, name, term) = match index.improper(type_of[&centre], &peripheral_types) {
            Some((row, order)) => (
                order,
                index.dash(concrete(row.i, row.j, row.k, row.l)),
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
                    format!("{}-{}-{centre_name}-{}", names[0], names[1], names[2]),
                    Bonded::from(estimate),
                )
            }
        };

        let id = out
            .add_improper(
                peripherals[order[0]],
                peripherals[order[1]],
                centre,
                peripherals[order[2]],
            )
            .map_err(malformed)?;
        out.set_improper_prop(id, keys::TYPE, name.clone())
            .map_err(malformed)?;
        types.insert(name, term);
    }
    Ok(types)
}

/// Assemble the [`ForceField`] from the rows the molecule's terms matched.
///
/// Every unit conversion of this module happens here; the tables hold upstream's
/// own numbers, untouched.
fn build_forcefield(
    set: GaffParameterSet,
    atom_types: &BTreeMap<&'static str, (&ParmMassRow, &ParmNonbondedRow)>,
    bond_types: &BTreeMap<String, Bonded>,
    angle_types: &BTreeMap<String, Bonded>,
    dihedral_types: &BTreeMap<String, Bonded>,
    improper_types: &BTreeMap<String, Bonded>,
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
        // GAFF/AMBER uses the unbuffered Coulomb form.  The constant is force-field
        // data (and differs measurably from CODATA and MMFF), while `delta = 0`
        // explicitly selects the unbuffered branch of the shared `coul/cut` kernel.
        ff.def_pairstyle(
            "coul/cut",
            &[
                ("coulomb", AMBER_COULOMB),
                ("dielectric", VACUUM_DIELECTRIC),
                ("delta", 0.0),
            ],
        );
    }

    if !bond_types.is_empty() {
        let style = ff.def_bondstyle("harmonic");
        for (name, term) in bond_types {
            // AMBER's K carries no ½; molrs's BondHarmonic does. Hence the 2 — and
            // it lands on a row and an estimate alike, because both arrive in the
            // table's own convention (see `gaff_estimator`).
            let [i, j] = ends(name);
            style.def_bondtype(
                i,
                j,
                &[
                    ("k0", 2.0 * param(&term.params, "k0")),
                    ("r0", param(&term.params, "r0")),
                ],
            );
            write_estimate(style, name, term);
        }
    }

    if !angle_types.is_empty() {
        let style = ff.def_anglestyle("harmonic");
        for (name, term) in angle_types {
            let [i, j, k] = ends(name);
            style.def_angletype(
                i,
                j,
                k,
                &[
                    ("k0", 2.0 * param(&term.params, "k0")),
                    // θ₀ is already radians: the candidate table is where the
                    // table's degrees were converted.
                    ("theta0", param(&term.params, "theta0")),
                ],
            );
            write_estimate(style, name, term);
        }
    }

    if !dihedral_types.is_empty() {
        let style = ff.def_dihedralstyle("periodic");
        for (name, term) in dihedral_types {
            // The cosine terms of one torsion (`k{m}`/`n{m}`/`d{m}`, radians) need
            // no conversion: a periodic barrier carries no ½ in either convention.
            let params: Vec<(&str, f64)> = term.params.iter().collect();
            let [i, j, k, l] = ends(name);
            style.def_dihedraltype(i, j, k, l, &params);
            write_estimate(style, name, term);
        }
    }

    if !improper_types.is_empty() {
        let style = ff.def_improperstyle("periodic");
        for (name, term) in improper_types {
            let [i, j, k, l] = ends(name);
            style.def_impropertype(
                i,
                j,
                k,
                l,
                &[
                    ("k", param(&term.params, "k")),
                    ("n", param(&term.params, "n")),
                    ("d", param(&term.params, "d")),
                ],
            );
            write_estimate(style, name, term);
        }
    }

    ff
}

/// One numeric param of a term.
///
/// # Panics
///
/// If the key is absent. Unreachable: every `Params` reaching here was built by
/// this module's own `*_params` helpers or copied by the estimator from one of
/// them, so the keys of an arity are fixed by construction.
fn param(params: &Params, key: &str) -> f64 {
    params
        .get(key)
        .unwrap_or_else(|| panic!("a bonded term of this arity always carries `{key}`"))
}

/// Record the provenance of a term that was estimated, and nothing for one that
/// was looked up.
fn write_estimate(style: &mut Style, name: &str, term: &Bonded) {
    if let Some(provenance) = &term.estimate {
        write_provenance(style, name, provenance);
    }
}

/// The atom types of a dash-form force-field type name (`c3-hc` -> `["c3", "hc"]`).
///
/// The name IS the term's atom types — every name this module builds is a join of
/// them — so splitting it back is a projection, not a parse.
///
/// # Panics
///
/// If the name does not hold exactly `N` types. Unreachable: every caller passes a
/// name this module built from `N` atom types a moment earlier.
fn ends<const N: usize>(name: &str) -> [&str; N] {
    let parts: Vec<&str> = name.split('-').collect();
    parts
        .try_into()
        .unwrap_or_else(|_: Vec<&str>| panic!("`{name}` is not a {N}-atom type name"))
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
