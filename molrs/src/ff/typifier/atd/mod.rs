//! Antechamber's `ATD` / `WILDATOM` atom-type engine — one engine, N tables.
//!
//! All seven `ATOMTYPE_*.DEF` files share one rule language, so they share one
//! interpreter: [`AtdTypifier`] is that interpreter, and [`AtdParameterSet`]
//! chooses the table it walks. The tables are `&'static` Rust data generated
//! from the upstream `.DEF` files (see [`crate::ff::params`]), so a typifier
//! carries no state beyond which table it names, and `typify()` parses nothing.
//!
//! The engine holds **no** per-table knowledge. That is a testable claim rather
//! than a stylistic one: the three tables disagree exactly where typing is hard
//! (imidazole's pyridine-type N is `24` under BCC, `28` under ABCG2 and `n2`
//! under GAS), so a table-specific special case that satisfied one column would
//! break another.
//!
//! ```no_run
//! use molrs::Atomistic;
//! use molrs::ff::typifier::Typifier;
//! use molrs::ff::typifier::atd::{AtdParameterSet, AtdTypifier};
//!
//! # fn main() -> Result<(), String> {
//! let mol = Atomistic::new();
//! let typed = AtdTypifier::new(AtdParameterSet::Bcc).typify(&mol)?;
//! # Ok(())
//! # }
//! ```
//!
//! # Layering
//!
//! Bond types are always **perceived** here, never read off the input: the rules
//! count `sb` / `db` / `ab` / `DL` bonds, which need the delocalized (9) and
//! aromatic (7/8) types that a bond *order* cannot express.

mod conjugate;
mod facts;
mod pattern;
mod rules;

pub(crate) use facts::antechamber_bond_type;

use molrs::perceive::Perceive;
use molrs::store::keys;
use molrs::{AtomId, Atomistic};

use self::facts::MolFacts;
use super::Typifier;
use crate::ff::params::{
    ATOMTYPE_ABCG2, ATOMTYPE_AMBER, ATOMTYPE_BCC, ATOMTYPE_GAS, ATOMTYPE_GFF, ATOMTYPE_GFF2,
    ATOMTYPE_SYBYL, AtdRule, AtdTable,
};

/// Which `ATOMTYPE_*.DEF` table an [`AtdTypifier`] walks.
///
/// This is the **atom-type** axis, and it is wider than the BCC-correction axis
/// it used to be conflated with: `ATOMTYPE_GAS.DEF` exists but there is no
/// `BCCPARM_GAS.DAT`, so GAS is a set of atom types with no correction family.
/// Only [`BccParameterSet`](super::am1bcc::BccParameterSet) — `Bcc` and `Abcg2`
/// — names both.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AtdParameterSet {
    /// `ATOMTYPE_BCC.DEF` — AM1-BCC atom types (`antechamber -at bcc`).
    Bcc,
    /// `ATOMTYPE_ABCG2.DEF` — ABCG2 atom types (`antechamber -at abcg2`).
    Abcg2,
    /// `ATOMTYPE_GAS.DEF` — Gasteiger atom types (`antechamber -at gas`).
    Gas,
    /// `ATOMTYPE_GFF.DEF` — GAFF atom types (`antechamber -at gaff`).
    Gff,
    /// `ATOMTYPE_GFF2.DEF` — GAFF2 atom types (`antechamber -at gaff2`).
    Gff2,
    /// `ATOMTYPE_AMBER.DEF` — AMBER atom types (`antechamber -at amber`).
    Amber,
    /// `ATOMTYPE_SYBYL.DEF` — SYBYL atom types (`antechamber -at sybyl`).
    Sybyl,
}

impl AtdParameterSet {
    /// The compile-time table this set names.
    pub fn table(self) -> AtdTable {
        match self {
            Self::Bcc => ATOMTYPE_BCC,
            Self::Abcg2 => ATOMTYPE_ABCG2,
            Self::Gas => ATOMTYPE_GAS,
            Self::Gff => ATOMTYPE_GFF,
            Self::Gff2 => ATOMTYPE_GFF2,
            Self::Amber => ATOMTYPE_AMBER,
            Self::Sybyl => ATOMTYPE_SYBYL,
        }
    }
}

/// The type every `.DEF` file's terminal catch-all row assigns: "no rule matched".
///
/// Antechamber's dummy type. Six of the seven tables end with an unconditional
/// `ATD DU &` row, so an atom no rule covers is not *unlabelled* — it is labelled
/// `DU`, and the label carries no chemistry. In the AMBER column that is genuinely
/// antechamber's answer (nitromethane's nitro oxygens, DMSO's sulfur), so the engine
/// must keep emitting it; but a consumer that needs *parameters* for the type — the
/// BCC correction tables have no `DU` row — has to read it as the refusal it is.
pub(crate) const DUMMY_TYPE: &str = "DU";

/// Why the engine could not type a molecule.
///
/// The typed twin of the `String` that [`Typifier::typify`] returns. A charge model
/// has to tell "no rule covers this atom" (a permanent property of the table — boron
/// and bare sulfur are the real cases) apart from "this graph is malformed", because
/// the C++ and Python bridges have to report them differently.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum AtdError {
    /// No rule of `table` matched the atom.
    MissingAtomType {
        /// The `.DEF` file that was walked.
        table: &'static str,
        /// The atom's 0-based index in graph atom order.
        atom: usize,
        /// The atom's element symbol.
        element: String,
    },
    /// The molecule's facts could not be derived — a defect in the input graph.
    Malformed {
        /// What the facts layer said.
        detail: String,
    },
}

impl std::fmt::Display for AtdError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingAtomType {
                table,
                atom,
                element,
            } => write!(f, "missing {table} atom type for atom {atom} ({element})"),
            Self::Malformed { detail } => write!(f, "{detail}"),
        }
    }
}

/// The ATD rule engine, bound to one atom-type table.
///
/// [`typify`](Typifier::typify) perceives antechamber bond types, derives the
/// facts each rule can ask about, and labels every atom with the first rule of
/// the table that matches it. An atom no rule matches is an **error**, not an
/// untyped or defaulted atom.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AtdTypifier {
    set: AtdParameterSet,
}

impl AtdTypifier {
    /// Bind the engine to the table `set` names.
    pub fn new(set: AtdParameterSet) -> Self {
        Self { set }
    }

    /// The parameter set this typifier walks.
    pub fn parameter_set(&self) -> AtdParameterSet {
        self.set
    }

    /// The types this table assigns, in graph atom order — **computed, not written**.
    ///
    /// The half of [`typify`](Typifier::typify) that a charge model wants: the atom
    /// types come back as a `Vec`, so the model can look its corrections up without
    /// ever putting a BCC code into the caller's [`keys::TYPE`] column (where their
    /// GAFF / OPLS force-field types live).
    ///
    /// # Arguments
    ///
    /// * `perceived` — a molecule whose bonds already carry perceived antechamber
    ///   bond types, i.e. the output of
    ///   [`Perceive::find_bond_types`](molrs::perceive::Perceive::find_bond_types).
    ///   The rules count `sb` / `db` / `ab` / `DL` bonds, so they cannot run on bond
    ///   *orders*.
    ///
    /// # Errors
    ///
    /// [`AtdError::MissingAtomType`] when no rule matches an atom;
    /// [`AtdError::Malformed`] when the graph's facts cannot be derived.
    pub(crate) fn types_of(&self, perceived: &Atomistic) -> Result<Vec<&'static str>, AtdError> {
        let table = self.set.table();
        let facts = MolFacts::new(perceived).map_err(|detail| AtdError::Malformed { detail })?;
        let atom_ids: Vec<AtomId> = perceived.atoms().map(|(aid, _)| aid).collect();

        // Pass 1 — the table: the first rule that matches each atom.
        let assigned: Vec<&'static AtdRule> = atom_ids
            .iter()
            .enumerate()
            .map(|(i, aid)| {
                rules::assign_rule(&table, *aid, &facts).ok_or_else(|| AtdError::MissingAtomType {
                    table: table.name,
                    atom: i,
                    element: perceived
                        .get_atom(*aid)
                        .ok()
                        .and_then(|atom| atom.get_str(keys::ELEMENT).map(str::to_owned))
                        .unwrap_or_default(),
                })
            })
            .collect::<Result<_, _>>()?;

        // Pass 2 — the conjugated 2-colouring, which is a fact about the whole
        // molecule and so cannot be folded into the per-atom loop above: half of
        // each conjugated system is renamed to the matched rule's `alternate`.
        Ok(conjugate::resolve_types(&atom_ids, &assigned, &facts))
    }
}

impl Typifier for AtdTypifier {
    type Mol = Atomistic;

    fn typify(&self, mol: &Atomistic) -> Result<Atomistic, String> {
        let mut out = Perceive::new().find_bond_types(mol);
        let types = self.types_of(&out).map_err(|e| e.to_string())?;

        let atom_ids: Vec<AtomId> = out.atoms().map(|(aid, _)| aid).collect();
        for (aid, atom_type) in atom_ids.iter().zip(types) {
            out.set_atom(*aid, keys::TYPE, atom_type)
                .map_err(|e| e.to_string())?;
        }
        Ok(out)
    }
}
