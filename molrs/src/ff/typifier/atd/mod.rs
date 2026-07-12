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
    ATOMTYPE_SYBYL, AtdTable,
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
}

impl Typifier for AtdTypifier {
    type Mol = Atomistic;

    fn typify(&self, mol: &Atomistic) -> Result<Atomistic, String> {
        let table = self.set.table();
        let mut out = Perceive::new().find_bond_types(mol);

        let facts = MolFacts::new(&out)?;
        let atom_ids: Vec<AtomId> = out.atoms().map(|(aid, _)| aid).collect();
        for aid in atom_ids {
            let atom_type = rules::assign_atom_type(&table, aid, &facts)
                .ok_or_else(|| format!("missing {} atom type for {aid:?}", table.name))?;
            out.set_atom(aid, keys::TYPE, atom_type)
                .map_err(|e| e.to_string())?;
        }
        Ok(out)
    }
}
