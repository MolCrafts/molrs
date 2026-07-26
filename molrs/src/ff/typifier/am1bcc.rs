//! The AM1-BCC *tables*: the atom-type table and the bond charge corrections.
//!
//! This module owns what the BCC parameter sets are — [`BccParameterSet`] names a
//! pair (`ATOMTYPE_*.DEF` + `BCCPARM*.DAT`), [`BCCAtomChargeTypifier`] walks the first and
//! [`BCCCorrectionTable`] / [`BCCCorrector`] apply the second. The charge *model*
//! that composes them — the push API `BccModel::correct(&mol, &am1)` — lives one
//! module over in [`ff::charge`](crate::ff::charge), because a charge model is not a
//! typifier: it returns charges rather than a relabelled graph, and it must never
//! write its internal BCC codes into the caller's [`keys::TYPE`] column, where their
//! GAFF / OPLS force-field types live.
//!
//! Atom typing itself lives one module over the other way, in [`atd`](super::atd):
//! the rules that assign `11` / `91` / `24` are antechamber's ATD language, shared
//! by every `ATOMTYPE_*.DEF` table, and this module drives that one engine with the
//! BCC (or ABCG2) table rather than owning a second copy of it. What is BCC-specific
//! is what stays here: the **correction** table, which no other parameter set has.
//!
//! The implementation is a native port surface: external programs are reference
//! material only, not runtime dependencies. A missing BCC atom type, bond type or
//! correction row is an error, never a defaulted value.

use molrs::store::keys;
use molrs::{AtomId, Atomistic, BondId};
use std::collections::HashMap;
use std::fmt;

use super::Typifier;
use super::atd::{AtdParameterSet, AtdTypifier, antechamber_bond_type};
use crate::ff::params::{BccAlias, BccCorrectionRow};

/// AM1-BCC correction-family selector.
///
/// The two variants are exactly the two `BCCPARM*.DAT` files that exist. This is
/// a **narrower** axis than [`AtdParameterSet`]: every correction family names an
/// atom-type table (via [`BccParameterSet::atd_set`]), but not every atom-type
/// table has a correction family — `ATOMTYPE_GAS.DEF` has no `BCCPARM_GAS.DAT`,
/// so GAS is reachable through [`AtdTypifier`] and cannot be a variant here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BccParameterSet {
    /// Original AM1-BCC corrections (`BCCPARM.DAT`, `-c bcc`).
    Bcc,
    /// ABCG2 corrections (`BCCPARM_ABCG2.DAT`, `-c abcg2`).
    Abcg2,
}

impl BccParameterSet {
    /// The set's bond charge corrections, as compile-time table data.
    fn corrections(self) -> &'static [BccCorrectionRow] {
        match self {
            Self::Bcc => crate::ff::params::BCC_CORRECTIONS,
            Self::Abcg2 => crate::ff::params::ABCG2_CORRECTIONS,
        }
    }

    /// The set's `CORR` alias rows.
    fn aliases(self) -> &'static [BccAlias] {
        match self {
            Self::Bcc => crate::ff::params::BCC_ALIASES,
            Self::Abcg2 => crate::ff::params::ABCG2_ALIASES,
        }
    }

    /// The atom-type table this correction family is defined against.
    ///
    /// A correction row is keyed on atom types, so a correction family is only
    /// meaningful next to the table those types come from: `BCCPARM.DAT` rows
    /// speak `ATOMTYPE_BCC.DEF`, `BCCPARM_ABCG2.DAT` rows speak
    /// `ATOMTYPE_ABCG2.DEF`. Pairing them any other way silently looks up the
    /// wrong rows.
    ///
    /// # Returns
    ///
    /// The [`AtdParameterSet`] whose atom types this family's rows are written in.
    pub fn atd_set(self) -> AtdParameterSet {
        match self {
            Self::Bcc => AtdParameterSet::Bcc,
            Self::Abcg2 => AtdParameterSet::Abcg2,
        }
    }
}

/// Graph-based BCC atom typifier: the [`AtdTypifier`] bound to a BCC table.
///
/// This is a named shorthand, not a second engine — `BCCAtomChargeTypifier::bcc()` and
/// `AtdTypifier::new(AtdParameterSet::Bcc)` label every atom identically because
/// the former *is* the latter. It exists because the AM1-BCC pipeline needs the
/// atom-type table and the correction family chosen together, and
/// [`BccParameterSet`] is the type that keeps that pair honest.
///
/// It **writes** its labels into the graph's [`keys::TYPE`] column, so it is for
/// callers who want a BCC-typed molecule and nothing else. Charges do **not** go
/// through it: `BccModel` keeps its BCC types to itself precisely so that a molecule
/// can carry GAFF types and BCC charges at the same time, which is what the standard
/// AM1-BCC workflow is.
#[derive(Debug, Clone)]
pub struct BCCAtomChargeTypifier {
    model: BccParameterSet,
}

impl Default for BCCAtomChargeTypifier {
    fn default() -> Self {
        Self::bcc()
    }
}

impl BCCAtomChargeTypifier {
    /// A typifier for the atom-type table `model` names.
    ///
    /// # Arguments
    ///
    /// * `model` — the correction family whose atom-type table to walk.
    ///
    /// # Returns
    ///
    /// The typifier bound to that table.
    pub fn parameter_set(model: BccParameterSet) -> Self {
        Self { model }
    }

    /// `ATOMTYPE_BCC.DEF` — the original AM1-BCC atom types.
    ///
    /// # Returns
    ///
    /// The typifier bound to `ATOMTYPE_BCC.DEF`.
    pub fn bcc() -> Self {
        Self::parameter_set(BccParameterSet::Bcc)
    }

    /// `ATOMTYPE_ABCG2.DEF` — the ABCG2 atom types.
    ///
    /// # Returns
    ///
    /// The typifier bound to `ATOMTYPE_ABCG2.DEF`.
    pub fn abcg2() -> Self {
        Self::parameter_set(BccParameterSet::Abcg2)
    }
}

impl Typifier for BCCAtomChargeTypifier {
    type Mol = Atomistic;

    /// Perceive BCC bond types, then label every atom from the set's
    /// `ATOMTYPE_*.DEF` rules.
    ///
    /// The bond types are always **perceived** (via
    /// [`Perceive::find_bond_types`](molrs::perceive::Perceive::find_bond_types)),
    /// never read off the input: the atom-type rules count `sb`/`db`/`ab`/`DL`
    /// bonds, so they need the delocalized (9) and aromatic (7/8) types that a bond
    /// *order* cannot express — and a supplied
    /// [`BCC_BOND_TYPE`](molrs::perceive::bond_type::BCC_BOND_TYPE) may be the
    /// unresolved aromatic precursor (10), which must be resolved, not trusted. To
    /// apply corrections with types of your own, drive [`BCCCorrector`] directly.
    ///
    /// # Arguments
    ///
    /// * `mol` — the molecule to type; left untouched.
    ///
    /// # Returns
    ///
    /// A clone of `mol` whose atoms carry BCC codes in [`keys::TYPE`] and whose
    /// bonds carry perceived antechamber bond types in
    /// [`BCC_BOND_TYPE`](molrs::perceive::bond_type::BCC_BOND_TYPE) — the bond's own
    /// [`keys::TYPE`], the caller's force-field label, is left untouched.
    ///
    /// # Errors
    ///
    /// A message naming the atom no rule of the table matched.
    fn typify(&self, mol: &Self::Mol) -> Result<Self::Mol, String> {
        AtdTypifier::new(self.model.atd_set()).typify(mol)
    }
}

/// Oriented BCC correction table.
///
/// A row `(left, right, bond_type, delta)` means a bond typed
/// `left|right|bond_type` adds `+delta` to the left atom and `-delta` to the
/// right atom. A reversed bond applies the same magnitude with reversed sign.
#[derive(Debug, Clone)]
pub struct BCCCorrectionTable {
    corrections: HashMap<(String, String, i32), f64>,
    aliases: HashMap<String, String>,
}

impl BCCCorrectionTable {
    /// A table built from explicit rows.
    ///
    /// This is the only way to build a table from scratch, and it *names its
    /// content*: what goes in comes out, and no parameter set leaks in behind the
    /// caller's back. There is deliberately no empty-by-default form — an empty
    /// correction table is not a default, it is a table that fails on every bond
    /// (`missing BCC correction for bond ..`) long after it was constructed. To get
    /// the corrections of a published set, name it: [`Self::bcc`], [`Self::abcg2`],
    /// or [`Self::parameter_set`].
    ///
    /// Aliases (`CORR` rows) are not rows and cannot be expressed here; add them
    /// with [`Self::alias`].
    ///
    /// # Arguments
    ///
    /// * `rows` — the correction rows to hold.
    ///
    /// # Returns
    ///
    /// A table holding exactly `rows`, with no aliases.
    pub fn from_rows(rows: &[BccCorrectionRow]) -> Self {
        let corrections = rows
            .iter()
            .map(|row| {
                (
                    (row.left.to_owned(), row.right.to_owned(), row.bond_type),
                    row.delta,
                )
            })
            .collect();
        Self {
            corrections,
            aliases: HashMap::new(),
        }
    }

    /// Build the table for a parameter set from its compile-time rows.
    ///
    /// # Arguments
    ///
    /// * `model` — the correction family to load.
    ///
    /// # Returns
    ///
    /// The family's rows and `CORR` aliases.
    ///
    /// # Errors
    ///
    /// Never, in practice: the tables are typed Rust data. The `Result` keeps the
    /// call shape of the former text-parsing constructor.
    pub fn parameter_set(model: BccParameterSet) -> Result<Self, String> {
        let mut table = Self::from_rows(model.corrections());
        for alias in model.aliases() {
            table = table.alias(alias.atom_type, alias.reference);
        }
        Ok(table)
    }

    /// `BCCPARM.DAT` — the original AM1-BCC corrections.
    ///
    /// # Returns
    ///
    /// The BCC family's table.
    ///
    /// # Errors
    ///
    /// Never, in practice; see [`Self::parameter_set`].
    pub fn bcc() -> Result<Self, String> {
        Self::parameter_set(BccParameterSet::Bcc)
    }

    /// `BCCPARM_ABCG2.DAT` — the ABCG2 corrections.
    ///
    /// # Returns
    ///
    /// The ABCG2 family's table.
    ///
    /// # Errors
    ///
    /// Never, in practice; see [`Self::parameter_set`].
    pub fn abcg2() -> Result<Self, String> {
        Self::parameter_set(BccParameterSet::Abcg2)
    }

    /// Add a single-bond row.
    ///
    /// # Arguments
    ///
    /// * `left` / `right` — the two BCC atom types.
    /// * `delta` — the increment added to `left` and subtracted from `right`.
    ///
    /// # Returns
    ///
    /// The table, with the row inserted.
    pub fn insert(self, left: impl Into<String>, right: impl Into<String>, delta: f64) -> Self {
        self.insert_bond_type(left, right, 1, delta)
    }

    /// Add a row for one bond type.
    ///
    /// # Arguments
    ///
    /// * `left` / `right` — the two BCC atom types.
    /// * `bond_type` — the antechamber bond type the row applies to.
    /// * `delta` — the increment added to `left` and subtracted from `right`.
    ///
    /// # Returns
    ///
    /// The table, with the row inserted.
    pub fn insert_bond_type(
        mut self,
        left: impl Into<String>,
        right: impl Into<String>,
        bond_type: i32,
        delta: f64,
    ) -> Self {
        self.corrections
            .insert((left.into(), right.into(), bond_type), delta);
        self
    }

    /// Add a `CORR` alias: `atom_type` corrects as `reference` does.
    ///
    /// # Arguments
    ///
    /// * `atom_type` — the type with no rows of its own.
    /// * `reference` — the type whose rows it borrows.
    ///
    /// # Returns
    ///
    /// The table, with the alias registered.
    pub fn alias(mut self, atom_type: impl Into<String>, reference: impl Into<String>) -> Self {
        self.aliases.insert(atom_type.into(), reference.into());
        self
    }

    /// The number of correction rows.
    ///
    /// # Returns
    ///
    /// The row count (aliases are not rows; see [`Self::alias_len`]).
    pub fn len(&self) -> usize {
        self.corrections.len()
    }

    /// Has the table no rows at all?
    ///
    /// # Returns
    ///
    /// `true` when it would fail on every bond.
    pub fn is_empty(&self) -> bool {
        self.corrections.is_empty()
    }

    /// The number of `CORR` aliases.
    ///
    /// # Returns
    ///
    /// The alias count.
    pub fn alias_len(&self) -> usize {
        self.aliases.len()
    }

    /// The increment for a typed bond, following `CORR` aliases.
    ///
    /// # Arguments
    ///
    /// * `left` / `right` — the endpoints' BCC atom types.
    /// * `bond_type` — the perceived antechamber bond type.
    ///
    /// # Returns
    ///
    /// The increment to add to `left` and subtract from `right`, or `None` when no
    /// row (and no aliased row) covers the bond.
    pub fn correction(&self, left: &str, right: &str, bond_type: i32) -> Option<f64> {
        if let Some(v) = self.direct_correction(left, right, bond_type) {
            return Some(v);
        }

        let left_alias = self.aliases.get(left).map(String::as_str);
        let right_alias = self.aliases.get(right).map(String::as_str);

        if let Some(alias) = left_alias
            && let Some(v) = self.direct_correction(alias, right, bond_type)
        {
            return Some(v);
        }
        if let Some(alias) = right_alias
            && let Some(v) = self.direct_correction(left, alias, bond_type)
        {
            return Some(v);
        }
        if let (Some(left_alias), Some(right_alias)) = (left_alias, right_alias)
            && let Some(v) = self.direct_correction(left_alias, right_alias, bond_type)
        {
            return Some(v);
        }
        None
    }

    fn direct_correction(&self, left: &str, right: &str, bond_type: i32) -> Option<f64> {
        self.corrections
            .get(&(left.to_owned(), right.to_owned(), bond_type))
            .copied()
            .or_else(|| {
                self.corrections
                    .get(&(right.to_owned(), left.to_owned(), bond_type))
                    .map(|v| -*v)
            })
    }
}

/// Why a correction pass could not be applied.
///
/// Kept typed rather than a message so that the charge model can hand the C++ and
/// Python bridges a `ChargeError` they can discriminate on: a bond with no row is a
/// permanent property of the parameter set, a malformed graph is the caller's.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum BccIncrementError {
    /// No row (and no aliased row) covers this bond.
    MissingRow {
        /// The bond, for the message.
        bond: BondId,
        /// BCC atom type of one endpoint.
        left: String,
        /// BCC atom type of the other endpoint.
        right: String,
        /// The perceived antechamber bond type.
        bond_type: i32,
    },
    /// The graph could not be read.
    Malformed {
        /// What the graph layer said.
        detail: String,
    },
}

impl fmt::Display for BccIncrementError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingRow {
                bond,
                left,
                right,
                bond_type,
            } => write!(
                f,
                "missing BCC correction for bond {bond:?}: {left}|{right}|{bond_type}"
            ),
            Self::Malformed { detail } => write!(f, "{detail}"),
        }
    }
}

/// The per-atom BCC increment of every atom, in graph atom order.
///
/// The one place the corrections are applied. Every row is added to one endpoint and
/// subtracted from the other, so the increments sum to zero and the pass conserves
/// the molecule's total charge exactly.
///
/// The atom types are an **argument**, not something read off `mol`: the charge model
/// perceives its BCC types into a `Vec` and never writes them into the caller's
/// graph, while [`BCCCorrector`] reads them from a graph the caller typed on purpose.
/// One loop, two ways to say what the types are.
///
/// # Arguments
///
/// * `table` — the correction rows to apply.
/// * `mol` — the molecule, whose bonds carry perceived antechamber bond types.
/// * `types` — the BCC atom type of every atom, in graph atom order.
///
/// # Returns
///
/// The increment for every atom, in graph atom order.
///
/// # Errors
///
/// [`BccIncrementError::MissingRow`] for a bond the table does not cover;
/// [`BccIncrementError::Malformed`] for a bond with no usable `type`.
pub(crate) fn bcc_increments(
    table: &BCCCorrectionTable,
    mol: &Atomistic,
    types: &[&str],
) -> Result<Vec<f64>, BccIncrementError> {
    let atom_ids: Vec<AtomId> = mol.atoms().map(|(id, _)| id).collect();
    let index: HashMap<AtomId, usize> = atom_ids
        .iter()
        .enumerate()
        .map(|(i, aid)| (*aid, i))
        .collect();

    let mut delta = vec![0.0; atom_ids.len()];
    for (bid, bond) in mol.bonds() {
        let (Some(&i), Some(&j)) = (index.get(&bond.nodes[0]), index.get(&bond.nodes[1])) else {
            return Err(BccIncrementError::Malformed {
                detail: format!("bond {bid:?} has an endpoint that is not an atom of the molecule"),
            });
        };
        let bond_type = antechamber_bond_type(&bond)
            .map_err(|detail| BccIncrementError::Malformed { detail })?;
        let Some(dq) = table.correction(types[i], types[j], bond_type) else {
            return Err(BccIncrementError::MissingRow {
                bond: bid,
                left: types[i].to_owned(),
                right: types[j].to_owned(),
                bond_type,
            });
        };
        delta[i] += dq;
        delta[j] -= dq;
    }
    Ok(delta)
}

/// Applies a [`BCCCorrectionTable`] to the AM1 base charges of a typed molecule.
///
/// A corrector *is* its table, so there is no way to build one without naming it:
/// a corrector with an empty table constructs happily and then rejects the first
/// bond it sees, which is a runtime failure standing in for a compile-time one.
///
/// This is the LOW-level half of AM1-BCC: the molecule must already carry BCC atom
/// types and BCC bond types, so the types are the caller's argument rather than the
/// corrector's job. The high-level half — perceive the types, correct, hand the
/// charges back without touching the graph — is `BccModel::correct` in
/// [`ff::charge`](crate::ff::charge).
#[derive(Debug, Clone)]
pub struct BCCCorrector {
    table: BCCCorrectionTable,
}

impl BCCCorrector {
    /// A corrector that applies `table`.
    ///
    /// # Arguments
    ///
    /// * `table` — the correction rows, built from a parameter set (or explicit rows).
    ///
    /// # Returns
    ///
    /// The corrector.
    pub fn new(table: BCCCorrectionTable) -> Self {
        Self { table }
    }

    /// Add the table's increments to a typed molecule's `charge` column, in place.
    ///
    /// # Arguments
    ///
    /// * `mol` — a molecule carrying BCC atom types ([`keys::TYPE`]), BCC bond types
    ///   ([`BCC_BOND_TYPE`](molrs::perceive::bond_type::BCC_BOND_TYPE)) and AM1 base
    ///   charges ([`keys::CHARGE`]). Its charges are replaced by the corrected ones.
    ///
    /// # Errors
    ///
    /// A message when an atom has no `type`, an atom has no base `charge`, or the
    /// table has no row for a bond — never a silent zero for any of them.
    pub fn apply(&self, mol: &mut Atomistic) -> Result<(), String> {
        let atom_ids: Vec<AtomId> = mol.atoms().map(|(id, _)| id).collect();
        let types: Vec<String> = atom_ids
            .iter()
            .map(|aid| {
                mol.get_atom(*aid)
                    .map_err(|e| e.to_string())?
                    .get_str(keys::TYPE)
                    .map(str::to_owned)
                    .ok_or_else(|| "BCCCorrector requires atom `type` labels".to_owned())
            })
            .collect::<Result<_, String>>()?;
        let types: Vec<&str> = types.iter().map(String::as_str).collect();

        let delta = bcc_increments(&self.table, mol, &types).map_err(|e| e.to_string())?;

        for (i, aid) in atom_ids.iter().copied().enumerate() {
            let q0 = mol
                .get_atom(aid)
                .map_err(|e| e.to_string())?
                .get_f64(keys::CHARGE)
                .ok_or_else(|| "BCCCorrector requires AM1 base `charge`".to_owned())?;
            mol.set_atom(aid, keys::CHARGE, q0 + delta[i])
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}
