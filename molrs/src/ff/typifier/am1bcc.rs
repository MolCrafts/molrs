//! AM1-BCC charge typifiers.
//!
//! The public shape is intentionally parallel to `MMFFTypifier` and
//! `OPLSAATypifier`: a struct owns the model/backend state, `typify()` returns
//! a labeled `Atomistic`, and `charge` is written as a canonical atom property.
//!
//! The implementation is a native port surface: external programs are reference
//! material only, not runtime dependencies. Missing AM1 backends, BCC atom
//! types, bond types, or correction rows return errors.
//!
//! Atom typing itself lives one module over, in [`atd`](super::atd): the rules
//! that assign `11` / `91` / `24` are antechamber's ATD language, shared by every
//! `ATOMTYPE_*.DEF` table, and this module drives that one engine with the BCC
//! (or ABCG2) table rather than owning a second copy of it. What is BCC-specific
//! is what stays here: the **correction** table, which no other parameter set has.

use molrs::store::keys;
use molrs::{AtomId, Atomistic};
use std::collections::HashMap;

use super::Typifier;
use super::atd::{AtdParameterSet, AtdTypifier, antechamber_bond_type};
use crate::ff::params::{BccAlias, BccCorrectionRow};

/// AM1 base-charge result supplied by an AM1 backend.
#[derive(Debug, Clone, PartialEq)]
pub struct AM1ChargeResult {
    pub charges: Vec<f64>,
    pub total_charge: Option<f64>,
    pub heat_of_formation_kcal_mol: Option<f64>,
    pub reference: String,
}

impl AM1ChargeResult {
    pub fn new(charges: Vec<f64>) -> Self {
        Self {
            charges,
            total_charge: None,
            heat_of_formation_kcal_mol: None,
            reference: String::new(),
        }
    }
}

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
    pub fn atd_set(self) -> AtdParameterSet {
        match self {
            Self::Bcc => AtdParameterSet::Bcc,
            Self::Abcg2 => AtdParameterSet::Abcg2,
        }
    }
}

/// Backend trait for obtaining AM1 base charges.
pub trait AM1ChargeBackend {
    fn compute_am1_charges(&self, mol: &Atomistic) -> Result<AM1ChargeResult, String>;
}

/// Guarded default backend used until Atomiverse's AM1 solver is linked in.
#[derive(Debug, Clone, Copy, Default)]
pub struct UnavailableAM1Backend;

impl AM1ChargeBackend for UnavailableAM1Backend {
    fn compute_am1_charges(&self, _mol: &Atomistic) -> Result<AM1ChargeResult, String> {
        Err("AM1-BCC requires an AM1ChargeBackend backed by Atomiverse AM1; no backend is configured".to_owned())
    }
}

/// AM1 base-charge typifier without bond charge corrections.
#[derive(Debug, Clone)]
pub struct AM1ChargeTypifier<B = UnavailableAM1Backend> {
    backend: B,
    total_charge: Option<f64>,
}

impl Default for AM1ChargeTypifier<UnavailableAM1Backend> {
    fn default() -> Self {
        Self::new(UnavailableAM1Backend)
    }
}

impl<B> AM1ChargeTypifier<B> {
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            total_charge: None,
        }
    }

    pub fn with_total_charge(mut self, total_charge: f64) -> Self {
        self.total_charge = Some(total_charge);
        self
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }
}

impl<B> Typifier for AM1ChargeTypifier<B>
where
    B: AM1ChargeBackend,
{
    type Mol = Atomistic;

    fn typify(&self, mol: &Atomistic) -> Result<Atomistic, String> {
        let am1 = self.backend.compute_am1_charges(mol)?;
        if am1.charges.len() != mol.n_atoms() {
            return Err(format!(
                "AM1 backend returned {} charges for {} atoms",
                am1.charges.len(),
                mol.n_atoms()
            ));
        }

        let mut out = mol.clone();
        let atom_ids: Vec<AtomId> = out.atoms().map(|(id, _)| id).collect();
        for (i, aid) in atom_ids.iter().copied().enumerate() {
            out.set_atom(aid, keys::CHARGE, am1.charges[i])
                .map_err(|e| e.to_string())?;
        }
        if let Some(target) = self.total_charge {
            normalize_total_charge(&mut out, target)?;
        }
        Ok(out)
    }
}

/// Graph-based BCC atom typifier: the [`AtdTypifier`] bound to a BCC table.
///
/// This is a named shorthand, not a second engine — `BCCAtomTypifier::bcc()` and
/// `AtdTypifier::new(AtdParameterSet::Bcc)` label every atom identically because
/// the former *is* the latter. It exists because the AM1-BCC pipeline needs the
/// atom-type table and the correction family chosen together, and
/// [`BccParameterSet`] is the type that keeps that pair honest.
#[derive(Debug, Clone)]
pub struct BCCAtomTypifier {
    model: BccParameterSet,
}

impl Default for BCCAtomTypifier {
    fn default() -> Self {
        Self::bcc()
    }
}

impl BCCAtomTypifier {
    pub fn parameter_set(model: BccParameterSet) -> Self {
        Self { model }
    }

    pub fn bcc() -> Self {
        Self::parameter_set(BccParameterSet::Bcc)
    }

    pub fn abcg2() -> Self {
        Self::parameter_set(BccParameterSet::Abcg2)
    }

    /// Perceive BCC bond types, then label every atom from the set's
    /// `ATOMTYPE_*.DEF` rules.
    ///
    /// The bond types are always **perceived** (via
    /// [`Perceive::find_bond_types`](molrs::perceive::Perceive::find_bond_types)),
    /// never read off the input: the atom-type rules count `sb`/`db`/`ab`/`DL`
    /// bonds, so they need the delocalized (9) and aromatic (7/8) types that a bond
    /// *order* cannot express — and an input `type` may be the unresolved aromatic
    /// precursor (10), which must be resolved, not trusted. To apply corrections
    /// with types of your own, drive [`BCCCorrector`] directly.
    pub fn typify(&self, mol: &Atomistic) -> Result<Atomistic, String> {
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
    /// Fallible only to keep the call shape of the former text-parsing
    /// constructor; the tables are typed Rust data, so this cannot fail.
    pub fn parameter_set(model: BccParameterSet) -> Result<Self, String> {
        let mut table = Self::from_rows(model.corrections());
        for alias in model.aliases() {
            table = table.alias(alias.atom_type, alias.reference);
        }
        Ok(table)
    }

    pub fn bcc() -> Result<Self, String> {
        Self::parameter_set(BccParameterSet::Bcc)
    }

    pub fn abcg2() -> Result<Self, String> {
        Self::parameter_set(BccParameterSet::Abcg2)
    }

    pub fn insert(self, left: impl Into<String>, right: impl Into<String>, delta: f64) -> Self {
        self.insert_bond_type(left, right, 1, delta)
    }

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

    pub fn alias(mut self, atom_type: impl Into<String>, reference: impl Into<String>) -> Self {
        self.aliases.insert(atom_type.into(), reference.into());
        self
    }

    pub fn len(&self) -> usize {
        self.corrections.len()
    }

    pub fn is_empty(&self) -> bool {
        self.corrections.is_empty()
    }

    pub fn alias_len(&self) -> usize {
        self.aliases.len()
    }

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

/// Applies a [`BCCCorrectionTable`] to the AM1 base charges of a typed molecule.
///
/// A corrector *is* its table, so there is no way to build one without naming it:
/// a corrector with an empty table constructs happily and then rejects the first
/// bond it sees, which is a runtime failure standing in for a compile-time one.
#[derive(Debug, Clone)]
pub struct BCCCorrector {
    table: BCCCorrectionTable,
}

impl BCCCorrector {
    /// A corrector that applies `table`.
    pub fn new(table: BCCCorrectionTable) -> Self {
        Self { table }
    }

    pub fn apply(&self, mol: &mut Atomistic) -> Result<(), String> {
        let atom_ids: Vec<AtomId> = mol.atoms().map(|(id, _)| id).collect();
        let mut index = HashMap::new();
        for (i, aid) in atom_ids.iter().copied().enumerate() {
            index.insert(aid, i);
        }

        let mut delta = vec![0.0; atom_ids.len()];
        let bond_rows: Vec<_> = mol.bonds().collect();
        for (bid, bond) in bond_rows {
            let a = bond.nodes[0];
            let b = bond.nodes[1];
            let bond_type = antechamber_bond_type(&bond)?;
            let a_label = mol
                .get_atom(a)
                .map_err(|e| e.to_string())?
                .get_str(keys::TYPE)
                .ok_or_else(|| "BCCCorrector requires atom `type` labels".to_owned())?
                .to_owned();
            let b_label = mol
                .get_atom(b)
                .map_err(|e| e.to_string())?
                .get_str(keys::TYPE)
                .ok_or_else(|| "BCCCorrector requires atom `type` labels".to_owned())?
                .to_owned();
            let Some(dq) = self.table.correction(&a_label, &b_label, bond_type) else {
                return Err(format!(
                    "missing BCC correction for bond {:?}: {a_label}|{b_label}|{bond_type}",
                    bid
                ));
            };
            delta[index[&a]] += dq;
            delta[index[&b]] -= dq;
        }

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

/// AM1-BCC typifier: AM1 backend + BCC atom typifier + correction table.
#[derive(Debug, Clone)]
pub struct AM1BCCTypifier<B = UnavailableAM1Backend> {
    backend: B,
    atom_typifier: BCCAtomTypifier,
    corrector: BCCCorrector,
    total_charge: Option<f64>,
}

impl<B> AM1BCCTypifier<B> {
    /// An AM1-BCC typifier over `backend`, correcting with `table`.
    ///
    /// The correction table is an argument because there is no default one: the
    /// two that exist are `BCCPARM.DAT` and `BCCPARM_ABCG2.DAT`, and the caller
    /// must say which — hence no parameterless form and no `Default`. To pair a
    /// published correction family with the atom-type table it is defined against
    /// (which is what you almost always want), use [`Self::bcc`] / [`Self::abcg2`]
    /// rather than assembling the pair by hand.
    ///
    /// Atoms are typed with [`BCCAtomTypifier::default`] — i.e. `ATOMTYPE_BCC.DEF`.
    /// A table whose rows speak a *different* atom-type language (ABCG2) needs the
    /// matching typifier too; see [`BccParameterSet::atd_set`] for why the pair
    /// must be kept honest, and override with [`Self::with_atom_typifier`].
    pub fn new(backend: B, table: BCCCorrectionTable) -> Self {
        Self {
            backend,
            atom_typifier: BCCAtomTypifier::default(),
            corrector: BCCCorrector::new(table),
            total_charge: None,
        }
    }

    /// The original AM1-BCC parameter set: `ATOMTYPE_BCC.DEF` + `BCCPARM.DAT`.
    pub fn bcc(backend: B) -> Result<Self, String> {
        Ok(Self::new(backend, BCCCorrectionTable::bcc()?)
            .with_atom_typifier(BCCAtomTypifier::bcc()))
    }

    /// The ABCG2 parameter set: `ATOMTYPE_ABCG2.DEF` + `BCCPARM_ABCG2.DAT`.
    pub fn abcg2(backend: B) -> Result<Self, String> {
        Ok(Self::new(backend, BCCCorrectionTable::abcg2()?)
            .with_atom_typifier(BCCAtomTypifier::abcg2()))
    }

    /// Replace the atom typifier (e.g. to match an ABCG2 correction table).
    pub fn with_atom_typifier(mut self, atom_typifier: BCCAtomTypifier) -> Self {
        self.atom_typifier = atom_typifier;
        self
    }

    /// Replace the correction table.
    pub fn with_correction_table(mut self, table: BCCCorrectionTable) -> Self {
        self.corrector = BCCCorrector::new(table);
        self
    }

    /// Replace the corrector wholesale.
    pub fn with_corrector(mut self, corrector: BCCCorrector) -> Self {
        self.corrector = corrector;
        self
    }

    /// Normalize the corrected charges to sum to `total_charge`.
    pub fn with_total_charge(mut self, total_charge: f64) -> Self {
        self.total_charge = Some(total_charge);
        self
    }

    /// The AM1 backend supplying base charges.
    pub fn backend(&self) -> &B {
        &self.backend
    }
}

impl<B> AM1BCCTypifier<B>
where
    B: AM1ChargeBackend,
{
    fn typify_am1bcc(&self, mol: &Atomistic) -> Result<Atomistic, String> {
        let am1 = self.backend.compute_am1_charges(mol)?;
        if am1.charges.len() != mol.n_atoms() {
            return Err(format!(
                "AM1 backend returned {} charges for {} atoms",
                am1.charges.len(),
                mol.n_atoms()
            ));
        }

        let mut out = self.atom_typifier.typify(mol)?;
        let atom_ids: Vec<AtomId> = out.atoms().map(|(id, _)| id).collect();
        for (i, aid) in atom_ids.iter().copied().enumerate() {
            out.set_atom(aid, keys::CHARGE, am1.charges[i])
                .map_err(|e| e.to_string())?;
        }

        self.corrector.apply(&mut out)?;
        if let Some(target) = self.total_charge.or(am1.total_charge) {
            normalize_total_charge(&mut out, target)?;
        }
        Ok(out)
    }
}

impl<B> Typifier for AM1BCCTypifier<B>
where
    B: AM1ChargeBackend,
{
    type Mol = Atomistic;

    fn typify(&self, mol: &Atomistic) -> Result<Atomistic, String> {
        self.typify_am1bcc(mol)
    }
}

fn normalize_total_charge(mol: &mut Atomistic, target: f64) -> Result<(), String> {
    let atom_ids: Vec<AtomId> = mol.atoms().map(|(id, _)| id).collect();
    if atom_ids.is_empty() {
        return Ok(());
    }
    let mut sum = 0.0;
    for aid in &atom_ids {
        sum += mol
            .get_atom(*aid)
            .map_err(|e| e.to_string())?
            .get_f64(keys::CHARGE)
            .ok_or_else(|| "cannot normalize missing charge".to_owned())?;
    }
    let shift = (target - sum) / atom_ids.len() as f64;
    for aid in atom_ids {
        let q = mol
            .get_atom(aid)
            .map_err(|e| e.to_string())?
            .get_f64(keys::CHARGE)
            .ok_or_else(|| "cannot normalize missing charge".to_owned())?;
        mol.set_atom(aid, keys::CHARGE, q + shift)
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}
