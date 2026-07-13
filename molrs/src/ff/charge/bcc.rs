//! [`BccModel`] — AM1-BCC and ABCG2, as one model over two parameter sets.
//!
//! ```text
//! QM charges ──▶ equivalence classes ──▶ class-mean ──▶ + BCCPARM increments ──▶ q
//! ```
//!
//! The seam is a **push**: the AM1 charges are an argument
//! ([`correct`](BccModel::correct) takes them as a slice), not something a backend
//! trait is asked for. Every implementor that trait ever had was a constant-carrier
//! that ignored the molecule it was handed and returned a `Vec<f64>` computed
//! elsewhere — so production had to *fake a backend* to hand molrs a vector, which is
//! the shape of a seam installed backwards. molrs does not solve AM1 (that is
//! Atomiverse's job); it corrects the charges AM1 produced.
//!
//! # The `type` column is the caller's
//!
//! The model perceives its BCC atom and bond types into local `Vec`s and never writes
//! them into `mol` — nor reads the ones `mol` arrives with. Both halves matter:
//!
//! * **write** — the standard AM1-BCC workflow needs GAFF types *and* BCC charges at
//!   the same time (GAFF for LJ and bonded terms, BCC for the electrostatics). A
//!   charge pass that relabelled `c3` as `11` would destroy the force field it was
//!   meant to complete;
//! * **read** — a molecule read from a LAMMPS data file carries integer *bond type
//!   ids* in the very prop the BCC bond type lives in. Re-interpreting `1, 2, 3` as
//!   BCC bond types would split acetate's two carboxylate oxygens by ~0.2 e without a
//!   word. So the working copy is stripped of both `type` columns before perception.

use molrs::Atomistic;
use molrs::store::keys;

use crate::ff::typifier::am1bcc::{
    BCCCorrectionTable, BccIncrementError, BccParameterSet, bcc_increments,
};
use crate::ff::typifier::atd::{AtdError, AtdTypifier, DUMMY_TYPE};
use molrs::perceive::Perceive;

use super::error::ChargeError;
use super::model::{ChargeModel, atom_ids, check_count, equivalence_average};

/// The AM1-BCC family: QM base charges plus bond charge corrections.
///
/// One engine, two parameter sets — [`BccParameterSet::Bcc`] reads `BCCPARM.DAT`
/// against `ATOMTYPE_BCC.DEF`, [`BccParameterSet::Abcg2`] reads `BCCPARM_ABCG2.DAT`
/// against `ATOMTYPE_ABCG2.DEF` — and no special case for either.
///
/// # Examples
///
/// ```
/// use molrs::Atomistic;
/// use molrs::ff::charge::{BccModel, BccParameterSet};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut mol = Atomistic::new();
/// let c = mol.add_atom_xyz("C", 0.0, 0.0, 0.0);
/// for [x, y, z] in [
///     [0.63, 0.63, 0.63],
///     [-0.63, -0.63, 0.63],
///     [-0.63, 0.63, -0.63],
///     [0.63, -0.63, -0.63],
/// ] {
///     let h = mol.add_atom_xyz("H", x, y, z);
///     mol.add_bond(c, h)?;
/// }
///
/// // The AM1 charges come from an AM1 solver; here, a zero base, so that what
/// // comes back is the BCC increments themselves: C–H is +0.0393 e per bond.
/// let q = BccModel::new(BccParameterSet::Bcc)?.correct(&mol, &[0.0; 5])?;
/// assert!((q[0] - 4.0 * 0.0393).abs() < 1e-12);
/// assert!((q[1] + 0.0393).abs() < 1e-12);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct BccModel {
    set: BccParameterSet,
    table: BCCCorrectionTable,
}

impl BccModel {
    /// The model for one correction family.
    ///
    /// The parameter set is an argument because there is no default one: the two that
    /// exist are `BCCPARM.DAT` and `BCCPARM_ABCG2.DAT`, and a model with no table is
    /// an object that constructs, type-checks and then fails on the first bond it
    /// sees.
    ///
    /// # Arguments
    ///
    /// * `set` — the correction family, which also names the atom-type table its rows
    ///   are written in (see [`BccParameterSet::atd_set`]).
    ///
    /// # Returns
    ///
    /// A model holding that family's correction rows and `CORR` aliases.
    ///
    /// # Errors
    ///
    /// [`ChargeError::Malformed`] if the compile-time table cannot be built — which
    /// it cannot fail to be; the `Result` is the parameter set's refusal point.
    pub fn new(set: BccParameterSet) -> Result<Self, ChargeError> {
        let table = BCCCorrectionTable::parameter_set(set)
            .map_err(|detail| ChargeError::Malformed { detail })?;
        Ok(Self { set, table })
    }

    /// The correction family this model applies.
    ///
    /// # Returns
    ///
    /// The [`BccParameterSet`] it was built with.
    pub fn parameter_set(&self) -> BccParameterSet {
        self.set
    }

    /// Correct already-equivalenced base charges with the bond charge increments.
    ///
    /// The pure BCC stage, and the whole push API: molecule in behind a shared
    /// reference, base charges in as a slice, corrected charges out. Nothing is
    /// written back — not the charges, and above all not the BCC atom types, which
    /// the model perceives for itself and keeps to itself.
    ///
    /// The base charges are expected to be **already averaged over the equivalence
    /// classes** (antechamber's `-eq 1`), because that is what the BCC stage consumes;
    /// [`ChargeModel::assign`] is the door that does the averaging for you.
    ///
    /// # Arguments
    ///
    /// * `mol` — the molecule; left untouched, `type` columns included.
    /// * `am1` — one base charge per atom, in graph atom order.
    ///
    /// # Returns
    ///
    /// `am1[i] + Σ increments` for every atom, in graph atom order. The increments are
    /// pairwise antisymmetric, so the total charge is conserved — including the AM1
    /// rounding residual, which is **carried through, never normalized away**:
    /// `am1bcc.c` ends at the increment loop, and matching antechamber means carrying
    /// its residual rather than removing it.
    ///
    /// # Errors
    ///
    /// [`ChargeError::ChargeCountMismatch`] when `am1` is not one charge per atom;
    /// [`ChargeError::MissingAtomType`] when the table cannot type an atom;
    /// [`ChargeError::MissingCorrection`] when it has no row for a bond.
    pub fn correct(&self, mol: &Atomistic, am1: &[f64]) -> Result<Vec<f64>, ChargeError> {
        check_count(mol, am1)?;

        let work = without_type_columns(mol)?;
        let perceived = Perceive::new().find_bond_types(&work);
        let atd = self.set.atd_set();
        let types = AtdTypifier::new(atd)
            .types_of(&perceived)
            .map_err(charge_error)?;
        reject_dummy_types(&perceived, &types, atd.table().name)?;

        let delta = bcc_increments(&self.table, &perceived, &types).map_err(charge_error_row)?;

        Ok(am1.iter().zip(delta).map(|(q, d)| q + d).collect())
    }
}

impl ChargeModel for BccModel {
    fn needs_equivalencing(&self) -> bool {
        // antechamber's `-eq 1`, the default for `-c bcc` and `-c abcg2`.
        true
    }

    fn assign(&self, mol: &Atomistic, qm: Option<&[f64]>) -> Result<Vec<f64>, ChargeError> {
        let qm = qm.ok_or(ChargeError::MissingQmCharges { model: "AM1-BCC" })?;
        check_count(mol, qm)?;

        let base = if self.needs_equivalencing() {
            equivalence_average(mol, qm)?
        } else {
            qm.to_vec()
        };
        self.correct(mol, &base)
    }
}

/// A copy of `mol` with both `type` columns removed.
///
/// What the model perceives from: the caller's atom types (GAFF `c3`, OPLS `opls_135`)
/// and bond types (LAMMPS bond-type ids) are *their* labels, in the same props BCC
/// uses for its own. Perception reads a bond's `type` as an aromaticity hint, so a
/// LAMMPS molecule whose bond-type id happened to be 7 would be perceived as aromatic;
/// stripping the columns first makes the model's answer a function of the molecule
/// alone, which is what `correct_ignores_the_atom_types_the_molecule_arrives_with`
/// asserts to the bit.
fn without_type_columns(mol: &Atomistic) -> Result<Atomistic, ChargeError> {
    let mut work = mol.clone();
    let malformed = |e: molrs::MolRsError| ChargeError::Malformed {
        detail: e.to_string(),
    };

    for aid in atom_ids(mol) {
        work.clear_atom(aid, keys::TYPE).map_err(malformed)?;
    }
    let bond_kind = work.bond_kind();
    let bond_ids: Vec<_> = work.bonds().map(|(bid, _)| bid).collect();
    for bid in bond_ids {
        work.clear_relation_prop(bond_kind, bid, keys::TYPE)
            .map_err(malformed)?;
    }
    Ok(work)
}

/// Refuse a molecule whose atoms the BCC table could not name.
///
/// `ATOMTYPE_BCC.DEF` and `ATOMTYPE_ABCG2.DEF` end with an unconditional catch-all
/// row that labels an unmatched atom [`DUMMY_TYPE`], so the engine returns a type for
/// *every* atom and "no rule matched" arrives as a label rather than an error. For
/// AM1-BCC that label is a refusal: `BCCPARM.DAT` has no `DU` row, and neither
/// correction family assigns `DU` to any atom of the 37-molecule antechamber oracle.
///
/// A bonded `DU` atom would fail anyway, at the correction lookup. A **bondless** one
/// would not — nothing looks its type up — so a bare sulfur (the one element whose
/// BCC rules all require at least one bond) would come back with its AM1 charge
/// silently uncorrected, which is exactly the plausible-looking answer molrs's
/// no-fallback-values rule exists to prevent.
fn reject_dummy_types(
    perceived: &Atomistic,
    types: &[&str],
    table: &'static str,
) -> Result<(), ChargeError> {
    let Some((atom, _)) = types.iter().enumerate().find(|(_, ty)| **ty == DUMMY_TYPE) else {
        return Ok(());
    };
    let element = atom_ids(perceived)
        .get(atom)
        .and_then(|aid| perceived.get_atom(*aid).ok())
        .and_then(|a| a.get_str(keys::ELEMENT).map(str::to_owned))
        .unwrap_or_default();
    Err(ChargeError::MissingAtomType {
        table,
        atom,
        element,
    })
}

/// An atom-typing failure, as a charge failure.
fn charge_error(err: AtdError) -> ChargeError {
    match err {
        AtdError::MissingAtomType {
            table,
            atom,
            element,
        } => ChargeError::MissingAtomType {
            table,
            atom,
            element,
        },
        AtdError::Malformed { detail } => ChargeError::Malformed { detail },
    }
}

/// A correction-lookup failure, as a charge failure.
fn charge_error_row(err: BccIncrementError) -> ChargeError {
    match err {
        BccIncrementError::MissingRow {
            left,
            right,
            bond_type,
            ..
        } => ChargeError::MissingCorrection {
            left,
            right,
            bond_type,
        },
        BccIncrementError::Malformed { detail } => ChargeError::Malformed { detail },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Methane, untyped — the molecule a user has.
    fn methane() -> Atomistic {
        let mut mol = Atomistic::new();
        let c = mol.add_atom_xyz("C", 0.0, 0.0, 0.0);
        for [x, y, z] in [
            [0.63, 0.63, 0.63],
            [-0.63, -0.63, 0.63],
            [-0.63, 0.63, -0.63],
            [0.63, -0.63, -0.63],
        ] {
            let h = mol.add_atom_xyz("H", x, y, z);
            mol.add_bond(c, h).expect("add C-H");
        }
        mol
    }

    /// The molecule comes back untouched: `correct` takes `&Atomistic` and writes
    /// nothing, not even the charges it computed.
    #[test]
    fn correct_writes_nothing_into_the_molecule() {
        let mol = methane();
        let model = BccModel::new(BccParameterSet::Bcc).expect("build the model");
        let q = model.correct(&mol, &[0.0; 5]).expect("correct methane");

        assert!((q[0] - 4.0 * 0.0393).abs() < 1e-12, "{}", q[0]);
        for (_, atom) in mol.atoms() {
            assert_eq!(atom.get_str(keys::TYPE), None, "no BCC code was written");
            assert_eq!(atom.get_f64(keys::CHARGE), None, "no charge was written");
        }
        for (_, bond) in mol.bonds() {
            assert!(
                !bond.props.contains_key(keys::TYPE),
                "no BCC bond type was written"
            );
        }
    }

    /// The increments cancel: what goes in as a total comes out as a total.
    #[test]
    fn the_increments_conserve_the_total_charge() {
        let am1 = [-0.266, 0.066, 0.066, 0.066, 0.066];
        let q = BccModel::new(BccParameterSet::Bcc)
            .expect("build the model")
            .correct(&methane(), &am1)
            .expect("correct methane");

        let before: f64 = am1.iter().sum();
        let after: f64 = q.iter().sum();
        assert!((after - before).abs() < 1e-14, "{before} -> {after}");
    }

    /// `assign(None)` refuses; it does not invent a base to correct.
    #[test]
    fn assign_without_qm_charges_is_an_error() {
        let model = BccModel::new(BccParameterSet::Bcc).expect("build the model");
        assert_eq!(
            model.assign(&methane(), None),
            Err(ChargeError::MissingQmCharges { model: "AM1-BCC" })
        );
    }
}
