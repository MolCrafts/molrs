//! The [`ChargeModel`] trait — one seam for every charge method.
//!
//! | model | needs QM input? | topology correction? | `needs_equivalencing` |
//! |---|---|---|---|
//! | Mulliken | yes | no (pass-through) | `false` (`-eq 0`) |
//! | AM1-BCC | yes | yes (bond increments) | `true` (`-eq 1`) |
//! | ABCG2 | yes | yes (bond increments) | `true` (`-eq 1`) |
//! | Gasteiger | **no** | yes (iterative) | `false` |
//!
//! The trait carries all four without any of them being a special case, which is
//! what says it has not quietly assumed "QM base charges plus a correction": the
//! QM charges are an `Option`, so a model that needs none simply ignores them, and
//! the correction is the model's own business, so a model with none returns what it
//! was handed.

use molrs::store::keys;
use molrs::{AtomId, Atomistic};

use molrs::perceive::equivalence::{EquivalenceOptions, average_charges, find_equivalence_classes};

use super::error::ChargeError;

/// A charge model: molecule (and optionally QM base charges) in, charges out.
///
/// Object-safe on purpose. The C++ and Python bridges choose a model at runtime, so
/// they must be able to hold one as `Box<dyn ChargeModel>`; and a trait whose users
/// have to downcast to find out what they are holding is a special case wearing a
/// trait's clothes.
///
/// Nothing is written back into the molecule. The charges are the return value, so
/// a caller keeps their force-field atom types and gets BCC charges — the standard
/// AM1-BCC workflow needs both at once.
pub trait ChargeModel {
    /// Does this model average its base charges over the topological-equivalence
    /// classes before correcting them?
    ///
    /// antechamber's `-eq`, whose default is **per charge method**: `1` for `bcc` /
    /// `abcg2` / `resp`, `0` for everything else. That makes it a property of the
    /// model, not of the graph and not of the caller's memory — hence a method here
    /// rather than a flag on either.
    ///
    /// # Returns
    ///
    /// `true` when [`assign`](Self::assign) applies the class-mean to the QM charges
    /// it is given.
    fn needs_equivalencing(&self) -> bool;

    /// The final charges of every atom, in graph atom order.
    ///
    /// This is the **whole** model: raw QM charges in (as an AM1 backend really
    /// hands them over — un-averaged, one conformer), final charges out. A model
    /// that declares [`needs_equivalencing`](Self::needs_equivalencing) applies the
    /// class-mean itself, because otherwise the declaration would be inert
    /// documentation that every caller had to re-implement.
    ///
    /// # Arguments
    ///
    /// * `mol` — the molecule; left untouched, including its `type` columns.
    /// * `qm` — QM base charges in graph atom order, or `None` when the caller has
    ///   none. A model that needs them fails with
    ///   [`ChargeError::MissingQmCharges`]; a model that does not (Gasteiger)
    ///   ignores the argument.
    ///
    /// # Returns
    ///
    /// One charge per atom, in graph atom order.
    ///
    /// # Errors
    ///
    /// [`ChargeError`] — missing QM input, a charge-count mismatch, an atom the
    /// parameter set cannot type, or a bond it cannot correct. Never a fallback
    /// value.
    fn assign(&self, mol: &Atomistic, qm: Option<&[f64]>) -> Result<Vec<f64>, ChargeError>;
}

/// The atoms of `mol`, in graph atom order — the order every charge slice is in.
pub(super) fn atom_ids(mol: &Atomistic) -> Vec<AtomId> {
    mol.atoms().map(|(aid, _)| aid).collect()
}

/// Reject a charge slice that does not belong to this molecule.
///
/// # Errors
///
/// [`ChargeError::ChargeCountMismatch`], carrying both counts, when the slice is
/// not one charge per atom.
pub(super) fn check_count(mol: &Atomistic, charges: &[f64]) -> Result<(), ChargeError> {
    let expected = mol.n_atoms();
    if charges.len() != expected {
        return Err(ChargeError::ChargeCountMismatch {
            expected,
            got: charges.len(),
        });
    }
    Ok(())
}

/// Average `qm` over the molecule's topological-equivalence classes (`-eq 1`).
///
/// The step antechamber runs between AM1 and BCC. A semi-empirical calculation is
/// done on one conformer, so its Mulliken charges are not symmetric — methanol's
/// three methyl hydrogens come out of `sqm` split `0.053 / 0.098 / 0.053` purely
/// because one of them eclipses the O–H — and the class-mean removes that artefact
/// before any bond-charge correction is applied.
///
/// The perception and the mean are molrs's one implementation of each
/// ([`find_equivalence_classes`] and [`average_charges`]); this only carries `qm`
/// through them, which is why the charges are laid onto a throwaway clone rather
/// than re-summed here. Two class-means would be two chances to disagree with
/// antechamber in the last bits.
///
/// # Arguments
///
/// * `mol` — the molecule to partition; left untouched.
/// * `qm` — one base charge per atom, in graph atom order.
///
/// # Returns
///
/// The class-averaged charges, in the same order.
///
/// # Errors
///
/// [`ChargeError::Malformed`] when the charges cannot be laid onto the graph or
/// read back off it.
pub(super) fn equivalence_average(mol: &Atomistic, qm: &[f64]) -> Result<Vec<f64>, ChargeError> {
    let ids = atom_ids(mol);
    let mut base = mol.clone();
    for (aid, q) in ids.iter().zip(qm) {
        base.set_atom(*aid, keys::CHARGE, *q)
            .map_err(|e| ChargeError::Malformed {
                detail: e.to_string(),
            })?;
    }

    let classes = find_equivalence_classes(&base, EquivalenceOptions::bcc());
    let averaged = average_charges(&base, &classes);

    ids.iter()
        .map(|aid| {
            averaged
                .get_atom(*aid)
                .ok()
                .and_then(|atom| atom.get_f64(keys::CHARGE))
                .ok_or_else(|| ChargeError::Malformed {
                    detail: format!("atom {aid:?} lost its charge while equivalencing"),
                })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Methanol's three methyl hydrogens, which `sqm` splits and `-eq 1` merges.
    fn methanol() -> Atomistic {
        let mut mol = Atomistic::new();
        let c = mol.add_atom_xyz("C", 0.0, 0.0, 0.0);
        let o = mol.add_atom_xyz("O", 1.43, 0.0, 0.0);
        let h1 = mol.add_atom_xyz("H", -0.63, 0.63, 0.63);
        let h2 = mol.add_atom_xyz("H", -0.63, -0.63, 0.63);
        let h3 = mol.add_atom_xyz("H", -0.63, 0.0, -0.89);
        let ho = mol.add_atom_xyz("H", 1.76, 0.89, 0.0);
        for (a, b) in [(c, o), (c, h1), (c, h2), (c, h3), (o, ho)] {
            mol.add_bond(a, b).expect("add bond");
        }
        mol
    }

    /// The class-mean lands on the three methyl hydrogens and nowhere else.
    ///
    /// The hydroxyl H is in a class of its own (it is bonded to O, not C), so it
    /// must come back bit-for-bit; the methyl three become their mean.
    #[test]
    fn equivalence_average_merges_the_methyl_hydrogens_only() {
        let raw = [-0.073, -0.326, 0.053, 0.098, 0.053, 0.195];
        let got = equivalence_average(&methanol(), &raw).expect("average methanol");

        let mean: f64 = (0.053 + 0.098 + 0.053) / 3.0;
        for (k, q) in got.iter().enumerate().take(5).skip(2) {
            assert_eq!(q.to_bits(), mean.to_bits(), "methyl H {k}");
        }
        assert_eq!(
            got[5].to_bits(),
            raw[5].to_bits(),
            "the hydroxyl H is alone"
        );
        assert_eq!(got[0].to_bits(), raw[0].to_bits(), "carbon is alone");
    }

    /// A charge slice that is not one-per-atom is refused with both counts.
    #[test]
    fn check_count_reports_both_counts() {
        let err = check_count(&methanol(), &[0.0; 3]).expect_err("6 atoms, 3 charges");
        assert_eq!(
            err,
            ChargeError::ChargeCountMismatch {
                expected: 6,
                got: 3
            }
        );
    }
}
