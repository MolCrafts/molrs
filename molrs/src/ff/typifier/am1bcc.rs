//! The AM1-BCC atom *typifier*: [`BccAtomChargeTypifier`] labels every atom with
//! the BCC (or ABCG2) atom type of the table a
//! [`crate::ff::charge::BccParameterSet`] names.

use molrs::core::Atomistic;
use std::sync::OnceLock;

use super::atd::AtdTypifier;
use super::{TypeAssignment, Typifier};
use crate::ff::charge::BccParameterSet;
use crate::ff::forcefield::ForceField;

/// Graph-based BCC atom typifier: the [`AtdTypifier`] bound to a BCC table.
///
/// This is a named shorthand, not a second engine — `BccAtomChargeTypifier::bcc()` and
/// `AtdTypifier::new(AtdParameterSet::Bcc)` label every atom identically because
/// the former *is* the latter. It exists because the AM1-BCC pipeline needs the
/// atom-type table and the correction family chosen together, and
/// [`BccParameterSet`] is the type that keeps that pair honest.
///
/// It **writes** its labels into the graph's [`keys::TYPE`](molrs::core::keys::TYPE) column, so it is for
/// callers who want a BCC-typed molecule and nothing else. Charges do **not** go
/// through it: `BccModel` keeps its BCC types to itself precisely so that a molecule
/// can carry GAFF types and BCC charges at the same time, which is what the standard
/// AM1-BCC workflow is.
///
/// The atom-type rules are antechamber's ATD language, shared by every
/// `ATOMTYPE_*.DEF` table, so this drives the one ATD engine ([`AtdTypifier`](super::AtdTypifier))
/// with the BCC (or ABCG2) table rather than owning a second copy of it. The
/// bond charge corrections — the half of AM1-BCC that turns types into
/// charges — are a charge model's, and live in [`ff::charge`](crate::ff::charge)
/// (`BccModel`), which perceives its BCC types for itself and never writes them
/// into the caller's [`keys::TYPE`](molrs::core::keys::TYPE) column.
#[derive(Debug, Clone)]
pub struct BccAtomChargeTypifier {
    model: BccParameterSet,
}

impl Default for BccAtomChargeTypifier {
    fn default() -> Self {
        Self::bcc()
    }
}

impl BccAtomChargeTypifier {
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

impl Typifier for BccAtomChargeTypifier {
    /// Perceive BCC bond types, then label every atom from the set's
    /// `ATOMTYPE_*.DEF` rules.
    ///
    /// The bond types are always **perceived** — as antechamber perceives them,
    /// bond orders judged from the connectivity (`AtdTypifier`'s default
    /// [`AtdBondOrders::Perceive`](super::atd::AtdBondOrders::Perceive)) —
    /// never read off the input: the atom-type rules count `sb`/`db`/`ab`/`DL`
    /// bonds, so they need the delocalized (9) and aromatic (7/8) types that a bond
    /// *order* cannot express — and a supplied
    /// [`BCC_BOND_TYPE`](molrs::core::keys::BCC_BOND_TYPE) may be the
    /// unresolved aromatic precursor (10), which must be resolved, not trusted.
    ///
    /// The match is the [`AtdTypifier`]'s over the set's table: `graph`'s
    /// bonds carry perceived antechamber bond types in
    /// [`BCC_BOND_TYPE`](molrs::core::keys::BCC_BOND_TYPE) — the bond's own
    /// [`keys::TYPE`](molrs::core::keys::TYPE), the caller's force-field label, is left untouched — and
    /// every atom's BCC code is a `type` value. It defines nothing.
    ///
    /// # Errors
    ///
    /// A message naming the atom no rule of the table matched.
    fn assign(&self, graph: &mut Atomistic) -> Result<TypeAssignment, String> {
        AtdTypifier::new(self.model.atd_set()).assign(graph)
    }

    /// An empty force field named `BCC`: the atom-type table assigns labels,
    /// not parameters. Built once.
    fn source_forcefield(&self) -> &ForceField {
        static LIBRARY: OnceLock<ForceField> = OnceLock::new();
        LIBRARY.get_or_init(|| ForceField::new("BCC"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::typifier::Typing;
    use molrs::core::keys;

    /// Methane, hand-built: C is atom 0, the four hydrogens follow.
    fn methane() -> Atomistic {
        let mut m = Atomistic::new();
        let c = m.add_atom_bare("C");
        for _ in 0..4 {
            let h = m.add_atom_bare("H");
            m.add_bond(c, h).unwrap();
        }
        m
    }

    /// Typing through the base stamps a non-empty `type` on every atom and
    /// defines nothing: the match is stamp-only, so the output holds no type.
    #[test]
    fn typing_stamps_every_atom_and_defines_no_type() {
        let mut typing = Typing::new(BccAtomChargeTypifier::bcc());
        let typed = typing.typify(&methane()).expect("methane types");

        assert_eq!(typed.atoms().count(), 5);
        for (id, atom) in typed.atoms() {
            let t = atom.get_str(keys::TYPE);
            assert!(t.is_some_and(|t| !t.is_empty()), "atom {id:?}: {t:?}");
        }
        assert!(
            typing
                .forcefield()
                .styles()
                .iter()
                .all(|s| s.defs().collect_type_params().is_empty()),
            "output holds no type: {:?}",
            typing.forcefield()
        );
    }

    /// The library a BCC atom typer matches against is empty: no styles.
    #[test]
    fn library_is_an_empty_forcefield() {
        let typing = Typing::new(BccAtomChargeTypifier::bcc());
        assert!(typing.typifier().source_forcefield().styles().is_empty());
    }
}
