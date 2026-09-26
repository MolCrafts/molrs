//! Bond class and localized bond number — the two orthogonal facts about a bond.
//!
//! A bond carries **two** independent properties, and conflating them is the
//! defect this module exists to prevent:
//!
//! * [`BondType`] — what *kind* of bond it is. Aromatic is one of the kinds,
//!   peer to single / double / triple. A class, never a number.
//! * [`BondNumber`] — the integer bond number a localized Lewis / Kekulé
//!   structure gives it. Always an integer.
//!
//! An aromatic bond is therefore `BondType::Aromatic` **and** a `BondNumber` of
//! `Single` or `Double`: benzene's ring is six `Aromatic` types over an
//! alternating `1,2,1,2,1,2` of numbers. There is no fractional bond number —
//! `1.5` said "aromatic" and "one-and-a-half bonds" at once, so a consumer
//! reading the number could not tell which was meant and drew six double bonds.
//!
//! Fractional bond orders from electronic structure (Wiberg, Mayer, resonance
//! averages) are real quantities, but they are *computed properties* with their
//! own keys — never these two.

use crate::error::MolRsError;
use crate::store::keys;
use crate::system::molgraph::{KindId, MolGraph, PropValue, RelationId};

/// The chemical class of a bond.
///
/// Stored under [`keys::BOND_TYPE`] as its
/// [`code`](BondType::code).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
pub enum BondType {
    /// The input did not say, and perception has not run.
    #[default]
    Unknown,
    Single,
    Double,
    Triple,
    /// Part of an aromatic system. The code `4` is this protocol's aromatic
    /// marker — not a quadruple bond, which is a [`BondNumber`], not a type.
    Aromatic,
}

/// The integer bond number of a localized Lewis / Kekulé structure.
///
/// Stored under [`keys::BOND_NUMBER`] as its
/// [`code`](BondNumber::code).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
pub enum BondNumber {
    /// Not yet assigned — an aromatic bond before kekulization, or an input
    /// that never stated one. Not a legal state for a standardized molecule.
    #[default]
    Unknown,
    Single,
    Double,
    Triple,
    Quadruple,
}

impl BondType {
    /// The stored code: 0 unknown, 1 single, 2 double, 3 triple, 4 aromatic.
    pub fn code(self) -> u32 {
        match self {
            BondType::Unknown => 0,
            BondType::Single => 1,
            BondType::Double => 2,
            BondType::Triple => 3,
            BondType::Aromatic => 4,
        }
    }

    /// Read a stored code. Anything outside `0..=4` is [`BondType::Unknown`] —
    /// an unreadable class is "we do not know", never a guess.
    pub fn from_code(code: u32) -> Self {
        match code {
            1 => BondType::Single,
            2 => BondType::Double,
            3 => BondType::Triple,
            4 => BondType::Aromatic,
            _ => BondType::Unknown,
        }
    }

    /// Read the class from a stored prop; missing or non-numeric is `Unknown`.
    pub fn from_prop(prop: Option<&PropValue>) -> Self {
        match prop.and_then(PropValue::as_f64) {
            Some(v) if v >= 0.0 => BondType::from_code(v.round() as u32),
            _ => BondType::Unknown,
        }
    }

    /// Is this an aromatic bond? The question a renderer must ask *before* it
    /// asks about the number.
    pub fn is_aromatic(self) -> bool {
        matches!(self, BondType::Aromatic)
    }

    /// The bond number a non-aromatic class implies.
    ///
    /// Single is one bond, double is two — the two facts agree by construction
    /// for everything but aromatic, whose number only a Kekulé assignment can
    /// decide. `None` for `Aromatic` and `Unknown`, because neither implies one.
    pub fn implied_number(self) -> Option<BondNumber> {
        match self {
            BondType::Single => Some(BondNumber::Single),
            BondType::Double => Some(BondNumber::Double),
            BondType::Triple => Some(BondNumber::Triple),
            BondType::Aromatic | BondType::Unknown => None,
        }
    }
}

impl BondNumber {
    /// The stored code: 0 unknown, 1 single, 2 double, 3 triple, 4 quadruple.
    pub fn code(self) -> u32 {
        match self {
            BondNumber::Unknown => 0,
            BondNumber::Single => 1,
            BondNumber::Double => 2,
            BondNumber::Triple => 3,
            BondNumber::Quadruple => 4,
        }
    }

    /// Read a stored code. Anything outside `0..=4` is [`BondNumber::Unknown`].
    pub fn from_code(code: u32) -> Self {
        match code {
            1 => BondNumber::Single,
            2 => BondNumber::Double,
            3 => BondNumber::Triple,
            4 => BondNumber::Quadruple,
            _ => BondNumber::Unknown,
        }
    }

    /// Read the number from a stored prop; missing or non-numeric is `Unknown`.
    pub fn from_prop(prop: Option<&PropValue>) -> Self {
        match prop.and_then(PropValue::as_f64) {
            Some(v) if v >= 0.0 => BondNumber::from_code(v.round() as u32),
            _ => BondNumber::Unknown,
        }
    }

    /// The bond class a localized number implies: the inverse of
    /// [`BondType::implied_number`], and the one number → class map.
    ///
    /// `Quadruple` is classed [`BondType::Double`]: no quadruple class exists
    /// (the class code `4` is aromatic), so it takes the highest multiple-bond
    /// class below it, as the SMILES reader classes `$`. `Unknown` implies
    /// [`BondType::Unknown`].
    pub fn implied_type(self) -> BondType {
        match self {
            BondNumber::Single => BondType::Single,
            BondNumber::Double | BondNumber::Quadruple => BondType::Double,
            BondNumber::Triple => BondType::Triple,
            BondNumber::Unknown => BondType::Unknown,
        }
    }

    /// The number as a count, for valence sums. `Unknown` counts as zero; a
    /// caller that cannot tolerate that must check for it.
    pub fn count(self) -> u32 {
        match self {
            BondNumber::Unknown => 0,
            _ => self.code(),
        }
    }
}

// Graph props are `Int`; the schema declares the *frame* columns `UInt`, and
// `to_frame` re-types on the way out. Writing the code in exactly one place is
// what keeps the two from drifting.
impl From<BondType> for PropValue {
    fn from(v: BondType) -> Self {
        PropValue::Int(v.code() as i32)
    }
}

impl From<BondNumber> for PropValue {
    fn from(v: BondNumber) -> Self {
        PropValue::Int(v.code() as i32)
    }
}

/// Stamp both facts about a bond — its class and its localized number — onto
/// relation `id` of `kind` in `graph`.
///
/// The two keys are written together because they are only meaningful together:
/// a class without a number leaves the bond un-standardized, and a number
/// without a class leaves a renderer no way to tell aromatic from double. It
/// lives here, beside the vocabulary, because two leaves write it —
/// [`Atomistic::set_bond_class`](crate::system::atomistic::Atomistic::set_bond_class)
/// and [`Fragment::add_bond`](crate::system::fragment::Fragment::add_bond).
///
/// # Errors
///
/// Returns [`MolRsError::NotFound`] when `kind` is unregistered or `id` names
/// no live relation of it, and [`MolRsError::Validation`] when the store
/// refuses either value.
pub(crate) fn write_bond_class(
    graph: &mut MolGraph,
    kind: KindId,
    id: RelationId,
    bond_type: BondType,
    bond_number: BondNumber,
) -> Result<(), MolRsError> {
    graph.set_relation_prop(kind, id, keys::BOND_TYPE, bond_type)?;
    graph.set_relation_prop(kind, id, keys::BOND_NUMBER, bond_number)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::keys;
    use crate::system::molgraph::MolGraph;

    /// The two-key write is one function because a class without a number
    /// leaves the bond un-standardized: both keys land, or neither does.
    #[test]
    fn write_bond_class_stamps_both_keys() {
        let mut graph = MolGraph::new();
        let kind = graph.register_kind("bonds", 2);
        let a = graph.add_node();
        let b = graph.add_node();
        let bid = graph.add_relation(kind, &[a, b]).unwrap();

        write_bond_class(
            &mut graph,
            kind,
            bid,
            BondType::Aromatic,
            BondNumber::Double,
        )
        .expect("both props are writable on a live relation");

        let rel = graph.get_relation(kind, bid).unwrap();
        assert_eq!(
            BondType::from_prop(rel.props.get(keys::BOND_TYPE)),
            BondType::Aromatic
        );
        assert_eq!(
            BondNumber::from_prop(rel.props.get(keys::BOND_NUMBER)),
            BondNumber::Double
        );
    }

    #[test]
    fn aromatic_is_a_type_not_a_number() {
        // The invariant this module exists for: `4` means aromatic as a type
        // and quadruple as a number, and neither of them is `1.5`.
        assert_eq!(BondType::Aromatic.code(), 4);
        assert_eq!(BondNumber::Quadruple.code(), 4);
        assert!(BondType::Aromatic.is_aromatic());
        assert_eq!(BondType::Aromatic.implied_number(), None);
    }

    #[test]
    fn a_plain_type_implies_its_own_number() {
        for (t, n) in [
            (BondType::Single, BondNumber::Single),
            (BondType::Double, BondNumber::Double),
            (BondType::Triple, BondNumber::Triple),
        ] {
            assert_eq!(t.implied_number(), Some(n));
            assert_eq!(t.code(), n.code());
        }
    }

    /// The one number → class map (amended 2026-09-26), the inverse of
    /// `BondType::implied_number`. Quadruple has no class of its own and is
    /// classed `Double`, as the SMILES reader classes `$`.
    #[test]
    fn bond_number_implied_type_maps_each_order() {
        for (n, t) in [
            (BondNumber::Single, BondType::Single),
            (BondNumber::Double, BondType::Double),
            (BondNumber::Triple, BondType::Triple),
            (BondNumber::Quadruple, BondType::Double),
        ] {
            assert_eq!(n.implied_type(), t, "{n:?}");
        }
    }

    #[test]
    fn an_unreadable_code_is_unknown_never_a_guess() {
        assert_eq!(BondType::from_code(9), BondType::Unknown);
        assert_eq!(BondNumber::from_code(9), BondNumber::Unknown);
        assert_eq!(BondType::from_prop(None), BondType::Unknown);
        assert_eq!(
            BondType::from_prop(Some(&PropValue::Str("ar".into()))),
            BondType::Unknown
        );
        assert_eq!(BondNumber::Unknown.count(), 0);
    }

    #[test]
    fn codes_round_trip() {
        for code in 0..=4u32 {
            assert_eq!(BondType::from_code(code).code(), code);
            assert_eq!(BondNumber::from_code(code).code(), code);
        }
    }

    #[test]
    fn a_fractional_prop_can_never_read_back_as_aromatic() {
        // The old encoding must not survive as a silent alias: 1.5 rounds to 2,
        // which is Double, not Aromatic. Nothing turns 1.5 into aromaticity.
        assert_eq!(
            BondType::from_prop(Some(&PropValue::F64(1.5))),
            BondType::Double
        );
    }
}
