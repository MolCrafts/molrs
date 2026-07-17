//! The currency of the interpolation seam: one bonded term, named by its atoms.
//!
//! [`BondedTerm`] used to live inside the OPLS typifier, which made it read as an
//! OPLS thing ("the two endpoint `opls_NNN` types"). It never was: it is the query
//! type of the generic [`ParameterInterpolator`](super::ParameterInterpolator)
//! seam, and GAFF speaks it too. A term is its **atom-type names**, in the order
//! the force field writes them; which force field named them is not this type's
//! business.

/// One bonded term awaiting parameters: its arity-tagged endpoint atom types.
///
/// Handed to a [`ParameterInterpolator`](super::ParameterInterpolator) when no
/// force-field table covers the term. Kept small and owned so an interpolator
/// needs no access to the molecular graph.
///
/// Proper terms ([`Bond`](Self::Bond), [`Angle`](Self::Angle),
/// [`Dihedral`](Self::Dihedral)) are **reversal-symmetric** — `i-j-k-l` and
/// `l-k-j-i` are the same term — and their slot order is the chain along the
/// bonds. An [`Improper`](Self::Improper) is not: see its own note.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BondedTerm {
    /// A bond: the two endpoint atom types.
    Bond([String; 2]),
    /// An angle: the three atom types, **vertex in the middle**.
    Angle([String; 3]),
    /// A dihedral: the four atom types along the chain, inner pair in the middle.
    Dihedral([String; 4]),
    /// An improper: the four atom types with the **centre third** (`i-j-k-l`,
    /// `k` central), which is AMBER's slot order and the order
    /// [`ImproperPeriodic`](crate::ff::potential::improper::periodic::ImproperPeriodic)
    /// reads.
    ///
    /// The three peripherals are an unordered **set** — an improper is a
    /// planarity constraint on a centre, not a walk along bonds — so a matcher
    /// must try them against a row's peripheral slots in any order, and must not
    /// simply reverse the quartet the way it would a proper term.
    Improper([String; 4]),
}

impl BondedTerm {
    /// The centre atom type of an [`Improper`](Self::Improper), or `None` for a
    /// proper term (which has no distinguished centre).
    pub fn improper_centre(&self) -> Option<&str> {
        match self {
            Self::Improper(types) => Some(&types[2]),
            _ => None,
        }
    }

    /// The three peripheral atom types of an [`Improper`](Self::Improper), in the
    /// term's own slot order, or `None` for a proper term.
    pub fn improper_peripherals(&self) -> Option<[&str; 3]> {
        match self {
            Self::Improper(t) => Some([t[0].as_str(), t[1].as_str(), t[3].as_str()]),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_improper_names_its_centre_third() {
        let improper = BondedTerm::Improper([
            "ca".to_owned(),
            "ca".to_owned(),
            "ca".to_owned(),
            "ha".to_owned(),
        ]);
        assert_eq!(improper.improper_centre(), Some("ca"));
        assert_eq!(improper.improper_peripherals(), Some(["ca", "ca", "ha"]));
    }

    #[test]
    fn a_proper_term_has_no_centre() {
        let bond = BondedTerm::Bond(["c3".to_owned(), "hc".to_owned()]);
        assert_eq!(bond.improper_centre(), None);
        assert_eq!(bond.improper_peripherals(), None);
    }
}
