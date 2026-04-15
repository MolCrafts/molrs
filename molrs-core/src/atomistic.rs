//! All-atom molecular graph with element-level semantics.
//!
//! [`Atomistic`] is a newtype wrapper around [`MolGraph`] that guarantees every
//! atom carries a `"element"` property (element symbol). This invariant is
//! required by subsystems like [`embed`](crate::embed) and
//! [`typifier`](crate::typifier) that look up element data.
//!
//! All [`MolGraph`] methods are available via `Deref`/`DerefMut`.
//!
//! # Examples
//!
//! ```
//! use molrs_core::atomistic::Atomistic;
//!
//! let mut mol = Atomistic::new();
//! let c = mol.add_atom_bare("C");
//! let h = mol.add_atom_bare("H");
//! mol.add_bond(c, h).unwrap();
//!
//! assert_eq!(mol.n_atoms(), 2);
//! ```

use std::ops::{Deref, DerefMut};

use crate::error::MolRsError;
use crate::molgraph::{Atom, AtomId, MolGraph};

/// All-atom molecular graph.
///
/// Invariant: every atom has a `"element"` property containing a valid element
/// symbol string.
#[derive(Debug, Clone)]
pub struct Atomistic(MolGraph);

impl Deref for Atomistic {
    type Target = MolGraph;
    fn deref(&self) -> &MolGraph {
        &self.0
    }
}

impl DerefMut for Atomistic {
    fn deref_mut(&mut self) -> &mut MolGraph {
        &mut self.0
    }
}

impl Default for Atomistic {
    fn default() -> Self {
        Self::new()
    }
}

impl Atomistic {
    /// Create an empty all-atom molecular graph.
    pub fn new() -> Self {
        Self(MolGraph::new())
    }

    /// Add an atom with element symbol and 3D coordinates.
    pub fn add_atom_xyz(&mut self, symbol: &str, x: f64, y: f64, z: f64) -> AtomId {
        self.0.add_atom(Atom::xyz(symbol, x, y, z))
    }

    /// Add an atom with element symbol only (no coordinates).
    ///
    /// Use this when building molecules for 3D coordinate generation — the
    /// [`embed`](crate::embed) pipeline will assign coordinates.
    pub fn add_atom_bare(&mut self, symbol: &str) -> AtomId {
        let mut a = Atom::new();
        a.set("element", symbol);
        self.0.add_atom(a)
    }

    /// Promote from a [`MolGraph`], validating all atoms have `"element"`.
    pub fn try_from_molgraph(mol: MolGraph) -> Result<Self, MolRsError> {
        for (id, atom) in mol.atoms() {
            if atom.get_str("element").is_none() {
                return Err(MolRsError::validation(format!(
                    "atom {:?} missing 'symbol' property",
                    id
                )));
            }
        }
        Ok(Self(mol))
    }

    /// Unwrap to the inner [`MolGraph`] (zero cost).
    pub fn into_inner(self) -> MolGraph {
        self.0
    }

    /// Borrow the inner [`MolGraph`].
    pub fn as_molgraph(&self) -> &MolGraph {
        &self.0
    }

    /// Mutably borrow the inner [`MolGraph`].
    pub fn as_molgraph_mut(&mut self) -> &mut MolGraph {
        &mut self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_and_add() {
        let mut mol = Atomistic::new();
        let c = mol.add_atom_bare("C");
        let h = mol.add_atom_bare("H");
        mol.add_bond(c, h).unwrap();

        assert_eq!(mol.n_atoms(), 2);
        assert_eq!(mol.n_bonds(), 1);
        assert_eq!(mol.get_atom(c).unwrap().get_str("element"), Some("C"));
    }

    #[test]
    fn test_add_atom_xyz() {
        let mut mol = Atomistic::new();
        let o = mol.add_atom_xyz("O", 0.0, 0.0, 0.0);

        let atom = mol.get_atom(o).unwrap();
        assert_eq!(atom.get_str("element"), Some("O"));
        assert_eq!(atom.get_f64("x"), Some(0.0));
    }

    #[test]
    fn test_try_from_molgraph_ok() {
        let mut g = MolGraph::new();
        g.add_atom(Atom::xyz("C", 0.0, 0.0, 0.0));
        g.add_atom(Atom::xyz("H", 1.0, 0.0, 0.0));

        let mol = Atomistic::try_from_molgraph(g);
        assert!(mol.is_ok());
        assert_eq!(mol.unwrap().n_atoms(), 2);
    }

    #[test]
    fn test_try_from_molgraph_missing_symbol() {
        let mut g = MolGraph::new();
        g.add_atom(Atom::new()); // no symbol

        let mol = Atomistic::try_from_molgraph(g);
        assert!(mol.is_err());
    }

    #[test]
    fn test_deref_gives_molgraph_methods() {
        let mut mol = Atomistic::new();
        let c1 = mol.add_atom_bare("C");
        let c2 = mol.add_atom_bare("C");
        mol.add_bond(c1, c2).unwrap();

        // These all come from MolGraph via Deref
        assert_eq!(mol.n_atoms(), 2);
        assert_eq!(mol.n_bonds(), 1);
        let frame = mol.to_frame();
        assert!(frame.get("atoms").is_some());
    }

    #[test]
    fn test_into_inner() {
        let mut mol = Atomistic::new();
        mol.add_atom_bare("C");

        let g: MolGraph = mol.into_inner();
        assert_eq!(g.n_atoms(), 1);
    }
}
