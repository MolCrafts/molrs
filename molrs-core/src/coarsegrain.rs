//! Coarse-grained molecular graph.
//!
//! [`CoarseGrain`] is a newtype wrapper around [`MolGraph`] where every node
//! represents a bead (a group of atoms) rather than a single atom. The
//! invariant is that every node carries a `"bead_type"` property.
//!
//! All [`MolGraph`] methods are available via `Deref`/`DerefMut`.
//!
//! # Examples
//!
//! ```
//! use molrs::coarsegrain::CoarseGrain;
//!
//! let mut cg = CoarseGrain::new();
//! let b1 = cg.add_bead("W", 0.0, 0.0, 0.0);
//! let b2 = cg.add_bead("W", 3.0, 0.0, 0.0);
//! cg.add_bond(b1, b2).unwrap();
//!
//! assert_eq!(cg.n_atoms(), 2);
//! assert_eq!(cg.n_bonds(), 1);
//! ```

use std::ops::{Deref, DerefMut};

use crate::error::MolRsError;
use crate::molgraph::{Atom, AtomId, MolGraph};

/// Coarse-grained molecular graph.
///
/// Invariant: every node has a `"bead_type"` property.
#[derive(Debug, Clone)]
pub struct CoarseGrain(MolGraph);

impl Deref for CoarseGrain {
    type Target = MolGraph;
    fn deref(&self) -> &MolGraph {
        &self.0
    }
}

impl DerefMut for CoarseGrain {
    fn deref_mut(&mut self) -> &mut MolGraph {
        &mut self.0
    }
}

impl Default for CoarseGrain {
    fn default() -> Self {
        Self::new()
    }
}

impl CoarseGrain {
    /// Create an empty coarse-grained molecular graph.
    pub fn new() -> Self {
        Self(MolGraph::new())
    }

    /// Add a bead with type name and 3D coordinates.
    pub fn add_bead(&mut self, bead_type: &str, x: f64, y: f64, z: f64) -> AtomId {
        let mut a = Atom::new();
        a.set("bead_type", bead_type);
        a.set("x", x);
        a.set("y", y);
        a.set("z", z);
        self.0.add_atom(a)
    }

    /// Add a bead with type name only (no coordinates).
    pub fn add_bead_bare(&mut self, bead_type: &str) -> AtomId {
        let mut a = Atom::new();
        a.set("bead_type", bead_type);
        self.0.add_atom(a)
    }

    /// Promote from a [`MolGraph`], validating all nodes have `"bead_type"`.
    pub fn try_from_molgraph(mol: MolGraph) -> Result<Self, MolRsError> {
        for (id, atom) in mol.atoms() {
            if atom.get_str("bead_type").is_none() {
                return Err(MolRsError::validation(format!(
                    "node {:?} missing 'bead_type' property",
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
    fn test_new_and_add_bead() {
        let mut cg = CoarseGrain::new();
        let b1 = cg.add_bead("W", 0.0, 0.0, 0.0);
        let b2 = cg.add_bead("P1", 3.0, 0.0, 0.0);
        cg.add_bond(b1, b2).unwrap();

        assert_eq!(cg.n_atoms(), 2);
        assert_eq!(cg.n_bonds(), 1);
    }

    #[test]
    fn test_bead_has_type() {
        let mut cg = CoarseGrain::new();
        let b = cg.add_bead("W", 1.0, 2.0, 3.0);

        let atom = cg.get_atom(b).unwrap();
        assert_eq!(atom.get_str("bead_type"), Some("W"));
        assert_eq!(atom.get_f64("x"), Some(1.0));
    }

    #[test]
    fn test_try_from_molgraph_ok() {
        let mut g = MolGraph::new();
        let mut a = Atom::new();
        a.set("bead_type", "W");
        a.set("x", 0.0);
        a.set("y", 0.0);
        a.set("z", 0.0);
        g.add_atom(a);

        let cg = CoarseGrain::try_from_molgraph(g);
        assert!(cg.is_ok());
    }

    #[test]
    fn test_try_from_molgraph_missing_bead_type() {
        let mut g = MolGraph::new();
        g.add_atom(Atom::new()); // no bead_type

        let cg = CoarseGrain::try_from_molgraph(g);
        assert!(cg.is_err());
    }

    #[test]
    fn test_into_inner() {
        let mut cg = CoarseGrain::new();
        cg.add_bead("W", 0.0, 0.0, 0.0);

        let g: MolGraph = cg.into_inner();
        assert_eq!(g.n_atoms(), 1);
    }
}
