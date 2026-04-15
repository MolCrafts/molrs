//! Substructure matching: run a compiled [`SmartsPattern`] against a target.
//!
//! The matcher finds all subgraph isomorphisms from the pattern's query atoms
//! onto a target [`MolGraph`](molrs::molgraph::MolGraph). Each match is the
//! ordered list of target atom indices that correspond to the pattern atoms.
//!
//! The planned implementation wraps `petgraph::algo::isomorphism` with
//! predicate-evaluation closures that honor SMARTS semantics (aromaticity,
//! ring membership, logical query operators, recursive `$(...)` patterns).
//!
//! # Status
//!
//! The public API is in place; the implementation lands in session 2 of the
//! ETKDGv3 port.

use molrs::molgraph::MolGraph;

use crate::smarts::pattern::{SmartsError, SmartsPattern};

/// A single substructure match: an ordered list of target-atom indices, one
/// per query atom in the compiled pattern. The order matches the pattern's
/// atom declaration order, so callers can index directly (e.g. the torsion
/// library's `i-j-k-l` quartets map to `m.0[0..4]`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Match(pub Vec<usize>);

/// Run substructure matching against a target molecule.
///
/// Kept as a trait (rather than an inherent method on [`SmartsPattern`]) so
/// that specialized matchers (e.g. a bond-local matcher used by the ETKDG
/// torsion-preference lookup) can share the same entry point while providing
/// different internal strategies. The `Send + Sync` bound is mandatory — the
/// ETKDG multi-conformer pipeline shares compiled patterns across rayon
/// workers.
pub trait SubstructureMatcher: Send + Sync {
    /// Return every subgraph isomorphism of `self` onto `target`.
    ///
    /// Returns an empty vector if no matches are found. Returns an error only
    /// if the pattern could not be evaluated (e.g. a recursive SMARTS failed
    /// to resolve).
    fn find_all(&self, target: &MolGraph) -> Result<Vec<Match>, SmartsError>;

    /// Return the first match, or `None` if the pattern does not occur in the
    /// target. Default impl calls [`Self::find_all`] and takes the head.
    fn find_first(&self, target: &MolGraph) -> Result<Option<Match>, SmartsError> {
        Ok(self.find_all(target)?.into_iter().next())
    }
}

impl SubstructureMatcher for SmartsPattern {
    fn find_all(&self, _target: &MolGraph) -> Result<Vec<Match>, SmartsError> {
        Err(SmartsError::NotYetImplemented)
    }
}
