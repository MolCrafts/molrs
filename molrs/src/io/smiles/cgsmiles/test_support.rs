//! Test-only constructors for the resolved-pair vocabulary.
//!
//! The `resolve` and `to_atomistic` test modules both spell out
//! [`ResolvedPair`]s and their ends; this is the single copy they share.
//! Compiled under `#[cfg(test)]` only, so it never reaches a shipped build.

use super::ast::{PairEnd, ResolvedPair};
use crate::io::smiles::chem::ast::BondKind;

/// An end at the last level: the instance and the port index into its body's
/// descriptor map.
pub(super) fn body(instance: usize, port: usize) -> PairEnd {
    PairEnd::Body { instance, port }
}

/// One resolved pair, spelled out.
pub(super) fn pair(
    edge: usize,
    bond: usize,
    src: PairEnd,
    dst: PairEnd,
    kind: BondKind,
) -> ResolvedPair {
    ResolvedPair {
        edge,
        bond,
        src,
        dst,
        kind,
    }
}
