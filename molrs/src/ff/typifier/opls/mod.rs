//! OPLS-AA SMARTS atom typifier.
//!
//! Mirrors [`mmff`](crate::ff::typifier::mmff): typing metadata
//! ([`OplsTypingMeta`]) is read *separately* from the potential
//! [`ForceField`](crate::ff::forcefield::ForceField), both from the same OPLS-AA
//! XML. [`OplsTypifier`] owns both and implements [`Typifier`], assigning
//! `opls_NNN` atom types by SMARTS matching with overrides / priority / layer
//! conflict resolution (replicating molpy's `_OplsAtomTypifier`).
//!
//! After atom typing, [`OplsTypifier::typify`] runs
//! [`assign_bonded`](assign::assign_bonded): every bond / angle / dihedral is
//! matched against the force field's bonded tables by OPLS specificity + overlay
//! layer (chain 2). [`OplsTypifier::build`] closes the loop to evaluable
//! potentials (`typify → to_frame → to_potentials`).
//!
//! # B-line reversal
//!
//! This reverses the "typifier does not sink (B-line)" decision of
//! `opls-ef-01-kernels-seam`: OPLS bonded-parameter assignment now happens in
//! Rust (here), not in a post-typify Python pass over a molpy `ForceField`.
//!
//! # Scope
//!
//! Only types carrying a SMARTS `def` participate; legacy `oplsaa.xml` rows
//! (`opls_001`–`opls_134`, no `def`) are out of scope for auto-typing. Improper
//! matching and pair/charge assignment are out of scope. Uncovered bonded terms
//! follow the [`NoMatch`](assign::NoMatch) policy; a consumer that wants to fill
//! them can supply its own [`Estimator`](assign::Estimator) via
//! [`assign_bonded_with`](assign::assign_bonded_with).

use molrs::Atomistic;

use crate::ff::forcefield::ForceField;
use crate::ff::forcefield::readers::{ForceFieldReader, opls::OplsXmlReader};
use crate::ff::potential::{Potentials, intramolecular_pairs};

use super::Typifier;

pub mod assign;
pub mod deps;
pub mod layered;
pub mod meta;
pub mod typing;

pub use assign::{
    BondedTerm, CandidateTables, Estimator, NoMatch, assign_bonded, assign_bonded_with,
};
pub use meta::{LAYER_PRIORITY_STRIDE, OplsTypeRow, OplsTypingMeta};
pub use typing::annotate_opls;

/// OPLS-AA typifier — owns typing metadata and force-field parameters.
///
/// Primary constructor [`from_xml_str`](Self::from_xml_str) parses both the
/// typing metadata ([`OplsTypingMeta`]) and the potential parameters
/// ([`ForceField`]) from a single OPLS-AA XML string, then precomputes the
/// bonded candidate tables ([`CandidateTables`]) once.
pub struct OplsTypifier {
    meta: OplsTypingMeta,
    ff: ForceField,
    tables: CandidateTables,
    /// No-match policy for bonded terms with no force-field candidate.
    no_match: NoMatch,
}

impl OplsTypifier {
    /// Build a typifier from an OPLS-AA / GROMACS XML string.
    ///
    /// Reads typing metadata and potential parameters in one call. The two are
    /// read by independent parsers from the same XML and never share state.
    /// The bonded candidate tables are built once from the parsed force field.
    /// Defaults to strict bonded matching ([`NoMatch::Error`]).
    ///
    /// # Errors
    ///
    /// Returns `Err` if either parse fails.
    pub fn from_xml_str(xml: &str) -> Result<Self, String> {
        let meta = crate::ff::forcefield::xml::read_opls_typing_xml_str(xml)?;
        let ff = OplsXmlReader::new().read_str(xml)?;
        Ok(Self::new(meta, ff))
    }

    /// Build a typifier from the embedded canonical OPLS-AA parameter set
    /// ([`molrs::data::OPLSAA_XML`](crate::core::data::OPLSAA_XML)).
    ///
    /// The XML is compiled into the binary, so this is the standalone path: the
    /// OPLS typifier needs no external file on disk. The embedded copy uses
    /// lowercase SMARTS `c` for aromatic ring carbons / hydrogens
    /// (RDKit-faithful aromatic matching), so benzene-type rings type exactly as
    /// molpy's ground truth. Mirrors
    /// [`MMFFTypifier::mmff94`](crate::ff::typifier::mmff::MMFFTypifier::mmff94).
    ///
    /// # Errors
    ///
    /// Returns `Err` if parsing the embedded XML fails (should not happen for the
    /// shipped data).
    pub fn oplsaa() -> Result<Self, String> {
        Self::from_xml_str(molrs::data::OPLSAA_XML)
    }

    /// Construct directly from already-parsed metadata and force field
    /// (strict bonded matching).
    pub fn new(meta: OplsTypingMeta, ff: ForceField) -> Self {
        let tables = CandidateTables::build(&ff, &meta);
        Self {
            meta,
            ff,
            tables,
            no_match: NoMatch::Error,
        }
    }

    /// Set the bonded no-match policy (chaining). `strict=true` →
    /// [`NoMatch::Error`]; `strict=false` → [`NoMatch::Skip`].
    pub fn with_strict(mut self, strict: bool) -> Self {
        self.no_match = if strict {
            NoMatch::Error
        } else {
            NoMatch::Skip
        };
        self
    }

    /// Access the typing metadata.
    pub fn meta(&self) -> &OplsTypingMeta {
        &self.meta
    }

    /// Access the potential force field.
    pub fn ff(&self) -> &ForceField {
        &self.ff
    }

    /// Typify atoms and assign bonded parameters in one step.
    ///
    /// `annotate_opls` types/charges the atoms, then
    /// [`assign_bonded`](assign::assign_bonded) labels every bond / angle /
    /// dihedral with the most specific matching force-field parameters.
    ///
    /// # Errors
    ///
    /// Propagates atom-typing and bonded-assignment errors.
    pub fn typify_full(&self, mol: &Atomistic) -> Result<Atomistic, String> {
        let typed = annotate_opls(mol, &self.meta, &self.ff)?;
        assign_bonded(&typed, &self.tables, self.no_match)
    }

    /// Typify a molecule and compile potentials in one step.
    ///
    /// `mol → typify (atoms + bonded) → Frame → Potentials`, mirroring
    /// [`MMFFTypifier::build`](crate::ff::typifier::mmff::MMFFTypifier::build).
    /// 1-2 / 1-3 exclusion + 1-4 scaling come from the force field's
    /// `special_bonds` (set by the reader) and the consumer-built
    /// [`intramolecular_pairs`] neighbour list inserted here.
    ///
    /// # Errors
    ///
    /// Propagates typing / assignment / compilation errors.
    pub fn build(&self, mol: &Atomistic) -> Result<Potentials, String> {
        let mut frame = self.typify_full(mol)?.to_frame();
        let pairs = intramolecular_pairs(&frame);
        frame.insert("pairs", pairs);
        self.ff.to_potentials(&frame)
    }
}

impl Typifier for OplsTypifier {
    /// Atom typing only (`opls_NNN` type / class / charge per atom).
    ///
    /// Bonded-parameter assignment is a deliberately separate step
    /// ([`typify_full`](Self::typify_full)) so callers can type atoms without
    /// requiring every bonded term to resolve — the chain-1 coverage gap means
    /// many real molecules are only partially typed. [`build`](Self::build)
    /// runs the full pipeline.
    fn typify(&self, mol: &Atomistic) -> Result<Atomistic, String> {
        annotate_opls(mol, &self.meta, &self.ff)
    }
}
