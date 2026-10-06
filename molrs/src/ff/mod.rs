//! Force fields: what a force field *is*, and everything that turns a molecule
//! into one priced by it.
//!
//! Each submodule has one job, and its path is the only path to its items —
//! nothing is re-exported at this root.
//!
//! | module | its one job |
//! |---|---|
//! | [`ir`] | the force-field IR (LAMMPS standard): categories, style specs, dimensions, the style registry and its vocabulary, engine forms |
//! | [`forcefield`] | the [`ForceField`](forcefield::ForceField) data model, and force-field files mapped to and from it ([`readers`](forcefield::readers), [`writers`](forcefield::writers), [`xml`](forcefield::xml)) |
//! | [`potential`] | kernels: a force field's terms evaluated on coordinates, and the compiler that binds them to a frame |
//! | [`typifier`] | typing: a molecular graph in, the force-field types (and per-instance parameters) it carries out |
//! | [`charge`] | charge models: a molecule (and QM charges) in, partial charges out |
//! | [`params`] | every parameter table molrs ships, as compile-time data |
//! | [`scale_lj`] | CL&Pol's SAPT-derived Lennard-Jones scaling of a force field |
//!
//! Structure and trajectory formats are not here: they are [`crate::io`]'s.
//! Force-field *file* formats are, because a reader's output is a force field,
//! not a frame; a reader may still use one of `io`'s low-level section parsers.

pub mod charge;
#[cfg(test)]
mod completeness;
pub(crate) mod constants;
#[cfg(test)]
mod engine_codec_check;
#[cfg(test)]
mod equivalence_check;
pub mod forcefield;
pub mod ir;
#[cfg(test)]
mod ir_invariance;
#[cfg(test)]
mod one_four;
#[cfg(test)]
mod openmm_check;
pub mod params;
pub mod potential;
pub mod scale_lj;
pub mod typifier;
