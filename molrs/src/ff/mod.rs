//! Force fields: what a force field *is*, and everything that turns a molecule
//! into one priced by it.
//!
//! Each submodule has one job, and its path is the only path to its items —
//! nothing is re-exported at this root.
//!
//! | module | its one job |
//! |---|---|
//! | [`ir`] | the force-field IR (adopts the LAMMPS standard): categories, style specs, dimensions, the style registry and its vocabulary, engine forms |
//! | [`forcefield`] | the [`ForceField`](forcefield::ForceField) data model: styles, types, their parameters, and the force field's own declarations (mixing, 1-4 pairs, torsion algebra, its record section) |
//! | [`potential`] | kernels: a force field's terms evaluated on coordinates, and the compiler that binds them to a frame |
//! | [`typifier`] | typing: a molecular graph in, the force-field types (and per-instance parameters) it carries out |
//! | [`charge`] | charge models: a molecule (and QM charges) in, partial charges out |
//! | [`params`] | every parameter table molrs ships, as compile-time data |
//! | [`clpol_scaling`] | CL&Pol's SAPT-derived Lennard-Jones scaling of a force field |
//!
//! No file format is here. Every file reader and writer — structure,
//! trajectory, and force-field files alike — is [`crate::io`]'s
//! ([`crate::io`] maps force-field files to and from a
//! [`ForceField`](forcefield::ForceField)). `ff` never depends on `io`
//! outside its `#[cfg(test)]` checks, which read engine files to compare
//! against.

pub mod charge;
pub mod clpol_scaling;
#[cfg(test)]
mod completeness;
#[cfg(test)]
mod engine_codec_check;
#[cfg(test)]
pub(crate) mod equivalence_check;
pub mod forcefield;
#[cfg(test)]
mod io_boundary;
pub mod ir;
#[cfg(test)]
mod ir_invariance;
#[cfg(test)]
mod one_four_lammps_check;
#[cfg(test)]
mod openmm_check;
pub mod params;
pub mod potential;
pub mod typifier;
