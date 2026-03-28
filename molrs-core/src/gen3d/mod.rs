//! 3D coordinate generation pipeline for molecular graphs.
//!
//! This module provides a practical `generate_3d` API using a staged workflow:
//! initial coordinate build -> coarse minimization -> rotor sampling ->
//! final minimization -> stereo sanity checks.

mod builder;
mod distance_geometry;
mod fragment_data;
mod geom;
mod optimizer;
mod options;
mod pipeline;
mod report;
mod rotor_search;
mod stereo_guard;

pub use options::{EmbedAlgorithm, ForceFieldKind, Gen3DOptions, Gen3DSpeed};
pub use report::{Gen3DReport, StageKind, StageReport};

use crate::atomistic::Atomistic;
use crate::error::MolRsError;

/// Generate 3D coordinates for an all-atom molecular graph.
///
/// Returns the generated molecule (with 3D coordinates) and a stage-by-stage
/// report. The input molecule is not modified.
///
/// Requires [`Atomistic`] (not raw `MolGraph`) because gen3d depends on
/// element symbols for bond-length estimation, ring geometry, and force-field
/// selection.
pub fn generate_3d(
    mol: &Atomistic,
    opts: &Gen3DOptions,
) -> Result<(Atomistic, Gen3DReport), MolRsError> {
    // Pipeline works on &MolGraph internally (via Deref).
    let (mol_out, report) = pipeline::generate_3d_impl(mol, opts)?;
    // gen3d only adds atoms via add_hydrogens (which sets "element") and never
    // removes "element" from existing atoms, so the invariant holds.
    let atomistic = Atomistic::try_from_molgraph(mol_out)
        .expect("gen3d pipeline preserves Atomistic invariant");
    Ok((atomistic, report))
}
