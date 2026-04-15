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

pub use options::{EmbedAlgorithm, ForceFieldKind, EmbedOptions, EmbedSpeed};
pub use report::{EmbedReport, StageKind, StageReport};

use molrs::atomistic::Atomistic;
use molrs::error::MolRsError;

/// Generate 3D coordinates for an all-atom molecular graph.
///
/// Returns the generated molecule (with 3D coordinates) and a stage-by-stage
/// report. The input molecule is not modified.
///
/// Requires [`Atomistic`] (not raw `MolGraph`) because embed depends on
/// element symbols for bond-length estimation, ring geometry, and force-field
/// selection.
pub fn generate_3d(
    mol: &Atomistic,
    opts: &EmbedOptions,
) -> Result<(Atomistic, EmbedReport), MolRsError> {
    // Pipeline works on &MolGraph internally (via Deref).
    let (mol_out, report) = pipeline::generate_3d_impl(mol, opts)?;
    // embed only adds atoms via add_hydrogens (which sets "element") and never
    // removes "element" from existing atoms, so the invariant holds.
    let atomistic = Atomistic::try_from_molgraph(mol_out)
        .expect("embed pipeline preserves Atomistic invariant");
    Ok((atomistic, report))
}
