//! 3D conformer generation for molecular graphs.
//!
//! The public API is the [`Conformer`] struct: construct it with the desired
//! [`ConformerOptions`], then call [`Conformer::generate`] to produce 3D
//! coordinates. Internally this runs a staged ETKDGv3 workflow: initial
//! coordinate build -> coarse minimization -> rotor sampling -> final
//! minimization -> stereo sanity checks.
//!
//! ```no_run
//! use molrs::conformer::{Conformer, ConformerOptions};
//! # fn run(mol: &molrs::system::atomistic::Atomistic) -> Result<(), molrs::error::MolRsError> {
//! let (mol_3d, report) = Conformer::new(ConformerOptions::default()).generate(mol)?;
//! # let _ = (mol_3d, report);
//! # Ok(())
//! # }
//! ```

pub mod distgeom;
mod graph;
mod options;
mod report;

/// ETKDGv3 conformer-embedding pipeline (the active [`Conformer`] backend).
pub mod etkdg;

pub use options::{ConformerOptions, ConformerSpeed, ForceFieldKind};
pub use report::{ConformerReport, ConformerStageReport, StageKind};

use molrs::error::MolRsError;
use molrs::system::atomistic::Atomistic;

/// 3D conformer generator for all-atom molecular graphs.
///
/// Holds the [`ConformerOptions`] supplied at construction; [`generate`] runs
/// the staged ETKDGv3 pipeline against an input molecule and never mutates it.
///
/// [`generate`]: Conformer::generate
#[derive(Debug, Clone)]
pub struct Conformer {
    opts: ConformerOptions,
}

impl Conformer {
    /// Build a generator from explicit options.
    pub fn new(opts: ConformerOptions) -> Self {
        Self { opts }
    }

    /// The options this generator was constructed with.
    pub fn options(&self) -> &ConformerOptions {
        &self.opts
    }

    /// Generate 3D coordinates for an all-atom molecular graph.
    ///
    /// Returns the generated molecule (with 3D coordinates) and a
    /// stage-by-stage report. The input molecule is not modified.
    ///
    /// Requires [`Atomistic`] (not raw `MolGraph`) because conformer
    /// generation depends on element symbols for bond-length estimation, ring
    /// geometry, and force-field selection.
    pub fn generate(&self, mol: &Atomistic) -> Result<(Atomistic, ConformerReport), MolRsError> {
        // The ETKDG pipeline operates on (and returns) an `Atomistic` directly:
        // it only adds atoms via `add_hydrogens` (which sets "element") and
        // never removes "element" from existing atoms, so the chemistry
        // invariant is preserved end to end.
        etkdg::generate_3d_impl(mol, &self.opts)
    }
}

impl Default for Conformer {
    fn default() -> Self {
        Self::new(ConformerOptions::default())
    }
}
