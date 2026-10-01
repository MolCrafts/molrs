//! 3D conformer generation for molecular graphs.
//!
//! The public API is the [`Conformer`] struct: construct it with the desired
//! [`ConformerOptions`], then call [`Conformer::generate`] to produce 3D
//! coordinates. Internally this runs a staged ETKDGv3 workflow: initial
//! coordinate build -> coarse minimization -> rotor sampling -> final
//! minimization -> stereo sanity checks.
//!
//! [`Conformer::generate`] is generic over [`ElementGraph`], so it returns the
//! same type it was handed: an [`Atomistic`] in, an `Atomistic` out, with its
//! ports ([`molrs::system::port`]) and `frag_id` labels intact. The hydrogens
//! the pipeline adds belong to no input unit, so relabelling them is the
//! caller's own step.
//!
//! ```no_run
//! use molrs::conformer::{Conformer, ConformerOptions};
//! use molrs::system::atomistic::Atomistic;
//! # fn run(mol: &Atomistic, unit: &Atomistic) -> Result<(), molrs::error::MolRsError> {
//! let conformer = Conformer::new(ConformerOptions::default());
//!
//! // An `Atomistic` in, an `Atomistic` out.
//! let (mol_3d, report) = conformer.generate(mol)?;
//! println!("{} atoms, {} stages", mol_3d.n_atoms(), report.stages.len());
//!
//! // A ported unit keeps its ports — then label the added hydrogens.
//! let (mut unit_3d, _report) = conformer.generate(unit)?;
//! let labelled = unit_3d.inherit_frag_ids();
//! println!("{labelled} of {} atoms inherited a frag_id", unit_3d.n_atoms());
//! # Ok(())
//! # }
//! ```

pub mod distgeom;
mod element_graph;
mod graph;
mod options;
mod report;

/// ETKDGv3 conformer-embedding pipeline (the active [`Conformer`] backend).
pub mod etkdg;

pub use element_graph::ElementGraph;
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

    /// Generate 3D coordinates for an element-bearing molecular graph.
    ///
    /// Generic over [`ElementGraph`], so **the returned type is the input
    /// type**: an [`Atomistic`] in, an `Atomistic` out. That bound is what the embedding actually requires — every node carries
    /// an `element`, which the pipeline reads for bond-length estimation, ring
    /// geometry and force-field selection. Coordinates are written in Å, and
    /// the input molecule is never modified.
    ///
    /// # What survives
    ///
    /// Every node, relation and property of the input graph, including its
    /// `ports` relations and `frag_id` properties: no stage of
    /// the pipeline removes a node, a relation or a property. The embed only
    /// appends hydrogen atoms and their bonds and writes `x` / `y` / `z`. The
    /// `ports` kind also rides through the MMFF staging `Frame` as an **unread
    /// block** — that frame is emitted one block per non-empty kind, and the
    /// potential compiler selects blocks by category name, so a `ports` block
    /// is carried across and never read.
    ///
    /// # `frag_id` on the hydrogens this adds
    ///
    /// With hydrogen addition on (the default) the output carries hydrogens
    /// that existed in no input fragment and therefore carry no `frag_id`.
    /// Labelling them is the caller's visible step, not a hidden hook inside
    /// this method: call `inherit_frag_ids` on the result, as the
    /// [module example](self) shows.
    ///
    /// # Cost
    ///
    /// Two clones of the graph where one existed: the promotion into the
    /// working [`Atomistic`] here, plus the clone hydrogen perception already
    /// made inside the pipeline. Both sit ahead of the distance-geometry solve
    /// that dominates the run, and neither is inside the retry loop; unwrapping
    /// and both promotions are moves.
    ///
    /// # Errors
    ///
    /// Returns [`MolRsError`] when the input graph is not element-bearing, when
    /// the embedding itself fails, and when the embedded graph cannot be
    /// promoted back into `M`.
    pub fn generate<M: ElementGraph>(&self, mol: &M) -> Result<(M, ConformerReport), MolRsError> {
        let work = Atomistic::try_from_molgraph(mol.as_molgraph().clone())?;
        let (out, report) = etkdg::generate_3d_impl(&work, &self.opts)?;
        Ok((M::try_from_molgraph(out.into_inner())?, report))
    }
}

impl Default for Conformer {
    fn default() -> Self {
        Self::new(ConformerOptions::default())
    }
}
