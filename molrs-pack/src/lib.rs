//! `molrs-pack` — faithful Rust port of Packmol molecular packing.
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use molrs_pack::{Target, Molpack, MoleculeConstraint, Restraint, InsideBoxConstraint};
//!
//! let c = InsideBoxConstraint::new([0.,0.,0.], [40.,40.,40.]);
//! let target = Target::new(&water_positions, &water_radii, 500)
//!     .with_name("water")
//!     .with_constraint(c);
//!
//! let result = Molpack::new()
//!     .pack(&[target], 20, Some(42))?;
//! ```

pub mod api;
pub mod cases;
pub mod cell;
pub mod constraint;
pub mod constraints;
pub mod context;
pub mod error;
pub mod euler;
pub mod frame;
pub mod gencan;
pub mod handler;
pub mod hook;
pub mod initial;
pub mod movebad;
pub mod objective;
pub mod packer;
pub mod target;
pub mod validation;

pub use cases::{ExampleCase, build_targets, example_dir_from_manifest, render_packmol_input};
pub use constraint::{
    AbovePlaneConstraint, AtomConstraint, BelowPlaneConstraint, InsideBoxConstraint,
    InsideSphereConstraint, MoleculeConstraint, OutsideSphereConstraint, RegionConstraint,
    Restraint,
};
pub use context::PackContext;
pub use error::PackError;
pub use frame::{compute_mol_ids, context_to_frame, finalize_frame, frame_to_coords};
pub use handler::{
    EarlyStopHandler, Handler, NullHandler, PhaseInfo, ProgressHandler, StepInfo, XYZHandler,
};
pub use hook::{Hook, HookRunner, TorsionMcHook, compute_excluded_pairs, self_avoidance_penalty};
pub use molrs::Element;
pub use molrs::types::F;
pub use packer::{Molpack, PackResult};
pub use target::{CenteringMode, Target};
pub use validation::{ValidationReport, ViolationMetrics, validate_from_targets};
