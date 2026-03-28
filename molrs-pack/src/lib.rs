//! `molrs-pack` — faithful Rust port of Packmol molecular packing.
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use molrs_pack::{InsideBoxConstraint, Molpack, Target};
//!
//! let water_positions = [
//!     [0.0, 0.0, 0.0],
//!     [0.96, 0.0, 0.0],
//!     [-0.24, 0.93, 0.0],
//! ];
//! let water_radii = [1.52, 1.20, 1.20];
//! let box_constraint = InsideBoxConstraint::new([0.0, 0.0, 0.0], [40.0, 40.0, 40.0]);
//! let target = Target::from_coords(&water_positions, &water_radii, 500)
//!     .with_name("water")
//!     .with_constraint(box_constraint);
//!
//! let mut packer = Molpack::new().tolerance(2.0).precision(0.01);
//! let result = packer.pack(&[target], 200, Some(42)).unwrap();
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
mod numerics;
pub mod objective;
pub mod packer;
mod random;
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
