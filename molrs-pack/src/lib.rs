//! # molrs-pack
//!
//! Packmol-grade molecular packing in pure Rust. Produces a non-overlapping
//! arrangement of N molecule types with copy counts and geometric restraints,
//! using a faithful port of Packmol's GENCAN-driven three-phase algorithm
//! (Martínez et al. 2009). Correctness is checked against Packmol's reference
//! output for five canonical workloads.
//!
//! ## Documentation map
//!
//! This crate is documented in four dedicated modules; start with
//! [`getting_started`] if you are new.
//!
//! - [`getting_started`] — install, hello-world packing, the three
//!   restraint scopes, handlers, relaxers, PBC, running the canonical
//!   examples.
//! - [`concepts`] — every abstraction defined in one place: `Restraint`,
//!   `Region`, `Relaxer`, `Handler`, `Objective`, `Target`, `Molpack`,
//!   `PackContext`; the scope equivalence law; the two-scale contract;
//!   the direction-3 extension pattern.
//! - [`architecture`] — module map, dependency graph, core-type
//!   relationships, full `pack()` lifecycle diagram, hot-path
//!   `evaluate()` walkthrough, invariants, design decisions.
//! - [`extending`] — tutorials for writing your own `Restraint` /
//!   `Region` / `Handler` / `Relaxer`; testing + benchmarking
//!   discipline; common pitfalls; contributing flow.
//!
//! Reference material (not rustdoc):
//!
//! - [Packmol alignment](https://github.com/MolCrafts/molrs/blob/master/molrs-pack/docs/packmol_alignment.md)
//!   — kind-number ↔ Rust struct mapping with Fortran pointers.
//! - Spec: `.claude/specs/molrs-pack-plugin-arch.md` (in the repo root's
//!   `.claude` dir) — full design document with performance gates and
//!   risk register.
//!
//! ## Quick example
//!
//! ```rust,no_run
//! use molrs_pack::{InsideBoxRestraint, Molpack, Target};
//!
//! let positions = [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]];
//! let radii     = [1.52, 1.20, 1.20];
//!
//! let target = Target::from_coords(&positions, &radii, 100)
//!     .with_name("water")
//!     .with_restraint(InsideBoxRestraint::new([0.0; 3], [40.0, 40.0, 40.0]));
//!
//! let result = Molpack::new()
//!     .tolerance(2.0)
//!     .precision(0.01)
//!     .pack(&[target], 200, Some(42))?;
//!
//! println!("converged = {}, natoms = {}", result.converged, result.natoms());
//! # Ok::<(), molrs_pack::PackError>(())
//! ```
//!
//! ## Public surface at a glance
//!
//! | Category | Items |
//! |---|---|
//! | Builder | [`Molpack`], [`PackResult`] |
//! | Target  | [`Target`], [`CenteringMode`] |
//! | Restraint trait + 14 concrete structs | [`Restraint`] + `InsideBox` / `InsideCube` / `InsideSphere` / `InsideEllipsoid` / `InsideCylinder` / `Outside*` variants / `AbovePlane` / `BelowPlane` / `AboveGaussian` / `BelowGaussian` — each suffixed `…Restraint` |
//! | Region trait + combinators + lift | [`Region`], [`RegionExt`], [`And`], [`Or`], [`Not`], [`FromRegion`], [`InsideBoxRegion`], [`InsideSphereRegion`], [`OutsideSphereRegion`], [`BBox`] |
//! | Handler trait + built-ins | [`Handler`], [`NullHandler`], [`ProgressHandler`], [`EarlyStopHandler`], [`XYZHandler`], [`StepInfo`], [`PhaseInfo`], [`PhaseReport`] |
//! | Relaxer trait + built-in | [`Relaxer`], [`RelaxerRunner`], [`TorsionMcRelaxer`] (aliased to `TorsionMcHook`) |
//! | Errors | [`PackError`] |
//! | Validation | [`validate_from_targets`], [`ValidationReport`], [`ViolationMetrics`] |
//! | Examples harness | [`ExampleCase`], [`build_targets`], [`example_dir_from_manifest`], [`render_packmol_input`] |
//!
//! ## Feature flags
//!
//! This crate inherits the workspace-wide `f64` / `f32` precision flag via
//! `molrs::types::F`. There are no pack-specific feature flags.

pub mod api;
pub mod cases;
pub mod cell;
pub mod constraints;
pub mod context;
pub mod error;
pub mod euler;
pub mod frame;
pub mod gencan;
pub mod handler;
pub mod initial;
pub mod movebad;
mod numerics;
pub mod objective;
pub mod packer;
mod random;
pub mod region;
pub mod relaxer;
pub mod restraint;
pub mod target;
pub mod validation;

pub use cases::{ExampleCase, build_targets, example_dir_from_manifest, render_packmol_input};
pub use context::PackContext;
pub use error::PackError;
pub use frame::{compute_mol_ids, context_to_frame, finalize_frame, frame_to_coords};
pub use handler::{
    EarlyStopHandler, Handler, NullHandler, PhaseInfo, PhaseReport, ProgressHandler, StepInfo,
    XYZHandler,
};
pub use molrs::Element;
pub use molrs::types::F;
pub use packer::{Molpack, PackResult};
pub use region::{
    And, BBox, FromRegion, InsideBoxRegion, InsideSphereRegion, Not, Or, OutsideSphereRegion,
    Region, RegionExt,
};
pub use relaxer::Relaxer as Hook;
pub use relaxer::RelaxerRunner as HookRunner;
pub use relaxer::TorsionMcRelaxer as TorsionMcHook;
pub use relaxer::{
    Relaxer, RelaxerRunner, TorsionMcRelaxer, compute_excluded_pairs, self_avoidance_penalty,
};
pub use restraint::{
    AboveGaussianRestraint, AbovePlaneRestraint, BelowGaussianRestraint, BelowPlaneRestraint,
    InsideBoxRestraint, InsideCubeRestraint, InsideCylinderRestraint, InsideEllipsoidRestraint,
    InsideSphereRestraint, OutsideBoxRestraint, OutsideCubeRestraint, OutsideCylinderRestraint,
    OutsideEllipsoidRestraint, OutsideSphereRestraint, Restraint,
};
pub use target::{CenteringMode, Target};
pub use validation::{ValidationReport, ViolationMetrics, validate_from_targets};

// ────────────────────────────────────────────────────────────────────────────
// Documentation modules (rustdoc-only; no runtime items).
// Content lives in `docs/*.md`, loaded via `include_str!` so each markdown
// file can be edited independently while rustdoc renders the whole chapter.
// ────────────────────────────────────────────────────────────────────────────

#[doc = include_str!("../docs/getting_started.md")]
pub mod getting_started {}

#[doc = include_str!("../docs/concepts.md")]
pub mod concepts {}

#[doc = include_str!("../docs/architecture.md")]
pub mod architecture {}

#[doc = include_str!("../docs/extending.md")]
pub mod extending {}
