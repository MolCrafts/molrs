//! # molrs
//!
//! Unified molecular simulation toolkit. A single crate whose sub-systems are
//! feature-gated modules: `core` (always on) plus `io`, `compute`, `smiles`,
//! `ff`, `conformer`, and `signal`.
//!
//! ```toml
//! molcrafts-molrs = { version = "0.12", features = ["io", "smiles"] }
//! ```
//!
//! Then:
//!
//! ```ignore
//! use molrs::Frame;              // core (always available)
//! use molrs::io::read_xyz;       // feature = "io"
//! use molrs::smiles::parse;      // feature = "smiles"
//! ```
//!
//! ## Features
//!
//! - `io`        — file I/O (PDB, XYZ, LAMMPS, CHGCAR, Cube, Zarr)
//! - `compute`   — trajectory analysis (RDF, MSD, clustering, tensors)
//! - `smiles`    — SMILES/SMARTS parser (lives in `io`)
//! - `ff`        — force fields (MMFF94, PME, typifier)
//! - `conformer` — 3D conformer generation
//! - `signal`    — signal processing (FFT-based ACF, windowing, frequency grids)
//! - `full`      — everything above
//! - `stream`    — MessagePack/JSON `Frame` wire encoding (not in `full`)
//! - `net`       — WebSocket Frame streaming + control commands (not in `full`)
//!
//! Core flags: `rayon` (default), `zarr`, `filesystem`, `blas`.
//!
//! ## Molecular packing
//!
//! The Packmol port lives in the standalone `molcrafts-molpack` crate
//! (<https://github.com/MolCrafts/molpack>); add it as a separate dependency
//! when needed.

#![warn(rustdoc::missing_crate_level_docs)]

// Let in-crate paths refer to this crate by its public name `molrs::` (e.g.
// `molrs::Frame`, `molrs::io::read_xyz`), matching how downstream code and
// doctests spell them. Sub-system modules below were absorbed from the former
// `molrs-*` member crates and rely on this alias for their cross-module paths.
extern crate self as molrs;

/// The version of the `molcrafts-molrs` crate compiled into this binary.
///
/// This is the crate every binder statically links, so its major.minor is the
/// ABI line of any FFI handle the binary mints — `molrs_ffi::abi` derives the
/// versioned capsule names and the handshake token from it. Downstream pins
/// major.minor only; layout of the FFI-crossing types is frozen within a minor
/// line (enforced by `molrs-ffi/tests/layout_snapshot.rs`).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

// Core is always compiled and its public surface is re-exported at the crate
// root, so `molrs::Frame`, `molrs::system::…`, `molrs::error::…` resolve exactly
// as they did when core was a separate crate.
pub mod core;
pub use crate::core::system::element::Element;
pub use crate::core::*;

/// Structure builders (graphene, nanotubes, self-avoiding walks, …).
///
/// Always compiled — builders sit above `core` and produce frames / paths
/// without depending on feature-gated analysis or force fields.
pub mod builder;
pub use crate::builder::{
    CarbonTubeBuilder, CarbonTubeError, FccLattice, GrapheneBuilder, GrapheneError, GrowthStrategy,
    OccupancyMode, OffLattice, SelfAvoidingWalk, WalkError, WalkOutput,
};

// Chemical perception: one layer above `core`, below `ff` / `io` / `conformer`.
// Always compiled — every consumer configuration already compiled these modules
// when they lived inside `core`, so keeping them unconditional reproduces the
// existing build graph exactly (feature-gating them would be a behaviour change,
// not a refactor). `optimize` above is the same shape: always on, no feature.
pub mod perceive;

// The crate-root surface that this layer used to publish via `pub use core::*`.
// It moves here verbatim, retargeted at `perceive`, so `molrs::find_rings`,
// `molrs::SmartsPattern`, `molrs::add_hydrogens`, … keep resolving. Deleting it
// would silently break 13 call sites — four of them in `ff/`, two of those only
// visible under `clippy -D warnings` as broken intra-doc links.
pub use crate::perceive::aromaticity::perceive_aromaticity;
pub use crate::perceive::hydrogens::{add_hydrogens, implicit_h_count, remove_hydrogens};
pub use crate::perceive::rings::{RingInfo, find_rings, max_ring_system_size};
pub use crate::perceive::smarts::{
    MatchOptions, Reaction, RingPrimitive, SmartsMatch, SmartsPattern,
};
pub use crate::perceive::stereo::{
    BondStereo, TetrahedralStereo, assign_bond_stereo_from_3d, assign_stereo_from_3d,
    chiral_volume, find_chiral_centers,
};

#[cfg(feature = "io")]
pub mod io;

#[cfg(feature = "signal")]
pub mod signal;

#[cfg(feature = "compute")]
pub mod compute;

// Force fields first: `optimize` depends on `ff::potential::Potential`.
#[cfg(feature = "ff")]
pub mod ff;

/// Geometry optimizers over [`ff::potential::Potential`].
///
/// Gated on `ff` — the optimizer depends on the force-field potential trait,
/// never the reverse.
#[cfg(feature = "ff")]
pub mod optimize;

/// In-process MD: velocity-Verlet / Langevin and shifted LJ.
/// Gated independently of `ff` — required pieces go in the constructor
/// (`VelocityVerlet::new(dt, potential, neighbors, mass)`); pair search is
/// core [`spatial::neighbors::VerletSkin`]. Frame/`ForceField` wiring lives
/// in molpy / molrs-python.
#[cfg(feature = "md")]
pub mod md;

/// Gasteiger/PEOE partial charges, at the crate root — `molrs::compute_gasteiger_charges`.
///
/// The name predates the charge models and the binders still reach for it here, so it
/// keeps resolving; it is a re-export of the **one** Gasteiger in the tree
/// ([`ff::charge::GasteigerModel`], `antechamber -c gas`), not a second one. It moved
/// out of `perceive` because a charge model belongs with the charge models, which is
/// also why it is now gated on `ff` — the layer that owns `GASPARM.DAT`.
#[cfg(feature = "ff")]
pub use crate::ff::charge::compute_gasteiger_charges;

#[cfg(feature = "conformer")]
pub mod conformer;

// `serde::Serialize`/`Deserialize` for the core model (Frame/Block/Column/
// SimBox). Impls only; no public items. Enabled by `serde` (and by `stream`).
#[cfg(feature = "serde")]
mod serialize;

/// Live `Frame` streaming: the transport encoding (MessagePack / JSON) over the
/// `serde`-serializable core model, plus the WebSocket server and control
/// commands that ride it. Kept out of `io` — and out of `full` — because it
/// pulls third-party runtime dependencies that `io` must not acquire.
#[cfg(feature = "stream")]
pub mod stream;

// `smiles` is a sub-module of `io`; expose it at the top level for ergonomics.
#[cfg(feature = "smiles")]
pub use crate::io::smiles;
