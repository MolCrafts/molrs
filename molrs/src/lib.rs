//! # molrs
//!
//! Unified molecular simulation toolkit. A single crate whose sub-systems are
//! modules. Four are always compiled — `op`, `core`, `perceive` and
//! `optimize` (whose force-field optimizers need `ff`) — and the rest are
//! feature-gated: `builder`, `io`, `signal`, `compute`, `ff`, `md`, `smiles`,
//! `conformer`, and `stream`.
//!
//! ```toml
//! molcrafts-molrs = { version = "0.15", features = ["io", "smiles"] }
//! ```
//!
//! Then:
//!
//! ```
//! # #[cfg(feature = "smiles")]
//! # {
//! use molrs::io::smiles::{parse_smiles, to_atomistic};
//!
//! let ir = parse_smiles("CCO")?;
//! let molecule = to_atomistic(&ir)?;
//! assert_eq!(molecule.n_atoms(), 3);
//! # }
//! # Ok::<(), Box<dyn std::error::Error>>(())
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
//! - `md`        — in-process molecular dynamics (enables `ff`)
//! - `voronoi`   — radical Voronoi tessellation (enables `compute`)
//! - `full`      — everything above
//! - `stream`    — MessagePack/JSON frames and native WebSocket streaming (not in `full`)
//!
//! Default: core only, plus `rayon`. Every sub-system is opt-in; name the
//! ones you use, or `full` for all of them. `default-features = false` also
//! drops `rayon` (wasm, Pyodide).
//! Storage and compute flags: `serde`, `rayon`, `zarr`, `zarr-codecs`,
//! `filesystem`.
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
/// line.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

// Op is always compiled: the numeric base beneath core (vector, rigid-motion,
// linear-algebra kernels); it names no other molrs module.
pub mod op;

// Core is always compiled and its public surface is re-exported at the crate
// root, so `molrs::Frame`, `molrs::system::…`, `molrs::error::…` resolve exactly
// as they did when core was a separate crate.
pub mod core;
pub use crate::core::system::element::Element;
pub use crate::core::*;

/// Structure builders (graphene, nanotubes, self-avoiding walks, trace
/// assembly, …).
///
/// Builders sit above `core` and produce frames / paths without depending on
/// feature-gated analysis or force fields; `full` includes them.
#[cfg(feature = "builder")]
pub mod builder;
#[cfg(feature = "builder")]
pub use crate::builder::{
    AssembleError, Assembler, AxisOrienter, CarbonTubeBuilder, CarbonTubeError, FccLattice,
    GrapheneBuilder, GrapheneError, GrowthPlacer, GrowthStrategy, OccupancyMode, OffLattice,
    OrientError, Orienter, ParentJoin, PlaceError, PlaceSite, Placer, SelfAvoidingWalk, SiteLink,
    SitePlacer, SiteView, WalkError, WalkOutput,
};

// Chemical perception: one layer above `core`, below `ff` / `io` / `conformer`.
// Always compiled — every consumer configuration already compiled these modules
// when they lived inside `core`, so keeping them unconditional reproduces the
// existing build graph exactly (feature-gating them would be a behaviour change,
// not a refactor).
pub mod perceive;

#[cfg(feature = "io")]
pub mod io;

#[cfg(feature = "signal")]
pub mod signal;

#[cfg(feature = "compute")]
pub mod compute;

#[cfg(feature = "ff")]
pub mod ff;

// Geometry optimization; always compiled (its module docs say what needs `ff`).
pub mod optimize;

/// In-process MD: velocity-Verlet / Langevin and shifted Lennard-Jones.
/// Consumes the one [`ff::potential::Potential`]/[`ff::potential::Potentials`]
/// seam (the `md` feature therefore enables `ff`) — required pieces go in the
/// constructor (`VelocityVerlet::new(dt, potential, neighbors, mass)`); pair
/// search is core [`spatial::neighbors::VerletSkin`]. Frame/`ForceField`
/// wiring lives in molpy / molrs-python.
#[cfg(feature = "md")]
pub mod md;

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
