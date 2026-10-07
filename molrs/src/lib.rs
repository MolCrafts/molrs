//! # molrs
//!
//! Unified molecular simulation toolkit. A single crate whose sub-systems are
//! modules. Four are always compiled — `op`, `core`, `perceive` and
//! `optimize` (whose force-field optimizers need `ff`) — and the rest are
//! feature-gated: `builder`, `io`, `signal`, `compute`, `ff`, `md`, `smiles`,
//! `conformer`, and `stream`.
//!
//! ```toml
//! molcrafts-molrs = { version = "0.16", features = ["io", "smiles"] }
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
//! - `builder`   — structure builders and site-graph assembly
//! - `io`        — file I/O (PDB, XYZ, LAMMPS, CHGCAR, Cube, …; `*.mrec`
//!   record files need `zarr`, their path doors `filesystem`)
//! - `compute`   — trajectory analysis (RDF, MSD, clustering, tensors)
//! - `smiles`    — SMILES/SMARTS parser (lives in `io`) and the SMARTS
//!   matcher (`perceive::smarts`) compiled from it
//! - `ff`        — force fields (MMFF94, PME, typifier)
//! - `conformer` — 3D conformer generation
//! - `signal`    — signal processing (FFT-based ACF, windowing, frequency grids)
//! - `md`        — in-process molecular dynamics: integrators and force
//!   providers; the kernels are `ff::potential` (enables `ff`)
//! - `voronoi`   — radical Voronoi tessellation (enables `compute`)
//! - `full`      — everything above
//! - `stream`    — MessagePack/JSON frames and native WebSocket streaming (not in `full`)
//!
//! ## Paths
//!
//! Every public item has exactly one path: `molrs::<subsystem>::Item`, where
//! the subsystem is a top-level module (`core`, `ff`, `io`, …) or a namespace
//! its facade keeps (`ff::potential::pair`, `core::keys`, …). Implementation files are private
//! and their facade re-exports them; nothing is flattened to the crate root.
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
// `molrs::core::Frame`, `molrs::io::read_xyz`), matching how downstream code and
// doctests spell them. Sub-system modules below were absorbed from the former
// `molrs-*` member crates and rely on this alias for their cross-module paths.
extern crate self as molrs;

// Op is always compiled: the numeric base beneath core (vector, rigid-motion,
// linear-algebra kernels), plus the whole-graph transforms of
// `op`'s geometry functions (`op::translate`, …), the one part of `op` that acts on a core `MolGraph`.
pub mod op;

// Core is always compiled: the data model (Frame, Block, Trajectory), the
// molecular graph (MolGraph, Atomistic, Topology, Element), space (SimBox,
// regions, neighbour search), numerics and units, all flat on `molrs::core`,
// with the vocabularies `core::keys`, `core::schema` and `core::constants`.
pub mod core;

// Structure builders (graphene, nanotubes, self-avoiding walks, trace
// assembly, …). Builders sit above `core` and produce frames / paths without
// depending on feature-gated analysis or force fields; `full` includes them.
#[cfg(feature = "builder")]
pub mod builder;

// Chemical perception: one layer above `core`, below `ff` / `conformer`.
// Always compiled, except the SMARTS matcher (`perceive::smarts`), which
// compiles its queries from the one SMARTS parser in `io::smiles` and so needs
// the `smiles` feature.
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

/// In-process MD: integrators (velocity-Verlet, Langevin) and force providers.
/// The energy kernels are not here — they are in [`ff::potential`], consumed
/// through the one [`ff::potential::Potential`]/[`ff::potential::Potentials`]
/// seam (the `md` feature therefore enables `ff`). Required pieces go in the
/// constructor (`VelocityVerlet::new(dt, forces, mass, simbox)`); pair search
/// is core [`core::VerletSkin`]. Frame/`ForceField` wiring lives
/// in molpy / molrs-python.
#[cfg(feature = "md")]
pub mod md;

#[cfg(feature = "conformer")]
pub mod conformer;

// `serde::Serialize`/`Deserialize` for the core model (Frame/Block/Column/
// SimBox). Impls only; no public items. Enabled by `serde` (and by `stream`).
#[cfg(feature = "serde")]
mod serialize;

// Live `Frame` streaming: the transport encoding (MessagePack / JSON) over the
// `serde`-serializable core model, plus the WebSocket server and control
// commands that ride it. Kept out of `io` — and out of `full` — because it
// pulls third-party runtime dependencies that `io` must not acquire.
#[cfg(feature = "stream")]
pub mod stream;
