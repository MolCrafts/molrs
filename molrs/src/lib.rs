//! # molrs
//!
//! Unified molecular simulation toolkit. A single crate whose sub-systems are
//! feature-gated modules: `core` (always on) plus `io`, `compute`, `smiles`,
//! `ff`, `conformer`, and `signal`.
//!
//! ```toml
//! molcrafts-molrs = { version = "0.1", features = ["io", "smiles"] }
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

// Core is always compiled and its public surface is re-exported at the crate
// root, so `molrs::Frame`, `molrs::system::…`, `molrs::error::…` resolve exactly
// as they did when core was a separate crate.
pub mod core;
pub use crate::core::*;

#[cfg(feature = "io")]
pub mod io;

#[cfg(feature = "signal")]
pub mod signal;

#[cfg(feature = "compute")]
pub mod compute;

pub mod optimize;

#[cfg(feature = "ff")]
pub mod ff;

#[cfg(feature = "conformer")]
pub mod conformer;

// `smiles` is a sub-module of `io`; expose it at the top level for ergonomics.
#[cfg(feature = "smiles")]
pub use crate::io::smiles;
