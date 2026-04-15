//! # molrs
//!
//! Unified façade for the molrs molecular simulation toolkit.
//!
//! This crate re-exports the molrs workspace crates under a single namespace.
//! Downstream users add one dependency and opt into sub-systems via features:
//!
//! ```toml
//! molcrafts-molrs = { version = "0.0.8", features = ["io", "smiles"] }
//! ```
//!
//! Then:
//!
//! ```ignore
//! use molrs::Frame;              // from molrs-core
//! use molrs::io::read_xyz;       // feature = "io"
//! use molrs::smiles::parse;      // feature = "smiles"
//! ```
//!
//! ## Features
//!
//! - `io`       — file I/O (PDB, XYZ, LAMMPS, CHGCAR, Cube, Zarr)
//! - `compute`  — trajectory analysis (RDF, MSD, clustering, tensors)
//! - `smiles`   — SMILES parser
//! - `ff`       — force fields (MMFF94, PME, typifier)
//! - `embed`    — 3D coordinate generation
//! - `full`     — everything above
//!
//! Core flags forwarded to `molrs-core`: `rayon`, `zarr`, `filesystem`, `blas`.
//!
//! ## Molecular packing
//!
//! The Packmol port lives in the standalone `molcrafts-molpack` crate
//! (<https://github.com/MolCrafts/molpack>); add it as a separate dependency
//! when needed.

#![warn(rustdoc::missing_crate_level_docs)]

// Core types at the top level (Frame, Block, MolGraph, SimBox, Element, …).
pub use molrs::*;

#[cfg(feature = "io")]
pub use molrs_io as io;

#[cfg(feature = "compute")]
pub use molrs_compute as compute;

#[cfg(feature = "smiles")]
pub use molrs_smiles as smiles;

#[cfg(feature = "ff")]
pub use molrs_ff as ff;

#[cfg(feature = "embed")]
pub use molrs_embed as embed;
