//! # molrs
//!
//! Unified façade for the molrs molecular simulation toolkit.
//!
//! This crate re-exports the molrs workspace crates under a single namespace.
//! Downstream users add one dependency and opt into sub-systems via features:
//!
//! ```toml
//! molcrafts-molrs = { version = "0.0.8", features = ["io", "smiles", "pack"] }
//! ```
//!
//! Then:
//!
//! ```ignore
//! use molrs::Frame;              // from molrs-core
//! use molrs::io::read_xyz;       // feature = "io"
//! use molrs::smiles::parse;      // feature = "smiles"
//! use molrs::pack::Molpack;      // feature = "pack"
//! ```
//!
//! ## Features
//!
//! - `io`       — file I/O (PDB, XYZ, LAMMPS, CHGCAR, Cube, Zarr)
//! - `compute`  — trajectory analysis (RDF, MSD, clustering, tensors)
//! - `smiles`   — SMILES parser
//! - `ff`       — force fields (MMFF94, PME, typifier)
//! - `gen3d`    — 3D coordinate generation
//! - `pack`     — molecular packing (Packmol port)
//! - `full`     — everything above
//!
//! Core flags forwarded to `molrs-core`: `rayon`, `zarr`, `filesystem`, `blas`.

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

#[cfg(feature = "gen3d")]
pub use molrs_gen3d as gen3d;

#[cfg(feature = "pack")]
pub use molrs_pack as pack;
