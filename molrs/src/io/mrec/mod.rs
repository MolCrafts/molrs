//! Scientific records (`*.mrec`), the MolRec record format.
//!
//! A **scientific record** is one self-describing package on disk: `meta`
//! plus at least one of a snapshot (`frame`), a topology (`system`), a
//! time-ordered frame sequence (`trajectory`), a force field (`forcefield`),
//! or a run `status`. On disk a record is a **directory** whose name
//! conventionally ends in `.mrec` (for example `water.mrec/`); inside, arrays
//! are stored with **Zarr V3** — an open format that cuts each array into
//! compressed chunks and records their shape in a small JSON document. The
//! Cargo feature that enables this module is still named `zarr` after that
//! encoding; the Zarr storage engine itself is crate-private. A closed
//! directory can be packed into a sibling `*.mrec.zip` archive.
//!
//! # The doors
//!
//! Whole records are read and written like every other format, by functions
//! of [`crate::io`]:
//!
//! - [`write_mrec_frame`](crate::io::write_mrec_frame) /
//!   [`read_mrec_frame`](crate::io::read_mrec_frame) — Structure (`meta` +
//!   `frame/`)
//! - [`write_mrec_system`](crate::io::write_mrec_system) /
//!   [`read_mrec_system`](crate::io::read_mrec_system) — System-def (`meta` +
//!   `system/`)
//! - [`write_mrec_trajectory`](crate::io::write_mrec_trajectory) /
//!   [`read_mrec_trajectory`](crate::io::read_mrec_trajectory) — Trajectory
//!   shape
//! - [`write_mrec_forcefield`](crate::io::write_mrec_forcefield) /
//!   [`read_mrec_forcefield`](crate::io::read_mrec_forcefield) — Force-field
//!   package (`meta` + `forcefield/`)
//! - [`write_mrec`](crate::io::write_mrec) / [`read_mrec`](crate::io::read_mrec)
//!   — a whole [`MolRec`], every section it holds;
//!   [`read_mrec_meta`](crate::io::read_mrec_meta) — the identity document
//! - [`write_mrec_storage`](crate::io::write_mrec_storage) /
//!   [`read_mrec_storage`](crate::io::read_mrec_storage) /
//!   [`read_mrec_frame_storage`](crate::io::read_mrec_frame_storage) — the same
//!   into / out of any open Zarr storage (in-memory, a host's), filesystem or
//!   not.
//!
//! # This module
//!
//! - A run too large to hold in memory: pin a [`SequenceSchema`], append with
//!   [`MrecWriter`], read one frame at a time with [`MrecReader`]
//!   ([`MrecReader::open`] opens a path, [`MrecReader::from_storage`] any open
//!   storage).
//! - [`section_names`] / [`section_names_storage`] — which sections a record
//!   holds.
//! - [`pack_mrec_zip`] / [`open_mrec_zip`] — collapse a closed directory into
//!   one `*.mrec.zip`, and open one for reading.
//! - [`ForceFieldSection`] — the `forcefield` section as data.
//! - [`validation`] — the runtime check of a record's frames.
//!   The language-neutral JSON Schema lives in molrec
//!   (`schema/core/record.schema.json`).
//!
//! The path-taking doors need the `filesystem` feature.
//!
//! Every writer creates the record root and its `meta/` group, written as the
//! producer supplied it; `meta` keys a reader does not recognise are kept as
//! they are. Identity of a store is the
//! `*.mrec/` path suffix plus a Zarr root.
//!
//! # Examples
//!
//! ```
//! # #[cfg(not(feature = "filesystem"))]
//! # fn main() {}
//! # #[cfg(feature = "filesystem")]
//! # fn main() -> Result<(), molrs::core::MolRsError> {
//! use molrs::io::{read_mrec_frame, write_mrec_frame};
//!
//! let dir = tempfile::tempdir().unwrap();
//! let path = dir.path().join("water.mrec");
//!
//! write_mrec_frame(&path, &molrs::core::Frame::new(), None, None)?;
//!
//! let loaded = read_mrec_frame(&path)?;
//! let sections = molrs::io::mrec::section_names(&path)?;
//! assert!(sections.iter().any(|s| s == "frame"));
//! let _ = loaded;
//! # Ok(())
//! # }
//! ```

mod forcefield_mapping;
pub(crate) mod forcefield_section;
mod record;
pub mod validation;
pub(crate) mod zarr_storage;

pub use forcefield_section::{EndpointKey, ForceFieldSection, SectionStyle, style_block_name};
pub use record::{MolRec, Observables};
pub use zarr_storage::{
    Compression, MrecReader, MrecWriter, SequenceSchema, dtype_from_schema_tag,
    section_names_storage,
};

#[cfg(feature = "filesystem")]
pub use zarr_storage::{open_mrec_zip, pack_mrec_zip, section_names};
