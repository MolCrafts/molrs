//! Public path and packing functions for scientific records (`*.mrec`).
//!
//! A **scientific record** is one self-describing package on disk: `meta`
//! plus at least one of a snapshot (`frame`), a topology (`system`), a
//! time-ordered frame sequence (`trajectory`), a force field (`forcefield`),
//! or a run `status`. This module is how a [`crate::store::Frame`],
//! [`crate::store::Trajectory`] or [`crate::store::ForceFieldSection`] becomes that
//! package, and how the package becomes those objects again.
//!
//! The in-memory codec working set that holds every section at once is
//! crate-internal. Callers write the object they have:
//!
//! - [`write_frame_file`] / [`read_frame_file`] — Structure (`meta` + `frame/`)
//! - [`write_system_file`] / [`read_system_file`] — System-def (`meta` + `system/`)
//! - [`write_trajectory_file`] / [`read_trajectory_file`] — Trajectory shape
//! - [`write_forcefield_file`] / [`read_forcefield_file`] — Force-field
//!   package (`meta` + `forcefield/`)
//!
//! On disk a record is a **directory** whose name conventionally ends in
//! `.mrec` (for example `water.mrec/`). Inside, arrays are stored with
//! **Zarr V3** — an open format that cuts each array into compressed chunks
//! and records their shape in a small JSON document. The Cargo feature that
//! enables this module is still named `zarr` after that encoding; the Zarr
//! adapter itself is crate-private. A closed directory can be packed into a
//! sibling `*.mrec.zip` archive. Paths ending in `.zarr` or `.zarr.zip` are
//! refused: those were the previous scientific suffixes and are not migrated.
//!
//! [`schema`] is the runtime check for path suffix and `meta` brand keys.
//! The language-neutral JSON Schema lives in molrec
//! (`schema/core/record.schema.json`).
//!
//! ## What to call
//!
//! - A snapshot: [`read_frame_file`] / [`write_frame_file`].
//! - A topology: [`read_system_file`] / [`write_system_file`].
//! - A trajectory: [`read_trajectory_file`] / [`write_trajectory_file`].
//! - A force field: [`read_forcefield_file`] / [`write_forcefield_file`], or
//!   [`crate::store::MolRec::forcefield`] beside other sections through
//!   [`write_record_file`].
//! - A run too large to hold in memory: pin a [`SequenceSchema`], append with
//!   [`MrecWriter`], read one frame at a time with [`MrecReader`].
//!   [`MrecReader::open`] takes any already-open store (including an
//!   in-memory one). [`open_trajectory_sequence`] is the filesystem-path
//!   opener for that cursor.
//! - Pack a closed directory: [`pack`] / [`open_packed`]. Those two, the
//!   `*_file` functions, and [`open_trajectory_sequence`] need the
//!   `filesystem` feature.
//!
//! Every writer creates the record root and its `meta/` group, and stamps
//! [`MOLREC_VERSION`](crate::store::MOLREC_VERSION) there over whatever the producer supplied — so
//! every record written by this version carries the contract it was written
//! in.
//!
//! Readers validate the key when it is present: it must be an integer `>= 1`
//! no newer than [`MOLREC_VERSION`](crate::store::MOLREC_VERSION). A version-1 store (molrs ≤ 0.15),
//! and one without the key (written before version 1), is converted on read —
//! the force-field IR's numbers whose meaning changed in version 2 are
//! re-expressed exactly, or the store is refused (the reader's own version-1
//! conversion, which needs no `ff`). `meta` comes back as stored. A version-1
//! trajectory is read, never appended to.
//!
//! Identity of a store is the `*.mrec/` path suffix plus a Zarr root, not this
//! key; the key says which contract wrote it.
//!
//! # Examples
//!
//! ```
//! # #[cfg(not(feature = "filesystem"))]
//! # fn main() {}
//! # #[cfg(feature = "filesystem")]
//! # fn main() -> Result<(), molrs::error::MolRsError> {
//! use molrs::io::mrec::{read_frame_file, write_frame_file};
//!
//! let dir = tempfile::tempdir().unwrap();
//! let path = dir.path().join("water.mrec");
//!
//! write_frame_file(&path, &molrs::store::Frame::new(), None, None)?;
//!
//! let loaded = read_frame_file(&path)?;
//! let sections = molrs::io::mrec::section_names(&path)?;
//! assert!(sections.iter().any(|s| s == "frame"));
//! let _ = loaded;
//! # Ok(())
//! # }
//! ```

#[doc(inline)]
pub use super::zarr::{Compression, MrecReader, MrecWriter, SequenceSchema, column_dtype};

/// Runtime validation of the mrec record schema (path suffix, `meta` brand).
#[doc(inline)]
pub use super::zarr::schema;

#[cfg(feature = "filesystem")]
#[doc(inline)]
pub use super::zarr::{
    open_trajectory_sequence, read_forcefield_file, read_frame_file, read_meta_file,
    read_record_file, read_system_file, read_trajectory_file, section_names, write_forcefield_file,
    write_frame_file, write_record_file, write_system_file, write_trajectory_file,
};

#[cfg(feature = "filesystem")]
#[doc(inline)]
pub use super::zarr::{open_packed, pack};

/// The store-taking record doors: a whole record into / out of any open
/// store, filesystem or not.
#[doc(inline)]
pub use super::zarr::{
    read_frame_section_store, read_record_store, section_names_store, write_record_store,
};
