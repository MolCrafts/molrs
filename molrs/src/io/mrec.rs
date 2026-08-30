//! Public path and packing functions for scientific records (`*.mrec`).
//!
//! A **scientific record** is one self-describing [`crate::Record`]: metadata
//! plus at least one of a snapshot (`frame`), a topology (`system`), a
//! time-ordered frame sequence (`trajectory`), or a run `status`. This module
//! is how that in-memory object becomes a file, and how a file becomes that
//! object again.
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
//! ## What to call
//!
//! - Whole records: [`read_record_file`] / [`write_record_file`].
//! - A record whose only state is a trajectory:
//!   [`read_trajectory_file`] / [`write_trajectory_file`].
//! - A run too large to hold in memory: pin a [`SequenceSchema`], append with
//!   [`FrameSequenceWriter`], read one frame at a time with [`FrameSequence`].
//!   [`FrameSequence::open`] takes any already-open store (including an
//!   in-memory one). [`open_trajectory_sequence`] is the filesystem-path
//!   opener for that cursor.
//! - Pack a closed directory: [`pack`] / [`open_packed`]. Those two, the
//!   `*_file` functions, and [`open_trajectory_sequence`] need the
//!   `filesystem` feature.
//!
//! Every writer writes [`crate::RECORD_FORMAT_NAME`] (`"mrec"`) and
//! [`crate::RECORD_SCHEMA_VERSION`] (`1`) into `meta`. A reader rejects any
//! other `format_name` — the format's identifying name — including the retired
//! spelling `"molrec"`, and any other schema version.
//!
//! # Examples
//!
//! ```
//! # #[cfg(not(feature = "filesystem"))]
//! # fn main() {}
//! # #[cfg(feature = "filesystem")]
//! # fn main() -> Result<(), molrs::MolRsError> {
//! use molrs::io::mrec::{read_record_file, write_record_file};
//!
//! let dir = tempfile::tempdir().unwrap();
//! let path = dir.path().join("water.mrec");
//!
//! let mut record = molrs::Record::new();
//! record.frame = Some(molrs::Frame::new());
//! write_record_file(&path, &record)?;
//!
//! let loaded = read_record_file(&path)?;
//! assert_eq!(loaded.meta["format_name"].as_str(), Some("mrec"));
//! assert_eq!(loaded.meta["record_schema_version"].as_u64(), Some(1));
//! # Ok(())
//! # }
//! ```

#[doc(inline)]
pub use super::zarr::{FrameSequence, FrameSequenceWriter, SequenceSchema};

#[cfg(feature = "filesystem")]
#[doc(inline)]
pub use super::zarr::{
    open_trajectory_sequence, read_record_file, read_trajectory_file, write_record_file,
    write_trajectory_file,
};

#[cfg(feature = "filesystem")]
#[doc(inline)]
pub use super::zarr::{open_packed, pack};
