//! Canonical molecular field-name constants.
//!
//! These are a re-export of [`crate::store::schema::consts`]. The schema is the
//! source of truth: a key is declared once, in its
//! [`ColumnSpec`](crate::store::schema::ColumnSpec), and the constant follows.
//!
//! This module used to hold the list itself, plus a `canonical_dtype` lookup
//! consulted by exactly one caller. Names and dtypes lived in separate tables
//! and nothing tied them together, so they could — and did — drift apart.
//!
//! # Examples
//!
//! ```
//! use molrs::store::keys;
//!
//! assert_eq!(keys::X, "x");
//! assert_eq!(keys::COORDS, [keys::X, keys::Y, keys::Z]);
//! ```

pub use crate::store::schema::consts::*;

// Frame meta keys. These name `frame.meta` entries, not block columns, so they
// sit outside the column schema.

/// Frame meta key: the atom-type inventory, packed as `"1:C,2:H"`.
///
/// Declares types no row needs to use; read by
/// [`TypeLabels`](crate::core::store::type_labels::TypeLabels).
pub const ATOM_TYPE_LABELS: &str = "atom_type_labels";
/// Frame meta key: the bond-type inventory, packed as `"1:c3-h1,2:c3-c3"`.
pub const BOND_TYPE_LABELS: &str = "bond_type_labels";
/// Frame meta key: the angle-type inventory, packed as `"id:label,…"`.
pub const ANGLE_TYPE_LABELS: &str = "angle_type_labels";
/// Frame meta key: the dihedral-type inventory, packed as `"id:label,…"`.
pub const DIHEDRAL_TYPE_LABELS: &str = "dihedral_type_labels";
/// Frame meta key: the improper-type inventory, packed as `"id:label,…"`.
pub const IMPROPER_TYPE_LABELS: &str = "improper_type_labels";
/// Frame meta key: the unit-preset name (`"real"`, `"lj"`, …) the frame's
/// numbers are in. Absent means the file or caller stated none.
///
/// The LAMMPS molecule-JSON reader writes it and its writer emits it back
/// (`io::data::lammps_molecule`); molrs converts no frame between presets.
pub const UNITS: &str = "units";

/// Canonical storage dtype for a key, if the vocabulary declares one.
///
/// Thin forwarder to [`crate::store::schema::column()`]. Unlike the old
/// hand-written table, this cannot disagree with what
/// [`Block::insert`](crate::store::block::Block::insert) enforces: both read the
/// same specs.
pub fn canonical_dtype(key: &str) -> Option<crate::store::block::DType> {
    crate::store::schema::column(key).map(|spec| spec.dtype)
}
