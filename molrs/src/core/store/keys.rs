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

/// Canonical storage dtype for a key, if the vocabulary declares one.
///
/// Thin forwarder to [`crate::store::schema::column`]. Unlike the old
/// hand-written table, this cannot disagree with what
/// [`Block::insert`](crate::store::block::Block::insert) enforces: both read the
/// same specs.
pub fn canonical_dtype(key: &str) -> Option<crate::store::block::DType> {
    crate::store::schema::column(key).map(|spec| spec.dtype)
}
