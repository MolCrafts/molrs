//! Column requirements for [`Neighbors`] tables — the one place a compute
//! kernel turns "this table never stored that column" into a
//! [`ComputeError::BadShape`].
//!
//! A [`Neighbors`] table is the list of particle pairs within a cutoff
//! distance, one row per pair. Besides the two particle indices it may carry
//! two *physical* columns, both taken under the **minimum-image convention**
//! (with periodic boundaries a pair has one separation per periodic copy of the
//! box; only the shortest is ever reported): `dist_sq`, the squared pair
//! distance `|r_j − r_i|²` in Å², and `disp`, the displacement `r_j − r_i`
//! itself in Å, pointing from `i` to `j` and left unnormalized. Which of them
//! a table holds is decided when it is materialized, by
//! [`NeighborsStorage`](molrs::spatial::neighbors::NeighborsStorage).
//!
//! A table reports a column it never stored as `None`, never as a fabricated
//! zero: `disp()` on an indices-only table is `None`, not a view full of
//! `(0, 0, 0)`, because a zero displacement means two coincident particles and
//! would answer confidently and wrongly. Every kernel that needs a column
//! therefore has to reject `None` itself, and each rejection carries the same
//! two facts — which column was missing, and how many pairs went unanswered
//! because of it.
//!
//! Both helpers are free functions rather than methods on [`Neighbors`]: the
//! owning type lives in `core`, one layer below this one, and must not learn
//! about [`ComputeError`] to answer a question `core` never asks.

use molrs::spatial::neighbors::Neighbors;
use molrs::types::{F, FNx3View};

use super::error::ComputeError;

/// The minimum-image displacement column (Å), or [`ComputeError::BadShape`].
///
/// On success the view has exactly [`n_pairs()`](Neighbors::n_pairs) rows, so a
/// caller may index rows `0..n_pairs` without further checks — which is the
/// point: reading an absent column would otherwise index an empty buffer (an
/// out-of-bounds panic natively, a `unreachable` trap under WASM). Row `k` is
/// `(dx, dy, dz)` for pair `k` in Å, pointing from `i` to `j` and
/// **unnormalized**: its length is the pair distance.
///
/// # Errors
///
/// [`ComputeError::BadShape`] when the table was materialized without
/// [`NeighborsStorage::disp`](molrs::spatial::neighbors::NeighborsStorage::disp)
/// — that is, with any policy other than `DISP` or `FULL`. The `expected` text
/// names the missing column and the pair count it was needed for, and `got`
/// describes the table that arrived; nothing is substituted for the column.
pub(crate) fn require_disp(nlist: &Neighbors) -> Result<FNx3View<'_>, ComputeError> {
    nlist
        .disp()
        .ok_or_else(|| missing_column("disp", nlist.n_pairs()))
}

/// The squared-distance column (Å²), or [`ComputeError::BadShape`].
///
/// On success the slice has exactly [`n_pairs()`](Neighbors::n_pairs) elements,
/// aligned row-for-row with the index columns. Element `k` is the squared
/// minimum-image distance of pair `k` in Å²; the square root is left to the
/// caller, since most consumers compare against a squared cutoff and never need
/// it.
///
/// # Errors
///
/// [`ComputeError::BadShape`] when the table was materialized without
/// [`NeighborsStorage::dist_sq`](molrs::spatial::neighbors::NeighborsStorage::dist_sq)
/// — that is, with any policy other than `DIST_SQ` or `FULL`. The `expected`
/// text names the missing column and the pair count it was needed for, and
/// `got` describes the table that arrived.
pub(crate) fn require_dist_sq(nlist: &Neighbors) -> Result<&[F], ComputeError> {
    nlist
        .dist_sq()
        .ok_or_else(|| missing_column("dist_sq", nlist.n_pairs()))
}

/// The shared error text: `expected` names the column *and* the pair count it
/// was needed for (a table with zero pairs is a different, harmless story),
/// `got` describes the table that arrived.
fn missing_column(column: &'static str, n_pairs: usize) -> ComputeError {
    ComputeError::BadShape {
        expected: format!("Neighbors with the {column} column for {n_pairs} pairs"),
        got: "indices-only / lean neighbor table".to_string(),
    }
}
