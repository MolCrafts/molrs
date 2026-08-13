//! Shared `#[cfg(test)]` neighbor-list construction for the compute unit tests.
//!
//! The per-kernel test modules used to pack the frame's separate `x`/`y`/`z`
//! columns into an owned [`Array2`](ndarray::Array2) and build a table from a
//! backend directly. This helper replaces that boilerplate with the engine's
//! column entry point ([`get_positions_ref`] → [`NeighborList::build_columns`]),
//! reading the three columns without an interleaved copy. Same pairs as the
//! `N × 3` path (proven by `engine_build_columns_matches_build`), so every
//! rewired test produces identical results.

use molrs::spatial::neighbors::{NeighborList, Neighbors, NeighborsStorage};
use molrs::store::frame_access::FrameAccess;
use molrs::types::F;

use super::util::get_positions_ref;

/// Build a self-query [`Neighbors`] table for `frame` at `cutoff` through the
/// engine, using the frame's own simulation box.
///
/// Reads the `atoms` block's `x`/`y`/`z` columns with [`get_positions_ref`] (a
/// zero-copy borrow for contiguous columns) and hands them to
/// [`NeighborList::build_columns`], then materializes every column
/// ([`NeighborsStorage::FULL`]) because the kernels under test read distances
/// and displacements alike.
///
/// Row order is unspecified — the cell-list backend materializes in parallel —
/// so a test that needs a fixed order sorts.
///
/// # Panics
/// Panics if the frame lacks `x`/`y`/`z` columns or has no simulation box.
pub(crate) fn nlist_from_frame<FA: FrameAccess>(frame: &FA, cutoff: F) -> Neighbors {
    let (xs, ys, zs) = get_positions_ref(frame).expect("frame must expose x/y/z columns");
    let simbox = frame
        .simbox_ref()
        .expect("frame must have a simulation box");
    let mut nl = NeighborList::new(cutoff);
    nl.build_columns(xs.slice(), ys.slice(), zs.slice(), simbox);
    nl.neighbors(NeighborsStorage::FULL)
}
