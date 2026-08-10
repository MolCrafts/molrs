//! Shared `#[cfg(test)]` neighbor-list construction for the compute unit tests.
//!
//! The per-kernel test modules used to pack the frame's separate `x`/`y`/`z`
//! columns into an owned [`Array2`](ndarray::Array2) and call `LinkCell::build`.
//! This helper replaces that boilerplate with the native SoA neighbor path
//! ([`get_positions_ref`] → [`LinkCell::build_soa`]), reading the three columns
//! directly. The two paths are byte-for-byte equivalent (proven by
//! `build_soa_matches_build_bitwise`), so every rewired test produces identical
//! results.

use molrs::spatial::neighbors::{LinkCell, NbListAlgo, Neighbors};
use molrs::store::frame_access::FrameAccess;
use molrs::types::F;

use super::util::get_positions_ref;

/// Build a self-query [`Neighbors`] table for `frame` at `cutoff` via the native
/// SoA path, using the frame's own simulation box.
///
/// Reads the `atoms` block's `x`/`y`/`z` columns with [`get_positions_ref`] (a
/// zero-copy borrow for contiguous columns) and feeds them straight to
/// [`LinkCell::build_soa`] — no interleaved `Array2` copy. Byte-for-byte
/// identical to the former per-file `LinkCell::build(pos.view(), box)` helpers.
///
/// # Panics
/// Panics if the frame lacks `x`/`y`/`z` columns or has no simulation box.
pub(crate) fn nlist_from_frame<FA: FrameAccess>(frame: &FA, cutoff: F) -> Neighbors {
    let (xs, ys, zs) = get_positions_ref(frame).expect("frame must expose x/y/z columns");
    let simbox = frame
        .simbox_ref()
        .expect("frame must have a simulation box");
    let mut lc = LinkCell::new().cutoff(cutoff);
    lc.build_soa(xs.slice(), ys.slice(), zs.slice(), simbox);
    lc.query().clone()
}
