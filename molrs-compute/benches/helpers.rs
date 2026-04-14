//! Shared fixtures for molrs-compute benches.
//!
//! Builds a synthetic [`Frame`] of N random atoms in a cubic periodic box
//! and a [`NeighborList`] ready for RDF / Cluster / tensor benches.

use molrs::block::Block;
use molrs::frame::Frame;
use molrs::neighbors::{LinkCell, NbList, NeighborList};
use molrs::region::simbox::SimBox;
use molrs::types::F;
use ndarray::{Array2, ArrayD, IxDyn, array};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub const BOX_SIZE: F = 30.0;
pub const CUTOFF: F = 4.0;

pub fn random_positions(n: usize, box_size: F, seed: u64) -> Array2<F> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut pts = Array2::<F>::zeros((n, 3));
    for i in 0..n {
        pts[[i, 0]] = rng.random::<F>() * box_size;
        pts[[i, 1]] = rng.random::<F>() * box_size;
        pts[[i, 2]] = rng.random::<F>() * box_size;
    }
    pts
}

pub fn pbc_simbox(size: F) -> SimBox {
    SimBox::cube(
        size,
        array![0.0 as F, 0.0 as F, 0.0 as F],
        [true, true, true],
    )
    .expect("invalid box length")
}

/// Build a Frame with an `"atoms"` block holding `x`/`y`/`z` columns
/// plus a PBC simbox. Positions are the columns of `pts` (shape `[n, 3]`).
pub fn frame_from_positions(pts: &Array2<F>, simbox: SimBox) -> Frame {
    let n = pts.nrows();
    let col = |axis: usize| -> ArrayD<F> {
        let mut v = Vec::with_capacity(n);
        for i in 0..n {
            v.push(pts[[i, axis]]);
        }
        ArrayD::from_shape_vec(IxDyn(&[n]), v).expect("column shape")
    };
    let mut atoms = Block::new();
    atoms.insert("x", col(0)).expect("insert x");
    atoms.insert("y", col(1)).expect("insert y");
    atoms.insert("z", col(2)).expect("insert z");

    let mut frame = Frame::new();
    frame.insert("atoms", atoms);
    frame.simbox = Some(simbox);
    frame
}

/// Build a self-query [`NeighborList`] for the given positions using LinkCell.
pub fn build_nlist(pts: &Array2<F>, simbox: &SimBox, cutoff: F) -> NeighborList {
    let mut nl = NbList(LinkCell::new().cutoff(cutoff));
    nl.build(pts.view(), simbox);
    nl.query().clone()
}

/// One-shot fixture: positions, simbox, frame, neighbor list.
#[allow(dead_code)]
pub struct Fixture {
    pub positions: Array2<F>,
    pub simbox: SimBox,
    pub frame: Frame,
    pub nlist: NeighborList,
}

pub fn fixture(n: usize, seed: u64) -> Fixture {
    let positions = random_positions(n, BOX_SIZE, seed);
    let simbox = pbc_simbox(BOX_SIZE);
    let nlist = build_nlist(&positions, &simbox, CUTOFF);
    let frame = frame_from_positions(&positions, simbox.clone());
    Fixture {
        positions,
        simbox,
        frame,
        nlist,
    }
}
