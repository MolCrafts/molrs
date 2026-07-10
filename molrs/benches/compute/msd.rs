//! `msd` category — mean squared displacement.
//!
//! Two regression points: the stateless batch [`MSD::compute`] and the
//! streaming [`MSDAccumulator`] hot path (flat blocked `x|y|z` per frame,
//! windowed lags).

use criterion::{Criterion, criterion_group};
use molrs::compute::msd::{MSD, MSDAccumulator};
use molrs::compute::traits::Compute;
use molrs::store::block::BlockDtype;
use molrs::types::F;

use crate::helpers;

/// Flatten a frame's `atoms` block into one blocked `x|y|z` slice of length
/// `3 * n_atoms`, the layout [`MSDAccumulator::accumulate`] consumes.
fn blocked_positions(frame: &molrs::store::frame::Frame) -> Vec<F> {
    let atoms = frame.get("atoms").unwrap();
    let mut out = Vec::new();
    for key in ["x", "y", "z"] {
        let col = atoms.get(key).unwrap();
        let arr = <F as BlockDtype>::from_column(col).unwrap();
        out.extend_from_slice(arr.as_slice().unwrap());
    }
    out
}

fn bench_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("msd/batch");
    helpers::configure(&mut group);
    let (frames_owned, _) = helpers::reg_pool();
    let frames: Vec<&_> = frames_owned.iter().collect();

    group.bench_function("reg", |b| {
        b.iter(|| {
            std::hint::black_box(MSD::new().compute(&frames, ()).unwrap());
        })
    });

    group.finish();
}

fn bench_accumulator(c: &mut Criterion) {
    let mut group = c.benchmark_group("msd/accumulator");
    helpers::configure(&mut group);
    let (frames_owned, _) = helpers::reg_pool();
    let flat: Vec<Vec<F>> = frames_owned.iter().map(blocked_positions).collect();
    let window = flat.len().saturating_sub(1);

    group.bench_function("reg", |b| {
        b.iter(|| {
            let mut acc = MSDAccumulator::new(window);
            for f in &flat {
                acc.accumulate(f).unwrap();
            }
            std::hint::black_box(acc.windowed_msd());
        })
    });

    group.finish();
}

criterion_group!(benches, bench_batch, bench_accumulator);
