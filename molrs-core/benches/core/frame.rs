//! Frame / Block access benches.
//!
//! Measures baseline cost of `Frame::clone` (the pattern we replaced
//! in molrs-wasm with borrow-based `with_frame`) plus cheap hot-path
//! operations: `Frame::get`, column slice extraction, `Block::insert`.

use criterion::{BenchmarkId, Criterion, criterion_group};
use molrs_core::block::{Block, BlockDtype};
use molrs_core::frame::Frame;
use molrs_core::types::F;
use ndarray::{ArrayD, IxDyn};

use crate::helpers;

const SIZES: &[usize] = &[500, 2_000, 10_000];
const BOX_SIZE: F = 30.0;

fn make_frame(n: usize, seed: u64) -> Frame {
    let pts = helpers::random_points(n, BOX_SIZE, seed);

    let col_from_axis = |axis: usize| -> ArrayD<F> {
        let mut v = Vec::with_capacity(n);
        for i in 0..n {
            v.push(pts[[i, axis]]);
        }
        ArrayD::from_shape_vec(IxDyn(&[n]), v).expect("col shape")
    };

    let mut atoms = Block::new();
    atoms.insert("x", col_from_axis(0)).unwrap();
    atoms.insert("y", col_from_axis(1)).unwrap();
    atoms.insert("z", col_from_axis(2)).unwrap();

    let mut frame = Frame::new();
    frame.insert("atoms", atoms);
    frame
}

fn bench_clone(c: &mut Criterion) {
    let mut group = c.benchmark_group("frame/clone");
    for &n in SIZES {
        let frame = make_frame(n, 42);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                std::hint::black_box(frame.clone());
            });
        });
    }
    group.finish();
}

fn bench_get(c: &mut Criterion) {
    let mut group = c.benchmark_group("frame/get");
    for &n in SIZES {
        let frame = make_frame(n, 42);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let atoms = frame.get("atoms").expect("atoms block");
                std::hint::black_box(atoms);
            });
        });
    }
    group.finish();
}

fn bench_col_slice(c: &mut Criterion) {
    // Simulates the inner hot operation: pull x/y/z columns as &[F] slices
    // (what `positions_from_frame` in molrs-wasm does before Nx3 assembly).
    let mut group = c.benchmark_group("frame/col_slice");
    for &n in SIZES {
        let frame = make_frame(n, 42);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let atoms = frame.get("atoms").unwrap();
                for key in ["x", "y", "z"] {
                    let col = atoms.get(key).unwrap();
                    let arr = <F as BlockDtype>::from_column(col).unwrap();
                    let slice = arr.as_slice().unwrap();
                    std::hint::black_box(slice);
                }
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_clone, bench_get, bench_col_slice);
