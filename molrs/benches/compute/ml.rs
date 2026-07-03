//! `ml` category — PCA projection + k-means (regression).
//!
//! [`Pca2`] projects a descriptor matrix to 2 components; [`KMeans`] then
//! clusters that projection. ONE small synthetic descriptor matrix.

use criterion::{Criterion, criterion_group};
use molrs::Frame;
use molrs::compute::ml::{KMeans, Pca2};
use molrs::compute::result::{ComputeResult, DescriptorRow};
use molrs::compute::traits::Compute;
use molrs::types::F;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// Descriptor row wrapping a `Vec<F>` (one observation).
#[derive(Clone)]
struct Row(Vec<F>);
impl DescriptorRow for Row {
    fn as_row(&self) -> &[F] {
        &self.0
    }
}
impl ComputeResult for Row {}

const N_PER_CLUSTER: usize = 200;
const DIMS: usize = 4;

/// Three loosely separated blobs of `DIMS`-dimensional rows.
fn make_rows(seed: u64) -> Vec<Row> {
    let centers = [0.0_f64, 10.0, 20.0];
    let mut rng = StdRng::seed_from_u64(seed);
    let mut rows = Vec::with_capacity(centers.len() * N_PER_CLUSTER);
    for &cx in &centers {
        for _ in 0..N_PER_CLUSTER {
            let r: Vec<F> = (0..DIMS)
                .map(|d| {
                    let base = if d == 0 { cx } else { 0.0 };
                    base + (rng.random::<F>() - 0.5)
                })
                .collect();
            rows.push(Row(r));
        }
    }
    rows
}

fn bench_pca(c: &mut Criterion) {
    let mut group = c.benchmark_group("ml/pca");
    crate::helpers::configure(&mut group);
    let rows = make_rows(42);
    let frame = Frame::new();
    group.bench_function("reg", |b| {
        b.iter(|| {
            std::hint::black_box(Pca2::<Row>::new().compute(&[&frame], &rows).unwrap());
        })
    });
    group.finish();
}

fn bench_kmeans(c: &mut Criterion) {
    let mut group = c.benchmark_group("ml/kmeans");
    crate::helpers::configure(&mut group);
    let rows = make_rows(42);
    let frame = Frame::new();
    let pca = Pca2::<Row>::new().compute(&[&frame], &rows).unwrap();
    let km = KMeans::new(3, 100, 42).unwrap();
    group.bench_function("reg", |b| {
        b.iter(|| {
            std::hint::black_box(km.compute(&[&frame], &pca).unwrap());
        })
    });
    group.finish();
}

criterion_group!(benches, bench_pca, bench_kmeans);
