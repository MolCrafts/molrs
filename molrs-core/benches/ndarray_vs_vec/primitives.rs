use criterion::{BenchmarkId, Criterion, criterion_group};
use molrs_core::types::F;
use ndarray::{Array1, Array2, Zip, s};

use super::SIZES;
use crate::helpers;

// ── Create: zeros ───────────────────────────────────────────────────

fn bench_create_zeros(c: &mut Criterion) {
    let mut group = c.benchmark_group("ndarray_vs_vec/create_zeros");

    for &n in SIZES {
        group.bench_with_input(BenchmarkId::new("ndarray", n), &n, |b, &n| {
            b.iter(|| std::hint::black_box(Array2::<f32>::zeros((n, 3))));
        });
        group.bench_with_input(BenchmarkId::new("vec", n), &n, |b, &n| {
            b.iter(|| std::hint::black_box(vec![[0.0f32; 3]; n]));
        });
    }

    group.finish();
}

// ── Create: from_iter ───────────────────────────────────────────────

fn bench_create_from_iter(c: &mut Criterion) {
    let mut group = c.benchmark_group("ndarray_vs_vec/create_from_iter");

    for &n in SIZES {
        group.bench_with_input(BenchmarkId::new("ndarray", n), &n, |b, &n| {
            b.iter(|| std::hint::black_box(Array1::from_iter((0..n).map(|i| i as f32))));
        });
        group.bench_with_input(BenchmarkId::new("vec", n), &n, |b, &n| {
            b.iter(|| std::hint::black_box((0..n).map(|i| i as f32).collect::<Vec<f32>>()));
        });
    }

    group.finish();
}

// ── Read: index (random access) ─────────────────────────────────────

fn bench_index(c: &mut Criterion) {
    let mut group = c.benchmark_group("ndarray_vs_vec/index");

    for &n in SIZES {
        let nd = helpers::random_points(n, 10.0, 42);
        let vc = helpers::random_points_native(n, 10.0, 42);
        let indices: Vec<usize> = (0..n.min(1000)).map(|i| (i * 7 + 13) % n).collect();

        group.bench_with_input(BenchmarkId::new("ndarray", n), &n, |b, _| {
            b.iter(|| {
                let mut s = 0.0 as F;
                for &i in &indices {
                    s += nd[[i, 0]] + nd[[i, 1]] + nd[[i, 2]];
                }
                std::hint::black_box(s)
            });
        });
        group.bench_with_input(BenchmarkId::new("vec", n), &n, |b, _| {
            b.iter(|| {
                let mut s = 0.0 as F;
                for &i in &indices {
                    s += vc[i][0] + vc[i][1] + vc[i][2];
                }
                std::hint::black_box(s)
            });
        });
    }

    group.finish();
}

// ── Read: slice ─────────────────────────────────────────────────────

fn bench_slice(c: &mut Criterion) {
    let mut group = c.benchmark_group("ndarray_vs_vec/slice");

    for &n in SIZES {
        let nd = helpers::random_1d_ndarray(n, 42);
        let vc = helpers::random_1d_vec(n, 42);
        let start = n / 4;
        let end = 3 * n / 4;

        group.bench_with_input(BenchmarkId::new("ndarray", n), &n, |b, _| {
            b.iter(|| {
                let sl = nd.slice(s![start..end]);
                std::hint::black_box(sl.sum())
            });
        });
        group.bench_with_input(BenchmarkId::new("vec", n), &n, |b, _| {
            b.iter(|| {
                let sl = &vc[start..end];
                std::hint::black_box(sl.iter().sum::<f32>())
            });
        });
    }

    group.finish();
}

// ── Read: sum ───────────────────────────────────────────────────────

fn bench_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("ndarray_vs_vec/sum");

    for &n in SIZES {
        let nd = helpers::random_1d_ndarray(n, 42);
        let vc = helpers::random_1d_vec(n, 42);

        group.bench_with_input(BenchmarkId::new("ndarray", n), &n, |b, _| {
            b.iter(|| std::hint::black_box(nd.sum()));
        });
        group.bench_with_input(BenchmarkId::new("vec", n), &n, |b, _| {
            b.iter(|| std::hint::black_box(vc.iter().sum::<f32>()));
        });
    }

    group.finish();
}

// ── Update: inplace scaled add (x += dt * v) ───────────────────────

fn bench_update_scaled_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("ndarray_vs_vec/update_scaled_add");

    for &n in SIZES {
        let x_nd_orig = helpers::random_1d_ndarray(n, 42);
        let v_nd = helpers::random_1d_ndarray(n, 99);
        let x_vc_orig = helpers::random_1d_vec(n, 42);
        let v_vc = helpers::random_1d_vec(n, 99);
        let dt = 0.001f32;

        group.bench_with_input(BenchmarkId::new("ndarray", n), &n, |b, _| {
            let mut x = x_nd_orig.clone();
            b.iter(|| {
                Zip::from(&mut x).and(&v_nd).for_each(|x, &v| *x += dt * v);
                std::hint::black_box(&x);
            });
        });
        group.bench_with_input(BenchmarkId::new("vec", n), &n, |b, _| {
            let mut x = x_vc_orig.clone();
            b.iter(|| {
                for (xi, &vi) in x.iter_mut().zip(v_vc.iter()) {
                    *xi += dt * vi;
                }
                std::hint::black_box(&x);
            });
        });
    }

    group.finish();
}

// ── Update: scatter (accumulate by index) ───────────────────────────

fn bench_scatter(c: &mut Criterion) {
    let mut group = c.benchmark_group("ndarray_vs_vec/scatter");

    for &n in SIZES {
        let pairs: Vec<(usize, usize)> = (0..n).map(|i| (i % n, (i * 7 + 3) % n)).collect();
        let values: Vec<[f32; 3]> = (0..n)
            .map(|i| {
                let v = i as f32 * 0.01;
                [v, -v, v * 0.5]
            })
            .collect();

        group.bench_with_input(BenchmarkId::new("ndarray", n), &n, |b, &n| {
            b.iter(|| {
                let mut acc = Array2::<f32>::zeros((n, 3));
                for (k, &(i, j)) in pairs.iter().enumerate() {
                    let f = values[k];
                    acc[[i, 0]] += f[0];
                    acc[[i, 1]] += f[1];
                    acc[[i, 2]] += f[2];
                    acc[[j, 0]] -= f[0];
                    acc[[j, 1]] -= f[1];
                    acc[[j, 2]] -= f[2];
                }
                std::hint::black_box(acc)
            });
        });
        group.bench_with_input(BenchmarkId::new("vec", n), &n, |b, &n| {
            b.iter(|| {
                let mut acc = vec![0.0f32; n * 3];
                for (k, &(i, j)) in pairs.iter().enumerate() {
                    let f = values[k];
                    acc[i * 3] += f[0];
                    acc[i * 3 + 1] += f[1];
                    acc[i * 3 + 2] += f[2];
                    acc[j * 3] -= f[0];
                    acc[j * 3 + 1] -= f[1];
                    acc[j * 3 + 2] -= f[2];
                }
                std::hint::black_box(acc)
            });
        });
    }

    group.finish();
}

// ── Matrix multiplication ───────────────────────────────────────────

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("ndarray_vs_vec/matmul");

    let mat_sizes: &[usize] = &[16, 64, 128, 512];

    for &n in mat_sizes {
        let a_nd = random_square(n, 42);
        let b_nd = random_square(n, 99);
        let a_vc = ndarray_to_flat(&a_nd);
        let b_vc = ndarray_to_flat(&b_nd);

        group.bench_with_input(BenchmarkId::new("ndarray", n), &n, |b, _| {
            b.iter(|| std::hint::black_box(a_nd.dot(&b_nd)));
        });
        group.bench_with_input(BenchmarkId::new("vec", n), &n, |b, &n| {
            b.iter(|| std::hint::black_box(matmul_naive(&a_vc, &b_vc, n)));
        });
    }

    group.finish();
}

fn random_square(n: usize, seed: u64) -> Array2<f32> {
    use rand::{Rng, SeedableRng, rngs::StdRng};
    let mut rng = StdRng::seed_from_u64(seed);
    Array2::from_shape_fn((n, n), |_| rng.random::<f32>())
}

fn ndarray_to_flat(a: &Array2<f32>) -> Vec<f32> {
    a.as_standard_layout().as_slice().unwrap().to_vec()
}

fn matmul_naive(a: &[f32], b: &[f32], n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; n * n];
    for i in 0..n {
        for k in 0..n {
            let a_ik = a[i * n + k];
            for j in 0..n {
                c[i * n + j] += a_ik * b[k * n + j];
            }
        }
    }
    c
}

criterion_group!(
    benches,
    bench_create_zeros,
    bench_create_from_iter,
    bench_index,
    bench_slice,
    bench_sum,
    bench_update_scaled_add,
    bench_scatter,
    bench_matmul,
);
