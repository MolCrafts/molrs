use criterion::{Criterion, criterion_group};
use molrs::spatial::region::simbox::SimBox;
use molrs::types::F;
use ndarray::array;

use crate::helpers;

fn bench_shortest_vector(c: &mut Criterion) {
    let mut group = c.benchmark_group("core/region/simbox/shortest_vector");
    helpers::configure(&mut group);

    let bx = SimBox::cube(
        10.0 as F,
        array![0.0 as F, 0.0 as F, 0.0 as F],
        [true, true, true],
    )
    .expect("invalid box length");
    let a = array![1.0 as F, 2.0 as F, 3.0 as F];
    let b = array![8.5 as F, 9.0 as F, 1.0 as F];

    group.bench_function("pbc_cube", |bencher| {
        bencher.iter(|| std::hint::black_box(bx.shortest_vector(a.view(), b.view())));
    });

    let bx_no_pbc = SimBox::cube(
        10.0 as F,
        array![0.0 as F, 0.0 as F, 0.0 as F],
        [false, false, false],
    )
    .expect("invalid box length");

    group.bench_function("no_pbc_cube", |bencher| {
        bencher.iter(|| std::hint::black_box(bx_no_pbc.shortest_vector(a.view(), b.view())));
    });

    group.finish();
}

fn bench_make_fractional(c: &mut Criterion) {
    let mut group = c.benchmark_group("core/region/simbox/make_fractional");
    helpers::configure(&mut group);

    let bx = SimBox::cube(
        10.0 as F,
        array![0.0 as F, 0.0 as F, 0.0 as F],
        [true, true, true],
    )
    .expect("invalid box length");
    let r = array![3.5 as F, 7.2 as F, 1.8 as F];

    group.bench_function("cube", |bencher| {
        bencher.iter(|| std::hint::black_box(bx.make_fractional_fast(r.view())));
    });

    group.finish();
}

fn bench_calc_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("core/region/simbox/calc_distance");
    helpers::configure(&mut group);

    let bx = SimBox::cube(
        10.0 as F,
        array![0.0 as F, 0.0 as F, 0.0 as F],
        [true, true, true],
    )
    .expect("invalid box length");
    let a = array![1.0 as F, 2.0 as F, 3.0 as F];
    let b = array![8.5 as F, 9.0 as F, 1.0 as F];

    group.bench_function("pbc_cube", |bencher| {
        bencher.iter(|| std::hint::black_box(bx.calc_distance2(a.view(), b.view())));
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_shortest_vector,
    bench_make_fractional,
    bench_calc_distance,
    bench_mic_variants_loop,
);

/// Compare the two surviving MIC variants over a realistic O(N²) pair-loop
/// workload. The four-variant family (`shortest_vector`,
/// `shortest_vector_fast`, `shortest_vector_fast_arr`, `shortest_vector_raw`)
/// was collapsed to two after a representative benchmark showed
/// `_fast` (Array1 output) carrying a 70% overhead vs the stack-array
/// path; this bench is the regression guard for that decision.
fn bench_mic_variants_loop(c: &mut Criterion) {
    use ndarray::Array2;
    use rand::RngExt;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    let bx = SimBox::cube(
        10.0 as F,
        array![0.0 as F, 0.0 as F, 0.0 as F],
        [true, true, true],
    )
    .unwrap();

    let n = 100;
    let mut rng = StdRng::seed_from_u64(42);
    let mut pts = Array2::<F>::zeros((n, 3));
    let mut flat: Vec<[F; 3]> = Vec::with_capacity(n);
    for i in 0..n {
        let p = [
            rng.random::<F>() * 10.0,
            rng.random::<F>() * 10.0,
            rng.random::<F>() * 10.0,
        ];
        pts[[i, 0]] = p[0];
        pts[[i, 1]] = p[1];
        pts[[i, 2]] = p[2];
        flat.push(p);
    }

    let mut group = c.benchmark_group("core/region/simbox/mic_pair_loop_100pts");
    helpers::configure(&mut group);

    group.bench_function("shortest_vector (F3View → Array1)", |bencher| {
        bencher.iter(|| {
            let mut acc: F = 0.0;
            for i in 0..n {
                let a = pts.row(i);
                for j in (i + 1)..n {
                    let b = pts.row(j);
                    let dr = bx.shortest_vector(a, b);
                    acc += dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                }
            }
            std::hint::black_box(acc)
        });
    });

    group.bench_function("shortest_vector_impl ([F;3] in & out)", |bencher| {
        bencher.iter(|| {
            let mut acc: F = 0.0;
            for (i, &a) in flat.iter().enumerate() {
                for &b in flat.iter().skip(i + 1) {
                    let dr = bx.shortest_vector_impl(a, b);
                    acc += dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                }
            }
            std::hint::black_box(acc)
        });
    });

    group.finish();
}
