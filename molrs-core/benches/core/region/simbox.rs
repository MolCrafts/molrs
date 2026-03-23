use criterion::{Criterion, criterion_group};
use molrs::region::simbox::SimBox;
use molrs::types::F;
use ndarray::array;

fn bench_shortest_vector(c: &mut Criterion) {
    let mut group = c.benchmark_group("core/region/simbox/shortest_vector");

    let bx = SimBox::cube(
        10.0 as F,
        array![0.0 as F, 0.0 as F, 0.0 as F],
        [true, true, true],
    )
    .expect("invalid box length");
    let a = array![1.0 as F, 2.0 as F, 3.0 as F];
    let b = array![8.5 as F, 9.0 as F, 1.0 as F];

    group.bench_function("pbc_cube", |bencher| {
        bencher.iter(|| std::hint::black_box(bx.shortest_vector_fast(a.view(), b.view())));
    });

    let bx_no_pbc = SimBox::cube(
        10.0 as F,
        array![0.0 as F, 0.0 as F, 0.0 as F],
        [false, false, false],
    )
    .expect("invalid box length");

    group.bench_function("no_pbc_cube", |bencher| {
        bencher.iter(|| std::hint::black_box(bx_no_pbc.shortest_vector_fast(a.view(), b.view())));
    });

    group.finish();
}

fn bench_make_fractional(c: &mut Criterion) {
    let mut group = c.benchmark_group("core/region/simbox/make_fractional");

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
    bench_calc_distance
);
