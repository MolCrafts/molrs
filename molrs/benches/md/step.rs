//! One force evaluation, in the four shapes that regress differently.

use std::time::Duration;

use criterion::{Criterion, criterion_group};
use molrs::Topology;
use molrs::ff::potential::Potential;
use molrs::ff::potential::bond::harmonic::BondHarmonic;
use molrs::ff::potential::pair::LJCut;
use molrs::ff::potential::pair::lj_cut::Mixing;
use molrs::md::{Comm, ForceProvider, GhostPairs, MicPairs, SpecialWeights};
use molrs::spatial::neighbors::{NeighborList, NeighborPolicy, VerletSkin};
use molrs::spatial::simbox::SimBox;
use molrs::system::bond_weights::BondDistanceWeights;
use molrs::types::{F, FNx3};
use ndarray::{Array2, array};

/// Small enough to stay a regression bench, large enough that the halo carries
/// a real number of copies rather than a handful.
const SIDE: usize = 8;
const SPACING: F = 3.0;
const CUTOFF: F = 6.0;
const SKIN: F = 1.0;

/// A jittered cubic lattice, so no two separations are identical and the
/// kernel cannot be short-circuited by a degenerate configuration.
fn lattice() -> (SimBox, FNx3) {
    let n = SIDE * SIDE * SIDE;
    let l = SIDE as F * SPACING;
    let bx = SimBox::cube(l, array![0.0, 0.0, 0.0], [true; 3]).unwrap();
    let mut pos = FNx3::zeros((n, 3));
    let mut seed = 0x1234_5678_u64;
    let mut jitter = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as F / (1u64 << 31) as F - 0.5) * 0.3
    };
    for a in 0..n {
        let (i, j, k) = (a % SIDE, (a / SIDE) % SIDE, a / (SIDE * SIDE));
        pos[[a, 0]] = i as F * SPACING + jitter();
        pos[[a, 1]] = j as F * SPACING + jitter();
        pos[[a, 2]] = k as F * SPACING + jitter();
    }
    (bx, pos)
}

fn lj(n: usize) -> LJCut {
    LJCut::typed(
        (0..n).map(|i| (i % 2) as u32).collect(),
        &[(0.3, 3.4), (0.5, 3.0)],
        Mixing::Arithmetic,
        CUTOFF,
        12,
        6,
        false,
        false,
    )
    .unwrap()
}

/// Dimers along x, so every atom carries a bonded term to re-resolve.
fn chain(n: usize) -> (Topology, BondHarmonic) {
    let edges: Vec<[usize; 2]> = (0..n)
        .step_by(2)
        .filter(|&a| a + 1 < n)
        .map(|a| [a, a + 1])
        .collect();
    let k = vec![100.0; edges.len()];
    let r0 = vec![3.0; edges.len()];
    let (ai, aj): (Vec<usize>, Vec<usize>) = edges.iter().map(|e| (e[0], e[1])).unzip();
    (
        Topology::from_edges(n, &edges),
        BondHarmonic::new(ai, aj, k, r0),
    )
}

fn skin_for(bx: &SimBox, pos: &FNx3) -> VerletSkin {
    VerletSkin::new(
        NeighborList::new(CUTOFF + SKIN),
        CUTOFF,
        NeighborPolicy {
            skin: SKIN,
            ..NeighborPolicy::default()
        },
        pos.view(),
        bx.clone(),
    )
    .unwrap()
}

fn bench_step(c: &mut Criterion) {
    let (bx, pos) = lattice();
    let n = pos.nrows();
    let no_fold = Array2::<i64>::zeros((n, 3));
    // One atom crossing — what makes the bonded re-resolution and the halo's
    // re-imaging run at all.
    let mut one_fold = Array2::<i64>::zeros((n, 3));
    one_fold[[0, 0]] = 1;

    let (topo, _) = chain(n);
    let weights =
        SpecialWeights::new(&topo.special_weights(&BondDistanceWeights::from_exclusion_depth(3)));

    let mut g = c.benchmark_group("md/step");
    g.warm_up_time(Duration::from_millis(200));
    g.measurement_time(Duration::from_millis(600));
    g.sample_size(10);

    let mut mic = MicPairs::new(lj(n), skin_for(&bx, &pos)).unwrap();
    g.bench_function("mic/plain", |b| {
        b.iter(|| mic.compute(pos.view(), no_fold.view()).unwrap())
    });

    let mut mic_w = MicPairs::from_members(
        vec![(Box::new(lj(n)) as Box<dyn Potential>, weights.clone())],
        skin_for(&bx, &pos),
    )
    .unwrap();
    g.bench_function("mic/weighted", |b| {
        b.iter(|| mic_w.compute(pos.view(), no_fold.view()).unwrap())
    });

    let comm = Comm::new(bx.clone(), pos.view(), CUTOFF, SKIN).unwrap();
    let mut ghost = GhostPairs::new(lj(n), comm).unwrap();
    g.bench_function("ghost/plain", |b| {
        b.iter(|| ghost.compute(pos.view(), no_fold.view()).unwrap())
    });

    let (_, bond) = chain(n);
    let comm = Comm::new(bx.clone(), pos.view(), CUTOFF, SKIN).unwrap();
    let mut ghost_b = GhostPairs::from_members(
        vec![
            (Box::new(lj(n)) as Box<dyn Potential>, weights.clone()),
            (
                Box::new(bond) as Box<dyn Potential>,
                SpecialWeights::default(),
            ),
        ],
        comm,
    )
    .unwrap();
    // A folding step and a quiet one: a run does both, and only the folding one
    // exercises the re-imaging and the bonded re-resolution.
    g.bench_function("ghost/bonded_fold", |b| {
        b.iter(|| {
            ghost_b.compute(pos.view(), one_fold.view()).unwrap();
            ghost_b.compute(pos.view(), no_fold.view()).unwrap()
        })
    });

    g.finish();
}

criterion_group!(benches, bench_step);
