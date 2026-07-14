//! Graph hot-path benches: aromaticity perception (per-atom valence over
//! incident bonds) and ring perception (native cycle basis). These exercise
//! the Python-reachable `perceive_aromaticity` / `find_rings` entry points and
//! guard the O(N) adjacency path against an accidental O(N^2) all-bonds scan.

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group};
use molrs::perceive::aromaticity::perceive_aromaticity;
use molrs::perceive::rings::find_rings;
use molrs::system::atomistic::Atomistic;

use crate::helpers;

// Sizes are *ring counts*; total atoms = 6 x this. One small regression size.
const SIZES: &[usize] = &[200];

/// `n_rings` independent 6-membered carbon rings (6 atoms + 6 bonds each).
/// Ring atoms are exactly what drives `perceive_aromaticity`'s per-atom
/// incident-bond / valence passes, so an O(N^2) all-bonds scan in
/// `incident_bonds` shows up as super-linear scaling here (an O(N) adjacency
/// path stays linear).
fn make_rings(n_rings: usize) -> Atomistic {
    let mut mol = Atomistic::new();
    for r in 0..n_rings {
        let base = r as f64 * 10.0;
        let ids: Vec<_> = (0..6)
            .map(|k| mol.add_atom_xyz("C", base + k as f64, 0.0, 0.0))
            .collect();
        for k in 0..6 {
            let _ = mol.add_bond(ids[k], ids[(k + 1) % 6]);
        }
    }
    mol
}

fn bench_perceive_aromaticity(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph/perceive_aromaticity");
    helpers::configure(&mut group);
    for &n in SIZES {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter_batched(
                || make_rings(n),
                |mut mol| perceive_aromaticity(&mut mol),
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn bench_find_rings(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph/find_rings");
    helpers::configure(&mut group);
    for &n in SIZES {
        let mol = make_rings(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| find_rings(&mol));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_perceive_aromaticity, bench_find_rings);
