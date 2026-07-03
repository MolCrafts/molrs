//! Topology hot-path benches: angle/dihedral perception and single-source BFS
//! distances over the bond graph. These exercise the Python-reachable
//! `generate_topology` / `topo_distances` entry points — the core graph kernels
//! that molpy's `Atomistic` now delegates to after the core-sink — and guard
//! the native 2-edge / 3-edge path enumeration against an accidental
//! super-linear regression.

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group};
use molrs::system::atomistic::{AtomId, Atomistic};

use crate::helpers;

// Linear-chain lengths in atoms. A chain has degree-2 interior nodes, so angle
// (N-2) and dihedral (N-3) perception is O(N) — super-linear scaling here flags
// a regression in the path-enumeration walk.
const SIZES: &[usize] = &[helpers::REG_N];

/// Linear carbon chain of `n` atoms (`n-1` bonds), returning the molecule and
/// its atom ids in chain order (ids[0] is an endpoint).
fn make_chain(n: usize) -> (Atomistic, Vec<AtomId>) {
    let mut mol = Atomistic::new();
    let ids: Vec<AtomId> = (0..n)
        .map(|k| mol.add_atom_xyz("C", k as f64, 0.0, 0.0))
        .collect();
    for k in 0..n.saturating_sub(1) {
        let _ = mol.add_bond(ids[k], ids[k + 1]);
    }
    (mol, ids)
}

fn bench_generate_topology(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph/generate_topology");
    helpers::configure(&mut group);
    for &n in SIZES {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            // Rebuild a fresh chain each iteration so perception always runs on
            // an un-perceived graph (build is untimed in the setup closure).
            b.iter_batched(
                || make_chain(n).0,
                |mut mol| mol.generate_topology(true, true, false).unwrap(),
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn bench_topo_distances(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph/topo_distances");
    helpers::configure(&mut group);
    for &n in SIZES {
        let (mol, ids) = make_chain(n);
        let source = ids[0];
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| mol.topo_distances(source));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_generate_topology, bench_topo_distances);
