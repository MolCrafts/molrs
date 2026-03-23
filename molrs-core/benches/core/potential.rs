use criterion::{Criterion, criterion_group};
use molrs::potential::Potential;
use molrs::potential::kernels::{BondHarmonic, PairLJ126};
use molrs::types::F;

fn random_coords(n_atoms: usize, seed: u64) -> Vec<F> {
    let mut state = seed;
    let mut coords = Vec::with_capacity(n_atoms * 3);
    for _ in 0..n_atoms * 3 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        coords.push((state as F / u64::MAX as F) * 10.0);
    }
    coords
}

fn bench_pair_lj126_forces(c: &mut Criterion) {
    for &n_pairs in &[100, 1_000, 10_000] {
        let n_atoms = n_pairs + 1;
        let atom_i: Vec<usize> = (0..n_pairs).collect();
        let atom_j: Vec<usize> = (1..=n_pairs).collect();
        let epsilon = vec![1.0 as F; n_pairs];
        let sigma = vec![1.0 as F; n_pairs];
        let pot = PairLJ126::new(atom_i, atom_j, epsilon, sigma);
        let coords = random_coords(n_atoms, 12345);

        c.bench_function(&format!("pair_lj126_forces_{n_pairs}"), |b| {
            b.iter(|| pot.forces(&coords))
        });
    }
}

fn bench_pair_lj126_energy(c: &mut Criterion) {
    let n_pairs = 10_000;
    let n_atoms = n_pairs + 1;
    let atom_i: Vec<usize> = (0..n_pairs).collect();
    let atom_j: Vec<usize> = (1..=n_pairs).collect();
    let epsilon = vec![1.0 as F; n_pairs];
    let sigma = vec![1.0 as F; n_pairs];
    let pot = PairLJ126::new(atom_i, atom_j, epsilon, sigma);
    let coords = random_coords(n_atoms, 12345);

    c.bench_function("pair_lj126_energy_10k", |b| b.iter(|| pot.energy(&coords)));
}

fn bench_bond_harmonic_forces(c: &mut Criterion) {
    let n_bonds = 10_000;
    let n_atoms = n_bonds + 1;
    let atom_i: Vec<usize> = (0..n_bonds).collect();
    let atom_j: Vec<usize> = (1..=n_bonds).collect();
    let k = vec![300.0 as F; n_bonds];
    let r0 = vec![1.5 as F; n_bonds];
    let pot = BondHarmonic::new(atom_i, atom_j, k, r0);
    let coords = random_coords(n_atoms, 54321);

    c.bench_function("bond_harmonic_forces_10k", |b| {
        b.iter(|| pot.forces(&coords))
    });
}

fn bench_potentials_forces(c: &mut Criterion) {
    use molrs::potential::Potentials;

    let n = 1000;
    let coords = random_coords(n + 1, 99999);

    // Bond potential
    let bond = BondHarmonic::new(
        (0..n).collect(),
        (1..=n).collect(),
        vec![300.0 as F; n],
        vec![1.5 as F; n],
    );
    // Pair potential
    let pair = PairLJ126::new(
        (0..n).collect(),
        (1..=n).collect(),
        vec![1.0 as F; n],
        vec![1.0 as F; n],
    );

    let mut pots = Potentials::new();
    pots.push(Box::new(bond));
    pots.push(Box::new(pair));

    c.bench_function("potentials_forces_bond+pair_1k", |b| {
        b.iter(|| pots.forces(&coords))
    });
}

criterion_group!(
    benches,
    bench_pair_lj126_forces,
    bench_pair_lj126_energy,
    bench_bond_harmonic_forces,
    bench_potentials_forces,
);
