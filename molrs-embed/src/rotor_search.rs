//! Rotatable bond sampling stage.

use rand::Rng;

use super::geom::{add, norm, rotate_about_axis, scale, sub};
use super::optimizer::{EnergyModel, steepest_descent};
use super::options::EmbedOptions;
use molrs::molgraph::MolGraph;
use molrs::rotatable::detect_rotatable_bonds_with_downstream;

/// Rotor-search stage summary.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RotorSearchResult {
    pub attempts: usize,
    pub improved: bool,
    pub energy_before: f64,
    pub energy_after: f64,
}

pub(crate) fn run(
    mol: &MolGraph,
    model: &EnergyModel,
    coords: &mut [[f64; 3]],
    opts: &EmbedOptions,
    rng: &mut impl Rng,
) -> RotorSearchResult {
    let rot_bonds = detect_rotatable_bonds_with_downstream(mol);
    let attempts = opts.rotor_attempts(rot_bonds.len());
    let energy_before = model.energy(coords);
    if rot_bonds.is_empty() || attempts == 0 {
        return RotorSearchResult {
            attempts: 0,
            improved: false,
            energy_before,
            energy_after: energy_before,
        };
    }

    let max_delta = opts.rotor_max_delta();
    let local_steps = match opts.speed {
        super::options::EmbedSpeed::Fast => 6,
        super::options::EmbedSpeed::Medium => 10,
        super::options::EmbedSpeed::Better => 16,
    };

    let mut best = coords.to_vec();
    let mut best_energy = energy_before;
    let mut trial = best.clone();

    for _ in 0..attempts {
        let bidx = rng.random_range(0..rot_bonds.len());
        let bond = &rot_bonds[bidx];
        let delta = rng.random_range(-max_delta..max_delta);

        trial.copy_from_slice(&best);
        rotate_around_bond(&mut trial, bond.j, bond.k, &bond.downstream, delta);
        let _ = steepest_descent(model, &mut trial, local_steps, 0.015, 1e-3);
        let e = model.energy(&trial);
        if e < best_energy {
            best_energy = e;
            best.copy_from_slice(&trial);
        }
    }

    let improved = best_energy + 1e-10 < energy_before;
    if improved {
        coords.copy_from_slice(&best);
    }

    RotorSearchResult {
        attempts,
        improved,
        energy_before,
        energy_after: best_energy.min(energy_before),
    }
}

fn rotate_around_bond(
    coords: &mut [[f64; 3]],
    j: usize,
    k: usize,
    downstream: &[usize],
    angle: f64,
) {
    let origin = coords[j];
    let axis = sub(coords[k], origin);
    let axis_len = norm(axis);
    if axis_len < 1e-12 {
        return;
    }

    for &idx in downstream {
        let local = sub(coords[idx], origin);
        let rotated = rotate_about_axis(local, axis, angle);
        coords[idx] = add(origin, scale(rotated, 1.0));
    }
}
