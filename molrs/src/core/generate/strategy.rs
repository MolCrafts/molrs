//! The two built-in [`GrowthStrategy`] implementations.
//!
//! Both are plain structs injected into
//! [`SelfAvoidingWalk`](super::walk::SelfAvoidingWalk) — there are no factory
//! functions. Overlap is judged by occupancy cells, never by distance:
//! [`FccLattice`] steps onto a global FCC lattice and uses
//! [`OccupancyMode::SameCell`] (distinct sites are already `>= bond_length`
//! apart); [`OffLattice`] grows in continuous space and uses
//! [`OccupancyMode::BlockClear`] with cell edge `excluded_radius`.

use rand::RngExt;
use rand::rngs::StdRng;

use super::occupancy::OccupancyMode;
use super::walk::GrowthStrategy;
use crate::spatial::simbox::SimBox;
use crate::types::F;

/// FCC-lattice growth: nearest-neighbour spacing equals `bond_length`, the box
/// edge is rounded up to a whole number of conventional FCC cells, and overlap
/// is exact-site occupancy.
pub struct FccLattice;

/// Conventional-cubic FCC cell edge for nearest-neighbour spacing `bond_length`.
fn fcc_cell_edge(bond_length: F) -> F {
    bond_length * std::f64::consts::SQRT_2
}

/// The 12 FCC nearest-neighbour step vectors, each of magnitude `bond_length`.
fn fcc_steps(bond_length: F) -> [[F; 3]; 12] {
    let s = bond_length / std::f64::consts::SQRT_2;
    let dirs = [
        [1.0, 1.0, 0.0],
        [1.0, -1.0, 0.0],
        [-1.0, 1.0, 0.0],
        [-1.0, -1.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 0.0, -1.0],
        [-1.0, 0.0, 1.0],
        [-1.0, 0.0, -1.0],
        [0.0, 1.0, 1.0],
        [0.0, 1.0, -1.0],
        [0.0, -1.0, 1.0],
        [0.0, -1.0, -1.0],
    ];
    dirs.map(|d| [d[0] * s, d[1] * s, d[2] * s])
}

/// FCC basis offsets (fractions of the conventional cell edge).
const FCC_BASIS: [[F; 3]; 4] = [
    [0.0, 0.0, 0.0],
    [0.5, 0.5, 0.0],
    [0.5, 0.0, 0.5],
    [0.0, 0.5, 0.5],
];

impl GrowthStrategy for FccLattice {
    fn occupancy_mode(&self, bond_length: F) -> OccupancyMode {
        // Cell small enough that each distinct FCC site maps to its own cell:
        // two distinct sites are >= bond_length apart, and any two points in one
        // cube of side bond_length/2 are < bond_length*sqrt(3)/2 < bond_length.
        OccupancyMode::SameCell {
            cell: bond_length * 0.5,
        }
    }

    fn adjust_box_edge(&self, edge: F, bond_length: F) -> F {
        // Round up to a whole number of FCC cells so the lattice tiles the box
        // commensurately (keeps non-overlap exact across periodic boundaries).
        let cell = fcc_cell_edge(bond_length);
        (edge / cell).ceil().max(1.0) * cell
    }

    fn propose_first(&self, simbox: &SimBox, bond_length: F, rng: &mut StdRng) -> [F; 3] {
        let a = simbox.lengths()[0];
        let edge = fcc_cell_edge(bond_length);
        let ncells = (a / edge).round().max(1.0) as i64;
        let i = rng.random_range(0..ncells) as F;
        let j = rng.random_range(0..ncells) as F;
        let k = rng.random_range(0..ncells) as F;
        let basis = FCC_BASIS[rng.random_range(0..FCC_BASIS.len())];
        [
            (i + basis[0]) * edge,
            (j + basis[1]) * edge,
            (k + basis[2]) * edge,
        ]
    }

    fn propose_step(&self, tip: [F; 3], bond_length: F, rng: &mut StdRng) -> [F; 3] {
        let steps = fcc_steps(bond_length);
        let d = steps[rng.random_range(0..steps.len())];
        [tip[0] + d[0], tip[1] + d[1], tip[2] + d[2]]
    }
}

/// Continuous-space growth: each step samples a random direction at fixed
/// `bond_length`; overlap is the background-grid `BlockClear` rule with cell
/// edge `excluded_radius`. Use `excluded_radius <= bond_length`.
pub struct OffLattice {
    /// Minimum allowed minimum-image separation between non-bonded monomers.
    pub excluded_radius: F,
}

/// Sample a unit vector uniformly on the sphere from two uniforms.
fn random_unit(rng: &mut StdRng) -> [F; 3] {
    let z = 2.0 * rng.random::<f64>() - 1.0;
    let phi = 2.0 * std::f64::consts::PI * rng.random::<f64>();
    let r = (1.0 - z * z).max(0.0).sqrt();
    [r * phi.cos(), r * phi.sin(), z]
}

impl GrowthStrategy for OffLattice {
    fn occupancy_mode(&self, _bond_length: F) -> OccupancyMode {
        OccupancyMode::BlockClear {
            cell: self.excluded_radius,
        }
    }

    fn propose_first(&self, simbox: &SimBox, _bond_length: F, rng: &mut StdRng) -> [F; 3] {
        let a = simbox.lengths()[0];
        [
            rng.random::<f64>() * a,
            rng.random::<f64>() * a,
            rng.random::<f64>() * a,
        ]
    }

    fn propose_step(&self, tip: [F; 3], bond_length: F, rng: &mut StdRng) -> [F; 3] {
        let d = random_unit(rng);
        [
            tip[0] + d[0] * bond_length,
            tip[1] + d[1] * bond_length,
            tip[2] + d[2] * bond_length,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fcc_steps_have_bond_length() {
        let b = 1.53;
        for d in fcc_steps(b) {
            let mag = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            assert!((mag - b).abs() < 1e-12, "step magnitude {mag} != {b}");
        }
    }

    #[test]
    fn fcc_box_edge_is_commensurate() {
        let b = 1.53;
        let cell = fcc_cell_edge(b);
        let edge = FccLattice.adjust_box_edge(10.63, b);
        let ratio = edge / cell;
        assert!(
            (ratio - ratio.round()).abs() < 1e-9,
            "edge not a cell multiple"
        );
        assert!(edge >= 10.63);
    }

    #[test]
    fn random_unit_is_normalized() {
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..1000 {
            let v = random_unit(&mut rng);
            let mag = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            assert!((mag - 1.0).abs() < 1e-12, "‖unit‖ = {mag}");
        }
    }
}
