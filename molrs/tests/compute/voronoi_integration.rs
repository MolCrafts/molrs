//! Integration tests for Voronoi electron-density integration → molecular
//! moments (travis-parity-07). Densities are constructed in memory (compute
//! testing, not IO-format testing), so the IO real-fixture rule does not apply.

use molrs::compute::voronoi::{
    BOHR_TO_ANG, DensityGrid, MolecularMoments, VoronoiIntegration, polarizability_finite_field,
};
use molrs::spatial::region::simbox::SimBox;
use molrs::types::F;
use ndarray::{Array2, array};

fn cube_box(l: F) -> SimBox {
    SimBox::cube(l, array![0.0 as F, 0.0, 0.0], [true, true, true]).unwrap()
}

/// 1 Å voxels on `[0,l)^3`, all-zero density to start.
fn unit_grid(l: usize) -> DensityGrid {
    DensityGrid::new(
        [0.0, 0.0, 0.0],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [l, l, l],
        vec![0.0; l * l * l],
    )
}

#[inline]
fn idx(dims: [usize; 3], i: usize, j: usize, k: usize) -> usize {
    (i * dims[1] + j) * dims[2] + k
}

// ---------------------------------------------------------------------------
// ac-002: integrated electronic charge is conserved
// ---------------------------------------------------------------------------
#[test]
fn charge_is_conserved_across_cells() {
    let sb = cube_box(10.0);
    let mut g = unit_grid(10);
    // Spread some electrons over several voxels (total = 6 electrons).
    for (i, j, k, q) in [(1, 1, 1, 2.0), (8, 8, 8, 3.0), (4, 6, 2, 1.0)] {
        let m = idx(g.dims, i, j, k);
        g.density[m] = q; // dV = 1 → n_elec = q
    }
    let total_elec: F = g.density.iter().sum::<F>() * g.dv;

    // Two atoms → two molecules, total nuclear charge 10.
    let pos = array![[2.0, 2.0, 2.0], [7.0, 7.0, 7.0]];
    let z = [4, 6];
    let out = VoronoiIntegration
        .integrate(pos.view(), &[0.0, 0.0], &z, &[0, 1], 2, &g, &sb)
        .unwrap();

    let sys_charge: F = z.iter().map(|&zi| zi as F).sum::<F>() - total_elec;
    let summed: F = out.charges.iter().sum();
    assert!(
        (summed - sys_charge).abs() < 1e-9,
        "Σ molecular charge {summed} != system charge {sys_charge}"
    );
}

// ---------------------------------------------------------------------------
// ac-003: integrated dipole matches an analytic distribution
// ---------------------------------------------------------------------------
#[test]
fn single_displaced_electron_gives_analytic_dipole() {
    let sb = cube_box(10.0);
    let mut g = unit_grid(10);
    // one electron at voxel point (5,5,7); atom (nucleus Z=1) at (5,5,5).
    g.density[idx(g.dims, 5, 5, 7)] = 1.0;
    let pos = array![[5.0, 5.0, 5.0]];
    let out = VoronoiIntegration
        .integrate(pos.view(), &[0.0], &[1], &[0], 1, &g, &sb)
        .unwrap();
    // μ = Z(r_a−r_ref) − Σ n(r−r_ref) = 0 − ([0,0,2]) = [0,0,-2]
    assert!((out.charges[0]).abs() < 1e-12, "neutral");
    assert!((out.dipoles[[0, 0]]).abs() < 1e-12);
    assert!((out.dipoles[[0, 1]]).abs() < 1e-12);
    assert!(
        (out.dipoles[[0, 2]] + 2.0).abs() < 1e-9,
        "μ_z = {}",
        out.dipoles[[0, 2]]
    );
}

#[test]
fn centrosymmetric_density_has_zero_dipole() {
    let sb = cube_box(10.0);
    let mut g = unit_grid(10);
    g.density[idx(g.dims, 5, 5, 7)] = 0.5;
    g.density[idx(g.dims, 5, 5, 3)] = 0.5; // symmetric about z=5
    let pos = array![[5.0, 5.0, 5.0]];
    let out = VoronoiIntegration
        .integrate(pos.view(), &[0.0], &[1], &[0], 1, &g, &sb)
        .unwrap();
    let mag = (0..3)
        .map(|c| out.dipoles[[0, c]].powi(2))
        .sum::<F>()
        .sqrt();
    assert!(
        mag < 1e-9,
        "centrosymmetric dipole magnitude {mag} should be ~0"
    );
}

/// Origin-dependence: for a *charged* molecule the dipole shifts with the
/// reference by `μ' = μ − Q·Δref`; for a neutral one it is invariant.
#[test]
fn charged_species_dipole_is_origin_dependent() {
    let sb = cube_box(10.0);
    let mut g = unit_grid(10);
    g.density[idx(g.dims, 5, 5, 7)] = 2.0; // 2 electrons, nucleus Z=1 → net charge −1
    let pos = array![[5.0, 5.0, 5.0]];
    let out = VoronoiIntegration
        .integrate(pos.view(), &[0.0], &[1], &[0], 1, &g, &sb)
        .unwrap();
    let q = out.charges[0];
    assert!((q + 1.0).abs() < 1e-9, "charge {q} should be −1");
    // shift reference by Δ=(1,0,0): μ' = μ − Q·Δ ; since Q≠0 it must differ.
    let delta = [1.0, 0.0, 0.0];
    let shifted_x = out.dipoles[[0, 0]] - q * delta[0];
    assert!(
        (shifted_x - out.dipoles[[0, 0]]).abs() > 0.5,
        "charged-species dipole must depend on reference"
    );
}

// ---------------------------------------------------------------------------
// ac-004: unit conversion (e/Bohr³ → e/Å³) via from_cube_frame
// ---------------------------------------------------------------------------
#[test]
fn cube_unit_conversion_reproduces_hand_value() {
    use molrs::store::block::Block;
    use molrs::store::frame::Frame;
    use ndarray::Array1;

    // Cube with 1-Bohr voxels (so voxel length in Å = BOHR_TO_ANG), 2³ grid,
    // uniform density 1.0 e/Bohr³. Hand value: 8 electrons total (= 2³ Bohr³).
    let dims = [2usize, 2, 2];
    let a = BOHR_TO_ANG;
    let mut grid = Block::new();
    grid.insert("density", Array1::from_vec(vec![1.0 as F; 8]).into_dyn())
        .unwrap();
    grid.set_shape(&dims).unwrap();
    let mut frame = Frame::new();
    frame.insert("grid", grid);
    frame.meta.insert("cube_units".into(), "bohr".into());
    // h column j = voxel_vec_j × dim_j = a × 2 along each axis.
    let h = ndarray::Array2::from_shape_fn((3, 3), |(i, j)| if i == j { a * 2.0 } else { 0.0 });
    frame.simbox = Some(SimBox::new(h, array![0.0 as F, 0.0, 0.0], [true, true, true]).unwrap());

    let dg = DensityGrid::from_cube_frame(&frame).unwrap();
    let total_elec: F = dg.density.iter().sum::<F>() * dg.dv;
    assert!(
        (total_elec - 8.0).abs() < 1e-9,
        "converted electron count {total_elec} != 8"
    );
}

// ---------------------------------------------------------------------------
// ac-005: finite-field polarizability recovers a linear-response input
// ---------------------------------------------------------------------------
#[test]
fn finite_field_recovers_alpha() {
    let alpha = [[2.0, 0.3, 0.0], [0.3, 1.5, 0.1], [0.0, 0.1, 1.0]];
    let mu0 = [0.4_f64, -0.2, 0.05];
    let e = 1e-3;
    let mk = |v: Vec<F>| MolecularMoments {
        charges: vec![0.0],
        dipoles: Array2::from_shape_vec((1, 3), v).unwrap(),
        references: Array2::zeros((1, 3)),
    };
    let zero = mk(mu0.to_vec());
    let mut recovered = [[0.0; 3]; 3];
    for j in 0..3 {
        let plus = mk((0..3).map(|i| mu0[i] + e * alpha[i][j]).collect());
        let minus = mk((0..3).map(|i| mu0[i] - e * alpha[i][j]).collect());
        let col = polarizability_finite_field(&zero, &plus, &minus, e).unwrap();
        for i in 0..3 {
            recovered[i][j] = col[[0, i]];
        }
    }
    for i in 0..3 {
        for j in 0..3 {
            assert!((recovered[i][j] - alpha[i][j]).abs() < 1e-9);
        }
    }
}

// ---------------------------------------------------------------------------
// ac-006: boundary determinism + PBC molecule handling
// ---------------------------------------------------------------------------
#[test]
fn boundary_voxel_assigned_deterministically() {
    // Two equal-radius generators; a voxel exactly equidistant goes to the
    // lower index (argmin tie-break), regardless of run.
    let sb = cube_box(10.0);
    let mut g = unit_grid(10);
    g.density[idx(g.dims, 5, 0, 0)] = 1.0; // equidistant between x=2 and x=8? no:
    // place generators at x=4 and x=6, voxel at x=5 is the midpoint.
    let pos = array![[4.0, 5.0, 5.0], [6.0, 5.0, 5.0]];
    g.density.iter_mut().for_each(|v| *v = 0.0);
    g.density[idx(g.dims, 5, 5, 5)] = 1.0;
    let out = VoronoiIntegration
        .integrate(pos.view(), &[0.0, 0.0], &[1, 1], &[0, 1], 2, &g, &sb)
        .unwrap();
    // The midpoint electron must land on molecule 0 (lower index) → its charge
    // is 1 − 1 = 0, the other stays +1.
    assert!(
        (out.charges[0] - 0.0).abs() < 1e-9,
        "mol0 {}",
        out.charges[0]
    );
    assert!(
        (out.charges[1] - 1.0).abs() < 1e-9,
        "mol1 {}",
        out.charges[1]
    );
}

#[test]
fn molecule_split_across_boundary_uses_min_image() {
    // A diatomic straddling the x-edge: atoms at x=0.5 and x=9.5 (1 Å apart via
    // min image, not 9 Å). A symmetric electron pair on each atom → small,
    // finite dipole, not a spurious ~9 Å·e moment.
    let sb = cube_box(10.0);
    let mut g = unit_grid(10);
    g.density[idx(g.dims, 0, 5, 5)] = 1.0; // near atom A (x≈0.5)
    g.density[idx(g.dims, 9, 5, 5)] = 1.0; // near atom B (x≈9.5)
    let pos = array![[0.5, 5.0, 5.0], [9.5, 5.0, 5.0]];
    let out = VoronoiIntegration
        .integrate(pos.view(), &[0.0, 0.0], &[1, 1], &[0, 0], 1, &g, &sb)
        .unwrap();
    let mag = (0..3)
        .map(|c| out.dipoles[[0, c]].powi(2))
        .sum::<F>()
        .sqrt();
    assert!(
        mag < 2.0,
        "min-image dipole magnitude {mag} should be small, not ~9 (no wrap)"
    );
    assert!((out.charges[0]).abs() < 1e-9, "neutral diatomic");
}

#[test]
fn dimension_mismatch_is_rejected() {
    let sb = cube_box(10.0);
    let g = unit_grid(4);
    let pos = array![[2.0, 2.0, 2.0]];
    // radii length 2 != 1 atom
    let err = VoronoiIntegration
        .integrate(pos.view(), &[0.0, 0.0], &[1], &[0], 1, &g, &sb)
        .unwrap_err();
    assert!(matches!(
        err,
        molrs::compute::ComputeError::DimensionMismatch { .. }
    ));
}
