#![cfg(feature = "f64")]
#![allow(clippy::needless_range_loop)]
//! Finite-difference gradient consistency tests.
//! Requires f64 precision for accurate numerical differentiation.

use molrs_pack::objective::{compute_f, compute_fg, compute_g};
use molrs_pack::{F, PackContext, Restraint};

// ── helpers ────────────────────────────────────────────────────────────────

/// Central finite-difference gradient for variable `i`.
fn finite_diff(x: &[F], sys: &mut PackContext, i: usize, h: F) -> F {
    let mut xp = x.to_vec();
    let mut xm = x.to_vec();
    xp[i] += h;
    xm[i] -= h;
    let fp = compute_f(&xp, sys);
    let fm = compute_f(&xm, sys);
    (fp - fm) / (2.0 * h)
}

/// Build a minimal PackContext for `nmol` single-atom molecules.
fn single_atom_system(nmol: usize) -> PackContext {
    let ntotat = nmol;
    let mut sys = PackContext::new(ntotat, nmol, 1);
    sys.ntype_with_fixed = 1;
    sys.nmols = vec![nmol];
    sys.natoms = vec![1];
    sys.idfirst = vec![0];
    sys.comptype = vec![true];
    sys.coor = vec![[0.0, 0.0, 0.0]];
    sys.radius = vec![1.0; ntotat];
    sys.radius_ini = vec![1.0; ntotat];
    sys.fscale = vec![1.0; ntotat];
    sys
}

fn setup_cells(sys: &mut PackContext, cell_n: usize, cell_len: F) {
    sys.ncells = [cell_n, cell_n, cell_n];
    sys.cell_length = [cell_len; 3];
    sys.pbc_min = [0.0; 3];
    sys.pbc_length = [cell_len * cell_n as F; 3];
    sys.resize_cell_arrays();
}

// ── pair penalty gradient ──────────────────────────────────────────────────

#[test]
fn gradient_pair_penalty() {
    let mut sys = single_atom_system(2);
    sys.restraints.clear();
    sys.iratom_offsets = vec![0, 0, 0];
    sys.iratom_data.clear();
    setup_cells(&mut sys, 1, 10.0);

    // x = [com0(3), com1(3), euler0(3), euler1(3)]
    let mut x = vec![0.0; 12];
    x[0] = 1.0;
    x[1] = 1.0;
    x[2] = 1.0;
    x[3] = 2.5;
    x[4] = 1.0;
    x[5] = 1.0;

    let _ = compute_f(&x, &mut sys);
    let mut g = vec![0.0; x.len()];
    compute_g(&x, &mut sys, &mut g);

    let h = 1e-7;
    for i in 0..6 {
        let gfd = finite_diff(&x, &mut sys, i, h);
        let err = (g[i] - gfd).abs();
        assert!(
            err < 1e-3,
            "pair gradient mismatch at var {i}: analytic={} fd={gfd} err={err}",
            g[i]
        );
    }
}

// ── box constraint gradient ────────────────────────────────────────────────

#[test]
fn gradient_box_constraint() {
    let mut sys = single_atom_system(1);
    sys.restraints = vec![Restraint::inside_box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])];
    sys.iratom_offsets = vec![0, 1];
    sys.iratom_data = vec![0];
    sys.init1 = true;

    // Place atom outside the box
    let mut x = vec![0.0; 6];
    x[0] = 1.2;
    x[1] = -0.1;
    x[2] = 0.3;

    let _ = compute_f(&x, &mut sys);
    let mut g = vec![0.0; x.len()];
    compute_g(&x, &mut sys, &mut g);

    let h = 1e-7;
    for i in 0..3 {
        let gfd = finite_diff(&x, &mut sys, i, h);
        let err = (g[i] - gfd).abs();
        assert!(
            err < 1e-5,
            "box constraint gradient mismatch at var {i}: analytic={} fd={gfd} err={err}",
            g[i]
        );
    }
}

// ── sphere constraint gradient ─────────────────────────────────────────────

#[test]
fn gradient_sphere_constraint() {
    let mut sys = single_atom_system(1);
    sys.restraints = vec![Restraint::inside_sphere([0.0, 0.0, 0.0], 3.0)];
    sys.iratom_offsets = vec![0, 1];
    sys.iratom_data = vec![0];
    sys.init1 = true;

    let mut x = vec![0.0; 6];
    x[0] = 4.0;
    x[1] = 1.0;
    x[2] = 0.0;

    let _ = compute_f(&x, &mut sys);
    let mut g = vec![0.0; x.len()];
    compute_g(&x, &mut sys, &mut g);

    let h = 1e-7;
    for i in 0..3 {
        let gfd = finite_diff(&x, &mut sys, i, h);
        let err = (g[i] - gfd).abs();
        assert!(
            err < 1e-4,
            "sphere constraint gradient mismatch at var {i}: analytic={} fd={gfd} err={err}",
            g[i]
        );
    }
}

// ── plane constraint gradient ──────────────────────────────────────────────

#[test]
fn gradient_above_plane_constraint() {
    let mut sys = single_atom_system(1);
    sys.restraints = vec![Restraint::above_plane([0.0, 0.0, 1.0], 5.0)];
    sys.iratom_offsets = vec![0, 1];
    sys.iratom_data = vec![0];
    sys.init1 = true;

    let mut x = vec![0.0; 6];
    x[0] = 0.0;
    x[1] = 0.0;
    x[2] = 3.0; // below plane z=5

    let _ = compute_f(&x, &mut sys);
    let mut g = vec![0.0; x.len()];
    compute_g(&x, &mut sys, &mut g);

    let h = 1e-7;
    for i in 0..3 {
        let gfd = finite_diff(&x, &mut sys, i, h);
        let err = (g[i] - gfd).abs();
        assert!(
            err < 1e-5,
            "above_plane gradient mismatch at var {i}: analytic={} fd={gfd} err={err}",
            g[i]
        );
    }
}

// ── rotation gradient (multi-atom molecules) ───────────────────────────────

#[test]
fn gradient_with_rotations() {
    let mut sys = PackContext::new(4, 2, 1);
    sys.ntype_with_fixed = 1;
    sys.nmols = vec![2];
    sys.natoms = vec![2];
    sys.idfirst = vec![0];
    sys.comptype = vec![true];
    sys.coor = vec![[0.0, 0.0, 0.0], [1.0, 0.2, -0.1]];

    sys.radius = vec![1.0; 4];
    sys.radius_ini = vec![1.0; 4];
    sys.fscale = vec![1.0; 4];

    sys.restraints.clear();
    sys.iratom_offsets = vec![0, 0, 0, 0, 0];
    sys.iratom_data.clear();

    setup_cells(&mut sys, 2, 5.0);

    // x = [com0(3), com1(3), euler0(3), euler1(3)]
    let mut x = vec![0.0; 12];
    x[0] = 1.0;
    x[1] = 1.0;
    x[2] = 1.0;
    x[3] = 2.1;
    x[4] = 1.4;
    x[5] = 1.3;
    x[6] = 0.3;
    x[7] = 0.5;
    x[8] = 0.7;
    x[9] = -0.4;
    x[10] = 0.2;
    x[11] = -0.6;

    let _ = compute_f(&x, &mut sys);
    let mut g = vec![0.0; x.len()];
    compute_g(&x, &mut sys, &mut g);

    let h = 1e-7;
    for i in 0..x.len() {
        let gfd = finite_diff(&x, &mut sys, i, h);
        let err = (g[i] - gfd).abs();
        assert!(
            err < 5e-3,
            "rotation gradient mismatch at var {i}: analytic={} fd={gfd} err={err}",
            g[i]
        );
    }
}

// ── constraint + pair penalty combined ─────────────────────────────────────

#[test]
fn gradient_combined_constraint_and_pairs() {
    let mut sys = PackContext::new(3, 3, 1);
    sys.ntype_with_fixed = 1;
    sys.nmols = vec![3];
    sys.natoms = vec![1];
    sys.idfirst = vec![0];
    sys.comptype = vec![true];
    sys.coor = vec![[0.0, 0.0, 0.0]];

    sys.radius = vec![1.0; 3];
    sys.radius_ini = vec![1.0; 3];
    sys.fscale = vec![1.0; 3];

    // Box constraint on all atoms
    sys.restraints = vec![Restraint::inside_box([0.0, 0.0, 0.0], [5.0, 5.0, 5.0])];
    sys.iratom_offsets = vec![0, 1, 1, 1]; // only first atom has constraint
    sys.iratom_data = vec![0];

    setup_cells(&mut sys, 1, 10.0);

    // x = [com0(3), com1(3), com2(3), euler0(3), euler1(3), euler2(3)]
    let mut x = vec![0.0; 18];
    x[0] = 6.0; // outside box
    x[1] = 2.0;
    x[2] = 2.0;
    x[3] = 3.0;
    x[4] = 2.0;
    x[5] = 2.0;
    x[6] = 3.5;
    x[7] = 2.5;
    x[8] = 2.0;

    let _ = compute_f(&x, &mut sys);
    let mut g = vec![0.0; x.len()];
    compute_g(&x, &mut sys, &mut g);

    let h = 1e-7;
    for i in 0..9 {
        let gfd = finite_diff(&x, &mut sys, i, h);
        let err = (g[i] - gfd).abs();
        assert!(
            err < 1e-3,
            "combined gradient mismatch at var {i}: analytic={} fd={gfd} err={err}",
            g[i]
        );
    }
}

#[test]
fn fused_function_and_gradient_matches_separate_evaluation() {
    let mut sys = PackContext::new(4, 2, 1);
    sys.ntype_with_fixed = 1;
    sys.nmols = vec![2];
    sys.natoms = vec![2];
    sys.idfirst = vec![0];
    sys.comptype = vec![true];
    sys.coor = vec![[0.0, 0.0, 0.0], [1.0, 0.2, -0.1]];

    sys.radius = vec![1.0; 4];
    sys.radius_ini = vec![1.0; 4];
    sys.fscale = vec![1.0; 4];

    sys.restraints = vec![Restraint::inside_box([0.0, 0.0, 0.0], [5.0, 5.0, 5.0])];
    sys.iratom_offsets = vec![0, 1, 1, 2, 2];
    sys.iratom_data = vec![0, 0];
    setup_cells(&mut sys, 2, 5.0);

    let x = vec![1.2, 1.0, 1.1, 2.4, 1.3, 1.2, 0.3, 0.5, 0.7, -0.4, 0.2, -0.6];

    let f_sep = compute_f(&x, &mut sys);
    let mut g_sep = vec![0.0; x.len()];
    compute_g(&x, &mut sys, &mut g_sep);

    let mut g_fused = vec![0.0; x.len()];
    let f_fused = compute_fg(&x, &mut sys, &mut g_fused);

    assert!(
        (f_sep - f_fused).abs() < 1e-10,
        "f mismatch: {f_sep} vs {f_fused}"
    );
    for (i, (&a, &b)) in g_sep.iter().zip(&g_fused).enumerate() {
        let err = (a - b).abs();
        assert!(err < 1e-10, "g mismatch at {i}: {a} vs {b} (err={err})");
    }
}
