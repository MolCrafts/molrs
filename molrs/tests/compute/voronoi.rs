//! Integration tests for the radical (Laguerre) Voronoi tessellation and its
//! domain / void consumers (spec travis-parity-06).

use molrs::compute::voronoi::{DomainAnalysis, RadicalVoronoi, VoidAnalysis};
use molrs::spatial::region::simbox::SimBox;
use molrs::types::F;
use ndarray::{Array2, array};

fn cube_box(l: F) -> SimBox {
    SimBox::cube(l, array![0.0 as F, 0.0, 0.0], [true, true, true]).unwrap()
}

fn ortho_box(lx: F, ly: F, lz: F) -> SimBox {
    SimBox::ortho(
        array![lx, ly, lz],
        array![0.0 as F, 0.0, 0.0],
        [true, true, true],
    )
    .unwrap()
}

/// `n`×`n`×`n` simple-cubic lattice (spacing 1) in a box of side `n`.
fn cubic_lattice(n: usize) -> (Array2<F>, SimBox) {
    let mut pts = Vec::new();
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                pts.push([i as F, j as F, k as F]);
            }
        }
    }
    let m = pts.len();
    let mut a = Array2::zeros((m, 3));
    for (r, p) in pts.iter().enumerate() {
        a[[r, 0]] = p[0];
        a[[r, 1]] = p[1];
        a[[r, 2]] = p[2];
    }
    (a, cube_box(n as F))
}

// ── ac-001 ──────────────────────────────────────────────────────────────────
#[test]
fn total_volume_equals_box_volume() {
    // Deterministic pseudo-random configuration with per-atom radii.
    let l: F = 10.0;
    let n = 30;
    let mut a = Array2::zeros((n, 3));
    let mut radii = Vec::with_capacity(n);
    let mut state: u64 = 0x1234_5678_9abc_def0;
    let mut rng = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as F) / ((1u64 << 53) as F)
    };
    for i in 0..n {
        a[[i, 0]] = rng() * l;
        a[[i, 1]] = rng() * l;
        a[[i, 2]] = rng() * l;
        radii.push(0.3 + 0.4 * rng());
    }
    let sb = cube_box(l);
    let cells = RadicalVoronoi.build(a.view(), &radii, &sb).unwrap();
    let rel = (cells.total_volume() - sb.volume()).abs() / sb.volume();
    assert!(
        rel < 1e-9,
        "Σvol={} box={} rel={rel:e}",
        cells.total_volume(),
        sb.volume()
    );
}

// ── ac-002 ──────────────────────────────────────────────────────────────────
#[test]
fn cubic_lattice_gives_unit_cubes() {
    let (a, sb) = cubic_lattice(3);
    let radii = vec![0.0; a.nrows()];
    let cells = RadicalVoronoi.build(a.view(), &radii, &sb).unwrap();
    for i in 0..cells.len() {
        assert!(
            (cells.volumes[i] - 1.0).abs() < 1e-9,
            "cell {i} vol {}",
            cells.volumes[i]
        );
        let nfaces = cells.faces[i].iter().filter(|f| f.area > 1e-9).count();
        assert_eq!(nfaces, 6, "cell {i} faces {nfaces}");
    }
    assert!((cells.total_volume() - 27.0).abs() < 1e-9);
}

#[test]
fn two_atom_radical_plane_is_analytic() {
    // Two atoms split along x, gaps both = g = Lx/2; analytic cell-0 width is
    // Lx/2 + (R0² − R1²)/g  (reduces to the midpoint bisector at equal radii).
    let lx: F = 8.0;
    let w: F = 4.0;
    let g = lx / 2.0;
    let (r0, r1) = (1.2, 0.5);
    let a = array![[lx * 0.25, 2.0, 2.0], [lx * 0.75, 2.0, 2.0]];
    let sb = ortho_box(lx, w, w);
    let cells = RadicalVoronoi.build(a.view(), &[r0, r1], &sb).unwrap();
    let width0 = lx / 2.0 + (r0 * r0 - r1 * r1) / g;
    let expect0 = width0 * w * w;
    assert!(
        (cells.volumes[0] - expect0).abs() < 1e-9,
        "cell0 {} expect {expect0}",
        cells.volumes[0]
    );
    assert!((cells.total_volume() - lx * w * w).abs() < 1e-9);

    // Equal radii → exact midpoint bisector → equal half-boxes.
    let cells_eq = RadicalVoronoi.build(a.view(), &[1.0, 1.0], &sb).unwrap();
    assert!((cells_eq.volumes[0] - lx * w * w / 2.0).abs() < 1e-9);
    assert!((cells_eq.volumes[1] - lx * w * w / 2.0).abs() < 1e-9);
}

// ── ac-003 ──────────────────────────────────────────────────────────────────
#[test]
fn periodic_neighbor_relation_is_symmetric_with_matching_areas() {
    // 2×2×2 lattice with mixed radii — exercises wrap-around neighbours.
    let (a, sb) = cubic_lattice(2);
    let n = a.nrows();
    let radii: Vec<F> = (0..n).map(|i| 0.4 + 0.05 * i as F).collect();
    let cells = RadicalVoronoi.build(a.view(), &radii, &sb).unwrap();

    for i in 0..n {
        for j in cells.neighbors(i) {
            let j = j as usize;
            // symmetry: j must list i
            assert!(cells.neighbors(j).contains(&(i as i64)), "{j} missing {i}");
            // shared face areas agree
            let ai: F = cells.faces[i]
                .iter()
                .filter(|f| f.neighbor == j as i64)
                .map(|f| f.area)
                .sum();
            let aj: F = cells.faces[j]
                .iter()
                .filter(|f| f.neighbor == i as i64)
                .map(|f| f.area)
                .sum();
            assert!(
                (ai - aj).abs() < 1e-9,
                "area {i}->{j}={ai} vs {j}->{i}={aj}"
            );
        }
    }
    // no residual box faces (fully bounded / wrapped)
    for i in 0..n {
        assert!(
            cells.faces[i]
                .iter()
                .all(|f| f.neighbor >= 0 || f.area < 1e-9)
        );
    }
}

// ── ac-004 ──────────────────────────────────────────────────────────────────
#[test]
fn domain_analysis_recovers_bilayer_and_percolating_domain() {
    // Bilayer: label by z-slab in a 4×4×4 lattice → exactly two domains.
    let (a, sb) = cubic_lattice(4);
    let n = a.nrows();
    let labels: Vec<i64> = (0..n).map(|_| 0).collect();
    let labels: Vec<i64> = labels
        .iter()
        .enumerate()
        .map(|(idx, _)| if a[[idx, 2]] < 2.0 { 0 } else { 1 })
        .collect();
    let cells = RadicalVoronoi.build(a.view(), &vec![0.0; n], &sb).unwrap();
    let dom = DomainAnalysis.analyze(&cells, &labels).unwrap();
    assert_eq!(dom.count, 2, "sizes {:?}", dom.sizes);
    assert_eq!(dom.sizes, vec![32, 32]);

    // Percolating: one label fills space except a single isolated site →
    // one large percolating domain + one singleton.
    let mut labels2 = vec![0i64; n];
    labels2[0] = 1;
    let dom2 = DomainAnalysis.analyze(&cells, &labels2).unwrap();
    assert_eq!(dom2.count, 2);
    assert_eq!(dom2.sizes, vec![n - 1, 1]);
    assert!((dom2.largest_fraction - (n - 1) as F / n as F).abs() < 1e-12);
}

// ── ac-005 ──────────────────────────────────────────────────────────────────
#[test]
fn void_analysis_recovers_single_cavity() {
    // 3×3×3 lattice with the centre atom removed; a probe at the empty site
    // recovers a single cavity ≈ the removed (unit) cell.
    let (full, sb) = cubic_lattice(3);
    let n = full.nrows();
    // centre site is (1,1,1)
    let centre = (0..n)
        .find(|&i| full[[i, 0]] == 1.0 && full[[i, 1]] == 1.0 && full[[i, 2]] == 1.0)
        .unwrap();

    // atoms = all but centre; append one probe at the centre site (index last)
    let mut pts: Vec<[F; 3]> = Vec::new();
    for i in 0..n {
        if i != centre {
            pts.push([full[[i, 0]], full[[i, 1]], full[[i, 2]]]);
        }
    }
    pts.push([1.0, 1.0, 1.0]); // probe
    let m = pts.len();
    let mut a = Array2::zeros((m, 3));
    for (r, p) in pts.iter().enumerate() {
        a[[r, 0]] = p[0];
        a[[r, 1]] = p[1];
        a[[r, 2]] = p[2];
    }
    let mut is_void = vec![false; m];
    is_void[m - 1] = true;

    let cells = RadicalVoronoi.build(a.view(), &vec![0.0; m], &sb).unwrap();
    let void = VoidAnalysis.analyze(&cells, &is_void, sb.volume()).unwrap();
    assert_eq!(
        void.cavity_volumes.len(),
        1,
        "cavities {:?}",
        void.cavity_volumes
    );
    assert!(
        (void.cavity_volumes[0] - 1.0).abs() < 1e-9,
        "cavity {}",
        void.cavity_volumes[0]
    );
    assert!((void.void_fraction - 1.0 / 27.0).abs() < 1e-9);
}

// ── ac-006: build gate (compiling under `voronoi` is the runtime check). ─────
#[test]
fn single_atom_cell_is_whole_box() {
    let l: F = 5.0;
    let a = array![[2.5, 2.5, 2.5]];
    let sb = cube_box(l);
    let cells = RadicalVoronoi.build(a.view(), &[0.0], &sb).unwrap();
    assert!(
        (cells.volumes[0] - l * l * l).abs() < 1e-9,
        "vol {}",
        cells.volumes[0]
    );
}
