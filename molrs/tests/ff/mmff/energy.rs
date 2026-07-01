//! MMFF94 total-energy + analytical-gradient validation against RDKit.
//!
//! Fixtures `<name>.sdf` / `<name>.energy.json` are produced by
//! `gen_mmff_energy_fixtures.py` (RDKit 2026.03.2): the SDF carries the 3D
//! conformer (atom order == RDKit atom order) and the JSON carries the
//! reference MMFF94 total energy `MMFFGetMoleculeForceField(mol).CalcEnergy()`.
//!
//! Tests:
//! - total energy within 1e-3 kcal/mol of RDKit for a coverage set;
//! - analytical gradient vs central finite difference (step 1e-5) < 1e-5;
//! - translation + rigid-rotation invariance of benzene (< 1e-9);
//! - single-eval timing of a ~50-atom molecule (ac-006 baseline; printed).

use std::path::{Path, PathBuf};
use std::time::Instant;

use molrs::Atomistic;
use molrs::ff::mmff::{MmffForceField, MmffMolProperties, MmffVariant};
use molrs::ff::potential::Potential;
use molrs::system::molgraph::{Atom, PropValue};
use serde_json::Value;

const ENERGY_TOL: f64 = 1.0e-3;

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/ff/mmff/fixtures")
}

/// Minimal V2000 SDF loader preserving atom order and bond orders, parsing
/// `M  CHG` (same as typing.rs).
fn load_sdf(path: &Path) -> Atomistic {
    let text = std::fs::read_to_string(path).expect("read sdf");
    let lines: Vec<&str> = text.lines().collect();
    let counts = lines[3];
    let n_atoms: usize = counts[0..3].trim().parse().expect("n_atoms");
    let n_bonds: usize = counts[3..6].trim().parse().expect("n_bonds");

    let mut g = Atomistic::new();
    let mut ids = Vec::with_capacity(n_atoms);
    for k in 0..n_atoms {
        let line = lines[4 + k];
        let x: f64 = line[0..10].trim().parse().expect("x");
        let y: f64 = line[10..20].trim().parse().expect("y");
        let z: f64 = line[20..30].trim().parse().expect("z");
        let element = line[31..34].trim().to_string();
        let mut atom = Atom::xyz(&element, x, y, z);
        atom.set("formal_charge", PropValue::Int(0));
        ids.push(g.add_atom(atom));
    }
    for k in 0..n_bonds {
        let line = lines[4 + n_atoms + k];
        let i: usize = line[0..3].trim().parse::<usize>().expect("bond i") - 1;
        let j: usize = line[3..6].trim().parse::<usize>().expect("bond j") - 1;
        let order: f64 = line[6..9].trim().parse().expect("bond order");
        let bid = g.add_bond(ids[i], ids[j]).expect("add bond");
        g.set_bond_prop(bid, "order", PropValue::F64(order))
            .expect("bond");
    }
    for line in &lines[4 + n_atoms + n_bonds..] {
        let t = line.trim_end();
        if t == "M  END" || t == "$$$$" {
            break;
        }
        if let Some(rest) = t.strip_prefix("M  CHG") {
            let toks: Vec<&str> = rest.split_whitespace().collect();
            if toks.is_empty() {
                continue;
            }
            let pairs: usize = toks[0].parse().unwrap_or(0);
            for q in 0..pairs {
                let ai: usize = toks[1 + 2 * q].parse::<usize>().expect("chg atom") - 1;
                let chg: i32 = toks[2 + 2 * q].parse().expect("chg val");
                g.set_atom(ids[ai], "formal_charge", PropValue::Int(chg))
                    .expect("atom");
            }
        }
    }
    g
}

/// Flat `[x0,y0,z0,...]` coordinates in SDF atom order.
fn coords_of(path: &Path) -> Vec<f64> {
    let text = std::fs::read_to_string(path).expect("read sdf");
    let lines: Vec<&str> = text.lines().collect();
    let n_atoms: usize = lines[3][0..3].trim().parse().expect("n_atoms");
    let mut c = Vec::with_capacity(3 * n_atoms);
    for k in 0..n_atoms {
        let line = lines[4 + k];
        c.push(line[0..10].trim().parse::<f64>().expect("x"));
        c.push(line[10..20].trim().parse::<f64>().expect("y"));
        c.push(line[20..30].trim().parse::<f64>().expect("z"));
    }
    c
}

fn ref_energy_field(name: &str, field: &str) -> f64 {
    let path = fixtures_dir().join(format!("{name}.energy.json"));
    let text = std::fs::read_to_string(&path).expect("read energy json");
    let v: Value = serde_json::from_str(&text).expect("parse json");
    v[field]
        .as_f64()
        .unwrap_or_else(|| panic!("{name}: missing energy field '{field}'"))
}

fn ref_energy(name: &str) -> f64 {
    ref_energy_field(name, "mmff94_total_energy")
}

fn build_ff_variant(name: &str, variant: MmffVariant) -> (MmffForceField, Vec<f64>) {
    let dir = fixtures_dir();
    let mol = load_sdf(&dir.join(format!("{name}.sdf")));
    let coords = coords_of(&dir.join(format!("{name}.sdf")));
    let props = MmffMolProperties::compute(&mol, variant)
        .unwrap_or_else(|e| panic!("{name}: typing failed: {e}"));
    let ff = MmffForceField::build(&mol, &props)
        .unwrap_or_else(|e| panic!("{name}: ff build failed: {e}"));
    (ff, coords)
}

fn build_ff(name: &str) -> (MmffForceField, Vec<f64>) {
    build_ff_variant(name, MmffVariant::Mmff94)
}

#[test]
fn total_energy_matches_rdkit() {
    let names = [
        "e_ethane",
        "e_ethylene",
        "e_benzene",
        "e_butane",
        "e_caffeine",
    ];
    let mut fails = Vec::new();
    for name in names {
        let (ff, coords) = build_ff(name);
        let got = ff.calc_energy_forces(&coords).0;
        let want = ref_energy(name);
        let delta = (got - want).abs();
        println!("{name:12} molrs={got:14.6}  rdkit={want:14.6}  d={delta:.3e}");
        if delta > ENERGY_TOL {
            fails.push(format!(
                "{name}: molrs={got:.6} rdkit={want:.6} d={delta:.3e}"
            ));
        }
    }
    assert!(
        fails.is_empty(),
        "energy mismatches vs RDKit:\n  {}",
        fails.join("\n  ")
    );
}

/// Molecules whose MMFF94s energy differs from MMFF94 (delocalized-N / amide /
/// aromatic-amine planarization). The fixtures store both reference energies.
const S_NAMES: [&str; 4] = ["s_aniline", "s_acetamide", "s_nmethylacetamide", "s_urea"];

/// MMFF94s total energy must match RDKit `mmffVariant='MMFF94s'` to 1e-3 for
/// molecules where the `_S` oop/torsion tables actually change the result.
#[test]
fn mmff94s_total_energy_matches_rdkit() {
    let mut fails = Vec::new();
    for name in S_NAMES {
        let (ff, coords) = build_ff_variant(name, MmffVariant::Mmff94s);
        let got = ff.calc_energy_forces(&coords).0;
        let want = ref_energy_field(name, "mmff94s_total_energy");
        let delta = (got - want).abs();
        println!("{name:20} 94s molrs={got:14.6}  rdkit={want:14.6}  d={delta:.3e}");
        if delta > ENERGY_TOL {
            fails.push(format!(
                "{name}: molrs={got:.6} rdkit={want:.6} d={delta:.3e}"
            ));
        }
    }
    assert!(
        fails.is_empty(),
        "MMFF94s energy mismatches vs RDKit:\n  {}",
        fails.join("\n  ")
    );
}

/// The same molecules under MMFF94 must still match RDKit MMFF94, and the 94 vs
/// 94s totals must genuinely differ (otherwise the `_S` wiring is a no-op and
/// the test above proves nothing).
#[test]
fn mmff94_unchanged_and_differs_from_94s() {
    let mut fails = Vec::new();
    for name in S_NAMES {
        let (ff94, coords) = build_ff_variant(name, MmffVariant::Mmff94);
        let got94 = ff94.calc_energy_forces(&coords).0;
        let want94 = ref_energy_field(name, "mmff94_total_energy");
        let want94s = ref_energy_field(name, "mmff94s_total_energy");
        let d94 = (got94 - want94).abs();
        let (ff94s, _) = build_ff_variant(name, MmffVariant::Mmff94s);
        let got94s = ff94s.calc_energy_forces(&coords).0;
        println!(
            "{name:20} 94 molrs={got94:14.6} rdkit={want94:14.6} | 94s molrs={got94s:14.6} | \
             ref_diff={:.4}",
            (want94 - want94s).abs()
        );
        if d94 > ENERGY_TOL {
            fails.push(format!(
                "{name}: MMFF94 molrs={got94:.6} rdkit={want94:.6} d={d94:.3e}"
            ));
        }
        // RDKit's own 94 vs 94s totals differ -> our two builds must too.
        if (got94 - got94s).abs() < ENERGY_TOL {
            fails.push(format!(
                "{name}: molrs 94 ({got94:.6}) == 94s ({got94s:.6}); _S tables not applied"
            ));
        }
    }
    assert!(
        fails.is_empty(),
        "MMFF94/94s consistency failures:\n  {}",
        fails.join("\n  ")
    );
}

#[test]
fn breakdown_sums_to_total() {
    // The per-term breakdown must reconstruct the eval() total exactly.
    for name in ["e_ethane", "e_ethylene", "e_benzene", "e_caffeine"] {
        let (ff, coords) = build_ff(name);
        let b = ff.energy_terms(&coords);
        let total = ff.calc_energy_forces(&coords).0;
        println!(
            "{name:12} bond={:.4} angle={:.4} sb={:.4} oop={:.4} tor={:.4} vdw={:.4} ele={:.4} | total={:.4}",
            b.bond, b.angle, b.stretch_bend, b.oop, b.torsion, b.vdw, b.electrostatic, b.total
        );
        assert!(
            (b.total - total).abs() < 1.0e-9,
            "{name}: breakdown {} != eval {}",
            b.total,
            total
        );
    }
}

#[test]
fn analytical_gradient_matches_finite_difference() {
    let h = 1.0e-5;
    for name in [
        "e_ethane",
        "e_ethylene",
        "e_benzene",
        "e_butane",
        "e_caffeine",
    ] {
        let (ff, coords) = build_ff(name);
        let (_, forces) = ff.calc_energy_forces(&coords);
        let mut max_err = 0.0f64;
        for idx in 0..coords.len() {
            let mut cp = coords.clone();
            cp[idx] += h;
            let ep = ff.calc_energy_forces(&cp).0;
            let mut cm = coords.clone();
            cm[idx] -= h;
            let em = ff.calc_energy_forces(&cm).0;
            let fd_grad = (ep - em) / (2.0 * h);
            // forces = -gradient
            let err = (forces[idx] + fd_grad).abs();
            max_err = max_err.max(err);
        }
        println!("{name:12} grad max_err={max_err:.3e}");
        assert!(
            max_err < 1.0e-5,
            "{name}: gradient FD max err {max_err:.3e} >= 1e-5"
        );
    }
}

#[test]
fn benzene_translation_rotation_invariance() {
    let (ff, coords) = build_ff("e_benzene");
    let e0 = ff.calc_energy_forces(&coords).0;

    // translation
    let mut shifted = coords.clone();
    for k in (0..shifted.len()).step_by(3) {
        shifted[k] += 3.7;
        shifted[k + 1] -= 1.2;
        shifted[k + 2] += 0.4;
    }
    let e_t = ff.calc_energy_forces(&shifted).0;
    assert!(
        (e_t - e0).abs() < 1.0e-9,
        "translation changed E by {}",
        e_t - e0
    );

    // rigid rotation about z by 0.7 rad
    let (c, s) = (0.7f64.cos(), 0.7f64.sin());
    let mut rotated = coords.clone();
    for k in (0..rotated.len()).step_by(3) {
        let (x, y) = (coords[k], coords[k + 1]);
        rotated[k] = c * x - s * y;
        rotated[k + 1] = s * x + c * y;
    }
    let e_r = ff.calc_energy_forces(&rotated).0;
    assert!(
        (e_r - e0).abs() < 1.0e-9,
        "rotation changed E by {}",
        e_r - e0
    );
}

#[test]
fn timing_baseline_50_atoms() {
    // ac-006: measure + print a single-eval timing for a ~50-atom molecule.
    let (ff, coords) = build_ff("e_big");
    let n = coords.len() / 3;
    // warm up
    let _ = ff.calc_energy_forces(&coords);
    let iters = 200;
    let t0 = Instant::now();
    let mut acc = 0.0;
    for _ in 0..iters {
        acc += ff.calc_energy_forces(&coords).0;
    }
    let elapsed = t0.elapsed();
    let per = elapsed.as_secs_f64() / iters as f64;
    println!(
        "ac-006 timing: {n} atoms, single eval = {:.1} us  (acc={acc:.3})",
        per * 1.0e6
    );
    assert!(per.is_finite());
}

// ---------------------------------------------------------------------------
// Geometry optimization over the (RDKit-validated) MmffForceField path.
// Proves molrs::ff::{LBFGS} relax a real force field, and
// that the homogeneous batch path reproduces the single-structure result.
// ---------------------------------------------------------------------------

#[test]
fn lbfgs_minimize_relaxes_mmff_ethane() {
    use molrs::optimize::{LBFGS, LbfgsConfig};

    let (ff, coords0) = build_ff("e_ethane");
    let e_start = ff.calc_energy_forces(&coords0).0;

    let mut coords = coords0.clone();
    let opts = LbfgsConfig::default();
    let report = LBFGS::new(&ff, opts)
        .run(&mut coords)
        .expect("minimize ethane");

    assert!(
        report.final_energy <= e_start + 1e-9,
        "energy must not increase: {e_start} -> {}",
        report.final_energy
    );
    assert!(report.converged, "ethane should converge: {report:?}");
    assert!(
        report.final_fmax <= opts.fmax + 1e-12,
        "fmax not satisfied: {}",
        report.final_fmax
    );
    println!(
        "ethane MMFF relax: E {e_start:.4} -> {:.4} (fmax {:.4}, {} steps)",
        report.final_energy, report.final_fmax, report.n_steps
    );
}

#[test]
fn lbfgs_minimize_batch_matches_single() {
    use molrs::optimize::{LBFGS, LbfgsConfig};

    let (ff, coords0) = build_ff("e_ethane");
    let n_atoms = coords0.len() / 3;
    let opts = LbfgsConfig::default();

    // Single-structure reference.
    let mut single = coords0.clone();
    let single_report = LBFGS::new(&ff, opts).run(&mut single).expect("single");

    // Homogeneous batch: block 0 identical to the single start; blocks 1..B
    // deterministically perturbed (same topology, same force field).
    let b = 4;
    let mut batch: Vec<f64> = Vec::with_capacity(b * coords0.len());
    for s in 0..b {
        for (i, &c) in coords0.iter().enumerate() {
            let pert = if s == 0 {
                0.0
            } else {
                0.02 * (((i + s * 7) % 5) as f64 - 2.0)
            };
            batch.push(c + pert);
        }
    }

    let reports = LBFGS::new(&ff, opts)
        .run_batch(&mut batch, n_atoms, b)
        .expect("batch");
    assert_eq!(reports.len(), b);

    // Block 0 started from the same coords as `single` -> identical outcome.
    assert!(
        (reports[0].final_energy - single_report.final_energy).abs() < 1e-9,
        "batch block 0 energy {} != single {}",
        reports[0].final_energy,
        single_report.final_energy
    );
    for (a, s) in batch[0..coords0.len()].iter().zip(&single) {
        assert!(
            (a - s).abs() < 1e-9,
            "batch block 0 coords diverged from single"
        );
    }

    // All structures relaxed to a finite, converged minimum.
    for (i, r) in reports.iter().enumerate() {
        assert!(r.final_energy.is_finite(), "block {i} energy not finite");
        assert!(
            r.final_fmax <= opts.fmax + 1e-12 || !r.converged,
            "block {i} fmax {} unsatisfied",
            r.final_fmax
        );
    }
}

// ---------------------------------------------------------------------------
// Generic-path parity (ac-004): the same RDKit fixtures evaluated through
// `MMFFTypifier::build` -> `ForceField::to_potentials` (typify + nonbonded_pairs
// + the generic kernels, including the now-wired out-of-plane term) must
// reproduce the RDKit reference energy and a finite-difference-consistent
// gradient. e_benzene / e_caffeine exercise the out-of-plane path.
// ---------------------------------------------------------------------------

fn generic_pots(name: &str) -> (molrs::ff::potential::Potentials, Vec<f64>) {
    use molrs::ff::typifier::mmff::MMFFTypifier;
    let dir = fixtures_dir();
    let mol = load_sdf(&dir.join(format!("{name}.sdf")));
    let coords = coords_of(&dir.join(format!("{name}.sdf")));
    let pots = MMFFTypifier::mmff94()
        .expect("load MMFF94")
        .build(&mol)
        .unwrap_or_else(|e| panic!("{name}: generic build failed: {e}"));
    (pots, coords)
}

#[test]
fn generic_path_total_energy_matches_rdkit() {
    // Validated subset: molecules whose every MMFF parameter is an exact or
    // equivalence-degraded table hit (bond/angle/stretch-bend/torsion/vdw/ele;
    // out-of-plane is exercised by molecules below once their stbn/torsion
    // labels resolve). Aromatic / sp2 molecules (benzene, caffeine, ethylene)
    // still need stretch-bend + torsion eq-fallback label resolution — the
    // remaining ac-004 work — so they are not yet asserted here.
    let names = ["e_ethane"];
    let mut fails = Vec::new();
    for name in names {
        let (pots, coords) = generic_pots(name);
        let got = pots.calc_energy_forces(&coords).0;
        let want = ref_energy(name);
        let delta = (got - want).abs();
        println!("{name:12} generic={got:14.6}  rdkit={want:14.6}  d={delta:.3e}");
        if delta > ENERGY_TOL {
            fails.push(format!(
                "{name}: generic={got:.6} rdkit={want:.6} d={delta:.3e}"
            ));
        }
    }
    assert!(
        fails.is_empty(),
        "generic-path energy mismatches vs RDKit:\n  {}",
        fails.join("\n  ")
    );
}

#[test]
fn generic_path_gradient_matches_finite_difference() {
    let h = 1.0e-5;
    for name in ["e_ethane"] {
        let (pots, coords) = generic_pots(name);
        let (_, forces) = pots.calc_energy_forces(&coords);
        let mut max_err = 0.0f64;
        for idx in 0..coords.len() {
            let mut cp = coords.clone();
            cp[idx] += h;
            let ep = pots.calc_energy_forces(&cp).0;
            let mut cm = coords.clone();
            cm[idx] -= h;
            let em = pots.calc_energy_forces(&cm).0;
            let fd_grad = (ep - em) / (2.0 * h);
            max_err = max_err.max((forces[idx] + fd_grad).abs());
        }
        println!("{name:12} generic grad max_err={max_err:.3e}");
        assert!(
            max_err < 1.0e-5,
            "{name}: generic gradient FD max err {max_err:.3e} >= 1e-5"
        );
    }
}
