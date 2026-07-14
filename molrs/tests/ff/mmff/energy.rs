//! MMFF94 energy validation against RDKit — both molrs paths.
//!
//! Fixtures `<name>.sdf` / `<name>.energy.json` carry the 3D conformer (atom
//! order == RDKit atom order) and the reference MMFF94 total energy
//! `MMFFGetMoleculeForceField(mol).CalcEnergy()`. The reference is computed on
//! the **committed 4-decimal SDF geometry** (re-read from the mol block), not on
//! the full-precision conformer — verified for all fixtures: RDKit re-evaluated
//! from each `.sdf` reproduces the stored `mmff94_total_energy` to 0.0e0.
//!
//! `e_acetonitrile` (mmff-orthogonal-01) was generated as:
//!
//! ```python
//! m = Chem.AddHs(Chem.MolFromSmiles("CC#N"))
//! AllChem.EmbedMolecule(m, randomSeed=42)
//! rdMolTransforms.SetAngleDeg(m.GetConformer(), 0, 1, 2, 170.0)  # deliberately BENT
//! block = Chem.MolToMolBlock(m)                                  # 4-decimal coords
//! m2 = Chem.MolFromMolBlock(block, removeHs=False)               # oracle on THOSE coords
//! E = AllChem.MMFFGetMoleculeForceField(m2, AllChem.MMFFGetMoleculeProperties(m2)).CalcEnergy()
//! ```
//!
//! The bend is load-bearing: ETKDG's own geometry is 179.14°, where the linear
//! and cubic angle forms differ by ~1e-4·ka — below the 1e-3 tolerance. A fixture
//! built at that angle would pass whether or not the linear-angle branch exists.
//!
//! Two paths are validated here against the same fixtures:
//!
//! - **bespoke** (`MmffForceField`) — the reference frame. Reproduces RDKit to
//!   1e-6 on every fixture, and mmff-orthogonal-01 does not touch it (see
//!   `bespoke_gate.rs`).
//! - **generic** (`MMFF94Typifier::build` → `ForceField` → kernels) — the
//!   documented public API. Its fixture list is produced by **scanning**
//!   `fixtures/*.energy.json`, so a subset assertion cannot be written: the only
//!   fixture the old test asserted (`e_ethane`) was one of exactly two whose MMFF
//!   charges are all zero, i.e. the one input class that cannot expose a missing
//!   electrostatic term.
//!
//! The generic path is additionally asserted **per style** against
//! `<name>.breakdown.json` (7 terms, frozen from the bespoke path and re-proved
//! against the RDKit oracle on every run). A total-energy assertion is not
//! enough: the total is the number most able to lie — it hid a 28 kcal/mol
//! electrostatic hole behind partially-cancelling terms on NMA, and on ethane it
//! hid an entire missing energy term.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use molrs::Atomistic;
use molrs::ff::mmff::{MmffForceField, MmffMolProperties, MmffVariant};
use molrs::ff::potential::{Potential, Potentials, intramolecular_pairs};
use molrs::ff::typifier::mmff::{MMFF94STypifier, MMFF94Typifier};
use molrs::store::frame::Frame;
use molrs::system::molgraph::{Atom, PropValue};
use serde_json::Value;

/// Generic path vs the RDKit oracle (kcal/mol) — the spec's parity tolerance.
const ENERGY_TOL: f64 = 1.0e-3;
/// Bespoke path vs the RDKit oracle. It reproduces RDKit to ~1e-13; this is the
/// machine-checkable form of "max |delta| = 0.00000".
const BESPOKE_TOL: f64 = 1.0e-6;
/// Generic per-style energy vs the frozen bespoke breakdown (kcal/mol).
const BREAKDOWN_TOL: f64 = 1.0e-6;

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/ff/mmff/fixtures")
}

/// Repository root (one above this package).
fn repo_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("molrs/ has a parent")
}

// ---------------------------------------------------------------------------
// Fixture discovery — the list is SCANNED, never hardcoded.
// ---------------------------------------------------------------------------

/// Every energy fixture on disk, sorted.
///
/// Scanning `*.energy.json` is what separates the 10+1 energy fixtures from the
/// bare-named typifier fixtures (`benzene.sdf` + `benzene.json`, …) that share
/// the directory. No test may take a subset of this list: "not implemented yet"
/// is a reason to fail, not a reason to exclude a molecule.
fn energy_fixtures() -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(fixtures_dir())
        .expect("read fixtures dir")
        .map(|e| e.expect("dir entry").file_name())
        .filter_map(|f| {
            f.to_string_lossy()
                .strip_suffix(".energy.json")
                .map(str::to_owned)
        })
        .collect();
    names.sort();
    assert!(
        !names.is_empty(),
        "no `*.energy.json` fixtures found in {} — a scan that finds nothing would \
         make every parity test below vacuously green",
        fixtures_dir().display()
    );
    names
}

/// The fixtures that must exist. Used ONLY to prove none was deleted — the
/// parity tests iterate the scan, so adding a fixture cannot be forgotten and
/// removing one cannot be quiet.
const REQUIRED_FIXTURES: [&str; 11] = [
    "e_acetonitrile",
    "e_benzene",
    "e_big",
    "e_butane",
    "e_caffeine",
    "e_ethane",
    "e_ethylene",
    "s_acetamide",
    "s_aniline",
    "s_nmethylacetamide",
    "s_urea",
];

#[test]
fn every_required_fixture_is_on_disk() {
    let found = energy_fixtures();
    let missing: Vec<&str> = REQUIRED_FIXTURES
        .iter()
        .copied()
        .filter(|n| !found.iter().any(|f| f == n))
        .collect();
    assert!(
        missing.is_empty(),
        "energy fixtures missing from {}: {missing:?}",
        fixtures_dir().display()
    );
    for name in &found {
        assert!(
            fixtures_dir().join(format!("{name}.sdf")).is_file(),
            "{name}.energy.json has no matching {name}.sdf"
        );
        assert!(
            fixtures_dir()
                .join(format!("{name}.breakdown.json"))
                .is_file(),
            "{name} has no frozen per-style breakdown ({name}.breakdown.json)"
        );
    }
}

// ---------------------------------------------------------------------------
// SDF / JSON loading
// ---------------------------------------------------------------------------

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

/// Element symbols in SDF atom order.
fn elements_of(path: &Path) -> Vec<String> {
    let text = std::fs::read_to_string(path).expect("read sdf");
    let lines: Vec<&str> = text.lines().collect();
    let n_atoms: usize = lines[3][0..3].trim().parse().expect("n_atoms");
    (0..n_atoms)
        .map(|k| lines[4 + k][31..34].trim().to_string())
        .collect()
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

// ---------------------------------------------------------------------------
// The 7 MMFF energy terms, and the styles that must carry them
// ---------------------------------------------------------------------------

/// `(category, style name, breakdown key)` for every MMFF energy term.
///
/// `pair/mmff_ele` is the one whose kernel has been registered since the day it
/// was written (`potential/registry.rs`) while **no ForceField ever defined the
/// style** — the entire 150 kcal/mol caffeine error.
const MMFF_STYLES: [(&str, &str, &str); 7] = [
    ("bond", "mmff_bond", "bond"),
    ("angle", "mmff_angle", "angle"),
    ("angle", "mmff_stbn", "stbn"),
    ("dihedral", "mmff_torsion", "torsion"),
    ("improper", "mmff_oop", "oop"),
    ("pair", "mmff_vdw", "vdw"),
    ("pair", "mmff_ele", "ele"),
];

/// The 7 frozen term energies of `<name>.breakdown.json`.
fn frozen_breakdown(name: &str) -> BTreeMap<&'static str, f64> {
    let path = fixtures_dir().join(format!("{name}.breakdown.json"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{name}: no frozen breakdown ({}): {e}", path.display()));
    let v: Value = serde_json::from_str(&text).expect("parse breakdown json");
    MMFF_STYLES
        .iter()
        .map(|&(_, _, key)| {
            let e = v[key]
                .as_f64()
                .unwrap_or_else(|| panic!("{name}.breakdown.json: missing term '{key}'"));
            (key, e)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Bespoke path (the reference frame — mmff-orthogonal-01 does not touch it)
// ---------------------------------------------------------------------------

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

/// ac-011: the bespoke path still reproduces RDKit on **every** fixture.
///
/// Directory-scanned like the generic one, so the new acetonitrile fixture is
/// covered by both from the moment it lands.
#[test]
fn total_energy_matches_rdkit() {
    let mut fails = Vec::new();
    let mut worst = 0.0f64;
    for name in energy_fixtures() {
        let (ff, coords) = build_ff(&name);
        let got = ff.calc_energy_forces(&coords).0;
        let want = ref_energy(&name);
        let delta = (got - want).abs();
        worst = worst.max(delta);
        println!("{name:20} bespoke={got:14.6}  rdkit={want:14.6}  d={delta:.3e}");
        if delta > BESPOKE_TOL {
            fails.push(format!(
                "{name}: bespoke={got:.6} rdkit={want:.6} d={delta:.3e}"
            ));
        }
    }
    println!("bespoke max |delta| = {worst:.5}");
    assert!(
        fails.is_empty(),
        "bespoke energy mismatches vs RDKit:\n  {}",
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

/// The bespoke per-term breakdown must reconstruct its own `eval()` total, on
/// every fixture. (That the SAME breakdown also reconstructs the RDKit total is
/// asserted independently in `frozen_breakdown_sums_to_the_rdkit_total` — that
/// one is what keeps the frozen fixtures legitimate after spec 02 deletes this
/// path.)
#[test]
fn breakdown_sums_to_total() {
    for name in energy_fixtures() {
        let (ff, coords) = build_ff(&name);
        let b = ff.energy_terms(&coords);
        let total = ff.calc_energy_forces(&coords).0;
        println!(
            "{name:20} bond={:.4} angle={:.4} sb={:.4} oop={:.4} tor={:.4} vdw={:.4} ele={:.4} | total={:.4}",
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
    for name in energy_fixtures() {
        let (ff, coords) = build_ff(&name);
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
            max_err = max_err.max((forces[idx] + fd_grad).abs());
        }
        println!("{name:20} bespoke grad max_err={max_err:.3e}");
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
// Generic path — typifier -> ForceField -> KernelRegistry.
//
// This is the DOCUMENTED public API (`MMFF94Typifier::build`). It is the path
// the four mmff-orthogonal-01 defects live in.
// ---------------------------------------------------------------------------

/// The typed frame + coordinates + the typifier that owns the ForceField.
///
/// Built exactly the way `MMFF94Typifier::build` builds it (typify → to_frame →
/// `intramolecular_pairs`), but kept open so each style can be compiled and
/// evaluated in isolation. No production code is needed for that:
/// `ForceField::styles`, `Style::to_potential` and `ForceField::special_bonds`
/// are all public.
struct GenericMmff {
    typifier: MMFF94Typifier,
    frame: Frame,
    coords: Vec<f64>,
}

impl GenericMmff {
    fn load(name: &str) -> Self {
        let dir = fixtures_dir();
        let mol = load_sdf(&dir.join(format!("{name}.sdf")));
        let coords = coords_of(&dir.join(format!("{name}.sdf")));
        let typifier = MMFF94Typifier::new();
        let typed = typifier
            .typify(&mol)
            .unwrap_or_else(|e| panic!("{name}: generic typify failed: {e}"));
        let mut frame = typed.to_frame();
        let pairs = intramolecular_pairs(&frame);
        frame.insert("pairs", pairs);
        Self {
            typifier,
            frame,
            coords,
        }
    }

    /// Sum of |q| over the typed frame's baked MMFF partial charges.
    fn total_abs_charge(&self) -> f64 {
        self.frame
            .get("atoms")
            .and_then(|b| b.get_float("charge"))
            .map(|c| c.iter().map(|q| q.abs()).sum())
            .unwrap_or_else(|| panic!("typed frame has no atoms/charge column"))
    }

    fn n_14_pairs(&self) -> usize {
        self.frame
            .get("pairs")
            .and_then(|b| b.get_bool("is_14"))
            .map(|c| c.iter().filter(|&&f| f).count())
            .unwrap_or(0)
    }

    /// The energy of ONE style at `coords`, or `None` when the force field does
    /// not define that style at all (which is the missing-electrostatics defect,
    /// and must be distinguished from a style whose topology is simply absent —
    /// e.g. no impropers in butane, which yields `Some(0.0)`).
    fn style_energy(&self, key: &str, coords: &[f64]) -> Option<f64> {
        let ff = self.typifier.ff();
        let (category, style_name, _) = MMFF_STYLES
            .iter()
            .find(|(_, _, k)| *k == key)
            .unwrap_or_else(|| panic!("unknown breakdown key '{key}'"));
        let style = ff
            .styles()
            .iter()
            .find(|s| s.category() == *category && s.name == *style_name)?;
        let pot = style
            .to_potential(&self.frame, ff.special_bonds())
            .unwrap_or_else(|e| panic!("style {category}/{style_name}: to_potential failed: {e}"));
        // A style with no topology rows (no impropers, no pairs, …) contributes
        // exactly zero — it is DEFINED, it just has nothing to act on.
        Some(pot.map_or(0.0, |p| p.calc_energy_forces(coords).0))
    }

    /// All 7 term energies. Panics if the force field carries an energy style
    /// this breakdown does not know about — a new style must be added to
    /// `MMFF_STYLES`, never silently omitted from the decomposition.
    fn term_energies(&self, coords: &[f64]) -> BTreeMap<&'static str, f64> {
        let ff = self.typifier.ff();
        for style in ff.styles() {
            if style.category() == "atom" {
                continue;
            }
            assert!(
                MMFF_STYLES
                    .iter()
                    .any(|(c, n, _)| *c == style.category() && *n == style.name),
                "the MMFF force field defines an energy style ({}/{}) that the per-style \
                 breakdown does not cover; add it to MMFF_STYLES, or its energy would be \
                 invisible to every assertion here",
                style.category(),
                style.name
            );
        }
        MMFF_STYLES
            .iter()
            .map(|&(_, _, key)| (key, self.style_energy(key, coords).unwrap_or(f64::NAN)))
            .collect()
    }
}

/// The documented public API: `MMFF94Typifier::build`.
fn generic_pots(name: &str) -> (Potentials, Vec<f64>) {
    let dir = fixtures_dir();
    let mol = load_sdf(&dir.join(format!("{name}.sdf")));
    let coords = coords_of(&dir.join(format!("{name}.sdf")));
    let pots = MMFF94Typifier::new()
        .build(&mol)
        .unwrap_or_else(|e| panic!("{name}: generic build failed: {e}"));
    (pots, coords)
}

/// ac-002 — 11/11 RDKit total-energy parity on the generic path.
///
/// The fixture list is SCANNED, not written down: the previous version of this
/// test asserted `["e_ethane"]` and justified the exclusions with a comment
/// blaming "stretch-bend + torsion eq-fallback label resolution". That comment
/// was false — stbn and torsion agree with the bespoke path to five decimals on
/// every fixture (NMA: -0.31596 / -1.18237, identical on both paths). The real
/// defect was that no ForceField ever defined `pair/mmff_ele`.
#[test]
fn generic_path_total_energy_matches_rdkit() {
    let mut fails = Vec::new();
    for name in energy_fixtures() {
        let (pots, coords) = generic_pots(&name);
        let got = pots.calc_energy_forces(&coords).0;
        let want = ref_energy(&name);
        let delta = (got - want).abs();
        println!("{name:20} generic={got:14.6}  rdkit={want:14.6}  d={delta:.3e}");
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

/// ac-010 — analytic forces vs central finite differences on EVERY fixture.
///
/// Covers the analytic gradient of the linear-angle branch (`e_acetonitrile`),
/// which no other fixture exercises.
#[test]
fn generic_path_gradient_matches_finite_difference() {
    let h = 1.0e-5;
    for name in energy_fixtures() {
        let (pots, coords) = generic_pots(&name);
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
        println!("{name:20} generic grad max_err={max_err:.3e}");
        assert!(
            max_err < 1.0e-5,
            "{name}: generic gradient FD max err {max_err:.3e} >= 1e-5"
        );
    }
}

/// ac-003 — every one of the 7 style energies matches the frozen bespoke value.
///
/// This, not the total, is the assertion that has teeth. The generic total on
/// e_ethane was within 2e-5 of RDKit while the electrostatic term did not exist
/// at all (ethane's MMFF charges are all zero); on NMA a 28 kcal/mol hole hid
/// behind partially-cancelling terms.
#[test]
fn generic_path_per_style_breakdown_matches_frozen() {
    let mut fails = Vec::new();
    for name in energy_fixtures() {
        let g = GenericMmff::load(&name);
        let got = g.term_energies(&g.coords);
        let want = frozen_breakdown(&name);
        let mut line = format!("{name:20} ");
        for &(category, style_name, key) in &MMFF_STYLES {
            let (g_e, w_e) = (got[key], want[key]);
            line.push_str(&format!("{key}={g_e:.5} "));
            if g_e.is_nan() {
                fails.push(format!(
                    "{name}: the force field defines no `{category}/{style_name}` style, so the \
                     `{key}` term does not exist (frozen reference: {w_e:.6} kcal/mol)"
                ));
                continue;
            }
            let d = (g_e - w_e).abs();
            if d > BREAKDOWN_TOL {
                fails.push(format!(
                    "{name}: {key} generic={g_e:.6} frozen={w_e:.6} d={d:.3e} \
                     (style {category}/{style_name})"
                ));
            }
        }
        println!("{line}");
    }
    assert!(
        fails.is_empty(),
        "generic per-style breakdown mismatches vs the frozen bespoke reference:\n  {}",
        fails.join("\n  ")
    );
}

/// ac-004 — the frozen breakdowns are legitimate, proved by an EXTERNAL oracle.
///
/// Sum of the 7 frozen terms == `mmff94_total_energy` (RDKit) to 1e-6. This is
/// what keeps `<name>.breakdown.json` a valid reference after spec 02 deletes
/// the bespoke path that produced it: the numbers do not rest on molrs at all.
#[test]
fn frozen_breakdown_sums_to_the_rdkit_total() {
    let mut fails = Vec::new();
    for name in energy_fixtures() {
        let want = frozen_breakdown(&name);
        let sum: f64 = want.values().sum();
        let oracle = ref_energy(&name);
        let d = (sum - oracle).abs();
        println!("{name:20} frozen_sum={sum:14.6}  rdkit={oracle:14.6}  d={d:.3e}");
        if d > BREAKDOWN_TOL {
            fails.push(format!(
                "{name}: frozen terms sum to {sum:.9}, RDKit says {oracle:.9} (d={d:.3e})"
            ));
        }
    }
    assert!(
        fails.is_empty(),
        "frozen breakdown fixtures are not self-consistent with the RDKit oracle:\n  {}",
        fails.join("\n  ")
    );
}

/// ac-005 — both MMFF force fields must DEFINE all 7 energy styles.
///
/// `mmff_ele_ctor` has been in the kernel registry since it was written; what is
/// missing is a style. A registered-but-unreferenced kernel is a term that does
/// not exist.
#[test]
fn mmff_forcefields_define_all_seven_energy_styles() {
    let ff94 = MMFF94Typifier::new();
    let ff94s = MMFF94STypifier::new();
    let mut fails = Vec::new();
    for (label, ff) in [("MMFF94", ff94.ff()), ("MMFF94s", ff94s.ff())] {
        for (category, style_name, key) in MMFF_STYLES {
            let found = ff
                .styles()
                .iter()
                .any(|s| s.category() == category && s.name == style_name);
            if !found {
                fails.push(format!(
                    "{label}: no `{category}/{style_name}` style — the `{key}` energy term \
                     cannot be computed by any consumer of the documented public API"
                ));
            }
        }
    }
    assert!(
        fails.is_empty(),
        "missing MMFF styles:\n  {}",
        fails.join("\n  ")
    );
}

/// ac-005 — the electrostatic parameters are DATA, declared once per XML.
///
/// The section is the reader's dispatch source (`<ElectrostaticParams
/// dielectric="1.0" delta="0.05" scale14="0.75"/>`), symmetric with
/// `<VdWParams>` — not a `def_pairstyle` conjured inside the reader. All four
/// copies must carry it, or a `data/` build and a `molrs/data/` build disagree
/// about whether MMFF has electrostatics.
#[test]
fn both_mmff_xmls_declare_the_electrostatic_section() {
    let mut fails = Vec::new();
    for rel in [
        "molrs/data/mmff94.xml",
        "molrs/data/mmff94s.xml",
        "data/mmff94.xml",
        "data/mmff94s.xml",
    ] {
        let path = repo_root().join(rel);
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let n = text.matches("<ElectrostaticParams").count();
        if n != 1 {
            fails.push(format!(
                "{rel}: {n} `<ElectrostaticParams` elements, want exactly 1"
            ));
        }
    }
    assert!(
        fails.is_empty(),
        "MMFF XML is missing the electrostatic section:\n  {}",
        fails.join("\n  ")
    );
}

/// MMFF's 1-4 rules live in `special_bonds` (house rule), not in the kernel.
///
/// Coulomb 1-4 is scaled by **0.75**; vdW 1-4 is **not scaled at all** (RDKit
/// semantics). The vdW weight is 1.0 — a no-op — and it must stay one: the next
/// reader of `lj14scale` will be tempted to "fix" it to 0.5 by analogy with
/// Amber, which would silently corrupt every MMFF vdW energy with a 1-4 pair.
#[test]
fn mmff_special_bonds_scale_coulomb_14_but_not_vdw() {
    for (label, ff) in [
        ("MMFF94", MMFF94Typifier::new().ff().clone()),
        ("MMFF94s", MMFF94STypifier::new().ff().clone()),
    ] {
        let sp = ff.special_bonds();
        assert!(
            (sp.coul_14() - 0.75).abs() < 1e-12,
            "{label}: coulomb 1-4 weight is {}, want 0.75 (MMFF buffered-Coulomb 1-4 scaling)",
            sp.coul_14()
        );
        assert!(
            (sp.lj_14() - 1.0).abs() < 1e-12,
            "{label}: vdW 1-4 weight is {}, want 1.0 — MMFF does NOT scale 1-4 vdW",
            sp.lj_14()
        );
    }
}

/// ac-006 (reverse assertion) — zero-charge molecules get EXACTLY zero
/// electrostatic energy.
///
/// e_ethane and e_butane are the only two fixtures with sum|q| = 0, which is why
/// asserting only `e_ethane` could never have exposed the missing electrostatic
/// term. Now that the term exists, they prove it was added in the RIGHT PLACE:
/// a term that produces spurious energy on a neutral molecule is not a fix.
#[test]
fn zero_charge_molecules_get_exactly_zero_electrostatic_energy() {
    for name in ["e_ethane", "e_butane"] {
        let g = GenericMmff::load(name);
        let q = g.total_abs_charge();
        assert!(
            q < 1e-12,
            "{name} was chosen because its MMFF charges are all zero, but sum|q| = {q}"
        );

        let ele = g.style_energy("ele", &g.coords).unwrap_or_else(|| {
            panic!(
                "{name}: the force field defines no `pair/mmff_ele` style — there is no \
                 electrostatic term to be zero"
            )
        });
        assert_eq!(
            ele, 0.0,
            "{name}: sum|q| = 0, so the electrostatic energy must be EXACTLY 0.0, not {ele:e}"
        );

        let (pots, coords) = generic_pots(name);
        let total = pots.calc_energy_forces(&coords).0;
        let d = (total - ref_energy(name)).abs();
        assert!(
            d < ENERGY_TOL,
            "{name}: total {total:.6} vs RDKit {:.6} (d={d:.3e})",
            ref_energy(name)
        );
    }
}

/// e_butane is the only zero-charge fixture WITH 1-4 pairs — so it is the one
/// molecule that can prove `lj14scale = 1.0` is a genuine no-op on vdW.
///
/// If someone ever scales MMFF's 1-4 vdW, butane's vdW term moves and this fails
/// while its (zero) electrostatics stays put.
#[test]
fn butane_has_14_pairs_and_its_vdw_is_unscaled() {
    let g = GenericMmff::load("e_butane");
    let n14 = g.n_14_pairs();
    assert!(
        n14 > 0,
        "e_butane must have 1-4 pairs, else the unscaled-vdW claim is vacuous"
    );
    let vdw = g
        .style_energy("vdw", &g.coords)
        .expect("MMFF defines pair/mmff_vdw");
    let want = frozen_breakdown("e_butane")["vdw"];
    let d = (vdw - want).abs();
    println!("e_butane {n14} 1-4 pairs, vdw={vdw:.6} frozen={want:.6} d={d:.3e}");
    assert!(
        d < BREAKDOWN_TOL,
        "e_butane vdw={vdw:.6} != frozen {want:.6} (d={d:.3e}); MMFF does not scale 1-4 vdW"
    );
}

// ---------------------------------------------------------------------------
// e_acetonitrile — the linear-angle fixture (defect 4).
// ---------------------------------------------------------------------------

/// The three atoms of the C-C#N angle, checked against the SDF's own element
/// column so the indices cannot silently rot.
fn acetonitrile_linear_centre() -> (usize, usize, usize) {
    let elements = elements_of(&fixtures_dir().join("e_acetonitrile.sdf"));
    assert_eq!(
        elements,
        vec!["C", "C", "N", "H", "H", "H"],
        "e_acetonitrile.sdf atom order changed; the linear centre is no longer 0-1-2"
    );
    (0, 1, 2)
}

fn angle_deg(coords: &[f64], i: usize, j: usize, k: usize) -> f64 {
    let v = |a: usize, b: usize| {
        [
            coords[3 * a] - coords[3 * b],
            coords[3 * a + 1] - coords[3 * b + 1],
            coords[3 * a + 2] - coords[3 * b + 2],
        ]
    };
    let (u, w) = (v(i, j), v(k, j));
    let dot = u[0] * w[0] + u[1] * w[1] + u[2] * w[2];
    let nu = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
    let nw = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
    (dot / (nu * nw)).clamp(-1.0, 1.0).acos().to_degrees()
}

/// ac-009 — the fixture's own anti-degeneracy self-check.
///
/// At exactly 180° BOTH angle forms give zero and the defect hides. Measured, per
/// unit ka: at ETKDG's 179.14° the linear and cubic forms differ by ~1e-4 (below
/// the 1e-3 tolerance); at the 170° this fixture is built at, by 0.16. If this
/// fixture ever straightens out, it stops testing what it was added to test.
#[test]
fn acetonitrile_fixture_is_bent_away_from_180() {
    let (i, j, k) = acetonitrile_linear_centre();
    let coords = coords_of(&fixtures_dir().join("e_acetonitrile.sdf"));
    let theta = angle_deg(&coords, i, j, k);
    let deviation = (180.0 - theta).abs();
    println!("e_acetonitrile C-C#N = {theta:.3} deg (deviation {deviation:.3} deg)");
    assert!(
        deviation >= 2.0,
        "e_acetonitrile C-C#N is {theta:.3} deg, only {deviation:.3} deg from linear. Below ~2 deg \
         the linear form E = 143.9325*ka*(1+cos t) and the cubic form agree to within the 1e-3 \
         parity tolerance, so this fixture would pass with OR without the linear-angle branch — \
         a fake fixture. Rebuild it with rdMolTransforms.SetAngleDeg(conf, 0, 1, 2, 170.0)."
    );
}

/// ac-009 — the angle term of a linear centre uses `E = 143.9325*ka*(1+cos t)`,
/// not the cubic form.
///
/// Measured on this geometry: the linear form gives 1.205521 kcal/mol (RDKit's
/// own value, term-for-term), the cubic form gives 1.279203 — a 0.074 kcal/mol
/// gap, 74x the parity tolerance. Every alkyne, nitrile and allene is wrong
/// until this branch exists, and no other fixture covers one.
#[test]
fn acetonitrile_angle_energy_uses_the_linear_form() {
    let g = GenericMmff::load("e_acetonitrile");
    let angle = g
        .style_energy("angle", &g.coords)
        .expect("MMFF defines angle/mmff_angle");
    let want = frozen_breakdown("e_acetonitrile")["angle"];
    let d = (angle - want).abs();
    println!("e_acetonitrile angle: generic={angle:.6} frozen={want:.6} d={d:.3e}");
    assert!(
        d < BREAKDOWN_TOL,
        "e_acetonitrile angle energy is {angle:.6}, want {want:.6} (d={d:.3e}). The cubic bend \
         form gives 1.279203 on this geometry; the linear centre (MmffProp.linh != 0) needs \
         E = 143.9325*ka*(1 + cos theta)."
    );
}

/// ac-009 (reverse assertion) — a linear centre contributes EXACTLY zero
/// stretch-bend.
///
/// Note what this does NOT say: acetonitrile's stretch-bend energy is
/// **-0.010941**, not zero — its three methyl angles carry stbn like any other
/// sp3 centre. Only the LINEAR centre is skipped. RDKit's oracle says the same
/// (`SetMMFFStretchBendTerm`-only energy = -0.010941).
///
/// So the exact statement is an invariance: the C-C#N stretch-bend term is the
/// only stbn row that could involve the nitrogen, so moving N — which changes
/// both r(C#N) and theta(C-C#N), and no methyl bond or methyl angle — must leave
/// the stretch-bend energy bit-for-bit unchanged. RDKit: delta = 0.000000 while
/// bond moves by +15.91 and angle by -0.90.
#[test]
fn acetonitrile_linear_centre_contributes_no_stretch_bend() {
    let (_, _, n) = acetonitrile_linear_centre();
    let g = GenericMmff::load("e_acetonitrile");

    let mut moved = g.coords.clone();
    moved[3 * n] += 0.15;
    moved[3 * n + 1] += 0.07;
    moved[3 * n + 2] -= 0.11;

    let stbn0 = g.style_energy("stbn", &g.coords).expect("angle/mmff_stbn");
    let stbn1 = g.style_energy("stbn", &moved).expect("angle/mmff_stbn");
    let bond0 = g.style_energy("bond", &g.coords).expect("bond/mmff_bond");
    let bond1 = g.style_energy("bond", &moved).expect("bond/mmff_bond");

    // The perturbation must actually perturb, or the invariance is vacuous.
    assert!(
        (bond1 - bond0).abs() > 1.0,
        "displacing N changed the bond energy by only {:.3e}; the probe is not probing",
        bond1 - bond0
    );
    // ... and stbn must be live, or "unchanged" would be trivially true.
    assert!(
        stbn0.abs() > 1e-6,
        "acetonitrile's stretch-bend energy is {stbn0:e}; its methyl angles should give \
         -0.010941, so an all-zero stbn means the style is not being evaluated at all"
    );

    assert_eq!(
        stbn0, stbn1,
        "displacing the nitrogen changed the stretch-bend energy ({stbn0:e} -> {stbn1:e}). The \
         C-C#N centre is LINEAR (MmffProp.linh != 0) and MMFF skips stretch-bend there entirely, \
         so no stbn row may reference N. Every other stbn row (H-C-H, H-C-C) is untouched by this \
         displacement."
    );
}
