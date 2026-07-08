//! Aromaticity perception validation against RDKit's default model.
//!
//! Fixtures are produced by `gen_aromaticity_fixtures.py` (RDKit 2026.03.2):
//!   * one **Kekulé** V2000 SDF per molecule (RDKit atom order preserved,
//!     integer bond orders) so molrs perceives aromaticity from scratch — we do
//!     NOT load RDKit's aromatic flags;
//!   * a `<name>.json` sidecar with per-atom `GetIsAromatic()`, per-bond
//!     aromatic flags, and formal charges in SDF/atom order;
//!   * `smarts_ref.json` with RDKit `GetSubstructMatches(uniquify=False)`
//!     reference tuples for the ac-004 aromatic primitives.
//!
//! Acceptance criteria covered:
//!   * ac-001 — per-atom `is_aromatic` matches RDKit for every atom.
//!   * ac-002 — per-bond aromatic flag matches RDKit.
//!   * ac-003 — cyclohexane and cyclopentadiene yield zero aromatic atoms.
//!   * ac-004 — SMARTS `a`/`c` primitives match RDKit end-to-end after
//!     *native* perception.
//!   * ac-005 — perception is idempotent.

use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::path::PathBuf;

use molrs::{
    Atom, AtomId, Atomistic, BondId, MatchOptions, PropValue, SmartsPattern, perceive_aromaticity,
};
use serde_json::Value;

const MOLECULES: &[&str] = &[
    "benzene",
    "pyridine",
    "pyrrole",
    "imidazole",
    "furan",
    "naphthalene",
    "indole",
    "caffeine",
    "biphenyl",
    "cyclohexane",
    "cyclopentadiene",
    "phenol",
];

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("core")
        .join("fixtures")
        .join("aromaticity")
}

/// Minimal V2000 molfile loader: element + connectivity + bond order, atom
/// order preserved. Coordinates parsed but unused.
fn load_sdf(name: &str) -> (Atomistic, Vec<AtomId>) {
    let path = fixtures_dir().join(format!("{name}.sdf"));
    let text = fs::read_to_string(&path).expect("read sdf");
    let lines: Vec<&str> = text.lines().collect();
    // Line index 3 is the counts line: "aaabbb...".
    let counts = lines[3];
    let n_atoms: usize = counts[0..3].trim().parse().expect("atom count");
    let n_bonds: usize = counts[3..6].trim().parse().expect("bond count");

    let mut g = Atomistic::new();
    let mut ids = Vec::with_capacity(n_atoms);
    for i in 0..n_atoms {
        let l = lines[4 + i];
        let x: f64 = l[0..10].trim().parse().unwrap_or(0.0);
        let y: f64 = l[10..20].trim().parse().unwrap_or(0.0);
        let z: f64 = l[20..30].trim().parse().unwrap_or(0.0);
        let sym = l[31..].split_whitespace().next().unwrap_or("C");
        ids.push(g.add_atom(Atom::xyz(sym, x, y, z)));
    }
    for b in 0..n_bonds {
        let l = lines[4 + n_atoms + b];
        let ai: usize = l[0..3].trim().parse::<usize>().unwrap() - 1;
        let aj: usize = l[3..6].trim().parse::<usize>().unwrap() - 1;
        let order: f64 = l[6..9].trim().parse().unwrap_or(1.0);
        let bid = g.add_bond(ids[ai], ids[aj]).expect("add bond");
        g.set_bond_prop(bid, "order", PropValue::F64(order))
            .unwrap();
    }
    (g, ids)
}

fn load_meta(name: &str) -> Value {
    let path = fixtures_dir().join(format!("{name}.json"));
    serde_json::from_str(&fs::read_to_string(path).expect("read json")).unwrap()
}

/// Read the perceived `is_aromatic` flag of an atom (0/1).
fn atom_is_aromatic(g: &Atomistic, id: AtomId) -> i64 {
    match g.get_atom(id).unwrap().get("is_aromatic") {
        Some(PropValue::Int(v)) => *v as i64,
        Some(PropValue::F64(v)) => (*v != 0.0) as i64,
        _ => 0,
    }
}

/// Whether a bond is perceived aromatic (order ~= 1.5).
fn bond_is_aromatic(g: &Atomistic, bid: BondId) -> bool {
    match g.get_bond(bid).unwrap().props.get("order") {
        Some(PropValue::F64(v)) => (v - 1.5).abs() < 1e-6,
        _ => false,
    }
}

// ── ac-001 + ac-002: per-atom and per-bond flags match RDKit ────────────────

#[test]
fn test_per_atom_and_bond_match_rdkit() {
    let mut atom_failures: Vec<String> = Vec::new();
    let mut bond_failures: Vec<String> = Vec::new();

    for &name in MOLECULES {
        let (mut g, ids) = load_sdf(name);
        let meta = load_meta(name);
        perceive_aromaticity(&mut g);

        // ac-001: per-atom
        let want_atom = meta["atom_aromatic"].as_array().unwrap();
        for (k, &id) in ids.iter().enumerate() {
            let got = atom_is_aromatic(&g, id);
            let want = want_atom[k].as_i64().unwrap();
            if got != want {
                atom_failures.push(format!("  {name} atom {k}: got {got}, want {want}"));
            }
        }

        // ac-002: per-bond. Build (i,j)->aromatic from molrs result, in SDF idx.
        let row_of: HashMap<AtomId, usize> =
            ids.iter().enumerate().map(|(r, &id)| (id, r)).collect();
        let mut got_bonds: BTreeSet<(usize, usize)> = BTreeSet::new();
        for (bid, bond) in g.bonds() {
            if bond_is_aromatic(&g, bid) {
                let i = row_of[&bond.nodes[0]];
                let j = row_of[&bond.nodes[1]];
                got_bonds.insert((i.min(j), i.max(j)));
            }
        }
        let mut want_bonds: BTreeSet<(usize, usize)> = BTreeSet::new();
        for entry in meta["bond_aromatic"].as_array().unwrap() {
            if entry["aromatic"].as_i64().unwrap() == 1 {
                let i = entry["i"].as_u64().unwrap() as usize;
                let j = entry["j"].as_u64().unwrap() as usize;
                want_bonds.insert((i.min(j), i.max(j)));
            }
        }
        if got_bonds != want_bonds {
            bond_failures.push(format!("  {name}: got {got_bonds:?}, want {want_bonds:?}"));
        }
    }

    assert!(
        atom_failures.is_empty(),
        "ac-001 per-atom mismatches:\n{}",
        atom_failures.join("\n")
    );
    assert!(
        bond_failures.is_empty(),
        "ac-002 per-bond mismatches:\n{}",
        bond_failures.join("\n")
    );
}

// ── ac-003: non-aromatic rings are not mislabeled ───────────────────────────

#[test]
fn test_nonaromatic_rings_zero() {
    for name in ["cyclohexane", "cyclopentadiene"] {
        let (mut g, _) = load_sdf(name);
        let n = perceive_aromaticity(&mut g);
        assert_eq!(n, 0, "{name} must have zero aromatic atoms");
    }
}

// ── ac-004: SMARTS aromatic primitives match RDKit end-to-end ───────────────

#[test]
fn test_smarts_primitives_match_rdkit() {
    let ref_path = fixtures_dir().join("smarts_ref.json");
    let smarts_ref: Value =
        serde_json::from_str(&fs::read_to_string(ref_path).expect("read smarts_ref")).unwrap();
    let patterns: Vec<String> = smarts_ref["patterns"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();

    let mut failures: Vec<String> = Vec::new();
    for &name in MOLECULES {
        let (mut g, ids) = load_sdf(name);
        // NATIVE perception — not transplanted flags.
        perceive_aromaticity(&mut g);
        let row_of: HashMap<AtomId, usize> =
            ids.iter().enumerate().map(|(r, &id)| (id, r)).collect();

        for p in &patterns {
            let pat = SmartsPattern::parse(p).unwrap_or_else(|e| panic!("parse {p:?}: {e}"));
            let got: BTreeSet<Vec<usize>> = pat
                .find(&g, MatchOptions::default())
                .iter()
                .map(|m| m.atoms().iter().map(|id| row_of[id]).collect())
                .collect();
            let want: BTreeSet<Vec<usize>> = smarts_ref["matches"][name][p]
                .as_array()
                .unwrap()
                .iter()
                .map(|t| {
                    t.as_array()
                        .unwrap()
                        .iter()
                        .map(|x| x.as_u64().unwrap() as usize)
                        .collect()
                })
                .collect();
            if got != want {
                failures.push(format!(
                    "  {name} :: {p}\n     got  {got:?}\n     want {want:?}"
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "ac-004 SMARTS mismatches:\n{}",
        failures.join("\n")
    );
}

// ── ac-005: idempotent ──────────────────────────────────────────────────────

#[test]
fn test_idempotent() {
    for &name in MOLECULES {
        let (mut g, ids) = load_sdf(name);
        let n1 = perceive_aromaticity(&mut g);
        let flags1: Vec<i64> = ids.iter().map(|&id| atom_is_aromatic(&g, id)).collect();
        let bonds1: Vec<bool> = g
            .bonds()
            .map(|(bid, _)| bond_is_aromatic(&g, bid))
            .collect();

        let n2 = perceive_aromaticity(&mut g);
        let flags2: Vec<i64> = ids.iter().map(|&id| atom_is_aromatic(&g, id)).collect();
        let bonds2: Vec<bool> = g
            .bonds()
            .map(|(bid, _)| bond_is_aromatic(&g, bid))
            .collect();

        assert_eq!(n1, n2, "{name} count differs between runs");
        assert_eq!(flags1, flags2, "{name} atom flags differ between runs");
        assert_eq!(bonds1, bonds2, "{name} bond flags differ between runs");
    }
}
