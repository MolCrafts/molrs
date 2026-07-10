//! OPLS-AA typifier parity harness vs molpy's `OPLSAATypifier` (spec
//! `opls-typifier-03-parity`, ac-001 … ac-004).
//!
//! Ground truth is a set of per-molecule JSON fixtures under `tests-data/opls/`,
//! produced by molpy's own `OPLSAATypifier` over a fixed real-molecule set
//! (`molpy/scripts/gen_opls_fixtures.py`). Each fixture carries:
//!
//! * the molecule definition (element + xyz per atom, bonds as `[i, j, order]`)
//!   so this test rebuilds the **exact same input topology** molpy typed;
//! * the per-atom `opls_NNN` type;
//! * every bond / angle / dihedral with its assigned force-field type *name* and
//!   numeric params, already expressed in **molrs canonical units** (the
//!   generator reconciles the molpy reader's deg/0.5-prefactor conventions, see
//!   that script's header).
//!
//! molrs types from its **embedded canonical** OPLS-AA force field
//! ([`molrs::data::OPLSAA_XML`] via [`OPLSAATypifier::oplsaa`]) — the durable,
//! committed-in-molrs copy — so the force-field source is independent of
//! `scripts/fetch-test-data.sh`. The generator fed molpy that same canonical
//! (c-corrected) XML, so this isolates *engine* semantics rather than
//! force-field-version drift.
//!
//! # Gating
//!
//! The whole suite skips cleanly (prints + returns) when `tests-data/opls/` is
//! absent, mirroring the GAFF/parmchk2 parity pattern. Run
//! `python molpy/scripts/gen_opls_fixtures.py` (with molpy installed) to
//! regenerate, or fetch via `scripts/fetch-test-data.sh`.
//!
//! # The C/c aromatic seam (gap closed)
//!
//! The molrs SMARTS engine is RDKit-faithful: it perceives aromatic atoms (from
//! order ~1.5 ring bonds) and matches them with lowercase `c`, reserving
//! uppercase `C` for aliphatic carbon. The OPLS XML's aromatic ring-carbon /
//! -hydrogen defs use lowercase `c`, so molrs now types the benzene / toluene
//! ring atoms exactly as molpy's ground truth (opls_145 ring C, opls_146 ring
//! H). `opls_parity_aromatic_per_atom_now_matches_molpy` asserts this: benzene
//! is full 12/12 per-atom parity; toluene agrees on every ring H and non-ipso
//! ring carbon, with the only residual being the two substituent-junction
//! carbons where molrs picks the more specific OPLS override (ipso
//! opls_141->opls_145, methyl opls_135->opls_148).
//!
//! The hard agreeable-set gate (ac-001 100% per-atom, ac-002 params-in-tolerance)
//! runs on the aliphatic / alcohol / ether set (the PEO-relevant chemistry); the
//! benzene/toluene fixtures keep molpy's `known_gap` flag so they stay out of
//! that *bonded-term-name* gate — molrs serializes an OPLS wildcard dihedral end
//! as the verbatim empty class (`-CA-CA-`) where molpy normalizes it to `*`
//! (`*-CA-CA-*`); that naming convention is a separate seam, not an aromatic
//! typing gap (the matched params agree).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use molrs::ff::typifier::opls::OPLSAATypifier;
use molrs::{Atom, AtomId, Atomistic};
use serde_json::Value;

use crate::helpers;

/// A bonded term as `(canonical atom indices, assigned type name, numeric params)`
/// collected off a molrs typed structure for comparison against the ground truth.
type MolrsTerm = (Vec<usize>, Option<String>, BTreeMap<String, f64>);

// --- spec tolerances (ac-002) ----------------------------------------------
const R0_ATOL: f64 = 0.02; // Å
const THETA_ATOL: f64 = 3.0 * std::f64::consts::PI / 180.0; // 3° in rad
const FK_RTOL: f64 = 0.10; // force-constant relative tolerance

/// Resolve the fixture directory, or `None` to skip cleanly when absent.
fn fixtures_dir() -> Option<PathBuf> {
    let dir = helpers::tests_data_dir().join("opls");
    dir.is_dir().then_some(dir)
}

/// Every `*.json` fixture (excluding `manifest.json`), sorted.
fn fixture_files(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", dir.display()))
        .map(|e| e.expect("dir entry").path())
        .filter(|p| {
            p.extension().is_some_and(|x| x == "json")
                && p.file_name().is_some_and(|n| n != "manifest.json")
        })
        .collect();
    files.sort();
    files
}

// ===========================================================================
// Fixture model
// ===========================================================================

struct Fixture {
    name: String,
    known_gap: bool,
    coverage: Vec<String>,
    elements: Vec<String>,
    coords: Vec<[f64; 3]>,
    bonds: Vec<(usize, usize, f64)>,
    atom_types: Vec<Option<String>>,
    bonds_gt: Vec<TermGt>,
    angles_gt: Vec<TermGt>,
    dihedrals_gt: Vec<TermGt>,
}

/// One bonded ground-truth term: canonical atom indices + type name + numeric
/// params (molrs-canonical units).
struct TermGt {
    atoms: Vec<usize>,
    ty: Option<String>,
    params: BTreeMap<String, f64>,
}

fn jf(v: &Value, k: &str) -> Option<f64> {
    v.get(k).and_then(Value::as_f64)
}

fn parse_term(v: &Value, param_keys: &[&str]) -> TermGt {
    let atoms = v["atoms"]
        .as_array()
        .expect("term.atoms")
        .iter()
        .map(|x| x.as_u64().expect("atom index") as usize)
        .collect();
    let ty = v
        .get("type")
        .and_then(Value::as_str)
        .map(str::to_string)
        .filter(|_| !v["type"].is_null());
    let mut params = BTreeMap::new();
    for k in param_keys {
        if let Some(val) = jf(v, k) {
            params.insert((*k).to_string(), val);
        }
    }
    TermGt { atoms, ty, params }
}

fn parse_fixture(path: &Path) -> Fixture {
    let text = std::fs::read_to_string(path).expect("read fixture");
    let v: Value = serde_json::from_str(&text).expect("parse fixture json");

    let mol = &v["molecule"];
    let elements: Vec<String> = mol["atoms"]
        .as_array()
        .expect("atoms")
        .iter()
        .map(|a| a["element"].as_str().expect("element").to_string())
        .collect();
    let coords: Vec<[f64; 3]> = mol["atoms"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| {
            [
                a["x"].as_f64().unwrap_or(0.0),
                a["y"].as_f64().unwrap_or(0.0),
                a["z"].as_f64().unwrap_or(0.0),
            ]
        })
        .collect();
    let bonds: Vec<(usize, usize, f64)> = mol["bonds"]
        .as_array()
        .expect("bonds")
        .iter()
        .map(|b| {
            let arr = b.as_array().unwrap();
            (
                arr[0].as_u64().unwrap() as usize,
                arr[1].as_u64().unwrap() as usize,
                arr.get(2).and_then(Value::as_f64).unwrap_or(1.0),
            )
        })
        .collect();

    let atom_types: Vec<Option<String>> = v["atom_types"]
        .as_array()
        .expect("atom_types")
        .iter()
        .map(|t| t.as_str().map(str::to_string))
        .collect();

    let terms = |key: &str, pk: &[&str]| -> Vec<TermGt> {
        v[key]
            .as_array()
            .map(|a| a.iter().map(|t| parse_term(t, pk)).collect())
            .unwrap_or_default()
    };

    Fixture {
        name: v["name"].as_str().unwrap_or("?").to_string(),
        known_gap: v["known_gap"].as_bool().unwrap_or(false),
        coverage: v["coverage"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default(),
        elements,
        coords,
        bonds,
        atom_types,
        bonds_gt: terms("bonds", &["r0", "k0"]),
        angles_gt: terms("angles", &["theta0", "k0"]),
        dihedrals_gt: terms("dihedrals", &["f1", "f2", "f3", "f4"]),
    }
}

/// Rebuild the molrs [`Atomistic`] from the fixture's molecule definition.
/// Atom iteration order == fixture index order (same as molpy fed).
fn build_mol(fx: &Fixture) -> Atomistic {
    let mut g = Atomistic::new();
    let mut ids: Vec<AtomId> = Vec::with_capacity(fx.elements.len());
    for (elem, c) in fx.elements.iter().zip(&fx.coords) {
        ids.push(g.add_atom(Atom::xyz(elem, c[0], c[1], c[2])));
    }
    for &(i, j, order) in &fx.bonds {
        let bid = g.add_bond(ids[i], ids[j]).expect("add_bond");
        let _ = g.set_bond_prop(bid, "order", order);
    }
    g
}

// ===========================================================================
// Per-term comparison helpers
// ===========================================================================

/// Canonical key for a bonded term: its atom indices sorted (orientation-free).
fn canon(atoms: &[usize]) -> Vec<usize> {
    let mut a = atoms.to_vec();
    a.sort_unstable();
    a
}

/// Build an index from atom-id → fixture index for the typed structure.
/// Relies on atom iteration order matching insertion order.
fn id_to_index(typed: &Atomistic) -> BTreeMap<AtomId, usize> {
    typed
        .atoms()
        .enumerate()
        .map(|(i, (id, _))| (id, i))
        .collect()
}

/// Whether two force constants agree within the relative tolerance (treating a
/// near-zero reference as absolute).
fn fk_ok(got: f64, want: f64) -> bool {
    if want.abs() < 1e-9 {
        got.abs() < 1e-6
    } else {
        (got - want).abs() / want.abs() < FK_RTOL
    }
}

/// Compare one molrs term's params against the ground-truth term.
/// Returns `Err(msg)` on the first mismatch.
fn compare_params(kind: &str, got: &BTreeMap<String, f64>, want: &TermGt) -> Result<(), String> {
    for (key, &wv) in &want.params {
        let gv = got
            .get(key)
            .copied()
            .ok_or_else(|| format!("{kind} term missing param {key}"))?;
        let ok = match key.as_str() {
            "r0" => (gv - wv).abs() < R0_ATOL,
            "theta0" => (gv - wv).abs() < THETA_ATOL,
            _ => fk_ok(gv, wv), // k0 / f1..f4
        };
        if !ok {
            return Err(format!(
                "{kind} param {key}: molrs {gv} vs molpy {wv} (atoms {:?})",
                want.atoms
            ));
        }
    }
    Ok(())
}

/// Read the molrs term's params (only the keys present in the ground truth) and
/// its `type` name, from a bonded relation's `props` map.
fn molrs_term(
    props: &std::collections::HashMap<String, molrs::system::molgraph::PropValue>,
    keys: &[&str],
) -> (Option<String>, BTreeMap<String, f64>) {
    let ty = props.get("type").and_then(|v| match v {
        molrs::system::molgraph::PropValue::Str(s) => Some(s.clone()),
        _ => None,
    });
    let mut params = BTreeMap::new();
    for k in keys {
        if let Some(val) = props.get(*k).and_then(|v| v.as_f64()) {
            params.insert((*k).to_string(), val);
        }
    }
    (ty, params)
}

/// Result of comparing one molecule's bonded terms.
#[derive(Default)]
struct TermReport {
    matched: usize,
    type_mismatch: usize,
    param_mismatch: usize,
    only_molrs: usize, // molrs parametrized a term molpy left bare (silent estimation guard)
    first_error: Option<String>,
}

/// Compare a molrs bonded relation set against the ground-truth term list,
/// matched by canonical atom-index tuple. `param_keys` selects the GT params.
fn compare_terms(kind: &str, molrs_terms: &[MolrsTerm], gt: &[TermGt]) -> TermReport {
    let mut report = TermReport::default();
    let gt_by_atoms: BTreeMap<Vec<usize>, &TermGt> =
        gt.iter().map(|t| (canon(&t.atoms), t)).collect();

    for (atoms, ty, params) in molrs_terms {
        let key = canon(atoms);
        match gt_by_atoms.get(&key) {
            Some(want) => {
                // molpy left this term unparameterized (no type/params): ac-004
                // requires molrs leave it bare too (no silent estimation).
                if want.ty.is_none() && want.params.is_empty() {
                    if ty.is_some() || !params.is_empty() {
                        report.only_molrs += 1;
                        report.first_error.get_or_insert(format!(
                            "{kind}: molrs parametrized a term molpy left bare (atoms {atoms:?})"
                        ));
                    } else {
                        report.matched += 1;
                    }
                    continue;
                }
                if ty != &want.ty {
                    report.type_mismatch += 1;
                    report.first_error.get_or_insert(format!(
                        "{kind} type: molrs {ty:?} vs molpy {:?} (atoms {atoms:?})",
                        want.ty
                    ));
                    continue;
                }
                if let Err(e) = compare_params(kind, params, want) {
                    report.param_mismatch += 1;
                    report.first_error.get_or_insert(e);
                    continue;
                }
                report.matched += 1;
            }
            None => {
                // A molrs term with no GT counterpart only matters if molrs
                // parametrized it (would be silent estimation).
                if ty.is_some() || !params.is_empty() {
                    report.only_molrs += 1;
                    report.first_error.get_or_insert(format!(
                        "{kind}: molrs parametrized an unmatched term (atoms {atoms:?})"
                    ));
                }
            }
        }
    }
    report
}

/// Collect a typed structure's bonded relations as `(atom-indices, type, params)`.
fn collect_molrs_terms(
    typed: &Atomistic,
    idx: &BTreeMap<AtomId, usize>,
) -> (Vec<MolrsTerm>, Vec<MolrsTerm>, Vec<MolrsTerm>) {
    let bonds = typed
        .bonds()
        .map(|(_, b)| {
            let atoms: Vec<usize> = b.nodes.iter().map(|n| idx[n]).collect();
            let (ty, p) = molrs_term(&b.props, &["r0", "k0"]);
            (atoms, ty, p)
        })
        .collect();
    let angles = typed
        .angles()
        .map(|(_, a)| {
            let atoms: Vec<usize> = a.nodes.iter().map(|n| idx[n]).collect();
            let (ty, p) = molrs_term(&a.props, &["theta0", "k0"]);
            (atoms, ty, p)
        })
        .collect();
    let dihedrals = typed
        .dihedrals()
        .map(|(_, d)| {
            let atoms: Vec<usize> = d.nodes.iter().map(|n| idx[n]).collect();
            let (ty, p) = molrs_term(&d.props, &["f1", "f2", "f3", "f4"]);
            (atoms, ty, p)
        })
        .collect();
    (bonds, angles, dihedrals)
}

// ===========================================================================
// Per-atom parity
// ===========================================================================

struct AtomParity {
    agree: usize,
    total: usize,
    /// (index, element, molpy type, molrs type) for each divergent atom.
    diverged: Vec<(usize, String, Option<String>, Option<String>)>,
}

fn atom_parity(fx: &Fixture, typed: &Atomistic) -> AtomParity {
    let molrs_types: Vec<Option<String>> = typed
        .atoms()
        .map(|(_, a)| a.get_str("type").map(str::to_string))
        .collect();
    assert_eq!(
        molrs_types.len(),
        fx.atom_types.len(),
        "{}: atom count mismatch (topology rebuild bug)",
        fx.name
    );

    let mut agree = 0;
    let mut diverged = Vec::new();
    for (i, (want, got)) in fx.atom_types.iter().zip(&molrs_types).enumerate() {
        if want == got {
            agree += 1;
        } else {
            diverged.push((i, fx.elements[i].clone(), want.clone(), got.clone()));
        }
    }
    AtomParity {
        agree,
        total: fx.atom_types.len(),
        diverged,
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[test]
fn opls_parity_agreeable_set_exact_per_atom_and_params() {
    // ac-001 + ac-002 + ac-004: on the agreeable set (non-gap fixtures —
    // aliphatic / alcohol / ether), molrs must assign the EXACT same per-atom
    // opls type as molpy (100%), and every bonded term molpy parametrized must
    // match within tolerance. Terms molpy left bare must stay bare in molrs
    // (no silent estimation). Skips cleanly when fixtures are absent.
    let Some(dir) = fixtures_dir() else {
        eprintln!(
            "skipping OPLS parity: tests-data/opls/ absent — run \
             `python molpy/scripts/gen_opls_fixtures.py` (molpy required)"
        );
        return;
    };

    // Source the FF from the embedded canonical OPLS-AA XML (durable, committed
    // in molrs — independent of `fetch-test-data.sh`), not the gitignored
    // tests-data copy. The fixtures (ground truth) stay gated separately.
    let typifier = OPLSAATypifier::oplsaa()
        .expect("build OPLSAATypifier from embedded OPLS-AA XML")
        // ac-004: lenient (no estimator) — matches the molpy non-strict run; a
        // term with no candidate is left bare, never estimated.
        .with_strict(false);

    let files = fixture_files(&dir);
    assert!(!files.is_empty(), "tests-data/opls has fixtures");

    let mut checked_agreeable = 0;
    let mut total_atoms = 0usize;
    let mut total_agree = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for path in &files {
        let fx = parse_fixture(path);
        if fx.known_gap {
            continue; // characterized separately; not a hard gate
        }
        checked_agreeable += 1;

        let mol = build_mol(&fx);
        let typed = typifier
            .typify(&mol)
            .unwrap_or_else(|e| panic!("{}: typify failed: {e}", fx.name));
        let idx = id_to_index(&typed);

        // --- per-atom (ac-001) ---
        let ap = atom_parity(&fx, &typed);
        total_atoms += ap.total;
        total_agree += ap.agree;
        if !ap.diverged.is_empty() {
            failures.push(format!(
                "{}: {}/{} atoms agree; diverged: {:?}",
                fx.name, ap.agree, ap.total, ap.diverged
            ));
        }

        // --- per-term (ac-002 + ac-004) ---
        let (mb, ma, md) = collect_molrs_terms(&typed, &idx);
        for (kind, got, gt) in [
            ("bond", &mb, &fx.bonds_gt),
            ("angle", &ma, &fx.angles_gt),
            ("dihedral", &md, &fx.dihedrals_gt),
        ] {
            let rep = compare_terms(kind, got, gt);
            if let Some(err) = rep.first_error {
                failures.push(format!("{}: {err}", fx.name));
            }
            // Every GT term that molpy parametrized must have a molrs match.
            let gt_param = gt.iter().filter(|t| t.ty.is_some()).count();
            if rep.matched + rep.type_mismatch + rep.param_mismatch < gt_param {
                failures.push(format!(
                    "{}: {kind} — molrs missing terms molpy parametrized \
                     (matched {} of {gt_param})",
                    fx.name, rep.matched
                ));
            }
        }
    }

    assert!(
        checked_agreeable > 0,
        "the agreeable (non-gap) fixture set is non-empty"
    );
    eprintln!(
        "OPLS parity (agreeable set): {}/{} atoms exact across {} molecules",
        total_agree, total_atoms, checked_agreeable
    );
    assert!(
        failures.is_empty(),
        "OPLS parity failures on the agreeable set:\n  {}",
        failures.join("\n  ")
    );
    // ac-001: per-atom parity is 100% on the agreeable set.
    assert_eq!(
        total_agree, total_atoms,
        "per-atom OPLS type parity must be 100% on the agreeable set"
    );
}

#[test]
fn opls_parity_fixture_coverage() {
    // ac-003: the fixture set must exercise a wildcard-end dihedral
    // (X-CT-CT-X style) and an overrides/layer-resolved atom type. Skips
    // cleanly when fixtures are absent.
    let Some(dir) = fixtures_dir() else {
        eprintln!("skipping OPLS coverage check: tests-data/opls/ absent");
        return;
    };

    let mut has_wildcard_dihedral = false;
    let mut has_layered = false;
    let mut has_overrides = false;
    let mut saw_x_ct_ct_x = false;

    for path in &fixture_files(&dir) {
        let fx = parse_fixture(path);
        for c in &fx.coverage {
            match c.as_str() {
                "wildcard_dihedral" => has_wildcard_dihedral = true,
                "layered_type" => has_layered = true,
                "overrides" => has_overrides = true,
                _ => {}
            }
        }
        // A directly-observable wildcard-end dihedral: an X-CT-CT-X / HC-CT-CT-HC
        // / OS-CT-CT-OS term, i.e. a dihedral whose ground-truth type name has a
        // wildcard end or whose central pair is CT-CT.
        for d in &fx.dihedrals_gt {
            if let Some(ty) = &d.ty
                && (ty.starts_with("*-") || ty.ends_with("-*") || ty.contains("-CT-CT-"))
            {
                saw_x_ct_ct_x = true;
            }
        }
    }

    assert!(
        has_wildcard_dihedral && saw_x_ct_ct_x,
        "fixture set must include a wildcard-end / X-CT-CT-X dihedral \
         (coverage flag {has_wildcard_dihedral}, observed term {saw_x_ct_ct_x})"
    );
    assert!(
        has_layered && has_overrides,
        "fixture set must include an overrides/layer-resolved atom type \
         (layered {has_layered}, overrides {has_overrides})"
    );
}

#[test]
fn opls_parity_aromatic_per_atom_now_matches_molpy() {
    // Aromatic per-atom parity (the closed C/c gap). The OPLS XML's aromatic
    // ring-carbon / -hydrogen defs use lowercase `c` (RDKit-faithful), so molrs
    // now perceives the benzene/toluene ring atoms as aromatic and types them
    // exactly as molpy's ground truth (opls_145 ring C, opls_146 ring H). This
    // was formerly a `known_gap` characterization that only *reported* the
    // divergence; it is now a hard parity assertion.
    //
    // The fixtures are still flagged `known_gap` by molpy's generator (so they
    // stay out of the strict agreeable-set term gate, which would also assert
    // bonded-term *names* — see the separate wildcard-naming note below); this
    // test re-reads those same fixtures and asserts the aromatic typing agrees.
    //
    // benzene: full 12/12 per-atom parity (ring C → opls_145, ring H → opls_146).
    //
    // toluene: agrees on every ring H and every non-ipso ring carbon; the only
    // residual divergences are the two substituent-junction carbons, where molrs
    // assigns the MORE specific OPLS type than molpy's ground truth:
    //   * the ipso ring carbon — molrs opls_145 (aromatic CA, which `overrides`
    //     the generic alkene opls_141) vs molpy opls_141; molpy's uppercase-C
    //     opls_141 def matched the ipso by atomic number, while molrs's aromatic
    //     `c` opls_145 def now correctly claims it;
    //   * the methyl carbon — molrs opls_148 (toluene CH3, which `overrides`
    //     opls_135) vs molpy opls_135; molrs matches opls_148 because its
    //     `[C;X4]([c;%opls_145])(H)(H)H` def now sees the aromatic-c neighbour.
    // Both are cases where molrs picks the chemistry-specific override; they are
    // asserted precisely so the residual stays characterized, not silent.
    let Some(dir) = fixtures_dir() else {
        eprintln!("skipping OPLS aromatic parity: tests-data/opls/ absent");
        return;
    };

    // Embedded canonical OPLS-AA XML — same durable source as the agreeable-set
    // test (independent of fetch-test-data).
    let typifier = OPLSAATypifier::oplsaa()
        .expect("build OPLSAATypifier from embedded OPLS-AA XML")
        .with_strict(false);

    let mut gap_molecules = 0;
    for path in &fixture_files(&dir) {
        let fx = parse_fixture(path);
        if !fx.known_gap {
            continue;
        }
        gap_molecules += 1;

        let mol = build_mol(&fx);
        let typed = typifier
            .typify(&mol)
            .unwrap_or_else(|e| panic!("{}: typify failed: {e}", fx.name));
        let ap = atom_parity(&fx, &typed);

        eprintln!(
            "OPLS aromatic [{}]: {}/{} atoms agree with molpy ({} residual)",
            fx.name,
            ap.agree,
            ap.total,
            ap.diverged.len()
        );
        for (i, elem, molpy_ty, molrs_ty) in &ap.diverged {
            eprintln!("    atom[{i}] {elem}: molpy {molpy_ty:?}  vs  molrs {molrs_ty:?}");
        }

        // Whatever the molecule, every ring hydrogen must agree (opls_146) and
        // must be typed (never bare) — the aromatic-H def `[H][c;%opls_145]`.
        for (i, elem, molpy_ty, molrs_ty) in &ap.diverged {
            assert_ne!(
                elem, "H",
                "{}: aromatic hydrogen at atom[{i}] diverged \
                 (molpy {molpy_ty:?} vs molrs {molrs_ty:?}) — the lowercase-c \
                 aromatic-H def should match it",
                fx.name
            );
        }

        match fx.name.as_str() {
            "benzene" => assert!(
                ap.diverged.is_empty(),
                "benzene: aromatic gap must be fully closed (got {}/{}); \
                 diverged: {:?}",
                ap.agree,
                ap.total,
                ap.diverged
            ),
            "toluene" => {
                // The two substituent-junction carbons are the only allowed
                // residual, and only because molrs assigns the more specific
                // override. Anything else is a regression.
                let allowed = |molpy: &Option<String>, molrs: &Option<String>| {
                    let pair = (molpy.as_deref(), molrs.as_deref());
                    pair == (Some("opls_141"), Some("opls_145")) // ipso C
                        || pair == (Some("opls_135"), Some("opls_148")) // methyl C
                };
                for (i, elem, molpy_ty, molrs_ty) in &ap.diverged {
                    assert!(
                        elem == "C" && allowed(molpy_ty, molrs_ty),
                        "toluene: unexpected residual at atom[{i}] {elem} \
                         (molpy {molpy_ty:?} vs molrs {molrs_ty:?}) — only the \
                         ipso (opls_141->opls_145) and methyl (opls_135->opls_148) \
                         override divergences are characterized"
                    );
                }
                // And those two divergences must actually be present (they are
                // the documented molrs-more-specific result, not silent agreement
                // nor a fresh gap).
                assert_eq!(
                    ap.diverged.len(),
                    2,
                    "toluene: expected exactly the 2 substituent-junction \
                     override divergences, got {}",
                    ap.diverged.len()
                );
            }
            other => panic!("unexpected known_gap fixture {other:?}"),
        }
    }
    assert!(
        gap_molecules > 0,
        "at least one aromatic fixture (benzene/toluene) is present"
    );
}
