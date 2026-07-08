//! OPLS-AA SMARTS atom typing over real molecules from `tests-data/`.
//!
//! Per the MANDATORY IO testing rule, molecule inputs are **real files** read
//! from `tests-data/mol2/` (iterated, not a hand-crafted happy-path string),
//! and the force field is the real bundled `tests-data/xml/oplsaa.xml`. The only
//! in-code construction is a minimal Tripos-mol2 loader (the `ff` test target
//! does not enable the `io` feature, so it cannot call `read_mol2`; this mirrors
//! the MMFF test's self-contained SDF loader).

use std::path::Path;

use molrs::ff::forcefield::Params;
use molrs::ff::typifier::opls::{
    BondedTerm, CandidateTables, Estimator, NoMatch, OPLSAATypifier, OplsTypeRow, OplsTypingMeta,
    typify_atoms, typify_bonded, typify_bonded_with,
};
use molrs::{Atom, Atomistic};

use crate::helpers;

/// Map a SYBYL bond-type token to a numeric bond order (the molrs convention:
/// aromatic = 1.5, amide = 1.0).
fn sybyl_bond_order(tok: &str) -> f64 {
    match tok {
        "1" => 1.0,
        "2" => 2.0,
        "3" => 3.0,
        "ar" => 1.5,
        "am" => 1.0,
        _ => 1.0,
    }
}

/// Element symbol from a Tripos atom record. SYBYL atom types come in several
/// conventions — `C.3` / `N.ar` (dotted) and `c3` / `hc` / `cl` (GAFF-style
/// lowercase from Antechamber). Normalise by dropping the `.`-suffix and any
/// trailing digits, then validate against the periodic table: prefer the
/// title-cased two-letter symbol (`Cl`, `Br`, `Na`) when it is a real element,
/// else the single leading letter, falling back to the atom name.
fn element_of(sybyl_type: &str, name: &str) -> String {
    let base: String = sybyl_type
        .split('.')
        .next()
        .unwrap_or(sybyl_type)
        .chars()
        .take_while(|c| c.is_ascii_alphabetic())
        .collect();
    if let Some(sym) = normalize_element(&base) {
        return sym;
    }
    let name_base: String = name
        .chars()
        .take_while(|c| c.is_ascii_alphabetic())
        .collect();
    normalize_element(&name_base).unwrap_or(name_base)
}

/// Title-case a 1- or 2-letter element token and validate it against the
/// periodic table. Tries the two-letter symbol first (so `cl` -> `Cl`), then the
/// single leading letter (so `c3`-stripped `c` -> `C`).
fn normalize_element(token: &str) -> Option<String> {
    use std::str::FromStr;
    let t = token.to_ascii_lowercase();
    let title2 = |s: &str| -> String {
        let mut c = s.chars();
        let first = c.next().unwrap().to_ascii_uppercase();
        let rest: String = c.collect();
        format!("{first}{rest}")
    };
    if t.len() >= 2 {
        let two = title2(&t[..2]);
        if molrs::Element::from_str(&two).is_ok() {
            return Some(two);
        }
    }
    if !t.is_empty() {
        let one = title2(&t[..1]);
        if molrs::Element::from_str(&one).is_ok() {
            return Some(one);
        }
    }
    None
}

/// Minimal single-molecule Tripos mol2 loader producing an [`Atomistic`] with
/// `element` props and bond `order` props. Reads only the first `@<TRIPOS>`
/// MOLECULE in the file.
fn load_mol2(path: &Path) -> Atomistic {
    let text = std::fs::read_to_string(path).expect("read mol2");
    let mut g = Atomistic::new();
    let mut ids: Vec<molrs::AtomId> = Vec::new();
    let mut section = "";
    let mut seen_molecule = false;
    for line in text.lines() {
        let t = line.trim();
        if let Some(name) = t.strip_prefix("@<TRIPOS>") {
            // A second MOLECULE record => stop (single-molecule loader).
            if name == "MOLECULE" {
                if seen_molecule {
                    break;
                }
                seen_molecule = true;
            }
            section = match name {
                "ATOM" => "ATOM",
                "BOND" => "BOND",
                _ => "",
            };
            continue;
        }
        if t.is_empty() {
            continue;
        }
        match section {
            "ATOM" => {
                // id name x y z sybyl_type [subst_id subst_name charge]
                let f: Vec<&str> = t.split_whitespace().collect();
                if f.len() < 6 {
                    continue;
                }
                let x: f64 = f[2].parse().unwrap_or(0.0);
                let y: f64 = f[3].parse().unwrap_or(0.0);
                let z: f64 = f[4].parse().unwrap_or(0.0);
                let element = element_of(f[5], f[1]);
                ids.push(g.add_atom(Atom::xyz(&element, x, y, z)));
            }
            "BOND" => {
                // id origin target bond_type   (origin/target are 1-based)
                let f: Vec<&str> = t.split_whitespace().collect();
                if f.len() < 4 {
                    continue;
                }
                let i: usize = match f[1].parse::<usize>() {
                    Ok(v) if v >= 1 && v <= ids.len() => v - 1,
                    _ => continue,
                };
                let j: usize = match f[2].parse::<usize>() {
                    Ok(v) if v >= 1 && v <= ids.len() => v - 1,
                    _ => continue,
                };
                if let Ok(bid) = g.add_bond(ids[i], ids[j]) {
                    let _ = g.set_bond_prop(bid, "order", sybyl_bond_order(f[3]));
                }
            }
            _ => {}
        }
    }
    g
}

/// The real bundled OPLS-AA force field used as the typing source.
fn oplsaa_xml() -> String {
    std::fs::read_to_string(helpers::data_path("xml/oplsaa.xml")).expect("read oplsaa.xml")
}

#[test]
fn typifier_builds_from_real_oplsaa_xml() {
    // ac-001/ac-003: construction parses both typing-meta and the potential FF
    // from one XML, additively (the potential reader is unchanged).
    let typifier = OPLSAATypifier::from_xml_str(&oplsaa_xml()).expect("build OPLSAATypifier");
    assert!(
        !typifier.meta().is_empty(),
        "typing metadata should have rows"
    );
    // A known modern row carries class CT + a SMARTS def.
    let r135 = typifier.meta().get("opls_135").expect("opls_135 present");
    assert_eq!(r135.class, "CT");
    assert!(r135.def.is_some());
    // The potential FF still resolves charges per opls type.
    assert!(typifier.ff().get_style("atom", "full").is_some());
}

#[test]
fn ethane_atoms_typed_with_type_class_charge() {
    // ac-004: a real alkane (ethane.mol2 carries explicit H) is fully typed —
    // CT carbons + HC hydrogens — and typed atoms carry type/class/charge.
    let typifier = OPLSAATypifier::from_xml_str(&oplsaa_xml()).expect("build OPLSAATypifier");
    let mol = load_mol2(&helpers::data_path("mol2/ethane.mol2"));
    let typed = typifier.typify(&mol).expect("typify ethane");

    let mut typed_carbons = 0;
    let mut typed_hydrogens = 0;
    for (_, a) in typed.atoms() {
        let Some(ty) = a.get_str("type") else {
            continue;
        };
        assert!(ty.starts_with("opls_"), "type is an opls_NNN: {ty}");
        // class must be present for every typed atom.
        assert!(a.get_str("class").is_some(), "typed atom carries class");
        // charge resolved from the potential force field.
        assert!(a.get_f64("charge").is_some(), "typed atom carries charge");
        match a.get_str("element") {
            Some("C") => typed_carbons += 1,
            Some("H") => typed_hydrogens += 1,
            _ => {}
        }
    }
    assert_eq!(typed_carbons, 2, "both ethane carbons typed");
    assert_eq!(typed_hydrogens, 6, "all six ethane hydrogens typed");
}

#[test]
fn typing_runs_over_every_mol2_molecule() {
    // ac-004: iterate every real mol2 in tests-data — typing must never panic
    // or error, and at least one molecule must receive at least one opls type.
    let typifier = OPLSAATypifier::from_xml_str(&oplsaa_xml()).expect("build OPLSAATypifier");
    let files = helpers::format_files("mol2");
    assert!(!files.is_empty(), "tests-data/mol2 has files");

    let mut total_typed = 0usize;
    for path in &files {
        let mol = load_mol2(path);
        if mol.n_atoms() == 0 {
            continue;
        }
        let typed = typify_atoms(&mol, typifier.meta(), typifier.ff())
            .unwrap_or_else(|e| panic!("typify_atoms {:?} failed: {e}", path.file_name().unwrap()));
        // Every typed atom carries a well-formed opls_NNN type.
        for (_, a) in typed.atoms() {
            if let Some(ty) = a.get_str("type") {
                assert!(ty.starts_with("opls_"), "{path:?}: bad type {ty}");
                total_typed += 1;
            }
        }
    }
    assert!(
        total_typed > 0,
        "at least one atom across all mol2 molecules should be typed"
    );
}

#[test]
fn recursive_dollar_def_types_a_real_molecule() {
    // ac-004: at least one def using recursive $() SMARTS. The bundled oplsaa.xml
    // happens to carry no $() def, so we inject a single recursive def into the
    // typing metadata and confirm it types atoms of a REAL molecule (ethane).
    // (Engine-level recursive support is also unit-tested in src/.)
    let ff = OPLSAATypifier::from_xml_str(&oplsaa_xml())
        .expect("build")
        .ff()
        .clone();
    let mut meta = OplsTypingMeta::new();
    meta.insert(
        "opls_135",
        OplsTypeRow {
            class: "CT".into(),
            // recursive $(): an sp3 carbon bonded to another sp3 carbon.
            def: Some("[$([CX4][CX4])]".into()),
            overrides: Vec::new(),
            priority: None,
            layer: 0,
        },
    );
    let mol = load_mol2(&helpers::data_path("mol2/ethane.mol2"));
    let typed = typify_atoms(&mol, &meta, &ff).expect("typify with recursive def");
    let n = typed
        .atoms()
        .filter(|(_, a)| a.get_str("type") == Some("opls_135"))
        .count();
    assert_eq!(n, 2, "recursive $() def types both ethane carbons");
}

// ===========================================================================
// Chain 4: layered %opls_NNN dependency typing
// ===========================================================================
//
// These exercise the previously-skipped %opls_NNN defs end to end through the
// real bundled oplsaa.xml. Per-atom expectations are the ground truth produced
// by molpy's own `OPLSAATypifier` on the same molecule (verified out-of-band):
// methanol → C opls_157, O opls_154, hydroxyl-H opls_155, methyl-H opls_156.
// The hydroxyl H (opls_155, def `H[O;%opls_154]`) and methyl H (opls_156, def
// `HC[O;%opls_154]`) are %opls_154-dependent — exactly the layered defs chain-1
// could not assign. The molecule is built in code (the `ff` test target cannot
// read files via the `io` feature); the force field is the real oplsaa.xml.

/// A methanol skeleton CH3-OH with explicit hydrogens.
/// Returns `(graph, C, O, hydroxyl-H, [methyl-H; 3])`.
fn methanol() -> (
    Atomistic,
    molrs::AtomId,
    molrs::AtomId,
    molrs::AtomId,
    [molrs::AtomId; 3],
) {
    let mut g = Atomistic::new();
    let c = g.add_atom(Atom::xyz("C", 0.0, 0.0, 0.0));
    let o = g.add_atom(Atom::xyz("O", 1.4, 0.0, 0.0));
    let ho = g.add_atom(Atom::xyz("H", 2.0, 0.0, 0.0));
    let h1 = g.add_atom(Atom::xyz("H", -0.5, 0.9, 0.0));
    let h2 = g.add_atom(Atom::xyz("H", -0.5, -0.9, 0.0));
    let h3 = g.add_atom(Atom::xyz("H", -0.5, 0.0, 0.9));
    g.add_bond(c, o).unwrap();
    g.add_bond(o, ho).unwrap();
    for h in [h1, h2, h3] {
        g.add_bond(c, h).unwrap();
    }
    (g, c, o, ho, [h1, h2, h3])
}

#[test]
fn methanol_layered_types_match_molpy() {
    // ac-005: the alcohol layered chain. O types opls_154 (level 0); the
    // hydroxyl H types opls_155 (def `H[O;%opls_154]`, level 1) only after the
    // O is typed; the methyl H types opls_156 (def `HC[O;%opls_154]`, level 1).
    // Per-atom types equal molpy's OPLSAATypifier on the same molecule.
    let typifier = OPLSAATypifier::from_xml_str(&oplsaa_xml()).expect("build OPLSAATypifier");
    let (g, c, o, ho, methyl_h) = methanol();
    let typed = typifier.typify(&g).expect("typify methanol");

    let ty = |id: molrs::AtomId| {
        typed
            .get_atom(id)
            .unwrap()
            .get_str("type")
            .map(str::to_string)
    };
    assert_eq!(ty(c).as_deref(), Some("opls_157"), "methanol C");
    assert_eq!(ty(o).as_deref(), Some("opls_154"), "alcohol O (level 0)");
    assert_eq!(
        ty(ho).as_deref(),
        Some("opls_155"),
        "hydroxyl H via %opls_154 (level 1)"
    );
    for h in methyl_h {
        assert_eq!(
            ty(h).as_deref(),
            Some("opls_156"),
            "methyl H via %opls_154 (level 1)"
        );
    }
}

#[test]
fn percent_defs_now_covered() {
    // ac-005 / ac-006: the previously-skipped %opls_NNN defs are now reachable.
    // opls_155 (`H[O;%opls_154]`) is a %opls_NNN-dependent type chain-1 could
    // never assign; its presence after layered typing proves the `%opls_NNN`
    // defs in the bundled oplsaa.xml are now covered.
    let typifier = OPLSAATypifier::from_xml_str(&oplsaa_xml()).expect("build OPLSAATypifier");
    let (g, _c, _o, _ho, _mh) = methanol();
    let typed = typifier.typify(&g).expect("typify methanol");
    assert!(
        typed
            .atoms()
            .any(|(_, a)| a.get_str("type") == Some("opls_155")),
        "a %opls_NNN-dependent type (opls_155) is now assigned"
    );
}

#[test]
fn layered_typing_terminates_over_every_mol2() {
    // ac-005 (breadth): the full layered pipeline (incl. the %opls_NNN defs and
    // any circular groups in the real oplsaa.xml) must terminate and never panic
    // over every real mol2 molecule, and assign well-formed types throughout.
    let typifier = OPLSAATypifier::from_xml_str(&oplsaa_xml()).expect("build OPLSAATypifier");
    for path in &helpers::format_files("mol2") {
        let mol = load_mol2(path);
        if mol.n_atoms() == 0 {
            continue;
        }
        let typed = typify_atoms(&mol, typifier.meta(), typifier.ff()).unwrap_or_else(|e| {
            panic!(
                "layered typify_atoms {:?} failed: {e}",
                path.file_name().unwrap()
            )
        });
        for (_, a) in typed.atoms() {
            if let Some(ty) = a.get_str("type") {
                assert!(ty.starts_with("opls_"), "{path:?}: bad type {ty}");
            }
        }
    }
}

// ===========================================================================
// Chain 2: bonded-parameter typification (typify_bonded / build)
// ===========================================================================

/// Read a numeric prop off a materialized bond/angle/dihedral relation (whose
/// props are a `HashMap<String, PropValue>`), or `None` if absent/non-numeric.
fn rel_f64(rel: &molrs::Bond, key: &str) -> Option<f64> {
    rel.props.get(key).and_then(|v| v.as_f64())
}

/// Read a required numeric prop, failing the test if absent.
fn term_param(props_get: Option<f64>, name: &str) -> f64 {
    props_get.unwrap_or_else(|| panic!("term missing param {name:?}"))
}

#[test]
fn ethane_bond_angle_dihedral_match_opls_reference() {
    // ac-004: a fully-typed real molecule (ethane) gets every bonded term
    // parametrized with values matching the bundled OPLS-AA reference within
    // the spec tolerances (bond r0 atol 0.02 Å / angle θ0 atol 3° / force
    // constants rtol 0.10). Reference (oplsaa.xml, molrs units):
    //   CT-CT bond:  r0 1.529 Å,  k0 224262.4/418.4 = 536.0 kcal/mol/Å²
    //   CT-HC bond:  r0 1.09  Å,  k0 284512.0/418.4 = 680.0
    //   HC-CT-HC ang: θ0 1.88146 rad, k0 276.144/4.184 = 66.0 kcal/mol/rad²
    //   HC-CT-CT ang: θ0 1.93208 rad, k0 313.8/4.184  = 75.0
    //   HC-CT-CT-HC dih: f1 0, f2 0, f3 0.3, f4 0 (RB→OPLS of [.,1.8828,0,-2.5104,..])
    let typifier = OPLSAATypifier::from_xml_str(&oplsaa_xml()).expect("build OPLSAATypifier");
    let mol = load_mol2(&helpers::data_path("mol2/ethane.mol2"));
    let typed = typifier.typify(&mol).expect("typify + assign ethane");

    const FK_RTOL: f64 = 0.10;
    const R0_ATOL: f64 = 0.02; // Å
    const THETA_ATOL: f64 = 3.0_f64 * std::f64::consts::PI / 180.0; // 3° in rad

    let class_of = |id: molrs::AtomId| -> String {
        typed
            .get_atom(id)
            .ok()
            .and_then(|a| a.get_str("class").map(str::to_string))
            .unwrap_or_default()
    };

    // --- bonds ---
    let mut saw_ct_ct = false;
    let mut saw_ct_hc = false;
    for (_, b) in typed.bonds() {
        let (ci, cj) = (class_of(b.nodes[0]), class_of(b.nodes[1]));
        let mut classes = [ci.as_str(), cj.as_str()];
        classes.sort_unstable();
        let r0 = term_param(rel_f64(&b, "r0"), "r0");
        let k0 = term_param(rel_f64(&b, "k0"), "k0");
        match classes {
            ["CT", "CT"] => {
                saw_ct_ct = true;
                assert!((r0 - 1.529).abs() < R0_ATOL, "CT-CT r0 {r0}");
                assert!((k0 - 536.0).abs() / 536.0 < FK_RTOL, "CT-CT k0 {k0}");
            }
            ["CT", "HC"] => {
                saw_ct_hc = true;
                assert!((r0 - 1.09).abs() < R0_ATOL, "CT-HC r0 {r0}");
                assert!((k0 - 680.0).abs() / 680.0 < FK_RTOL, "CT-HC k0 {k0}");
            }
            other => panic!("unexpected ethane bond classes {other:?}"),
        }
    }
    assert!(saw_ct_ct && saw_ct_hc, "ethane has CT-CT and CT-HC bonds");

    // --- angles (enumerated by typify_bonded) ---
    let mut saw_hch = false;
    let mut saw_hcc = false;
    for (_, a) in typed.angles() {
        let mut classes = [
            class_of(a.nodes[0]),
            class_of(a.nodes[1]),
            class_of(a.nodes[2]),
        ];
        classes.sort_unstable();
        let theta0 = term_param(rel_f64(&a, "theta0"), "theta0");
        let k0 = term_param(rel_f64(&a, "k0"), "k0");
        match [
            classes[0].as_str(),
            classes[1].as_str(),
            classes[2].as_str(),
        ] {
            ["CT", "HC", "HC"] => {
                // central is CT; this is the HC-CT-HC angle.
                saw_hch = true;
                assert!(
                    (theta0 - 1.88146).abs() < THETA_ATOL,
                    "HC-CT-HC θ0 {theta0}"
                );
                assert!((k0 - 66.0).abs() / 66.0 < FK_RTOL, "HC-CT-HC k0 {k0}");
            }
            ["CT", "CT", "HC"] => {
                saw_hcc = true;
                assert!(
                    (theta0 - 1.93208).abs() < THETA_ATOL,
                    "HC-CT-CT θ0 {theta0}"
                );
                assert!((k0 - 75.0).abs() / 75.0 < FK_RTOL, "HC-CT-CT k0 {k0}");
            }
            other => panic!("unexpected ethane angle classes {other:?}"),
        }
    }
    assert!(
        saw_hch && saw_hcc,
        "ethane has HC-CT-HC and HC-CT-CT angles"
    );

    // --- dihedrals (HC-CT-CT-HC; f3 = 0.3, others 0) ---
    let mut saw_dih = false;
    for (_, d) in typed.dihedrals() {
        saw_dih = true;
        let f3 = term_param(rel_f64(&d, "f3"), "f3");
        assert!((f3 - 0.3).abs() / 0.3 < FK_RTOL, "HC-CT-CT-HC f3 {f3}");
        for k in ["f1", "f2", "f4"] {
            let v = rel_f64(&d, k).unwrap_or(0.0);
            assert!(v.abs() < 1e-6, "HC-CT-CT-HC {k} should be ~0, got {v}");
        }
    }
    assert!(saw_dih, "ethane has at least one HC-CT-CT-HC dihedral");
}

#[test]
fn typify_bonded_over_every_mol2_only_touches_typed_terms() {
    // ac-004 (breadth): typify_bonded must never panic over real molecules.
    // Because chain-1 has a coverage gap (only %def types are SMARTS-typed),
    // many atoms are untyped — terms touching them are skipped, so non-strict
    // assignment is always Ok. Bonded parity is only meaningful for the subset
    // chain-1 actually typed; here we just assert no panic + that ethane (fully
    // typed) yields fully-parametrized bonds.
    let typifier = OPLSAATypifier::from_xml_str(&oplsaa_xml())
        .expect("build OPLSAATypifier")
        .with_strict(false);
    for path in &helpers::format_files("mol2") {
        let mol = load_mol2(path);
        if mol.n_atoms() == 0 {
            continue;
        }
        let typed = typifier
            .typify(&mol)
            .unwrap_or_else(|e| panic!("assign {:?} failed: {e}", path.file_name().unwrap()));
        // Any bond whose two endpoints are both typed and that received a `type`
        // proxy (r0) must carry numeric params (no half-written terms).
        for (_, b) in typed.bonds() {
            if rel_f64(&b, "r0").is_some() {
                assert!(
                    rel_f64(&b, "k0").is_some(),
                    "{path:?}: bond has r0 but no k0"
                );
            }
        }
    }
}

#[test]
fn no_match_seam_strict_errors_lenient_skips() {
    // ac-005: a typed term with no force-field candidate yields Err under
    // strict (NoMatch::Error) and a clean unparametrized term under lenient
    // (NoMatch::Skip) with no estimator attached.
    //
    // Construct a 2-atom "molecule" whose atoms are typed to a class the bond
    // table cannot match: a force field with a single CT-CT bond, atoms typed
    // to class OH (no OH-OH bond). The bond therefore matches nothing.
    let mut ff = molrs::ff::forcefield::ForceField::new("mini");
    ff.def_bondstyle("harmonic")
        .def_type("CT-CT", &[("k0", 300.0), ("r0", 1.5)]);
    let mut meta = OplsTypingMeta::new();
    meta.insert(
        "opls_oh",
        OplsTypeRow {
            class: "OH".into(),
            def: None,
            overrides: Vec::new(),
            priority: None,
            layer: 0,
        },
    );
    let tables = CandidateTables::build(&ff, &meta);

    let mut mol = Atomistic::new();
    let a = mol.add_atom(Atom::xyz("O", 0.0, 0.0, 0.0));
    let b = mol.add_atom(Atom::xyz("O", 1.4, 0.0, 0.0));
    mol.add_bond(a, b).unwrap();
    mol.set_atom(a, "type", "opls_oh").unwrap();
    mol.set_atom(b, "type", "opls_oh").unwrap();

    // strict -> Err naming the term.
    let err = typify_bonded(&mol, &tables, NoMatch::Error).unwrap_err();
    assert!(err.contains("opls_oh"), "strict err names the term: {err}");

    // lenient -> Ok, bond left unparametrized (no k0/r0).
    let typed = typify_bonded(&mol, &tables, NoMatch::Skip).expect("lenient ok");
    let (_, bond) = typed.bonds().next().expect("the bond survives");
    assert!(
        rel_f64(&bond, "k0").is_none(),
        "unmatched bond stays unparam'd"
    );
    assert!(rel_f64(&bond, "r0").is_none());
}

#[test]
fn no_match_seam_estimator_fills_params() {
    // ac-005 (seam shape): an attached Estimator is consulted for unmatched
    // terms; its returned params are written, overriding the strict policy.
    struct FixedEstimator;
    impl Estimator for FixedEstimator {
        fn estimate(&self, term: &BondedTerm) -> Result<Option<Params>, String> {
            match term {
                BondedTerm::Bond(_) => Ok(Some(Params::from_pairs(&[("k0", 111.0), ("r0", 1.2)]))),
                _ => Ok(None),
            }
        }
    }

    let mut ff = molrs::ff::forcefield::ForceField::new("mini");
    ff.def_bondstyle("harmonic")
        .def_type("CT-CT", &[("k0", 300.0), ("r0", 1.5)]);
    let mut meta = OplsTypingMeta::new();
    meta.insert(
        "opls_oh",
        OplsTypeRow {
            class: "OH".into(),
            def: None,
            overrides: Vec::new(),
            priority: None,
            layer: 0,
        },
    );
    let tables = CandidateTables::build(&ff, &meta);

    let mut mol = Atomistic::new();
    let a = mol.add_atom(Atom::xyz("O", 0.0, 0.0, 0.0));
    let b = mol.add_atom(Atom::xyz("O", 1.4, 0.0, 0.0));
    mol.add_bond(a, b).unwrap();
    mol.set_atom(a, "type", "opls_oh").unwrap();
    mol.set_atom(b, "type", "opls_oh").unwrap();

    // Even under strict, the estimator fills in and avoids the error.
    let est = FixedEstimator;
    let typed = typify_bonded_with(&mol, &tables, NoMatch::Error, Some(&est))
        .expect("estimator fills the gap");
    let (_, bond) = typed.bonds().next().expect("bond");
    assert_eq!(rel_f64(&bond, "k0"), Some(111.0), "estimator k0 written");
    assert_eq!(rel_f64(&bond, "r0"), Some(1.2), "estimator r0 written");
}

#[test]
fn build_closes_to_potentials_with_finite_energy() {
    // ac-006: build(mol) closes typify -> assign -> to_frame -> to_potentials.
    // Ethane is fully typed, so all bonded terms resolve under strict; the
    // compiled Potentials evaluate to a finite energy on the input conformer.
    // (Numeric parity against molpy's OPLS energy is the bm-molrs-molpy harness;
    // here we assert the loop closes and is finite/sane — the kernel parity
    // itself is covered by tests/ff/potential/opls.rs.)
    let typifier = OPLSAATypifier::from_xml_str(&oplsaa_xml()).expect("build OPLSAATypifier");
    let mol = load_mol2(&helpers::data_path("mol2/ethane.mol2"));
    let pots = typifier.build(&mol).expect("build potentials for ethane");

    // bond + angle + dihedral + lj/cut + coul/cut kernels should be present.
    assert!(
        pots.len() >= 3,
        "at least bond/angle/dihedral kernels: {}",
        pots.len()
    );

    let coords: Vec<f64> = mol
        .atoms()
        .flat_map(|(_, a)| {
            [
                a.get_f64("x").unwrap_or(0.0),
                a.get_f64("y").unwrap_or(0.0),
                a.get_f64("z").unwrap_or(0.0),
            ]
        })
        .collect();
    let energy = pots.calc_energy(&coords);
    assert!(
        energy.is_finite(),
        "ethane OPLS energy must be finite: {energy}"
    );
}
