//! chem-perceive-15 architecture + reverse gates.
//!
//! These tests are the whole-chain acceptance: they assert the five "only one"
//! architecture promises (ac-001..ac-004) and the reverse runtime gates that
//! historically caught real defects on this chain (ac-007).
//!
//! **Acceptance rule:** a gate that finds a production defect stays RED. Do not
//! quietly fix production code inside this file — open a separate spec.
//!
//! Run:
//! ```text
//! cargo test -p molcrafts-molrs --features full --test architecture_gate
//! ```

#![cfg(feature = "ff")]

use std::fs;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

fn crate_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn src_dir() -> PathBuf {
    crate_dir().join("src")
}

fn params_dir() -> PathBuf {
    src_dir().join("ff").join("params")
}

/// Walk every `*.rs` under `root` (non-recursive helper used with a stack).
fn walk_rs_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
                out.push(path);
            }
        }
    }
    out.sort();
    out
}

/// Strip `//` line comments and `/* … */` block comments for crude identifier scans.
/// Bias: stripping can only remove matches, never invent them.
fn strip_comments(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let bytes = src.as_bytes();
    let mut i = 0;
    let mut in_block = false;
    let mut in_line = false;
    while i < bytes.len() {
        if in_line {
            if bytes[i] == b'\n' {
                in_line = false;
                out.push('\n');
            }
            i += 1;
            continue;
        }
        if in_block {
            if bytes[i] == b'*' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
                in_block = false;
                i += 2;
            } else {
                if bytes[i] == b'\n' {
                    out.push('\n');
                }
                i += 1;
            }
            continue;
        }
        if bytes[i] == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
            in_line = true;
            i += 2;
            continue;
        }
        if bytes[i] == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
            in_block = true;
            i += 2;
            continue;
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

// ===========================================================================
// ac-001 / ac-002 — ONE place, ONE form; no runtime table parsing
// ===========================================================================

#[test]
fn ac001_params_dir_is_flat_and_is_the_only_home() {
    let params = params_dir();
    assert!(
        params.is_dir(),
        "molrs/src/ff/params/ must exist (only home of parameter tables)"
    );

    // Flat: no subdirectories at all (generated/ is the historical sin).
    let mut subdirs = Vec::new();
    for entry in fs::read_dir(&params).expect("read params dir") {
        let entry = entry.expect("entry");
        if entry.file_type().expect("ft").is_dir() {
            subdirs.push(entry.file_name().to_string_lossy().into_owned());
        }
    }
    assert!(
        subdirs.is_empty(),
        "molrs/src/ff/params/ must be flat; found subdirs: {subdirs:?}"
    );

    // Historical path that must not return.
    assert!(
        !params.join(concat!("gener", "ated")).exists(),
        "molrs/src/ff/params/generated/ must not exist"
    );
    assert!(
        !src_dir().join("ff").join("mmff").join("tables.rs").exists(),
        "molrs/src/ff/mmff/tables.rs must not exist"
    );

    // molrs/data/ does not exist (neither under crate root nor workspace-style).
    assert!(
        !crate_dir().join("data").exists(),
        "molrs/data/ must not exist"
    );
}

#[test]
fn ac001_no_include_str_macro_in_production_src() {
    // Assemble the needle so this gate cannot exempt itself by naming the macro
    // in a way that would also match its own source if it lived under src/.
    let needle = concat!("include", "_str", "!");
    let mut hits = Vec::new();
    for path in walk_rs_files(&src_dir()) {
        let text = fs::read_to_string(&path).unwrap_or_default();
        let code = strip_comments(&text);
        if code.contains(needle) {
            hits.push(path.display().to_string());
        }
    }
    assert!(
        hits.is_empty(),
        "include_str! must not appear in production code under molrs/src (comments OK). hits: {hits:?}"
    );
}

#[test]
fn ac001_identifier_purge_gate() {
    // Spec forbids identifier *names* containing the two provenance tokens
    // (mod/file/type/fn). The words may appear in comments/docs. `generate`
    // (`crate::builder` / the word "generate" as a verb) is a different word and is OK.
    //
    // Needles assembled with concat! so this gate's own `fn` name never contains
    // them as a contiguous identifier (which would trip the gate itself).
    let bad_a = concat!("gener", "ated");
    let bad_b = concat!("gener", "ator");

    let mut violations = Vec::new();

    // File / directory names under src/ and tests/
    for root in [src_dir(), crate_dir().join("tests")] {
        if !root.exists() {
            continue;
        }
        let mut stack = vec![root];
        while let Some(dir) = stack.pop() {
            let Ok(entries) = fs::read_dir(&dir) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                let name = entry.file_name().to_string_lossy().to_lowercase();
                if name.contains(bad_a) || name.contains(bad_b) {
                    violations.push(format!("path name: {}", path.display()));
                }
                if path.is_dir() {
                    stack.push(path);
                }
            }
        }
    }

    // Identifiers in source: mod/fn/struct/enum/type/trait names only.
    let id_re = regex_lite::Regex::new(&format!(
        r"(?m)\b(mod|fn|struct|enum|type|trait)\s+([A-Za-z0-9_]*(?:{bad_a}|{bad_b})[A-Za-z0-9_]*)\b"
    ))
    .expect("regex");

    for root in [src_dir(), crate_dir().join("tests")] {
        if !root.exists() {
            continue;
        }
        for path in walk_rs_files(&root) {
            let text = fs::read_to_string(&path).unwrap_or_default();
            let code = strip_comments(&text);
            for cap in id_re.captures_iter(&code) {
                let kind = cap.get(1).unwrap().as_str();
                let name = cap.get(2).unwrap().as_str();
                violations.push(format!(
                    "{kind} {name} in {}",
                    path.strip_prefix(crate_dir()).unwrap_or(&path).display()
                ));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "provenance tokens must not appear in identifiers (file/mod/type/fn). hits:\n{}",
        violations.join("\n")
    );
}

#[test]
fn ac002_shipped_params_path_has_no_runtime_table_parse() {
    // Shipped tables live under ff/params as typed Rust. Runtime JSON/XML parse
    // of those tables is forbidden there. (Caller-supplied XML via from_xml_str
    // still uses forcefield/xml.rs — that is a user input path, not a table home.)
    let params = params_dir();
    let needles = [
        concat!("serde_json", "::", "from_str"),
        concat!("serde_json", "::", "from_slice"),
        concat!("serde_json", "::", "from_reader"),
        concat!("roxmltree", "::"),
        concat!("include", "_str", "!"),
    ];
    let mut hits = Vec::new();
    for path in walk_rs_files(&params) {
        let text = fs::read_to_string(&path).unwrap_or_default();
        let code = strip_comments(&text);
        for n in &needles {
            if code.contains(n) {
                hits.push(format!("{} contains {n}", path.display()));
            }
        }
    }
    assert!(
        hits.is_empty(),
        "ff/params must not parse table text at runtime: {hits:?}"
    );
}

// ===========================================================================
// ac-003 — ONE perception layer, ONE interpolation seam, ONE MMFF path
// ===========================================================================

#[test]
fn ac003_no_legacy_chem_module_alias() {
    // The perceive layer replaced the old top-level chem module alias.
    // Needle assembled with concat! so this file never holds the contiguous path.
    let needle = concat!("molrs", "::", "chem");
    let crate_root = crate_dir();
    let workspace = crate_root
        .parent()
        .expect("molrs crate has a parent workspace root");
    let roots = [
        workspace.join("molrs"),
        workspace.join("molrs-cxxapi"),
        workspace.join("molrs-python"),
        workspace.join("molrs-ffi"),
        workspace.join("molrs-wasm"),
        workspace.join("molrs-capi"),
    ];
    let mut hits = Vec::new();
    for root in &roots {
        if !root.exists() {
            continue;
        }
        let mut stack = vec![root.clone()];
        while let Some(dir) = stack.pop() {
            let dir_name = dir.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if matches!(
                dir_name,
                "target" | ".venv" | "venv" | "node_modules" | ".tox" | "__pycache__" | "worktrees"
            ) {
                continue;
            }
            let Ok(entries) = fs::read_dir(&dir) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                    continue;
                }
                let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
                if !matches!(ext, "rs" | "py" | "pyi" | "toml" | "md") {
                    continue;
                }
                // This gate file may mention the alias only via concat!; after
                // expansion the binary has the needle but the source must not
                // contain a contiguous hit. Still scan production trees fully.
                let text = fs::read_to_string(&path).unwrap_or_default();
                if text.contains(needle) {
                    hits.push(path.display().to_string());
                }
            }
        }
    }
    assert!(
        hits.is_empty(),
        "legacy chem module alias must have 0 hits; found in: {hits:?}"
    );
}

#[test]
fn ac003_parameter_interpolator_has_exactly_one_implementor() {
    // ParameterInterpolator + Parmchk2Estimator is the only interpolation seam.
    let mut impls = Vec::new();
    for path in walk_rs_files(&src_dir()) {
        let text = fs::read_to_string(&path).unwrap_or_default();
        let code = strip_comments(&text);
        // Match: impl … ParameterInterpolator for TypeName
        for line in code.lines() {
            let line = line.trim();
            if !line.starts_with("impl") || !line.contains("ParameterInterpolator") {
                continue;
            }
            if let Some(idx) = line.find(" for ") {
                let rest = &line[idx + 5..];
                let name = rest.split([' ', '<', '{']).next().unwrap_or("").trim();
                if !name.is_empty() && name != "T" {
                    impls.push((name.to_string(), path.display().to_string()));
                }
            }
        }
    }
    assert_eq!(
        impls.len(),
        1,
        "ParameterInterpolator must have exactly one implementor \
         (Parmchk2Estimator); found {impls:?}"
    );
    assert_eq!(
        impls[0].0, "Parmchk2Estimator",
        "sole ParameterInterpolator implementor must be Parmchk2Estimator, got {:?}",
        impls[0]
    );
}

#[test]
fn ac003_no_build_mmff_potentials_free_function() {
    let needle = concat!("build", "_mmff", "_potentials");
    let mut hits = Vec::new();
    for path in walk_rs_files(&src_dir()) {
        let text = fs::read_to_string(&path).unwrap_or_default();
        let code = strip_comments(&text);
        if code.contains(needle) {
            hits.push(path.display().to_string());
        }
    }
    assert!(
        hits.is_empty(),
        "build_mmff_potentials free function must not exist: {hits:?}"
    );
}

// ===========================================================================
// ac-004 — ParamSource::PerInstance is bidirectional (semantics, not spelling)
// ===========================================================================

/// Built-in styles that MUST be registered PerInstance (kernel ignores type rows).
const PER_INSTANCE_STYLES: &[(&str, &str)] = &[
    ("pair", "coul/cut"),
    ("bond", "mmff_bond"),
    ("angle", "mmff_angle"),
    ("angle", "mmff_stbn"),
    ("dihedral", "mmff_torsion"),
    ("improper", "mmff_oop"),
    ("bond", "uff_bond"),
    ("angle", "uff_angle"),
    ("dihedral", "uff_torsion"),
    ("pair", "uff_lj"),
    ("improper", "uff_inversion"),
    ("kspace", "pme"),
];

/// Built-in styles that MUST remain TypeRows (kernel indexes `tp`).
const TYPE_ROWS_STYLES: &[(&str, &str)] = &[
    ("bond", "harmonic"),
    ("bond", "class2"),
    ("bond", "morse"),
    ("angle", "harmonic"),
    ("angle", "class2"),
    ("dihedral", "opls"),
    ("dihedral", "charmm"),
    ("dihedral", "multi/harmonic"),
    ("dihedral", "periodic"),
    ("dihedral", "fourier"),
    ("dihedral", "class2"),
    ("pair", "lj/cut"),
    ("pair", "lj/class2"),
    ("pair", "buck"),
    ("pair", "morse"),
    ("pair", "thole"),
    ("pair", "coul/tt"),
    ("pair", "mmff_vdw"),
    ("improper", "harmonic"),
    ("improper", "cvff"),
    ("improper", "periodic"),
];

#[test]
fn ac004_param_source_registration_is_declared() {
    use molrs::ff::potential::{ParamSource, lookup_param_source};

    for &(cat, name) in PER_INSTANCE_STYLES {
        let src = lookup_param_source(cat, name);
        assert_eq!(
            src,
            Some(ParamSource::PerInstance),
            "{cat}/{name} must be registered ParamSource::PerInstance, got {src:?}"
        );
    }
    for &(cat, name) in TYPE_ROWS_STYLES {
        let src = lookup_param_source(cat, name);
        assert_eq!(
            src,
            Some(ParamSource::TypeRows),
            "{cat}/{name} must be registered ParamSource::TypeRows, got {src:?}"
        );
    }
}

#[test]
fn ac004_param_source_is_bidirectional_on_semantics_not_spelling() {
    // Half A — source: a kernel ctor whose second parameter is *ignored by
    // binding* (leading underscore, any spelling: `_tp`, `_type_params`, …)
    // must be registered PerInstance. Half B — registration: every PerInstance
    // style's ctor must ignore its type-params (leading underscore).
    //
    // This is deliberately broader than a grep for the spelling `_tp`, which
    // historically missed `pme_ctor` / `pair_coul_cut_ctor` (`_type_params`).
    use molrs::ff::potential::{ParamSource, lookup_param_source};

    // Map well-known ctor function → (category, style name).
    let ctor_to_style: &[(&str, &str, &str)] = &[
        ("pair_coul_cut_ctor", "pair", "coul/cut"),
        ("pme_ctor", "kspace", "pme"),
        ("mmff_bond_ctor", "bond", "mmff_bond"),
        ("mmff_angle_ctor", "angle", "mmff_angle"),
        ("mmff_stbn_ctor", "angle", "mmff_stbn"),
        ("mmff_torsion_ctor", "dihedral", "mmff_torsion"),
        ("mmff_oop_ctor", "improper", "mmff_oop"),
        ("mmff_vdw_ctor", "pair", "mmff_vdw"),
        ("uff_bond_ctor", "bond", "uff_bond"),
        ("uff_angle_ctor", "angle", "uff_angle"),
        ("uff_torsion_ctor", "dihedral", "uff_torsion"),
        ("uff_lj_ctor", "pair", "uff_lj"),
        ("uff_inversion_ctor", "improper", "uff_inversion"),
        ("bond_harmonic_ctor", "bond", "harmonic"),
        ("bond_class2_ctor", "bond", "class2"),
        ("bond_morse_ctor", "bond", "morse"),
        ("angle_harmonic_ctor", "angle", "harmonic"),
        ("angle_class2_ctor", "angle", "class2"),
        ("dihedral_opls_ctor", "dihedral", "opls"),
        ("dihedral_charmm_ctor", "dihedral", "charmm"),
        ("dihedral_multi_harmonic_ctor", "dihedral", "multi/harmonic"),
        ("dihedral_periodic_ctor", "dihedral", "periodic"),
        ("dihedral_class2_ctor", "dihedral", "class2"),
        ("pair_lj_cut_ctor", "pair", "lj/cut"),
        ("pair_lj_class2_ctor", "pair", "lj/class2"),
        ("pair_buck_ctor", "pair", "buck"),
        ("pair_morse_ctor", "pair", "morse"),
        ("pair_thole_ctor", "pair", "thole"),
        ("pair_tang_toennies_ctor", "pair", "coul/tt"),
        ("improper_harmonic_ctor", "improper", "harmonic"),
        ("improper_cvff_ctor", "improper", "cvff"),
        ("improper_periodic_ctor", "improper", "periodic"),
    ];

    // Discover second-parameter names for every `pub fn *_ctor` under potential/.
    let pot_root = src_dir().join("ff").join("potential");
    let mut second_param: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    for path in walk_rs_files(&pot_root) {
        let text = fs::read_to_string(&path).unwrap_or_default();
        let code = strip_comments(&text);
        // Crude multi-line: pub fn name_ctor( … first, second_name : …
        let lines: Vec<&str> = code.lines().collect();
        for i in 0..lines.len() {
            let line = lines[i].trim();
            if !line.contains("fn ") || !line.contains("_ctor") {
                continue;
            }
            // Find "fn NAME"
            let Some(fn_pos) = line.find("fn ") else {
                continue;
            };
            let after = &line[fn_pos + 3..];
            let name = after
                .split(|c: char| c == '(' || c == '<' || c.is_whitespace())
                .next()
                .unwrap_or("");
            if !name.ends_with("_ctor") {
                continue;
            }
            // Collect signature text until we see the second parameter.
            let mut sig = String::new();
            for line in lines.iter().take(i.saturating_add(8)).skip(i) {
                sig.push(' ');
                sig.push_str(line.trim());
                if sig.matches(',').count() >= 2 && sig.contains(':') {
                    break;
                }
            }
            // After first comma: the type-params argument.
            let Some(comma) = sig.find(',') else {
                continue;
            };
            let rest = sig[comma + 1..].trim();
            let param = rest
                .split(':')
                .next()
                .unwrap_or("")
                .trim()
                .trim_start_matches("mut ")
                .trim();
            if !param.is_empty() {
                second_param.insert(name.to_string(), param.to_string());
            }
        }
    }

    let mut failures = Vec::new();
    for &(ctor, cat, style) in ctor_to_style {
        let Some(param) = second_param.get(ctor) else {
            failures.push(format!("could not locate second param of {ctor}"));
            continue;
        };
        let ignores = param.starts_with('_');
        let src = lookup_param_source(cat, style);
        match (ignores, src) {
            (true, Some(ParamSource::PerInstance)) => {}
            (false, Some(ParamSource::TypeRows)) => {}
            (true, other) => failures.push(format!(
                "{ctor} ignores type-params as `{param}` but {cat}/{style} is registered {other:?} \
                 (must be PerInstance)"
            )),
            (false, other) => failures.push(format!(
                "{ctor} reads type-params as `{param}` but {cat}/{style} is registered {other:?} \
                 (must be TypeRows)"
            )),
        }
    }

    // Smoke: PerInstance coul/cut truly ignores type-params content.
    // Two zero-charge atoms with one pair; empty tp and garbage tp must both
    // construct and both give exactly 0 energy (TypeRows kernels would differ
    // or fail on unknown type labels).
    {
        use molrs::Atomistic;
        use molrs::ff::forcefield::Params;
        use molrs::ff::potential::{Potential, lookup_kernel};
        use molrs::system::molgraph::Atom;

        // Build a typed frame via the graph so we do not need ndarray in tests.
        let mut mol = Atomistic::new();
        let mut a0 = Atom::xyz("He", 0.0, 0.0, 0.0);
        a0.set("charge", 0.0_f64);
        let mut a1 = Atom::xyz("He", 1.5, 0.0, 0.0);
        a1.set("charge", 0.0_f64);
        let _i0 = mol.add_atom(a0);
        let _i1 = mol.add_atom(a1);
        // No bond → intramolecular_pairs will include (0,1) as a non-1-4 pair.
        let mut frame = mol.to_frame();
        frame.insert("pairs", molrs::ff::potential::intramolecular_pairs(&frame));

        let mut style = Params::new();
        style.set("coulomb", 332.0716);
        style.set("dielectric", 1.0);
        style.set("coulomb14scale", 0.75);
        style.set("delta", 0.05);

        let ctor = lookup_kernel("pair", "coul/cut").expect("coul/cut kernel");
        let empty: &[(&str, &Params)] = &[];
        let garbage_params = Params::from_pairs(&[("k", 999.0)]);
        let garbage: &[(&str, &Params)] = &[("ZZZ-YYY", &garbage_params)];

        match (ctor(&style, empty, &frame), ctor(&style, garbage, &frame)) {
            (Ok(a), Ok(b)) => {
                let coords = [0.0, 0.0, 0.0, 1.5, 0.0, 0.0];
                let ea = Potential::calc_energy(&*a, &coords);
                let eb = Potential::calc_energy(&*b, &coords);
                if ea != 0.0 || eb != 0.0 || ea != eb {
                    failures.push(format!(
                        "coul/cut must ignore tp and give exactly 0.0 for zero charges; \
                         empty_tp={ea}, garbage_tp={eb}"
                    ));
                }
            }
            (Err(e), _) | (_, Err(e)) => {
                failures.push(format!("coul/cut ctor failed (empty or garbage tp): {e}"));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "ParamSource bidirectional semantic gate failed:\n{}",
        failures.join("\n")
    );
}

// ===========================================================================
// ac-005 / ac-006 — golden fixtures live in molpy tests (no live tools)
// ===========================================================================
// molpy/tests/test_typifier/test_mmff.py  — RDKit-locked MMFF types/energy
// molpy/tests/test_typifier/test_atd.py   — antechamber-locked ATD types
// cxxapi am1bcc_bridge + am1bcc_reference.rs — full charge matrix (C++ bridge)

#[test]
#[ignore = "ac-005 remainder: full 37-molecule charge/parmchk matrix beyond \
            molpy ATD/MMFF goldens; see cxxapi am1bcc_bridge."]
fn ac005_end_to_end_external_oracles_stub() {}

#[test]
#[ignore = "ac-006: covered by molpy typifier goldens; expand as needed there."]
fn ac006_python_rust_bit_parity_stub() {}

// ===========================================================================
// ac-007 — REVERSE gates (runtime, live molrs APIs)
// ===========================================================================

#[cfg(feature = "smiles")]
mod reverse {
    use molrs::Atomistic;
    use molrs::ff::potential::{extract_coords, intramolecular_pairs};
    use molrs::ff::typifier::mmff::{MMFF94STypifier, MMFF94Typifier};
    use molrs::io::smiles::{parse_smiles, to_atomistic};
    use molrs::perceive::Perceive;

    /// SMILES → Atomistic with explicit H. Perception is graph-in/graph-out.
    fn mol_from_smiles(smiles: &str) -> Atomistic {
        let ir = parse_smiles(smiles).unwrap_or_else(|e| panic!("parse {smiles}: {e}"));
        let mol = to_atomistic(&ir).unwrap_or_else(|e| panic!("atomize {smiles}: {e}"));
        // Rings + aromaticity (for aromatic SMILES) then explicit hydrogens.
        let mol = Perceive::new().find_rings(&mol);
        let mol = Perceive::new().find_aromaticity(&mol);
        Perceive::new().find_hydrogens(&mol)
    }

    /// Assign crude non-overlapping coordinates when the SMILES path has none.
    /// Energies only need *some* geometry; charges/impropers need none.
    fn ensure_coords(mol: &mut Atomistic) {
        let ids: Vec<_> = mol.atoms().map(|(id, _)| id).collect();
        for (i, id) in ids.into_iter().enumerate() {
            let atom = mol.get_atom(id).expect("atom");
            if atom.get_f64("x").is_some() {
                continue;
            }
            // Place on a mild helix so no two atoms coincide.
            let t = i as f64;
            let _ = mol.set_atom(id, "x", t * 1.2);
            let _ = mol.set_atom(id, "y", (t * 0.7).sin());
            let _ = mol.set_atom(id, "z", (t * 0.5).cos());
        }
    }

    fn mmff94_frame(smiles: &str) -> molrs::Frame {
        let mut mol = mol_from_smiles(smiles);
        ensure_coords(&mut mol);
        let typed = MMFF94Typifier::new()
            .typify(&mol)
            .unwrap_or_else(|e| panic!("MMFF94 typify {smiles}: {e}"));
        typed.to_frame()
    }

    /// Electrostatic energy alone (pair/coul/cut), not the total.
    fn coul_energy(frame: &molrs::Frame, typifier_ff: &molrs::ff::forcefield::ForceField) -> f64 {
        use molrs::ff::potential::Potential;

        let mut frame = frame.clone();
        if frame.get("pairs").is_none() {
            frame.insert("pairs", intramolecular_pairs(&frame));
        }
        let style = typifier_ff
            .get_style("pair", "coul/cut")
            .expect("MMFF force field must declare pair/coul/cut");
        let pot = style
            .to_potential(&frame, typifier_ff.special_bonds())
            .expect("coul style → potential")
            .expect("coul style must produce a potential when pairs exist");
        let coords = extract_coords(&frame).expect("coords");
        Potential::calc_energy(&*pot, &coords)
    }

    #[test]
    fn ac007_zero_charge_molecule_has_exactly_zero_electrostatics() {
        // Ethane: one of the few molecules whose MMFF charges are all zero.
        // A missing coulomb style used to hide behind ethane's total energy.
        let typifier = MMFF94Typifier::new();
        let frame = mmff94_frame("CC");

        let charges = frame
            .get("atoms")
            .expect("atoms")
            .get_float("charge")
            .expect("charge column");
        assert!(
            charges.iter().all(|&q| q == 0.0),
            "ethane MMFF charges must all be exactly 0.0 for this gate to be meaningful; got {charges:?}"
        );

        let e_ele = coul_energy(&frame, typifier.ff());
        assert_eq!(
            e_ele, 0.0,
            "zero-charge molecule must have EXACTLY 0.0 electrostatic energy, not approximately; got {e_ele}"
        );
    }

    #[test]
    fn ac007_benzene_has_impropers() {
        // Benzene had ZERO impropers silently before this chain. oop energy of
        // 0.0 on a planar ring is nearly right, which is how it hid.
        let frame = mmff94_frame("c1ccccc1");
        let n = frame.get("impropers").and_then(|b| b.nrows()).unwrap_or(0);
        assert!(
            n > 0,
            "benzene MUST have impropers after MMFF typify (got {n})"
        );
    }

    #[test]
    fn ac007_nitrate_oxygens_have_equal_charges() {
        // [O-][N+](=O)[O-] — three chemically equivalent oxygens.
        let frame = mmff94_frame("[O-][N+](=O)[O-]");
        let atoms = frame.get("atoms").expect("atoms");
        let elements = atoms.get_string("element").expect("element");
        let charges = atoms.get_float("charge").expect("charge");

        let o_charges: Vec<f64> = elements
            .iter()
            .zip(charges.iter())
            .filter(|(el, _)| el.eq_ignore_ascii_case("O"))
            .map(|(_, &q)| q)
            .collect();
        assert_eq!(
            o_charges.len(),
            3,
            "nitrate must have 3 oxygens, got {}",
            o_charges.len()
        );
        let q0 = o_charges[0];
        for (i, &q) in o_charges.iter().enumerate() {
            assert!(
                (q - q0).abs() < 1e-12,
                "nitrate oxygen charges must be equal within 1e-12; O0={q0}, O{i}={q}"
            );
        }
    }

    #[test]
    fn ac007_acetate_carboxylate_oxygens_have_equal_charges() {
        // CC(=O)[O-] — the two carboxylate oxygens must carry equal charge.
        // Historically they differed by ~0.2014 e when equivalencing was missing.
        let frame = mmff94_frame("CC(=O)[O-]");
        let atoms = frame.get("atoms").expect("atoms");
        let elements = atoms.get_string("element").expect("element");
        let charges = atoms.get_float("charge").expect("charge");
        let types = atoms.get_string("type");

        // Prefer MMFF carboxylate oxygen type (32 = O2CM) when present; else all O.
        let mut o_charges: Vec<f64> = Vec::new();
        for i in 0..elements.len() {
            if !elements[i].eq_ignore_ascii_case("O") {
                continue;
            }
            if let Some(types) = types {
                if types[i] == "32" || types[i] == "O2CM" {
                    o_charges.push(charges[i]);
                }
            } else {
                o_charges.push(charges[i]);
            }
        }
        // Fallback: if type filter left us short, use every oxygen except if we
        // only expected carboxylate pair — acetate has exactly two O.
        if o_charges.len() < 2 {
            o_charges = elements
                .iter()
                .zip(charges.iter())
                .filter(|(el, _)| el.eq_ignore_ascii_case("O"))
                .map(|(_, &q)| q)
                .collect();
        }
        assert!(
            o_charges.len() >= 2,
            "acetate must expose ≥2 oxygen charges, got {o_charges:?}"
        );
        // The two most-negative (carboxylate) or the typed pair — for acetate
        // both oxygens should match each other when type-filtered; when not,
        // both oxygens are the carboxylate pair.
        let q0 = o_charges[0];
        for (i, &q) in o_charges.iter().enumerate() {
            assert!(
                (q - q0).abs() < 1e-12,
                "acetate carboxylate oxygen charges must be equal within 1e-12; \
                 O0={q0}, O{i}={q}; all={o_charges:?}"
            );
        }
    }

    #[test]
    fn ac007_mmff94_and_mmff94s_are_bit_identical_on_ethane() {
        // No delocalized N → MMFF94 and MMFF94s MUST be bit-identical. Else a
        // "they differ" test can pass on a difference that does not exist.
        let mut mol = mol_from_smiles("CC");
        ensure_coords(&mut mol);

        let t94 = MMFF94Typifier::new();
        let t94s = MMFF94STypifier::new();

        let mut f94 = t94.typify(&mol).expect("typify 94").to_frame();
        let mut f94s = t94s.typify(&mol).expect("typify 94s").to_frame();
        f94.insert("pairs", intramolecular_pairs(&f94));
        f94s.insert("pairs", intramolecular_pairs(&f94s));

        let pots94 = t94.ff().to_potentials(&f94).expect("potentials 94");
        let pots94s = t94s.ff().to_potentials(&f94s).expect("potentials 94s");
        let coords = extract_coords(&f94).expect("coords");

        let e94 = pots94.calc_energy(&coords);
        let e94s = pots94s.calc_energy(&coords);
        assert_eq!(
            e94.to_bits(),
            e94s.to_bits(),
            "MMFF94 and MMFF94s must be BIT-IDENTICAL on ethane (no delocalized N); \
             e94={e94}, e94s={e94s}"
        );
    }
}

// ===========================================================================
// Tiny local regex — avoid adding a regex crate dep for a one-shot gate.
// ===========================================================================

/// Minimal regex surface used by the identifier gate (no new crate dep).
mod regex_lite {
    pub struct Regex {
        /// Lowercased needles that must appear inside the captured name.
        needles: Vec<String>,
        /// Source pattern kept only for error messages.
        _pat: String,
    }

    pub struct Captures<'a> {
        kind: &'a str,
        name: &'a str,
        kind_range: (usize, usize),
        name_range: (usize, usize),
    }

    impl<'a> Captures<'a> {
        pub fn get(&self, i: usize) -> Option<Match<'a>> {
            match i {
                1 => Some(Match {
                    text: self.kind,
                    start: self.kind_range.0,
                    end: self.kind_range.1,
                }),
                2 => Some(Match {
                    text: self.name,
                    start: self.name_range.0,
                    end: self.name_range.1,
                }),
                _ => None,
            }
        }
    }

    pub struct Match<'a> {
        text: &'a str,
        start: usize,
        end: usize,
    }

    impl<'a> Match<'a> {
        pub fn as_str(&self) -> &'a str {
            self.text
        }
        #[allow(dead_code)]
        pub fn start(&self) -> usize {
            self.start
        }
        #[allow(dead_code)]
        pub fn end(&self) -> usize {
            self.end
        }
    }

    impl Regex {
        pub fn new(pat: &str) -> Result<Self, String> {
            // We only need to detect keyword + name containing generated/generator.
            // Extract the two needles from the pattern if present; else fail.
            let mut needles = Vec::new();
            for n in ["generated", "generator"] {
                if pat.to_lowercase().contains(n) {
                    needles.push(n.to_string());
                }
            }
            if needles.is_empty() {
                return Err("regex_lite: expected generated/generator needles".into());
            }
            Ok(Self {
                needles,
                _pat: pat.to_string(),
            })
        }

        pub fn captures_iter<'a>(&'a self, text: &'a str) -> CapturesIter<'a> {
            CapturesIter {
                text,
                needles: &self.needles,
                pos: 0,
            }
        }
    }

    pub struct CapturesIter<'a> {
        text: &'a str,
        needles: &'a [String],
        pos: usize,
    }

    impl<'a> Iterator for CapturesIter<'a> {
        type Item = Captures<'a>;

        fn next(&mut self) -> Option<Self::Item> {
            const KEYWORDS: &[&str] = &["mod", "fn", "struct", "enum", "type", "trait"];
            let bytes = self.text.as_bytes();
            while self.pos < self.text.len() {
                // Find a keyword at a word boundary.
                let rest = &self.text[self.pos..];
                let mut best: Option<(usize, &'static str)> = None;
                for kw in KEYWORDS {
                    if let Some(rel) = rest.find(kw) {
                        let abs = self.pos + rel;
                        // Word boundary before.
                        if abs > 0 {
                            let prev = bytes[abs - 1] as char;
                            if prev.is_ascii_alphanumeric() || prev == '_' {
                                continue;
                            }
                        }
                        // Word boundary after keyword.
                        let after = abs + kw.len();
                        if after < bytes.len() {
                            let next = bytes[after] as char;
                            if next.is_ascii_alphanumeric() || next == '_' {
                                continue;
                            }
                        }
                        match best {
                            Some((b, _)) if rel >= b => {}
                            _ => best = Some((rel, *kw)),
                        }
                    }
                }
                let (rel, kw) = best?;
                let kw_start = self.pos + rel;
                let after_kw = kw_start + kw.len();
                // Skip whitespace.
                let mut i = after_kw;
                while i < bytes.len() && (bytes[i] as char).is_whitespace() {
                    i += 1;
                }
                // Read identifier.
                let name_start = i;
                while i < bytes.len() {
                    let c = bytes[i] as char;
                    if c.is_ascii_alphanumeric() || c == '_' {
                        i += 1;
                    } else {
                        break;
                    }
                }
                if i == name_start {
                    self.pos = after_kw;
                    continue;
                }
                let name = &self.text[name_start..i];
                self.pos = i;
                let lower = name.to_ascii_lowercase();
                if self.needles.iter().any(|n| lower.contains(n.as_str())) {
                    return Some(Captures {
                        kind: kw,
                        name,
                        kind_range: (kw_start, kw_start + kw.len()),
                        name_range: (name_start, i),
                    });
                }
            }
            None
        }
    }
}
