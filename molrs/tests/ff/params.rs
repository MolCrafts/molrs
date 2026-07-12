//! Antechamber parameter tables (`ff::params`).
//!
//! Three jobs:
//!
//! 1. **Manifest guard** — `MANIFEST.sha256` records the SHA-256 of every
//!    emitted `.rs`. This test recomputes them and needs **no AmberTools**, so
//!    unlike the drift guard below it actually runs in CI. It is what stops a
//!    hand-edit of the ~27k committed table lines from going unnoticed.
//! 2. **Drift guard** — with `$AMBERHOME` set, re-running the generator must
//!    byte-reproduce every committed `.rs`. That additionally catches an
//!    upstream AmberTools change. Without `$AMBERHOME` it skips: the committed
//!    `.rs` is the single in-repo source of truth, AmberTools merely upstream.
//!    CI has no AmberTools, so this one is permanently skipped there — hence (1).
//! 3. **Column semantics** — the tables carry meaning, not just numbers. The
//!    `GASPARM` `d` column in particular is chi+, a normalisation denominator,
//!    NOT a quartic coefficient, and it must stay a distinct named field.

use std::path::{Path, PathBuf};
use std::process::Command;

use sha2::{Digest, Sha256};

use molrs::ff::params::{
    ABCG2_ALIASES, ABCG2_CORRECTIONS, ATOMTYPE_ABCG2, ATOMTYPE_AMBER, ATOMTYPE_BCC, ATOMTYPE_GAS,
    ATOMTYPE_GFF, ATOMTYPE_GFF2, ATOMTYPE_SYBYL, BCC_ALIASES, BCC_CORRECTIONS, GASTEIGER_PARAMS,
};

/// Repository root (the workspace dir, one above this package).
fn repo_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("molrs/ has a parent")
}

fn generated_dir() -> PathBuf {
    repo_root().join("molrs/src/ff/params/generated")
}

fn chi_plus(atom_type: &str) -> f64 {
    GASTEIGER_PARAMS
        .iter()
        .find(|row| row.atom_type == atom_type)
        .unwrap_or_else(|| panic!("no GASPARM row for `{atom_type}`"))
        .chi_plus
}

/// Every committed table must still hash to what `MANIFEST.sha256` recorded.
///
/// This is the drift check that **runs in CI**. The byte-regeneration guard
/// below needs `$AMBERHOME`, which CI does not have, so it is permanently
/// skipped there — leaving ~27k lines of generated table protected by nothing
/// but a `DO NOT HAND-EDIT` comment. Recomputing the hashes needs no AmberTools
/// and catches exactly that: a hand-edit to a generated file.
///
/// A deliberate regeneration updates the tables AND the manifest together, so
/// this stays green; an edit to only one of them does not.
#[test]
fn committed_tables_match_the_manifest_hashes() {
    let dir = generated_dir();
    let manifest = std::fs::read_to_string(dir.join("MANIFEST.sha256"))
        .expect("MANIFEST.sha256 is committed alongside the generated tables");

    let mut checked = 0usize;
    for line in manifest.lines() {
        let Some(rest) = line.strip_prefix("emitted ") else {
            continue;
        };
        let mut it = rest.split_whitespace();
        let (want, name) = (
            it.next().expect("manifest row has a hash"),
            it.next().expect("manifest row has a file name"),
        );

        let bytes = std::fs::read(dir.join(name))
            .unwrap_or_else(|e| panic!("manifest lists `{name}` but it is not readable: {e}"));
        let got = format!("{:x}", Sha256::digest(&bytes));

        assert_eq!(
            got, want,
            "generated table `{name}` does not match MANIFEST.sha256.\n\
             It was hand-edited, or regenerated without committing the manifest.\n\
             Re-run: AMBERHOME=... python scripts/gen_param_tables.py"
        );
        checked += 1;
    }

    assert_eq!(
        checked, 11,
        "manifest should cover all 11 generated files (10 tables + mod.rs)"
    );
}

/// The generator must byte-reproduce every committed table.
///
/// Skips when `$AMBERHOME` is unset, so contributors and CI without AmberTools
/// are never blocked by it. That skip is why
/// [`committed_tables_match_the_manifest_hashes`] exists — it is the guard CI
/// actually runs.
#[test]
fn generator_byte_reproduces_the_committed_tables() {
    if std::env::var_os("AMBERHOME").is_none() {
        eprintln!("skipping param-table drift guard: $AMBERHOME is not set");
        return;
    }

    let out_dir = std::env::temp_dir().join(format!("molrs-param-drift-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&out_dir);

    let python = std::env::var("MOLRS_PYTHON").unwrap_or_else(|_| "python3".to_owned());
    let output = Command::new(&python)
        .arg(repo_root().join("scripts/gen_param_tables.py"))
        .arg("--out-dir")
        .arg(&out_dir)
        .output()
        .unwrap_or_else(|e| panic!("could not run `{python} scripts/gen_param_tables.py`: {e}"));
    assert!(
        output.status.success(),
        "generator failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let committed = generated_dir();
    let mut names: Vec<_> = std::fs::read_dir(&committed)
        .expect("read committed generated/")
        .map(|e| e.expect("dir entry").file_name())
        .collect();
    names.sort();
    // 10 tables + mod.rs + MANIFEST.sha256. The manifest is compared too: it
    // carries the upstream source hashes, so a change in AmberTools itself
    // shows up here even when the emitted tables happen to be unaffected.
    assert_eq!(
        names.len(),
        12,
        "expected 10 tables + mod.rs + MANIFEST.sha256"
    );

    let mut drifted = Vec::new();
    for name in &names {
        let want = std::fs::read(committed.join(name)).expect("read committed table");
        let got = std::fs::read(out_dir.join(name)).unwrap_or_default();
        if want != got {
            drifted.push(name.to_string_lossy().into_owned());
        }
    }
    let _ = std::fs::remove_dir_all(&out_dir);

    assert!(
        drifted.is_empty(),
        "generated tables drifted from the committed .rs: {drifted:?}\n\
         re-run `AMBERHOME=... python scripts/gen_param_tables.py` and commit the result"
    );
}

/// `GASPARM.DAT`'s five value columns carry three different meanings; they must
/// stay three differently-named fields, never an anonymous `[f64; 5]`.
#[test]
fn gasteiger_columns_keep_their_semantics() {
    // `a`/`b`/`c` are the electronegativity polynomial; `d` is chi+ (the
    // damping denominator); `formal_charge` is the seed charge q0.
    let h = GASTEIGER_PARAMS
        .iter()
        .find(|row| row.atom_type == "h")
        .expect("GASPARM has an `h` row");
    assert!((h.a - 7.17).abs() < 1e-12);
    assert!((h.b - 6.24).abs() < 1e-12);
    assert!((h.c - -0.56).abs() < 1e-12);
    assert!(
        (h.chi_plus - 20.02).abs() < 1e-12,
        "chi+ for H is the `d` column"
    );
    assert!((h.seed_charge - 0.0).abs() < 1e-12);

    assert!((chi_plus("h") - 20.02).abs() < 1e-12);
    assert!((chi_plus("c3") - 19.04).abs() < 1e-12);

    // chi+ is NOT a quartic coefficient: for every heavy atom it is the sum
    // a+b+c, exactly as the upstream file's trailing comment states.
    for row in GASTEIGER_PARAMS.iter().filter(|r| r.atom_type != "h") {
        let sum = row.a + row.b + row.c;
        assert!(
            (row.chi_plus - sum).abs() < 5e-2,
            "{}: chi+ {} should track a+b+c {}",
            row.atom_type,
            row.chi_plus,
            sum
        );
    }

    // Seed charges are the formal-charge column: the ionic types carry them.
    let anion = GASTEIGER_PARAMS
        .iter()
        .find(|row| row.atom_type == "o-1")
        .expect("GASPARM has an `o-1` row");
    assert!((anion.seed_charge - -1.0).abs() < 1e-12);
}

/// Row counts, pinned against the upstream tables the generator read.
#[test]
fn bcc_correction_tables_have_the_upstream_row_counts() {
    assert_eq!(BCC_CORRECTIONS.len(), 405, "BCCPARM.DAT rows");
    assert_eq!(BCC_ALIASES.len(), 0, "BCCPARM.DAT declares no CORR rows");
    assert_eq!(ABCG2_CORRECTIONS.len(), 439, "BCCPARM_ABCG2.DAT rows");
    assert_eq!(ABCG2_ALIASES.len(), 36, "BCCPARM_ABCG2.DAT CORR rows");
    assert_eq!(GASTEIGER_PARAMS.len(), 37, "GASPARM.DAT rows");

    // A spot value straight off BCCPARM.DAT line 3: `3  11  13  1  -0.0753`.
    let row = BCC_CORRECTIONS
        .iter()
        .find(|r| r.left == "11" && r.right == "13" && r.bond_type == 1)
        .expect("11|13|1 is in BCCPARM.DAT");
    assert!((row.delta - -0.0753).abs() < 1e-12);
}

/// All seven `.DEF` files parsed, and every rule kept its file order.
#[test]
fn every_atomtype_def_became_a_typed_table() {
    let tables = [
        (ATOMTYPE_BCC, "ATOMTYPE_BCC.DEF"),
        (ATOMTYPE_ABCG2, "ATOMTYPE_ABCG2.DEF"),
        (ATOMTYPE_GAS, "ATOMTYPE_GAS.DEF"),
        (ATOMTYPE_GFF, "ATOMTYPE_GFF.DEF"),
        (ATOMTYPE_GFF2, "ATOMTYPE_GFF2.DEF"),
        (ATOMTYPE_AMBER, "ATOMTYPE_AMBER.DEF"),
        (ATOMTYPE_SYBYL, "ATOMTYPE_SYBYL.DEF"),
    ];
    for (table, name) in tables {
        assert_eq!(table.name, name);
        assert!(!table.rules.is_empty(), "{name} has rules");
        assert!(!table.wildatoms.is_empty(), "{name} declares WILDATOMs");
    }

    // The BCC table is the one the AM1-BCC typifier walks; pin its shape.
    assert_eq!(ATOMTYPE_BCC.rules.len(), 159, "ATD rows minus the DU row");
    assert_eq!(ATOMTYPE_BCC.wildatoms.len(), 4);

    // Rule order is load-bearing (first match wins), so the first rule must
    // still be the 3-membered-ring sp3 carbon that opens the file.
    let first = &ATOMTYPE_BCC.rules[0];
    assert_eq!(first.atom_type, "11");
    assert_eq!(first.atomic_number, Some(6));
    assert_eq!(first.degree, Some(4));

    // `XB` is file-local: C3 N2 N3 O2 S2 P2 in BCC, but N/P in GFF.
    let bcc_xb = ATOMTYPE_BCC
        .wildatoms
        .iter()
        .find(|w| w.name == "XB")
        .expect("BCC declares XB");
    assert_eq!(bcc_xb.specs.len(), 6);
    let gff_xb = ATOMTYPE_GFF
        .wildatoms
        .iter()
        .find(|w| w.name == "XB")
        .expect("GFF declares XB");
    assert_eq!(gff_xb.specs.len(), 2);
}
