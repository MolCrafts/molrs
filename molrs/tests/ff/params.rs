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
        checked, 15,
        "manifest should cover all 15 generated files (14 tables + mod.rs). \
         chem-perceive-10 ac-001 adds two: `gaff_equiv.rs` and `gaff_empirical.rs`, \
         the estimator's substitution and empirical-constant tables, which are still \
         runtime-parsed JSON (see `gaff_equiv_and_empirical_are_generated_tables`)"
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
    // 14 tables + mod.rs + MANIFEST.sha256. The manifest is compared too: it
    // carries the upstream source hashes, so a change in AmberTools itself
    // shows up here even when the emitted tables happen to be unaffected.
    assert_eq!(
        names.len(),
        16,
        "expected 14 tables + mod.rs + MANIFEST.sha256"
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

/// Hydrogen is the row where chi+ and `a+b+c` DIFFER — and chi+ is the one that is right.
///
/// The heavy rows above satisfy `d == a+b+c` to the last decimal, which is what says
/// `d` is chi+ = chi(q=1), the cation electronegativity, rather than a fourth
/// polynomial coefficient. Hydrogen is the documented exception and it is the whole
/// reason the two cannot be conflated:
///
/// ```text
/// chi_plus(h) = 20.02          <- the `d` column, and what PEOE divides by
/// a + b + c   = 7.17 + 6.24 - 0.56 = 12.85
/// ```
///
/// H⁺ is a bare proton: it has no valence orbital left, so its polynomial chi+ is
/// physically meaningless and a fixed 20.02 eV is substituted. A model that "helpfully"
/// computed chi+ as `a+b+c` — the formula that works for all 36 other rows — would put
/// hydrogen's divisor 36% low and inflate every single H transfer.
///
/// This is one half of ac-002; the other half is that no code path uses `d` as a `q^3`
/// coefficient, which is checked against the SOURCE in
/// `charge::gasteiger_source::the_chi_plus_column_is_never_a_cubic_term`.
#[test]
fn hydrogens_chi_plus_is_not_its_polynomial_sum() {
    let h = GASTEIGER_PARAMS
        .iter()
        .find(|row| row.atom_type == "h")
        .expect("GASPARM has an `h` row");

    let polynomial_sum = h.a + h.b + h.c;
    assert!(
        (polynomial_sum - 12.85).abs() < 1e-12,
        "H's a+b+c is {polynomial_sum}, not 12.85 — the GASPARM row has moved"
    );
    assert!(
        (h.chi_plus - 20.02).abs() < 1e-12,
        "H's chi+ is the `d` column"
    );

    // The load-bearing assertion: the two are NOT the same number.
    assert!(
        (h.chi_plus - polynomial_sum).abs() > 7.0,
        "H's chi+ ({}) and its a+b+c ({polynomial_sum}) have converged. They are 7.17 \
         apart upstream, and that gap is the hydrogen special case: chi+(H) is a fixed \
         20.02 eV because H+ is a bare proton with no polynomial to evaluate. If they \
         ever became equal, `chi_plus` could be derived instead of read — and that \
         derivation is exactly the bug this pins.",
        h.chi_plus
    );

    // The contrast case: for carbon, the two DO agree, which is why `a+b+c` looks
    // like a safe shortcut right up until it meets hydrogen.
    let c3 = GASTEIGER_PARAMS
        .iter()
        .find(|row| row.atom_type == "c3")
        .expect("GASPARM has a `c3` row");
    assert!((chi_plus("c3") - 19.04).abs() < 1e-12);
    assert!(
        (c3.chi_plus - (c3.a + c3.b + c3.c)).abs() < 1e-12,
        "c3: chi+ 19.04 == a+b+c 7.98+9.18+1.88"
    );
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
    assert_eq!(
        ATOMTYPE_BCC.rules.len(),
        160,
        "all ATD rows INCLUDING the terminal DU catch-all (restored in spec 06)"
    );
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

/// The conjugated pairs `PARMCHK.DAT` declares, in its own header's words:
///
/// ```text
/// #   equivalent_flag:  1 for cc/ce/cg/nc/ne/pc/pe
/// #                     2 for cd/cf/ch/nd/nf/pd/pf
/// #                     0 for others
/// ```
///
/// Flag 1 is the name an `ATOMTYPE_GFF*.DEF` rule emits; flag 2 is the name
/// antechamber renames it to on the other colour of a conjugated system. The
/// pairing is upstream DATA — this is the table that declares it — which is why it
/// belongs on `AtdRule` and not in a `match` inside the engine.
const PARMCHK_PAIRS: [(&str, &str); 7] = [
    ("cc", "cd"),
    ("ce", "cf"),
    ("cg", "ch"),
    ("nc", "nd"),
    ("ne", "nf"),
    ("pc", "pd"),
    ("pe", "pf"),
];

/// Every conjugated GAFF type carries its `PARMCHK.DAT` partner.
///
/// The generator dropped the terminal `DU` row once, and nothing noticed until the
/// AMBER column of the oracle came up two atoms short. `alternate` is the same
/// shape of fact — a column of an upstream table that the emitted `.rs` is the only
/// in-repo copy of — so it gets the same kind of guard: pin the whole pairing set,
/// per table, by name.
#[test]
fn every_conjugated_gaff_type_carries_its_parmchk_alternate() {
    for table in [ATOMTYPE_GFF, ATOMTYPE_GFF2] {
        for (phase_1, phase_2) in PARMCHK_PAIRS {
            let rules: Vec<_> = table
                .rules
                .iter()
                .filter(|rule| rule.atom_type == phase_1)
                .collect();
            assert!(
                !rules.is_empty(),
                "{} declares no `{phase_1}` rule, so this pin is vacuous",
                table.name
            );
            for rule in rules {
                assert_eq!(
                    rule.alternate,
                    Some(phase_2),
                    "{}: the `{phase_1}` rule must carry its PARMCHK.DAT partner `{phase_2}` \
                     (equivalent_flag 1 -> 2); without it the 2-colouring pass has no name to \
                     rename half a conjugated system to",
                    table.name
                );
            }
        }
    }
}

/// The alternate names are not rows of the `.DEF` at all.
///
/// This is *why* `AtdRule::alternate` has to exist. `ATOMTYPE_GFF.DEF` can emit
/// `cc` and never `cd`: `cd` appears nowhere in the file. So `cd` cannot come from
/// walking the table — it can only come from the alternate column — and any engine
/// that reproduces `antechamber -at gaff` on pyrrole is reading it from here.
///
/// It is also what keeps `no_assigned_type_is_absent_from_the_table_being_walked`
/// honest after that gate is widened to `atom_type ∪ alternate`: the union is a
/// strictly larger set only because of these seven names, each of which is upstream
/// data rather than an engine invention.
#[test]
fn the_alternate_names_are_not_atd_rows_of_the_def_themselves() {
    for table in [ATOMTYPE_GFF, ATOMTYPE_GFF2] {
        let declared: Vec<&str> = table.rules.iter().map(|rule| rule.atom_type).collect();
        for (phase_1, phase_2) in PARMCHK_PAIRS {
            assert!(
                declared.contains(&phase_1),
                "{}: `{phase_1}` should be an ATD row",
                table.name
            );
            assert!(
                !declared.contains(&phase_2),
                "{} now declares an ATD row for `{phase_2}`. Upstream, it does not: the \
                 `.DEF` only ever emits the phase-1 name and antechamber renames half of \
                 each conjugated system afterwards. If this ever became true upstream, the \
                 alternate column — and the pass that applies it — would need re-deriving, \
                 not just this assertion relaxing",
                table.name
            );
        }
    }
}

/// A type with `equivalent_flag: 0` has no alternate — including the ones whose
/// names *look* like a conjugated pair.
///
/// The trap is AMBER: `ATOMTYPE_AMBER.DEF` has real `CC` and `CD` rows (parm94's
/// histidine carbons), and `PARMCHK.DAT` gives both `equivalent_flag 0`. They are
/// not a conjugated pair, they are not even the same alphabet, and AMBER types
/// 37/37 today. A generator that derived `alternate` from the *spelling* of a type
/// rather than from the flag column would pair them, 2-colour the AMBER table, and
/// break a passing column — so `CC -> None` is the assertion that keeps the
/// alternate column tied to `PARMCHK.DAT` instead of to a naming convention.
#[test]
fn a_type_with_equivalent_flag_zero_has_no_alternate() {
    // GAFF types whose PARMCHK.DAT equivalent_flag is 0.
    for table in [ATOMTYPE_GFF, ATOMTYPE_GFF2] {
        for atom_type in [
            "c3", "c2", "ca", "cp", "n", "na", "nb", "os", "s", "ha", "hc", "h1",
        ] {
            for rule in table.rules.iter().filter(|r| r.atom_type == atom_type) {
                assert_eq!(
                    rule.alternate, None,
                    "{}: `{atom_type}` has equivalent_flag 0 in PARMCHK.DAT — it is not half \
                     of a conjugated pair and must not be renamed by the 2-colouring pass",
                    table.name
                );
            }
        }
    }

    // The uppercase AMBER table: `CC`/`CD` are parm94 types, flag 0, not a pair.
    let amber_cc: Vec<_> = ATOMTYPE_AMBER
        .rules
        .iter()
        .filter(|rule| rule.atom_type == "CC")
        .collect();
    assert!(
        !amber_cc.is_empty(),
        "ATOMTYPE_AMBER.DEF declares CC rows; this pin is vacuous without them"
    );
    for rule in amber_cc {
        assert_eq!(
            rule.alternate, None,
            "ATOMTYPE_AMBER.DEF's `CC` is parm94's histidine carbon (equivalent_flag 0), not \
             GAFF's conjugated `cc`. Pairing it with `CD` would 2-colour the AMBER column, \
             which reproduces antechamber 37/37 today"
        );
    }
}

/// chem-perceive-10 ac-001 — the estimator's two tables are generated Rust too.
///
/// `gaff_equiv.json` (6159 lines) and `gaff_empirical.json` (87) are the last two
/// parameter tables molrs keeps as text: parmchk2's equivalence / substitution rows
/// with their penalty weights and defaults, and the Badger / Wang2004 empirical
/// constants. Every other table in this directory was compiled years of bugs ago;
/// these two are still `include_str!` + `serde_json::from_str` at runtime
/// (`ff/typifier/estimate/tables.rs`), and ac-001 finishes the job.
///
/// The gate is deliberately about the FILES, not about a Rust symbol: it has to be
/// able to fail before the tables exist. What it pins is that they arrive the same
/// way every other table did — emitted into `generated/`, hashed into
/// `MANIFEST.sha256`, and byte-reproduced by the generator (which the two guards
/// above then enforce on every run, without needing to know these two by name).
#[test]
fn gaff_equiv_and_empirical_are_generated_tables() {
    let dir = generated_dir();
    let manifest = std::fs::read_to_string(dir.join("MANIFEST.sha256"))
        .expect("MANIFEST.sha256 is committed alongside the generated tables");

    let mut missing = Vec::new();
    for table in ["gaff_equiv.rs", "gaff_empirical.rs"] {
        if !dir.join(table).is_file() {
            missing.push(format!("{table}: not emitted into `generated/`"));
        } else if !manifest.contains(table) {
            missing.push(format!(
                "{table}: emitted, but MANIFEST.sha256 does not hash it"
            ));
        }
    }

    assert!(
        missing.is_empty(),
        "the estimator's tables are still runtime-parsed JSON, not compiled data:\n  {}\n\
         Emit them from `scripts/gen_param_tables.py` (PARMCHK.DAT is already one of its \
         hashed sources) and delete the `include_str!` in \
         `src/ff/typifier/estimate/tables.rs`.",
        missing.join("\n  ")
    );
}

/// Every script in `scripts/` must at least parse.
///
/// `gen_am1bcc_oracle.py` sat with an `IndentationError` for several commits: the
/// oracle drift guard covers `gen_param_tables.py`, and nothing else ever ran the
/// other scripts, so a generator that could not start looked exactly like one that
/// was simply not needed.
#[test]
fn every_generator_script_compiles() {
    let scripts = repo_root().join("scripts");
    let python = std::env::var("MOLRS_PYTHON").unwrap_or_else(|_| "python3".to_owned());
    let mut broken = Vec::new();
    for entry in std::fs::read_dir(&scripts).expect("read scripts/") {
        let path = entry.expect("dir entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("py") {
            continue;
        }
        let out = Command::new(&python)
            .arg("-m")
            .arg("py_compile")
            .arg(&path)
            .output()
            .unwrap_or_else(|e| panic!("could not run `{python} -m py_compile`: {e}"));
        if !out.status.success() {
            broken.push(format!(
                "{}: {}",
                path.file_name().expect("file name").to_string_lossy(),
                String::from_utf8_lossy(&out.stderr).trim().to_owned()
            ));
        }
    }
    assert!(
        broken.is_empty(),
        "scripts that do not compile:\n{}",
        broken.join("\n")
    );
}
