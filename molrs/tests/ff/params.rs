//! Antechamber parameter tables (`ff::params`).
//!
//! Two jobs:
//!
//! 1. **Manifest guard** — `MANIFEST.sha256` records the SHA-256 of every emitted
//!    `.rs`. This test recomputes them and needs **no AmberTools**, so it actually
//!    runs in CI. It is what stops a hand-edit of the ~50k committed table lines
//!    from going unnoticed, and after chem-perceive-14 it is the **only** standing
//!    drift guard over them.
//! 2. **Column semantics** — the tables carry meaning, not just numbers. The
//!    `GASPARM` `d` column in particular is chi+, a normalisation denominator,
//!    NOT a quartic coefficient, and it must stay a distinct named field.
//!
//! # The AmberTools drift guard is gone, and that is the point
//!
//! There used to be a third job here: with the AmberTools install-root environment
//! variable set, re-run `scripts/gen_param_tables.py` and require it to byte-reproduce
//! every committed `.rs`. It opened with `if env::var_os(…).is_none() { return; }` — so
//! on every CI run, and on every contributor machine without AmberTools, it printed a
//! skip line and passed. **It never once ran in CI.**
//!
//! chem-perceive-14 ac-003 deletes it outright (owner's ruling: CI must have zero
//! entanglement with the AmberTools install root; byte-reproduction is verified once,
//! locally, during implementation, and the result recorded in the spec body). Deleted,
//! not skipped: a test that skips itself where it runs buys the appearance of coverage
//! and none of the substance, and it crowded out the guard that does work — the
//! manifest hash check above, which needs nothing installed and catches the failure
//! that actually happens.
//!
//! `tables_gate::no_test_couples_ci_to_ambertools` is the grep that keeps it deleted.

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

/// The one place every parameter table lives — flat, no `generated/` subdirectory
/// (chem-perceive-14 ac-001).
fn params_dir() -> PathBuf {
    repo_root().join("molrs/src/ff/params")
}

fn chi_plus(atom_type: &str) -> f64 {
    GASTEIGER_PARAMS
        .iter()
        .find(|row| row.atom_type == atom_type)
        .unwrap_or_else(|| panic!("no GASPARM row for `{atom_type}`"))
        .chi_plus
}

/// The marker every generator-emitted table carries in its header doc.
///
/// This — not a hardcoded file count — is what tells a generated table from a
/// hand-written one, and it is what makes [`committed_tables_match_the_manifest_hashes`]
/// two-sided without having to know any table by name.
const GENERATED_MARKER: &str = "regenerate with `scripts/gen_param_tables.py`";

/// Every committed table hashes to what `MANIFEST.sha256` recorded — and every
/// generated table is IN the manifest.
///
/// After chem-perceive-14 this is the **only** standing drift guard over ~50k lines of
/// committed force-field numbers: ac-003 deletes the byte-regeneration test that needed
/// an AmberTools install (and which never ran in CI anyway — see the module header), and
/// byte-reproduction is verified once, locally, by the implementer. Recomputing the
/// hashes needs nothing installed and catches the failure that actually happens: a
/// hand-edit to a generated file.
///
/// It is deliberately **two-sided**, because each side alone is trivially defeated:
///
/// * *forward* — every `emitted` row names a file that exists and still hashes to the
///   recorded digest. Catches a hand-edit to a table.
/// * *backward* — every `.rs` under `ff/params/` that declares itself generated has an
///   `emitted` row. Catches a table added to the tree, or converted from XML by this very
///   spec, without a manifest row — which would otherwise ship ~5,000 fresh numbers
///   guarded by nothing at all, while the forward pass stayed happily green.
///
/// A deliberate regeneration updates the tables AND the manifest together, so this stays
/// green; an edit to only one of them does not.
#[test]
fn committed_tables_match_the_manifest_hashes() {
    let dir = params_dir();
    let manifest = std::fs::read_to_string(dir.join("MANIFEST.sha256")).unwrap_or_else(|e| {
        panic!(
            "{}/MANIFEST.sha256 is not readable ({e}).\n\
             chem-perceive-14 ac-001 flattens `ff/params/generated/` into `ff/params/`, \
             and the manifest moves with the tables it hashes.",
            dir.display()
        )
    });

    // Forward: every manifest row is honoured.
    let mut listed = Vec::new();
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
             It was hand-edited, or regenerated without committing the manifest."
        );
        listed.push(name.to_owned());
    }

    // Backward: every generated table is in the manifest.
    let mut unhashed = Vec::new();
    for entry in std::fs::read_dir(&dir).expect("read ff/params/") {
        let path = entry.expect("dir entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let src = std::fs::read_to_string(&path).expect("read a parameter table");
        if !src.contains(GENERATED_MARKER) {
            continue; // hand-written: the module root, the moved RDKit port, the resolver
        }
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .expect("file name")
            .to_owned();
        if !listed.contains(&name) {
            unhashed.push(name);
        }
    }
    unhashed.sort();
    assert!(
        unhashed.is_empty(),
        "these tables declare themselves generated but MANIFEST.sha256 does not hash \
         them:\n  {}\n\
         An unhashed table is ~thousands of force-field numbers with no drift guard at \
         all — and the forward pass above cannot see it, because it only walks rows the \
         manifest already has. Emit a manifest row from `scripts/gen_param_tables.py`.",
        unhashed.join("\n  ")
    );

    // Non-vacuity: 14 AmberTools tables + `oplsaa`, the one XML set that becomes a table
    // of its own. (MMFF is ONE shared table merged with the moved RDKit port, so whether
    // `mmff.rs` carries a manifest row is the implementer's call — the backward pass
    // above settles it either way. And there is no `mmff94s.rs` at all: that XML is
    // deleted, not converted.)
    //
    // A floor, not an equality — whether the generator still emits its own `mod.rs` once
    // `generated/` is flattened away is likewise the implementer's call, and the backward
    // pass is what actually guarantees nothing escapes.
    assert!(
        listed.len() >= 15,
        "MANIFEST.sha256 hashes only {} emitted files, expected at least 15 (14 \
         AmberTools tables + oplsaa). Before chem-perceive-14 it hashes 15 and lives one \
         directory down, in `generated/`. Listed: {listed:?}",
        listed.len()
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
/// way every other table did — emitted into the one parameter directory, hashed into
/// `MANIFEST.sha256`, and byte-reproduced by the generator (which the guard above then
/// enforces on every run, without needing to know these two by name).
#[test]
fn gaff_equiv_and_empirical_are_generated_tables() {
    let dir = params_dir();
    let manifest = std::fs::read_to_string(dir.join("MANIFEST.sha256"))
        .expect("MANIFEST.sha256 is committed alongside the generated tables");

    let mut missing = Vec::new();
    for table in ["gaff_equiv.rs", "gaff_empirical.rs"] {
        if !dir.join(table).is_file() {
            missing.push(format!(
                "{table}: not present directly under `ff/params/` (chem-perceive-14 \
                 ac-001 flattens `generated/` away — the tables are first-class source)"
            ));
        } else if !manifest.contains(table) {
            missing.push(format!(
                "{table}: present, but MANIFEST.sha256 does not hash it"
            ));
        }
    }

    assert!(
        missing.is_empty(),
        "the estimator's tables are not where every other parameter table lives:\n  {}\n\
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
