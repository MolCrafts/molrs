//! ac-005 / ac-006 / ac-007 — the deletion, and the reverse protection.
//!
//! `mmff-orthogonal-01` proved the generic path (typifier → ForceField →
//! KernelRegistry) reproduces RDKit on 11/11 fixtures, term by term. That makes
//! the bespoke path a *provably redundant second implementation*, and this spec
//! removes it. These are the machine-checkable forms of that removal.
//!
//! Three criteria, and the third one points the other way:
//!
//! * **ac-005** — 4,065 dead type-def rows and their five readers are gone.
//! * **ac-006** — the bespoke energy assembly and the wrong classifiers are gone
//!   repo-wide, including the Python door (a Rust cleanup that leaves the broken
//!   door open in Python has cleaned nothing).
//! * **ac-007** — **`ff/mmff/energy/params.rs` SURVIVES**, out of `energy/`.
//!
//! ac-007 is the one that matters most, because it is the one a deletion list
//! gets wrong. That 800-odd-line file lived under `energy/` but it is not an
//! energy file — it is the **resolver**: the RDKit-faithful `bond_type` /
//! `angle_type` / `torsion_type` with four-level equivalence degradation and the
//! empirical rules that invent parameters from covalent radii when the table
//! misses. `frame_builder.rs` imports it as `eparams`; it is the one correct
//! implementation of MMFF's context rules in the tree. A deletion that swallows
//! it destroys exactly what the deletion was supposed to consolidate onto.
//!
//! # Where the subject moved (`chem-perceive-14-all-tables`)
//!
//! Two of these gates named artefacts that spec 14 has since renamed or deleted.
//! They are **retargeted, not relaxed** — the property each was guarding is the
//! same property, one hop away:
//!
//! * The resolver is now **`ff/mmff/resolve.rs`**. Those lines are an ALGORITHM,
//!   and `params.rs` was the wrong name for them (`ff/params/` is the home of
//!   *tables*, and the resolver holds none). ac-007 follows the file: it must
//!   still exist, still be the ONLY implementation of the three context rules
//!   anywhere under `src/ff/`, and still not be filed under `ff/params/`.
//! * The four MMFF XML copies are **deleted**; the shipped parameter set is the
//!   compiled table [`molrs::ff::params::mmff`], assembled into a `ForceField` by
//!   `ff/typifier/mmff/embedded.rs`. The row census that used to count XML tags
//!   now counts **type-def rows in the ForceField skeleton** — which is what those
//!   rows always *were*. Deleting the census along with its file would have left
//!   the property unguarded on the one path that ships.
//!
//! # Companion gate
//!
//! `bespoke_gate.rs` pins the SHA-256 of every file under `src/ff/mmff/` and
//! exists to stop spec 01 from touching them. It is this file's mirror image, and
//! the two cannot both be green: the deletion changes that tree. Spec 01's gate
//! says so itself — "the deletion of the bespoke path belongs to
//! mmff-orthogonal-02, which removes this gate along with the tree it guards".
//! **`bespoke_gate.rs` and `BESPOKE.sha256` are deleted by the GREEN commit.**
//!
//! # Scanner bias
//!
//! Line comments are stripped from `.rs` (`//`) and `.py` / `.pyi` (`#`) before
//! scanning; Markdown is scanned raw. So prose *about* a deleted symbol survives
//! (a comment reading "MmffForceField was removed in 0.7" is not a regression)
//! while any surviving *code* reference is a failure. Markdown is raw because a
//! doc page that still documents the deleted API is precisely the "broken door
//! left open" this criterion exists to close. `CHANGELOG.md` is excluded — it
//! must name what it removed.
//!
//! The dead-symbol needles below are assembled with `concat!` so that this file's
//! own source does not literally contain them. That is not a trick to be clever
//! with: it means the gate scans its own directory honestly, with **no
//! self-exemption by path**. A gate that skips a file is a gate someone can move
//! code into.

use std::path::{Path, PathBuf};

use molrs::ff::params::mmff::MMFF_VDW;
use molrs::ff::potential::{ParamSource, lookup_param_source};
use molrs::ff::typifier::mmff::{MMFF94STypifier, MMFF94Typifier};

fn repo_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("molrs/ has a parent")
}

fn src_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

// ---------------------------------------------------------------------------
// Scanning
// ---------------------------------------------------------------------------

/// Strip line comments introduced by `marker`.
fn strip_line_comments(src: &str, marker: &str) -> String {
    src.lines()
        .map(|line| match line.find(marker) {
            Some(at) => &line[..at],
            None => line,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The scannable text of a file, or `None` if it is not a source/doc file.
fn scannable(path: &Path) -> Option<String> {
    let text = std::fs::read_to_string(path).ok()?;
    match path.extension().and_then(|e| e.to_str())? {
        "rs" => Some(strip_line_comments(&text, "//")),
        "py" | "pyi" => Some(strip_line_comments(&text, "#")),
        "md" => Some(text),
        _ => None,
    }
}

/// Every scannable file under `root`, recursively.
fn scan_tree(root: &Path, out: &mut Vec<(PathBuf, String)>) {
    if root.is_file() {
        if let Some(text) = scannable(root) {
            out.push((root.to_owned(), text));
        }
        return;
    }
    if !root.is_dir() {
        return;
    }
    for entry in std::fs::read_dir(root).expect("read a scan root") {
        let path = entry.expect("dir entry").path();
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        // Build output and the CHANGELOG (which must name what it removed).
        if name == "target" || name == "CHANGELOG.md" || name.starts_with('.') {
            continue;
        }
        if path.is_dir() {
            scan_tree(&path, out);
        } else if let Some(text) = scannable(&path) {
            out.push((path, text));
        }
    }
}

/// Every place a consumer could still reach the deleted API from.
///
/// Both crates, both languages, plus the docs. `molrs-python` lives in the same
/// git repository as `molrs`, so deleting a public Rust API without deleting its
/// binder does not even compile — but the `.pyi`, `__init__.py`, the examples and
/// the docs site are text, and text does not fail to compile. It just lies.
fn deletion_scan_roots() -> Vec<PathBuf> {
    let root = repo_root();
    [
        "molrs/src",
        "molrs/tests",
        "molrs/examples",
        "molrs-python/src",
        "molrs-python/python",
        "molrs-python/tests",
        "molrs-python/examples",
        "molrs-python/site-src",
        "docs",
        "README.md",
    ]
    .iter()
    .map(|r| root.join(r))
    .filter(|p| p.exists())
    .collect()
}

fn deletion_scan() -> Vec<(PathBuf, String)> {
    let mut out = Vec::new();
    for root in deletion_scan_roots() {
        scan_tree(&root, &mut out);
    }
    assert!(
        out.len() > 100,
        "the deletion scan found only {} files — a scan that reads nothing reports no \
         violations",
        out.len()
    );
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// `path:line: text` for every occurrence of `needle` in the scanned tree.
fn hits(sources: &[(PathBuf, String)], needle: &str) -> Vec<String> {
    let root = repo_root();
    sources
        .iter()
        .flat_map(|(path, text)| {
            let rel = path
                .strip_prefix(root)
                .unwrap_or(path)
                .display()
                .to_string();
            text.lines()
                .enumerate()
                .filter(|(_, line)| line.contains(needle))
                .map(|(n, line)| format!("{rel}:{}: {}", n + 1, line.trim()))
                .collect::<Vec<_>>()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// ac-005 — the 4,065 dead rows and their five readers
//
// The rows used to be XML tags (`<Bond >`, `<Angle >`, `<StretchBend >`,
// `<Torsion >`, `<Oop >`) in four copies of a parameter file. `chem-perceive-14`
// deleted the XML and compiled the parameter set into `ff/params/mmff.rs`, so the
// tag count cannot be taken any more — but the *rows* were never really tags. They
// were **type-def rows in the ForceField skeleton**: `Style::defs`, the thing
// `Style::to_potential` hands a kernel as `tp`. That is where the census belongs
// now, and it is a strictly better place to take it — it sees every route by which
// a dead row could re-enter the tree, not just the one that went through text.
// ---------------------------------------------------------------------------

/// The five styles whose type-def rows were the 4,065 dead XML rows.
///
/// Each is `ParamSource::PerInstance`: its kernel reads the columns the typifier
/// baked into the Frame (`kb`/`r0`, `ka`/`theta0`, `kba_*`, `v1`/`v2`/`v3`,
/// `koop`) and ignores `tp` entirely. MMFF's context rules — aromaticity, ring
/// size, equivalence degradation, empirical fallbacks — are not a
/// `(type_i, type_j, …) → params` table and cannot be made into one, which is why
/// the rows were dead on the day they were written.
const DEAD_TYPE_ROW_STYLES: [(&str, &str); 5] = [
    ("bond", "mmff_bond"),
    ("angle", "mmff_angle"),
    ("angle", "mmff_stbn"),
    ("dihedral", "mmff_torsion"),
    ("improper", "mmff_oop"),
];

/// vdW is the one MMFF style that genuinely IS a per-atom-type table: 95 types,
/// 95 rows, and `mmff_vdw_ctor` opens by indexing `tp`.
const VDW_TYPE_ROWS: usize = 95;

/// The dead type-def rows are gone from the ForceField tree; the 95 live ones are not.
///
/// Both halves matter and they pull in opposite directions, which is the point:
/// over-reach is as much a failure as under-reach. A deletion that took the vdW
/// rows with the rest would leave `mmff_vdw_ctor` — a `TypeRows` kernel — with
/// nothing to resolve from, and `Style::to_potential` would refuse to build it.
#[test]
fn the_dead_type_rows_are_gone_and_the_live_ones_are_not() {
    let mut fails = Vec::new();

    // The data half: the 95 vdW rows still exist as data at all.
    if MMFF_VDW.len() != VDW_TYPE_ROWS {
        fails.push(format!(
            "`ff::params::mmff::MMFF_VDW` has {} rows, want {VDW_TYPE_ROWS} — this is REAL data \
             (the per-type alpha / n_eff / a_i / g_i / DA table) and no part of the deletion \
             touches it",
            MMFF_VDW.len()
        ));
    }

    // The tree half: what the shipped ForceField actually carries, per front door.
    for (label, ff) in [
        ("MMFF94", MMFF94Typifier::new().ff().clone()),
        ("MMFF94s", MMFF94STypifier::new().ff().clone()),
    ] {
        for (category, name) in DEAD_TYPE_ROW_STYLES {
            let Some(style) = ff.get_style(category, name) else {
                fails.push(format!(
                    "{label}: no `{category}/{name}` style at all — the row census below would be \
                     vacuous, and the term would not be computed by any consumer of the public API"
                ));
                continue;
            };
            let rows = style.defs.collect_type_params().len();
            if rows != 0 {
                fails.push(format!(
                    "{label}: `{category}/{name}` carries {rows} type-def rows, want 0 — no code \
                     reads them. They existed only to satisfy `Style::to_potential`'s \
                     empty-type-params guard, which `ParamSource::PerInstance` now answers honestly"
                ));
            }
            let source = lookup_param_source(category, name);
            if source != Some(ParamSource::PerInstance) {
                fails.push(format!(
                    "{label}: `{category}/{name}` is registered as {source:?}, want \
                     `PerInstance` — a `TypeRows` registration is exactly the escape hatch that \
                     let MMFF declare itself table-driven and then be fed 4,065 rows of type-def \
                     no kernel reads. The zero-row assertion above depends on this: with \
                     `TypeRows`, zero rows is an ERROR, and the rows would have to come back."
                ));
            }
        }

        match ff.get_style("pair", "mmff_vdw") {
            None => fails.push(format!(
                "{label}: no `pair/mmff_vdw` style — the deletion over-reached and took the one \
                 MMFF style that IS a type table"
            )),
            Some(style) => {
                let rows = style.defs.collect_type_params().len();
                if rows != VDW_TYPE_ROWS {
                    fails.push(format!(
                        "{label}: `pair/mmff_vdw` carries {rows} type-def rows, want \
                         {VDW_TYPE_ROWS} — one per MMFF atom type. `mmff_vdw_ctor` resolves its \
                         parameters FROM these rows; they are not the dead kind"
                    ));
                }
                let source = lookup_param_source("pair", "mmff_vdw");
                if source != Some(ParamSource::TypeRows) {
                    fails.push(format!(
                        "{label}: `pair/mmff_vdw` is registered as {source:?}, want `TypeRows` — \
                         if it ever became `PerInstance`, the 95 rows above could be deleted \
                         without any test noticing"
                    ));
                }
            }
        }
    }

    assert!(
        fails.is_empty(),
        "MMFF type-def row census failed:\n  {}",
        fails.join("\n  ")
    );
}

/// The five readers that parsed those rows.
///
/// Deleting the data while leaving the reader behind leaves a loaded gun: the
/// next regeneration of the XML, or a user-supplied parameter file, walks the
/// rows straight back into a `TypeRows` style.
#[test]
fn the_five_dead_xml_readers_are_gone() {
    let path = src_dir().join("ff/forcefield/xml.rs");
    let text = strip_line_comments(
        &std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read xml.rs: {e}")),
        "//",
    );

    let dead = [
        "parse_mmff_bonds",
        "parse_mmff_angles",
        "parse_mmff_stbn",
        "parse_mmff_torsions",
        "parse_mmff_oop",
    ];
    let found: Vec<&str> = dead.iter().copied().filter(|n| text.contains(n)).collect();
    assert!(
        found.is_empty(),
        "`ff/forcefield/xml.rs` still declares/dispatches the dead MMFF readers: {found:?}. \
         The rows they parse are gone; a reader without data is the shape that lets the data \
         come back."
    );

    // The reverse: the two live readers must NOT have been deleted with them.
    for live in ["parse_mmff_vdw", "parse_mmff_ele"] {
        assert!(
            text.contains(live),
            "`{live}` is gone from xml.rs — vdW is a real 95-row type table and the \
             electrostatic section is what mmff-orthogonal-01 added to close the 150 kcal/mol \
             hole. Neither is dead data."
        );
    }
}

/// The generator must stop emitting the five dead sections.
///
/// `scripts/mmff_to_xml.py` is what produced the 4,065 rows from RDKit's
/// `Params.cpp`. Deleting its output without deleting its emitters means the next
/// regeneration invites the dead data straight back in — and the next
/// regeneration is `chem-perceive-14-all-tables`, which compiles the XML into
/// committed, test-protected Rust tables.
#[test]
fn the_xml_generator_no_longer_emits_the_dead_sections() {
    let path = repo_root().join("scripts/mmff_to_xml.py");
    let text = strip_line_comments(
        &std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read mmff_to_xml.py: {e}")),
        "#",
    );

    let dead_sections = [
        "BondStretchParams",
        "AngleBendParams",
        "StretchBendParams",
        "TorsionParams",
        "OutOfPlaneParams",
    ];
    let found: Vec<&str> = dead_sections
        .iter()
        .copied()
        .filter(|s| text.contains(s))
        .collect();
    assert!(
        found.is_empty(),
        "`scripts/mmff_to_xml.py` still emits the dead MMFF sections: {found:?}. Regenerating \
         the XML would restore all 4,065 rows."
    );

    for live in ["VdWParams", "ElectrostaticParams"] {
        assert!(
            text.contains(live),
            "`scripts/mmff_to_xml.py` no longer emits `{live}` — a regenerated XML would lose \
             {}",
            if live == "VdWParams" {
                "the 95 real vdW type rows"
            } else {
                "the electrostatic section, reopening the 150 kcal/mol hole"
            }
        );
    }
}

// ---------------------------------------------------------------------------
// ac-006 — the bespoke path and the wrong classifiers, repo-wide
// ---------------------------------------------------------------------------

/// The bespoke energy-assembly symbols.
///
/// Spelled through `concat!` so this file's own source does not contain them —
/// see the module docs. The gate scans its own directory like any other.
const DEAD_SYMBOLS: [&str; 3] = [
    concat!("Mmff", "ForceField"),
    concat!("Mmff", "EnergyBreakdown"),
    concat!("build_mmff", "_potentials"),
];

#[test]
fn the_bespoke_energy_symbols_are_gone_repo_wide() {
    let sources = deletion_scan();
    let mut fails = Vec::new();
    for symbol in DEAD_SYMBOLS {
        let found = hits(&sources, symbol);
        if !found.is_empty() {
            fails.push(format!(
                "`{symbol}` ({} hits):\n    {}",
                found.len(),
                found.join("\n    ")
            ));
        }
    }
    assert!(
        fails.is_empty(),
        "the bespoke MMFF energy path is still reachable. mmff-orthogonal-01 proved the generic \
         path reproduces RDKit on 11/11 fixtures term-by-term, so these are a redundant second \
         implementation — and a second implementation of a force field is a second set of \
         numbers to be wrong.\n\n{}\n\n(CHANGELOG.md is excluded: it must name what it removed.)",
        fails.join("\n\n")
    );
}

#[test]
fn the_bespoke_energy_module_tree_is_gone() {
    let dir = src_dir().join("ff/mmff/energy");
    assert!(
        !dir.exists(),
        "{} still exists. The energy assembly layer (bond/angle/stretchbend/torsion/oop/\
         nonbonded/geom) is the redundant second implementation.\n\
         NOTE: `energy/params.rs` is NOT part of it — it is the resolver and must be MOVED to \
         `ff/mmff/resolve.rs`, not deleted (ac-007).",
        dir.display()
    );
}

#[test]
fn the_wrong_classifier_module_is_gone() {
    let path = src_dir().join("ff/typifier/mmff/classify.rs");
    assert!(
        !path.exists(),
        "{} still exists. Its three classifiers reimplement — incorrectly — the context rules \
         that `ff/mmff/resolve.rs` already implements correctly:\n\
           * `typify_bond`: aromatic bond -> type 1. RDKit says 0: after MMFF aromaticity \
             perception an aromatic bond is AROMATIC, never SINGLE, and `getMMFFBondType` \
             requires SINGLE. Backwards.\n\
           * `typify_angle(bt_ij, bt_jk)`: the SIGNATURE cannot express the rule. RDKit's \
             `getMMFFAngleType` needs topology — 3-/4-membered-ring membership promotes the \
             angle type to 3..8, and two bond types cannot say whether the angle is in a ring.\n\
           * `typify_dihedral`: same disease; no 4-/5-ring torsion types.",
        path.display()
    );
}

/// No `build` / `typify_bond` / `typify_angle` / `typify_dihedral` on the MMFF
/// front doors.
///
/// Scoped to `src/ff/typifier/mmff/` on purpose. `OPLSAATypifier::build` is
/// explicitly **out of scope** for this spec (the owner's ruling named the three
/// MMFF doors), so a gate scoped at the whole typifier tree would fail on code
/// that is deliberately being left alone.
///
/// The needles carry their opening paren so that `typify_bonded_topology(` — a
/// real function `frame_builder` calls, whose name contains `typify_bond` — is
/// not mistaken for the deleted classifier.
#[test]
fn the_mmff_front_doors_expose_no_build_or_classify_methods() {
    let dir = src_dir().join("ff/typifier/mmff");
    let mut sources = Vec::new();
    scan_tree(&dir, &mut sources);
    assert!(
        !sources.is_empty(),
        "no sources under {} — the gate would pass vacuously",
        dir.display()
    );

    let mut fails = Vec::new();
    for needle in [
        "fn build(",
        "typify_bond(",
        "typify_angle(",
        "typify_dihedral(",
    ] {
        let found = hits(&sources, needle);
        if !found.is_empty() {
            fails.push(format!("`{needle}`:\n    {}", found.join("\n    ")));
        }
    }
    assert!(
        fails.is_empty(),
        "the MMFF front doors still expose the deleted API.\n\n{}\n\n\
         `build` is the convenience that made MMFF a special case; the three classifiers are \
         wrong. The surviving contract is `typify(mol) -> Atomistic` — labels and charges — \
         after which MMFF walks the same ForceField route as every other force field.",
        fails.join("\n\n")
    );
}

// ---------------------------------------------------------------------------
// ac-007 — the REVERSE protection: the resolver survives
// ---------------------------------------------------------------------------

/// The three context rules. Declaring one of these IS being the resolver.
///
/// The leading `fn ` and the trailing `(` are load-bearing: `mmff_bond_type(` (a
/// real function in `ff/mmff/charges.rs`, which asks the resolver rather than
/// reimplementing it) contains `bond_type(` and must not be mistaken for a second
/// declaration.
const RESOLVER_RULES: [&str; 3] = ["fn bond_type(", "fn angle_type(", "fn torsion_type("];

/// A `src/`-relative path, for messages and for comparing "which file declares it".
fn rel_to_src(path: &Path) -> String {
    path.strip_prefix(src_dir())
        .unwrap_or(path)
        .display()
        .to_string()
}

/// The resolver, wherever it is filed, is the ONE implementation of the context rules.
///
/// It is the RDKit-faithful resolver: `bond_type` / `angle_type` / `torsion_type`
/// (with the ring rules the deleted classifiers could not express), four-level
/// equivalence degradation, and the empirical rules that invent parameters from
/// covalent radii on a table miss. Every number the MMFF kernels read — `kb`,
/// `r0`, `ka`, `theta0`, `kba_*`, `v1/v2/v3`, `koop` — comes from here, through
/// `frame_builder`.
///
/// # Its name, twice corrected
///
/// It was `ff/mmff/energy/params.rs` and spec 02 moved it to `ff/mmff/params.rs`.
/// `chem-perceive-14` renamed it again, to **`ff/mmff/resolve.rs`**: those lines
/// are an ALGORITHM, `ff/params/` is the home of *tables*, and the file holds
/// none — calling it `params.rs` was the same misnaming in the other direction.
/// So this gate no longer pins one path. It pins the property the path was only
/// ever a proxy for:
///
/// 1. the resolver exists, at `ff/mmff/resolve.rs`, and is not a stub;
/// 2. it declares the three context rules;
/// 3. **nothing else under `src/ff/` declares them** — not `classify.rs`, not a
///    fresh copy under `ff/params/`, not anywhere. That is the half a rename can
///    silently break, and the half that made ac-007 the reverse-protection
///    criterion in the first place;
/// 4. and neither of the two names it used to have is back, which is what
///    distinguishes a move from a copy.
///
/// The line-count floor is the anti-gutting clause. "Renamed" must mean renamed,
/// not re-derived into a stub that happens to have the right file name.
#[test]
fn the_rdkit_faithful_resolver_survives_outside_energy() {
    let resolver_rel = "ff/mmff/resolve.rs";
    let resolver = src_dir().join(resolver_rel);
    assert!(
        resolver.is_file(),
        "{} does not exist. The ~810-line resolver lived under `energy/` but it is NOT an energy \
         file, and it is not a parameter table either — it is the one correct implementation of \
         MMFF's context rules in this tree, and the typifier imports it. A deletion list that \
         swallowed `energy/params.rs` along with `energy/` destroyed exactly the thing this \
         consolidation is consolidating ONTO; a rename that loses it does the same damage more \
         quietly.",
        resolver.display()
    );

    let text = std::fs::read_to_string(&resolver).expect("read the resolver");
    let lines = text.lines().count();
    assert!(
        lines >= 700,
        "{} has only {lines} lines. The resolver is ~810; a file this short is a stub, not a \
         rename. The equivalence degradation and the empirical rules ARE the file.",
        resolver.display()
    );

    // Neither name it used to have may be back: a copy is not a move.
    for old in ["ff/mmff/energy/params.rs", "ff/mmff/params.rs"] {
        let path = src_dir().join(old);
        assert!(
            !path.exists(),
            "{} still exists — the resolver was copied, not moved. Two copies of the context \
             rules is the defect this spec is deleting, reintroduced by its own fix.",
            path.display()
        );
    }

    // The RDKit-faithful trio must actually be in there...
    for needle in RESOLVER_RULES {
        assert!(
            text.contains(needle),
            "the resolver does not declare `{needle}` — these three ARE the context rules (and \
             the reason the `classify.rs` versions were wrong)"
        );
    }

    // ...and nowhere else under `src/ff/`, including `ff/params/`.
    let mut sources = Vec::new();
    scan_tree(&src_dir().join("ff"), &mut sources);
    assert!(
        !sources.is_empty(),
        "no sources under src/ff — vacuous gate"
    );

    let mut fails = Vec::new();
    for needle in RESOLVER_RULES {
        let declared_in: Vec<String> = sources
            .iter()
            .filter(|(_, text)| text.contains(needle))
            .map(|(path, _)| rel_to_src(path))
            .collect();
        if declared_in != [resolver_rel] {
            fails.push(format!(
                "`{needle}` is declared in {declared_in:?}, want exactly [\"{resolver_rel}\"]"
            ));
        }
    }
    assert!(
        fails.is_empty(),
        "MMFF's context rules have more than one implementation under `src/ff/`:\n  {}\n\n\
         There is exactly ONE — the RDKit-faithful resolver, validated against RDKit on 11/11 \
         fixtures. A second one is not a second opinion; it is a second set of numbers to be \
         wrong, and the last one (`classify.rs`) got benzene's aromatic bonds backwards and \
         could not express a ring-membership angle type at all. `ff/params/` is not a home for \
         it either: that directory holds TABLES, and these are rules.",
        fails.join("\n  ")
    );
}

/// The resolver's module path. Its FULL crate path, and that is deliberate.
///
/// The old needle was the bare `"mmff::params"`, and it had a hole: a file could
/// satisfy it by importing `crate::ff::typifier::mmff::params::MMFFParams` — a
/// real, unrelated module (the 95 nine-column typing rows) whose path also spells
/// `mmff::params`. Grepping for the full path closes that, and the call check
/// below closes it again from the other side: an import that is not *used* for the
/// labels no longer passes.
const RESOLVER_MODULE: &str = "crate::ff::mmff::resolve";

/// `frame_builder` derives its labels from the resolver, and from nothing else.
///
/// After `classify.rs` is gone, the bond / angle / dihedral type labels must come
/// from the one correct implementation. The labels are only provenance — the
/// per-instance kernels read Frame columns, not labels — but "only provenance" is
/// not a licence to be wrong, and they HAVE been wrong (see
/// `tests/ff/typifier/mmff_labels.rs`, which asserts the values).
///
/// Three things, and the middle one is the one with teeth:
///
/// 1. the resolver is imported, by its full crate path;
/// 2. the three context rules are **called through whatever name that import
///    binds** — so the import cannot be a decoration that satisfies a grep while
///    the labels come from somewhere else;
/// 3. no `classify`, and no resurrected `mmff::energy`.
#[test]
fn frame_builder_imports_the_resolver_and_no_second_classifier() {
    let path = src_dir().join("ff/typifier/mmff/frame_builder.rs");
    let text = strip_line_comments(
        &std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read frame_builder.rs: {e}")),
        "//",
    );

    let use_line = text
        .lines()
        .find(|line| line.contains("use ") && line.contains(RESOLVER_MODULE))
        .unwrap_or_else(|| {
            panic!(
                "`frame_builder.rs` does not import `{RESOLVER_MODULE}` — the resolver is where \
                 every bond / angle / dihedral type label must come from"
            )
        });

    // What the import binds the resolver's items to: `use … resolve as eparams;`
    // -> `eparams::`, `use … resolve;` -> `resolve::`, `use … resolve::{…};` -> the
    // items are in scope bare. Whichever spelling, the three rules must be CALLED.
    let tail = use_line
        .split_once(RESOLVER_MODULE)
        .map(|(_, tail)| tail)
        .unwrap_or_default();
    let prefix = if tail.trim_start().starts_with("::") {
        // `use …::resolve::{bond_type, …};` — the items are in scope bare.
        String::new()
    } else if let Some((_, alias)) = tail.split_once(" as ") {
        // `use …::resolve as eparams;`
        format!("{}::", alias.trim().trim_end_matches(';').trim())
    } else {
        // `use …::resolve;`
        "resolve::".to_owned()
    };

    let mut missing = Vec::new();
    for rule in ["bond_type(", "angle_type(", "torsion_type("] {
        let call = format!("{prefix}{rule}");
        if !text.contains(&call) {
            missing.push(call);
        }
    }
    assert!(
        missing.is_empty(),
        "`frame_builder.rs` imports the resolver (`{}`) but never calls {missing:?} through it. \
         An import is not a derivation: the bond / angle / dihedral type codes must come from the \
         RDKit-faithful rules, which are the only ones in the tree that can see aromaticity and \
         ring membership.",
        use_line.trim()
    );

    assert!(
        !text.contains("mmff::energy"),
        "`frame_builder.rs` still imports from `ff::mmff::energy`, which no longer exists as an \
         energy module. The resolver lives at `{RESOLVER_MODULE}`."
    );
    assert!(
        !text.contains("classify"),
        "`frame_builder.rs` still references `classify` — the second (wrong) classifier is \
         deleted, and its labels are re-derived from the resolver"
    );
}

/// No second classifier anywhere under `src/ff/`.
///
/// Wider than the typifier tree, because "delete the duplicate" means the
/// duplicate does not reappear one directory over. The five names below are the
/// complete contents of `classify.rs`: three wrong classifiers and two
/// equivalence-degradation loops that existed to key type-row lookups which no
/// longer exist.
#[test]
fn no_second_classifier_survives_anywhere_in_ff() {
    let mut sources = Vec::new();
    scan_tree(&src_dir().join("ff"), &mut sources);
    assert!(
        !sources.is_empty(),
        "no sources under src/ff — vacuous gate"
    );

    let mut fails = Vec::new();
    for needle in [
        "typify_bond(",
        "typify_angle(",
        "typify_dihedral(",
        "resolve_angle_label(",
        "resolve_oop_label(",
    ] {
        let found = hits(&sources, needle);
        if !found.is_empty() {
            fails.push(format!("`{needle}`:\n    {}", found.join("\n    ")));
        }
    }
    assert!(
        fails.is_empty(),
        "a second MMFF classifier still exists under src/ff/:\n\n{}\n\n\
         There is exactly one correct implementation of MMFF's bond / angle / torsion type \
         rules (`ff/mmff/resolve.rs`, validated against RDKit on 11/11 fixtures). Every label in \
         the tree derives from it, or it is not the only one.",
        fails.join("\n\n")
    );
}
