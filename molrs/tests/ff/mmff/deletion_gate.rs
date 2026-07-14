//! ac-005 / ac-006 / ac-007 — the deletion, and the reverse protection.
//!
//! `mmff-orthogonal-01` proved the generic path (typifier → ForceField →
//! KernelRegistry) reproduces RDKit on 11/11 fixtures, term by term. That makes
//! the bespoke path a *provably redundant second implementation*, and this spec
//! removes it. These are the machine-checkable forms of that removal.
//!
//! Three criteria, and the third one points the other way:
//!
//! * **ac-005** — 4,065 dead XML rows and their five readers are gone.
//! * **ac-006** — the bespoke energy assembly and the wrong classifiers are gone
//!   repo-wide, including the Python door (a Rust cleanup that leaves the broken
//!   door open in Python has cleaned nothing).
//! * **ac-007** — **`ff/mmff/energy/params.rs` SURVIVES**, moved to
//!   `ff/mmff/params.rs`.
//!
//! ac-007 is the one that matters most, because it is the one a deletion list
//! gets wrong. That 826-line file lives under `energy/` but it is not an energy
//! file — it is the **parameter resolver**: the RDKit-faithful `bond_type` /
//! `angle_type` / `torsion_type` with four-level equivalence degradation and the
//! empirical rules that invent parameters from covalent radii when the table
//! misses. `frame_builder.rs` imports it as `eparams`; it is the one correct
//! implementation of MMFF's context rules in the tree. A deletion that swallows
//! it destroys exactly what the deletion was supposed to consolidate onto.
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
// ---------------------------------------------------------------------------

/// The five dead XML row tags, with the count each must have after the deletion.
///
/// The trailing space is load-bearing: it distinguishes the `<Bond ` *rows* from
/// the `<BondStretchParams>` section wrapper and the `<BondChargeIncrements>`
/// section, which are different elements and are not all being deleted.
const DEAD_XML_ROWS: [&str; 5] = ["<Bond ", "<Angle ", "<StretchBend ", "<Torsion ", "<Oop "];

/// The rows that must SURVIVE, and their exact counts.
///
/// `<VdW >` is a genuine per-atom-type table (95 types, 95 rows) that
/// `mmff_vdw_ctor` really does read. `<ElectrostaticParams>` is the style-level
/// section mmff-orthogonal-01 added — the one whose absence was the 150 kcal/mol
/// hole. Pinning both stops the deletion from over-reaching in either direction.
const SURVIVING_XML: [(&str, usize); 2] = [("<VdW ", 95), ("<ElectrostaticParams", 1)];

/// All four copies of the parameter set. A `data/` build and a `molrs/data/`
/// build must not disagree about what MMFF contains.
const MMFF_XMLS: [&str; 4] = [
    "molrs/data/mmff94.xml",
    "molrs/data/mmff94s.xml",
    "data/mmff94.xml",
    "data/mmff94s.xml",
];

#[test]
fn the_dead_xml_rows_are_gone_and_the_live_ones_are_not() {
    let mut fails = Vec::new();
    for rel in MMFF_XMLS {
        let path = repo_root().join(rel);
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));

        for tag in DEAD_XML_ROWS {
            let n = text.matches(tag).count();
            if n != 0 {
                fails.push(format!(
                    "{rel}: {n} `{tag}` rows, want 0 — no code reads them. They exist only to \
                     satisfy `Style::to_potential`'s empty-type-params guard, which \
                     `ParamSource::PerInstance` now answers honestly"
                ));
            }
        }
        for (tag, want) in SURVIVING_XML {
            let n = text.matches(tag).count();
            if n != want {
                fails.push(format!(
                    "{rel}: {n} `{tag}` elements, want {want} — this one is REAL data and must \
                     survive the deletion"
                ));
            }
        }
    }
    assert!(
        fails.is_empty(),
        "MMFF XML row census failed:\n  {}",
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
         NOTE: `energy/params.rs` is NOT part of it — it is the parameter resolver and must be \
         MOVED to `ff/mmff/params.rs`, not deleted (ac-007).",
        dir.display()
    );
}

#[test]
fn the_wrong_classifier_module_is_gone() {
    let path = src_dir().join("ff/typifier/mmff/classify.rs");
    assert!(
        !path.exists(),
        "{} still exists. Its three classifiers reimplement — incorrectly — the context rules \
         that `ff/mmff/params.rs` already implements correctly:\n\
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

/// `energy/params.rs` is not an energy file. It must survive, out of `energy/`.
///
/// It is the RDKit-faithful parameter resolver: `bond_type` / `angle_type` /
/// `torsion_type` (with the ring rules the deleted classifiers could not express),
/// four-level equivalence degradation, and the empirical rules that invent
/// parameters from covalent radii on a table miss. Every number the MMFF kernels
/// read — `kb`, `r0`, `ka`, `theta0`, `kba_*`, `v1/v2/v3`, `koop` — comes from
/// here, through `frame_builder`.
///
/// The line-count floor is the anti-gutting clause. "Moved" must mean moved, not
/// re-derived into a stub that happens to have the right file name.
#[test]
fn the_rdkit_faithful_resolver_survives_outside_energy() {
    let moved = src_dir().join("ff/mmff/params.rs");
    assert!(
        moved.is_file(),
        "{} does not exist. The 826-line resolver lived under `energy/` but it is NOT an energy \
         file — it is the one correct implementation of MMFF's context rules in this tree, and \
         the typifier imports it. A deletion list that swallowed `energy/params.rs` along with \
         `energy/` destroyed exactly the thing this consolidation is consolidating ONTO.",
        moved.display()
    );

    let text = std::fs::read_to_string(&moved).expect("read the moved resolver");
    let lines = text.lines().count();
    assert!(
        lines >= 700,
        "{} has only {lines} lines. The resolver is 826; a file this short is a stub, not a \
         move. The equivalence degradation and the empirical rules ARE the file.",
        moved.display()
    );

    let old = src_dir().join("ff/mmff/energy/params.rs");
    assert!(
        !old.exists(),
        "{} still exists — the resolver was copied, not moved. Two copies of the context rules \
         is the defect this spec is deleting, reintroduced by its own fix.",
        old.display()
    );

    // The RDKit-faithful trio must actually be in there.
    for needle in ["fn bond_type(", "fn angle_type(", "fn torsion_type("] {
        assert!(
            text.contains(needle),
            "the moved resolver does not declare `{needle}` — these three ARE the context rules \
             (and the reason the `classify.rs` versions were wrong)"
        );
    }
}

/// `frame_builder` derives its labels from the resolver, and from nothing else.
///
/// After `classify.rs` is gone, the bond / angle / dihedral type labels must come
/// from the one correct implementation. The labels are only provenance — the
/// per-instance kernels read Frame columns, not labels — but "only provenance" is
/// not a licence to be wrong, and today they ARE wrong (see
/// `tests/ff/typifier/mmff_labels.rs`, which asserts the values).
#[test]
fn frame_builder_imports_the_resolver_and_no_second_classifier() {
    let path = src_dir().join("ff/typifier/mmff/frame_builder.rs");
    let text = strip_line_comments(
        &std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read frame_builder.rs: {e}")),
        "//",
    );

    assert!(
        text.contains("mmff::params"),
        "`frame_builder.rs` does not import `crate::ff::mmff::params` — the resolver is where \
         every bond / angle / dihedral type label must now come from"
    );
    assert!(
        !text.contains("mmff::energy"),
        "`frame_builder.rs` still imports from `ff::mmff::energy`, which no longer exists as an \
         energy module. The resolver moved to `ff::mmff::params`."
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
         rules (`ff/mmff/params.rs`, validated against RDKit on 11/11 fixtures). Every label in \
         the tree derives from it, or it is not the only one.",
        fails.join("\n\n")
    );
}
