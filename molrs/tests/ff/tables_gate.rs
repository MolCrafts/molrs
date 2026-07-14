//! Structural gates for chem-perceive-14: **one place, one form**.
//!
//! Parameter data in molrs lives in four forms across three places — typed Rust under
//! `ff/params/generated/`, typed Rust in `ff/mmff/tables.rs`, raw XML in `data/*.xml`
//! re-parsed on every construction, and, until the owner deleted it, a dead Open Babel
//! fragment library in `data/gen3d/`. The owner's ruling collapses that to one:
//!
//! > "请你将所有的 data 统一成一个形式，放在统一的地方"
//! > "如果是原位的，那就不要叫 generated 这种傻逼的名字！"
//! > "`_tables` 这个名字不好，换个名字"
//!
//! | in the tree today | form | size |
//! |---|---|---|
//! | `ff/params/generated/` ×14 + `mod.rs` | committed typed Rust | 49,977 lines |
//! | `ff/mmff/tables.rs` | committed typed Rust (RDKit `Params.cpp` port, BSD-3) | 51,621 lines |
//! | `ff/mmff/params.rs` | the **resolver** — an algorithm, misnamed | 808 lines |
//! | `data/{mmff94,mmff94s,oplsaa}.xml` | raw XML, `include_str!` + runtime parse | 481.5 KB |
//!
//! The end state:
//!
//! ```text
//! ff/params/            <- EVERY parameter table, flat. No `generated/`, no `_tables`
//!     mod.rs            <- row types only
//!     gaff.rs  gaff2.rs  bccparm.rs  bccparm_abcg2.rs  gasparm.rs
//!     gaff_equiv.rs  gaff_empirical.rs
//!     atomtype_{gff,gff2,bcc,abcg2,amber,sybyl,gas}.rs
//!     oplsaa.rs         <- NEW (oplsaa.xml -> typed Rust)
//!     mmff.rs           <- ALL MMFF parameter DATA, merged:
//!                          (a) the RDKit Params.cpp port (17 statics + 17 accessors)
//!                          (b) the 199 entries left in mmff94.xml
//! ff/mmff/
//!     resolve.rs        <- the 808-line RESOLVER, renamed from params.rs. NOT a table.
//! ```
//!
//! Three naming rulings are baked into the gates below, and each has a reason:
//!
//! * **No `generated/`.** These tables are first-class source, not a build artefact.
//!   *"Generated" names how they arrived, not what they are* — and how they arrived
//!   belongs in each file's header doc, not in a directory name.
//! * **No `_tables` suffix under `ff/params/`.** Everything in there *is* a table; a
//!   `_tables` suffix says "table table".
//! * **`ff/mmff/params.rs` → `ff/mmff/resolve.rs`, and it does NOT move into
//!   `ff/params/`.** Those 808 lines are **not parameters, they are an algorithm**
//!   (`bond_type` / `angle_type` / `torsion_type`, four-level equivalence degradation,
//!   empirical rules). `params.rs` was a naming error in spec 02, corrected here.
//!
//! The gates here are on the **tree**, not on Rust symbols, and deliberately so — the
//! same choice `params::gaff_equiv_and_empirical_are_generated_tables` makes, for the
//! same reason: a gate has to be able to fail *before* the tables exist, and a test
//! that names a symbol which does not compile yet takes the whole `ff` target down
//! with it, so nothing else in the suite can run while the change is in flight.
//!
//! # The XMLs are not what they were when this spec was written
//!
//! The spec body's Summary says "1204 KB of XML" across three files. It is **481.5 KB**,
//! and after this spec it is **two** source files, not three:
//!
//! `mmff-orthogonal-02` deleted 4,065 type-def rows from each MMFF XML — rows no kernel
//! ever read, which existed only to satisfy an `is_empty()` guard — and
//! `mmff-orthogonal-01` added an `<ElectrostaticParams>` section. That left
//! `mmff94.xml` and `mmff94s.xml` differing by **exactly two lines**: the
//! `<ForceField name=…>` attribute. The real MMFF94-vs-MMFF94s delta (11 Oop + 42
//! Torsion rows) lives entirely in the RDKit port (`MMFF_OOP` vs `MMFF_OOP_S`,
//! `MMFF_TOR` vs `MMFF_TOR_S`), selected by `MmffVariant`.
//!
//! So `mmff94s.xml` is deleted and there is **ONE shared MMFF table**. The two front
//! doors differ by exactly two things — the ForceField **name string** and the
//! **`MmffVariant`** — which now lives in the type system instead of being duplicated
//! in data. Emitting two byte-identical 199-entry tables that differ only in a name
//! string would have been precisely the dead weight the owner has already rejected three
//! times.
//!
//! Note this is a **consequence** of deleting the dead rows, not a fact that was true
//! earlier: when this chain first claimed `mmff94s.xml` was redundant it was **wrong**,
//! and the owner was right to reject it — back then the XML really did carry the 11 + 42
//! differing rows and really did feed them to the ForceField tree.

use std::path::{Path, PathBuf};

/// Repository root (the workspace dir, one above this package).
fn repo_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("molrs/ has a parent")
}

fn src_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("data")
}

/// The one place every parameter table lives.
fn params_dir() -> PathBuf {
    src_dir().join("ff/params")
}

fn tests_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests")
}

/// Shorten a path for a failure message.
fn shown(path: &Path) -> String {
    path.strip_prefix(repo_root())
        .unwrap_or(path)
        .display()
        .to_string()
}

/// The three XML files that must leave the tree. All three are deleted; only **two** of
/// them are a table's source — `mmff94s.xml` differs from `mmff94.xml` by two lines (the
/// `<ForceField name=…>` attribute) and contributes nothing, so it is deleted outright
/// rather than converted.
const XML_FILES: [&str; 3] = ["mmff94", "mmff94s", "oplsaa"];

/// Every table that must sit directly under `ff/params/`, by module stem.
///
/// The first fourteen are today's `generated/` tables (AmberTools `.DAT` / `.DEF` and the
/// two parmchk2 JSON assets); `oplsaa` is the one XML set this spec converts to a table
/// of its own.
///
/// **`mmff` is deliberately absent from this list** and is checked separately by
/// [`the_rdkit_mmff_port_is_merged_into_ff_params`]: it is a *merge* of a moved file (the
/// RDKit port — not a generator output, so not necessarily hashed by `MANIFEST.sha256`)
/// with the 199 entries left in `mmff94.xml`. There is no `mmff94.rs` and no
/// `mmff94s.rs`: **one shared MMFF table**, two front doors that differ by a name string
/// and an `MmffVariant`.
const PARAM_TABLES: [&str; 15] = [
    "atomtype_abcg2",
    "atomtype_amber",
    "atomtype_bcc",
    "atomtype_gas",
    "atomtype_gff",
    "atomtype_gff2",
    "atomtype_sybyl",
    "bccparm",
    "bccparm_abcg2",
    "gaff",
    "gaff2",
    "gaff_empirical",
    "gaff_equiv",
    "gasparm",
    "oplsaa",
];

/// Every `.rs` under `molrs/src`, with line comments stripped.
///
/// Comments are stripped for the same reason `ff/typifier/source_gate.rs` strips them:
/// this file's own prose, and `core/data.rs`'s, mentions `include_str!` and must not
/// trip the gate that forbids it. Stripping text can only remove a match, never invent
/// one, so the scanner's bias stays towards false negatives.
fn src_sources() -> Vec<(PathBuf, String)> {
    let mut out = Vec::new();
    collect(&src_dir(), &mut out);
    assert!(
        !out.is_empty(),
        "no .rs found under {} — every gate in this file would pass vacuously",
        src_dir().display()
    );
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

fn collect(dir: &Path, out: &mut Vec<(PathBuf, String)>) {
    for entry in std::fs::read_dir(dir).expect("read the molrs source tree") {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            collect(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            let src = std::fs::read_to_string(&path).expect("read a molrs source file");
            out.push((path, strip_line_comments(&src)));
        }
    }
}

/// Drop everything from the first `//` of each line.
fn strip_line_comments(src: &str) -> String {
    src.lines()
        .map(|line| match line.find("//") {
            Some(at) => &line[..at],
            None => line,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// `file:line: <text>` for every line of `molrs/src` containing `needle`.
fn src_lines_containing(needle: &str) -> Vec<String> {
    src_sources()
        .iter()
        .flat_map(|(path, src)| {
            src.lines()
                .enumerate()
                .filter(|(_, line)| line.contains(needle))
                .map(|(n, line)| format!("{}:{}: {}", shown(path), n + 1, line.trim()))
                .collect::<Vec<_>>()
        })
        .collect()
}

fn manifest() -> String {
    let path = params_dir().join("MANIFEST.sha256");
    std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "{} is not readable ({e}).\n\
             chem-perceive-14 flattens `ff/params/generated/` into `ff/params/`, and the \
             manifest moves with the tables it hashes.",
            shown(&path)
        )
    })
}

// ---------------------------------------------------------------------------
// ac-001 — one form: no parameter table is raw text
// ---------------------------------------------------------------------------

/// ac-001 — molrs embeds no parameter table as raw text.
///
/// The criterion words this as an absolute: `grep -rn 'include_str!' molrs/src` returns
/// **0 hits**. Today it returns exactly three, all in `src/core/data.rs`, and all three
/// are parameter tables:
///
/// ```text
/// src/core/data.rs:7:  pub const MMFF94_XML  = include_str!(.../data/mmff94.xml)
/// src/core/data.rs:11: pub const MMFF94S_XML = include_str!(.../data/mmff94s.xml)
/// src/core/data.rs:24: pub const OPLSAA_XML  = include_str!(.../data/oplsaa.xml)
/// ```
///
/// That is 481.5 KB of XML sitting in the binary as unparsed text, re-parsed on every
/// `MMFF94Typifier::new()` / `MMFF94STypifier::new()` / `OPLSAATypifier::oplsaa()`
/// call. Deleting the consts is not enough on its own — a table could be re-embedded
/// anywhere — so the gate is on the *form*, tree-wide, exactly as the criterion words
/// it.
#[test]
fn no_parameter_table_is_embedded_as_raw_text() {
    let hits = src_lines_containing("include_str!");
    assert!(
        hits.is_empty(),
        "`include_str!` must have 0 hits in molrs/src — every parameter table is a \
         typed, compiled `.rs` under `ff::params`, not text parsed at runtime. \
         Still embedded:\n  {}",
        hits.join("\n  ")
    );
}

/// ac-001 — all three parameter XMLs are deleted from the tree.
///
/// `mmff94.xml` and `oplsaa.xml` are redundant once their content lives in the committed
/// `.rs`: keeping them would leave two copies of the same numbers, one of which nothing
/// reads and nothing checks.
///
/// `mmff94s.xml` is a different case and deserves naming: it is **not** converted at all.
/// It differs from `mmff94.xml` by two lines — the `<ForceField name=…>` attribute — so
/// there is nothing in it to convert. One shared MMFF table serves both front doors.
#[test]
fn the_three_parameter_xmls_are_deleted() {
    let still_there: Vec<String> = XML_FILES
        .iter()
        .map(|stem| data_dir().join(format!("{stem}.xml")))
        .filter(|path| path.exists())
        .map(|path| {
            let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            format!("{} ({} KB)", shown(&path), size / 1024)
        })
        .collect();

    assert!(
        still_there.is_empty(),
        "the XML parameter sets are still on disk after their content was compiled \
         into `ff::params`:\n  {}\n\
         Two copies of the same numbers, and only one of them is checked by anything. \
         (`mmff94s.xml` is deleted outright, not converted — it carries no numbers that \
         `mmff94.xml` does not.)",
        still_there.join("\n  ")
    );
}

/// ac-001 — `molrs/data/` is gone entirely, not merely emptied of XML.
///
/// The spec's testing strategy states it as a directory-level fact ("`molrs/data/`
/// 目录不再存在"), and that is the form that holds: a surviving `data/` is an invitation
/// to drop the next parameter file back into it, which is precisely the four-forms /
/// three-places sprawl this spec exists to end.
#[test]
fn the_data_directory_is_gone() {
    let dir = data_dir();
    let contents: Vec<String> = std::fs::read_dir(&dir)
        .map(|rd| {
            rd.filter_map(Result::ok)
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default();

    assert!(
        !dir.exists(),
        "{} still exists, holding: {}\n\
         Parameter data has ONE home — `molrs/src/ff/params/`. The last file out turns \
         off the lights: remove the directory, not just its contents.",
        shown(&dir),
        if contents.is_empty() {
            "(nothing)".to_owned()
        } else {
            contents.join(", ")
        }
    );
}

/// ac-001 — the legacy Open Babel coordinate-template library stays absent.
///
/// `data/gen3d/{rigid,ring}-fragments.txt` were Open Babel's fragment templates: not
/// RDKit ETKDG parameters, and never loaded by molrs — zero references in the tree.
/// The owner deleted them outright rather than convert dead data to typed Rust, so
/// this gate pins the deletion. It is **green today**; it is here to stay green.
#[test]
fn legacy_gen3d_fragment_libraries_are_removed() {
    let dir = data_dir().join("gen3d");
    assert!(
        !dir.exists(),
        "{} must not be restored: molrs generates 3D coordinates with ETKDG and has no \
         fragment-template backend, so these files would be dead data compiled into the \
         binary — the exact thing this spec removes",
        shown(&dir)
    );
}

// ---------------------------------------------------------------------------
// ac-001 — one place: `ff/params/`, flat
// ---------------------------------------------------------------------------

/// ac-001 — no directory named `generated` survives anywhere under `molrs/src`.
///
/// Verbatim from the owner: *"如果是原位的，那就不要叫 generated 这种傻逼的名字！"*
///
/// The tables are checked in, reviewed, grepped, diffed and stepped through like any
/// other source. A directory called `generated` says "build artefact — regenerate me,
/// do not read me", which is false of every file in it, and it is the name that let
/// parameter data drift into a second-class corner of the tree in the first place.
/// Provenance is a header comment's job, not a directory's.
///
/// The test's own NAME obeys the rule it enforces: it says what must be true, not
/// which forbidden word it looks for. The word survives in the prose and in the
/// needle below, where it is a string being searched for — never an identifier.
#[test]
fn no_table_directory_is_named_after_its_provenance() {
    let mut found = Vec::new();
    find_dirs_named(&src_dir(), "generated", &mut found);
    assert!(
        found.is_empty(),
        "these directories are named `generated`:\n  {}\n\
         Flatten the tables into `molrs/src/ff/params/` and record how each arrived in \
         its own header doc. \"Generated\" is how they got here, not what they are.",
        found
            .iter()
            .map(|p| shown(p))
            .collect::<Vec<_>>()
            .join("\n  ")
    );
}

fn find_dirs_named(dir: &Path, name: &str, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        if path.file_name().and_then(|n| n.to_str()) == Some(name) {
            out.push(path.clone());
        }
        find_dirs_named(&path, name, out);
    }
}

/// ac-001 — all seventeen parameter tables sit directly under `ff/params/`, declared.
///
/// Fourteen are there today (one directory down, under `generated/`); three arrive with
/// this spec. What the gate pins is that they arrive the way the others do — a `.rs` in
/// the one parameter directory, declared by the module root — so that
/// `params::committed_tables_match_the_manifest_hashes` covers them for free, without
/// having to know any of them by name.
#[test]
fn every_parameter_table_lives_directly_under_ff_params() {
    let dir = params_dir();
    let mod_rs = std::fs::read_to_string(dir.join("mod.rs")).expect("ff/params/mod.rs");

    let mut missing = Vec::new();
    for stem in PARAM_TABLES {
        let table = format!("{stem}.rs");
        if !dir.join(&table).is_file() {
            missing.push(format!("{table}: not a file in `ff/params/`"));
            continue;
        }
        if !mod_rs.contains(&format!("pub mod {stem};")) {
            missing.push(format!(
                "{table}: present, but `ff/params/mod.rs` does not declare `pub mod {stem};` \
                 — it is not compiled at all"
            ));
        }
    }

    assert!(
        missing.is_empty(),
        "parameter tables are not in the one place they belong:\n  {}\n\
         `git mv` the fourteen out of `generated/`, emit `oplsaa.rs` from \
         `scripts/gen_param_tables.py`, and declare each in `ff/params/mod.rs`. \
         (MMFF is one shared table, `mmff.rs`, checked by \
         `the_rdkit_mmff_port_is_merged_into_ff_params`.)",
        missing.join("\n  ")
    );
}

/// ac-001 — the RDKit port lands in `ff/params/mmff.rs` with **every value intact**.
///
/// `ff/mmff/tables.rs` is the odd one out: already typed Rust (17 static tables and 17
/// binary-search accessors, ported from RDKit's `Params.cpp`, BSD-3), bigger than every
/// generated table put together, and living nowhere near them. It is parameter data, so
/// it goes where parameter data goes — merged into the **one shared MMFF table**
/// alongside the 199 entries left in `mmff94.xml`.
///
/// # Why 17 hashes and not one
///
/// The first version of this gate hashed the whole file body (`//!` header stripped):
/// the port has no `use` statement in it, so "not one line changed" was literally
/// checkable in a single digest. That is no longer possible — `ff/params/mmff.rs` is a
/// **merge**, so it legitimately carries content the port never had (the vdW rows, the
/// atom properties, the style skeleton), and a whole-file hash would fail on a correct
/// implementation.
///
/// So the pin moves down one level, to the 17 static tables themselves — **51,222 of the
/// port's 51,621 lines**, which is every number in it. Each is hashed by name, from its
/// `pub static NAME…= &[` line through its closing `];`. This is *stronger* than the
/// whole-file digest was, not weaker:
///
/// * it is invariant to where the statics sit in the merged file, what is added around
///   them, and how the header doc is reworded — none of which is a force-field change;
/// * it still catches the only thing that matters — a single re-rounded or mistyped
///   value in ~50,000 — and now it names *which table* moved;
/// * and it is not fooled by a merge that dropped a whole static, because each of the 17
///   is looked up by name.
///
/// Reformatting a table, re-deriving a value or dropping an accessor is a force-field
/// change wearing a refactor's clothes. The MMFF suites would not see it: the numbers in
/// `MMFF_ANGLE` alone (2,342 rows) dwarf everything the energy fixtures touch.
#[test]
fn the_rdkit_mmff_port_is_merged_into_ff_params() {
    /// SHA-256 of each `pub static` block of the RDKit port — from its declaration line
    /// through its closing `];` — as of 2026-07-14. Together they are 51,222 of the
    /// port's 51,621 lines: every parameter value it holds.
    const PORT_STATICS: [(&str, &str, usize); 17] = [
        (
            "MMFF_AROM",
            "22ef308b872085f491981840ca9baa31438478cd37b1e2542a0c204c688bda06",
            3,
        ),
        (
            "MMFF_DEF",
            "4244282804c92b11821a1389aaf9e8c71a1a5acaacc5a1795cf1e8f0c4bb5e9f",
            382,
        ),
        (
            "MMFF_PROP",
            "815ea8644f4326079690a3df778e7e0835051e2e595b0377098dc7ec6aa27026",
            1047,
        ),
        (
            "MMFF_PBCI",
            "56d49631d6a4e339bb4fc0f67755bdbd42839229e281d7af60ff6bde64278277",
            497,
        ),
        (
            "MMFF_CHG",
            "372253f182e83f556471923303dd0f30df2043b1e335e5db37473df81c1f4334",
            2990,
        ),
        (
            "MMFF_BOND",
            "fe90d41b1004856ffefec954832bb0717c30ba386c93510fc953d04cf0b9dc18",
            3453,
        ),
        (
            "MMFF_BNDK",
            "34e7234faed20f5531650a1cd2de5276c5958ea36708d474e6dd23d5460c93ee",
            350,
        ),
        (
            "MMFF_HERSCHBACH_LAURIE",
            "51777b2e3bd82565e48c3c8f31067a416bc1c23039c99462d0fa71729326cded",
            177,
        ),
        (
            "MMFF_COV_RAD_PAU_ELE",
            "b48d2dd9629c26bd165cbcbac2afbb5af9c90559bd4b31fc47439e3d4c4bc534",
            92,
        ),
        (
            "MMFF_ANGLE",
            "84728a5a8ef41f05ff189e5d543c47b7d4331f353789846f5f64009e77131214",
            18738,
        ),
        (
            "MMFF_STBN",
            "2dabc1a3f344a9791fa516daa906525529d3f8b6c9c927a0b21f3a1cea32f3b4",
            2258,
        ),
        (
            "MMFF_DFSB",
            "5b8454d211f65006892b284248b360bb528f63b8762d0180b3858f35b615bc2a",
            212,
        ),
        (
            "MMFF_OOP",
            "59710fb8c1dd44697a54cb71802ec814a95cc76bb0c03d0e48106969a29be8e6",
            821,
        ),
        (
            "MMFF_OOP_S",
            "0d16000b12f253bdd6d3f7525630abef72d6ce74307fd8be450483d702068190",
            821,
        ),
        (
            "MMFF_TOR",
            "34d9d02c69bcf5a2b285532e82d82a3a0cbb81b8bdb8f2ffd77b01a42396ba76",
            9262,
        ),
        (
            "MMFF_TOR_S",
            "2718f5845ca66ab9ffc4791b336343b11dd1845582773f923cc22aaf85f9223d",
            9262,
        ),
        (
            "MMFF_VDW",
            "6f4a241e0ee3b971da51b8cb32efa3015edf19f898437c4b638099e158b95bb9",
            857,
        ),
    ];

    let old = src_dir().join("ff/mmff/tables.rs");
    let new = params_dir().join("mmff.rs");

    assert!(
        !old.exists(),
        "{} still exists: the MMFF parameter tables are parameter data and belong in \
         `ff/params/`, with every other table. (And `tables.rs` is a banned name there — \
         everything under `ff/params/` is a table.) `git mv` it to {}.",
        shown(&old),
        shown(&new)
    );
    assert!(
        new.is_file(),
        "{} does not exist. It is the ONE shared MMFF table: the RDKit `Params.cpp` port \
         (51,621 lines — the single largest table in the tree) merged with the 199 \
         entries left in `mmff94.xml`. Both `MMFF94Typifier` and `MMFF94STypifier` read \
         it; they differ only by a name string and an `MmffVariant`.",
        shown(&new)
    );

    let src = std::fs::read_to_string(&new).expect("read ff/params/mmff.rs");

    // Provenance lives in the header doc, per the owner's ruling; a directory named
    // `generated` used to say "this came from somewhere else", and now the header must.
    let header: String = src
        .lines()
        .take_while(|l| l.trim_start().starts_with("//!") || l.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        header.contains("Params.cpp"),
        "{} does not record its provenance in its header doc. This one came from RDKit's \
         `Code/ForceField/MMFF/Params.cpp` (BSD-3) — plus `mmff94.xml` via \
         `scripts/gen_param_tables.py`. Both belong in the header.",
        shown(&new)
    );

    // Every static, by name, hashed from its declaration through its closing `];`.
    let lines: Vec<&str> = src.lines().collect();
    let mut drifted = Vec::new();
    for (name, want, want_lines) in PORT_STATICS {
        let Some(start) = lines
            .iter()
            .position(|l| l.starts_with(&format!("pub static {name}:")))
        else {
            drifted.push(format!(
                "{name}: no `pub static {name}:` in the merged table"
            ));
            continue;
        };
        let Some(len) = lines[start..].iter().position(|l| l.trim() == "];") else {
            drifted.push(format!("{name}: never closed with `];`"));
            continue;
        };
        let block = lines[start..=start + len].join("\n");
        let got = {
            use sha2::{Digest, Sha256};
            format!("{:x}", Sha256::digest(block.as_bytes()))
        };
        if got != want {
            drifted.push(format!(
                "{name}: {} lines (was {want_lines}), sha256 {got} != {want}",
                len + 1
            ));
        }
    }

    assert!(
        drifted.is_empty(),
        "{} does not carry the RDKit port's tables unchanged:\n  {}\n\
         These 17 statics are 51,222 lines — every parameter value the port holds. The \
         merge may add content around them and reword the header; it may not touch a \
         number. Re-deriving a value or re-rounding a column is a force-field change \
         wearing a refactor's clothes, and no energy fixture in the suite would see it \
         (`MMFF_ANGLE` alone has 2,342 rows).",
        shown(&new),
        drifted.join("\n  ")
    );

    // The 17 binary-search accessors are the port's whole API; a drop would not change
    // any static's hash, so it gets its own check.
    const ACCESSORS: [&str; 17] = [
        "mmff_def",
        "mmff_prop",
        "mmff_pbci",
        "mmff_chg",
        "mmff_bond",
        "mmff_bndk",
        "mmff_herschbach_laurie",
        "mmff_cov_rad_pau_ele",
        "mmff_angle",
        "mmff_stbn",
        "mmff_dfsb",
        "mmff_oop",
        "mmff_oop_s",
        "mmff_tor",
        "mmff_tor_s",
        "mmff_vdw",
        "mmff_is_arom",
    ];
    let lost: Vec<&str> = ACCESSORS
        .into_iter()
        .filter(|f| !src.contains(&format!("pub fn {f}(")))
        .collect();
    assert!(
        lost.is_empty(),
        "{} lost accessors in the move: {lost:?}",
        shown(&new)
    );
}

/// ac-001 — the MMFF resolver is `ff/mmff/resolve.rs`, and it stays **out** of `params/`.
///
/// Those 808 lines are **not parameters, they are an algorithm**: `bond_type` /
/// `angle_type` / `torsion_type`, the four-level equivalence degradation, and the
/// empirical-rule fallbacks that invent parameters from covalent radii and
/// electronegativities when every lookup has failed. Calling the file `params.rs` was a
/// naming error in spec 02, and the owner corrected it here: it is renamed to
/// `resolve.rs` and it belongs with the typifier front end, **not** under `ff/params/`.
///
/// The file's own header doc still carries the old instruction — *"the resolver should
/// travel with its tables"* — and that instruction is now **superseded**. It was written
/// when `ff/params/` was going to be the home of anything MMFF-shaped. It is not: it is
/// the home of *tables*. Proximity to the rows it degrades through is a real
/// consideration, but it does not outweigh putting an algorithm in a directory whose
/// entire contract, asserted by [`every_file_under_ff_params_is_a_table`], is that
/// everything in it is data. An implementer who reads that header and dutifully moves the
/// file will be caught here.
///
/// # Why this one is not hashed, when the port is
///
/// [`the_rdkit_mmff_port_is_merged_into_ff_params`] pins 17 SHA-256s because the port is
/// ~50,000 **numbers**, and a single re-rounded value is invisible to every energy test
/// in the suite. The resolver is the opposite on both counts: 765 lines of **logic**,
/// whose imports (`use crate::ff::mmff::tables::{…}`) *must* change for the rename to
/// compile at all — so a body hash would fail on a correct rename. And its behaviour is
/// already covered where behaviour belongs: `ff/mmff/energy.rs` checks MMFF94 and MMFF94s
/// total energies against RDKit's own oracle term-by-term on 11 fixtures, and
/// `ff/typifier/mmff_variant.rs` pins the resolved `koop` / `(v1, v2, v3)` per centre. A
/// logic change here moves those numbers. A hash would add brittleness, not coverage.
#[test]
fn the_mmff_resolver_is_renamed_to_resolve_and_stays_out_of_params() {
    /// The resolver's whole entry-point surface, as `ff/typifier/mmff/frame_builder.rs`
    /// calls it (aliased there as `eparams`). All `pub(crate)` — the resolver is an
    /// internal contract between the tables and the typifier, not public API.
    const ENTRY_POINTS: [&str; 8] = [
        "bond_type",
        "angle_type",
        "torsion_type",
        "bond_params",
        "angle_params",
        "stretch_bend_params",
        "oop_params",
        "torsion_params",
    ];

    let old = src_dir().join("ff/mmff/params.rs");
    let new = src_dir().join("ff/mmff/resolve.rs");

    assert!(
        !old.exists(),
        "{} still exists. Those 808 lines are an ALGORITHM, not parameters — `bond_type` \
         / `angle_type` / `torsion_type`, four-level equivalence degradation, empirical \
         rules. `params.rs` was a naming error in spec 02; `git mv` it to {}.\n\
         (Do not confuse it with `ff/typifier/mmff/params.rs` — that is `MMFFParams`, the \
         typing metadata, and it stays where it is.)",
        shown(&old),
        shown(&new)
    );
    assert!(
        new.is_file(),
        "{} does not exist. The resolver is renamed, not moved: it stays in `ff/mmff/` \
         with the typifier front end.",
        shown(&new)
    );

    let src = std::fs::read_to_string(&new).expect("read ff/mmff/resolve.rs");
    let missing: Vec<&str> = ENTRY_POINTS
        .into_iter()
        .filter(|f| !src.contains(&format!("fn {f}(")))
        .collect();
    assert!(
        missing.is_empty(),
        "{} lost entry points in the rename: {missing:?}. This is a `git mv` plus import \
         fixes (its `use crate::ff::mmff::tables::{{…}}` now points at \
         `crate::ff::params::mmff`, and `frame_builder.rs` imports it as `eparams`). \
         Change no logic — the resolver's numbers are pinned against RDKit by \
         `ff/mmff/energy.rs` and `ff/typifier/mmff_variant.rs`.",
        shown(&new)
    );

    // And it must NOT have been dragged into the table directory.
    let strays: Vec<String> = std::fs::read_dir(params_dir())
        .expect("read ff/params/")
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("rs"))
        .filter(|p| {
            std::fs::read_to_string(p)
                .is_ok_and(|s| ENTRY_POINTS.iter().all(|f| s.contains(&format!("fn {f}("))))
        })
        .map(|p| shown(&p))
        .collect();
    assert!(
        strays.is_empty(),
        "the MMFF resolver has been moved INTO the table directory: {strays:?}\n\
         It is an algorithm, and `ff/params/` holds only data. Its own header doc asks to \
         `travel with its tables` — that instruction is superseded by the owner's naming \
         ruling and should be corrected in the move, not obeyed."
    );
}

/// ac-001 — everything under `ff/params/` is a table, so nothing there is named
/// `*_tables` or `tables.rs`.
///
/// Owner, verbatim: *"`_tables` 这个名字不好，换个名字"*. The directory's whole contract is
/// that it holds parameter tables and nothing else — so a `_tables` suffix on a file
/// inside it says "table table", and a bare `tables.rs` says nothing at all. The name has
/// one job here: to say **which** force field's data this is (`gaff2`, `oplsaa`, `mmff`).
///
/// This is also the gate that catches the laziest possible implementation of this spec:
/// `git mv ff/mmff/tables.rs ff/params/tables.rs` and call it done.
#[test]
fn every_file_under_ff_params_is_a_table() {
    let dir = params_dir();
    let badly_named: Vec<String> = std::fs::read_dir(&dir)
        .expect("read ff/params/")
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("rs"))
        .filter(|p| {
            let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or_default();
            stem == "tables" || stem.ends_with("_tables")
        })
        .map(|p| shown(&p))
        .collect();

    assert!(
        badly_named.is_empty(),
        "these files are named `*_tables` / `tables` inside the table directory:\n  {}\n\
         Everything under `ff/params/` IS a table — the suffix says \"table table\". Name \
         the file after the force field whose data it holds (`mmff.rs`, `oplsaa.rs`, \
         `gaff2.rs`).",
        badly_named.join("\n  ")
    );
}

/// ac-001 — no parameter table is dead weight.
///
/// A table no code path can reach is not a parameter set, it is binary size and a
/// maintenance bill. This is the gate that **held this spec back**: converting
/// `mmff94s.xml` while nothing read it would have produced a typed table, hashed in the
/// manifest and byte-reproduced by the generator, and read by nobody — the same dead
/// weight as `data/gen3d/`, which the owner deleted rather than convert. So
/// chem-perceive-14 was resequenced behind `mmff-typifier-split`, which gave MMFF94s
/// its first real reader (`MMFF94STypifier`, via `MmffEngine::embedded`). All three
/// XML sets now have consumers, and this gate is what keeps it that way.
///
/// Mechanism-agnostic on purpose: it asserts the module is *declared* and that *some*
/// public item of it is named from outside `ff/params/`. Not any particular constructor,
/// re-export style or const name — those are the implementer's to choose.
#[test]
fn no_parameter_table_is_unreachable() {
    let dir = params_dir();
    let mod_rs = std::fs::read_to_string(dir.join("mod.rs")).expect("ff/params/mod.rs");
    let sources = src_sources();

    let mut tables = 0usize;
    let mut orphans = Vec::new();
    for entry in std::fs::read_dir(&dir).expect("read ff/params/") {
        let path = entry.expect("dir entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .expect("table file stem");
        if stem == "mod" {
            continue;
        }
        tables += 1;

        // `mod {stem};` rather than `pub mod {stem};`: the MMFF resolver is crate-internal
        // (`pub(crate)`), and reachability — not publicity — is what this gate is about.
        if !mod_rs.contains(&format!("mod {stem};")) {
            orphans.push(format!(
                "{stem}.rs: not declared in `ff/params/mod.rs` — it is not even compiled"
            ));
            continue;
        }

        let items = public_item_names(&path);
        assert!(
            !items.is_empty(),
            "{stem}.rs declares no reachable `const` / `static` / `fn` — nothing for a \
             consumer to name"
        );
        let reached = items.iter().any(|name| {
            sources
                .iter()
                .any(|(p, src)| !p.starts_with(&dir) && names_item(src, name))
        });
        if !reached {
            orphans.push(format!(
                "{stem}.rs: none of its public items ({}) is named anywhere in molrs/src \
                 outside `ff/params/` — it is compiled into the binary and read by nobody",
                items.join(", ")
            ));
        }
    }

    assert!(
        tables >= PARAM_TABLES.len(),
        "only {tables} tables found directly under `ff/params/`, expected at least {} — \
         this gate would pass vacuously. (Before chem-perceive-14 they are one directory \
         down, in `generated/`.)",
        PARAM_TABLES.len()
    );
    assert!(
        orphans.is_empty(),
        "parameter tables that no code can reach:\n  {}\n\
         Wire each table to a consumer, or do not ship it.",
        orphans.join("\n  ")
    );
}

/// The `const` / `static` / `fn` identifiers a table exports to the rest of the crate.
///
/// `pub(crate)` counts: the question this gate asks is "can any code outside `ff/params/`
/// reach it", not "is it in the public API".
fn public_item_names(path: &Path) -> Vec<String> {
    const KEYWORDS: [&str; 6] = [
        "pub const ",
        "pub static ",
        "pub fn ",
        "pub(crate) const ",
        "pub(crate) static ",
        "pub(crate) fn ",
    ];
    let src = std::fs::read_to_string(path).expect("read a parameter table");
    src.lines()
        .filter_map(|line| {
            let trimmed = line.trim_start();
            let rest = KEYWORDS.iter().find_map(|kw| trimmed.strip_prefix(kw))?;
            let name = rest.split([':', ' ', '(', '<']).next()?;
            (!name.is_empty()).then(|| name.to_owned())
        })
        .collect()
}

/// Whether `src` names `item` as an identifier rather than as a substring of a longer
/// one (`MMFF94S` must not be "found" inside `MMFF94S_XML`).
fn names_item(src: &str, item: &str) -> bool {
    src.match_indices(item).any(|(at, _)| {
        let before = src[..at].chars().next_back();
        let after = src[at + item.len()..].chars().next();
        let boundary = |c: Option<char>| !c.is_some_and(|c| c.is_alphanumeric() || c == '_');
        boundary(before) && boundary(after)
    })
}

// ---------------------------------------------------------------------------
// ac-003 — provenance stays, AmberTools coupling goes
// ---------------------------------------------------------------------------

/// ac-003 — the manifest records each XML's **source** hash, not just the emitted one.
///
/// `MANIFEST.sha256` has two kinds of row, and both matter:
///
/// ```text
/// emitted <sha256>  bccparm.rs                          <- catches a hand-edit
/// source  <sha256>  dat/antechamber/BCCPARM.DAT         <- catches an upstream change
/// ```
///
/// For the AmberTools tables the source can be re-read from an AmberTools install on
/// demand. For these the source is *deleted* by ac-001 — so the `source` row is the only
/// surviving record of which bytes the committed table was derived from. Drop it and the
/// provenance of 481.5 KB of numbers is gone: `git log` would still have the XML, but
/// nothing in the tree would say *which revision of it* these tables came from.
///
/// **Two rows, not three.** `mmff94s.xml` is deleted without being converted — it differs
/// from `mmff94.xml` by the `<ForceField name=…>` attribute and nothing else, so it is
/// the source of no table and has no provenance to record. Recording it anyway would be a
/// lie of exactly the useful-looking kind: a `source` row implies "some committed table
/// was derived from these bytes", and none was.
///
/// The hashes are the ones the XMLs have **today**, re-derived 2026-07-14. `mmff94.xml`'s
/// moved since this spec was drafted (`mmff-orthogonal-02` deleted 4,065 dead type-def
/// rows; `mmff-orthogonal-01` added `<ElectrostaticParams>`); `oplsaa.xml` is untouched
/// and still hashes to `d997039c…`. That is exactly why the reference fixtures had to be
/// re-frozen from the *current* XML: freezing the old dump would have compiled 4,065 dead
/// rows into committed, test-protected Rust — the one outcome the orthogonal-02 chain
/// existed to prevent.
///
/// The same hashes are pinned in the header of each frozen reference fixture
/// (`tests/ff/fixtures/tables/*.reference.txt`), which is what ties the equivalence
/// test's expected values to these specific source bytes.
#[test]
fn the_manifest_records_the_xml_source_hashes() {
    const XML_SOURCE_HASHES: [(&str, &str); 2] = [
        (
            "mmff94.xml",
            "9d9c41db11529da54a301e446cc912b11bdec43d43bc466e8fcd5eac45da72a5",
        ),
        (
            "oplsaa.xml",
            "d997039c15e24f63272bcee55d0f27622d5d11d00f78e572dea364b405c09af2",
        ),
    ];

    let manifest = manifest();
    let sources: Vec<&str> = manifest
        .lines()
        .filter_map(|line| line.strip_prefix("source "))
        .collect();

    let mut missing = Vec::new();
    for (file, hash) in XML_SOURCE_HASHES {
        let row = sources
            .iter()
            .find(|row| row.trim_start().starts_with(hash) && row.contains(file));
        if row.is_none() {
            missing.push(format!("{file}: no `source {hash}  …{file}` row"));
        }
    }

    assert!(
        missing.is_empty(),
        "MANIFEST.sha256 does not record where the XML tables came from:\n  {}\n\
         ac-001 deletes the XMLs, so this row is the last record of the provenance of \
         481.5 KB of force-field numbers. Hash the source in the generator, as it already \
         does for every upstream AmberTools table.",
        missing.join("\n  ")
    );
}

/// ac-003 — **no test in the suite is coupled to an AmberTools install.**
///
/// From the owner (the variable name elided — this file has to talk about the rule to
/// explain it, and must not become the one hit the rule forbids; the un-elided quote is
/// in `.claude/specs/chem-perceive-14-all-tables.md`):
///
/// > "ci 不要和 …HOME 有任何牵连，只在实施过程中验证一次！"
/// >
/// > *CI must have no entanglement whatsoever with the AmberTools install root; verify
/// > it once, during implementation.*
///
/// The criterion is a grep, and this is that grep: the AmberTools install-root
/// environment variable is named **nowhere** under `molrs/tests`. Byte-for-byte
/// regeneration is validated exactly once, locally, by the implementer, and the result
/// is recorded in the spec body — it is not a CI job.
///
/// The rule is *delete*, not *skip*, and the distinction is the whole point. The drift
/// guard this replaces (`params::generator_byte_reproduces_the_committed_tables`) began
/// with `if env::var_os(…).is_none() { return; }`, so on every CI run, on every
/// contributor machine without AmberTools, it printed a skip line and passed. It has
/// never once run in CI. A test that skips itself where it runs buys the *appearance* of
/// coverage and none of the substance, and it crowds out the guard that does work:
/// `params::committed_tables_match_the_manifest_hashes`, which recomputes the SHA-256 of
/// every committed table, needs nothing installed, and catches the failure that actually
/// happens — a hand-edit to ~50k lines of generated numbers.
///
/// The needle is spelled `concat!` so that this file, which must talk about the variable
/// to explain itself, does not itself become the one hit the gate forbids.
#[test]
fn no_test_couples_ci_to_ambertools() {
    let needle = concat!("AMBER", "HOME");

    let mut hits = Vec::new();
    let mut sources = Vec::new();
    collect_raw(&tests_dir(), &mut sources);
    assert!(
        !sources.is_empty(),
        "no .rs found under {} — this gate would pass vacuously",
        tests_dir().display()
    );
    for (path, src) in &sources {
        for (n, line) in src.lines().enumerate() {
            if line.contains(needle) {
                hits.push(format!("{}:{}: {}", shown(path), n + 1, line.trim()));
            }
        }
    }

    assert!(
        hits.is_empty(),
        "the test suite still names the AmberTools install root:\n  {}\n\
         CI must have ZERO coupling to AmberTools. Delete the test — do not make it skip. \
         Byte-for-byte regeneration is verified once, locally, during implementation, and \
         recorded in the spec; the standing guard is the manifest hash check, which needs \
         nothing installed. (A comment that merely mentions the variable is also a hit: \
         the criterion is a plain grep, and a grep that has to be read with exceptions is \
         not a gate.)",
        hits.join("\n  ")
    );
}

/// Every `.rs` under `dir`, verbatim (no comment stripping — the grep above is literal).
fn collect_raw(dir: &Path, out: &mut Vec<(PathBuf, String)>) {
    for entry in std::fs::read_dir(dir).expect("read the molrs test tree") {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            collect_raw(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            let src = std::fs::read_to_string(&path).expect("read a molrs test file");
            out.push((path, src));
        }
    }
}
