//! chem-perceive-15 — **the whole-chain architecture gate.**
//!
//! Sixteen specs ran on this chain. Each verified its own slice; nothing verified
//! the whole. This target is the whole: the five "only one of these exists"
//! promises, turned from intentions into machine-checked gates.
//!
//! Every gate here corresponds to a defect that **actually happened**. None is a
//! hypothetical risk:
//!
//! | Promise | What grew back when nobody was watching |
//! |---|---|
//! | one place, one form for parameters | 4,065 rows of XML no code read, alive only to bribe an `is_empty()` guard |
//! | one perception layer | a `chem` alias left behind by a rename |
//! | one interpolation seam | a second estimator stack (reworked out once already) |
//! | one MMFF path | a bespoke energy layer + a second classifier that answered the *opposite* of RDKit |
//! | a `tp`-ignoring ctor is not a Style | 8 of them, registered as if they read type rows |
//!
//! # Why the gates are REVERSE assertions
//!
//! A forward assertion ("X exists, and it is correct") is fooled by *"X was added,
//! but in the wrong place, and the old one is still there too."* Every gate below
//! asserts that something is **absent** — a second implementation, a second home, a
//! second spelling — because absence is the only property a duplicate cannot satisfy.
//!
//! # Why the gates cannot exempt themselves
//!
//! A gate that skips its own file is a gate with a hole exactly the shape of the
//! next defect. This file is scanned by every gate in it. That is only possible
//! because no needle is ever *written* here: each is assembled at compile time with
//! [`concat!`], so the string this file searches for does not occur in this file.
//! Take the `concat!` away and the target fails — on itself. That is the intended
//! behaviour, and `gate_scans_itself` proves the scanner really does read this file.
//!
//! The scanner reads **code only** — comments and string literals are blanked
//! ([`code_only`]) — so the prose above (which names every forbidden thing) is not a
//! violation of anything, and a violation cannot hide inside a `//` or a `"…"`.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

/// `molrs/` — the merged crate's root.
fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// The workspace root — the parent of `molrs/`, i.e. where the sibling binder
/// workspaces (`molrs-python`, `molrs-ffi`, `molrs-wasm`, `molrs-capi`,
/// `molrs-cxxapi`) live. Promise 2 is about **all** of them, not just this crate.
fn repo_root() -> PathBuf {
    crate_root()
        .parent()
        .expect("molrs/ has a parent")
        .to_path_buf()
}

fn src_dir() -> PathBuf {
    crate_root().join("src")
}

fn tests_dir() -> PathBuf {
    crate_root().join("tests")
}

/// Every `.rs` file under `dir`, recursively, sorted.
fn rust_files(dir: &Path) -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
    }
    let mut out = Vec::new();
    walk(dir, &mut out);
    out.sort();
    out
}

/// Every directory under `dir`, recursively.
fn dirs(dir: &Path) -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                out.push(path.clone());
                walk(&path, out);
            }
        }
    }
    let mut out = Vec::new();
    walk(dir, &mut out);
    out.sort();
    out
}

/// Path relative to the workspace root, for readable failure messages.
fn rel(path: &Path) -> String {
    path.strip_prefix(repo_root())
        .unwrap_or(path)
        .display()
        .to_string()
}

// ---------------------------------------------------------------------------
// The scanner — code only
// ---------------------------------------------------------------------------

/// Blank every comment and every string / char literal, keeping byte offsets.
///
/// The gates below search for things that must not be **in the code**. Provenance
/// prose in a header doc is not only allowed, it is where provenance is *required*
/// to live; a failure message that names the forbidden thing is not the forbidden
/// thing. So the scanner blanks:
///
/// * `//` line comments and `/* … */` block comments (nesting, as Rust allows),
/// * `"…"` strings with escapes, `r"…"` / `r#"…"#` raw strings, `'c'` chars.
///
/// Blanked spans become spaces rather than being removed, so line and column
/// numbers still point at the real source.
fn code_only(src: &str) -> String {
    let b = src.as_bytes();
    let mut out = vec![b' '; b.len()];
    let mut i = 0usize;

    // Copy a byte range through unchanged.
    macro_rules! keep {
        ($from:expr, $to:expr) => {
            out[$from..$to].copy_from_slice(&b[$from..$to])
        };
    }

    while i < b.len() {
        // Preserve newlines everywhere so line numbers survive.
        if b[i] == b'\n' {
            out[i] = b'\n';
            i += 1;
            continue;
        }

        // Line comment
        if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'/' {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
            continue;
        }

        // Block comment (nesting)
        if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
            let mut depth = 1usize;
            i += 2;
            while i < b.len() && depth > 0 {
                if b[i] == b'\n' {
                    out[i] = b'\n';
                }
                if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
                    depth += 1;
                    i += 2;
                } else if b[i] == b'*' && i + 1 < b.len() && b[i + 1] == b'/' {
                    depth -= 1;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            continue;
        }

        // Raw string: r"…" or r#…"…"#…
        if b[i] == b'r' && i + 1 < b.len() && (b[i + 1] == b'"' || b[i + 1] == b'#') {
            let mut j = i + 1;
            let mut hashes = 0usize;
            while j < b.len() && b[j] == b'#' {
                hashes += 1;
                j += 1;
            }
            if j < b.len() && b[j] == b'"' {
                j += 1;
                loop {
                    if j >= b.len() {
                        break;
                    }
                    if b[j] == b'\n' {
                        out[j] = b'\n';
                    }
                    if b[j] == b'"' {
                        let close = b[j + 1..]
                            .iter()
                            .take(hashes)
                            .filter(|&&c| c == b'#')
                            .count();
                        if close == hashes {
                            j += 1 + hashes;
                            break;
                        }
                    }
                    j += 1;
                }
                i = j;
                continue;
            }
        }

        // String literal
        if b[i] == b'"' {
            i += 1;
            while i < b.len() {
                if b[i] == b'\n' {
                    out[i] = b'\n';
                }
                if b[i] == b'\\' {
                    // An escaped LINE CONTINUATION still contains a newline, and
                    // dropping it slides every later line number by one — which is how
                    // a scanner reports a violation at an innocent line and is believed.
                    if i + 1 < b.len() && b[i + 1] == b'\n' {
                        out[i + 1] = b'\n';
                    }
                    i += 2;
                    continue;
                }
                if b[i] == b'"' {
                    i += 1;
                    break;
                }
                i += 1;
            }
            continue;
        }

        // Char literal — but NOT a lifetime (`'a`), which is code.
        if b[i] == b'\'' {
            let is_char = (i + 2 < b.len() && b[i + 1] != b'\\' && b[i + 2] == b'\'')
                || (i + 3 < b.len() && b[i + 1] == b'\\');
            if is_char {
                let mut j = i + 1;
                if b[j] == b'\\' {
                    j += 1;
                }
                while j < b.len() && b[j] != b'\'' {
                    j += 1;
                }
                i = j + 1;
                continue;
            }
        }

        keep!(i, i + 1);
        i += 1;
    }

    String::from_utf8_lossy(&out).into_owned()
}

/// `needle` occurs in `hay` as a whole identifier-ish token (not inside a longer
/// word). `snake_case` neighbours count as the same word, so `_generated_` matches
/// `generated`; `degenerate` does not match `generate`.
fn contains_word(hay: &str, needle: &str) -> bool {
    hay.match_indices(needle).any(|(at, _)| {
        let before = hay[..at].chars().next_back();
        let after = hay[at + needle.len()..].chars().next();
        let boundary = |c: Option<char>| !c.is_some_and(|c| c.is_alphanumeric() || c == '_');
        boundary(before) && boundary(after)
    })
}

/// Every `(file, line, text)` in `files` whose CODE contains `needle`.
fn code_hits(files: &[PathBuf], needle: &str) -> Vec<(PathBuf, usize, String)> {
    let mut hits = Vec::new();
    for path in files {
        let Ok(src) = fs::read_to_string(path) else {
            continue;
        };
        for (n, line) in code_only(&src).lines().enumerate() {
            if line.contains(needle) {
                hits.push((path.clone(), n + 1, line.trim().to_owned()));
            }
        }
    }
    hits
}

fn render(hits: &[(PathBuf, usize, String)]) -> String {
    let mut s = String::new();
    for (path, line, text) in hits {
        let _ = writeln!(s, "  {}:{line}: {text}", rel(path));
    }
    s
}

// ---------------------------------------------------------------------------
// The scanner is real
// ---------------------------------------------------------------------------

/// The scanner reads THIS file, and blanks comments and strings while doing it.
///
/// Everything below rests on two claims: that `rust_files(tests_dir())` really
/// includes this gate (so no gate can exempt itself by omission), and that
/// [`code_only`] really blanks prose (so the forbidden words in the module doc above
/// are not themselves failures). Neither claim is observable from a green run — a
/// scanner that silently read *nothing* would make every gate here vacuously green,
/// which is the exact failure mode this whole spec exists to hunt.
#[test]
fn gate_scans_itself() {
    let files = rust_files(&tests_dir());
    let me = Path::new(file!())
        .file_name()
        .expect("this file has a name")
        .to_owned();
    assert!(
        files.iter().any(|f| f.file_name() == Some(&me)),
        "the test-tree scan does not include the gate file itself ({me:?}); a gate that \
         cannot see itself is a gate with a hole exactly the shape of the next defect"
    );

    let src = fs::read_to_string(crate_root().join("tests").join(&me)).expect("read self");
    let code = code_only(&src);
    // The module doc above names the forbidden identifiers in prose. If `code_only`
    // did not blank comments, THIS gate would report itself — which is precisely the
    // property being asserted, in the one direction that can be asserted safely.
    let banned = concat!("gener", "ated");
    assert!(
        src.contains(banned),
        "this file's prose is supposed to name the forbidden word (in a comment)"
    );
    assert!(
        !code.contains(banned),
        "`code_only` failed to blank a comment: the forbidden word `{banned}` survives \
         into the scanned code of the gate file itself, so every scan below is reading \
         prose as if it were code"
    );
}

// ===========================================================================
// PROMISE 1 — one place, one form for parameters
// ===========================================================================

/// `ff/params/` is the only home, and it is FLAT.
///
/// The tables used to live in three places at once (`ff/params/generated/`,
/// `ff/mmff/tables.rs`, `molrs/data/*.xml`) in two forms (Rust consts and parsed
/// XML). Three homes is three chances for a consumer to read the stale one — and
/// one of the three, the XML, was read by nothing at all and existed only to make an
/// `is_empty()` guard pass.
#[test]
fn parameters_have_exactly_one_home_and_it_is_flat() {
    let params = src_dir().join("ff/params");
    assert!(
        params.is_dir(),
        "{} is the one home of every parameter table, and it does not exist",
        rel(&params)
    );

    let subdirs = dirs(&params);
    assert!(
        subdirs.is_empty(),
        "`ff/params/` must be FLAT — a subdirectory is a second home wearing the first \
         one's address:\n{}",
        subdirs
            .iter()
            .map(|d| format!("  {}\n", rel(d)))
            .collect::<String>()
    );

    // The two homes that were deleted, and must not come back — by any name.
    for gone in [
        crate_root().join("data"),
        src_dir().join("ff/mmff/tables.rs"),
        src_dir().join("data"),
    ] {
        assert!(
            !gone.exists(),
            "{} exists again — parameters have one home (`ff/params/`), and it is not this one",
            rel(&gone)
        );
    }

    assert!(
        !rust_files(&params).is_empty(),
        "`ff/params/` is empty; a scan that finds nothing makes every gate here vacuous"
    );
}

/// No parameter table is a DATA FILE anywhere in the crate.
///
/// "One form" is the other half of "one place": a committed `.rs` const is checked by
/// the compiler, an `.xml` / `.DAT` / `.DEF` is checked by nothing until it is parsed
/// — at runtime, in a user's process. A malformed table must be a COMPILE error.
///
/// Test *fixtures* are not parameters and live under `tests/`, so only `src/` is
/// scanned. That distinction is exactly the one the deleted `molrs/data/` blurred.
#[test]
fn no_parameter_table_survives_as_a_data_file() {
    let mut found = Vec::new();
    fn walk(dir: &Path, found: &mut Vec<PathBuf>) {
        let Ok(entries) = fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, found);
                continue;
            }
            let ext = path
                .extension()
                .map(|e| e.to_string_lossy().to_uppercase())
                .unwrap_or_default();
            if matches!(
                ext.as_str(),
                "XML" | "DAT" | "DEF" | "JSON" | "FF" | "FRCMOD"
            ) {
                found.push(path);
            }
        }
    }
    walk(&src_dir(), &mut found);
    assert!(
        found.is_empty(),
        "parameter-shaped data files under `src/` — a table molrs must PARSE is a table \
         whose malformation is a runtime error in a user's process:\n{}",
        found
            .iter()
            .map(|p| format!("  {}\n", rel(p)))
            .collect::<String>()
    );
}

/// Zero `include_str!` in code — the whole `src/` tree.
///
/// The needle is assembled with `concat!`: the scanner reads this file too, and a
/// gate that had to exempt itself would be one edit away from exempting the next
/// offender. (Four occurrences survive in `src/` **as prose**, in the header docs
/// that record what each table replaced. `code_only` blanks them, which is the
/// point: provenance belongs in the doc.)
#[test]
fn nothing_embeds_a_parameter_file_at_compile_time() {
    let needle = concat!("include", "_str", "!");
    let hits = code_hits(&rust_files(&src_dir()), needle);
    assert!(
        hits.is_empty(),
        "`{needle}` in code — a table that arrives as embedded TEXT is a table that is \
         parsed at runtime, and a malformed one is a user's runtime error rather than \
         our compile error:\n{}",
        render(&hits)
    );
}

/// **No BUILT-IN parameter table is parsed at runtime** — and the gate says that on
/// semantics, not on the presence of a parser.
///
/// The claim is *not* "molrs contains no parser". `MMFF94Typifier::from_xml_str(xml)`
/// and `OPLSAATypifier::from_xml_str(xml)` parse XML, and must: that XML is the
/// **caller's**, handed in as an argument, and reading a force field a user gives you
/// is a feature. What must not exist is a parse of text molrs *manufactured for
/// itself* — a built-in table shipped as a string and re-parsed on every construction.
/// That is the shape whose malformation is a runtime error in a user's process
/// instead of a compile error in ours.
///
/// The two are told apart by the SIGNATURE, which is the only place the difference
/// is visible: a function that parses text it was **given** takes text
/// (`&str` / `&[u8]` / `String` / `Path` / `impl Read`); a function that parses text
/// it **made** takes none. `MmffEngine::embedded(variant, name)` and
/// `MMFF94Typifier::new()` take no text, and must therefore contain no parse.
///
/// Scoped to the force-field path — the tables (`ff/params`), the code that reads
/// them (`ff/typifier`, `ff/mmff`, `ff/charge`) and the force fields they build
/// (`ff/forcefield/gaff.rs`).
#[test]
fn no_builtin_parameter_table_is_parsed_at_runtime() {
    let mut files = Vec::new();
    for sub in ["ff/params", "ff/typifier", "ff/mmff", "ff/charge"] {
        files.extend(rust_files(&src_dir().join(sub)));
    }
    files.push(src_dir().join("ff/forcefield/gaff.rs"));

    let parsers = [
        concat!("serde_json", "::", "from_"),
        concat!("include", "_str", "!"),
        concat!("quick", "_xml"),
        concat!("roxml", "tree"),
        concat!("read_forcefield", "_xml"),
        concat!("read_mmff_params", "_xml"),
        concat!("from_reader", "("),
    ];
    // A parameter of one of these types is text the CALLER supplied.
    let text_input = ["&str", "& str", "String", "&[u8]", "Path", "Read", "&mut R"];

    let mut hits = Vec::new();
    for path in &files {
        for item in fn_items(path) {
            if !parsers.iter().any(|p| item.body.contains(p)) {
                continue;
            }
            if text_input.iter().any(|t| item.args.contains(t)) {
                continue; // parses what it was handed — that is a reader, and it is fine
            }
            hits.push((
                path.clone(),
                item.line,
                format!(
                    "fn {}({}) parses text it was never given",
                    item.name, item.args
                ),
            ));
        }
    }
    assert!(
        hits.is_empty(),
        "a BUILT-IN parameter table is parsed at runtime:\n{}\n\
         A function that takes no text and parses some is parsing a table molrs shipped \
         as a string. A malformed table must be a COMPILE error — not an error we hand \
         to a user, in their process, at their call site.",
        render(&hits)
    );
}

/// **Provenance is not a name.** No artifact of the parameter system is named after
/// the process that produced it.
///
/// `generated` says how a table ARRIVED. It says nothing about what it IS — and once
/// a name lies, every later reader is misled by it (the same mistake as calling an
/// 808-line algorithm `params.rs`). Provenance belongs in the header doc, where a
/// reader who wants it will look, and where it cannot be welded onto the public
/// surface.
///
/// # The ruling this gate encodes — stated, not ducked
///
/// The ban is on **artifacts of the parameter system** being named after their
/// production: directories, files, modules, types. It is not a ban on the *spelling*.
/// `scripts/gen_param_tables.py` **is** a generator — that is what it IS, not how it
/// arrived — so a test whose subject is that script may call it one, and a const that
/// holds the provenance marker a table's header must carry may call it that. This is
/// the same distinction the whole gate rests on, applied to itself; enforcing the
/// spelling instead would be the exact error `param_source_is_bidirectional` exists
/// to condemn (a grep finds spellings, a gate finds semantics).
///
/// So the rule is:
///
/// * **`src/` — strict.** Zero identifiers, of any kind. Nothing shipped has a
///   generator as its subject.
/// * **Directories, files, modules, types — strict, everywhere.** These are artifacts.
/// * **`tests/` functions and consts** — the word is allowed only where the identifier
///   also names its subject as a `script` or a `marker`, i.e. where the thing being
///   named is the generation itself and not a parameter table.
///
/// If the owner rejects that reading, tightening this gate is one line (delete the
/// `names_a_generation_artifact` clause) plus two renames in `tests/ff/params.rs`.
#[test]
fn provenance_is_not_a_name() {
    let banned = [concat!("gener", "ated"), concat!("gener", "ator")];

    // --- directories and files (artifacts) — strict, everywhere ---
    let mut named = Vec::new();
    for root in [src_dir(), tests_dir()] {
        for path in dirs(&root).into_iter().chain(rust_files(&root)) {
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().to_lowercase())
                .unwrap_or_default();
            if banned.iter().any(|b| name.contains(b)) {
                named.push(path);
            }
        }
    }
    assert!(
        named.is_empty(),
        "these directories / files are named after how their contents ARRIVED:\n{}\n\
         A table is a table. Where it came from belongs in its header doc.",
        named
            .iter()
            .map(|p| format!("  {}\n", rel(p)))
            .collect::<String>()
    );

    // --- identifiers ---
    // In `src`, any identifier. In `tests`, module and type names always; functions
    // and consts unless the identifier names a generation artifact (a script, or the
    // provenance marker a script stamps) rather than a parameter table.
    let names_a_generation_artifact =
        |ident: &str| ident.contains("script") || ident.contains("marker");

    let mut hits = Vec::new();
    for (root, strict) in [(src_dir(), true), (tests_dir(), false)] {
        for path in rust_files(&root) {
            let Ok(text) = fs::read_to_string(&path) else {
                continue;
            };
            for (n, line) in code_only(&text).lines().enumerate() {
                for ident in line.split(|c: char| !(c.is_alphanumeric() || c == '_')) {
                    let lower = ident.to_lowercase();
                    if !banned.iter().any(|b| lower.contains(b)) {
                        continue;
                    }
                    if !strict {
                        let is_type_or_mod = line.contains("struct ")
                            || line.contains("enum ")
                            || line.contains("trait ")
                            || line.contains("mod ")
                            || line.contains("type ");
                        if !is_type_or_mod && names_a_generation_artifact(&lower) {
                            continue;
                        }
                    }
                    hits.push((path.clone(), n + 1, ident.to_owned()));
                }
            }
        }
    }
    assert!(
        hits.is_empty(),
        "identifiers named after how a thing was produced:\n{}\n\
         Provenance goes in the header doc. A name says what a thing IS.",
        render(&hits)
    );
}

// ===========================================================================
// PROMISE 2 — one perception layer
// ===========================================================================

/// The `chem` alias is gone — from EVERY workspace, not just this one.
///
/// A rename that leaves an alias behind has not renamed anything; it has added a
/// second name. And an alias is invisible to a behavioural test: both paths reach the
/// same code, so every test stays green while the tree carries two spellings of one
/// layer. The only place it is visible is the source, and the only source that
/// matters is *all* of it — the binder workspaces are separate crates, so a `cargo
/// test` in this one would never have noticed.
///
/// Scope note, stated because the scanner cannot state it for itself: the alias this
/// gate hunts is the **crate-root** one — the old name of the perception layer. The
/// SMILES *AST* module (`io::smiles::chem`) is a different module with an unfortunate
/// name, reached only through its parents, and it is not what promise 2 is about.
/// Widening the needle to a bare `mod chem` would flag it, and a gate that cries wolf
/// is a gate that gets an allowlist.
///
/// This doc comment does not spell the forbidden path, and that is not squeamishness:
/// `tests/perceive/chem_alias.rs` forbids the literal in **any** source line, comments
/// included ("a gate that reads nothing passes, and this one has a specific nothing it
/// must not read"). It is the stricter rule, it is the house's, and it caught this very
/// file when the first draft spelled the path out in prose. A gate that exempted itself
/// from a neighbour's gate would be the same self-exemption this target exists to
/// forbid — so the neighbour wins.
#[test]
fn the_chem_alias_is_gone_from_every_workspace() {
    // The path spellings of the crate-root alias. Only the crate root can produce
    // them: a nested `chem` module is reached through its parents.
    let needles = [
        concat!("molrs", "::", "chem"),
        concat!("crate", "::", "chem"),
    ];

    // …and the alias at its source: a re-export at the crate root.
    let lib = src_dir().join("lib.rs");
    let lib_code = code_only(&fs::read_to_string(&lib).expect("read lib.rs"));
    for decl in [
        concat!("mod ", "chem"),
        concat!("pub use ", "chem"),
        concat!("as ", "chem;"),
    ] {
        assert!(
            !lib_code.contains(decl),
            "`{decl}` at the crate root ({}) — that is the `chem` alias, back where it \
             was. One perception layer means one name for it.",
            rel(&lib)
        );
    }

    let mut files = Vec::new();
    for ws in [
        "molrs",
        "molrs-python",
        "molrs-ffi",
        "molrs-wasm",
        "molrs-capi",
        "molrs-cxxapi",
    ] {
        let root = repo_root().join(ws);
        if root.is_dir() {
            files.extend(rust_files(&root));
        }
    }
    assert!(
        files.len() > 100,
        "the sibling-workspace scan found only {} files — a scan that finds nothing \
         passes vacuously, which is the failure this gate exists to prevent",
        files.len()
    );

    let mut hits = Vec::new();
    for needle in needles {
        hits.extend(code_hits(&files, needle));
    }
    assert!(
        hits.is_empty(),
        "the `chem` alias survives:\n{}\nOne perception layer means ONE name for it \
         (`perceive`). An alias is a second name that no behavioural test can see.",
        render(&hits)
    );
}

/// `perceive` sits ABOVE `core` and BELOW `ff` / `io` / `conformer`.
///
/// The spine is `core → perceive → {io, ff} → conformer`. A `use crate::ff::…` inside
/// `perceive` would invert it — and would do so silently, because a cycle in the
/// module graph is legal Rust. The layer would still *work*; it would just no longer
/// be a layer, and the next thing to sink into `core` would drag a force field with it.
#[test]
fn perceive_depends_on_nothing_above_it() {
    let above = [
        concat!("crate", "::", "ff"),
        concat!("crate", "::", "io"),
        concat!("crate", "::", "conformer"),
        concat!("crate", "::", "compute"),
        concat!("molrs", "::", "ff"),
        concat!("molrs", "::", "io"),
        concat!("molrs", "::", "conformer"),
    ];
    let files = rust_files(&src_dir().join("perceive"));
    assert!(
        !files.is_empty(),
        "`src/perceive/` has no files — the layer this promise is about does not exist"
    );

    let mut hits = Vec::new();
    for needle in above {
        hits.extend(code_hits(&files, needle));
    }
    assert!(
        hits.is_empty(),
        "`perceive` reaches UP the dependency spine (core -> perceive -> {{io, ff}} -> \
         conformer):\n{}\nA layer that depends on what depends on it is not a layer.",
        render(&hits)
    );
}

// ===========================================================================
// PROMISE 3 — one interpolation seam
// ===========================================================================

/// `ParameterInterpolator` has exactly ONE implementor in `src`: `Parmchk2Estimator`.
///
/// This chain was already reworked once for precisely this sin: a second force field
/// arrived, and with it a second estimator stack — two ways to invent a missing
/// parameter, which is two sets of numbers to be wrong, drifting apart at the speed
/// of whichever one somebody last edited. **A different force field is a different
/// `TypifierParameterContext`, not a different estimator.**
///
/// The test tree is deliberately NOT scanned. `tests/ff/typifier/opls.rs` implements
/// the trait for a stub (`FixedEstimator`) precisely to prove the seam is generic —
/// that the interpolator is swappable is the *claim*; a test double is how you show
/// it, and it ships to nobody. A second implementor in `src/` is a second stack.
#[test]
fn there_is_exactly_one_interpolation_seam() {
    let needle = concat!("impl ", "ParameterInterpolator", " for ");
    let hits = code_hits(&rust_files(&src_dir()), needle);

    assert_eq!(
        hits.len(),
        1,
        "`ParameterInterpolator` must have exactly ONE implementor in `src/`; found \
         {}:\n{}\nA second estimator stack is a second set of invented parameters. \
         Swap the `TypifierParameterContext`, not the estimator.",
        hits.len(),
        render(&hits)
    );
    assert!(
        hits[0].2.contains("Parmchk2Estimator"),
        "the one implementor must be `Parmchk2Estimator`; found `{}` at {}:{}",
        hits[0].2,
        rel(&hits[0].0),
        hits[0].1
    );
}

// ===========================================================================
// PROMISE 4 — one MMFF path
// ===========================================================================

/// MMFF has no bespoke energy layer, no free-function assembler, no second
/// classifier — and **owns no kernel**.
///
/// Every one of these existed. The bespoke layer duplicated the seven term kernels
/// the generic path already had; the second classifier answered `1` where RDKit
/// answers `0` for an aromatic bond, and the tests passed because they asserted the
/// two doors *agreed with each other* — which they did, wrongly. And `mmff_ele` was a
/// whole electrostatic kernel MMFF owned privately, so the force field that was
/// supposed to *declare* its electrostatics could omit it entirely and still look
/// complete. It is now `pair/coul/cut`, parameterised.
///
/// A doc comment that says "the deleted `mmff_ele` kernel …" is history, not a
/// kernel; `code_only` blanks it. What this gate hunts is a live symbol.
#[test]
fn mmff_has_one_path_and_owns_no_kernel() {
    let src = rust_files(&src_dir());

    // A bespoke assembly layer, by any of its historical names.
    let assemblers = [
        concat!("build_mmff", "_potentials"),
        concat!("MmffForce", "Field"),
        concat!("MMFFForce", "Field"),
    ];
    let mut hits = Vec::new();
    for needle in assemblers {
        hits.extend(code_hits(&src, needle));
    }
    assert!(
        hits.is_empty(),
        "a bespoke MMFF energy layer is back:\n{}\nMMFF computes energies the way every \
         other force field does: typify -> Frame -> ForceField::to_potentials.",
        render(&hits)
    );

    // The MMFF-owned electrostatic kernel.
    let kernel = [
        concat!("mmff", "_ele", "_ctor"),
        concat!("MMFF", "Electrostatic"),
    ];
    let mut hits = Vec::new();
    for needle in kernel {
        hits.extend(code_hits(&src, needle));
    }
    assert!(
        hits.is_empty(),
        "MMFF owns an electrostatic kernel again:\n{}\nMMFF's electrostatics is \
         `pair/coul/cut` PARAMETERISED (k / D / delta come from the style). A kernel that \
         supplies the force field's own constants is not reading the force field — and a \
         force field that need not declare its electrostatics is one that can silently \
         omit them. That was the 150 kcal/mol hole in caffeine.",
        render(&hits)
    );

    // The deleted trees.
    for gone in [
        src_dir().join("ff/mmff/energy"),
        src_dir().join("ff/mmff/classify.rs"),
    ] {
        assert!(
            !gone.exists(),
            "{} is back — a second implementation of MMFF is a second set of numbers to \
             be wrong, and this one already answered the OPPOSITE of RDKit for an \
             aromatic bond while every test agreed with it",
            rel(&gone)
        );
    }
}

// ===========================================================================
// PROMISE 5 — a registered ctor that ignores `tp` is not a Style
// ===========================================================================

/// One kernel constructor: its name, the identifier it binds `tp` to, and whether
/// the BODY ever reads it.
#[derive(Debug)]
struct Ctor {
    name: String,
    file: PathBuf,
    line: usize,
    /// The body reads the type-params argument, whatever it is spelled.
    reads_type_params: bool,
}

/// Extract every `fn *_ctor(…)` under `src/ff/potential/`, and decide — **by usage,
/// not by spelling** — whether it reads its type-params argument.
///
/// This is the whole point of the gate. The previous criterion tested the *spelling*
/// of the binding (`_tp`, with Rust's leading underscore for "unused") and therefore
/// missed `pme_ctor` and `pair_coul_cut_ctor`, which spell it `_type_params` and
/// ignore it just as completely: 8 violators, not the 6 the grep found. Worse, the
/// spelling criterion cannot see the deliberate version at all — a ctor that binds
/// `type_params` and then never reads it. Usage can see all three.
fn kernel_ctors() -> Vec<Ctor> {
    let mut out = Vec::new();
    for path in rust_files(&src_dir().join("ff/potential")) {
        for item in fn_items(&path) {
            if !item.name.ends_with("_ctor") {
                continue;
            }
            // The type-params argument is the SECOND one — the signature is fixed by
            // `KernelConstructor`, so its position is a fact, not a convention.
            let args = split_top_level(&item.args);
            if args.len() < 2 {
                continue;
            }
            let binding = args[1]
                .split(':')
                .next()
                .unwrap_or("")
                .trim()
                .trim_start_matches("mut ")
                .trim()
                .to_owned();

            // Semantics: does the body READ the binding? A leading underscore is Rust's
            // *declaration* of intent — but a declaration is what the last gate believed,
            // and it was wrong twice. The fact is whether the identifier occurs at all.
            let reads = contains_word(&item.body, &binding);

            out.push(Ctor {
                name: item.name,
                file: path.clone(),
                line: item.line,
                reads_type_params: reads,
            });
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// One `fn` — its name, argument list, body, and 1-based line. Comments and string
/// literals are already blanked; what is left is what the compiler sees.
struct FnItem {
    name: String,
    args: String,
    body: String,
    line: usize,
}

/// Blank every `#[cfg(test)]` module body.
///
/// The gates that reason about *functions* reason about SHIPPED functions. An inline
/// unit test may register a dummy kernel (`my_ctor`) or parse a string fixture — both
/// are legitimate, neither ships, and counting them would make the registry gate
/// report a kernel no user can reach. (The gates that reason about *text* —
/// `include_str!`, `mmff_ele` — keep scanning the whole file: a forbidden symbol is
/// forbidden wherever it is compiled.)
fn strip_cfg_test(code: &str) -> String {
    let mut out = code.to_owned();
    let needle = concat!("#[cfg(", "test)]");
    while let Some(at) = out.find(needle) {
        let bytes = out.as_bytes();
        let Some(open) = bytes
            .iter()
            .skip(at)
            .position(|&b| b == b'{')
            .map(|p| p + at)
        else {
            break;
        };
        let mut depth = 0usize;
        let mut end = open;
        for (k, &b) in bytes.iter().enumerate().skip(open) {
            match b {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = k;
                        break;
                    }
                }
                _ => {}
            }
        }
        if end <= open {
            break;
        }
        let blanked: String = out[at..=end]
            .chars()
            .map(|c| if c == '\n' { '\n' } else { ' ' })
            .collect();
        out.replace_range(at..=end, &blanked);
    }
    out
}

/// Every `fn` in a file, with its argument list and its body.
///
/// Crude but honest: it balances `(` and `{`, which is all the gates need, and it
/// reads [`code_only`] output so a brace in a comment or a string cannot fool it.
/// Inline `#[cfg(test)]` modules are blanked — see [`strip_cfg_test`].
fn fn_items(path: &Path) -> Vec<FnItem> {
    let Ok(text) = fs::read_to_string(path) else {
        return Vec::new();
    };
    let code = strip_cfg_test(&code_only(&text));
    let bytes = code.as_bytes();
    let mut out = Vec::new();

    let balanced = |from: usize, open_b: u8, close_b: u8| -> Option<(usize, usize)> {
        let start = bytes.iter().skip(from).position(|&b| b == open_b)? + from;
        let mut depth = 0usize;
        for (k, &b) in bytes.iter().enumerate().skip(start) {
            if b == open_b {
                depth += 1;
            } else if b == close_b {
                depth -= 1;
                if depth == 0 {
                    return Some((start, k));
                }
            }
        }
        None
    };

    let mut at = 0usize;
    while let Some(found) = code[at..].find("fn ") {
        let kw = at + found;
        at = kw + 3;

        let rest = &code[at..];
        let name_len = rest
            .find(|c: char| !(c.is_alphanumeric() || c == '_'))
            .unwrap_or(0);
        if name_len == 0 {
            continue;
        }
        let name = rest[..name_len].to_owned();

        let Some((open, close)) = balanced(at + name_len, b'(', b')') else {
            continue;
        };
        // A trait method declaration (`fn f(&self);`) has no body.
        let Some((brace, end)) = balanced(close, b'{', b'}') else {
            continue;
        };
        if code[close..brace].contains(';') {
            continue;
        }

        out.push(FnItem {
            name,
            args: code[open + 1..close].to_owned(),
            body: code[brace + 1..end].to_owned(),
            line: code[..kw].bytes().filter(|&b| b == b'\n').count() + 1,
        });
    }
    out
}

/// Split an argument list on top-level commas (generics and tuples nest).
fn split_top_level(args: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let (mut depth, mut start) = (0i32, 0usize);
    for (i, c) in args.char_indices() {
        match c {
            '(' | '[' | '<' => depth += 1,
            ')' | ']' | '>' => depth -= 1,
            ',' if depth == 0 => {
                out.push(args[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    let tail = args[start..].trim();
    if !tail.is_empty() {
        out.push(tail);
    }
    out
}

/// One `r.register…(category, name, ctor[, ParamSource::X])` in `KernelRegistry::builtin`.
#[derive(Debug)]
struct Registration {
    category: String,
    style: String,
    ctor: String,
    per_instance: bool,
}

/// Every `r.register…(…)` call inside `KernelRegistry::builtin()` — and ONLY inside it.
///
/// Bounded to that one body on purpose. `register` / `register_with` are also
/// *declared* in this file (their own parameters are literally named `category`,
/// `name`, `ctor`), and the inline unit tests register kernels of their own. Parsing
/// either would have the gate reasoning about a registry no user ever gets.
fn registrations() -> Vec<Registration> {
    let path = src_dir().join("ff/potential/registry.rs");
    let text = fs::read_to_string(&path).expect("read registry.rs");

    let builtin = fn_items(&path)
        .into_iter()
        .find(|f| f.name == "builtin")
        .expect("registry.rs declares `builtin()`");

    // `fn_items` hands back the body with comments and strings blanked, so the
    // *offsets* are still the offsets of the original text. Find each call in the
    // blanked body (so a `.register` inside a comment cannot be seen), then read the
    // arguments back out of the ORIGINAL text at the same offsets.
    let code = strip_cfg_test(&code_only(&text));
    let body_at = code
        .find(&builtin.body)
        .expect("the body came from this text");
    let bytes = builtin.body.as_bytes();

    let mut out = Vec::new();
    let mut at = 0usize;
    while let Some(found) = builtin.body[at..].find(".register") {
        let call = at + found;
        let Some(rel_open) = builtin.body[call..].find('(') else {
            break;
        };
        let open = call + rel_open;
        let mut depth = 0usize;
        let mut close = open;
        for (k, &b) in bytes.iter().enumerate().skip(open) {
            match b {
                b'(' => depth += 1,
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        close = k;
                        break;
                    }
                }
                _ => {}
            }
        }
        at = close + 1;

        let raw = &text[body_at + open + 1..body_at + close];
        let args = split_top_level(raw);
        if args.len() < 3 {
            continue;
        }
        let unquote = |s: &str| s.trim().trim_matches('"').to_owned();
        out.push(Registration {
            category: unquote(args[0]),
            style: unquote(args[1]),
            ctor: args[2]
                .trim()
                .rsplit("::")
                .next()
                .unwrap_or("")
                .trim()
                .to_owned(),
            per_instance: args.get(3).is_some_and(|a| a.contains("PerInstance")),
        });
    }
    out
}

/// **A ctor ignores its type-params IF AND ONLY IF it is registered `PerInstance`.**
///
/// Both directions, because both lies are the same lie:
///
/// * *"I ignore `tp` but I registered as `TypeRows`"* — the MMFF bug. It forces dead
///   type rows into the parameter file to satisfy `Style::to_potential`'s
///   `is_empty()` guard (4,065 of them), and once the lie is in place nothing notices
///   that a whole style was never defined at all.
/// * *"I registered as `PerInstance` but I really do read `tp`"* — a style whose type
///   rows may now be silently empty. That is the *next* 150 kcal/mol hole, wearing
///   the fix as a disguise.
///
/// The verdict is by USAGE. `tests/ff/potential/param_source_gate.rs` asks the same
/// question of the *spelling* and says so in its own docs ("the scanner's bias is
/// towards false negatives … cannot see a ctor that binds `tp` and then never reads
/// it"). This one reads the body.
#[test]
fn param_source_is_bidirectional_on_semantics_not_spelling() {
    let ctors = kernel_ctors();
    let regs = registrations();

    assert!(
        ctors.len() >= 15,
        "found only {} kernel ctors — the parser is broken, and a broken parser makes \
         this gate vacuously green",
        ctors.len()
    );
    assert!(
        regs.len() >= 15,
        "found only {} registrations — see above",
        regs.len()
    );

    let by_name: BTreeMap<&str, &Ctor> = ctors.iter().map(|c| (c.name.as_str(), c)).collect();
    let mut wrong = Vec::new();

    for reg in &regs {
        let Some(ctor) = by_name.get(reg.ctor.as_str()) else {
            wrong.push(format!(
                "  {}/{} registers `{}`, which is not a ctor in `ff/potential/`",
                reg.category, reg.style, reg.ctor
            ));
            continue;
        };
        match (ctor.reads_type_params, reg.per_instance) {
            (false, false) => wrong.push(format!(
                "  {}/{}: `{}` ({}:{}) NEVER READS its type-params, but is registered \
                 TypeRows.\n      -> it is not a table-driven style. Register it \
                 `ParamSource::PerInstance`, or the empty-type-params guard has to be \
                 bribed with rows no code reads.",
                reg.category,
                reg.style,
                ctor.name,
                rel(&ctor.file),
                ctor.line
            )),
            (true, true) => wrong.push(format!(
                "  {}/{}: `{}` ({}:{}) DOES read its type-params, but is registered \
                 PerInstance.\n      -> its type rows may now be silently empty, and \
                 `Style::to_potential` will no longer refuse them.",
                reg.category,
                reg.style,
                ctor.name,
                rel(&ctor.file),
                ctor.line
            )),
            _ => {}
        }
    }

    assert!(
        wrong.is_empty(),
        "ParamSource lies ({} of them):\n{}\n\nA registered kernel constructor that \
         ignores `tp` is not a Style.",
        wrong.len(),
        wrong.join("\n")
    );

    // Every ctor in the tree is reachable from the registry: an unregistered kernel is
    // dead code that no gate above can see, and the place a ninth violator would hide.
    let registered: BTreeSet<&str> = regs.iter().map(|r| r.ctor.as_str()).collect();
    let orphans: Vec<String> = ctors
        .iter()
        .filter(|c| !registered.contains(c.name.as_str()))
        .map(|c| format!("  {} ({}:{})", c.name, rel(&c.file), c.line))
        .collect();
    assert!(
        orphans.is_empty(),
        "kernel ctors that no built-in registration reaches:\n{}\nAn unregistered ctor \
         is invisible to the gate above — which is exactly where the next one would sit.",
        orphans.join("\n")
    );
}

// ===========================================================================
// THE SUBSET GATE — the chain's most expensive lesson
// ===========================================================================

/// Every `[ … ]` list of string literals in a file, as `(line, names)`.
///
/// Balanced on the BLANKED text (so a bracket inside a comment or a string cannot
/// open a group), then the literals are read back out of the ORIGINAL text at the
/// same offsets — because the literals are exactly what was blanked, and exactly what
/// this gate is looking for.
fn string_lists(path: &Path) -> Vec<(usize, Vec<String>)> {
    let Ok(text) = fs::read_to_string(path) else {
        return Vec::new();
    };
    let code = code_only(&text);
    let bytes = code.as_bytes();
    let mut out = Vec::new();

    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] != b'[' {
            i += 1;
            continue;
        }
        let mut depth = 0usize;
        let mut end = i;
        for (k, &b) in bytes.iter().enumerate().skip(i) {
            match b {
                b'[' => depth += 1,
                b']' => {
                    depth -= 1;
                    if depth == 0 {
                        end = k;
                        break;
                    }
                }
                _ => {}
            }
        }
        if end <= i {
            break;
        }

        let raw = &text[i + 1..end];
        let quotes: Vec<usize> = raw.match_indices('"').map(|(q, _)| q).collect();
        let names: Vec<String> = quotes
            .chunks(2)
            .filter(|c| c.len() == 2)
            .map(|c| raw[c[0] + 1..c[1]].to_owned())
            .collect();
        if !names.is_empty() {
            let line = code[..i].bytes().filter(|&b| b == b'\n').count() + 1;
            out.push((line, names));
        }
        i = end + 1;
    }
    out
}

/// **No test may assert on a hand-picked subset of its fixtures.**
///
/// `generic_path_total_energy_matches_rdkit` asserted on `["e_ethane"]` — one of
/// exactly TWO fixtures whose MMFF charges are all zero, i.e. the one input class
/// that *structurally cannot* expose a missing electrostatic term. The other ten sat
/// on disk, unread, for a month, behind a comment blaming the wrong cause. The bug
/// was 150 kcal/mol on caffeine and the suite was green.
///
/// So: where a fixture list can be directory-scanned, it MUST be. A hardcoded list of
/// fixture names is allowed only if it is **complete** — i.e. it can prove none was
/// deleted, but it cannot select. A subset assertion should not be something you
/// justify; it should be something you cannot write.
///
/// "Not yet implemented" is not a reason to exclude a fixture. It is a reason to fail.
#[test]
fn no_test_asserts_on_a_subset_of_its_fixtures() {
    // A fixture's FAMILY is the set of companion files it carries.
    //
    // `e_ethane` carries {.sdf, .energy.json, .breakdown.json} — an energy fixture.
    // `benzene` carries {.sdf, .json} — a typing fixture. `embed_alanine_r` carries
    // {.sdf} alone. Grouping by a single suffix instead would fuse all three into one
    // "*.sdf family" of 21, and then every honest test in the tree looks like a subset
    // assertion — and a gate that cries wolf is a gate that gets an allowlist.
    //
    // The signature is a FACT on disk, not a judgement: a fixture that carries an
    // `mmff94s` reference is an MMFF94s fixture, whatever anyone names it.
    let mut families: BTreeMap<(PathBuf, String), BTreeSet<String>> = BTreeMap::new();
    for dir in dirs(&tests_dir()) {
        if dir.file_name().and_then(|n| n.to_str()) != Some("fixtures") {
            continue;
        }
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        let mut suffixes: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let Some(dot) = name.find('.') else { continue };
            let (stem, suffix) = name.split_at(dot);
            suffixes
                .entry(stem.to_owned())
                .or_default()
                .insert(suffix.to_owned());
        }
        for (stem, sig) in suffixes {
            let key = sig.into_iter().collect::<Vec<_>>().join("+");
            families.entry((dir.clone(), key)).or_default().insert(stem);
        }
    }
    assert!(
        !families.is_empty(),
        "no fixture families found under {} — a scan that finds nothing makes this gate \
         vacuously green, which is the very defect it is here to catch",
        rel(&tests_dir())
    );

    let mut sins = Vec::new();

    for path in rust_files(&tests_dir()) {
        // Every `[ … ]` in the file: a list literal, however many LINES it spans.
        //
        // A per-line scan would only ever have caught the historical `["e_ethane"]`
        // because it happened to be written on one line. `const REQUIRED_FIXTURES:
        // [&str; 11] = [\n  "e_acetonitrile",\n  …` is the same list with newlines in
        // it, and a gate that a line break defeats is not a gate.
        for (line, names) in string_lists(&path) {
            if names.len() < 2 {
                continue; // one name is a PROBE of one molecule, not a subset of a loop
            }
            for ((dir, sig), stems) in &families {
                // A list is a SELECTION over a family only if every name in it is a
                // member of that family. Otherwise the list is about something else and
                // the overlap is a coincidence of chemistry: `core/aromaticity.rs` names
                // benzene and pyridine (and furan, which is no fixture at all), and the
                // antechamber oracle names 37 molecules, five of which happen to share a
                // name with an MMFF fixture. Flagging those is how a gate earns an
                // allowlist — and an allowlist is where the next defect sits down.
                if !names.iter().all(|n| stems.contains(n)) {
                    continue;
                }
                let named: BTreeSet<&String> =
                    names.iter().filter(|l| stems.contains(*l)).collect();
                if named.len() < 2 || named.len() >= stems.len() {
                    continue;
                }
                let missing: Vec<&str> = stems
                    .iter()
                    .filter(|s| !named.iter().any(|n| *n == *s))
                    .map(String::as_str)
                    .collect();
                sins.push(format!(
                    "  {}:{}: names {} of the {} fixtures in {} that carry `{sig}`, \
                     omitting {missing:?}\n      -> scan the directory and select with a \
                     PREDICATE (the reference field the fixture carries, the chemistry the \
                     molecule has). A list you can write by hand is a list you can shorten \
                     by hand — and the one that mattered had been shortened to ONE.",
                    rel(&path),
                    line,
                    named.len(),
                    stems.len(),
                    rel(dir),
                ));
            }
        }
    }

    assert!(
        sins.is_empty(),
        "tests that assert on a HAND-PICKED SUBSET of their fixtures:\n{}\n\n\
         This is the single pattern behind the worst defect on this chain.",
        sins.join("\n")
    );
}

/// No test SKIPS a fixture by name.
///
/// The other shape of the same sin. A completeness list can be defeated by iterating
/// all of it and then `filter`ing one out — the loop still *looks* exhaustive. A
/// `.find(…)` naming one molecule is fine (that is a probe of one property); a
/// `filter` or a `continue` that drops a case from a loop asserting a GENERAL property
/// is a subset assertion with the evidence swept behind it.
#[test]
fn no_test_skips_a_fixture_by_name() {
    let mut sins = Vec::new();
    for path in rust_files(&tests_dir()) {
        let Ok(text) = fs::read_to_string(&path) else {
            continue;
        };
        let code = code_only(&text);
        for (n, (raw, line)) in text.lines().zip(code.lines()).enumerate() {
            // The comparison must be against a hardcoded NAME. `c.name == r.ctor`
            // compares two things the test computed — that is a join, not a skip list,
            // and `param_source_gate.rs` uses exactly that shape to hunt orphans. The
            // sin is a name the author TYPED. So the right-hand side must be a literal:
            // read it from the raw line, since `code_only` has blanked the literals out.
            let compares_a_name =
                line.contains(concat!(".name ", "==")) || line.contains(concat!(".name ", "!="));
            let against_a_literal = raw.contains(concat!(".name ", "== \""))
                || raw.contains(concat!(".name ", "!= \""));
            if !(compares_a_name && against_a_literal) {
                continue;
            }
            // Is it a filter/skip, or a probe? `.find(…)` picks one molecule to examine
            // a property of; `.filter(…)` drops one from a loop that asserts a general
            // one, and the loop still looks exhaustive afterwards.
            let filtered = line.contains(concat!(".filter", "("))
                || line.contains(concat!(".retain", "("))
                || line.contains(concat!(".skip_while", "("))
                || line.contains("continue");
            if filtered {
                sins.push(format!("  {}:{}: {}", rel(&path), n + 1, raw.trim()));
            }
        }
    }
    assert!(
        sins.is_empty(),
        "tests that drop a fixture from an assertion loop by NAME:\n{}\n\n\
         A loop that skips a case is a subset assertion that looks exhaustive. If a \
         molecule cannot pass, that is a failure, not an exclusion.",
        sins.join("\n")
    );
}

/// "Not yet implemented" is not a reason to exclude a fixture — it is a reason to fail.
///
/// The comment next to `["e_ethane"]` blamed "stretch-bend + torsion eq-fallback label
/// resolution". It was false: stbn and torsion agreed to five decimals on every
/// fixture. The comment misdirected every reader for a month, which is worse than no
/// comment — a wrong reason is an *alibi*, and an alibi is how a defect gets to stay.
///
/// So the excuse is only a sin when it is doing an excuse's WORK: sitting next to a
/// thing that skips. A doc comment that *forbids* the excuse (like this one, or
/// `energy.rs`'s "no test may take a subset of this list: 'not implemented yet' is a
/// reason to fail") is prose about the rule, not an instance of it — and a gate that
/// could not tell those apart would forbid its own justification.
#[test]
fn no_fixture_is_excluded_for_being_unimplemented() {
    let excuses = [
        concat!("not yet ", "implemented"),
        concat!("not ", "implemented yet"),
        concat!("un", "implemented"),
        concat!("TO", "DO"),
        concat!("FIX", "ME"),
    ];
    // The constructs an excuse would be excusing.
    let exclusions = [
        concat!("#[", "ignore"),
        concat!("return", ";"),
        concat!("continue", ";"),
        concat!(".filter", "("),
    ];
    const WINDOW: usize = 4;

    let mut sins = Vec::new();
    for path in rust_files(&tests_dir()) {
        let Ok(text) = fs::read_to_string(&path) else {
            continue;
        };
        let lines: Vec<&str> = text.lines().collect();
        let code = code_only(&text);
        let code_lines: Vec<&str> = code.lines().collect();

        for (n, line) in lines.iter().enumerate() {
            let lower = line.to_lowercase();
            if !excuses.iter().any(|e| lower.contains(&e.to_lowercase())) {
                continue;
            }
            let lo = n.saturating_sub(WINDOW);
            let hi = (n + WINDOW + 1).min(code_lines.len());
            let excuses_something = code_lines[lo..hi]
                .iter()
                .any(|l| exclusions.iter().any(|x| l.contains(x)));
            if excuses_something {
                sins.push(format!("  {}:{}: {}", rel(&path), n + 1, line.trim()));
            }
        }
    }
    assert!(
        sins.is_empty(),
        "an excuse next to the thing it excuses:\n{}\n\nA fixture that cannot pass is a \
         FAILING test, not an excluded one. The comment that justified `[\"e_ethane\"]` \
         was not merely an excuse — it named the WRONG CAUSE, and every reader believed \
         it for a month.",
        sins.join("\n")
    );
}

/// No test passes VACUOUSLY when its input is missing.
///
/// The last shape of the same disease: a test that returns early — green — because
/// the file it needed was not there. It reports nothing, it asserts nothing, and it
/// counts as coverage. `tests/ff/tables_gate.rs` already says it out loud: *"CI must
/// have ZERO coupling to AmberTools. Delete the test — do not make it skip."*
#[test]
fn no_test_returns_green_when_its_input_is_absent() {
    let mut sins = Vec::new();
    for path in rust_files(&tests_dir()) {
        let Ok(text) = fs::read_to_string(&path) else {
            continue;
        };
        let lines: Vec<&str> = text.lines().collect();
        let code = code_only(&text);
        let code_lines: Vec<&str> = code.lines().collect();
        for (n, line) in code_lines.iter().enumerate() {
            // `eprintln!("skipping: …"); return;` — the shape, whatever the words.
            if !line.contains(concat!("print", "ln!")) {
                continue;
            }
            let says_skip = lines[n].to_lowercase().contains("skip");
            let returns = code_lines
                .get(n + 1..n + 3)
                .is_some_and(|w| w.iter().any(|l| l.trim() == "return;"));
            if says_skip && returns {
                sins.push(format!("  {}:{}: {}", rel(&path), n + 1, lines[n].trim()));
            }
        }
    }
    assert!(
        sins.is_empty(),
        "tests that go GREEN by skipping themselves when their input is absent:\n{}\n\n\
         A test that asserts nothing is not coverage — it is a green light with no lamp \
         behind it. Make the input unconditional, or delete the test.",
        sins.join("\n")
    );
}
