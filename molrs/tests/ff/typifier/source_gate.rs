//! Source-level gates over `molrs/src/ff/typifier/`.
//!
//! Two acceptance criteria of `chem-perceive-05` are structural, not behavioural:
//! *no runtime table parsing on the typify path* (ac-003) and *no constructor
//! that yields an empty parameter table* (ac-004). Both are properties of what
//! the source may CONTAIN, and both are invisible to a behavioural test — an
//! engine that re-parses its table on every call still returns the right types,
//! and a typifier built with an empty table still constructs. So they are read
//! off the source, exactly as the acceptance contract words them.
//!
//! The scanner is deliberately crude (line-based, line-comments stripped). Its
//! bias is towards false negatives: stripping text can only remove a match, never
//! invent one, so it can miss a violation but cannot fail an innocent file.

use std::path::{Path, PathBuf};

/// The typifier source tree.
pub fn typifier_src_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src/ff/typifier")
}

/// The charge-model source tree — where `BccModel` and the push API live.
///
/// The ac-004 gate has to follow the model here. `AM1BCCTypifier` used to be the
/// parameter-table-owning model under `src/ff/typifier/`, and it is now `BccModel`
/// under `src/ff/charge/`; a gate still scoped to the tree the model has LEFT is a
/// gate that passes because it is looking at nothing.
pub fn charge_src_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src/ff/charge")
}

/// Every `.rs` under the typifier tree, with line comments stripped.
///
/// Comments are stripped so that a doc-comment *mentioning* a forbidden form
/// (this file's own prose ends up in rustdoc, and so does the engine's) does not
/// trip the gate. Block comments are not stripped: the tree uses none, and a
/// naive state machine would be more likely to be wrong than useful.
pub fn typifier_sources() -> Vec<(PathBuf, String)> {
    let mut out = Vec::new();
    collect(&typifier_src_dir(), &mut out);
    assert!(
        !out.is_empty(),
        "no .rs found under {} — the gate would pass vacuously",
        typifier_src_dir().display()
    );
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// Every `.rs` of the charge-model tree, with line comments stripped.
///
/// Accepts both shapes the module can have — the `charge.rs` single file it was,
/// and the `charge/` directory the models split it into — because a gate that only
/// knew one of them would read an empty tree across the change and report no
/// violations from a source it never opened.
pub fn charge_sources() -> Vec<(PathBuf, String)> {
    let dir = charge_src_dir();
    let mut out = Vec::new();
    if dir.is_dir() {
        collect(&dir, &mut out);
    } else {
        let file = dir.with_extension("rs");
        if file.is_file() {
            let src = std::fs::read_to_string(&file).expect("read the charge source file");
            out.push((file, strip_line_comments(&src)));
        }
    }
    assert!(
        !out.is_empty(),
        "no charge source found at {} (or {}.rs) — the gate would pass vacuously",
        dir.display(),
        dir.display()
    );
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// Every source that may own a parameter table: the typifier tree AND the charge
/// tree.
///
/// The ac-004 footgun is "a model you can construct without naming its parameter
/// set", and it does not care which module the model sits in. Scoping the gate to
/// one tree would let the next `Default` land in the other.
pub fn parameterized_model_sources() -> Vec<(PathBuf, String)> {
    let mut out = typifier_sources();
    out.extend(charge_sources());
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// The whole force-field tree — `src/ff/`, with line comments stripped.
///
/// Wider than [`typifier_sources`] because chem-perceive-10's criterion is wider:
/// the estimator's tables were the LAST runtime text parse on the force-field path,
/// and "last" is a claim about `ff/`, not about `ff/typifier/`. A gate scoped to the
/// typifier would go green the moment the parse moved one directory up.
pub fn ff_sources() -> Vec<(PathBuf, String)> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/ff");
    let mut out = Vec::new();
    collect(&dir, &mut out);
    assert!(
        !out.is_empty(),
        "no .rs found under {} — the gate would pass vacuously",
        dir.display()
    );
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// One source file of the force-field tree, by path relative to `src/ff/`, with
/// its line comments AND its inline `#[cfg(test)]` module stripped.
///
/// The estimator's table module is named directly by the acceptance criterion, so
/// the gate on it reads it directly rather than filtering a tree scan — a filter
/// that matched nothing would pass silently, which is the failure mode this whole
/// file exists to avoid.
///
/// The test module is cut because these gates are about what the module *does at
/// runtime*, and a unit test's `.expect("os present")` is neither a runtime panic
/// nor a table parse. Without the cut the `.expect(` gate would demand the module's
/// own tests be rewritten to satisfy a criterion about its production path — a false
/// positive that would train the next reader to weaken the gate.
pub fn ff_source(relative: &str) -> (PathBuf, String) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/ff")
        .join(relative);
    let src =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    (path, strip_test_module(&strip_line_comments(&src)))
}

/// Everything before the inline `#[cfg(test)]` module.
///
/// Line-based, like the rest of this scanner: `#[cfg(test)]` on a source of its own
/// line is how every module in the tree opens its tests, and a form it does not
/// recognise leaves MORE text to match, never less — so the bias stays towards
/// false negatives.
fn strip_test_module(src: &str) -> String {
    match src
        .lines()
        .position(|line| line.trim_start().starts_with("#[cfg(test)]"))
    {
        Some(at) => src.lines().take(at).collect::<Vec<_>>().join("\n"),
        None => src.to_owned(),
    }
}

/// The ATD engine tree — `src/ff/typifier/atd/`, with line comments stripped.
///
/// Narrower than [`typifier_sources`] on purpose: the *engine* is what must stay
/// table-generic. Its callers may legitimately name a parameter set (`am1bcc.rs`
/// maps `BccParameterSet::Bcc` to `AtdParameterSet::Bcc`), so scoping a
/// "no per-set branch" gate at the whole typifier tree would fail on code that is
/// doing exactly the right thing.
pub fn atd_engine_sources() -> Vec<(PathBuf, String)> {
    let engine = typifier_src_dir().join("atd");
    let mut out = Vec::new();
    collect(&engine, &mut out);
    assert!(
        !out.is_empty(),
        "no .rs found under {} — the gate would pass vacuously",
        engine.display()
    );
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

fn collect(dir: &Path, out: &mut Vec<(PathBuf, String)>) {
    for entry in std::fs::read_dir(dir).expect("read the typifier source tree") {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            collect(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            let src = std::fs::read_to_string(&path).expect("read a typifier source file");
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

/// `file:line: <text>` for every line of the typifier tree containing `needle`.
pub fn lines_containing(needle: &str) -> Vec<String> {
    lines_containing_in(&typifier_sources(), needle)
}

/// `file:line: <text>` for every line of `sources` containing `needle`.
pub fn lines_containing_in(sources: &[(PathBuf, String)], needle: &str) -> Vec<String> {
    sources
        .iter()
        .flat_map(|(path, src)| {
            src.lines()
                .enumerate()
                .filter(|(_, line)| line.contains(needle))
                .map(|(n, line)| format!("{}:{}: {}", path.display(), n + 1, line.trim()))
                .collect::<Vec<_>>()
        })
        .collect()
}

/// The model types whose construction must name a parameter set.
///
/// Each of these owns (or wraps) a parameter table. Constructed without one, the
/// object exists, type-checks and then fails on every single bond with
/// `missing BCC correction` — the "constructible but non-functional" shape
/// molrs's no-fallback-values rule forbids.
///
/// `BccModel` replaces `AM1BCCTypifier` here: same object (the model that owns a
/// BCCPARM table), new module and new seam. The gate follows the model, not the
/// name — otherwise deleting the typifier would silently retire the criterion.
const PARAMETERIZED_MODELS: [&str; 3] = ["BCCCorrectionTable", "BCCCorrector", "BccModel"];

/// Every `Default` a [`PARAMETERIZED_MODELS`] type gets, derived or hand-written.
///
/// `Default` is the purest form of the footgun: a parameter table conjured out of
/// nothing. There is no default set of bond charge corrections — there is
/// `BCCPARM.DAT` and there is `BCCPARM_ABCG2.DAT`, and the caller must say which.
///
/// Both spellings are caught, because they are the same bug:
///   * `#[derive(.., Default)]` on the struct (today: `BCCCorrectionTable`);
///   * `impl Default for X` (today: `BCCCorrector`, `AM1BCCTypifier`).
pub fn default_impls_of_parameterized_models() -> Vec<String> {
    let mut found = Vec::new();
    for (path, src) in parameterized_model_sources() {
        let mut derives_default = false;
        for (n, line) in src.lines().enumerate() {
            let text = line.trim();
            let at = |what: &str| format!("{}:{}: {what}", path.display(), n + 1);

            // `impl Default for BCCCorrector { .. }`
            if let Some(target) = text.strip_prefix("impl Default for ")
                && PARAMETERIZED_MODELS.iter().any(|m| target.contains(m))
            {
                found.push(at(text));
            }

            // `#[derive(Debug, Clone, Default)]` followed by `pub struct BCCCorrectionTable`
            if text.starts_with("#[derive(") {
                derives_default = text.contains("Default");
                continue;
            }
            if derives_default
                && let Some(name) = struct_name(text)
                && PARAMETERIZED_MODELS.iter().any(|m| name.starts_with(m))
            {
                found.push(at(&format!("#[derive(.., Default)] on {name}")));
            }
            if !text.is_empty() && !text.starts_with('#') {
                derives_default = false;
            }
        }
    }
    found
}

/// The declared name of `struct X ..` / `pub struct X ..`, if this line is one.
fn struct_name(text: &str) -> Option<&str> {
    let rest = text
        .strip_prefix("pub struct ")
        .or_else(|| text.strip_prefix("struct "))
        .or_else(|| text.strip_prefix("pub(crate) struct "))?;
    Some(
        rest.split(['<', ' ', '{', '(', ';'])
            .next()
            .unwrap_or_default(),
    )
}

/// A `new` that hands back one of [`PARAMETERIZED_MODELS`] with no parameter set,
/// no table, and no `Result` to say it could not build one.
///
/// Allowed forms, i.e. what a fixed constructor looks like:
///   * `fn new(set: BccParameterSet, ..) -> Self`  — names the parameter set;
///   * `fn new(table: BCCCorrectionTable) -> Self` — takes a table someone else
///     built from a parameter set (this is `BCCCorrector::new`, which is fine);
///   * `fn new(..) -> Result<Self, _>`             — may refuse to build.
///
/// Forbidden: `fn new() -> Self` and `fn new(backend: B) -> Self`, which is
/// exactly what `BCCCorrectionTable::new` and `AM1BCCTypifier::new` used to be.
pub fn parameterless_constructors() -> Vec<String> {
    let mut found = Vec::new();
    for (path, src) in parameterized_model_sources() {
        let mut current_impl: Option<String> = None;
        for (n, line) in src.lines().enumerate() {
            let text = line.trim();
            if text.starts_with("impl ") || text.starts_with("impl<") {
                current_impl = Some(text.to_owned());
            }
            let guarded = current_impl
                .as_deref()
                .is_some_and(|header| PARAMETERIZED_MODELS.iter().any(|m| header.contains(m)));
            if !guarded || !text.contains("fn new(") {
                continue;
            }

            let args = arguments_of(text);
            let names_a_parameter_set = args.contains("ParameterSet") || args.contains("Table");
            let can_refuse = text.contains("-> Result<");
            if !names_a_parameter_set && !can_refuse {
                found.push(format!("{}:{}: {text}", path.display(), n + 1));
            }
        }
    }
    found
}

/// The text between the first `(` and the last `)` of a signature line.
fn arguments_of(signature: &str) -> &str {
    let Some(open) = signature.find('(') else {
        return "";
    };
    let Some(close) = signature.rfind(')') else {
        return "";
    };
    if close <= open {
        return "";
    }
    &signature[open + 1..close]
}
