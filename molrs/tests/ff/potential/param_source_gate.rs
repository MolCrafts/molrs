//! ac-002 — **a registered kernel constructor that ignores `tp` is not a Style.**
//!
//! MMFF's bond / angle / stretch-bend / torsion / out-of-plane / electrostatic
//! parameters are resolved *per instance* by the typifier (aromaticity, ring
//! size, four-level equivalence degradation, and — on a table miss — empirical
//! rules invented from covalent radii) and baked into Frame columns. They are
//! not, and cannot be, a `(type_i, type_j, …) → params` table. The kernels read
//! those columns and ignore their `tp` argument entirely.
//!
//! That is *correct*. The bug was that MMFF registered itself as a style that
//! **uses** type rows anyway, so `Style::to_potential`'s
//! `type_params.is_empty()` guard had to be bribed with 4,065 rows of XML that
//! no code ever reads — and once the lie was in place, nothing noticed that the
//! electrostatic style was never defined at all. That was the 150 kcal/mol hole
//! on caffeine.
//!
//! So `ParamSource` names the thing, and this gate makes the naming enforceable:
//!
//! > **ctor ignores its type-params  ⟺  registered `ParamSource::PerInstance`**
//!
//! Asserted in BOTH directions, because both lies are the same lie:
//!   * "I ignore `tp` but I registered as `TypeRows`" — the MMFF bug, which
//!     forces dead type rows into the parameter file to satisfy a guard.
//!   * "I registered as `PerInstance` but I really do read `tp`" — a style whose
//!     type rows may now be silently empty, which is the *next* 150 kcal/mol
//!     hole wearing the fix as a disguise.
//!
//! # Why a source-level gate
//!
//! Neither direction is observable at runtime. A kernel that ignores `tp`
//! constructs perfectly and returns plausible energies; that is exactly why the
//! defect survived. The claim is about what the *source declares*, so the source
//! is what gets read — the same reasoning (and the same crude, comment-stripping
//! scanner) as `tests/ff/typifier/source_gate.rs`.
//!
//! # There is no exemption list in this file
//!
//! Deliberately. The moment a gate grows an allowlist, the next tp-ignoring
//! kernel is one line of allowlist away from invisible. A kernel that genuinely
//! resolves nothing from `tp` says so by being registered `PerInstance` — that
//! is the entire point, and it costs one enum variant.
//!
//! The scanner's bias is towards **false negatives**: it reads parameter *names*
//! (`_tp` / `_type_params` — the leading underscore is Rust's own "I ignore
//! this") and cannot see a ctor that binds `tp` and then never reads it. That
//! shape is a deliberate lie, not an accident, and no cheap scanner catches it.
//! What this gate does catch is the accident.

use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Source scanning
// ---------------------------------------------------------------------------

fn potential_src_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src/ff/potential")
}

fn registry_path() -> PathBuf {
    potential_src_dir().join("registry.rs")
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

/// Everything before the inline `#[cfg(test)]` module.
///
/// `registry.rs`'s own unit tests call `r.register("pair", "lj/cut", …)`; without
/// this cut the gate would parse those as production registrations and reason
/// about a registry that does not exist outside `cargo test`.
fn strip_test_module(src: &str) -> String {
    match src
        .lines()
        .position(|line| line.trim_start().starts_with("#[cfg(test)]"))
    {
        Some(at) => src.lines().take(at).collect::<Vec<_>>().join("\n"),
        None => src.to_owned(),
    }
}

/// Every `.rs` under `src/ff/potential/`, recursively, comment-stripped.
///
/// The acceptance criterion words the scan as `potential/*/*.rs` (the kernel
/// families live one directory down). This walks the **whole** tree instead,
/// which is a strict superset: a tp-ignoring kernel must not become invisible by
/// moving up one directory.
fn potential_sources() -> Vec<(PathBuf, String)> {
    fn walk(dir: &Path, out: &mut Vec<(PathBuf, String)>) {
        for entry in std::fs::read_dir(dir).expect("read the potential source tree") {
            let path = entry.expect("dir entry").path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
                let src = std::fs::read_to_string(&path).expect("read a potential source file");
                out.push((path, strip_line_comments(&src)));
            }
        }
    }
    let mut out = Vec::new();
    walk(&potential_src_dir(), &mut out);
    assert!(
        !out.is_empty(),
        "no .rs found under {} — the gate would pass vacuously",
        potential_src_dir().display()
    );
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

// ---------------------------------------------------------------------------
// The two things being compared
// ---------------------------------------------------------------------------

/// A kernel constructor as *declared* in source.
#[derive(Debug, Clone)]
struct Ctor {
    /// e.g. `mmff_bond_ctor`
    name: String,
    /// `src/ff/potential/bond/mmff.rs`
    file: String,
    /// The second parameter (the `&[(&str, &Params)]` type-params slice) is
    /// bound with a leading underscore, i.e. the kernel resolves nothing from it.
    ignores_tp: bool,
    /// The declared name of that second parameter, for the failure message.
    tp_param: String,
}

/// A registration as *declared* in `registry.rs`.
#[derive(Debug, Clone)]
struct Registration {
    category: String,
    style: String,
    ctor: String,
    /// The registration names `ParamSource::PerInstance`.
    ///
    /// A plain three-argument `register(cat, name, ctor)` is the documented thin
    /// wrapper for `TypeRows`, so "names neither" reads as `TypeRows` — which is
    /// exactly today's state, and exactly what makes this gate RED today.
    per_instance: bool,
}

/// The text between a signature's opening `(` and its matching `)`.
///
/// Depth-counted, because the middle parameter's own type contains parentheses:
/// `&[(&str, &Params)]`.
fn signature_args(src: &str, open_paren: usize) -> Option<&str> {
    let bytes = src.as_bytes();
    let mut depth = 0usize;
    for (i, &b) in bytes.iter().enumerate().skip(open_paren) {
        match b {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&src[open_paren + 1..i]);
                }
            }
            _ => {}
        }
    }
    None
}

/// Split an argument list on its **top-level** commas.
fn split_top_level(args: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let (mut depth, mut start) = (0usize, 0usize);
    for (i, c) in args.char_indices() {
        match c {
            '(' | '[' | '<' => depth += 1,
            ')' | ']' | '>' => depth = depth.saturating_sub(1),
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

/// Every `pub fn *_ctor` in the potential tree, with its second parameter read.
fn scan_ctors() -> Vec<Ctor> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut out = Vec::new();
    for (path, src) in potential_sources() {
        let file = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");

        let mut cursor = 0usize;
        while let Some(found) = src[cursor..].find("pub fn ") {
            let start = cursor + found + "pub fn ".len();
            cursor = start;

            let Some(paren_off) = src[start..].find('(') else {
                break;
            };
            let paren = start + paren_off;
            let name = src[start..paren].trim();
            if !name.ends_with("_ctor") || name.contains(char::is_whitespace) {
                continue;
            }

            let args = signature_args(&src, paren)
                .unwrap_or_else(|| panic!("{file}: unbalanced parens in `{name}`"));
            let params = split_top_level(args);
            assert_eq!(
                params.len(),
                3,
                "{file}: `{name}` has {} parameters, not 3 — every kernel constructor must \
                 match `KernelConstructor = fn(&Params, &[(&str, &Params)], &Frame)`, and a \
                 signature this gate cannot read is a signature it cannot police",
                params.len()
            );

            let tp_param = params[1]
                .split(':')
                .next()
                .unwrap_or_default()
                .trim()
                .to_owned();
            out.push(Ctor {
                ignores_tp: tp_param.starts_with('_'),
                name: name.to_owned(),
                file: file.clone(),
                tp_param,
            });
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// Every kernel registration in `registry.rs`'s production code.
///
/// Parsed from source rather than from the built registry because
/// `KernelRegistry` cannot be asked "which `ParamSource` did this style declare,
/// and which ctor function was it" — and the pairing of *those two facts* is the
/// whole invariant.
///
/// The parse is deliberately method-name-agnostic: it keys off the string
/// literals and the `*_ctor` path, not off `register` vs `register_with` vs
/// whatever the implementation calls the `ParamSource`-carrying entry point. A
/// gate that hardcoded the method name would go green the day someone added a
/// second one.
fn scan_registrations() -> Vec<Registration> {
    let src = std::fs::read_to_string(registry_path()).expect("read registry.rs");
    let src = strip_test_module(&strip_line_comments(&src));

    let mut out = Vec::new();
    for stmt in src.split(';') {
        if !stmt.contains(".register") {
            continue;
        }
        // String literals, in order: ("category", "style_name", ...).
        let literals: Vec<&str> = stmt
            .match_indices('"')
            .collect::<Vec<_>>()
            .chunks(2)
            .filter(|c| c.len() == 2)
            .map(|c| &stmt[c[0].0 + 1..c[1].0])
            .collect();
        // The ctor: the last path segment ending in `_ctor`.
        let ctor = stmt
            .split(|c: char| !(c.is_alphanumeric() || c == '_'))
            .rfind(|t| t.ends_with("_ctor"));
        let (Some(ctor), true) = (ctor, literals.len() >= 2) else {
            // `pub fn register_kernel(category, name, ctor)` forwards its
            // arguments — no literals, no ctor path. Not a registration.
            continue;
        };
        out.push(Registration {
            category: literals[0].to_owned(),
            style: literals[1].to_owned(),
            ctor: ctor.to_owned(),
            per_instance: stmt.contains("PerInstance"),
        });
    }
    out
}

// ---------------------------------------------------------------------------
// Non-vacuity — asserted first, because every claim below rests on it
// ---------------------------------------------------------------------------

/// A gate that scans nothing passes everything.
///
/// Floors, not exact counts: today the tree declares 28 constructors and the
/// registry seeds 29 entries (`dihedral/periodic` is registered twice, as
/// `periodic` and `fourier`). Pinning the exact numbers would turn every new
/// kernel into a failing test for no reason; pinning a floor catches the failure
/// mode that matters — a scanner that silently stopped finding anything.
#[test]
fn the_gate_reads_a_real_registry_and_a_real_kernel_tree() {
    let ctors = scan_ctors();
    let regs = scan_registrations();

    assert!(
        ctors.len() >= 25,
        "scanned only {} kernel constructors under {} — the scanner is broken, and a broken \
         scanner reports no violations",
        ctors.len(),
        potential_src_dir().display()
    );
    assert!(
        regs.len() >= 25,
        "parsed only {} registrations from {} — the scanner is broken",
        regs.len(),
        registry_path().display()
    );

    // Both classes must be present, or the biconditional is only half-tested.
    let ignoring = ctors.iter().filter(|c| c.ignores_tp).count();
    let using = ctors.len() - ignoring;
    assert!(
        ignoring > 0,
        "no constructor in the tree ignores its type-params. If that is genuinely true, this \
         gate has nothing to say — but MMFF's six per-instance kernels and `pme` / `coul/cut` \
         all do, so it means the scanner stopped reading parameter names"
    );
    assert!(
        using > 0,
        "every constructor in the tree ignores its type-params — the scanner is misreading \
         parameter names, because `lj/cut`, `harmonic`, `opls` and `mmff_vdw` all resolve \
         from `tp`"
    );

    // Every registered ctor must be one the scanner actually found, or the two
    // halves of the invariant are describing different populations.
    let unknown: Vec<String> = regs
        .iter()
        .filter(|r| !ctors.iter().any(|c| c.name == r.ctor))
        .map(|r| format!("{}/{} -> {}", r.category, r.style, r.ctor))
        .collect();
    assert!(
        unknown.is_empty(),
        "registry.rs registers constructors this gate never scanned: {unknown:?}"
    );
}

// ---------------------------------------------------------------------------
// ac-002 — the invariant, both directions
// ---------------------------------------------------------------------------

/// Direction 1: **ignores `tp`  ⟹  registered `PerInstance`.**
///
/// This is the MMFF bug in general form. A kernel that resolves nothing from its
/// type rows, registered as a `TypeRows` style, has to be handed type rows it
/// will not read — so the parameter file grows dead data whose only job is to
/// satisfy a guard, and the guard stops meaning anything.
///
/// Today this is RED for **eight** constructors, not six. The six MMFF kernels
/// spell the ignored parameter `_tp`; `pme_ctor` and `pair_coul_cut_ctor` spell
/// it `_type_params` — but they ignore it just as completely (both read per-atom
/// `charge` from the Frame, which is per-instance data by construction). The
/// spec's `grep -l '_tp: &\[(&str, &Params)\]'` returns exactly the five MMFF
/// files because it greps a *spelling*; the invariant is about the *semantics*,
/// so both of them are in scope and both must be registered `PerInstance`.
#[test]
fn every_ctor_that_ignores_type_params_is_registered_per_instance() {
    let ctors = scan_ctors();
    let regs = scan_registrations();

    let mut fails = Vec::new();
    for reg in &regs {
        let Some(ctor) = ctors.iter().find(|c| c.name == reg.ctor) else {
            continue; // covered by the non-vacuity test
        };
        if ctor.ignores_tp && !reg.per_instance {
            fails.push(format!(
                "{}/{} -> {} ({}): binds its type-params as `{}` (ignored) but is registered \
                 TypeRows",
                reg.category, reg.style, ctor.name, ctor.file, ctor.tp_param
            ));
        }
    }

    assert!(
        fails.is_empty(),
        "a registered kernel constructor that ignores `tp` is not a Style.\n\
         These {} styles resolve nothing from their type rows, yet claim to be table-driven — \
         so `Style::to_potential`'s empty-type-params guard must be fed rows that no code \
         reads:\n  {}\n\n\
         Fix: register them `ParamSource::PerInstance` (their parameters are baked into Frame \
         columns by the typifier), and let the guard consult the ParamSource instead of \
         demanding dead rows.",
        fails.len(),
        fails.join("\n  ")
    );
}

/// Direction 2: **registered `PerInstance`  ⟹  ignores `tp`.**
///
/// The reverse lie, and the more dangerous one after this spec lands. A style
/// registered `PerInstance` is exempted from the "has type definitions" check —
/// so if its kernel actually *does* resolve from `tp`, its type rows may now be
/// silently empty and every parameter it looks up silently missing. That is the
/// same failure as the 150 kcal/mol electrostatic hole, wearing the fix as a
/// disguise.
///
/// `pair/mmff_vdw` is the live example of the boundary: MMFF's van der Waals
/// parameters genuinely ARE a 95-row per-atom-type table (`mmff_vdw_ctor` builds
/// a `HashMap<&str, &Params>` from `tp` on its first line), so it stays
/// `TypeRows` while the other six MMFF styles do not.
#[test]
fn every_per_instance_registration_has_a_ctor_that_ignores_type_params() {
    let ctors = scan_ctors();
    let regs = scan_registrations();

    let mut fails = Vec::new();
    for reg in &regs {
        let Some(ctor) = ctors.iter().find(|c| c.name == reg.ctor) else {
            continue;
        };
        if reg.per_instance && !ctor.ignores_tp {
            fails.push(format!(
                "{}/{} -> {} ({}): registered PerInstance but binds its type-params as `{}` \
                 (used)",
                reg.category, reg.style, ctor.name, ctor.file, ctor.tp_param
            ));
        }
    }

    assert!(
        fails.is_empty(),
        "these styles are registered PerInstance — which exempts them from the \
         empty-type-params guard — but their kernels really do read `tp`:\n  {}\n\n\
         A PerInstance style may be handed ZERO type rows. A kernel that resolves parameters \
         from those rows will therefore resolve nothing, silently. Either the kernel should \
         read its parameters from Frame columns (then rename the parameter `_tp`), or the \
         style is TypeRows.",
        fails.join("\n  ")
    );
}

/// Every constructor in the tree is reachable through the registry.
///
/// `mmff_ele_ctor` was in the registry from the day it was written while **no
/// ForceField ever defined the style** — a kernel nobody can reach is an energy
/// term that does not exist. The mirror image (a ctor nobody registers) is the
/// same dead weight, and it is also how a tp-ignoring kernel could hide from the
/// two tests above: they iterate *registrations*, so an unregistered ctor is
/// invisible to them. This is the assertion that closes that door.
#[test]
fn every_kernel_ctor_is_registered() {
    let ctors = scan_ctors();
    let regs = scan_registrations();

    let orphans: Vec<String> = ctors
        .iter()
        .filter(|c| !regs.iter().any(|r| r.ctor == c.name))
        .map(|c| format!("{} ({})", c.name, c.file))
        .collect();

    assert!(
        orphans.is_empty(),
        "these kernel constructors are declared but never registered, so no ForceField can \
         reach them — and neither can the ParamSource invariant above:\n  {}",
        orphans.join("\n  ")
    );
}
