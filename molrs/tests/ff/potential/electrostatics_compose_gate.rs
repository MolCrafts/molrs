//! mmff-ele-compose ac-001 / ac-004 — MMFF owns no kernel, and no constant.
//!
//! `mmff-orthogonal-02` stopped MMFF being a snowflake at the **ForceField** layer.
//! This spec pushes the same principle one level down: **the KERNEL layer must not
//! have an MMFF-specific kernel either.** MMFF's electrostatics is a *buffered
//! Coulomb* — a parameterization of the generic kernel, not a kernel of its own
//! (the physics is asserted in `coul_buffered.rs`; this file asserts the *absence*).
//!
//! Two removals, and they are different in kind:
//!
//! * **ac-001 — the kernel.** `MMFFElectrostatic` / `mmff_ele_ctor` / the
//!   `pair/mmff_ele` registration. `coul_cut.rs`'s own module doc said its kernel
//!   was "per-atom, **mirroring `mmff_ele`**": two implementations of one physics,
//!   and the author knew.
//!
//! * **ac-004 — the constants.** `COULOMB_MMFF` (332.0716) and `ELE_BUFFER`
//!   (0.05 Å) do not belong in `ff/constants.rs`. **A constant in `ff/constants.rs`
//!   claims to be a property of the universe; these are a property of one force
//!   field.** They move into `MMFF_ELE_STYLE` in `ff/params/mmff.rs`, where MMFF's
//!   other parameters live — and from there into the style, which is the only way a
//!   kernel is allowed to learn them.
//!
//! # Two kinds of assertion, because there are two kinds of failure
//!
//! The **registry** half is a runtime fact: a style whose kernel is not registered
//! cannot be built. The **source** half is not observable at runtime at all — a
//! dead-but-registered kernel constructs perfectly and returns plausible energies,
//! which is precisely how `mmff_ele_ctor` survived from the day it was written while
//! no ForceField ever defined its style (the 150 kcal/mol caffeine hole). So the
//! source is what gets read, with the same comment-stripping scanner the house
//! already uses in `mmff/deletion_gate.rs` and `param_source_gate.rs`.
//!
//! # Scanner bias (deliberate, and the same as the house's)
//!
//! `//` line comments are stripped before scanning, so *prose about* a deleted
//! symbol survives ("the deleted `mmff_ele` kernel …" in a doc comment is not a
//! regression) while any surviving **code** reference — a registration, a `pub use`,
//! a style-name string literal, a `const` — is a failure. Every real hit of the old
//! kernel is code: the registry entry, the `pub use` in `pair/mod.rs`, the `"mmff_ele"`
//! style name in `ff/params/mmff.rs` and in the XML reader, the `pub(crate) const` in
//! `ff/constants.rs`.
//!
//! The needles are assembled with `concat!` so this file's own source does not
//! literally contain them — the house convention, and what lets the gate scan
//! honestly with **no self-exemption by path**.
//!
//! # The over-reach guard
//!
//! `MMFFVdW` is **not** deleted, and `the_vdw_kernel_is_not_swept_into_the_deletion`
//! says so. vdW genuinely IS a per-atom-type lookup — 95 rows, and `mmff_vdw_ctor`
//! opens by indexing `tp`. A deletion that swept it in with the electrostatics would
//! take out a table that is actually read.

use std::path::{Path, PathBuf};

use molrs::ff::potential::{ParamSource, lookup_kernel, lookup_param_source};

// ---------------------------------------------------------------------------
// Scanning
// ---------------------------------------------------------------------------

fn src_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
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

/// Every `.rs` file under `molrs/src`, comment-stripped.
fn scan_src() -> Vec<(PathBuf, String)> {
    fn walk(dir: &Path, out: &mut Vec<(PathBuf, String)>) {
        for entry in std::fs::read_dir(dir).expect("read a source dir") {
            let path = entry.expect("dir entry").path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
                let text = std::fs::read_to_string(&path).expect("read a source file");
                out.push((path, strip_line_comments(&text)));
            }
        }
    }
    let mut out = Vec::new();
    walk(&src_dir(), &mut out);
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// `path:line: text` for every occurrence of `needle` in the scanned tree.
fn hits(sources: &[(PathBuf, String)], needle: &str) -> Vec<String> {
    let root = src_dir();
    sources
        .iter()
        .flat_map(|(path, text)| {
            let rel = path
                .strip_prefix(&root)
                .unwrap_or(path)
                .display()
                .to_string();
            text.lines()
                .enumerate()
                .filter(|(_, line)| line.contains(needle))
                .map(|(n, line)| format!("src/{rel}:{}: {}", n + 1, line.trim()))
                .collect::<Vec<_>>()
        })
        .collect()
}

/// A scan that reads nothing reports no violations.
fn scanned_source() -> Vec<(PathBuf, String)> {
    let sources = scan_src();
    assert!(
        sources.len() > 200,
        "the source scan found only {} files under {} — a scan that reads nothing would make \
         every deletion gate below vacuously green",
        sources.len(),
        src_dir().display()
    );
    assert!(
        !hits(&sources, "pair_coul_cut_ctor").is_empty(),
        "the scanner cannot see `pair_coul_cut_ctor`, which certainly exists — it is reading the \
         wrong tree, or stripping too much"
    );
    sources
}

// ---------------------------------------------------------------------------
// ac-001 — pair/mmff_ele is gone
// ---------------------------------------------------------------------------

/// `pair/mmff_ele` is not in the kernel registry, and `pair/coul/cut` is.
///
/// The runtime half of ac-001. A registered kernel is a kernel any force field can
/// reach by name: as long as `pair/mmff_ele` is registered, MMFF owns a kernel.
#[test]
fn the_mmff_electrostatic_kernel_is_not_in_the_registry() {
    let dead_style = concat!("mmff", "_ele");

    assert!(
        lookup_kernel("pair", dead_style).is_none(),
        "`pair/{dead_style}` is still a registered kernel. MMFF's electrostatics is a BUFFERED \
         COULOMB — `E = k·qᵢqⱼ / (D·(r + δ))` — which is `pair/coul/cut` at k = 332.0716, D = 1, \
         δ = 0.05. It is a parameterization of the generic kernel, not a kernel of its own, and \
         the owner's ruling is that it must be composed, not written."
    );
    assert!(
        lookup_param_source("pair", dead_style).is_none(),
        "`pair/{dead_style}` still carries a ParamSource registration"
    );

    // Non-vacuity: the kernel it composes into must exist, and must be honest about
    // where its parameters come from (charges are per-atom Frame data, never type rows).
    assert!(
        lookup_kernel("pair", "coul/cut").is_some(),
        "`pair/coul/cut` is not registered — the assertion above would be vacuously true, and \
         MMFF would have no electrostatics at all"
    );
    assert_eq!(
        lookup_param_source("pair", "coul/cut"),
        Some(ParamSource::PerInstance),
        "`pair/coul/cut` must stay `PerInstance`: its charges are per-atom columns on the Frame, \
         and it reads no type rows"
    );
}

/// The MMFF electrostatic kernel is gone from `molrs/src` — every code reference.
///
/// The source half of ac-001. Registration, `pub use`, the struct, the ctor, and the
/// `"mmff_ele"` style-name string wherever a force field or a table still spells it.
#[test]
fn the_mmff_electrostatic_kernel_is_gone_from_the_source() {
    let sources = scanned_source();

    let mut fails = Vec::new();
    for needle in [concat!("mmff", "_ele"), concat!("MMFF", "Electrostatic")] {
        let found = hits(&sources, needle);
        if !found.is_empty() {
            fails.push(format!(
                "`{needle}` still appears in {} place(s):\n    {}",
                found.len(),
                found.join("\n    ")
            ));
        }
    }

    assert!(
        fails.is_empty(),
        "MMFF still owns an electrostatic kernel:\n  {}\n\n\
         MMFF's electrostatics must be `pair/coul/cut` parameterised by the force field \
         (k = 332.0716, D = 1.0, δ = 0.05). Note the XML reader for `<ElectrostaticParams>` is \
         NOT deleted — it must survive under a name that does not carry the dead style's \
         (e.g. `parse_electrostatics`) and must define `coul/cut`; `mmff/deletion_gate.rs` \
         asserts it still exists.",
        fails.join("\n  ")
    );
}

// ---------------------------------------------------------------------------
// ac-004 — Halgren's constants are MMFF's parameters, not molrs's constants
// ---------------------------------------------------------------------------

/// `COULOMB_MMFF` / `ELE_BUFFER` are gone from `molrs/src`.
///
/// A constant in `ff/constants.rs` claims to be a property of the universe. 332.0716
/// is a property of ONE force field — Halgren chose it; CODATA says 332.06371, and
/// `coul/cut` used that. Both are correct. A shared constant cannot represent that,
/// and a kernel holding either one is a kernel holding data nobody handed it.
#[test]
fn halgrens_constants_are_not_molrs_constants() {
    let sources = scanned_source();

    let mut fails = Vec::new();
    for needle in [concat!("COULOMB", "_MMFF"), concat!("ELE", "_BUFFER")] {
        let found = hits(&sources, needle);
        if !found.is_empty() {
            fails.push(format!(
                "`{needle}` still appears in {} place(s):\n    {}",
                found.len(),
                found.join("\n    ")
            ));
        }
    }
    assert!(
        fails.is_empty(),
        "MMFF's electrostatic constants are still molrs constants:\n  {}\n\n\
         They belong in `MMFF_ELE_STYLE` in `ff/params/mmff.rs`, next to MMFF's other \
         parameters, and reach the kernel through the style — the same way `dielectric` and \
         `delta` already do.",
        fails.join("\n  ")
    );

    // The reverse: the constants that ARE universal must not be swept out with them.
    let constants = std::fs::read_to_string(src_dir().join("ff/constants.rs"))
        .expect("ff/constants.rs must still exist — MDYNE_A_TO_KCAL and DEG2RAD live there");
    for survivor in ["MDYNE_A_TO_KCAL", "DEG2RAD"] {
        assert!(
            constants.contains(survivor),
            "`{survivor}` is gone from `ff/constants.rs`. The deletion is of the two constants \
             that were a force field's property, not of the file"
        );
    }
}

/// MMFF's electrostatic constants live in MMFF's TABLE, and the typifier reads them
/// from there rather than re-typing them.
///
/// The positive half of ac-004: it is not enough that `ff/constants.rs` lost them —
/// they must land in `ff/params/mmff.rs`, which is where `MMFF_VDW_STYLE`'s
/// combining-rule constants already live. (Comments are stripped, so the `332.0716`
/// already present in `MmffEleStyle`'s doc comment does not satisfy this: only a real
/// field does.)
#[test]
fn mmffs_electrostatic_constants_live_in_mmffs_table() {
    let table = strip_line_comments(
        &std::fs::read_to_string(src_dir().join("ff/params/mmff.rs"))
            .expect("read ff/params/mmff.rs"),
    );
    assert!(
        table.contains("MMFF_ELE_STYLE"),
        "`ff/params/mmff.rs` has no `MMFF_ELE_STYLE` — MMFF's electrostatic data block"
    );
    assert!(
        table.contains("332.0716"),
        "`ff/params/mmff.rs` does not carry Halgren's Coulomb constant (332.0716) as DATA. It is \
         MMFF's parameter — it must sit in `MMFF_ELE_STYLE` beside `dielectric` and `delta`, not \
         in `ff/constants.rs` (where it claims to be a property of the universe) and not inside a \
         kernel (where it is data nobody handed it)."
    );

    let embedded = strip_line_comments(
        &std::fs::read_to_string(src_dir().join("ff/typifier/mmff/embedded.rs"))
            .expect("read ff/typifier/mmff/embedded.rs"),
    );
    assert!(
        !embedded.contains("332.0716"),
        "`ff/typifier/mmff/embedded.rs` spells the Coulomb constant as a literal. It must read it \
         from `MMFF_ELE_STYLE` — that file's own doc says \"Nothing here is a number: every value \
         comes from the table\", and a second copy of a constant is how the two values drifted \
         apart in the first place."
    );
}

// ---------------------------------------------------------------------------
// The over-reach guard — vdW is a REAL type table and stays
// ---------------------------------------------------------------------------

/// `MMFFVdW` survives. (GREEN today, and it must stay green.)
///
/// The electrostatic kernel goes because it reads **no type rows** (`_tp`) and is a
/// generic Coulomb wearing a force field's name. vdW is the opposite: 95 rows, one per
/// MMFF atom type, and `mmff_vdw_ctor` opens by building a `HashMap<&str, &Params>`
/// from `tp`. It is a genuine `TypeRows` style. A deletion list that pattern-matched on
/// "MMFF" would take it, and `Style::to_potential` would then refuse to build it at all.
#[test]
fn the_vdw_kernel_is_not_swept_into_the_deletion() {
    let sources = scanned_source();
    for needle in ["MMFFVdW", "mmff_vdw_ctor"] {
        assert!(
            !hits(&sources, needle).is_empty(),
            "`{needle}` is gone from `molrs/src`. MMFF's vdW is a REAL per-atom-type lookup (95 \
             rows, read from `tp`) — it is not the snowflake this spec deletes, and sweeping it in \
             would remove a table that is actually read"
        );
    }
    assert!(
        lookup_kernel("pair", "mmff_vdw").is_some(),
        "`pair/mmff_vdw` is no longer registered — the deletion over-reached"
    );
    assert_eq!(
        lookup_param_source("pair", "mmff_vdw"),
        Some(ParamSource::TypeRows),
        "`pair/mmff_vdw` must stay `TypeRows`: if it ever became `PerInstance`, its 95 type rows \
         could be deleted without any test noticing"
    );
}
