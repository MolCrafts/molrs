//! ac-002 — the XML → Rust conversion is a **pure representation change**: not one
//! number may move.
//!
//! # Why the existing suites are not enough
//!
//! The spec's testing strategy says "the existing MMFF94 / OPLS typifier and potential
//! tests stay green with no assertion value changed". They will — and that proves much
//! less than it sounds like. Those suites assert on a few dozen molecules and a handful
//! of energies. What the three XMLs actually hold, **counted from the current files on
//! 2026-07-14** (not from the spec body, which predates `mmff-orthogonal-01`/`-02`):
//!
//! | | mmff94 | mmff94s | oplsaa |
//! |---|---|---|---|
//! | dump entries | 199 | 199 | 4,727 |
//! | of which | 95 vdW pair rows, 95 atom-property rows, 7 style headers, `special_bonds` | identical rows; only the force-field **name** differs | 813 atom, 300 bond, 932 angle, 1,048 dihedral, 813 pair, 813 SMARTS typing rows |
//!
//! **5,125 entries.** A generator that dropped one column, or rounded one value to six
//! significant figures, or silently skipped the rows whose `def` attribute is absent,
//! would sail straight through caffeine and benzene and land in someone's simulation
//! eighteen months later. The existing tests cover what they cover; they are not a
//! conversion contract.
//!
//! # Why the MMFF half is 199 lines and not 4,358 — and why there is only ONE MMFF table
//!
//! It **was** 4,358 when this test was first written. `mmff-orthogonal-02` then deleted
//! 4,065 type-def rows from each MMFF XML: bond / angle / stretch-bend / torsion / oop
//! rows that no kernel ever read, which existed only to satisfy an `is_empty()` guard.
//! The MMFF energy parameters live — and always lived — in the RDKit `Params.cpp` port
//! (`ff/mmff/tables.rs`, 51,621 lines), which the typifier's resolver reads and bakes
//! into Frame columns per interaction (`ParamSource::PerInstance`).
//!
//! That left `mmff94.xml` and `mmff94s.xml` differing by **two lines** — the
//! `<ForceField name=…>` attribute. So `mmff94s.xml` is deleted without being converted,
//! and both front doors read **one shared table**, `ff/params/mmff.rs`: the port merged
//! with the 199 entries the XML still carries. Emitting two byte-identical 199-entry
//! tables that differ only in a name string would be exactly the dead weight this whole
//! chain exists to remove. `MMFF94Typifier` and `MMFF94STypifier` now differ by precisely
//! two things — a name string and an `MmffVariant` — and both live in the type system
//! rather than in duplicated data.
//!
//! So chem-perceive-14 handles the two halves differently, and the tests follow:
//!
//! * the **port** is *moved and merged*, not converted —
//!   `tables_gate::the_rdkit_mmff_port_is_merged_into_ff_params` hashes each of its 17
//!   static tables (51,222 lines: every number it holds) so the merge cannot touch a
//!   value;
//! * the **XML** is *converted* — and what is left of it to convert is these 199 entries,
//!   which this file freezes field-for-field.
//!
//! This is why the reference fixtures had to be re-frozen from the current XML rather
//! than reused. Re-blessing the old 4,358-line dump would have compiled the 4,065 dead
//! rows into committed, test-protected Rust tables — resurrecting, behind a hash guard,
//! exactly the dead weight the orthogonal chain had just removed.
//!
//! Both MMFF fixtures are kept, and both are still worth their bytes: `mmff94s` pins that
//! `MMFF94STypifier` names its own force field, which — with one shared table — is the
//! only thing about it a row-comparing test can see. See
//! [`one_shared_mmff_table_and_the_two_front_doors_differ_only_in_name`] for the half
//! this file is blind to, and who covers it.
//!
//! What survives in the MMFF XML is small but not negligible, and **none** of it is
//! reachable from an energy assertion: the 95 vdW rows (`a_i`, `alpha`, `n_eff`, `g_i`,
//! `da`), the 95 nine-column atom-property rows that drive typing, and — newest and
//! therefore likeliest for a generator written against the old shape to miss entirely —
//! the `<ElectrostaticParams>` section `mmff-orthogonal-01` added, which lands as the
//! `mmff_ele` style's `dielectric` / `delta` and as `special_bonds.coul[2] = 0.75`.
//!
//! # The contract
//!
//! For each of the three parameter sets: dump **every field of every entry** the runtime
//! XML parser produces, canonically and at full precision, and freeze it. The compiled
//! table must reproduce that dump **exactly** — byte for byte, not to a tolerance.
//!
//! Zero tolerance is the correct tolerance here, and it is the one place in this repo
//! where that is true. Nothing is being *computed*: the same IEEE-754 doubles that the
//! XML parser produced today must come out of the compiled table tomorrow. An `f64` is
//! printed with `{:?}`, which is Rust's shortest round-tripping form — two doubles print
//! the same string if and only if they are the same double — so string equality of the
//! dump *is* bit equality of the parameters. A relative tolerance, even 1e-12, would
//! quietly accept a table that had been emitted with `%.12g`.
//!
//! # How this catches a one-field drift in a 5,000-row table
//!
//! The fixture is one line per entry, carrying every field of it. A generator bug shows
//! up as a localised line diff, and the failure names the entry *and* the field:
//!
//! ```text
//! oplsaa: the compiled table does not reproduce the XML parse.
//!   1 of 4,727 lines differ:
//!   line 2,914:
//!     want: style[0001] angle harmonic angle[00417] name=… itom=CT jtom=CT ktom=OH :: k=0.9302 theta0=1.9740849710721253
//!     got : style[0001] angle harmonic angle[00417] name=… itom=CT jtom=CT ktom=OH :: k=0.9302 theta0=1.974085
//! ```
//!
//! — read directly: "the generator wrote `theta0` with six decimals, and only this row
//! is off". One row in 4,727, one field in that row, one digit in that field. No energy
//! test in the suite would have moved by more than 1e-7 kcal/mol.
//!
//! Every failure mode of a table generator lands here as a diff rather than as a
//! plausible-looking number:
//!
//! | generator bug | how it shows |
//! |---|---|
//! | a column dropped | the `key=` term is gone from every entry line of that style |
//! | a value re-rounded | the value text differs, at the digit where it was truncated |
//! | rows skipped (empty attribute, comment row, duplicate key) | entry lines missing, and the line counts differ |
//! | degrees emitted where the reader normalised to radians | every angle value differs by 57.3× |
//! | entry order changed | the `[NNNNN]` index moves — and `get_bondtype` takes the *first* match, so order is behaviour |
//! | a numeric `1.0` emitted as the string `"1.0"` | numbers dump as `k=1.0`, strings as `k="1.0"` |
//! | a `-0.0` normalised to `0.0` | `{:?}` prints them differently |
//! | typing metadata dropped (`sbmb`, `overrides`, SMARTS `def`) | its own line per type — and no energy test can see these at all |
//!
//! # Provenance of the fixtures
//!
//! `fixtures/tables/*.reference.txt` were produced **from the XML, by the runtime XML
//! parser, before it was deleted** — that is what makes them a valid reference rather
//! than a restatement of the generator's own output. Each carries the source path and
//! the source SHA-256 in its header, and the same hashes are pinned in
//! `tables_gate::the_manifest_records_the_xml_source_hashes`.
//!
//! They are deliberately **not re-blessable**: regenerating one needs
//! `molrs/data/*.xml`, which ac-001 deletes and `tables_gate` then keeps deleted. A
//! future maintainer cannot quietly re-bless a drift away — they would have to restore
//! the XML, and that shows up in the diff. (If a fixture ever genuinely must be rebuilt,
//! `git show <rev>:molrs/data/mmff94.xml` and the dump functions below are all it takes.)

use std::path::{Path, PathBuf};

use molrs::ff::forcefield::{ForceField, StyleDefs};
use molrs::ff::typifier::mmff::{MMFF94STypifier, MMFF94Typifier, MMFFParams};
use molrs::ff::typifier::opls::OPLSAATypifier;
use molrs::ff::typifier::opls::meta::OplsTypingMeta;

// ---------------------------------------------------------------------------
// The canonical dump
// ---------------------------------------------------------------------------

/// Exact, round-tripping decimal form of an `f64`.
///
/// `{:?}` on `f64` is the shortest decimal string that parses back to the same bits, so
/// this is a lossless rendering, not a formatted one. `1e-4` stays `0.0001`, `-0.0`
/// stays `-0.0`. Do not "tidy" this into `{:.12}` — that is exactly the mis-rounding the
/// test exists to catch, moved into the test.
fn exact(v: f64) -> String {
    format!("{v:?}")
}

/// Every numeric and string parameter of one entry, key-sorted, on one line.
///
/// `Params` is backed by a `HashMap`, so iteration order is not stable across runs —
/// sorting the keys is what makes the dump reproducible at all. (The *entries* are not
/// sorted; see [`dump_force_field`].)
///
/// One line per entry rather than one per field, because the identity prefix
/// (`style[0002] dihedral opls dihedral[00100] name=… itom=… jtom=… ktom=… ltom=…`) is
/// ~75 characters and would be repeated on every one of an entry's parameter lines,
/// several-fold inflating a fixture committed to a spec whose entire purpose is deleting
/// redundant copies of the same numbers. Nothing is lost: the diff still names the entry
/// and shows every field of it, so a one-field drift is still read straight off the
/// failure.
///
/// Numbers are `key=<exact>`; strings are `key="…"`. The two are spelled differently so
/// that a numeric `1.0` and the string `"1.0"` — which a sloppy generator could easily
/// confuse, since XML has no types — do not dump identically.
fn params_suffix(params: &molrs::ff::forcefield::Params) -> String {
    let mut parts = Vec::new();

    let mut numeric: Vec<(&str, f64)> = params.iter().collect();
    numeric.sort_by(|a, b| a.0.cmp(b.0));
    parts.extend(
        numeric
            .into_iter()
            .map(|(k, v)| format!("{k}={}", exact(v))),
    );

    let mut strings: Vec<(&str, &str)> = params.iter_strings().collect();
    strings.sort_by(|a, b| a.0.cmp(b.0));
    parts.extend(strings.into_iter().map(|(k, v)| format!("{k}={v:?}")));

    if parts.is_empty() {
        "::".to_owned()
    } else {
        format!(":: {}", parts.join(" "))
    }
}

/// Every style, every type definition, every field — in **file order**.
///
/// Entry order is preserved (and pinned, via the `[NNNNN]` index) because it is
/// load-bearing: `ForceField::get_bondtype` and friends scan and take the first match,
/// so a generator that emitted the same rows in a different order would be a different
/// force field for any lookup with more than one candidate. Sorting the dump would hide
/// precisely that.
///
/// The identity line (`name=… itom=… jtom=…`) is emitted for every entry even when it
/// carries no params, so that a dropped entry is visible as a missing line rather than
/// as nothing at all.
fn dump_force_field(ff: &ForceField) -> Vec<String> {
    let mut out = Vec::new();
    out.push(format!("forcefield name={}", ff.name));

    let special = ff.special_bonds();
    out.push(format!(
        "forcefield special_bonds lj=[{}] coul=[{}]",
        special
            .lj
            .iter()
            .copied()
            .map(exact)
            .collect::<Vec<_>>()
            .join(", "),
        special
            .coul
            .iter()
            .copied()
            .map(exact)
            .collect::<Vec<_>>()
            .join(", "),
    ));

    for (si, style) in ff.styles().iter().enumerate() {
        let head = format!("style[{si:04}] {} {}", style.category(), style.name);
        out.push(format!("{head} @style {}", params_suffix(&style.params)));

        match &style.defs {
            StyleDefs::Atom(types) => out.extend(types.iter().enumerate().map(|(ti, t)| {
                format!(
                    "{head} atom[{ti:05}] name={} {}",
                    t.name,
                    params_suffix(&t.params)
                )
            })),
            StyleDefs::Bond(types) => out.extend(types.iter().enumerate().map(|(ti, t)| {
                format!(
                    "{head} bond[{ti:05}] name={} itom={} jtom={} {}",
                    t.name,
                    t.itom,
                    t.jtom,
                    params_suffix(&t.params)
                )
            })),
            StyleDefs::Angle(types) => out.extend(types.iter().enumerate().map(|(ti, t)| {
                format!(
                    "{head} angle[{ti:05}] name={} itom={} jtom={} ktom={} {}",
                    t.name,
                    t.itom,
                    t.jtom,
                    t.ktom,
                    params_suffix(&t.params)
                )
            })),
            StyleDefs::Dihedral(types) => out.extend(types.iter().enumerate().map(|(ti, t)| {
                format!(
                    "{head} dihedral[{ti:05}] name={} itom={} jtom={} ktom={} ltom={} {}",
                    t.name,
                    t.itom,
                    t.jtom,
                    t.ktom,
                    t.ltom,
                    params_suffix(&t.params)
                )
            })),
            StyleDefs::Improper(types) => out.extend(types.iter().enumerate().map(|(ti, t)| {
                format!(
                    "{head} improper[{ti:05}] name={} itom={} jtom={} ktom={} ltom={} {}",
                    t.name,
                    t.itom,
                    t.jtom,
                    t.ktom,
                    t.ltom,
                    params_suffix(&t.params)
                )
            })),
            StyleDefs::Pair(types) => out.extend(types.iter().enumerate().map(|(ti, t)| {
                format!(
                    "{head} pair[{ti:05}] name={} itom={} jtom={} {}",
                    t.name,
                    t.itom,
                    t.jtom,
                    params_suffix(&t.params)
                )
            })),
            StyleDefs::KSpace => out.push(format!("{head} kspace (no per-type defs)")),
        }
    }

    out
}

/// The MMFF typing metadata — the half of the XML that is **not** the force field.
///
/// `MMFFParams` is the `<AtomProperties>` table: nine integer columns per MMFF type id.
/// It never appears in an energy, so no potential test can see it, and every one of its
/// columns changes what gets typed:
///
/// * `sbmb` selects the stretch-bend parameter row;
/// * `arom` / `pilp` / `mltb` drive aromatic and delocalised bond classification;
/// * `crd` / `val` gate the type assignment itself.
///
/// A generator that dropped `sbmb` would produce a force field that still parses, still
/// runs, and quietly picks the wrong stretch-bend row for every conjugated system. This
/// dump is the only thing in the suite that would notice.
///
/// `MMFFParams` exposes lookup, not iteration, so the type-id space is swept. MMFF types
/// run 1..=99; the sweep goes well past that so that a generator which invented an
/// out-of-range id is caught rather than ignored.
fn dump_mmff_props(params: &MMFFParams) -> Vec<String> {
    let mut out = Vec::new();
    for type_id in 0..=1024u32 {
        let Some(p) = params.get_prop(type_id) else {
            continue;
        };
        out.push(format!(
            "mmff prop type_id={} atno={} crd={} val={} pilp={} mltb={} arom={} linh={} sbmb={}",
            p.type_id, p.atno, p.crd, p.val, p.pilp, p.mltb, p.arom, p.linh, p.sbmb
        ));
    }
    out
}

/// The OPLS typing metadata — class, SMARTS def, overrides, priority, layer.
///
/// Also invisible to every energy test, and also load-bearing. `def` is the SMARTS
/// pattern the typifier matches on; `overrides` and `priority` decide *which* of several
/// matching types wins; `layer` is what lets CL&P / CL&Pol overlay OPLS at all. A
/// conversion that dropped `overrides` would leave every energy assertion in the suite
/// intact and re-type half of any molecule with a functional group.
///
/// Sorted by type name: `OplsTypingMeta` is a `HashMap`, so it has no file order to
/// preserve (unlike the `ForceField` entry lists, whose order decides lookups).
fn dump_opls_meta(meta: &OplsTypingMeta) -> Vec<String> {
    let mut rows: Vec<_> = meta.iter().collect();
    rows.sort_by(|a, b| a.0.cmp(b.0));
    rows.iter()
        .map(|(name, row)| {
            format!(
                "opls row name={name} class={} def={:?} overrides=[{}] priority={:?} layer={}",
                row.class,
                row.def,
                row.overrides.join(","),
                row.priority,
                row.layer
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------
// The three shipped parameter sets
// ---------------------------------------------------------------------------
//
// Each is reached through its **public front door**, never through the data behind it.
// That is what makes this file survive the conversion untouched: today the doors parse
// `molrs::data::*_XML` at runtime, tomorrow they read a compiled table, and the test
// cannot tell — which is exactly the property ac-002 asserts.

/// The MMFF94 parameter set exactly as molrs ships it.
fn shipped_mmff94() -> Vec<String> {
    let t = MMFF94Typifier::new();
    let mut lines = dump_force_field(t.ff());
    lines.extend(dump_mmff_props(t.params()));
    lines
}

/// The MMFF94s parameter set exactly as molrs ships it.
///
/// `MMFF94STypifier` is the reader `mmff-typifier-split` gave this parameter set, and
/// it is why chem-perceive-14 could land at all. Before that split, `mmff94s.xml` had
/// **no consumer**: nothing named `MMFF94S_XML`, and MMFF94s *energy* was served from
/// the RDKit port in `ff/mmff/tables.rs`. Converting it then would have produced a typed
/// table, hashed in the manifest and byte-reproduced by the generator, and read by
/// nobody — dead weight of exactly the kind the owner twice refused (and deleted, in the
/// case of `data/gen3d/`). So the spec was resequenced behind the split, and
/// `tables_gate::no_parameter_table_is_unreachable` now keeps every table honest.
///
/// Its rows are now identical to MMFF94's — see
/// [`the_two_mmff_parameter_sets_differ_only_in_their_name`], which is what stops this
/// front door from being wired to the wrong table without anyone noticing.
fn shipped_mmff94s() -> Vec<String> {
    let t = MMFF94STypifier::new();
    let mut lines = dump_force_field(t.ff());
    lines.extend(dump_mmff_props(t.params()));
    lines
}

/// The OPLS-AA parameter set exactly as molrs ships it.
fn shipped_oplsaa() -> Vec<String> {
    let t = OPLSAATypifier::oplsaa().expect("molrs ships OPLS-AA");
    let mut lines = dump_force_field(t.ff());
    lines.extend(dump_opls_meta(t.meta()));
    lines
}

// ---------------------------------------------------------------------------
// Fixture comparison
// ---------------------------------------------------------------------------

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/ff/fixtures/tables")
}

/// Which compiled table backs each parameter set — flat under `ff/params/`, no
/// `generated/`.
///
/// **Both MMFF front doors map to the same file.** There is no `mmff94.rs` and no
/// `mmff94s.rs`: `mmff94.xml` and `mmff94s.xml` differ by the `<ForceField name=…>`
/// attribute and nothing else, so emitting two byte-identical 199-entry tables would be
/// pure dead weight. One shared `mmff.rs`; the two typifiers differ by a name string and
/// an `MmffVariant`.
fn compiled_table(stem: &str) -> PathBuf {
    let file = match stem {
        "mmff94" | "mmff94s" => "mmff",
        other => other,
    };
    Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("src/ff/params/{file}.rs"))
}

/// The frozen pre-conversion dump: `#` header lines and blanks dropped.
fn reference(stem: &str) -> Vec<String> {
    let path = fixture_dir().join(format!("{stem}.reference.txt"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read the frozen reference {}: {e}", path.display()));
    text.lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(str::to_owned)
        .collect()
}

/// Refuse to run the comparison until there is something new to compare.
///
/// Without this the test is a tautology. The fixture was produced by the runtime XML
/// parser; until the compiled table exists, `shipped_*()` is *also* the runtime XML
/// parser, so the comparison is that parser against itself — green, and worth nothing.
/// The house calls this "passing vacuously" (`ff/typifier/source_gate.rs` guards against
/// it in four places); here it is the difference between a drift detector and a
/// decoration.
///
/// So this is the RED. Before the conversion the test fails here, saying so. After it,
/// this check passes and the line-by-line comparison below becomes the real contract —
/// and stays red for exactly as long as any one of ~10,000 values is wrong.
fn require_compiled_table(stem: &str) {
    let path = compiled_table(stem);
    let file = path
        .file_name()
        .and_then(|f| f.to_str())
        .expect("table file name");
    assert!(
        path.is_file(),
        "`src/ff/params/{file}` does not exist yet, so the {stem} force field below is \
         still being built by parsing XML at runtime — the very parser that produced the \
         reference fixture. Comparing them would compare that parser against itself and \
         prove nothing.\n\
         This assertion is the RED for ac-002: it lifts as soon as the table is compiled, \
         and from then on the comparison below is a real, field-for-field check of the \
         conversion."
    );
}

/// The exact size of each frozen dump, measured 2026-07-14 from the current XML.
///
/// Pinned as an equality, not a `>` floor. A floor is the wrong shape for a
/// non-vacuity guard here: the MMFF dumps are legitimately *small* (199 lines — see the
/// module header), so any floor loose enough to admit them is far too loose to notice a
/// fixture that had lost 90% of its content. The exact count is also the one number that
/// says out loud "the MMFF XML shrank by 4,065 rows and that was on purpose", so a future
/// reader who re-freezes a 4,358-line dump gets a failure instead of a silent pass.
const EXPECTED_ENTRIES: [(&str, usize); 3] = [("mmff94", 199), ("mmff94s", 199), ("oplsaa", 4727)];

fn expected_entries(stem: &str) -> usize {
    EXPECTED_ENTRIES
        .iter()
        .find(|(name, _)| *name == stem)
        .map(|(_, n)| *n)
        .unwrap_or_else(|| panic!("no expected entry count for `{stem}`"))
}

/// Compare the shipped parameter set against the frozen dump, and say precisely where
/// they part company.
fn assert_reproduces_the_xml_parse(stem: &str, got: &[String]) {
    let want = reference(stem);
    assert_eq!(
        want.len(),
        expected_entries(stem),
        "the frozen reference for {stem} has {} lines, not the {} it was frozen with — \
         the FIXTURE has changed, not the table. Re-blessing a reference is how a \
         conversion bug becomes the new expected value; restore it from git.",
        want.len(),
        expected_entries(stem)
    );

    let mut diffs = Vec::new();
    for (n, (w, g)) in want.iter().zip(got.iter()).enumerate() {
        if w != g {
            diffs.push(format!("  line {}:\n    want: {w}\n    got : {g}", n + 1));
        }
    }

    let mut report = String::new();
    if want.len() != got.len() {
        report.push_str(&format!(
            "  entry count changed: the XML parse produced {} lines, the compiled table \
             produces {} ({} {})\n",
            want.len(),
            got.len(),
            got.len().abs_diff(want.len()),
            if got.len() > want.len() {
                "invented"
            } else {
                "dropped"
            }
        ));
        let (long, short) = if got.len() > want.len() {
            (got, want.len())
        } else {
            (want.as_slice(), got.len())
        };
        for line in long.iter().skip(short).take(5) {
            report.push_str(&format!("    first unmatched: {line}\n"));
        }
    }

    if !diffs.is_empty() {
        report.push_str(&format!(
            "  {} of {} lines differ:\n{}\n",
            diffs.len(),
            want.len(),
            diffs
                .iter()
                .take(10)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n")
        ));
        if diffs.len() > 10 {
            report.push_str(&format!("    … and {} more\n", diffs.len() - 10));
        }
    }

    assert!(
        report.is_empty(),
        "{stem}: the compiled table does not reproduce the XML parse.\n{report}\n\
         chem-perceive-14 is a PURE REPRESENTATION change — the compiled table must hold \
         the same IEEE-754 doubles and the same strings the XML parser produced, in the \
         same order. The reference was frozen from `molrs/data/{stem}.xml` before it was \
         deleted; do not re-bless it, fix the generator."
    );
}

// ---------------------------------------------------------------------------
// ac-002
// ---------------------------------------------------------------------------

/// ac-002 — MMFF94: every bond, angle, stretch-bend, torsion, oop, vdW and atom-property
/// field survives the conversion unchanged.
#[test]
fn mmff94_compiled_table_reproduces_the_xml_parse_field_for_field() {
    require_compiled_table("mmff94");
    assert_reproduces_the_xml_parse("mmff94", &shipped_mmff94());
}

/// ac-002 — MMFF94s: the 108 rows that differ from MMFF94 are the whole point of the
/// variant, and they are invisible to every energy test that does not sit on an amide.
#[test]
fn mmff94s_compiled_table_reproduces_the_xml_parse_field_for_field() {
    require_compiled_table("mmff94s");
    assert_reproduces_the_xml_parse("mmff94s", &shipped_mmff94s());
}

/// ac-002 — OPLS-AA: force field **and** the SMARTS/override/layer typing metadata.
#[test]
fn oplsaa_compiled_table_reproduces_the_xml_parse_field_for_field() {
    require_compiled_table("oplsaa");
    assert_reproduces_the_xml_parse("oplsaa", &shipped_oplsaa());
}

// ---------------------------------------------------------------------------
// Guards on the reference itself
// ---------------------------------------------------------------------------

/// The frozen references are complete enough to be worth comparing against.
///
/// A dump that silently covered only the atom types would make all three tests above
/// green forever. This pins the **row shape** of each reference — how many entries of
/// each kind — so that a *fixture* that lost its content is as loud a failure as a
/// *table* that did.
///
/// Counting `<kind>[` (the entry-index marker the dump emits) rather than a bare
/// substring is what makes the count mean anything: `improper` also appears in the
/// `mmff_oop` *style header* line, so a `contains("improper")` check would report the
/// MMFF reference as carrying improper rows when it carries exactly zero of them. That
/// is the trap this test walked into before `mmff-orthogonal-02`, and the zeros below
/// are now load-bearing facts, not absences.
///
/// Green today and must stay green: it is a property of the committed fixtures, which
/// nothing in this spec should touch.
#[test]
fn the_frozen_references_have_the_row_shape_the_xml_declares() {
    // (stem, [(entry kind, exact count)])
    let expected: [(&str, &[(&str, usize)]); 3] = [
        // The MMFF energy rows are NOT here — they are in the RDKit port. What the XML
        // still carries is vdW, and the zeros are the assertion: `mmff-orthogonal-02`
        // deleted 4,065 bond/angle/torsion/oop type-defs that no kernel read, and this
        // is what stops a generator from putting them back.
        (
            "mmff94",
            &[
                ("pair", 95),
                ("bond", 0),
                ("angle", 0),
                ("dihedral", 0),
                ("improper", 0),
                ("atom", 0),
            ],
        ),
        (
            "mmff94s",
            &[
                ("pair", 95),
                ("bond", 0),
                ("angle", 0),
                ("dihedral", 0),
                ("improper", 0),
                ("atom", 0),
            ],
        ),
        (
            "oplsaa",
            &[
                ("atom", 813),
                ("bond", 300),
                ("angle", 932),
                ("dihedral", 1048),
                ("pair", 813),
                ("improper", 0),
            ],
        ),
    ];

    for (stem, kinds) in expected {
        let lines = reference(stem);
        assert_eq!(lines.len(), expected_entries(stem), "{stem}: entry count");

        for (kind, want) in kinds {
            let got = lines
                .iter()
                .filter(|l| l.contains(&format!("{kind}[")))
                .count();
            assert_eq!(
                got, *want,
                "{stem}.reference.txt has {got} `{kind}` rows, expected {want}. A count \
                 that DROPPED means the fixture lost content and the equivalence test \
                 above is now checking part of a table; a count that GREW on an MMFF set \
                 means the dead type-defs `mmff-orthogonal-02` deleted have come back."
            );
        }
    }

    // The typing metadata — the half of each XML that no energy test can see.
    let mmff_props = reference("mmff94")
        .iter()
        .filter(|l| l.starts_with("mmff prop "))
        .count();
    assert_eq!(
        mmff_props, 95,
        "the MMFF reference must carry all 95 nine-column `<AtomProperties>` rows: \
         `sbmb` picks the stretch-bend row, `arom`/`pilp`/`mltb` drive bond \
         classification, `crd`/`val` gate the type assignment. None of them appears in \
         an energy, so this dump is the only thing in the suite that would notice one \
         going missing."
    );

    let opls_rows = reference("oplsaa")
        .iter()
        .filter(|l| l.starts_with("opls row "))
        .count();
    assert_eq!(
        opls_rows, 813,
        "the OPLS reference must carry all 813 typing rows (SMARTS `def`, `overrides`, \
         `priority`, `layer`). `overrides` and `priority` decide which of several \
         matching types wins, and `layer` is what lets CL&P / CL&Pol overlay OPLS at \
         all — a conversion that dropped them would leave every energy assertion in the \
         suite intact and re-type half of any molecule with a functional group."
    );
}

/// The `<ElectrostaticParams>` section survives the conversion.
///
/// This is the newest data in either MMFF XML — `mmff-orthogonal-01` added it — and it
/// is therefore the row a generator written against the *old* XML shape is likeliest to
/// miss entirely, silently, while every other test in this file stays green. It lands in
/// two different places, and both are checked:
///
/// * `dielectric` and `delta` on the electrostatic pair style (the dielectric and the
///   electrostatic buffering constant of MMFF's Coulomb term) — and, since
///   `mmff-ele-compose`, `coulomb` beside them (see below);
/// * `scale14="0.75"` as `special_bonds.coul[2]` — the 1-4 Coulomb scale factor, which
///   is not a style parameter at all but a property of the force field.
///
/// A generator that emitted the styles and the vdW rows but skipped this section would
/// produce an MMFF force field whose 1-4 electrostatics are scaled by 1.0 instead of
/// 0.75. That is a real energy error on every molecule with a 1-4 pair, and the entry
/// count would not move.
///
/// # The style is `pair/coul/cut`, and it carries THREE numbers (mmff-ele-compose)
///
/// It was `pair/mmff_ele`. MMFF's electrostatics is a **buffered Coulomb** —
/// `E = k·qᵢqⱼ / (D·(r + δ))` — which is the generic kernel at `k = 332.0716`,
/// `D = 1.0`, `δ = 0.05`: a *parameterization*, not a kernel of its own. The Coulomb
/// constant therefore becomes DATA on the style, exactly like `dielectric` and `delta`
/// — it is the one number of the three that was never data at all, and the one that
/// cannot be shared (Halgren's 332.0716 vs CODATA's 332.06371, 2.4e-5 apart, is worth
/// more than the 1e-3 RDKit parity tolerance on caffeine).
///
/// **No value moves here.** `dielectric` and `delta` are the numbers they always were;
/// `coulomb` is the number the *kernel* always held (`ff/constants.rs::COULOMB_MMFF`),
/// relocated to where a force field's parameter belongs. The frozen reference records
/// the ForceField the shipped table builds, so it records that relocation.
#[test]
fn the_mmff_electrostatics_section_is_in_the_frozen_reference() {
    for stem in ["mmff94", "mmff94s"] {
        let lines = reference(stem);

        let ele = lines
            .iter()
            .find(|l| l.contains("pair coul/cut") && l.contains("@style"))
            .unwrap_or_else(|| {
                panic!(
                    "{stem}.reference.txt has no `coul/cut` pair style. \
                     `<ElectrostaticParams>` (mmff-orthogonal-01) is the newest section \
                     of the MMFF XML and the easiest for a generator to not know about — \
                     and since `mmff-ele-compose` it lands on the GENERIC Coulomb style, \
                     because MMFF's electrostatics is a parameterization of it and not a \
                     kernel of MMFF's own."
                )
            });
        assert!(
            ele.contains("dielectric=1.0") && ele.contains("delta=0.05"),
            "{stem}: the electrostatic style must carry MMFF's dielectric (1.0) and \
             electrostatic buffering delta (0.05). Got: {ele}"
        );
        assert!(
            ele.contains("coulomb=332.0716"),
            "{stem}: the electrostatic style must carry MMFF's COULOMB CONSTANT \
             (Halgren's 332.0716) as a style param. It used to live in the kernel \
             (`ff/constants.rs::COULOMB_MMFF`), which is how MMFF could be evaluated \
             with a constant nobody handed it — and how the generic kernel's CODATA \
             332.06371 would silently take its place the moment the two were merged. \
             Got: {ele}"
        );

        let special = lines
            .iter()
            .find(|l| l.starts_with("forcefield special_bonds"))
            .expect("every dump carries the special_bonds line");
        assert!(
            special.contains("coul=[0.0, 0.0, 0.75]"),
            "{stem}: `scale14=\"0.75\"` from `<ElectrostaticParams>` must survive as the \
             1-4 Coulomb weight. A conversion that dropped it would scale 1-4 \
             electrostatics by 1.0 — a real energy error on every molecule with a 1-4 \
             pair, and one that moves no entry count. Got: {special}"
        );
    }
}

/// mmff-ele-compose ac-002/ac-003 — **every** force field that uses the generic Coulomb
/// style declares the constant it is evaluated with. OPLS included.
///
/// The kernel no longer has a Coulomb constant of its own to fall back on: `coulomb` and
/// `dielectric` are force-field data and a style that omits them is an `Err`
/// (`potential/coul_buffered.rs`). OPLS's `pair/coul/cut` was defined with **empty
/// params** (`ff.def_pairstyle("coul/cut", &[])`), which is the same shape as the
/// backdoor this spec closes — it just happened to agree with the kernel's private copy
/// of CODATA's 332.06371.
///
/// So the OPLS force field must now say so out loud, and the frozen reference records it.
/// **The number does not move**: 332.06371 is `molrs::units::constants::COULOMB_REAL`,
/// the value `coul/cut` has always used — it is now *stated* rather than *assumed*.
///
/// RED until `ff/typifier/opls/embedded.rs` (and the OPLS/LAMMPS readers) declare it.
#[test]
fn the_generic_coulomb_style_declares_the_constant_it_uses() {
    let lines = reference("oplsaa");
    let coul = lines
        .iter()
        .find(|l| l.contains("pair coul/cut") && l.contains("@style"))
        .expect("oplsaa.reference.txt has a `pair coul/cut` style");

    assert!(
        coul.contains("coulomb=332.06371"),
        "oplsaa's `pair/coul/cut` does not declare its Coulomb constant. CODATA's 332.06371 is \
         what the kernel has always used — but the kernel is no longer allowed to KNOW that: a \
         style with no `coulomb` is a style whose force field never spoke, and the kernel would \
         be supplying the physics. (MMFF's is Halgren's 332.0716 — same kernel, different force \
         field, and the difference is above the RDKit parity tolerance.) Got: {coul}"
    );
    assert!(
        coul.contains("dielectric=1.0"),
        "oplsaa's `pair/coul/cut` does not declare its dielectric. `unwrap_or(1.0)` in the kernel \
         is the kernel pretending the force field spoke; OPLS is parameterised in vacuum and must \
         say D = 1.0 itself. Got: {coul}"
    );
}

/// ONE shared MMFF table: the two front doors' ForceField trees differ by the **name**,
/// and by nothing else this file can see.
///
/// This test used to assert the opposite: "the two references differ on exactly 11
/// out-of-plane and 42 torsion rows, Halgren's 1999 re-parameterisation of delocalised
/// trivalent nitrogen". That was true of the XML it was written against. It is not true
/// of the XML in the tree today, and the difference is the whole reason the reference
/// fixtures had to be re-frozen rather than reused.
///
/// `mmff-orthogonal-02` deleted the type-def rows from both MMFF XMLs, so the 11 + 42
/// went with them — leaving `mmff94.xml` and `mmff94s.xml` differing by two lines, the
/// `<ForceField name=…>` attribute. `mmff94s.xml` is therefore deleted outright and
/// **one** table (`ff/params/mmff.rs`) serves both typifiers.
///
/// **The 11 + 42 claim did not disappear — it moved to where it was always true.** Those
/// rows never entered an energy: `mmff_oop` and `mmff_torsion` are
/// `ParamSource::PerInstance` styles, and the numbers the kernels read are the `koop` and
/// `(v1, v2, v3)` the *resolver* looks up in the RDKit port (`MMFF_OOP` vs `MMFF_OOP_S`,
/// `MMFF_TOR` vs `MMFF_TOR_S` — 11 and 42 differing rows respectively, measured) and bakes
/// into Frame columns.
///
/// # The gap this leaves, and who closes it
///
/// With one shared table, the two front doors differ by exactly two things: the name
/// string and the `MmffVariant`. **This file can see the first and is blind to the
/// second** — the variant never reaches the `ForceField` tree, so a front door wired to
/// the wrong `MmffVariant` would reproduce this fixture perfectly and ship MMFF94's
/// numbers under MMFF94s's name.
///
/// So this test does two things. It pins the name (the half it *can* see), and it asserts
/// that the tests which see the other half **still exist** — because merging two tables
/// into one is exactly the change during which someone deletes "redundant" variant tests,
/// and after that nothing in the suite would notice the variant being ignored entirely.
///
/// Green today; it must stay green.
#[test]
fn one_shared_mmff_table_and_the_two_front_doors_differ_only_in_name() {
    let (a, b) = (reference("mmff94"), reference("mmff94s"));
    assert_eq!(
        a.len(),
        b.len(),
        "MMFF94 and MMFF94s dump the same {} entries",
        expected_entries("mmff94")
    );

    let differing: Vec<(usize, &String, &String)> = a
        .iter()
        .zip(b.iter())
        .enumerate()
        .filter(|(_, (x, y))| x != y)
        .map(|(n, (x, y))| (n, x, y))
        .collect();

    assert_eq!(
        differing.len(),
        1,
        "exactly one dump line may differ between the two MMFF front doors — the \
         force-field name. Every parameter row is identical, which is precisely why they \
         share ONE table: the variant's 11 oop + 42 torsion rows live in the RDKit port, \
         not in the XML (see this test's doc). Differing lines:\n{}",
        differing
            .iter()
            .take(10)
            .map(|(n, x, y)| format!("  line {}:\n    94 : {x}\n    94s: {y}", n + 1))
            .collect::<Vec<_>>()
            .join("\n")
    );

    let (_, name94, name94s) = differing[0];
    assert_eq!(name94.as_str(), "forcefield name=MMFF94");
    assert_eq!(
        name94s.as_str(),
        "forcefield name=MMFF94s",
        "`MMFF94STypifier` must name its own force field. With one shared table this is \
         the ONLY difference the ForceField tree carries, so it is the only thing a \
         row-comparing test can check — every other assertion in this file passes either \
         way, because the rows really are identical."
    );

    // The variant is invisible to everything above. These are the tests that DO see it;
    // if the merge deletes them as "redundant", nothing in the suite would catch an
    // `MmffVariant` that is ignored, mis-wired, or hard-coded.
    let variant_suite =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/ff/typifier/mmff_variant.rs");
    let src = std::fs::read_to_string(&variant_suite).unwrap_or_else(|e| {
        panic!(
            "{} is gone ({e}). It is the ONLY thing standing between MMFF94s and being \
             silently identical to MMFF94, now that the two share a parameter table.",
            variant_suite.display()
        )
    });

    const VARIANT_GUARDS: [&str; 5] = [
        // the koop sign flip — MMFF94s's whole reason to exist
        "fn mmff94s_bakes_positive_koop_on_every_n_centre",
        "fn mmff94_bakes_a_different_koop_on_every_n_centre",
        "fn n_centre_koop_matches_the_measured_table",
        "fn mmff94s_bakes_different_torsion_coefficients",
        // and the non-vacuity twin: identical where the variant should NOT bite
        "fn typed_frames_are_bit_identical_without_a_delocalised_nitrogen",
    ];
    let lost: Vec<&str> = VARIANT_GUARDS
        .into_iter()
        .filter(|guard| !src.contains(guard))
        .collect();
    assert!(
        lost.is_empty(),
        "the MMFF variant guards have been deleted: {lost:?}\n\
         The 94-vs-94s difference is `MmffVariant` selecting `MMFF_OOP_S` / `MMFF_TOR_S` \
         in the shared table, and it reaches an energy ONLY through the `koop` and \
         `(v1, v2, v3)` baked onto the Frame. These tests are what check that. Merging the \
         two tables into one does not make them redundant — it makes them the only \
         coverage left."
    );
}
