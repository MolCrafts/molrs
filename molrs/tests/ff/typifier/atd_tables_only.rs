//! ac-003 (chem-perceive-06) — the engine is unchanged: tables only.
//!
//! chem-perceive-06 claims GAFF / GAFF2 / AMBER / SYBYL atom typing is a **table**,
//! not code: four `ATOMTYPE_*.DEF` files handed to the `AtdTypifier` that
//! chem-perceive-05 already built. `atd_antechamber.rs` shows the four tables
//! reproduce antechamber 37/37; this file shows the engine did not learn about
//! them — that the seven parameter sets reach the rule matcher through one code
//! path whose only per-set step is `AtdParameterSet::table()`.
//!
//! **Why source gates rather than `git diff`.** The acceptance criterion is worded
//! as a diff ("no logic change in the engine for this spec"), but a diff-shaped
//! test is a test that passes forever the moment the spec is merged: the diff is
//! empty against the wrong base, and nothing stops the eighth table from arriving
//! with an `if set == Sybyl` beside it. The property that actually has to hold —
//! *the engine contains no per-set and no per-table branch* — is a property of the
//! source, and it keeps holding for table number eight.
//!
//! The scope is the engine tree (`src/ff/typifier/atd/`), not the whole typifier
//! tree. `am1bcc.rs` legitimately maps `BccParameterSet::Bcc -> AtdParameterSet::Bcc`;
//! that is a *caller* choosing a table, which is the API working.

use std::collections::{HashMap, HashSet};

use molrs::ff::typifier::atd::AtdParameterSet;

use super::source_gate::atd_engine_sources;

/// Every parameter set: the value, its source spelling, the table constant it must
/// map to, and the upstream `.DEF` that constant carries.
///
/// The source-text columns are what the two source gates below grep for, and the
/// value column is what the behavioural gate calls — so a variant renamed in one
/// place and not the other fails to compile.
const SETS: [(AtdParameterSet, &str, &str, &str); 7] = [
    (
        AtdParameterSet::Bcc,
        "Bcc",
        "ATOMTYPE_BCC",
        "ATOMTYPE_BCC.DEF",
    ),
    (
        AtdParameterSet::Abcg2,
        "Abcg2",
        "ATOMTYPE_ABCG2",
        "ATOMTYPE_ABCG2.DEF",
    ),
    (
        AtdParameterSet::Gas,
        "Gas",
        "ATOMTYPE_GAS",
        "ATOMTYPE_GAS.DEF",
    ),
    (
        AtdParameterSet::Gff,
        "Gff",
        "ATOMTYPE_GFF",
        "ATOMTYPE_GFF.DEF",
    ),
    (
        AtdParameterSet::Gff2,
        "Gff2",
        "ATOMTYPE_GFF2",
        "ATOMTYPE_GFF2.DEF",
    ),
    (
        AtdParameterSet::Amber,
        "Amber",
        "ATOMTYPE_AMBER",
        "ATOMTYPE_AMBER.DEF",
    ),
    (
        AtdParameterSet::Sybyl,
        "Sybyl",
        "ATOMTYPE_SYBYL",
        "ATOMTYPE_SYBYL.DEF",
    ),
];

/// The only lines the engine may spend on knowing which set it was given:
/// `Self::Gff => ATOMTYPE_GFF,` — seven of them, one per set.
fn table_lookup_arms() -> Vec<String> {
    SETS.iter()
        .map(|(_, variant, table, _)| format!("Self::{variant} => {table},"))
        .collect()
}

/// Does this line of source name a specific parameter set?
fn names_a_set(text: &str) -> bool {
    SETS.iter().any(|(_, variant, _, _)| {
        text.contains(&format!("Self::{variant}"))
            || text.contains(&format!("AtdParameterSet::{variant}"))
    })
}

// ── 1. no per-set branch ────────────────────────────────────────────────────

/// The engine names a parameter set ONLY to look up its table.
///
/// This is ac-003's real content. A `match set { Gff => .., _ => .. }` anywhere
/// past `table()` — a GAFF-only fixup, a "SYBYL needs one more pass" — is the
/// engine learning a table, and it is what turns "one engine, N tables" back into
/// N engines. Every arm is also *checked against its table*, so a `Self::Gff2 =>
/// ATOMTYPE_GFF,` typo (which would make `gff2_atom_types_match_antechamber` test
/// the GFF table twice) is a failure here, not a silent pass there.
#[test]
fn the_engine_names_a_parameter_set_only_where_it_picks_a_table() {
    let arms = table_lookup_arms();
    let mut found: HashSet<&str> = HashSet::new();
    let mut offenders: Vec<String> = Vec::new();

    for (path, src) in atd_engine_sources() {
        for (n, line) in src.lines().enumerate() {
            let text = line.trim();
            if !names_a_set(text) {
                continue;
            }
            match arms.iter().position(|arm| arm == text) {
                Some(i) => {
                    found.insert(SETS[i].1);
                }
                None => offenders.push(format!("{}:{}: {text}", path.display(), n + 1)),
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "the ATD engine branches on which parameter set it was given, outside the \
         `AtdParameterSet::table()` lookup. Atom typing for a new force field is a \
         table (chem-perceive-06 added four without touching the engine); a per-set \
         branch here is the engine learning one:\n{}",
        offenders.join("\n")
    );
    assert_eq!(
        found.len(),
        SETS.len(),
        "only {} of the {} parameter sets reach a table through `Self::X => ATOMTYPE_X,`; \
         the rest are routed some other way, and this gate cannot see how",
        found.len(),
        SETS.len(),
    );
}

/// The rule matcher does not know which table it is walking.
///
/// `facts.rs` / `pattern.rs` / `rules.rs` receive an `&AtdTable` and read it as
/// data. If any of them so much as mentions `AtdParameterSet`, the rule language
/// has stopped being table-generic, and the next `.DEF` file will need code.
#[test]
fn the_rule_matcher_never_mentions_a_parameter_set() {
    let offenders: Vec<String> = atd_engine_sources()
        .iter()
        .filter(|(path, _)| path.file_name().and_then(|f| f.to_str()) != Some("mod.rs"))
        .flat_map(|(path, src)| {
            src.lines()
                .enumerate()
                .filter(|(_, line)| line.contains("AtdParameterSet"))
                .map(|(n, line)| format!("{}:{}: {}", path.display(), n + 1, line.trim()))
                .collect::<Vec<_>>()
        })
        .collect();

    assert!(
        offenders.is_empty(),
        "the ATD rule matcher names a parameter set; it must take the table as data \
         (`&AtdTable`) and know nothing about which one it is:\n{}",
        offenders.join("\n")
    );
}

// ── 2. no per-table branch ──────────────────────────────────────────────────

/// The engine names a table constant only to import it or to hand it back.
///
/// The sibling hole to the gate above: `if table == ATOMTYPE_GFF { .. }` inside
/// `rules.rs` mentions no parameter set at all, and would sail through it. Both
/// spellings of "the engine knows about a particular table" are closed, so the
/// only thing the engine may do with `ATOMTYPE_*` is return it from `table()`.
#[test]
fn the_engine_names_a_table_constant_only_to_import_it_or_to_return_it() {
    let arms: HashSet<String> = table_lookup_arms().into_iter().collect();
    let mut offenders: Vec<String> = Vec::new();

    for (path, src) in atd_engine_sources() {
        // `use crate::ff::params::{ ATOMTYPE_BCC, .. };` wraps over several lines;
        // every line of the statement is an import, not a use of the constant.
        let mut in_use = false;
        for (n, line) in src.lines().enumerate() {
            let text = line.trim();
            in_use |= text.starts_with("use ");
            let importing = in_use;
            if in_use && text.ends_with(';') {
                in_use = false;
            }
            if !text.contains("ATOMTYPE_") || importing || arms.contains(text) {
                continue;
            }
            offenders.push(format!("{}:{}: {text}", path.display(), n + 1));
        }
    }

    assert!(
        offenders.is_empty(),
        "the ATD engine uses a specific `ATOMTYPE_*` table outside the `table()` \
         lookup — a per-table special case wearing a different hat:\n{}",
        offenders.join("\n")
    );
}

// ── 3. the lookup itself is honest ──────────────────────────────────────────

/// Each parameter set resolves to its own upstream `.DEF`, and no two share one.
///
/// The behavioural counterpart of gate 1: it reads the mapping through the public
/// API instead of through the source text, so it survives a reformat, and it is
/// what makes the four new parity tests non-vacuous. Two variants pointing at the
/// same table would give one table two names and quietly test it twice.
#[test]
fn every_parameter_set_resolves_to_its_own_upstream_table() {
    let mut owner: HashMap<&str, AtdParameterSet> = HashMap::new();
    for (set, _, _, def) in SETS {
        let table = set.table();
        assert_eq!(
            table.name, def,
            "{set:?} claims {def} but walks {}",
            table.name
        );
        assert!(
            !table.rules.is_empty(),
            "{set:?} walks {}, which declares no ATD rules — every atom would fail to \
             type, and the gates above would pass on an engine that types nothing",
            table.name
        );
        if let Some(other) = owner.insert(table.name, set) {
            panic!(
                "{set:?} and {other:?} both walk {} — one of them is an alias of the other, \
                 and its antechamber parity test is testing the wrong table",
                table.name
            );
        }
    }
}
