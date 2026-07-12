//! ac-002 — one ATD rule engine, N atom-type tables.
//!
//! `AtdTypifier` is antechamber's `ATD`/`WILDATOM` rule engine, parameterized by
//! the `ATOMTYPE_*.DEF` table it walks. The gate: the SAME engine must reproduce
//! `antechamber -at bcc`, `-at abcg2` and `-at gas` on all 37 oracle molecules.
//!
//! The three columns are not restatements of each other — they disagree exactly
//! where atom typing is hard. Imidazole's pyridine-type aromatic N is `24` under
//! BCC but `28` under ABCG2 and `n2` under GAS, and GAS types benzene's ring
//! carbons `c2` where both numeric tables say `16`. So no table-specific special
//! case inside the engine can satisfy all three columns at once: that is what
//! makes "one engine, N tables" a test rather than a claim.
//!
//! GAS is the column that cannot be reached through the BCC-correction API at
//! all: `ATOMTYPE_GAS.DEF` exists, but there is no `BCCPARM_GAS.DAT`, so GAS is
//! an atom-type table with no correction family. It can only be driven by a
//! typifier parameterized on the TABLE, which is precisely the axis this spec
//! separates out of `BccParameterSet`.

use std::collections::HashSet;

use molrs::Atomistic;
use molrs::ff::params::{ATOMTYPE_ABCG2, ATOMTYPE_BCC, ATOMTYPE_GAS};
use molrs::ff::typifier::Typifier;
use molrs::ff::typifier::am1bcc::BCCAtomTypifier;
use molrs::ff::typifier::atd::{AtdParameterSet, AtdTypifier};

use super::antechamber_oracle::{AntechamberCase, CASES};
use super::oracle_mol::{atom_types, build_case, report};

/// Which oracle column a parameter set must reproduce.
type OracleColumn = fn(&'static AntechamberCase) -> &'static [&'static str];

/// Type every oracle molecule with `set`; report the ones that disagree.
fn typing_failures(set: AtdParameterSet, want: OracleColumn) -> Vec<String> {
    let typifier = AtdTypifier::new(set);
    let mut failures = Vec::new();
    for case in CASES {
        let (mol, ids) = build_case(case);
        let typed = match typifier.typify(&mol) {
            Ok(typed) => typed,
            Err(e) => {
                failures.push(format!("  {:22} ERROR {e}", case.name));
                continue;
            }
        };
        let got = atom_types(&typed, &ids);
        let wrong: Vec<String> = want(case)
            .iter()
            .enumerate()
            .filter(|(k, want_k)| got[*k].as_deref() != Some(**want_k))
            .map(|(k, want_k)| format!("{k}{}: got {:?} want {want_k}", case.elements[k], got[k]))
            .collect();
        if !wrong.is_empty() {
            failures.push(format!(
                "  {:22} {}/{} atoms wrong: {}",
                case.name,
                wrong.len(),
                case.elements.len(),
                wrong.join(", ")
            ));
        }
    }
    failures
}

/// ABCG2 atom types, 37/37 against `antechamber -at abcg2`.
#[test]
fn abcg2_atom_types_match_antechamber() {
    let failures = typing_failures(AtdParameterSet::Abcg2, |case| case.abcg2_atom_types);
    report("ABCG2 atom typing (-at abcg2)", &failures, CASES.len());
}

/// GAS atom types, 37/37 against `antechamber -at gas`.
///
/// GAS has no BCC correction table, so this test is only expressible through the
/// table-parameterized engine.
#[test]
fn gas_atom_types_match_antechamber() {
    let failures = typing_failures(AtdParameterSet::Gas, |case| case.gas_atom_types);
    report("GAS atom typing (-at gas)", &failures, CASES.len());
}

/// BCC atom types, 37/37, driven through the generic engine rather than the
/// BCC-specific typifier.
#[test]
fn bcc_atom_types_match_antechamber_through_the_generic_engine() {
    let failures = typing_failures(AtdParameterSet::Bcc, |case| case.bcc_atom_types);
    report("BCC atom typing via AtdTypifier", &failures, CASES.len());
}

/// The BCC-correction typifier must *be* the ATD engine with the BCC table — not
/// a second implementation that merely agrees with it on the oracle.
///
/// Without this, "one engine, N tables" could be satisfied by keeping the old
/// hand-rolled BCC path alive next to a new generic one, and the two would drift
/// the first time a rule was fixed in only one of them.
#[test]
fn the_bcc_typifier_agrees_atom_for_atom_with_the_atd_engine() {
    let generic = AtdTypifier::new(AtdParameterSet::Bcc);
    let bcc = BCCAtomTypifier::bcc();
    let mut failures = Vec::new();
    for case in CASES {
        let (mol, ids) = build_case(case);
        let via_generic = generic.typify(&mol).map(|typed| atom_types(&typed, &ids));
        let via_bcc = bcc.typify(&mol).map(|typed| atom_types(&typed, &ids));
        if via_generic != via_bcc {
            failures.push(format!(
                "  {:22} AtdTypifier(Bcc) {via_generic:?} != BCCAtomTypifier::bcc() {via_bcc:?}",
                case.name
            ));
        }
    }
    report(
        "BCCAtomTypifier vs AtdTypifier(Bcc)",
        &failures,
        CASES.len(),
    );
}

/// The engine may only assign types the table it is walking declares.
///
/// molrs's no-fallback-values rule at the typifier boundary. A type in the output
/// that is in no rule of the table was invented by the engine — i.e. a per-table
/// special case, the thing ac-002 exists to forbid — and it would surface much
/// later as a wrong charge rather than as an error.
///
/// (A lone iron atom is NOT such a case, tempting though the test is: `Fe` is a
/// real `ATOMTYPE_BCC.DEF` row, and typing it `Fe` is the table doing its job.)
#[test]
fn no_assigned_type_is_absent_from_the_table_being_walked() {
    for (set, table) in [
        (AtdParameterSet::Bcc, ATOMTYPE_BCC),
        (AtdParameterSet::Abcg2, ATOMTYPE_ABCG2),
        (AtdParameterSet::Gas, ATOMTYPE_GAS),
    ] {
        let declared: HashSet<&str> = table.rules.iter().map(|rule| rule.atom_type).collect();
        let typifier = AtdTypifier::new(set);
        for case in CASES {
            let (mol, ids) = build_case(case);
            let typed = typifier
                .typify(&mol)
                .unwrap_or_else(|e| panic!("{set:?} {}: {e}", case.name));
            for (k, got) in atom_types(&typed, &ids).iter().enumerate() {
                let got = got
                    .as_deref()
                    .unwrap_or_else(|| panic!("{set:?} {}: atom {k} got no type", case.name));
                assert!(
                    declared.contains(got),
                    "{set:?} {}: atom {k}{} was typed `{got}`, which {} does not declare — \
                     the engine invented a type",
                    case.name,
                    case.elements[k],
                    table.name
                );
            }
        }
    }
}

/// Edge case: an empty molecule types to an empty molecule.
///
/// Zero atoms is a legitimate input (a filtered selection, a trajectory frame with
/// nothing in it); it must not be an error and must not panic on the rule loop.
#[test]
fn an_empty_molecule_types_to_an_empty_molecule() {
    let empty = Atomistic::new();
    for set in [
        AtdParameterSet::Bcc,
        AtdParameterSet::Abcg2,
        AtdParameterSet::Gas,
    ] {
        let typed = AtdTypifier::new(set)
            .typify(&empty)
            .unwrap_or_else(|e| panic!("{set:?}: typifying an empty molecule errored: {e}"));
        assert_eq!(
            typed.n_atoms(),
            0,
            "{set:?}: an empty molecule gained atoms"
        );
    }
}

/// Non-vacuity guard for this whole file, at the fixture level.
///
/// The three-column claim is only meaningful because the columns differ. If a
/// regenerated oracle ever collapsed them (e.g. an `-at` flag silently ignored),
/// every test above would still pass while testing one table three times.
#[test]
fn the_three_oracle_type_columns_actually_disagree() {
    let bcc_vs_abcg2 = CASES
        .iter()
        .filter(|case| case.bcc_atom_types != case.abcg2_atom_types)
        .count();
    assert!(
        bcc_vs_abcg2 > 0,
        "BCC and ABCG2 typed all 37 molecules identically — the oracle's `-at abcg2` \
         run did not take effect, and `abcg2_atom_types_match_antechamber` is vacuous"
    );

    let gas_vs_bcc = CASES
        .iter()
        .filter(|case| case.gas_atom_types != case.bcc_atom_types)
        .count();
    assert_eq!(
        gas_vs_bcc,
        CASES.len(),
        "GAS uses a different type alphabet (`c3`, `n2`) than BCC's numeric codes, \
         so it must differ on every molecule"
    );
}
