//! ac-002 (chem-perceive-05) and ac-001/ac-002 (chem-perceive-06) — one ATD rule
//! engine, seven atom-type tables.
//!
//! `AtdTypifier` is antechamber's `ATD`/`WILDATOM` rule engine, parameterized by
//! the `ATOMTYPE_*.DEF` table it walks. The gate: the SAME engine must reproduce
//! `antechamber -at {bcc,abcg2,gas,gaff,gaff2,amber,sybyl}` on all 37 oracle
//! molecules — 37/37 per column, seven columns.
//!
//! The columns are not restatements of each other — they disagree exactly where
//! atom typing is hard. Imidazole's pyridine-type aromatic N is `24` under BCC,
//! `28` under ABCG2, `n2` under GAS, `nd` under GAFF/GAFF2, `NB` under AMBER and
//! `N.2` under SYBYL; N-methylacetamide's amide N is `n` under GAFF but `ns`
//! under GAFF2. So no table-specific special case inside the engine can satisfy
//! all seven columns at once: that is what makes "one engine, N tables" a test
//! rather than a claim.
//!
//! chem-perceive-06 asserts the four new force fields are a **table** change with
//! no engine change. These four columns are what adjudicates that assertion — and
//! they do not merely confirm it. They show the shared engine is under-general in
//! two places the coarse BCC / ABCG2 / GAS tables cannot see: the `AR1`/`AR2`
//! ring classification (5-membered heteroaromatics), and atoms that match no rule
//! (antechamber's terminal `DU` / `ANY` row). Those are engine/table facts, not
//! test bugs, and the tests below stay red until they are fixed.
//!
//! Note the flag/table spelling gap: `antechamber -at gaff` walks
//! `ATOMTYPE_GFF.DEF` and `-at gaff2` walks `ATOMTYPE_GFF2.DEF`, so
//! `AtdParameterSet::Gff` / `Gff2` are named after the tables, not the flags.
//!
//! GAS is the column that cannot be reached through the BCC-correction API at
//! all: `ATOMTYPE_GAS.DEF` exists, but there is no `BCCPARM_GAS.DAT`, so GAS is
//! an atom-type table with no correction family. It can only be driven by a
//! typifier parameterized on the TABLE, which is precisely the axis this spec
//! separates out of `BccParameterSet`. GFF / GFF2 / AMBER / SYBYL are four more
//! of the same shape.

use std::collections::HashSet;

use molrs::Atomistic;
use molrs::ff::params::{
    ATOMTYPE_ABCG2, ATOMTYPE_AMBER, ATOMTYPE_BCC, ATOMTYPE_GAS, ATOMTYPE_GFF, ATOMTYPE_GFF2,
    ATOMTYPE_SYBYL, AtdTable,
};
use molrs::ff::typifier::Typifier;
use molrs::ff::typifier::am1bcc::BCCAtomTypifier;
use molrs::ff::typifier::atd::{AtdParameterSet, AtdTypifier};

use super::antechamber_oracle::{AntechamberCase, CASES};
use super::oracle_mol::{atom_types, build_case, report};

/// Which oracle column a parameter set must reproduce.
type OracleColumn = fn(&'static AntechamberCase) -> &'static [&'static str];

/// Every parameter set, the table it names, and the oracle column that table's
/// `antechamber -at` run produced.
///
/// One list, used by every gate below, so a table added to `AtdParameterSet`
/// without an oracle column (or vice versa) fails to compile rather than
/// silently going untested.
const SETS: [(AtdParameterSet, AtdTable, OracleColumn); 7] = [
    (AtdParameterSet::Bcc, ATOMTYPE_BCC, |case| {
        case.bcc_atom_types
    }),
    (AtdParameterSet::Abcg2, ATOMTYPE_ABCG2, |case| {
        case.abcg2_atom_types
    }),
    (AtdParameterSet::Gas, ATOMTYPE_GAS, |case| {
        case.gas_atom_types
    }),
    (AtdParameterSet::Gff, ATOMTYPE_GFF, |case| {
        case.gff_atom_types
    }),
    (AtdParameterSet::Gff2, ATOMTYPE_GFF2, |case| {
        case.gff2_atom_types
    }),
    (AtdParameterSet::Amber, ATOMTYPE_AMBER, |case| {
        case.amber_atom_types
    }),
    (AtdParameterSet::Sybyl, ATOMTYPE_SYBYL, |case| {
        case.sybyl_atom_types
    }),
];

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

// ── chem-perceive-06: four more tables ──────────────────────────────────────
//
// The 2026-06-19 note said GAFF typing must be delegated to AmberTools because a
// native typifier would be a second implementation of antechamber. chem-perceive-06
// reverses that: GAFF typing is `ATOMTYPE_GFF.DEF` handed to the engine that
// already exists. Each test below is the same call as the three above with a
// different table — which is the whole point: if a table needs engine code, the
// reversal was premature, and these tests are where that shows.

/// ac-001 — GAFF atom types, 37/37 against `antechamber -at gaff`.
///
/// `-at gaff` walks `ATOMTYPE_GFF.DEF`; the type alphabet is lowercase (`c3`,
/// `ca`, `os`, `h1`), disjoint from BCC's numeric codes, so a typifier wired to
/// the wrong table cannot accidentally pass this.
#[test]
fn gff_atom_types_match_antechamber() {
    let failures = typing_failures(AtdParameterSet::Gff, |case| case.gff_atom_types);
    report("GAFF atom typing (-at gaff)", &failures, CASES.len());
}

/// ac-001 — GAFF2 atom types, 37/37 against `antechamber -at gaff2`.
///
/// GAFF2 is not GAFF with a new name: it splits types GAFF conflates (the amide N
/// of N-methylacetamide is `n` under GAFF and `ns` under GAFF2), and the two
/// tables disagree on 6 of the 37 molecules. So this test is not a copy of the
/// one above — a `Gff2` variant quietly aliased to `ATOMTYPE_GFF` fails it.
#[test]
fn gff2_atom_types_match_antechamber() {
    let failures = typing_failures(AtdParameterSet::Gff2, |case| case.gff2_atom_types);
    report("GAFF2 atom typing (-at gaff2)", &failures, CASES.len());
}

/// ac-002 — AMBER atom types, 37/37 against `antechamber -at amber`.
///
/// The parm94-style uppercase types (`CT`, `CA`, `O2`, `H1`). AMBER is the table
/// that reaches its own last row: `ATOMTYPE_AMBER.DEF` ends with `ATD DU &`, a
/// constraint-free catch-all, and antechamber uses it — nitromethane's two nitro
/// oxygens and DMSO's sulfoxide sulfur come back `DU`, and those are the oracle's
/// answer, so molrs must produce them too. It is the only oracle column that
/// exercises the fall-through row every `.DEF` ends with (`ANY` in SYBYL), and the
/// generator drops that row today (`ff/params.rs`: "ATD rows minus the DU row"),
/// so this test cannot pass without a decision about it.
#[test]
fn amber_atom_types_match_antechamber() {
    let failures = typing_failures(AtdParameterSet::Amber, |case| case.amber_atom_types);
    report("AMBER atom typing (-at amber)", &failures, CASES.len());
}

/// ac-002 — SYBYL atom types, 37/37 against `antechamber -at sybyl`.
///
/// The mol2 type alphabet (`C.3`, `C.ar`, `N.am`, `O.co2`). It is the hybridization
/// -and-context table: `O.co2` for both carboxylate oxygens of acetate is a
/// delocalized-bond fact, not an element fact.
#[test]
fn sybyl_atom_types_match_antechamber() {
    let failures = typing_failures(AtdParameterSet::Sybyl, |case| case.sybyl_atom_types);
    report("SYBYL atom typing (-at sybyl)", &failures, CASES.len());
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
///
/// **`declared` is `atom_type ∪ alternate`, not `atom_type` alone.** antechamber
/// answers `cd` for half of pyrrole's ring, and no `ATOMTYPE_GFF.DEF` row emits
/// `cd`: the `.DEF` only ever emits the phase-1 name (`cc`), and a second pass
/// 2-colours each conjugated system, renaming one colour to the partner
/// `PARMCHK.DAT` declares for it (`equivalent_flag` 1 -> 2). That partner is now a
/// column of the rule — `AtdRule::alternate` — so it IS in the table, and the
/// principle this gate defends is untouched: an emitted type still has to come
/// from a row of the table being walked. What is widened is the row, not the rule.
/// A type from neither column is still an invention, and `cd` on a molecule with
/// no conjugation is still caught — by the parity columns, which is where a wrong
/// *choice* between two legal names belongs.
#[test]
fn no_assigned_type_is_absent_from_the_table_being_walked() {
    for (set, table, _) in SETS {
        let declared: HashSet<&str> = table
            .rules
            .iter()
            .flat_map(|rule| std::iter::once(rule.atom_type).chain(rule.alternate))
            .collect();
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
    for (set, _, _) in SETS {
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
/// The seven-column claim is only meaningful because the columns differ. If a
/// regenerated oracle ever collapsed two of them — an `-at` flag silently ignored
/// falls back to antechamber's default (`gaff`), which would make the AMBER column
/// a copy of the GFF one — the parity test for the collapsed column would still
/// pass while testing another table twice.
///
/// The tightest pair is GFF vs GFF2 (they disagree on only 6 of 37 molecules, all
/// via the amide/sulfonamide N split); every other pair disagrees on all 37.
#[test]
fn every_pair_of_oracle_type_columns_disagrees_somewhere() {
    for (i, (set_a, _, column_a)) in SETS.iter().enumerate() {
        for (set_b, _, column_b) in &SETS[i + 1..] {
            let differ = CASES
                .iter()
                .filter(|case| column_a(case) != column_b(case))
                .count();
            assert!(
                differ > 0,
                "{set_a:?} and {set_b:?} typed all {} oracle molecules identically — one of \
                 the two `-at` runs did not take effect, and its parity test is vacuous",
                CASES.len()
            );
        }
    }
}

/// The five symbolic tables disagree with BCC's numeric codes on EVERY molecule.
///
/// Separate from the pairwise gate because it is a stronger, cheaper-to-check
/// claim: `ATOMTYPE_BCC.DEF` types are integers (`11`, `16`, `91`) while GAS /
/// GFF / GFF2 / AMBER / SYBYL are symbols (`c3`, `ca`, `CT`, `C.ar`). A single
/// molecule where a symbolic column equals the numeric one would mean the oracle
/// column was written from the wrong run — and unlike the pairwise gate, this one
/// names which molecule.
#[test]
fn the_symbolic_tables_never_agree_with_the_numeric_bcc_codes() {
    for (set, _, column) in SETS
        .iter()
        .filter(|(set, _, _)| !matches!(set, AtdParameterSet::Bcc | AtdParameterSet::Abcg2))
    {
        let same: Vec<&str> = CASES
            .iter()
            .filter(|case| column(case) == case.bcc_atom_types)
            .map(|case| case.name)
            .collect();
        assert!(
            same.is_empty(),
            "{set:?} uses a symbolic type alphabet, so it cannot equal BCC's numeric \
             codes — yet it does on: {}",
            same.join(", ")
        );
    }
}
