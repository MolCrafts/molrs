//! chem-perceive-10 ac-003 — the parmchk2 missing-parameter oracle. **RED until 11.**
//!
//! Spec 09 landed `gaff.dat` / `gaff2.dat` as compiled tables and matched them
//! **exactly**: a term no wildcard-free row covers comes back as a
//! [`MissingTerm`], not a parameter. That is the honest half of the job, and it
//! leaves the other half visible — 153 dihedrals (gaff) / 139 (gaff2) that molrs
//! cannot parameterise and AMBER can.
//!
//! `parmchk2` is the tool that closes exactly that gap, so it is the oracle for
//! closing it here. Fed the same typed molecule, it writes an frcmod naming every
//! term the table does not cover, the parameter it estimated, the term it copied,
//! and the **penalty score** it charged. The two `*_parmchk_terms` columns of
//! [`antechamber_oracle`](super::antechamber_oracle) are those frcmods, verbatim.
//!
//! # What the oracle actually says, and why it is smaller than the miss list
//!
//! parmchk2 emits **43 terms under gaff and 58 under gaff2** — not 153 / 139. The
//! difference is not a disagreement, it is the wildcard tier: `X -c3-c3-X` matches
//! any quartet whose inner pair is `c3-c3`, so LEaP finds those parameters in
//! `gaff.dat` itself and parmchk2 has nothing to write. It writes a row only when
//! **no** row matches — wildcards included — and must then substitute atom types
//! and pay a penalty. `X -c3-ce-X` does not exist, which is why methyl
//! methacrylate's `hc-c3-ce-c2` is in the frcmod and `hc-c3-c3-hc` is not.
//!
//! So reproducing this column takes BOTH halves of the cascade, and the tests below
//! separate them:
//!
//! - **wildcard matching** — [`molrs_estimates_exactly_the_terms_parmchk2_estimates`]
//!   fails if molrs estimates a term parmchk2 found in the table. An estimator
//!   bolted on without the wildcard tier passes every other test here and fails
//!   that one, because it would fabricate an analogy for ~145 terms AMBER simply
//!   looks up.
//! - **analogy + penalty** — [`molrs_reproduces_every_parmchk2_parameter_value`]
//!   and [`molrs_reproduces_every_parmchk2_penalty_tier`] pin what it produces for
//!   the terms that genuinely have no row.
//!
//! # The oracle is torsion-only, and that is a finding
//!
//! Every BOND, ANGLE, MASS and NONBON term of all 37 molecules is an exact hit in
//! both tables — parmchk2's frcmods contain not one of them. Hence
//! [`the_oracle_estimates_only_torsions`] and
//! [`molrs_misses_only_torsions`], which pin the two sides of that
//! agreement, and hence the warning in [`CASES`]: **this oracle does not exercise
//! the empirical (Badger / Wang2004 Eq.5) fallback at all.** Nothing in these 37
//! molecules reaches it. Its formulas are pinned by the unit tests in
//! `ff::typifier::estimate` against published GAFF values; passing the tests in this
//! file says nothing whatsoever about them.

use std::collections::BTreeSet;

use molrs::Atomistic;
use molrs::ff::forcefield::gaff::{GaffError, GaffParameterSet, MissingTerm, gaff_forcefield};
use molrs::ff::forcefield::{DihedralType, ForceField, ImproperType, StyleDefs};
use molrs::ff::typifier::Typifier;
use molrs::ff::typifier::atd::{AtdParameterSet, AtdTypifier};
use molrs::ff::typifier::estimate::PenaltyTier;

use super::antechamber_oracle::{AntechamberCase, CASES, ParmchkTerm};
use super::oracle_mol::{build_case, report};

/// Which oracle column a parm table's estimates live in.
type OracleColumn = fn(&'static AntechamberCase) -> &'static [ParmchkTerm];

/// The parm table, the `ATOMTYPE_*.DEF` that types a molecule for it, and the
/// parmchk2 column that pairing produced.
///
/// Typing with one table and parameterising with the other is the category error
/// `GaffParameterSet` documents; keeping the pair in one list is what stops a test
/// from committing it by accident.
const SETS: [(GaffParameterSet, AtdParameterSet, OracleColumn); 2] = [
    (GaffParameterSet::Gaff, AtdParameterSet::Gff, |case| {
        case.gaff_parmchk_terms
    }),
    (GaffParameterSet::Gaff2, AtdParameterSet::Gff2, |case| {
        case.gaff2_parmchk_terms
    }),
];

/// Every (molecule, parm table) pair — the denominator of every report below.
const PAIRS: usize = 37 * 2;

/// Table transcription, so the **exact** tolerance column.
///
/// Both sides carry the same decimal literals: `gaff.dat`'s torsions have at most
/// three decimals and its impropers one, and parmchk2 prints them without loss. The
/// only arithmetic between the fixture and the force field is AMBER's own
/// `barrier / divisor` and `degrees.to_radians()`, both of which molrs performs on
/// the same inputs. Anything above f64 round-off is therefore a real disagreement,
/// not a precision artefact.
const TOL: f64 = 1e-10;

/// The provenance key an estimated term carries (`ff::typifier::estimate`, module
/// docs): the flag that says "this parameter was estimated, not matched".
const ESTIMATED: &str = "estimated";

/// The provenance key holding the estimate's total penalty score.
const ESTIMATE_PENALTY: &str = "estimate_penalty";

// ---------------------------------------------------------------------------
// oracle shape — what parmchk2 says there is to do
// ---------------------------------------------------------------------------

/// parmchk2 estimates nothing but torsions for these 37 molecules.
///
/// The whole BOND / ANGLE / MASS / NONBON side of both tables covers this chemistry
/// exactly. This is the fixture's own shape, asserted so that the two tests which
/// LEAN on it — [`molrs_misses_only_torsions`] and, by omission, every
/// claim about the empirical fallback — cannot be quietly invalidated by a
/// regenerated oracle.
#[test]
fn the_oracle_estimates_only_torsions() {
    let mut offenders = Vec::new();
    let mut terms = 0usize;
    for case in CASES {
        for (set, _, column) in SETS {
            for term in column(case) {
                terms += 1;
                if !matches!(term.kind, "DIHE" | "IMPROPER") {
                    offenders.push(format!(
                        "  {} [{}]: {} {}",
                        case.name,
                        set.name(),
                        term.kind,
                        term.types.join("-")
                    ));
                }
            }
        }
    }

    assert!(
        terms > 0,
        "the parmchk2 columns are empty — the oracle would be vacuous"
    );
    assert!(
        offenders.is_empty(),
        "parmchk2 now estimates a non-torsion term for the oracle molecules.\n\
         Every bond, angle, mass and NONBON term used to be an exact hit in both \
         tables, and two things in this file rest on that: the `molrs_misses_only_\
         torsions` gate, and the claim that this oracle does NOT exercise the \
         empirical bond/angle formulas. Both need revisiting:\n{}",
        offenders.join("\n")
    );
}

// ---------------------------------------------------------------------------
// the exact half — molrs and parmchk2 already agree here (GREEN)
// ---------------------------------------------------------------------------

/// molrs is short of exactly the same *kind* of term parmchk2 is: torsions only.
///
/// The counterpart of [`the_oracle_estimates_only_torsions`], from molrs's side. It
/// is the corroboration that spec 09's exact matcher and parmchk2 read the same
/// tables the same way — and, going forward, the guard that spec 11's fallback does
/// not start inventing bond or angle parameters where the table already had one.
#[test]
fn molrs_misses_only_torsions() {
    let mut failures = Vec::new();
    for case in CASES {
        for (set, atd, _) in SETS {
            let Err(GaffError::Missing { terms, .. }) = parameterise(case, atd, set) else {
                continue; // Ok, or a defect the other tests report
            };
            let bad: Vec<String> = terms
                .iter()
                .filter(|t| !matches!(t, MissingTerm::Dihedral(_)))
                .map(ToString::to_string)
                .collect();
            if !bad.is_empty() {
                failures.push(format!(
                    "  {} [{}]: {}",
                    case.name,
                    set.name(),
                    bad.join(", ")
                ));
            }
        }
    }
    report(
        "molrs reports a missing term parmchk2 had no trouble with",
        &failures,
        PAIRS,
    );
}

// ---------------------------------------------------------------------------
// the estimated half — RED until chem-perceive-11
// ---------------------------------------------------------------------------

/// Every oracle molecule comes out of molrs fully parameterised.
///
/// The headline of spec 11: a GAFF-typed molecule gets a force field, not an error
/// listing 153 torsions. RED today by construction — spec 09 is exact-match-only.
#[test]
fn molrs_parameterises_every_oracle_molecule() {
    let mut failures = Vec::new();
    for case in CASES {
        for (set, atd, _) in SETS {
            if let Err(err) = parameterise(case, atd, set) {
                let detail = match &err {
                    GaffError::Missing { terms, .. } => format!("{} term(s) missing", terms.len()),
                    other => other.to_string(),
                };
                failures.push(format!("  {} [{}]: {detail}", case.name, set.name()));
            }
        }
    }
    report("molrs cannot parameterise the molecule", &failures, PAIRS);
}

/// molrs falls back on exactly the terms parmchk2 falls back on — no more, no less.
///
/// This is the wildcard tier's test, and it is the one an estimator bolted on
/// without it fails. `X -c3-c3-X` covers `hc-c3-c3-hc`; parmchk2 matches it and
/// stays silent, so molrs must too. An estimator reached for every term spec 09
/// calls missing would instead *substitute atom types* for ~145 terms AMBER simply
/// looks up — producing plausible numbers, passing every value test below (those
/// only check the terms parmchk2 DID estimate), and silently disagreeing with AMBER
/// on the vast majority of the molecule.
///
/// The converse direction matters just as much: a term parmchk2 had to estimate and
/// molrs did not is a term molrs matched to a row parmchk2 could not find — i.e. a
/// wildcard tier that is too eager.
///
/// The estimated set is read off the provenance convention `ff::typifier::estimate`
/// documents (`estimated` = 1.0 on the term's params); a force field that estimates
/// without recording it is not auditable and fails here.
#[test]
fn molrs_estimates_exactly_the_terms_parmchk2_estimates() {
    let mut failures = Vec::new();
    for case in CASES {
        for (set, atd, column) in SETS {
            let ff = match parameterise(case, atd, set) {
                Ok(ff) => ff,
                Err(err) => {
                    failures.push(format!("  {} [{}]: {err}", case.name, set.name()));
                    continue;
                }
            };

            let want: BTreeSet<String> = column(case).iter().map(canonical_term).collect();
            let got: BTreeSet<String> = estimated_terms(&ff);

            let fabricated: Vec<&String> = got.difference(&want).collect();
            let missed: Vec<&String> = want.difference(&got).collect();
            if !fabricated.is_empty() || !missed.is_empty() {
                failures.push(format!(
                    "  {} [{}]: estimated but parmchk2 matched it in the table: {:?}; \
                     parmchk2 estimated but molrs did not: {:?}",
                    case.name,
                    set.name(),
                    fabricated,
                    missed
                ));
            }
        }
    }
    report(
        "molrs's estimated-term set disagrees with parmchk2's",
        &failures,
        PAIRS,
    );
}

/// Every parameter parmchk2 estimated comes out of molrs with the same value.
///
/// Values are compared in the force field's units, converted from the fixture's
/// upstream ones exactly as `forcefield::gaff` converts the table: `k = barrier /
/// divisor` per cosine term, phases and θ₀ in radians, and a torsion's negative
/// periodicity read as AMBER's "another term follows" flag rather than as a
/// multiplicity. The cosine terms of one torsion are compared as a SET: their order
/// within the group is the table's, and nothing downstream depends on it.
#[test]
fn molrs_reproduces_every_parmchk2_parameter_value() {
    let mut failures = Vec::new();
    for case in CASES {
        for (set, atd, column) in SETS {
            let ff = match parameterise(case, atd, set) {
                Ok(ff) => ff,
                Err(err) => {
                    failures.push(format!("  {} [{}]: {err}", case.name, set.name()));
                    continue;
                }
            };
            for group in term_groups(column(case)) {
                if let Err(why) = check_value(&ff, &group) {
                    failures.push(format!("  {} [{}]: {why}", case.name, set.name()));
                }
            }
        }
    }
    report(
        "molrs's estimated parameter differs from parmchk2's",
        &failures,
        PAIRS,
    );
}

/// Every estimate lands in the same confidence band parmchk2 put it in.
///
/// The penalty score is not decoration: it is what tells a user whether a parameter
/// is usable as-is (`< 10`), needs review (`10–50`), or needs re-fitting (`> 50`) —
/// [`PenaltyTier`]'s three CGenFF bands. Reproducing the value while charging the
/// wrong penalty ships a parameter with a false confidence label, which is worse
/// than shipping no parameter at all, so the tier is asserted separately from the
/// value.
///
/// Compared as a TIER, not as a float: the bands are the contract, and pinning
/// parmchk2's exact score would pin the arithmetic of its weight table rather than
/// the meaning of the result. Terms parmchk2 scored are the only ones checked — it
/// prints no score for an improper it took from the default (a default is not a
/// substitution, so there is nothing to score), nor on the leading rows of a
/// multi-cosine group, which it scores as a whole.
#[test]
fn molrs_reproduces_every_parmchk2_penalty_tier() {
    let mut failures = Vec::new();
    let mut checked = 0usize;
    for case in CASES {
        for (set, atd, column) in SETS {
            let ff = match parameterise(case, atd, set) {
                Ok(ff) => ff,
                Err(err) => {
                    failures.push(format!("  {} [{}]: {err}", case.name, set.name()));
                    continue;
                }
            };
            for group in term_groups(column(case)) {
                let Some(want) = group.iter().find_map(|t| t.penalty) else {
                    continue; // parmchk2 charged no penalty for this term
                };
                checked += 1;
                let head = group[0];
                let Some(got) = estimate_penalty(&ff, head) else {
                    failures.push(format!(
                        "  {} [{}]: {} {} carries no `{ESTIMATE_PENALTY}` — parmchk2 \
                         charged {want:.1} ({:?}) for it",
                        case.name,
                        set.name(),
                        head.kind,
                        head.types.join("-"),
                        PenaltyTier::of(want),
                    ));
                    continue;
                };
                if PenaltyTier::of(got) != PenaltyTier::of(want) {
                    failures.push(format!(
                        "  {} [{}]: {} {} scored {got:.1} ({:?}), parmchk2 scored \
                         {want:.1} ({:?}) — `{}`",
                        case.name,
                        set.name(),
                        head.kind,
                        head.types.join("-"),
                        PenaltyTier::of(got),
                        PenaltyTier::of(want),
                        head.comment,
                    ));
                }
            }
        }
    }

    assert!(
        checked > 0 || !failures.is_empty(),
        "no penalty was compared — the gate would be vacuous"
    );
    report(
        "molrs's estimate lands in the wrong confidence band",
        &failures,
        PAIRS,
    );
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Type the oracle molecule with the ATD table, then parameterise it from the parm
/// table it pairs with.
///
/// The molecule is built from the INPUT half of the case only (element, geometry,
/// bond order, formal charge) — never from antechamber's answers — because the
/// shared builder is the guard that keeps every oracle test honest.
fn parameterise(
    case: &AntechamberCase,
    atd: AtdParameterSet,
    set: GaffParameterSet,
) -> Result<ForceField, GaffError> {
    let (mol, _) = build_case(case);
    let typed: Atomistic = AtdTypifier::new(atd)
        .typify(&mol)
        .unwrap_or_else(|e| panic!("{}: ATD typing failed: {e}", case.name));
    gaff_forcefield(set, &typed).map(|(_, ff)| ff)
}

/// The rows of one frcmod term, grouped.
///
/// A torsion with several cosine terms is several frcmod rows for ONE force-field
/// type, and parmchk2 prints its penalty on the last of them, so the rows have to be
/// reassembled before either can be compared.
fn term_groups(terms: &'static [ParmchkTerm]) -> Vec<Vec<&'static ParmchkTerm>> {
    let mut groups: Vec<Vec<&ParmchkTerm>> = Vec::new();
    for term in terms {
        match groups
            .iter_mut()
            .find(|g| g[0].kind == term.kind && g[0].types == term.types)
        {
            Some(group) => group.push(term),
            None => groups.push(vec![term]),
        }
    }
    groups
}

/// A term's identity, orientation-independent: `KIND i-j-k-l`.
///
/// A bonded term and its reverse are the same term, so the two spellings must
/// canonicalise to one. An improper is NOT reversible — its centre is the third
/// slot and the other three are an unordered set — so it canonicalises by sorting
/// the peripherals around the fixed centre.
fn canonical_term(term: &ParmchkTerm) -> String {
    canonical(term.kind, term.types)
}

fn canonical(kind: &str, types: &[&str]) -> String {
    let key = if kind == "IMPROPER" {
        let mut peripherals = [types[0], types[1], types[3]];
        peripherals.sort_unstable();
        format!(
            "{}-{}-{}-{}",
            peripherals[0], peripherals[1], types[2], peripherals[2]
        )
    } else {
        let forward = types.join("-");
        let backward = types.iter().rev().copied().collect::<Vec<_>>().join("-");
        forward.min(backward)
    };
    format!("{kind} {key}")
}

/// Every term of `ff` that carries the `estimated` provenance flag, canonicalised.
fn estimated_terms(ff: &ForceField) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for style in ff.styles() {
        match &style.defs {
            StyleDefs::Bond(types) => {
                for t in types
                    .iter()
                    .filter(|t| is_estimated(t.params.get(ESTIMATED)))
                {
                    out.insert(canonical("BOND", &[&t.itom, &t.jtom]));
                }
            }
            StyleDefs::Angle(types) => {
                for t in types
                    .iter()
                    .filter(|t| is_estimated(t.params.get(ESTIMATED)))
                {
                    out.insert(canonical("ANGLE", &[&t.itom, &t.jtom, &t.ktom]));
                }
            }
            StyleDefs::Dihedral(types) => {
                for t in types
                    .iter()
                    .filter(|t| is_estimated(t.params.get(ESTIMATED)))
                {
                    out.insert(canonical("DIHE", &[&t.itom, &t.jtom, &t.ktom, &t.ltom]));
                }
            }
            StyleDefs::Improper(types) => {
                for t in types
                    .iter()
                    .filter(|t| is_estimated(t.params.get(ESTIMATED)))
                {
                    out.insert(canonical("IMPROPER", &[&t.itom, &t.jtom, &t.ktom, &t.ltom]));
                }
            }
            _ => {}
        }
    }
    out
}

/// The `estimated` flag is numeric `1.0` (the provenance convention has no bool).
fn is_estimated(flag: Option<f64>) -> bool {
    flag.is_some_and(|v| v != 0.0)
}

/// The penalty molrs charged for the force-field type covering `term`.
fn estimate_penalty(ff: &ForceField, term: &ParmchkTerm) -> Option<f64> {
    match term.kind {
        "DIHE" => find_dihedral(ff, term.types)?.params.get(ESTIMATE_PENALTY),
        "IMPROPER" => find_improper(ff, term.types)?.params.get(ESTIMATE_PENALTY),
        _ => None,
    }
}

/// Compare one frcmod term against the force field molrs built.
fn check_value(ff: &ForceField, group: &[&ParmchkTerm]) -> Result<(), String> {
    let head = group[0];
    let label = format!("{} {}", head.kind, head.types.join("-"));
    match head.kind {
        "DIHE" => {
            let ty = find_dihedral(ff, head.types)
                .ok_or_else(|| format!("{label} is not in the force field at all"))?;
            // AMBER: k = PK / IDIVF, phase in degrees, |PN| the multiplicity (the
            // sign is the "another term follows" flag, not a periodicity).
            let want = sorted_terms(group.iter().map(|t| {
                (
                    t.values[1] / t.values[0],
                    t.values[3].abs(),
                    t.values[2].to_radians(),
                )
            }));
            let got = sorted_terms((1..).map_while(|m| {
                Some((
                    ty.params.get(&format!("k{m}"))?,
                    ty.params.get(&format!("n{m}"))?,
                    ty.params.get(&format!("d{m}"))?,
                ))
            }));
            compare(&label, &want, &got, head)
        }
        "IMPROPER" => {
            let ty = find_improper(ff, head.types)
                .ok_or_else(|| format!("{label} is not in the force field at all"))?;
            // An improper's barrier is never divided (there is no IDIVF column).
            let want = sorted_terms(
                group
                    .iter()
                    .map(|t| (t.values[0], t.values[2], t.values[1].to_radians())),
            );
            let got = sorted_terms(
                [(ty.params.get("k"), ty.params.get("n"), ty.params.get("d"))]
                    .into_iter()
                    .filter_map(|(k, n, d)| Some((k?, n?, d?))),
            );
            compare(&label, &want, &got, head)
        }
        // Nothing else occurs; `the_oracle_estimates_only_torsions` is what says so.
        other => Err(format!("{label}: unhandled frcmod kind `{other}`")),
    }
}

/// The cosine terms of one torsion, in a comparable order.
fn sorted_terms(terms: impl Iterator<Item = (f64, f64, f64)>) -> Vec<(f64, f64, f64)> {
    let mut out: Vec<(f64, f64, f64)> = terms.collect();
    out.sort_by(|a, b| a.partial_cmp(b).expect("no NaN in a force-field parameter"));
    out
}

fn compare(
    label: &str,
    want: &[(f64, f64, f64)],
    got: &[(f64, f64, f64)],
    term: &ParmchkTerm,
) -> Result<(), String> {
    if want.len() != got.len() {
        return Err(format!(
            "{label}: parmchk2 wrote {} cosine term(s), molrs built {} — `{}`",
            want.len(),
            got.len(),
            term.comment
        ));
    }
    for (w, g) in want.iter().zip(got) {
        let off = (w.0 - g.0).abs() > TOL || (w.1 - g.1).abs() > TOL || (w.2 - g.2).abs() > TOL;
        if off {
            return Err(format!(
                "{label}: molrs (k={:.6}, n={}, d={:.6} rad) vs parmchk2 \
                 (k={:.6}, n={}, d={:.6} rad) — parmchk2 got there by `{}`",
                g.0, g.1, g.2, w.0, w.1, w.2, term.comment
            ));
        }
    }
    Ok(())
}

/// The force field's dihedral type for `types`, in either orientation.
fn find_dihedral<'a>(ff: &'a ForceField, types: &[&str]) -> Option<&'a DihedralType> {
    let StyleDefs::Dihedral(defs) = &ff.get_style("dihedral", "periodic")?.defs else {
        return None;
    };
    let want = canonical("DIHE", types);
    defs.iter()
        .find(|t| canonical("DIHE", &[&t.itom, &t.jtom, &t.ktom, &t.ltom]) == want)
}

/// The force field's improper type for `types` — centre third, peripherals unordered.
fn find_improper<'a>(ff: &'a ForceField, types: &[&str]) -> Option<&'a ImproperType> {
    let StyleDefs::Improper(defs) = &ff.get_style("improper", "periodic")?.defs else {
        return None;
    };
    let want = canonical("IMPROPER", types);
    defs.iter()
        .find(|t| canonical("IMPROPER", &[&t.itom, &t.jtom, &t.ktom, &t.ltom]) == want)
}
