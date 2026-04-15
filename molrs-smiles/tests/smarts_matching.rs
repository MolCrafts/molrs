//! RED tests for the SMARTS substructure matcher.
//!
//! These tests define the public-API contract that the session-2 matcher
//! implementation must satisfy. Every test that asserts a successful match
//! currently fails because [`SmartsPattern::compile`] returns
//! [`SmartsError::NotYetImplemented`]; a handful of contract tests verify
//! that the unimplemented states return clean errors rather than panicking.
//!
//! Source of truth: `docs/etkdgv3-port-spec.md` §7.1.

use molrs_smiles::smarts::{Match, SmartsError, SmartsPattern, SubstructureMatcher};
use molrs_smiles::{parse_smiles, to_atomistic};

/// Build a MolGraph target from a SMILES string. Centralized here so every
/// matcher test exercises the same ingestion path.
fn mol(smiles: &str) -> molrs::atomistic::Atomistic {
    let ir = parse_smiles(smiles).unwrap_or_else(|e| panic!("parse `{smiles}`: {e}"));
    to_atomistic(&ir).unwrap_or_else(|e| panic!("atomize `{smiles}`: {e}"))
}

/// Sort a match-list by each match's first atom index, so assertions are
/// order-independent.
fn sorted_matches(mut v: Vec<Match>) -> Vec<Match> {
    v.sort_by_key(|m| m.0.first().copied().unwrap_or(usize::MAX));
    v
}

// ---------------------------------------------------------------------------
// Atom queries
// ---------------------------------------------------------------------------

#[test]
fn element_query_carbon_matches_every_carbon_in_methane() {
    let pat = SmartsPattern::compile("[C]").expect("compile `[C]`");
    let target = mol("C");
    let matches = pat.find_all(&target).expect("find_all");
    assert_eq!(matches.len(), 1, "methane has exactly one carbon");
    assert_eq!(matches[0].0, vec![0]);
}

#[test]
fn element_query_carbon_matches_every_carbon_in_ethanol() {
    let pat = SmartsPattern::compile("[C]").expect("compile");
    let target = mol("CCO");
    let matches = sorted_matches(pat.find_all(&target).expect("find_all"));
    assert_eq!(matches.len(), 2, "ethanol has two carbons");
    assert_eq!(matches[0].0, vec![0]);
    assert_eq!(matches[1].0, vec![1]);
}

#[test]
fn hybridization_query_x4_matches_only_sp3_carbons() {
    let pat = SmartsPattern::compile("[C;X4]").expect("compile");
    let target = mol("C=CC"); // propene: C0=C1-C2; only C2 is sp3
    let matches = pat.find_all(&target).expect("find_all");
    assert_eq!(matches.len(), 1, "only the terminal methyl is sp3");
    assert_eq!(matches[0].0, vec![2]);
}

#[test]
fn aromaticity_query_lowercase_c_matches_only_aromatic_carbons() {
    let pat = SmartsPattern::compile("[c]").expect("compile");
    let target = mol("c1ccccc1C"); // toluene: 6 aromatic + 1 aliphatic carbon
    let matches = pat.find_all(&target).expect("find_all");
    assert_eq!(matches.len(), 6, "benzene ring carbons only");
}

#[test]
fn ring_size_query_r6_matches_only_atoms_in_six_membered_rings() {
    let pat = SmartsPattern::compile("[c;r6]").expect("compile");
    let target = mol("c1ccc2ccccc2c1"); // naphthalene: all 10 carbons in r6
    let matches = pat.find_all(&target).expect("find_all");
    assert_eq!(matches.len(), 10);
}

#[test]
fn atomic_number_query_hash6_matches_every_carbon() {
    let pat = SmartsPattern::compile("[#6]").expect("compile");
    let target = mol("CCN"); // two carbons, one nitrogen
    let matches = pat.find_all(&target).expect("find_all");
    assert_eq!(matches.len(), 2);
}

// ---------------------------------------------------------------------------
// Bond queries
// ---------------------------------------------------------------------------

#[test]
fn single_bond_query_matches_only_single_bonds() {
    let pat = SmartsPattern::compile("C-C").expect("compile");
    let target = mol("C=CC"); // C=C-C: only the C-C (indices 1-2) is single
    let matches = pat.find_all(&target).expect("find_all");
    assert_eq!(matches.len(), 2, "ordered pairs (1,2) and (2,1)");
}

#[test]
fn double_bond_query_matches_only_double_bonds() {
    let pat = SmartsPattern::compile("C=C").expect("compile");
    let target = mol("C=CC");
    let matches = pat.find_all(&target).expect("find_all");
    assert_eq!(matches.len(), 2, "ordered pairs (0,1) and (1,0)");
}

#[test]
fn aromatic_bond_query_matches_only_aromatic_bonds() {
    let pat = SmartsPattern::compile("c:c").expect("compile");
    let target = mol("c1ccccc1"); // 6 aromatic bonds, each matched both directions
    let matches = pat.find_all(&target).expect("find_all");
    assert_eq!(matches.len(), 12);
}

#[test]
fn any_bond_query_matches_every_bond() {
    let pat = SmartsPattern::compile("*~*").expect("compile");
    let target = mol("CCO"); // 2 bonds, each matched both directions
    let matches = pat.find_all(&target).expect("find_all");
    assert_eq!(matches.len(), 4);
}

// ---------------------------------------------------------------------------
// Connectivity
// ---------------------------------------------------------------------------

#[test]
fn three_atom_chain_matches_every_triple_in_propane() {
    let pat = SmartsPattern::compile("CCC").expect("compile");
    let target = mol("CCC"); // propane: one linear chain, two orderings
    let matches = pat.find_all(&target).expect("find_all");
    assert_eq!(matches.len(), 2, "(0,1,2) and (2,1,0)");
}

#[test]
fn ring_pattern_matches_cyclohexane() {
    let pat = SmartsPattern::compile("C1CCCCC1").expect("compile");
    let target = mol("C1CCCCC1");
    let matches = pat.find_all(&target).expect("find_all");
    assert!(
        !matches.is_empty(),
        "cyclohexane should match the ring pattern"
    );
}

#[test]
fn ring_pattern_does_not_match_linear_chain() {
    let pat = SmartsPattern::compile("C1CCCCC1").expect("compile");
    let target = mol("CCCCCC"); // n-hexane, linear — no ring
    let matches = pat.find_all(&target).expect("find_all");
    assert!(matches.is_empty());
}

// ---------------------------------------------------------------------------
// Contract: unimplemented features return clean errors, never panic
// ---------------------------------------------------------------------------

#[test]
fn recursive_smarts_returns_not_yet_implemented_for_now() {
    // Session 2 may implement this; until then it must error cleanly.
    let result = SmartsPattern::compile("[$(C=O)]");
    match result {
        Err(SmartsError::NotYetImplemented) | Err(SmartsError::Parse(_)) => {}
        Ok(_) => panic!("recursive SMARTS should not compile yet"),
    }
}

#[test]
fn malformed_smarts_returns_parse_error_not_panic() {
    let result = SmartsPattern::compile("[C;");
    assert!(
        matches!(result, Err(SmartsError::Parse(_)) | Err(SmartsError::NotYetImplemented)),
        "malformed SMARTS must error, not panic; got {result:?}"
    );
}

#[test]
fn empty_smarts_returns_parse_error_not_panic() {
    let result = SmartsPattern::compile("");
    assert!(
        matches!(result, Err(SmartsError::Parse(_)) | Err(SmartsError::NotYetImplemented)),
        "empty SMARTS must error, not panic; got {result:?}"
    );
}
