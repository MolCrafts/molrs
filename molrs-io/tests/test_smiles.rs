//! RED tests for the SMARTS substructure matcher.
//!
//! These tests define the public-API contract that the session-2 matcher
//! implementation must satisfy. Every test that asserts a successful match
//! currently fails because [`SmartsPattern::compile`] returns
//! [`SmartsError::NotYetImplemented`]; a handful of contract tests verify
//! that the unimplemented states return clean errors rather than panicking.
//!
//! Source of truth: `docs/etkdgv3-port-spec.md` §7.1.

#![cfg(feature = "smiles")]

use molrs_io::smiles::smarts::{Match, SmartsError, SmartsPattern, SubstructureMatcher};
use molrs_io::smiles::{parse_smiles, to_atomistic};

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
fn recursive_smarts_compiles_and_matches_carbonyl_carbon() {
    // `[$(C=O)]` — match every atom whose environment is that of a
    // carbonyl carbon. On acetic acid (CH3-C(=O)-OH) only C1 (the carbon
    // double-bonded to an oxygen) satisfies the recursive environment.
    let pat = SmartsPattern::compile("[$(C=O)]").expect("compile recursive SMARTS");
    let target = mol("CC(=O)O");
    let matches = pat.find_all(&target).expect("find_all");
    assert_eq!(matches.len(), 1, "only the carbonyl carbon matches");
    assert_eq!(matches[0].0, vec![1]);
}

#[test]
fn malformed_smarts_returns_parse_error_not_panic() {
    let result = SmartsPattern::compile("[C;");
    assert!(
        matches!(
            result,
            Err(SmartsError::Parse(_)) | Err(SmartsError::NotYetImplemented)
        ),
        "malformed SMARTS must error, not panic; got {result:?}"
    );
}

#[test]
fn empty_smarts_returns_parse_error_not_panic() {
    let result = SmartsPattern::compile("");
    assert!(
        matches!(
            result,
            Err(SmartsError::Parse(_)) | Err(SmartsError::NotYetImplemented)
        ),
        "empty SMARTS must error, not panic; got {result:?}"
    );
}

// ===========================================================================
// Expanded Daylight §3 coverage — RED tests for the full matcher.
//
// Each test is labelled with the Daylight primitive(s) it exercises so the
// implementer can map coverage → spec.
// ===========================================================================

// ---------------------------------------------------------------------------
// Atom primitives — atomic number  (Daylight §3.1 `#<n>`)
// ---------------------------------------------------------------------------

#[test]
fn atomic_number_hash7_matches_every_nitrogen() {
    // `[#7]` — atomic number 7.
    let pat = SmartsPattern::compile("[#7]").expect("compile");
    let target = mol("CCN"); // one N at index 2
    let matches = pat.find_all(&target).expect("find_all");
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].0, vec![2]);
}

#[test]
fn atomic_number_hash8_matches_every_oxygen() {
    // `[#8]` — atomic number 8.
    let pat = SmartsPattern::compile("[#8]").expect("compile");
    let target = mol("CCO"); // one O at index 2
    let matches = pat.find_all(&target).expect("find_all");
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].0, vec![2]);
}

#[test]
fn atomic_number_hash16_matches_every_sulfur() {
    // `[#16]` — atomic number 16.
    let pat = SmartsPattern::compile("[#16]").expect("compile");
    let target = mol("CCS"); // one S at index 2
    let matches = pat.find_all(&target).expect("find_all");
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].0, vec![2]);
}

// ---------------------------------------------------------------------------
// Atom primitives — formal charge  (Daylight §3.1 `+<n>` / `-<n>`)
// ---------------------------------------------------------------------------

#[test]
fn positive_charge_query_matches_cation() {
    // `[+]` — formal charge +1.
    let pat = SmartsPattern::compile("[+]").expect("compile");
    let target = mol("[NH4+]"); // ammonium
    let matches = pat.find_all(&target).expect("find_all");
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].0, vec![0]);
}

#[test]
fn explicit_positive_charge_plus1_matches_cation() {
    // `[+1]` — formal charge +1 written explicitly.
    let pat = SmartsPattern::compile("[+1]").expect("compile");
    let target = mol("[NH4+]");
    let matches = pat.find_all(&target).expect("find_all");
    assert_eq!(matches.len(), 1);
}

#[test]
fn negative_charge_query_matches_anion() {
    // `[-]` — formal charge -1.
    let pat = SmartsPattern::compile("[-]").expect("compile");
    let target = mol("[OH-]"); // hydroxide
    let matches = pat.find_all(&target).expect("find_all");
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].0, vec![0]);
}

#[test]
fn double_negative_charge_matches_minus_two() {
    // `[--]` — formal charge -2 (shorthand for `[-2]`).
    let pat = SmartsPattern::compile("[--]").expect("compile");
    let target = mol("[O-2]"); // oxide
    let matches = pat.find_all(&target).expect("find_all");
    assert_eq!(matches.len(), 1);
}

#[test]
fn neutral_carbon_does_not_match_positive_charge_query() {
    // Negative: `[+]` must not hit neutral atoms.
    let pat = SmartsPattern::compile("[+]").expect("compile");
    let target = mol("CCO");
    let matches = pat.find_all(&target).expect("find_all");
    assert!(matches.is_empty());
}

// ---------------------------------------------------------------------------
// Atom primitives — hydrogen counts  (Daylight §3.1 `H<n>` / `h<n>`)
// ---------------------------------------------------------------------------

#[test]
fn total_h_count_h3_matches_methyl_carbons() {
    // `[CH3]` — total H-count 3 on carbon, i.e. methyl.
    let pat = SmartsPattern::compile("[CH3]").expect("compile");
    let target = mol("CCO"); // C0 (CH3), C1 (CH2), O (OH)
    let matches = pat.find_all(&target).expect("find_all");
    assert_eq!(matches.len(), 1, "only C0 has 3 hydrogens");
    assert_eq!(matches[0].0, vec![0]);
}

#[test]
fn implicit_h_count_lowercase_h1_matches_aromatic_ch() {
    // `[ch1]` — implicit H-count 1 on aromatic carbon.
    let pat = SmartsPattern::compile("[ch1]").expect("compile");
    let target = mol("c1ccccc1"); // every aromatic C carries one implicit H
    let matches = pat.find_all(&target).expect("find_all");
    assert_eq!(matches.len(), 6);
}

// ---------------------------------------------------------------------------
// Atom primitives — degree  (Daylight §3.1 `D<n>`)
// ---------------------------------------------------------------------------

#[test]
fn degree_query_d1_matches_only_terminal_atoms() {
    // `[D1]` — exactly one explicit bond (terminal heavy atom).
    let pat = SmartsPattern::compile("[D1]").expect("compile");
    let target = mol("CCO"); // C0:D1, C1:D2, O:D1
    let matches = sorted_matches(pat.find_all(&target).expect("find_all"));
    assert_eq!(matches.len(), 2);
    assert_eq!(matches[0].0, vec![0]);
    assert_eq!(matches[1].0, vec![2]);
}

#[test]
fn degree_query_d3_matches_branch_point() {
    // `[D3]` — exactly three explicit bonds (e.g. isobutane's central C).
    let pat = SmartsPattern::compile("[D3]").expect("compile");
    let target = mol("CC(C)C"); // isobutane — C1 is D3
    let matches = pat.find_all(&target).expect("find_all");
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].0, vec![1]);
}

// ---------------------------------------------------------------------------
// Atom primitives — valence  (Daylight §3.1 `v<n>`)
// ---------------------------------------------------------------------------

#[test]
fn valence_query_v4_matches_tetravalent_carbons() {
    // `[v4]` — total valence 4 (every sp3 carbon in methane/ethane).
    let pat = SmartsPattern::compile("[v4]").expect("compile");
    let target = mol("CC"); // both carbons tetravalent
    let matches = pat.find_all(&target).expect("find_all");
    assert_eq!(matches.len(), 2);
}

// ---------------------------------------------------------------------------
// Atom primitives — isotope + atom class  (Daylight §3.1 `<n>X`, `:<n>`)
// ---------------------------------------------------------------------------

#[test]
fn isotope_query_matches_c13() {
    // `[13C]` — isotope mass number 13 on carbon.
    let pat = SmartsPattern::compile("[13C]").expect("compile");
    let target = mol("[13C]CO"); // only atom 0 is 13C
    let matches = pat.find_all(&target).expect("find_all");
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].0, vec![0]);
}

#[test]
fn atom_class_query_matches_mapped_atom() {
    // `[C:1]` — carbon with atom map / class 1.
    let pat = SmartsPattern::compile("[C:1]").expect("compile");
    let target = mol("[CH4:1]"); // methane with class 1
    let matches = pat.find_all(&target).expect("find_all");
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].0, vec![0]);
}

// ---------------------------------------------------------------------------
// Atom logical operators  (Daylight §3.2: `!` > `&` > `,` > `;`)
// ---------------------------------------------------------------------------

#[test]
fn or_query_c_or_n_matches_both_elements() {
    // `[C,N]` — OR: matches carbon or nitrogen.
    let pat = SmartsPattern::compile("[C,N]").expect("compile");
    let target = mol("CCN"); // C0, C1, N2
    let matches = pat.find_all(&target).expect("find_all");
    assert_eq!(matches.len(), 3);
}

#[test]
fn high_and_query_c_and_x4_matches_sp3_carbons() {
    // `[C&X4]` — explicit high-precedence AND; equivalent to `[CX4]`.
    let pat = SmartsPattern::compile("[C&X4]").expect("compile");
    let target = mol("C=CC"); // only C2 is sp3
    let matches = pat.find_all(&target).expect("find_all");
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].0, vec![2]);
}

#[test]
fn low_and_query_c_ring_matches_only_ring_carbons() {
    // `[C;R]` — low-precedence AND: aliphatic carbon AND in a ring.
    let pat = SmartsPattern::compile("[C;R]").expect("compile");
    let target = mol("C1CCCCC1C"); // cyclohexane + methyl: 6 ring C + 1 exocyclic
    let matches = pat.find_all(&target).expect("find_all");
    assert_eq!(matches.len(), 6);
}

#[test]
fn not_query_excludes_matching_atoms() {
    // `[!C]` — NOT carbon. Methane has only one C → no matches.
    let pat = SmartsPattern::compile("[!C]").expect("compile");
    let target = mol("CCO"); // only O matches
    let matches = pat.find_all(&target).expect("find_all");
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].0, vec![2]);
}

#[test]
fn mixed_precedence_or_inside_low_and() {
    // `[C,N;R]` — (C OR N) AND-low IN-RING. Daylight precedence: OR binds tighter than `;`.
    let pat = SmartsPattern::compile("[C,N;R]").expect("compile");
    let target = mol("N1CCCCC1C"); // piperidine + exocyclic methyl
    // Ring atoms: N0 + C1..C5 = 6 ring heavy atoms; exocyclic C6 excluded.
    let matches = pat.find_all(&target).expect("find_all");
    assert_eq!(matches.len(), 6);
}

// ---------------------------------------------------------------------------
// Bond queries  (Daylight §3.3)
// ---------------------------------------------------------------------------

#[test]
fn triple_bond_query_matches_only_triple_bonds() {
    // `#` — triple bond. Acetylene has one C#C bond.
    let pat = SmartsPattern::compile("C#C").expect("compile");
    let target = mol("C#C");
    let matches = pat.find_all(&target).expect("find_all");
    assert_eq!(matches.len(), 2, "(0,1) and (1,0)");
}

#[test]
fn ring_bond_query_matches_only_ring_bonds() {
    // `@` — ring bond. Methylcyclohexane: 6 ring bonds, exocyclic C-C is not.
    let pat = SmartsPattern::compile("C@C").expect("compile");
    let target = mol("C1CCCCC1C"); // 6 ring bonds × 2 orderings = 12
    let matches = pat.find_all(&target).expect("find_all");
    assert_eq!(matches.len(), 12);
}

#[test]
fn or_bond_query_single_or_double_matches_both() {
    // `-,=` — bond OR: single or double.
    let pat = SmartsPattern::compile("C-,=C").expect("compile");
    let target = mol("C=CC"); // bonds: C0=C1 (double), C1-C2 (single)
    let matches = pat.find_all(&target).expect("find_all");
    assert_eq!(matches.len(), 4, "2 bonds × 2 orderings");
}

#[test]
fn not_bond_query_excludes_double_bonds() {
    // `!=` — bond NOT double.
    let pat = SmartsPattern::compile("C!=C").expect("compile");
    let target = mol("C=CC"); // only C1-C2 is not-double
    let matches = pat.find_all(&target).expect("find_all");
    assert_eq!(matches.len(), 2, "(1,2) and (2,1)");
}

// ---------------------------------------------------------------------------
// Ring primitives  (Daylight §3.1 `R<n>` / `r<n>`)
// ---------------------------------------------------------------------------

#[test]
fn ring_membership_r_any_matches_every_ring_atom() {
    // `[R]` — in any ring.
    let pat = SmartsPattern::compile("[R]").expect("compile");
    let target = mol("C1CCCCC1C"); // 6 ring atoms, 1 exocyclic
    let matches = pat.find_all(&target).expect("find_all");
    assert_eq!(matches.len(), 6);
}

#[test]
fn ring_membership_r0_matches_only_acyclic_atoms() {
    // `[R0]` — in zero rings (acyclic).
    let pat = SmartsPattern::compile("[R0]").expect("compile");
    let target = mol("C1CCCCC1C"); // only atom 6 is acyclic
    let matches = pat.find_all(&target).expect("find_all");
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].0, vec![6]);
}

#[test]
fn ring_membership_r2_matches_atoms_in_two_or_more_rings() {
    // `[R2]` — atom is a member of ≥2 SSSR rings (ring-fusion atoms).
    // Naphthalene's two fusion atoms sit in both rings.
    let pat = SmartsPattern::compile("[R2]").expect("compile");
    let target = mol("c1ccc2ccccc2c1"); // naphthalene — 2 fusion carbons
    let matches = pat.find_all(&target).expect("find_all");
    assert_eq!(matches.len(), 2);
}

#[test]
fn ring_size_r5_matches_only_five_membered_ring_atoms() {
    // `[r5]` — atom in a 5-membered ring.
    let pat = SmartsPattern::compile("[r5]").expect("compile");
    let target = mol("c1ccc2[nH]ccc2c1"); // indole: 6-ring (6 C) + 5-ring (N + 4 C sharing 2)
    // Indole 5-ring has 5 atoms; 2 are shared with the 6-ring but still count as r5.
    let matches = pat.find_all(&target).expect("find_all");
    assert_eq!(matches.len(), 5);
}

// ---------------------------------------------------------------------------
// Connectivity / branching  (Daylight §3 examples)
// ---------------------------------------------------------------------------

#[test]
fn carboxylic_acid_branch_pattern_matches_acetic_acid() {
    // `C(=O)O` — carbonyl carbon with =O branch + OH. Acetic acid CC(=O)O:
    // atom 1 is the carboxylic carbon, =O is atom 2, OH is atom 3.
    let pat = SmartsPattern::compile("C(=O)O").expect("compile");
    let target = mol("CC(=O)O");
    let matches = pat.find_all(&target).expect("find_all");
    // Match indices: (C=1, O=2, O=3). With O's swapped the pattern could also
    // hypothetically match, but atom 2 is double-bonded and atom 3 is single,
    // so only one orientation is valid.
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].0, vec![1, 2, 3]);
}

#[test]
fn cc_chain_finds_all_overlapping_pairs_in_propane() {
    // `CC` on propane: pairs (0,1), (1,0), (1,2), (2,1) = 4 overlapping matches.
    let pat = SmartsPattern::compile("CC").expect("compile");
    let target = mol("CCC");
    let matches = sorted_matches(pat.find_all(&target).expect("find_all"));
    assert_eq!(matches.len(), 4);
}

#[test]
fn cco_chain_on_ethanol_matches_both_orderings() {
    // `CCO` on ethanol CCO: pattern is directional, so only one ordering (0,1,2).
    // BUT reverse mapping onto a target is not a valid isomorphism here since
    // O cannot map onto C; only one match exists.
    let pat = SmartsPattern::compile("CCO").expect("compile");
    let target = mol("CCO");
    let matches = pat.find_all(&target).expect("find_all");
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].0, vec![0, 1, 2]);
}

#[test]
fn aromatic_ring_pattern_matches_benzene() {
    // `c1ccccc1` — aromatic 6-ring. Benzene: 6 ring atoms × 2 directions = 12.
    let pat = SmartsPattern::compile("c1ccccc1").expect("compile");
    let target = mol("c1ccccc1");
    let matches = pat.find_all(&target).expect("find_all");
    assert_eq!(matches.len(), 12);
}

#[test]
fn aromatic_ring_pattern_matches_toluene_ring_only() {
    // `c1ccccc1` on toluene: only the aromatic ring atoms match.
    let pat = SmartsPattern::compile("c1ccccc1").expect("compile");
    let target = mol("c1ccccc1C");
    let matches = pat.find_all(&target).expect("find_all");
    assert_eq!(
        matches.len(),
        12,
        "ring atoms in 2 directions; methyl excluded"
    );
}

#[test]
fn aromatic_ring_pattern_matches_naphthalene_both_rings() {
    // `c1ccccc1` on naphthalene should match both fused 6-rings.
    // Each ring has 12 directed mappings → 24 total.
    let pat = SmartsPattern::compile("c1ccccc1").expect("compile");
    let target = mol("c1ccc2ccccc2c1");
    let matches = pat.find_all(&target).expect("find_all");
    assert_eq!(matches.len(), 24);
}

// ---------------------------------------------------------------------------
// Negative tests — patterns that must NOT match
// ---------------------------------------------------------------------------

#[test]
fn nitrogen_query_finds_nothing_in_methane() {
    // `[N]` — element nitrogen. Methane has none.
    let pat = SmartsPattern::compile("[N]").expect("compile");
    let target = mol("C");
    let matches = pat.find_all(&target).expect("find_all");
    assert!(matches.is_empty());
}

#[test]
fn double_bond_finds_nothing_in_ethane() {
    // `C=C` — double bond. Ethane (CC) is all single-bonded.
    let pat = SmartsPattern::compile("C=C").expect("compile");
    let target = mol("CC");
    let matches = pat.find_all(&target).expect("find_all");
    assert!(matches.is_empty());
}

#[test]
fn ring_query_finds_nothing_in_n_hexane() {
    // `[R]` — in any ring. n-Hexane is acyclic.
    let pat = SmartsPattern::compile("[R]").expect("compile");
    let target = mol("CCCCCC");
    let matches = pat.find_all(&target).expect("find_all");
    assert!(matches.is_empty());
}
