//! OPLS-AA bonded-parameter matching (bonds / angles / dihedrals).
//!
//! Given a typed [`Atomistic`] (atoms carry `type` = `opls_NNN` and `class` from
//! [`typify_atoms`](super::typify_atoms)), this module enumerates every bond /
//! angle / dihedral and resolves the most specific matching bonded type from the
//! potential [`ForceField`](crate::ff::forcefield::ForceField)'s `bond` /
//! `angle` / `dihedral` style tables, writing the winning type's numeric params
//! onto the term.
//!
//! # Why a bespoke matcher (not [`Style::get_bondtype`])
//!
//! OPLS-AA keys its bonded forces on **class** names (`CT`, `HC`, …) and uses
//! wildcard end atoms heavily (`X-CT-CT-X`). The force field's
//! [`get_bondtype`](crate::ff::forcefield::Style::get_bondtype) is *exact-key +
//! symmetric only* — no wildcard, no specificity ordering — so it cannot pick
//! the best of several overlapping candidates. Keeping `get_bondtype` exact is
//! deliberate (see the spec's "Out of scope"); the specificity ranking is the
//! **typifier's** job and lives here.
//!
//! This is a 1:1 Rust replica of molpy's
//! `typifier/atomistic.ForceField{Bond,Angle,Dihedral}Typifier`
//! (`_end_score` / `_sequence_score` + `(score, layer)` ranking).
//!
//! # Wildcard vocabulary
//!
//! molpy normalizes an empty / absent class attribute to `"*"` at XML read time;
//! the molrs [`OplsXmlReader`](crate::ff::forcefield::readers::opls::OplsXmlReader)
//! transcribes the `class*` attributes verbatim, so an OPLS wildcard end arrives
//! here as the **empty string** `""`. To stay bit-for-bit compatible with
//! molpy's matcher, [`end_score`] treats `""`, `"*"`, and `"X"` all as the
//! score-0 wildcard.
//!
//! # No-match seam (estimator)
//!
//! A term that matches no candidate is routed through the [`Estimator`] seam: if
//! an estimator is attached, it is asked to estimate the missing params;
//! otherwise the configured strict policy applies (`strict=true` → `Err`,
//! `strict=false` → the term is left unparametrized). The
//! [`ff-parameter-estimator`] is not built yet; this is the opt-in injection
//! point it will wire into.
//!
//! [`ff-parameter-estimator`]: https://github.com/MolCrafts/molrs

use std::collections::HashMap;

use molrs::{AtomId, Atomistic};

use crate::ff::forcefield::{ForceField, Params, StyleDefs};

use super::meta::OplsTypingMeta;

/// Specificity of one bonded-type *end pattern* against one atom.
///
/// Mirrors molpy's `_end_score`:
/// - exact `opls_NNN` type match → `3`;
/// - class match → `1`;
/// - wildcard (`""`, `"*"`, or `"X"`) → `0`;
/// - no match → `None`.
///
/// `pattern` is the bonded-type endpoint name (a class name, a wildcard, or — in
/// principle — a type name). `atom_type` is the atom's `opls_NNN` type;
/// `atom_class` is its resolved class (`None` if the type has no class mapping).
fn end_score(pattern: &str, atom_type: &str, atom_class: Option<&str>) -> Option<i64> {
    if is_wildcard(pattern) {
        return Some(0);
    }
    if pattern == atom_type {
        return Some(3);
    }
    if let Some(cls) = atom_class
        && pattern == cls
    {
        return Some(1);
    }
    None
}

/// Whether a bonded-type endpoint name is a wildcard. OPLS XML wildcards arrive
/// as `""` (verbatim from the reader); `"*"` and `"X"` are accepted for parity
/// with molpy / generic force-field conventions.
fn is_wildcard(pattern: &str) -> bool {
    pattern.is_empty() || pattern == "*" || pattern == "X"
}

/// Best specificity of an ordered bonded-term pattern against an ordered atom
/// list, trying forward and end-for-end-reversed orientations.
///
/// Mirrors molpy's `_sequence_score`: bonded terms are symmetric under
/// reversal, so both orientations are scored and the larger total returned. An
/// orientation in which any end scores `None` is rejected; the whole call
/// returns `None` only if *both* orientations have a non-matching end.
///
/// `atoms` is `[(type, class), ...]`, the same length as `pattern`.
fn sequence_score(pattern: &[&str], atoms: &[(&str, Option<&str>)]) -> Option<i64> {
    debug_assert_eq!(pattern.len(), atoms.len());
    let score_in = |order: &mut dyn Iterator<Item = &(&str, Option<&str>)>| -> Option<i64> {
        let mut total = 0;
        for (pat, (at_type, at_class)) in pattern.iter().zip(order) {
            total += end_score(pat, at_type, *at_class)?;
        }
        Some(total)
    };
    let forward = score_in(&mut atoms.iter());
    let reversed = score_in(&mut atoms.iter().rev());
    match (forward, reversed) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (Some(a), None) | (None, Some(a)) => Some(a),
        (None, None) => None,
    }
}

/// The winning bonded-type match: its force-field type *name* (e.g. `"CT-CT"`)
/// and the numeric params to write onto the term. The name is written as the
/// term's `type` label so the generic `ForceField::to_potentials` path can
/// re-resolve params from the bond/angle/dihedral style at compile time.
struct Match<'a> {
    name: &'a str,
    params: &'a Params,
}

/// A bonded-term candidate from the force field: its type name, endpoint class
/// pattern (length 2 / 3 / 4), the precomputed overlay layer, and the params to
/// write on a match.
struct Candidate {
    /// Force-field type name (e.g. `"CT-CT"`), written as the term's `type`.
    name: String,
    /// Endpoint pattern (class names / wildcards), e.g. `["X", "CT", "CT", "X"]`.
    pattern: Vec<String>,
    /// Overlay layer = max layer over the pattern's classes (CL&P / CL&Pol).
    layer: u32,
    /// Numeric params (e.g. `k0`/`r0`) to copy onto the matched term.
    params: Params,
}

/// Per-arity candidate tables, built once from the force field.
///
/// `bonds` / `angles` / `dihedrals` mirror molpy's `_bond_table` /
/// `_angle_table` / `_dihedral_table`. Built from the OPLS potential styles
/// (`("bond","harmonic")`, `("angle","harmonic")`, `("dihedral","opls")`).
pub struct CandidateTables {
    bonds: Vec<Candidate>,
    angles: Vec<Candidate>,
    dihedrals: Vec<Candidate>,
    /// `opls_NNN` → class (for resolving each atom's class at match time).
    type_to_class: HashMap<String, String>,
}

impl CandidateTables {
    /// Build the candidate tables and the type→class map from a force field +
    /// typing metadata.
    ///
    /// The class→layer map (used to compute each candidate's overlay layer) is
    /// derived from `meta`: `class → max(layer)` over every type carrying that
    /// class, replicating molpy's `_build_type_class_layer`.
    pub fn build(ff: &ForceField, meta: &OplsTypingMeta) -> Self {
        let (type_to_class, class_to_layer) = build_type_class_layer(meta);

        let layer_of = |classes: &[&str]| -> u32 {
            classes
                .iter()
                .map(|c| class_to_layer.get(*c).copied().unwrap_or(0))
                .max()
                .unwrap_or(0)
        };

        let bonds = match ff.get_style("bond", "harmonic").map(|s| &s.defs) {
            Some(StyleDefs::Bond(types)) => types
                .iter()
                .map(|t| Candidate {
                    name: t.name.clone(),
                    pattern: vec![t.itom.clone(), t.jtom.clone()],
                    layer: layer_of(&[&t.itom, &t.jtom]),
                    params: t.params.clone(),
                })
                .collect(),
            _ => Vec::new(),
        };

        let angles = match ff.get_style("angle", "harmonic").map(|s| &s.defs) {
            Some(StyleDefs::Angle(types)) => types
                .iter()
                .map(|t| Candidate {
                    name: t.name.clone(),
                    pattern: vec![t.itom.clone(), t.jtom.clone(), t.ktom.clone()],
                    layer: layer_of(&[&t.itom, &t.jtom, &t.ktom]),
                    params: t.params.clone(),
                })
                .collect(),
            _ => Vec::new(),
        };

        let dihedrals = match ff.get_style("dihedral", "opls").map(|s| &s.defs) {
            Some(StyleDefs::Dihedral(types)) => types
                .iter()
                .map(|t| Candidate {
                    name: t.name.clone(),
                    pattern: vec![
                        t.itom.clone(),
                        t.jtom.clone(),
                        t.ktom.clone(),
                        t.ltom.clone(),
                    ],
                    layer: layer_of(&[&t.itom, &t.jtom, &t.ktom, &t.ltom]),
                    params: t.params.clone(),
                })
                .collect(),
            _ => Vec::new(),
        };

        Self {
            bonds,
            angles,
            dihedrals,
            type_to_class,
        }
    }

    /// Resolve an atom's `(type, class)` pair from its `opls_NNN` type name.
    fn atom_of<'a>(&'a self, atom_type: &'a str) -> (&'a str, Option<&'a str>) {
        (
            atom_type,
            self.type_to_class.get(atom_type).map(String::as_str),
        )
    }

    /// Pick the best-ranked candidate for `atoms` from `table`. Highest
    /// `(score, layer)` wins; `None` if no candidate matches.
    fn best<'a>(table: &'a [Candidate], atoms: &[(&str, Option<&str>)]) -> Option<Match<'a>> {
        let mut best_key: Option<(i64, u32)> = None;
        let mut best: Option<Match<'a>> = None;
        for cand in table {
            let pat: Vec<&str> = cand.pattern.iter().map(String::as_str).collect();
            let Some(score) = sequence_score(&pat, atoms) else {
                continue;
            };
            let key = (score, cand.layer);
            if best_key.is_none_or(|cur| key > cur) {
                best_key = Some(key);
                best = Some(Match {
                    name: &cand.name,
                    params: &cand.params,
                });
            }
        }
        best
    }
}

/// Map each `opls_NNN` type to its class, and each class to its highest overlay
/// layer. Replicates molpy's `_build_type_class_layer`, sourced from chain-1's
/// [`OplsTypingMeta`] (each row carries `class` + `layer`).
fn build_type_class_layer(
    meta: &OplsTypingMeta,
) -> (HashMap<String, String>, HashMap<String, u32>) {
    let mut type_to_class = HashMap::new();
    let mut class_to_layer: HashMap<String, u32> = HashMap::new();
    for (name, row) in meta.iter() {
        type_to_class.insert(name.clone(), row.class.clone());
        if !row.class.is_empty() && row.class != "*" {
            let e = class_to_layer.entry(row.class.clone()).or_insert(0);
            *e = (*e).max(row.layer);
        }
    }
    (type_to_class, class_to_layer)
}

/// Strict-mode policy for a term that matches no candidate (and has no
/// estimator attached).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoMatch {
    /// Strict: a missing bonded parameter is a hard error.
    Error,
    /// Lenient: leave the term unparametrized and continue.
    Skip,
}

/// One bonded term awaiting parameters: its arity-tagged endpoint types.
///
/// Handed to an [`Estimator`] when no force-field candidate matches. Kept small
/// and owned so an estimator needs no access to `Atomistic` internals.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BondedTerm {
    /// A bond: the two endpoint `opls_NNN` types.
    Bond([String; 2]),
    /// An angle: the three endpoint `opls_NNN` types (centre in the middle).
    Angle([String; 3]),
    /// A dihedral: the four endpoint `opls_NNN` types.
    Dihedral([String; 4]),
}

/// No-match seam for filling uncovered bonded terms.
///
/// [`typify_bonded_with`] calls [`estimate`](Estimator::estimate) for any bonded
/// term the force-field tables do not cover. An implementation returns:
/// - `Ok(Some(params))` — estimated params to write onto the term;
/// - `Ok(None)` — declined; fall back to the strict policy;
/// - `Err(_)` — hard failure, propagated.
///
/// This is a force-field-agnostic extension point: molrs ships no estimator
/// (parameter estimation for uncovered terms is delegated to external tooling
/// such as AmberTools). A consumer that wants automatic fill-in implements this
/// trait and passes it to [`typify_bonded_with`]. The default [`typify_bonded`]
/// path attaches none and the [`NoMatch`] policy decides.
pub trait Estimator {
    /// Estimate parameters for an uncovered bonded `term`, or decline (`None`).
    fn estimate(&self, term: &BondedTerm) -> Result<Option<Params>, String>;
}

/// Typify bonded parameters onto a typed molecule, with the strict no-match
/// policy and no estimator (the default closed-loop path).
///
/// Convenience wrapper over [`typify_bonded_with`] with `estimator = None`. See
/// that function for the full contract.
///
/// # Errors
///
/// Returns `Err` if a term matches no candidate and `policy` is
/// [`NoMatch::Error`], or if writing a label onto the graph fails.
pub fn typify_bonded(
    mol_typed: &Atomistic,
    tables: &CandidateTables,
    policy: NoMatch,
) -> Result<Atomistic, String> {
    typify_bonded_with(mol_typed, tables, policy, None)
}

/// Typify bonded parameters onto a typed molecule, choosing each term's params
/// by the OPLS specificity + layer ranking, with an optional estimator seam.
///
/// For every enumerated bond / angle / dihedral:
/// 1. resolve each endpoint atom's `(type, class)`;
/// 2. scan the matching candidate table for the highest `(score, layer)`;
/// 3. on a match, copy the winning type's numeric params onto the term (e.g.
///    `k0`/`r0` for bonds, `k0`/`theta0` for angles, `f1..f4` for dihedrals),
///    matching molpy's `term.data.update(**type.params.kwargs)`;
/// 4. on no match, ask `estimator` (if any); if it declines or is absent, apply
///    `policy`.
///
/// Angles and dihedrals are enumerated from the bond graph via
/// the shared typifier topology helper (clearing any pre-existing generated
/// ones), mirroring the MMFF typifier. Only atoms that chain-1 actually typed participate:
/// a term with any untyped endpoint is skipped (its params are the consumer's
/// concern, not a hard error here) — full per-atom coverage is chain 3.
///
/// # Errors
///
/// Returns `Err` if a term matches no candidate, the estimator declines, and
/// `policy` is [`NoMatch::Error`]; if the estimator itself errors; or if a
/// graph write / topology enumeration fails.
pub fn typify_bonded_with(
    mol_typed: &Atomistic,
    tables: &CandidateTables,
    policy: NoMatch,
    estimator: Option<&dyn Estimator>,
) -> Result<Atomistic, String> {
    let mut out = mol_typed.clone();

    // Per-atom `opls_NNN` type, by id. Atoms with no `type` are untyped (chain-1
    // coverage gap) — terms touching them are skipped below.
    let type_of: HashMap<AtomId, String> = out
        .atoms()
        .filter_map(|(id, a)| a.get_str("type").map(|t| (id, t.to_string())))
        .collect();

    // --- bonds (already present from the input topology) ---
    let bond_rows: Vec<_> = out
        .bonds()
        .map(|(id, b)| (id, b.nodes[0], b.nodes[1]))
        .collect();
    for (id, i, j) in bond_rows {
        let (Some(ti), Some(tj)) = (type_of.get(&i), type_of.get(&j)) else {
            continue; // untyped endpoint — skip
        };
        let atoms = [tables.atom_of(ti), tables.atom_of(tj)];
        match CandidateTables::best(&tables.bonds, &atoms) {
            Some(m) => write_match(&mut out, BondedKind::Bond(id), &m)?,
            None => {
                let term = BondedTerm::Bond([ti.clone(), tj.clone()]);
                resolve_no_match(&mut out, BondedKind::Bond(id), &term, policy, estimator)?;
            }
        }
    }

    // --- enumerate angles + dihedrals from the bond graph (clear existing) ---
    crate::ff::typifier::topology::typify_bonded_topology(&mut out)?;

    // --- angles ---
    let angle_rows: Vec<_> = out
        .angles()
        .map(|(id, a)| (id, a.nodes[0], a.nodes[1], a.nodes[2]))
        .collect();
    for (id, i, j, k) in angle_rows {
        let (Some(ti), Some(tj), Some(tk)) = (type_of.get(&i), type_of.get(&j), type_of.get(&k))
        else {
            continue;
        };
        let atoms = [tables.atom_of(ti), tables.atom_of(tj), tables.atom_of(tk)];
        match CandidateTables::best(&tables.angles, &atoms) {
            Some(m) => write_match(&mut out, BondedKind::Angle(id), &m)?,
            None => {
                let term = BondedTerm::Angle([ti.clone(), tj.clone(), tk.clone()]);
                resolve_no_match(&mut out, BondedKind::Angle(id), &term, policy, estimator)?;
            }
        }
    }

    // --- dihedrals ---
    let dih_rows: Vec<_> = out
        .dihedrals()
        .map(|(id, d)| (id, d.nodes[0], d.nodes[1], d.nodes[2], d.nodes[3]))
        .collect();
    for (id, i, j, k, l) in dih_rows {
        let (Some(ti), Some(tj), Some(tk), Some(tl)) = (
            type_of.get(&i),
            type_of.get(&j),
            type_of.get(&k),
            type_of.get(&l),
        ) else {
            continue;
        };
        let atoms = [
            tables.atom_of(ti),
            tables.atom_of(tj),
            tables.atom_of(tk),
            tables.atom_of(tl),
        ];
        match CandidateTables::best(&tables.dihedrals, &atoms) {
            Some(m) => write_match(&mut out, BondedKind::Dihedral(id), &m)?,
            None => {
                let term = BondedTerm::Dihedral([ti.clone(), tj.clone(), tk.clone(), tl.clone()]);
                resolve_no_match(&mut out, BondedKind::Dihedral(id), &term, policy, estimator)?;
            }
        }
    }

    Ok(out)
}

/// A bonded relation id tagged by arity, so the param-writer dispatches to the
/// right `set_*_prop`.
enum BondedKind {
    Bond(molrs::BondId),
    Angle(molrs::AngleId),
    Dihedral(molrs::DihedralId),
}

/// Write a matched force-field type onto a bonded term: the `type` *name*
/// (so `ForceField::to_potentials` can re-resolve params from the bond/angle/
/// dihedral style) plus every numeric param (mirroring molpy's
/// `term.data["type"] = name; term.data.update(**type.params.kwargs)`).
fn write_match(out: &mut Atomistic, kind: BondedKind, m: &Match<'_>) -> Result<(), String> {
    set_term_prop(out, &kind, "type", m.name.to_string())?;
    write_params(out, kind, m.params)
}

/// Copy every param onto the bonded term (no `type` label; used by the
/// [`Estimator`] path, which synthesizes params with no force-field type name).
/// Both the numeric params (`k0`/`r0`/…) and any string params are written — the
/// latter carries an estimator's provenance convention (e.g. `estimate_method` /
/// `estimate_analog`) when one is supplied.
fn write_params(out: &mut Atomistic, kind: BondedKind, params: &Params) -> Result<(), String> {
    for (key, val) in params.iter() {
        set_term_prop(out, &kind, key, val)?;
    }
    for (key, val) in params.iter_strings() {
        set_term_prop(out, &kind, key, val)?;
    }
    Ok(())
}

/// Set one property on the bonded relation, dispatching by arity.
fn set_term_prop(
    out: &mut Atomistic,
    kind: &BondedKind,
    key: &str,
    val: impl Into<molrs::system::molgraph::PropValue>,
) -> Result<(), String> {
    match kind {
        BondedKind::Bond(id) => out.set_bond_prop(*id, key, val),
        BondedKind::Angle(id) => out.set_angle_prop(*id, key, val),
        BondedKind::Dihedral(id) => out.set_dihedral_prop(*id, key, val),
    }
    .map_err(|e| e.to_string())
}

/// Resolve a term that matched no candidate: try the estimator, else apply the
/// strict policy.
fn resolve_no_match(
    out: &mut Atomistic,
    kind: BondedKind,
    term: &BondedTerm,
    policy: NoMatch,
    estimator: Option<&dyn Estimator>,
) -> Result<(), String> {
    if let Some(est) = estimator
        && let Some(params) = est.estimate(term)?
    {
        return write_params(out, kind, &params);
    }
    match policy {
        NoMatch::Error => Err(format!("OPLS: no bonded type for {term:?}")),
        NoMatch::Skip => Ok(()), // leave the term unparametrized
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- end_score: 3 / 1 / 0 / None ---------------------------------------

    #[test]
    fn end_score_exact_type_is_3() {
        assert_eq!(end_score("opls_135", "opls_135", Some("CT")), Some(3));
    }

    #[test]
    fn end_score_class_is_1() {
        // Pattern is the class name (OPLS bonded forces key on class).
        assert_eq!(end_score("CT", "opls_135", Some("CT")), Some(1));
    }

    #[test]
    fn end_score_wildcard_is_0() {
        // Empty string (OPLS XML reader's wildcard), "*", and "X" all score 0.
        assert_eq!(end_score("", "opls_135", Some("CT")), Some(0));
        assert_eq!(end_score("*", "opls_135", Some("CT")), Some(0));
        assert_eq!(end_score("X", "opls_135", Some("CT")), Some(0));
    }

    #[test]
    fn end_score_no_match_is_none() {
        assert_eq!(end_score("OH", "opls_135", Some("CT")), None);
        // Class given but does not equal the pattern, and no type match either.
        assert_eq!(end_score("HC", "opls_140", Some("CT")), None);
    }

    #[test]
    fn end_score_no_class_falls_through_to_none() {
        // No class resolved, pattern is neither the type nor a wildcard => None.
        assert_eq!(end_score("CT", "opls_135", None), None);
    }

    // --- sequence_score: additive, symmetric, None-propagating -------------

    #[test]
    fn sequence_score_sums_per_end() {
        // CT-CT bond against two CT atoms: 1 + 1 = 2.
        let atoms = [("opls_135", Some("CT")), ("opls_136", Some("CT"))];
        assert_eq!(sequence_score(&["CT", "CT"], &atoms), Some(2));
    }

    #[test]
    fn sequence_score_is_reversal_symmetric() {
        // Pattern CT-OH matched against atoms (OH, CT) only works reversed.
        let atoms = [("opls_154", Some("OH")), ("opls_135", Some("CT"))];
        assert_eq!(sequence_score(&["CT", "OH"], &atoms), Some(2));
        // Forward-only would fail (CT vs OH-class, OH vs CT-class) -> reversed wins.
    }

    #[test]
    fn sequence_score_returns_best_orientation() {
        // Forward: exact(3) + wildcard(0) = 3; reversed: wildcard(0)+class(1)=1.
        // Best = 3.
        let atoms = [("opls_135", Some("CT")), ("opls_140", Some("HC"))];
        assert_eq!(sequence_score(&["opls_135", ""], &atoms), Some(3));
    }

    #[test]
    fn sequence_score_none_when_no_orientation_matches() {
        let atoms = [("opls_135", Some("CT")), ("opls_140", Some("HC"))];
        // Pattern OH-OH matches neither end in either orientation.
        assert_eq!(sequence_score(&["OH", "OH"], &atoms), None);
    }

    // --- ranking: specificity beats wildcard; layer breaks ties ------------

    fn cand(pattern: &[&str], layer: u32, k0: f64) -> Candidate {
        Candidate {
            name: pattern.join("-"),
            pattern: pattern.iter().map(|s| s.to_string()).collect(),
            layer,
            params: Params::from_pairs(&[("k0", k0)]),
        }
    }

    #[test]
    fn ranking_fully_resolved_beats_wildcard() {
        // Two bond candidates both match a CT-HC bond: the fully class-resolved
        // CT-HC (score 2) must beat the wildcard X-HC (score 0+1=1).
        let table = vec![cand(&["", "HC"], 0, 1.0), cand(&["CT", "HC"], 0, 2.0)];
        let atoms = [("opls_135", Some("CT")), ("opls_140", Some("HC"))];
        let best = CandidateTables::best(&table, &atoms).expect("a match");
        assert_eq!(
            best.params.get("k0"),
            Some(2.0),
            "fully-resolved candidate wins"
        );
    }

    #[test]
    fn ranking_equal_score_higher_layer_wins() {
        // Two CT-HC candidates with equal specificity; the higher overlay layer
        // (CL&P/CL&Pol) wins the tie.
        let table = vec![cand(&["CT", "HC"], 0, 1.0), cand(&["CT", "HC"], 2, 9.0)];
        let atoms = [("opls_135", Some("CT")), ("opls_140", Some("HC"))];
        let best = CandidateTables::best(&table, &atoms).expect("a match");
        assert_eq!(
            best.params.get("k0"),
            Some(9.0),
            "higher-layer candidate wins"
        );
    }

    #[test]
    fn ranking_no_candidate_matches_is_none() {
        let table = vec![cand(&["OH", "OH"], 0, 1.0)];
        let atoms = [("opls_135", Some("CT")), ("opls_140", Some("HC"))];
        assert!(CandidateTables::best(&table, &atoms).is_none());
    }

    // --- build_type_class_layer (class -> max layer) -----------------------

    #[test]
    fn class_to_layer_takes_the_max() {
        use crate::ff::typifier::opls::OplsTypeRow;
        let mut meta = OplsTypingMeta::new();
        let row = |class: &str, layer: u32| OplsTypeRow {
            class: class.to_string(),
            def: Some("[C]".into()),
            overrides: Vec::new(),
            priority: None,
            layer,
        };
        meta.insert("opls_a", row("CT", 0));
        meta.insert("opls_b", row("CT", 2)); // same class, higher layer
        let (t2c, c2l) = build_type_class_layer(&meta);
        assert_eq!(t2c.get("opls_a").map(String::as_str), Some("CT"));
        assert_eq!(c2l.get("CT"), Some(&2), "class layer is the max over types");
    }
}
