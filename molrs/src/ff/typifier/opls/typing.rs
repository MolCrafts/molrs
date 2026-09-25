//! OPLS-AA SMARTS atom typing (dependency-aware, layered).
//!
//! [`typify_atoms`] drives the [`LayeredTypingEngine`]: it processes the type
//! defs level by level so that a def referencing a
//! previously-assigned type via `%opls_NNN` (e.g. benzene's aromatic-H type
//! `opls_146` = `[H][C;%opls_145]`) is matched only after its dependency is
//! resolved. The engine returns the per-atom `opls_NNN` assignment, which this
//! function turns into the node [`Annotation`]s of a [`Match`](crate::ff::typifier::Match):
//! each type's `type` (with its `atom/full` row's `mass` and `charge`) and
//! `class`. It writes nothing; the typing base stamps the match.
//!
//! # SMARTS reuse
//!
//! Matching uses the always-compiled molrs SMARTS engine
//! ([`SmartsPattern`](molrs::SmartsPattern)) with the context-label extension
//! ([`MatchOptions::labels`](molrs::MatchOptions::labels)):
//! the engine feeds back the current assignment map as the label context so
//! `%opls_NNN` predicates can read it. Each `def` is the SMARTS for the type's
//! *target* atom: by RDKit convention the engine roots a match at query atom 0,
//! so the typed atom of every match is `match[0]`.
//!
//! # Conflict resolution
//!
//! When several defs match the same atom *within a level*, the winner is chosen
//! by, in order:
//! 1. higher priority (overrides / explicit / layer — see [`OplsTypingMeta`]);
//! 2. more specific pattern (more query atoms);
//! 3. earlier definition order (stable tie-break, by sorted `opls_NNN` name).
//!
//! Step 2 is a specificity proxy; molpy's exact `(score, pattern_size,
//! definition_order)` tie-break (which also counts query *edges*) is a chain-3
//! per-atom-parity refinement, not required here.
//!
//! # Levels
//!
//! Standalone (no-`%opls_NNN`) defs are level 0 — the chain-1 single-pass case.
//! `%opls_NNN`-referencing defs resolve in dependency order; mutually-dependent
//! defs form a circular group resolved by fixed-point iteration (see
//! [`layered`](super::layered)).
//!
//! # Out of scope
//!
//! - Legacy rows with no `def` are skipped (cannot be SMARTS-matched).
//! - Bonded-term (bond / angle / dihedral) labeling is chain 2.

use std::collections::HashMap;

use molrs::system::molgraph::PropValue;
use molrs::{AtomId, Atomistic};

use crate::ff::forcefield::{ForceField, Params};
use crate::ff::typifier::Annotation;

use super::layered::LayeredTypingEngine;
use super::meta::OplsTypingMeta;

/// The OPLS-AA atom typing of one graph.
pub(crate) struct AtomTyping {
    /// The `opls_NNN` type of every atom a def typed; an atom no def typed is
    /// absent.
    pub(crate) types: HashMap<AtomId, String>,
    /// The node annotations, positional against `graph.atoms()`.
    pub(crate) nodes: Vec<Vec<(String, Annotation)>>,
}

/// Type the atoms of `mol` with OPLS-AA atom types.
///
/// Drives the [`LayeredTypingEngine`] over `meta`. Every atom assigned a type
/// gets two annotations:
/// - `type` → [`Annotation::Type`] under the `atom/full` style of `ff`, named
///   by the `opls_NNN` type, with that row's numeric params (`mass`, `charge`)
///   — or a plain [`Annotation::Value`] when `ff` has no such row. The row's
///   string metadata (`type_`, `def_`, …, from the XML reader) is typing input
///   and is neither stamped nor defined;
/// - `class` → [`Annotation::Value`] of the type's class, when `meta` has one.
///
/// Atoms typed by no def get no annotation (and stay untyped); the strict
/// [`OPLSAATypifier`](super::OPLSAATypifier) refuses such a molecule.
///
/// # Errors
///
/// Returns `Err` if any SMARTS `def` is malformed (fail-fast).
pub(crate) fn typify_atoms(
    mol: &Atomistic,
    meta: &OplsTypingMeta,
    ff: &ForceField,
) -> Result<AtomTyping, String> {
    let engine = LayeredTypingEngine::build(meta)?;
    let types = engine.assign(mol);
    let atom_full = ff.get_style("atom", "full");

    let nodes = mol
        .atoms()
        .map(|(id, _)| {
            let Some(type_name) = types.get(&id) else {
                return Vec::new();
            };
            let typed = match atom_full.and_then(|s| s.get_atomtype(type_name)) {
                Some(row) => Annotation::Type {
                    style: "full".to_owned(),
                    name: type_name.clone(),
                    endpoints: Some(Vec::new()),
                    params: Params::from_pairs(&row.params.iter().collect::<Vec<_>>()),
                },
                None => Annotation::Value(PropValue::Str(type_name.clone())),
            };
            let mut annotations = vec![("type".to_owned(), typed)];
            if let Some(row) = meta.get(type_name) {
                annotations.push((
                    "class".to_owned(),
                    Annotation::Value(PropValue::Str(row.class.clone())),
                ));
            }
            annotations
        })
        .collect();

    Ok(AtomTyping { types, nodes })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::typifier::Match;
    use crate::ff::typifier::opls::meta::OplsTypeRow;
    use molrs::Atom;

    /// `mol` with the node annotations of [`typify_atoms`] written onto it
    /// through the typing base's one execution path.
    fn typed_graph(
        mol: &Atomistic,
        meta: &OplsTypingMeta,
        ff: &ForceField,
    ) -> Result<Atomistic, String> {
        let typing = typify_atoms(mol, meta, ff)?;
        let mut graph = mol.clone();
        let mut m = Match {
            nodes: typing.nodes,
            ..Match::default()
        };
        m.declare_styles_of(ff);
        m.write_onto(&mut graph, &mut ff.empty_like())?;
        Ok(graph)
    }

    /// Build a tiny ethane-like skeleton C-C with explicit H neighbours so the
    /// `[C;X4](C)(H)(H)H` style defs have something to match. (Pure-function
    /// unit fixture — real-molecule typing lives in tests/ff/typifier/opls.rs.)
    fn ethane() -> Atomistic {
        let mut g = Atomistic::new();
        let c0 = g.add_atom(Atom::xyz("C", 0.0, 0.0, 0.0));
        let c1 = g.add_atom(Atom::xyz("C", 1.5, 0.0, 0.0));
        g.add_bond(c0, c1).unwrap();
        for c in [c0, c1] {
            for k in 0..3 {
                let h = g.add_atom(Atom::xyz("H", 0.3 * (k as f64 + 1.0), 0.8, 0.0));
                g.add_bond(c, h).unwrap();
            }
        }
        g
    }

    fn meta_with(rows: &[(&str, OplsTypeRow)]) -> OplsTypingMeta {
        let mut m = OplsTypingMeta::new();
        for (name, row) in rows {
            m.insert(*name, row.clone());
        }
        m
    }

    fn row(class: &str, def: Option<&str>, overrides: &[&str]) -> OplsTypeRow {
        OplsTypeRow {
            class: class.to_string(),
            def: def.map(str::to_string),
            overrides: overrides.iter().map(|s| s.to_string()).collect(),
            priority: None,
            layer: 0,
        }
    }

    #[test]
    fn typing_assigns_carbon_and_hydrogen() {
        let m = meta_with(&[
            ("opls_135", row("CT", Some("[C;X4](C)(H)(H)H"), &[])),
            ("opls_140", row("HC", Some("H[C;X4]"), &[])),
        ]);
        let ff = ForceField::new("OPLS-AA");
        let typed = typed_graph(&ethane(), &m, &ff).unwrap();

        let mut n_ct = 0;
        let mut n_hc = 0;
        for (_, a) in typed.atoms() {
            match a.get_str("type") {
                Some("opls_135") => {
                    n_ct += 1;
                    assert_eq!(a.get_str("class"), Some("CT"));
                }
                Some("opls_140") => n_hc += 1,
                _ => {}
            }
        }
        assert_eq!(n_ct, 2, "both methyl carbons typed CT");
        assert_eq!(n_hc, 6, "all six H typed HC");
    }

    #[test]
    fn higher_priority_overrides_wins() {
        // Two defs both match every C; opls_special overrides opls_generic, so it
        // gains priority and must win on the carbons.
        let m = meta_with(&[
            ("opls_generic", row("CG", Some("[C]"), &[])),
            ("opls_special", row("CS", Some("[C;X4]"), &["opls_generic"])),
        ]);
        let ff = ForceField::new("OPLS-AA");
        let typed = typed_graph(&ethane(), &m, &ff).unwrap();
        for (id, a) in typed.atoms() {
            if matches!(a.get_str("element"), Some("C")) {
                assert_eq!(
                    a.get_str("type"),
                    Some("opls_special"),
                    "carbon {id:?} should take the higher-priority override"
                );
            }
        }
    }

    #[test]
    fn charge_written_from_forcefield() {
        let m = meta_with(&[("opls_140", row("HC", Some("H[C;X4]"), &[]))]);
        let mut ff = ForceField::new("OPLS-AA");
        ff.def_style("atom", "full", Params::new())
            .unwrap()
            .def_type(
                "opls_140",
                Params::from_pairs(&[("mass", 1.008), ("charge", 0.06)]),
            )
            .unwrap();
        let typed = typed_graph(&ethane(), &m, &ff).unwrap();
        let h = typed
            .atoms()
            .find(|(_, a)| a.get_str("type") == Some("opls_140"))
            .expect("a hydrogen was typed");
        assert_eq!(h.1.get_f64("charge"), Some(0.06));
    }

    #[test]
    fn malformed_def_fails_fast() {
        // Unbalanced bracket — a broken force-field def, must Err (never drop).
        let m = meta_with(&[("opls_bad", row("X", Some("[C"), &[]))]);
        let ff = ForceField::new("OPLS-AA");
        let err = typed_graph(&ethane(), &m, &ff).unwrap_err();
        assert!(err.contains("opls_bad"), "err names the type: {err}");
    }

    #[test]
    fn recursive_dollar_def_matches() {
        // Recursive $() SMARTS: an sp3 carbon bonded to another sp3 carbon.
        // Exercises the engine's recursive `$(...)` support (rooted at the
        // candidate atom). Both ethane carbons match.
        let m = meta_with(&[("opls_rec", row("CT", Some("[$([CX4][CX4])]"), &[]))]);
        let ff = ForceField::new("OPLS-AA");
        let typed = typed_graph(&ethane(), &m, &ff).unwrap();
        let n = typed
            .atoms()
            .filter(|(_, a)| a.get_str("type") == Some("opls_rec"))
            .count();
        assert_eq!(n, 2, "recursive def should match both sp3 carbons");
    }

    #[test]
    fn typed_atom_ref_def_without_dependency_types_nothing() {
        // A %opls_NNN def is now supported (layered), not skipped. With no
        // matching dependency present (nothing is ever typed opls_145), the
        // `%opls_145` predicate never holds, so the def matches nothing — and
        // it is NOT an error. (Real layered typing is covered in
        // src/ff/typifier/opls/layered.rs and tests/ff/typifier/opls.rs.)
        let m = meta_with(&[("opls_ref", row("HA", Some("[H][C;%opls_145]"), &[]))]);
        let ff = ForceField::new("OPLS-AA");
        let typed = typed_graph(&ethane(), &m, &ff).unwrap();
        assert!(typed.atoms().all(|(_, a)| a.get_str("type").is_none()));
    }

    #[test]
    fn layered_dependency_def_types_after_its_dependency() {
        // opls_154 (alcohol O) then opls_155 (H[O;%opls_154]): the hydroxyl H
        // is typed only after the O is typed opls_154 — exercising the full
        // typify_atoms layered path end to end on a constructed ethanol.
        let mut g = Atomistic::new();
        let cm = g.add_atom(Atom::xyz("C", 0.0, 0.0, 0.0));
        let ch = g.add_atom(Atom::xyz("C", 1.5, 0.0, 0.0));
        let o = g.add_atom(Atom::xyz("O", 2.5, 0.0, 0.0));
        let ho = g.add_atom(Atom::xyz("H", 3.3, 0.0, 0.0));
        g.add_bond(cm, ch).unwrap();
        g.add_bond(ch, o).unwrap();
        g.add_bond(o, ho).unwrap();
        for k in 0..3 {
            let h = g.add_atom(Atom::xyz("H", 0.3 * (k as f64 + 1.0), 0.9, 0.0));
            g.add_bond(cm, h).unwrap();
        }
        for k in 0..2 {
            let h = g.add_atom(Atom::xyz("H", 1.5 + 0.3 * (k as f64), 0.9, 0.0));
            g.add_bond(ch, h).unwrap();
        }

        let m = meta_with(&[
            ("opls_154", row("OH", Some("[O;X2](H)([!H])"), &[])),
            ("opls_155", row("HO", Some("H[O;%opls_154]"), &[])),
        ]);
        let ff = ForceField::new("OPLS-AA");
        let typed = typed_graph(&g, &m, &ff).unwrap();

        assert_eq!(
            typed.get_atom(o).unwrap().get_str("type"),
            Some("opls_154"),
            "alcohol O typed opls_154"
        );
        assert_eq!(
            typed.get_atom(ho).unwrap().get_str("type"),
            Some("opls_155"),
            "hydroxyl H typed opls_155 via the %opls_154 dependency"
        );
    }
}
