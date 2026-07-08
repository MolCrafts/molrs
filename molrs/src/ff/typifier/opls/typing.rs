//! OPLS-AA SMARTS atom typing (dependency-aware, layered).
//!
//! [`typify_atoms`] drives the [`LayeredTypingEngine`]: it processes the type
//! defs level by level so that a def referencing a
//! previously-assigned type via `%opls_NNN` (e.g. benzene's aromatic-H type
//! `opls_146` = `[H][C;%opls_145]`) is matched only after its dependency is
//! resolved. The engine returns the per-atom `opls_NNN` assignment, which this
//! function writes — together with each type's `class` and `charge` — onto a
//! labeled copy of the input [`Atomistic`].
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

use molrs::Atomistic;

use crate::ff::forcefield::ForceField;

use super::layered::LayeredTypingEngine;
use super::meta::OplsTypingMeta;

/// Typify atoms with OPLS-AA atom types, returning a labeled copy.
///
/// Drives the [`LayeredTypingEngine`] over `meta`: every atom assigned a type
/// gets that type's `type` (`opls_NNN`), `class`, and `charge` (`e`, from the
/// potential `ForceField`'s `("atom","full")` style) written onto the returned
/// [`Atomistic`]. Atoms typed by no def are left untyped (no `type` prop) —
/// strict-mode failure is the consumer's policy.
///
/// `ff` supplies per-type charges; a type absent from the atom style simply
/// yields no `charge` prop. Bonded-term labeling is out of scope (chain 2).
///
/// # Errors
///
/// Returns `Err` if any SMARTS `def` is malformed (fail-fast), or if writing a
/// label onto the graph fails.
pub fn typify_atoms(
    mol: &Atomistic,
    meta: &OplsTypingMeta,
    ff: &ForceField,
) -> Result<Atomistic, String> {
    let engine = LayeredTypingEngine::build(meta)?;
    let assignments = engine.typify(mol);

    // Per-type charge from the potential ForceField's atom style.
    let charge_of = |type_name: &str| -> Option<f64> {
        ff.get_style("atom", "full")?
            .get_atomtype(type_name)?
            .params
            .get("charge")
    };

    let mut out = mol.clone();
    for (atom_id, type_name) in &assignments {
        out.set_atom(*atom_id, "type", type_name.clone())
            .map_err(|e| e.to_string())?;
        if let Some(row) = meta.get(type_name) {
            out.set_atom(*atom_id, "class", row.class.clone())
                .map_err(|e| e.to_string())?;
        }
        if let Some(q) = charge_of(type_name) {
            out.set_atom(*atom_id, "charge", q)
                .map_err(|e| e.to_string())?;
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::typifier::opls::meta::OplsTypeRow;
    use molrs::Atom;

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
        let typed = typify_atoms(&ethane(), &m, &ff).unwrap();

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
        let typed = typify_atoms(&ethane(), &m, &ff).unwrap();
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
        ff.def_atomstyle("full")
            .def_atomtype("opls_140", &[("mass", 1.008), ("charge", 0.06)]);
        let typed = typify_atoms(&ethane(), &m, &ff).unwrap();
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
        let err = typify_atoms(&ethane(), &m, &ff).unwrap_err();
        assert!(err.contains("opls_bad"), "err names the type: {err}");
    }

    #[test]
    fn recursive_dollar_def_matches() {
        // Recursive $() SMARTS: an sp3 carbon bonded to another sp3 carbon.
        // Exercises the engine's recursive `$(...)` support (rooted at the
        // candidate atom). Both ethane carbons match.
        let m = meta_with(&[("opls_rec", row("CT", Some("[$([CX4][CX4])]"), &[]))]);
        let ff = ForceField::new("OPLS-AA");
        let typed = typify_atoms(&ethane(), &m, &ff).unwrap();
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
        let typed = typify_atoms(&ethane(), &m, &ff).unwrap();
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
        let typed = typify_atoms(&g, &m, &ff).unwrap();

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
