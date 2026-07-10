//! Layered, dependency-aware OPLS-AA atom typing.
//!
//! [`LayeredTypingEngine`] drives atom typing level by level, using the
//! [`OplsDependencyAnalyzer`] to order defs so that a def referencing
//! `%opls_NNN` is only matched *after* the `opls_NNN` type has been assigned in
//! an earlier level. It accumulates a `HashMap<AtomId, String>` of assigned
//! types and feeds it back as the SMARTS **context-label map** (so `%opls_NNN`
//! predicates can read the current assignments).
//!
//! This replicates molpy's `LayeredTypingEngine`
//! (`molpy/typifier/layered_engine.py`):
//! - levels processed in ascending order ([`LayeredTypingEngine::typify`]);
//! - a normal level is resolved in a single pass — matching under the current
//!   assignment context, resolving per-atom conflicts by priority, then merging
//!   (new assignments override) — mirroring molpy's `_resolve_level`;
//! - a level containing a circular-dependency group is resolved by fixed-point
//!   iteration (`max_iterations = 10`, converging on assignment-map equality),
//!   mirroring `_resolve_circular`.
//!
//! Per-atom conflict resolution reuses chain-1's ranking
//! (`priority ≫ specificity ≫ earlier definition order`), so a molecule with no
//! `%opls_NNN` defs collapses to the chain-1 single-pass behaviour: everything
//! lands at level 0 and is resolved in one pass.

use std::collections::HashMap;

use molrs::{AtomId, Atomistic, MatchOptions, SmartsPattern};

use super::deps::OplsDependencyAnalyzer;
use super::meta::OplsTypingMeta;

/// Maximum fixed-point iterations for a circular-dependency level (mirrors
/// molpy's `max_iterations` default).
pub const MAX_CIRCULAR_ITERATIONS: usize = 10;

/// A compiled, ranked def: its SMARTS pattern plus conflict-resolution inputs.
struct RankedDef {
    name: String,
    pattern: SmartsPattern,
    priority: i64,
    /// Specificity proxy: number of query atoms.
    specificity: usize,
    /// Stable definition order (index after sorting type names).
    order: usize,
}

/// A conflict-resolution rank key (all "higher wins").
#[derive(Clone, Copy, PartialEq, Eq)]
struct Rank {
    priority: i64,
    specificity: usize,
    order: usize,
}

impl Rank {
    /// Whether `self` beats `other` — higher priority, then higher specificity,
    /// then EARLIER definition order (reused from chain-1's `typify_atoms`).
    fn beats(&self, other: &Rank) -> bool {
        (
            self.priority,
            self.specificity,
            std::cmp::Reverse(self.order),
        ) > (
            other.priority,
            other.specificity,
            std::cmp::Reverse(other.order),
        )
    }
}

/// Dependency-aware, level-by-level OPLS atom typing engine.
pub struct LayeredTypingEngine {
    analyzer: OplsDependencyAnalyzer,
    /// Compiled defs grouped by topological level (index = level).
    by_level: Vec<Vec<RankedDef>>,
}

impl LayeredTypingEngine {
    /// Build the engine from typing metadata.
    ///
    /// Compiles every def-carrying type's SMARTS, computes dependency levels,
    /// and buckets the compiled defs by level. A malformed SMARTS `def` is a
    /// broken force-field definition and returns `Err` (fail-fast) — matching
    /// chain-1's behaviour. The monatomic-ion bare-element fix-up (`Li` →
    /// `[Li]`) is preserved.
    pub fn build(meta: &OplsTypingMeta) -> Result<Self, String> {
        let analyzer = OplsDependencyAnalyzer::new(meta);
        let priorities = meta.priorities();

        // Deterministic definition order: sort by type name.
        let mut named: Vec<(&String, &super::meta::OplsTypeRow)> = meta.iter().collect();
        named.sort_by(|a, b| a.0.cmp(b.0));

        let max_level = analyzer.max_level().unwrap_or(0);
        let mut by_level: Vec<Vec<RankedDef>> = (0..=max_level).map(|_| Vec::new()).collect();

        for (order, (name, row)) in named.into_iter().enumerate() {
            let Some(def) = row.def.as_deref() else {
                continue; // legacy / no-def row
            };
            let pattern = compile_def(def).map_err(|e| {
                format!("OPLS type {name:?}: failed to parse SMARTS def {def:?}: {e}")
            })?;
            let specificity = pattern.num_query_atoms();
            let level = analyzer.level(name).unwrap_or(0);
            if level >= by_level.len() {
                by_level.resize_with(level + 1, Vec::new);
            }
            by_level[level].push(RankedDef {
                name: name.clone(),
                pattern,
                priority: *priorities.get(name).unwrap_or(&0),
                specificity,
                order,
            });
        }

        Ok(Self { analyzer, by_level })
    }

    /// Resolve every atom's OPLS type, returning `atom → opls_NNN`.
    ///
    /// Levels are processed in ascending order; the accumulated assignment map
    /// is threaded into each level's SMARTS matching as the context-label map,
    /// so `%opls_NNN` defs see the prior levels' results. A level whose defs lie
    /// in a circular-dependency group is resolved by fixed-point iteration.
    pub fn typify(&self, mol: &Atomistic) -> HashMap<AtomId, String> {
        let mut assignments: HashMap<AtomId, String> = HashMap::new();
        for level in 0..self.by_level.len() {
            let defs = &self.by_level[level];
            if defs.is_empty() {
                continue;
            }
            let is_circular = defs.iter().any(|d| self.analyzer.is_circular(&d.name));
            assignments = if is_circular {
                self.resolve_circular(defs, mol, assignments)
            } else {
                self.resolve_level(defs, mol, assignments)
            };
        }
        assignments
    }

    /// Resolve a single (non-circular) level: match every def at this level
    /// against `mol` under the `current` label context, pick the best-ranked
    /// type per atom among this level's matches, then merge (level winners
    /// override prior assignments). Mirrors molpy's `_resolve_level`.
    fn resolve_level(
        &self,
        defs: &[RankedDef],
        mol: &Atomistic,
        current: HashMap<AtomId, String>,
    ) -> HashMap<AtomId, String> {
        let mut best: HashMap<AtomId, (String, Rank)> = HashMap::new();
        for d in defs {
            let rank = Rank {
                priority: d.priority,
                specificity: d.specificity,
                order: d.order,
            };
            for m in d.pattern.find(
                mol,
                MatchOptions {
                    labels: Some(&current),
                    root: None,
                    limit: None,
                },
            ) {
                // Target atom is the root (query atom 0), per RDKit convention.
                let Some(&target) = m.atoms.first() else {
                    continue;
                };
                match best.get(&target) {
                    Some((_, cur)) if !rank.beats(cur) => {}
                    _ => {
                        best.insert(target, (d.name.clone(), rank));
                    }
                }
            }
        }

        // Merge: this level's winners override prior assignments.
        let mut result = current;
        for (atom, (type_name, _)) in best {
            result.insert(atom, type_name);
        }
        result
    }

    /// Resolve a circular-dependency level by fixed-point iteration: repeatedly
    /// run [`resolve_level`](Self::resolve_level) (each pass feeding the prior
    /// pass's assignments back as the label context) until the assignment map
    /// stops changing or `MAX_CIRCULAR_ITERATIONS` is reached. Mirrors molpy's
    /// `_resolve_circular`.
    fn resolve_circular(
        &self,
        defs: &[RankedDef],
        mol: &Atomistic,
        current: HashMap<AtomId, String>,
    ) -> HashMap<AtomId, String> {
        let mut assignments = current;
        for _ in 0..MAX_CIRCULAR_ITERATIONS {
            let prev = assignments.clone();
            assignments = self.resolve_level(defs, mol, assignments);
            if assignments == prev {
                break;
            }
        }
        assignments
    }

    /// Access the underlying dependency analyzer (levels / circular groups).
    pub fn analyzer(&self) -> &OplsDependencyAnalyzer {
        &self.analyzer
    }
}

/// Compile a single SMARTS `def`, with the OPLS monatomic-ion bare-element
/// fix-up (`Li` → `[Li]`) preserved from chain-1's `typing::compile_def`.
fn compile_def(def: &str) -> Result<SmartsPattern, molrs::MolRsError> {
    match SmartsPattern::parse(def) {
        Ok(p) => Ok(p),
        Err(e) => {
            if is_bare_element_symbol(def) {
                SmartsPattern::parse(&format!("[{def}]"))
            } else {
                Err(e)
            }
        }
    }
}

/// Whether `def` is exactly one element symbol (`Li`, `Na`, `Br`, `C`).
fn is_bare_element_symbol(def: &str) -> bool {
    let b = def.as_bytes();
    match b.len() {
        1 => b[0].is_ascii_uppercase(),
        2 => b[0].is_ascii_uppercase() && b[1].is_ascii_lowercase(),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::typifier::opls::meta::OplsTypeRow;
    use molrs::Atom;

    fn row(class: &str, def: Option<&str>, overrides: &[&str]) -> OplsTypeRow {
        OplsTypeRow {
            class: class.to_string(),
            def: def.map(str::to_string),
            overrides: overrides.iter().map(|s| s.to_string()).collect(),
            priority: None,
            layer: 0,
        }
    }

    fn meta_with(rows: &[(&str, OplsTypeRow)]) -> OplsTypingMeta {
        let mut m = OplsTypingMeta::new();
        for (name, r) in rows {
            m.insert(*name, r.clone());
        }
        m
    }

    /// Ethanol skeleton C-C-O with explicit Hs; returns (graph, O id, H-on-O id).
    fn ethanol() -> (Atomistic, AtomId, AtomId) {
        let mut g = Atomistic::new();
        let cm = g.add_atom(Atom::xyz("C", 0.0, 0.0, 0.0));
        let ch = g.add_atom(Atom::xyz("C", 1.5, 0.0, 0.0));
        let o = g.add_atom(Atom::xyz("O", 2.5, 0.0, 0.0));
        let ho = g.add_atom(Atom::xyz("H", 3.3, 0.0, 0.0));
        g.add_bond(cm, ch).unwrap();
        g.add_bond(ch, o).unwrap();
        g.add_bond(o, ho).unwrap();
        for c in [cm, ch] {
            let n = if c == cm { 3 } else { 2 };
            for k in 0..n {
                let h = g.add_atom(Atom::xyz("H", 0.3 * (k as f64 + 1.0), 0.9, 0.0));
                g.add_bond(c, h).unwrap();
            }
        }
        (g, o, ho)
    }

    #[test]
    fn level_zero_only_matches_chain1_behaviour() {
        // No %-defs: everything is level 0, resolved in one pass.
        let meta = meta_with(&[
            ("opls_135", row("CT", Some("[C;X4](C)(H)(H)H"), &[])),
            ("opls_140", row("HC", Some("H[C;X4]"), &[])),
        ]);
        let engine = LayeredTypingEngine::build(&meta).unwrap();
        assert_eq!(engine.analyzer().max_level(), Some(0));

        let (g, _o, _ho) = ethanol();
        let assigned = engine.typify(&g);
        // The methyl carbon (CH3 on a C) types opls_135; its 3 H type opls_140.
        let n135 = assigned.values().filter(|t| *t == "opls_135").count();
        let n140 = assigned.values().filter(|t| *t == "opls_140").count();
        assert_eq!(n135, 1, "one CH3 carbon");
        assert!(n140 >= 1, "methyl Hs typed");
    }

    #[test]
    fn dependent_def_resolves_after_its_dependency() {
        // opls_154 (alcohol O, level 0) then opls_155 (H[O;%opls_154], level 1):
        // the hydroxyl H must type opls_155 only after the O is opls_154.
        let meta = meta_with(&[
            ("opls_154", row("OH", Some("[O;X2](H)([!H])"), &[])),
            ("opls_155", row("HO", Some("H[O;%opls_154]"), &[])),
        ]);
        let engine = LayeredTypingEngine::build(&meta).unwrap();
        assert_eq!(engine.analyzer().level("opls_154"), Some(0));
        assert_eq!(engine.analyzer().level("opls_155"), Some(1));

        let (g, o, ho) = ethanol();
        let assigned = engine.typify(&g);
        assert_eq!(
            assigned.get(&o).map(String::as_str),
            Some("opls_154"),
            "alcohol O typed at level 0"
        );
        assert_eq!(
            assigned.get(&ho).map(String::as_str),
            Some("opls_155"),
            "hydroxyl H typed at level 1 via %opls_154"
        );
    }

    #[test]
    fn missing_dependency_leaves_dependent_untyped() {
        // Without the level-0 O def, %opls_154 is never satisfied, so opls_155
        // never matches (the H stays untyped) — and the engine still terminates.
        let meta = meta_with(&[("opls_155", row("HO", Some("H[O;%opls_154]"), &[]))]);
        let engine = LayeredTypingEngine::build(&meta).unwrap();
        let (g, _o, ho) = ethanol();
        let assigned = engine.typify(&g);
        assert!(
            !assigned.contains_key(&ho),
            "no dependency assigned -> dependent stays untyped"
        );
    }

    #[test]
    fn circular_level_iterates_to_fixed_point() {
        // A constructed two-cycle that still terminates: opls_x and opls_y are a
        // circular group at max_level+1. With a level-0 seed (opls_base on the
        // methyl C), the fixed-point loop converges (here neither cyclic def can
        // actually fire on ethanol, so it converges immediately to the seed) and
        // the engine returns without exhausting the iteration cap.
        let meta = meta_with(&[
            ("opls_base", row("CT", Some("[C;X4](C)(H)(H)H"), &[])),
            ("opls_x", row("X", Some("[C;%opls_base][O;%opls_y]"), &[])),
            ("opls_y", row("Y", Some("[N;%opls_x]"), &[])),
        ]);
        let engine = LayeredTypingEngine::build(&meta).unwrap();
        assert_eq!(engine.analyzer().circular_groups().len(), 1);
        let (g, _o, _ho) = ethanol();
        let assigned = engine.typify(&g);
        // The base def still types the methyl carbon; the cyclic defs (needing
        // an O/N neighbour with cyclic types) never fire on ethanol.
        assert_eq!(
            assigned.values().filter(|t| *t == "opls_base").count(),
            1,
            "level-0 seed survives the circular level"
        );
        assert!(
            !assigned.values().any(|t| t == "opls_x" || t == "opls_y"),
            "cyclic defs cannot fire on ethanol"
        );
    }

    #[test]
    fn malformed_def_fails_fast() {
        let meta = meta_with(&[("opls_bad", row("X", Some("[C"), &[]))]);
        match LayeredTypingEngine::build(&meta) {
            Ok(_) => panic!("malformed def should fail fast"),
            Err(e) => assert!(e.contains("opls_bad"), "err names the type: {e}"),
        }
    }
}
