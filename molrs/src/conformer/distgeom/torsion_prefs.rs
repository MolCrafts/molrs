//! ETKDGv3 experimental torsion-angle preferences (CrystalFF) + SMARTS matcher.
//!
//! Data and selection semantics ported from RDKit (BSD-3, Copyright (C)
//! 2017-2023 Sereina Riniker and other RDKit contributors):
//!   * `$RDBASE/Code/GraphMol/ForceFieldHelpers/CrystalFF/`
//!     `torsionPreferences_{v2,smallrings,macrocycles}.in` — the full
//!     ETKDGv2/v3 SMARTS → `(signs, V)` parameter tables (embedded verbatim in
//!     [`super::torsion_tables`], extracted by `gen_torsion_tables.py`),
//!   * `.../TorsionPreferences.cpp` (`getExperimentalTorsions`) — pattern
//!     matching + first-match-wins dedup per rotatable bond,
//!   * `$RDBASE/Code/ForceField/CrystalFF/TorsionAngleM6.h` — the M6 potential
//!     `V = Σ_m V_m·(1 + s_m·cos(m·x))`.
//!
//! ## Faithful port (this replaces the former representative subset)
//!
//! Every torsion assignment now flows through the **full** three-table data
//! set and the project SMARTS engine ([`molrs::chem::smarts::SmartsPattern`]). For
//! each rotatable bond we reproduce RDKit's exact selection:
//!
//! 1. The three tables are matched in RDKit concatenation order — v2, then
//!    small-rings, then macrocycles ([`super::torsion_tables`]). The first
//!    pattern (in that global order) that matches a bond wins; a bond receives
//!    at most one experimental torsion.
//! 2. Ring-size gating is *intrinsic to the SMARTS* (macrocycle patterns carry
//!    `r{9-}`, small-ring patterns `r{-8}` / `r{5-6}` / …), so there is no
//!    separate ring-membership dispatch: the right table simply fails to match
//!    bonds of the wrong ring class. This is exactly how RDKit layers them.
//!
//! The patterns are passed verbatim to the core SMARTS engine: as of
//! `core-perception-02-smarts-rings` the engine parses and evaluates RDKit's
//! **ring-size range** `r{lo-hi}` / `r{-hi}` / `r{lo-}` and **ring
//! connectivity** `x<n>` natively, so the former strip-token + post-check shim
//! is gone (validated against RDKit `GetExperimentalTorsions` in
//! `tests/embed/torsions.rs`).

use std::collections::HashMap;

use molrs::chem::smarts::{MatchOptions, SmartsPattern};
use molrs::system::atomistic::{AtomId, Atomistic};
use molrs::system::molgraph::PropValue;

use super::mol_features::Perceived;
use super::torsion_tables::{self, TorsionRow};

/// One assigned experimental torsion: four atoms + the M6 `(signs, V)` set.
#[derive(Clone, Debug)]
pub struct TorsionConstraint {
    /// Ordered atom indices `i-j-k-l` (the rotatable bond is `j-k`).
    pub atoms: [usize; 4],
    /// Per-order signs `s1..s6`.
    pub signs: [i8; 6],
    /// Per-order force constants `V1..V6`.
    pub force_constants: [f64; 6],
    /// The originating pattern SMARTS (for diagnostics / spec traceability).
    pub pattern: &'static str,
}

/// Which source table a pattern came from (for diagnostics + `tests`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TorsionTable {
    /// `torsionPreferences_v2` — acyclic / general bonds.
    V2,
    /// `torsionPreferences_smallrings` — small-ring bonds.
    SmallRings,
    /// `torsionPreferences_macrocycles` — ring bonds of size ≥ 9.
    Macrocycles,
}

// ---------------------------------------------------------------------------
// Pattern compilation
// ---------------------------------------------------------------------------

/// A compiled table entry: the engine pattern, the M6 parameters, and
/// provenance.
struct CompiledPattern {
    smarts: &'static str,
    pattern: SmartsPattern,
    signs: [i8; 6],
    v: [f64; 6],
    table: TorsionTable,
}

/// Compile one `(smarts, signs, V)` row into a [`CompiledPattern`].
///
/// The SMARTS is parsed verbatim by the core engine, which natively supports
/// every primitive these tables use (including `r{lo-hi}` ring-size ranges and
/// `x<n>` ring connectivity).
fn compile_row(row: &TorsionRow, table: TorsionTable) -> Option<CompiledPattern> {
    let (smarts, signs, v) = (row.0, row.1, row.2);
    let pattern = SmartsPattern::parse(smarts).ok()?;
    Some(CompiledPattern {
        smarts,
        pattern,
        signs,
        v,
        table,
    })
}

/// Compile the tables for ETKDGv3 in RDKit concatenation order.
///
/// RDKit's `ETKDGv3()` sets `useSmallRingTorsions = false` and
/// `useMacrocycleTorsions = true`, so the concatenated parameter set is
/// **v2 ++ macrocycles** (the small-ring table is *not* included). We mirror
/// that exactly: matching v2 first, then macrocycles, first-match-wins. Rows
/// whose residue fails to parse are skipped (none in the validated set).
fn compile_all() -> Vec<CompiledPattern> {
    let mut out = Vec::new();
    for row in torsion_tables::V2 {
        if let Some(p) = compile_row(row, TorsionTable::V2) {
            out.push(p);
        }
    }
    for row in torsion_tables::MACROCYCLES {
        if let Some(p) = compile_row(row, TorsionTable::Macrocycles) {
            out.push(p);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Matching molecule (aromaticity transplanted from perception)
// ---------------------------------------------------------------------------

/// Build a working copy of `mol` whose atoms / bonds carry the `is_aromatic`
/// flag from the project perception, so the SMARTS engine's `a` / `c` / `:`
/// queries agree with RDKit (the engine reads `is_aromatic`, see
/// `molrs::chem::smarts` aromaticity convention).
fn aromatic_working_copy(mol: &Atomistic, p: &Perceived) -> Atomistic {
    let mut g = mol.clone();
    for (i, &aid) in p.atom_ids.iter().enumerate() {
        if p.atoms[i].aromatic {
            let _ = g.set_atom(aid, "is_aromatic", 1_i32);
        }
    }
    // Flag aromatic bonds so `:` and `BondFacts.aromatic` agree.
    let bond_ids: Vec<_> = g
        .bonds()
        .map(|(bid, b)| (bid, [b.nodes[0], b.nodes[1]]))
        .collect();
    let idx_of: HashMap<AtomId, usize> = p
        .atom_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();
    for (bid, [a, b]) in bond_ids {
        let (Some(&ia), Some(&ib)) = (idx_of.get(&a), idx_of.get(&b)) else {
            continue;
        };
        if p.is_aromatic_bond(ia, ib) {
            let _ = g.set_bond_prop(bid, "is_aromatic", PropValue::Int(1));
        }
    }
    g
}

// ---------------------------------------------------------------------------
// Assignment
// ---------------------------------------------------------------------------

/// A single assigned torsion with its provenance, for diagnostics / tests.
#[derive(Clone, Debug)]
pub struct AssignedTorsion {
    pub constraint: TorsionConstraint,
    pub table: TorsionTable,
}

/// Assign experimental torsions to `mol` by matching the full ETKDGv3 tables
/// (v2 ++ small-rings ++ macrocycles) through the SMARTS engine, reproducing
/// RDKit `getExperimentalTorsions`: the first matching pattern (global table
/// order) wins per rotatable bond; one torsion per bond.
///
/// `p` is the perception of `mol` (aromaticity / hybridization / rings); it is
/// reused to transplant aromatic flags onto the matching copy. The `r{…}` /
/// `x<n>` ring primitives are evaluated by the core SMARTS engine directly.
pub fn assign_with_provenance(mol: &Atomistic, p: &Perceived) -> Vec<AssignedTorsion> {
    let work = aromatic_working_copy(mol, p);
    let patterns = compile_all();

    let idx_of: HashMap<AtomId, usize> = p
        .atom_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();

    // RDKit keys "done" by central-bond index; we key by the unordered central
    // atom-index pair, which is equivalent for a simple molecular graph.
    let mut done: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
    let mut out: Vec<AssignedTorsion> = Vec::new();

    for cp in &patterns {
        for m in cp.pattern.find(&work, MatchOptions::default()) {
            // Resolve atom-map labels :1..:4 to mol-atom indices.
            let mut idx = [usize::MAX; 4];
            let mut ok = true;
            for (qi, &aid) in m.atoms.iter().enumerate() {
                if let Some(lbl) = cp.pattern.map_label(qi)
                    && (1..=4).contains(&lbl)
                {
                    match idx_of.get(&aid) {
                        Some(&ix) => idx[(lbl - 1) as usize] = ix,
                        None => {
                            ok = false;
                            break;
                        }
                    }
                }
            }
            if !ok || idx.contains(&usize::MAX) {
                continue;
            }

            let (j, k) = (idx[1], idx[2]);
            let key = if j < k { (j, k) } else { (k, j) };
            if done.contains(&key) {
                continue;
            }
            // The central pair must be a real bond.
            if !p.adj[j].contains(&k) {
                continue;
            }
            done.insert(key);
            out.push(AssignedTorsion {
                constraint: TorsionConstraint {
                    atoms: idx,
                    signs: cp.signs,
                    force_constants: cp.v,
                    pattern: cp.smarts,
                },
                table: cp.table,
            });
        }
    }
    out
}

/// Public entry point used by [`super::build_constraints`]: the bare
/// [`TorsionConstraint`] list (provenance dropped).
pub fn assign_experimental_torsions(mol: &Atomistic, p: &Perceived) -> Vec<TorsionConstraint> {
    assign_with_provenance(mol, p)
        .into_iter()
        .map(|a| a.constraint)
        .collect()
}
