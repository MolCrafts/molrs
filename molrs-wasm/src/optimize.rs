//! Geometry optimization — the WASM face of `molrs::optimize` (`LBFGS`).
//!
//! ```js
//! const pots   = new UFFTypifier().toPotentials(typed);
//! const nl     = new NeighborList(12.5);         // or NeighborList.bruteForce
//! nl.build(typed);
//! const report = new LBFGS(pots, nl.neighbors()).run(typed, 200);
//! ```
//!
//! The non-bonded pairs come from a [`NeighborList`](crate::core::NeighborList)
//! and nowhere else: the optimizer builds no pair list of its own.

use std::collections::HashSet;
use std::sync::Arc;

use js_sys::Uint32Array;
use wasm_bindgen::prelude::*;

use molrs::core::Block as RsBlock;
use molrs::core::Frame as RsFrame;
use molrs::ff::forcefield::ForceField as RsForceField;
use molrs::ff::potential::{Potential, PotentialCompiler, Potentials as RsPotentials};
use molrs::op::types::Idx;
use molrs::optimize::{LBFGS as RsLBFGS, Optimizer, set_free_mask};
use ndarray::Array1;

use crate::core::Neighbors;
use crate::core::frame::Frame;
use crate::ff::Potentials;

// ── LBFGS ───────────────────────────────────────────────────────────────────

/// Limited-memory BFGS.
///
/// Construct with potentials and a neighbor table, then `run(frame, nSteps)`.
///
/// The table is a [`NeighborList`](crate::core::NeighborList)'s
/// [`Neighbors`], built at the force field's non-bonded cutoff (or
/// `NeighborList.bruteForce` for a small molecule). Its pairs are installed
/// as the frame's `pairs` with the force field's `special_bonds` applied: a
/// 1-2 / 1-3 pair the weights exclude is dropped, a dihedral's end pair is
/// flagged 1-4 — the rules of `molrs::ff::potential::intramolecular_pairs`.
#[wasm_bindgen(js_name = LBFGS)]
pub struct LBFGS {
    ff: RsForceField,
    /// Latest compiled kernels (rebuilt at each `run` after pair install).
    pots: Arc<RsPotentials>,
    /// Neighbor pair indices `(i, j)` from the constructor's table.
    pair_i: Vec<u32>,
    pair_j: Vec<u32>,
    fmax: f64,
    max_step: f64,
    memory: usize,
}

#[wasm_bindgen(js_class = LBFGS)]
impl LBFGS {
    /// Bind `pots` and the [`Neighbors`] table from
    /// [`NeighborList::neighbors`](crate::core::NeighborList::neighbors).
    ///
    /// Knobs: `fmax` (default 0.05), `maxStep` (0.2), `memory` (8).
    /// Step count is the second argument of [`run`](Self::run).
    #[wasm_bindgen(constructor)]
    pub fn new(
        pots: &Potentials,
        neighbors: &Neighbors,
        fmax: Option<f64>,
        max_step: Option<f64>,
        memory: Option<usize>,
    ) -> LBFGS {
        LBFGS {
            ff: pots.ff.clone(),
            pots: Arc::clone(&pots.inner),
            pair_i: neighbors.inner.query_point_indices().to_vec(),
            pair_j: neighbors.inner.point_indices().to_vec(),
            fmax: fmax.unwrap_or(0.05).max(1e-8),
            max_step: max_step.unwrap_or(0.2).max(1e-6),
            memory: memory.unwrap_or(8).max(1),
        }
    }

    /// Minimize `frame` coordinates **in place** for up to `nSteps` iterations.
    ///
    /// Installs the constructor's neighbour pairs, recompiles potentials,
    /// then runs L-BFGS. Optional `fixed`: dense atom indices held fixed.
    pub fn run(
        &mut self,
        frame: &Frame,
        n_steps: Option<usize>,
        fixed: Option<Uint32Array>,
    ) -> Result<OptReport, JsValue> {
        let max_steps = n_steps.unwrap_or(200).max(1);

        // Install pairs + recompile, then minimize — all on the same borrow.
        let report = frame
            .inner
            .with_mut(|rs| -> Result<molrs::optimize::OptReport, String> {
                let pairs =
                    pairs_from_indices(rs, &self.pair_i, &self.pair_j, self.ff.special_bonds())?;
                rs.insert("pairs", pairs);
                let compiled = PotentialCompiler::new(&self.ff)
                    .compile(rs)
                    .map_err(|e| format!("compile: {e}"))?;
                self.pots = Arc::new(compiled);

                if let Some(ref fixed) = fixed {
                    apply_fixed_mask(rs, fixed)?;
                }

                let pot: Arc<dyn Potential> = Arc::clone(&self.pots) as Arc<dyn Potential>;
                let mut opt = RsLBFGS::new(pot, self.fmax, max_steps, self.max_step, self.memory);
                let report = Optimizer::run(&mut opt, rs)?;

                if fixed.is_some()
                    && let Some(atoms) = rs.get_mut("atoms")
                {
                    let _ = atoms.remove("free");
                }
                Ok(report)
            })
            .map_err(|e| JsValue::from_str(&e.to_string()))?
            .map_err(|e| JsValue::from_str(&e))?;

        Ok(OptReport {
            steps: report.n_steps,
            energy: report.final_energy,
            max_force: report.final_fmax,
            converged: report.converged,
        })
    }
}

// ── OptReport ───────────────────────────────────────────────────────────────

#[wasm_bindgen(js_name = OptReport)]
pub struct OptReport {
    steps: usize,
    energy: f64,
    max_force: f64,
    converged: bool,
}

#[wasm_bindgen(js_class = OptReport)]
impl OptReport {
    #[wasm_bindgen(getter)]
    pub fn steps(&self) -> usize {
        self.steps
    }
    #[wasm_bindgen(getter)]
    pub fn energy(&self) -> f64 {
        self.energy
    }
    #[wasm_bindgen(getter, js_name = maxForce)]
    pub fn max_force(&self) -> f64 {
        self.max_force
    }
    #[wasm_bindgen(getter)]
    pub fn converged(&self) -> bool {
        self.converged
    }
}

// ── pair install ────────────────────────────────────────────────────────────

/// Build a `pairs` block from spatial neighbour indices by the rules of
/// `molrs::ff::potential::intramolecular_pairs`: a 1-2 / 1-3 pair is dropped
/// unless `special` keeps its class, and a dihedral's end pair is flagged 1-4
/// unless it is also a 1-2 or 1-3 pair (a ring closes it).
fn pairs_from_indices(
    frame: &RsFrame,
    i: &[u32],
    j: &[u32],
    special: &molrs::ff::forcefield::SpecialBonds,
) -> Result<RsBlock, String> {
    let [keep_12, keep_13] = special.compiled_inclusion()?;
    if i.len() != j.len() {
        return Err(format!(
            "neighbor list length mismatch: i={} j={}",
            i.len(),
            j.len()
        ));
    }
    let pairs_12 = end_pairs(frame, "bonds", "atomi", "atomj");
    let pairs_13 = end_pairs(frame, "angles", "atomi", "atomk");
    let set_14 = end_pairs(frame, "dihedrals", "atomi", "atoml");

    let mut pi: Vec<Idx> = Vec::new();
    let mut pj: Vec<Idx> = Vec::new();
    let mut p14: Vec<bool> = Vec::new();
    let mut seen = HashSet::new();

    for (&a, &b) in i.iter().zip(j.iter()) {
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        let key = (lo as usize, hi as usize);
        if !seen.insert(key) {
            continue;
        }
        let (is_12, is_13) = (pairs_12.contains(&key), pairs_13.contains(&key));
        if (!keep_12 && is_12) || (!keep_13 && is_13) {
            continue;
        }
        pi.push(lo as Idx);
        pj.push(hi as Idx);
        p14.push(set_14.contains(&key) && !is_12 && !is_13);
    }

    let mut pairs = RsBlock::new();
    if !pi.is_empty() {
        pairs
            .insert("atomi", Array1::from_vec(pi).into_dyn())
            .map_err(|e| e.to_string())?;
        pairs
            .insert("atomj", Array1::from_vec(pj).into_dyn())
            .map_err(|e| e.to_string())?;
        pairs
            .insert("is_14", Array1::from_vec(p14).into_dyn())
            .map_err(|e| e.to_string())?;
    }
    Ok(pairs)
}

fn end_pairs(frame: &RsFrame, block: &str, col_a: &str, col_b: &str) -> HashSet<(usize, usize)> {
    let Some(b) = frame.get(block) else {
        return HashSet::new();
    };
    let (Some(a_col), Some(b_col)) = (
        b.get(col_a).and_then(|c| c.as_uint()),
        b.get(col_b).and_then(|c| c.as_uint()),
    ) else {
        return HashSet::new();
    };
    a_col
        .iter()
        .zip(b_col.iter())
        .map(|(&i, &j)| {
            let (i, j) = (i as usize, j as usize);
            if i < j { (i, j) } else { (j, i) }
        })
        .collect()
}

fn apply_fixed_mask(frame: &mut RsFrame, fixed: &Uint32Array) -> Result<(), String> {
    let n = frame.get("atoms").and_then(|b| b.nrows()).unwrap_or(0);
    if n == 0 || fixed.length() == 0 {
        return Ok(());
    }
    let mut free = vec![true; n];
    for i in 0..fixed.length() {
        let idx = fixed.get_index(i) as usize;
        if idx < n {
            free[idx] = false;
        }
    }
    set_free_mask(frame, &free)
}
