//! Force-field WASM face — mirrors native molrs composition.
//!
//! ```js
//! const typifier = new UFFTypifier();
//! const typed    = typifier.typify(frame);
//! const pots     = typifier.toPotentials(typed);   // no .ff()
//! const report   = new LBFGS(pots /*, neighborList */).run(typed, 200);
//! // no neighborList → Optimizer builds a bruteforce (topology) pair list
//! ```
//!
//! No `typifyUff` / `insertIntramolecularPairs` / `typifier.ff()` façades.

use std::collections::HashSet;
use std::sync::Arc;

use js_sys::Uint32Array;
use wasm_bindgen::prelude::*;

use molrs::ff::forcefield::ForceField as RsForceField;
use molrs::ff::potential::{
    intramolecular_pairs as topology_pairs, Potential, Potentials as RsPotentials,
};
use molrs::ff::typifier::mmff::{
    MMFF94STypifier as RsMMFF94S, MMFF94Typifier as RsMMFF94,
};
use molrs::ff::typifier::uff::UFFTypifier as RsUFF;
use molrs::optimize::{set_free_mask, LBFGS as RsLBFGS, Optimizer};
use molrs::store::block::Block as RsBlock;
use molrs::store::frame::Frame as RsFrame;
use molrs::system::atomistic::Atomistic;
use molrs::types::U;
use ndarray::Array1;

use crate::compute::NeighborList;
use crate::core::frame::Frame;

// ── Typifiers ───────────────────────────────────────────────────────────────

macro_rules! wasm_typifier {
    (
        $(#[$meta:meta])*
        $JsName:ident, $RsType:ty, $ctor:expr
    ) => {
        $(#[$meta])*
        #[wasm_bindgen(js_name = $JsName)]
        pub struct $JsName {
            inner: $RsType,
        }

        #[wasm_bindgen(js_class = $JsName)]
        impl $JsName {
            #[wasm_bindgen(constructor)]
            pub fn new() -> $JsName {
                $JsName { inner: $ctor }
            }

            /// Typify a molecular [`Frame`]. Returns a **new** labeled frame.
            ///
            /// Native: `typifier.typify(&mol)?.to_frame()`.
            pub fn typify(&self, frame: &Frame) -> Result<Frame, JsValue> {
                let mol = frame.with_frame(|rs| {
                    Atomistic::from_frame(rs).map_err(|e| {
                        JsValue::from_str(&format!("Frame → Atomistic: {e}"))
                    })
                })?;
                let typed = self
                    .inner
                    .typify(&mol)
                    .map_err(|e| JsValue::from_str(&e))?;
                Frame::from_rs(typed.to_frame())
            }

            /// Compile molecule-bound potentials from a **typed** frame.
            ///
            /// Non-bonded terms need a `pairs` block; [`LBFGS::run`] installs
            /// that list (from a caller-supplied [`NeighborList`] or an internal
            /// bruteforce topology list) and recompiles before minimizing.
            /// Calling this alone with no `pairs` yields bonded-only kernels.
            ///
            /// Native: `typifier.ff().to_potentials(&frame)?` — the FF handle
            /// stays private; WASM collapses that to one method on the typifier.
            #[wasm_bindgen(js_name = toPotentials)]
            pub fn to_potentials(&self, frame: &Frame) -> Result<Potentials, JsValue> {
                let pots = frame.with_frame(|rs| {
                    self.inner
                        .ff()
                        .to_potentials(rs)
                        .map_err(|e| JsValue::from_str(&format!("to_potentials: {e}")))
                })?;
                Ok(Potentials {
                    ff: self.inner.ff().clone(),
                    inner: Arc::new(pots),
                })
            }
        }

        impl Default for $JsName {
            fn default() -> Self {
                Self::new()
            }
        }
    };
}

wasm_typifier!(
    /// Universal Force Field typifier (full RDKit default table).
    UFFTypifier,
    RsUFF,
    RsUFF::new()
);

wasm_typifier!(
    /// MMFF94 typifier.
    MMFF94Typifier,
    RsMMFF94,
    RsMMFF94::new()
);

wasm_typifier!(
    /// MMFF94s typifier (static / planar amide N).
    MMFF94STypifier,
    RsMMFF94S,
    RsMMFF94S::new()
);

// ── Potentials ──────────────────────────────────────────────────────────────

/// Compiled kernels. Holds the force-field skeleton so [`LBFGS`] can recompile
/// after installing a neighbour list.
#[wasm_bindgen(js_name = Potentials)]
pub struct Potentials {
    ff: RsForceField,
    inner: Arc<RsPotentials>,
}

#[wasm_bindgen(js_class = Potentials)]
impl Potentials {
    /// `{ energy: number, forces: Float64Array }` for flat 3N coordinates.
    #[wasm_bindgen(js_name = energyForces)]
    pub fn energy_forces(&self, coords: &js_sys::Float64Array) -> Result<JsValue, JsValue> {
        let mut buf = vec![0.0; coords.length() as usize];
        coords.copy_to(&mut buf);
        let (e, f) = self.inner.calc_energy_forces(&buf);
        let obj = js_sys::Object::new();
        js_sys::Reflect::set(&obj, &"energy".into(), &JsValue::from_f64(e))?;
        let fa = js_sys::Float64Array::new_with_length(f.len() as u32);
        fa.copy_from(&f);
        js_sys::Reflect::set(&obj, &"forces".into(), &fa)?;
        Ok(obj.into())
    }
}

// ── LBFGS ───────────────────────────────────────────────────────────────────

/// Pair source for non-bonded terms.
enum PairSource {
    /// O(N²) topology list: all i<j except 1-2 / 1-3, 1-4 flagged
    /// ([`topology_pairs`]). Default when no [`NeighborList`] is given.
    BruteForceTopology,
    /// Spatial neighbour list indices (1-2 / 1-3 still excluded at install).
    NeighborList { i: Vec<u32>, j: Vec<u32> },
}

/// Limited-memory BFGS.
///
/// Construct with potentials (and optional neighbor list), then
/// `run(frame, nSteps)`. If no neighbor list is given, the optimizer builds
/// an internal bruteforce topology pair list (all nonbonded pairs excluding
/// 1-2 / 1-3).
#[wasm_bindgen(js_name = LBFGS)]
pub struct LBFGS {
    ff: RsForceField,
    /// Latest compiled kernels (rebuilt at each `run` after pair install).
    pots: Arc<RsPotentials>,
    pairs: PairSource,
    fmax: f64,
    max_step: f64,
    memory: usize,
}

#[wasm_bindgen(js_class = LBFGS)]
impl LBFGS {
    /// Bind `pots`. Optional spatial [`NeighborList`]; if omitted, a bruteforce
    /// topology pair list is used at [`run`](Self::run) time.
    ///
    /// Knobs: `fmax` (default 0.05), `maxStep` (0.2), `memory` (8).
    /// Step count is the second argument of [`run`](Self::run).
    #[wasm_bindgen(constructor)]
    pub fn new(
        pots: &Potentials,
        neighbor_list: Option<NeighborList>,
        fmax: Option<f64>,
        max_step: Option<f64>,
        memory: Option<usize>,
    ) -> LBFGS {
        let pairs = match neighbor_list {
            Some(nl) => {
                let i = nl.inner.query_point_indices().to_vec();
                let j = nl.inner.point_indices().to_vec();
                PairSource::NeighborList { i, j }
            }
            None => PairSource::BruteForceTopology,
        };
        LBFGS {
            ff: pots.ff.clone(),
            pots: Arc::clone(&pots.inner),
            pairs,
            fmax: fmax.unwrap_or(0.05).max(1e-8),
            max_step: max_step.unwrap_or(0.2).max(1e-6),
            memory: memory.unwrap_or(8).max(1),
        }
    }

    /// Minimize `frame` coordinates **in place** for up to `nSteps` iterations.
    ///
    /// Installs the pair list (from the constructor's neighbour list or an
    /// internal bruteforce topology list), recompiles potentials, then runs
    /// L-BFGS. Optional `fixed`: dense atom indices held fixed.
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
                install_pairs(rs, &self.pairs)?;
                let compiled = self
                    .ff
                    .to_potentials(rs)
                    .map_err(|e| format!("to_potentials: {e}"))?;
                self.pots = Arc::new(compiled);

                if let Some(ref fixed) = fixed {
                    apply_fixed_mask(rs, fixed)?;
                }

                let pot: Arc<dyn Potential> = Arc::clone(&self.pots) as Arc<dyn Potential>;
                let mut opt =
                    RsLBFGS::new(pot, self.fmax, max_steps, self.max_step, self.memory);
                let report = Optimizer::run(&mut opt, rs)?;

                if fixed.is_some() {
                    if let Some(atoms) = rs.get_mut("atoms") {
                        let _ = atoms.remove("free");
                    }
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

// NeighborList.inner is private — expose a crate-visible accessor in compute.rs
// via a method we add below, or duplicate indices through public JS API.

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

fn install_pairs(frame: &mut RsFrame, source: &PairSource) -> Result<(), String> {
    let block = match source {
        PairSource::BruteForceTopology => topology_pairs(frame),
        PairSource::NeighborList { i, j } => pairs_from_indices(frame, i, j)?,
    };
    frame.insert("pairs", block);
    Ok(())
}

/// Build a `pairs` block from spatial neighbour indices, dropping 1-2 / 1-3
/// and flagging 1-4 from topology (same exclusions as [`topology_pairs`]).
fn pairs_from_indices(frame: &RsFrame, i: &[u32], j: &[u32]) -> Result<RsBlock, String> {
    if i.len() != j.len() {
        return Err(format!(
            "neighbor list length mismatch: i={} j={}",
            i.len(),
            j.len()
        ));
    }
    let excluded_12 = end_pairs(frame, "bonds", "atomi", "atomj");
    let excluded_13 = end_pairs(frame, "angles", "atomi", "atomk");
    let set_14 = end_pairs(frame, "dihedrals", "atomi", "atoml");

    let mut pi: Vec<U> = Vec::new();
    let mut pj: Vec<U> = Vec::new();
    let mut p14: Vec<bool> = Vec::new();
    let mut seen = HashSet::new();

    for (&a, &b) in i.iter().zip(j.iter()) {
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        let key = (lo as usize, hi as usize);
        if !seen.insert(key) {
            continue;
        }
        if excluded_12.contains(&key) || excluded_13.contains(&key) {
            continue;
        }
        pi.push(lo as U);
        pj.push(hi as U);
        p14.push(set_14.contains(&key));
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

fn end_pairs(
    frame: &RsFrame,
    block: &str,
    col_a: &str,
    col_b: &str,
) -> HashSet<(usize, usize)> {
    let Some(b) = frame.get(block) else {
        return HashSet::new();
    };
    let (Some(a_col), Some(b_col)) = (b.get_uint(col_a), b.get_uint(col_b)) else {
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
