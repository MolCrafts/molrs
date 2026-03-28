//! Handler trait and built-in handlers for packing progress callbacks.

use molrs::types::F;
use std::io::BufWriter;
use std::path::PathBuf;
use std::time::Instant;

use crate::context::PackContext;
use crate::frame::compute_mol_ids;
use crate::numerics::objective_small_floor;

// ── Info structs ─────────────────────────────────────────────────────────────

/// Information about the current packing phase.
#[derive(Debug, Clone, Copy)]
pub struct PhaseInfo {
    /// 0-based phase index.
    pub phase: usize,
    /// Total number of phases (ntype + 1).
    pub total_phases: usize,
    /// If `Some(itype)`, this is a per-type compaction phase.
    /// If `None`, this is the final all-types phase.
    pub molecule_type: Option<usize>,
}

/// Per-iteration progress snapshot.
#[derive(Debug, Clone)]
pub struct StepInfo {
    /// 0-based loop iteration within the current phase.
    pub loop_idx: usize,
    /// Maximum loops for this phase.
    pub max_loops: usize,
    /// Current phase info.
    pub phase: PhaseInfo,
    /// Max inter-molecular overlap violation (0.0 = no overlap).
    pub fdist: F,
    /// Max constraint violation (0.0 = all constraints satisfied).
    pub frest: F,
    /// Improvement from last iteration, as percentage (positive = improving).
    pub improvement_pct: F,
    /// Current radius scaling factor (starts at discale, decays to 1.0).
    pub radscale: F,
    /// Convergence precision target.
    pub precision: F,
    /// Hook acceptance rates: `(type_index, acceptance_rate)`.
    pub hook_acceptance: Vec<(usize, F)>,
}

// ── Trait ─────────────────────────────────────────────────────────────────────

/// Callback interface called by [`crate::packer::Molpack`] during packing.
pub trait Handler: Send {
    /// Called immediately at the start of [`pack`][crate::packer::Molpack::pack],
    /// before any computation. Use this for immediate user feedback.
    fn on_start(&mut self, _ntotat: usize, _ntotmol: usize) {}

    /// Called once after initialization completes, with valid `xcart` positions.
    /// Use this to write the initial conformation (e.g. [`XYZHandler`]).
    fn on_initial(&mut self, _sys: &PackContext) {}

    /// Called after each outer optimization loop iteration.
    fn on_step(&mut self, info: &StepInfo, sys: &PackContext);

    /// Called at the start of each packing phase (per-type and all-types).
    /// Allows stateful handlers to reset between phases.
    fn on_phase_start(&mut self, _info: &PhaseInfo) {}

    /// Called once after the packing loop finishes (convergence or max loops).
    fn on_finish(&mut self, _sys: &PackContext) {}

    /// Return `true` to request early termination of the packing loop.
    fn should_stop(&self) -> bool {
        false
    }
}

// ── NullHandler ───────────────────────────────────────────────────────────────

/// A no-op handler.
pub struct NullHandler;

impl Handler for NullHandler {
    fn on_step(&mut self, _info: &StepInfo, _sys: &PackContext) {}
}

// ── XYZHandler ────────────────────────────────────────────────────────────────

/// Writes packing snapshots as a multi-frame XYZ trajectory.
///
/// Writes a frame on every `every`-th step (starting from step 0).
/// No automatic initial or final writes.
pub struct XYZHandler {
    path: PathBuf,
    /// Write every `n` steps (must be >= 1).
    every: usize,
    file: Option<BufWriter<std::fs::File>>,
    /// Precomputed global molecule ID per atom (constant across all frames).
    mol_ids: Vec<usize>,
}

impl XYZHandler {
    /// Create a new XYZ trajectory handler.
    ///
    /// `every` controls writing frequency: a frame is written on every step
    /// where `loop_idx % every == 0` (step 0 is always included).
    /// `every` must be >= 1.
    pub fn new(path: impl Into<PathBuf>, every: usize) -> Self {
        assert!(every >= 1, "every must be >= 1");
        Self {
            path: path.into(),
            every,
            file: None,
            mol_ids: Vec::new(),
        }
    }

    fn open(&mut self) {
        if self.file.is_some() {
            return;
        }
        match std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&self.path)
        {
            Ok(f) => self.file = Some(BufWriter::new(f)),
            Err(e) => log::warn!("XYZHandler: cannot open {}: {e}", self.path.display()),
        }
    }

    fn write_frame(&mut self, comment: &str, sys: &PackContext) {
        self.open();
        let Some(ref mut w) = self.file else { return };
        use std::io::Write;
        let nat = sys.xcart.len();
        let _ = writeln!(w, "{nat}");
        let _ = writeln!(w, "Properties=species:S:1:pos:R:3:mol:I:1  {comment}");
        for (icart, pos) in sys.xcart.iter().enumerate() {
            let elem = sys
                .elements
                .get(icart)
                .and_then(|e| *e)
                .map(|e| e.symbol())
                .unwrap_or("X");
            let mol_id = self.mol_ids.get(icart).copied().unwrap_or(0);
            let _ = writeln!(
                w,
                "{elem}  {:.6}  {:.6}  {:.6}  {mol_id}",
                pos[0], pos[1], pos[2]
            );
        }
        let _ = w.flush();
    }
}

impl Handler for XYZHandler {
    fn on_initial(&mut self, sys: &PackContext) {
        self.mol_ids = compute_mol_ids(sys);
    }

    fn on_step(&mut self, info: &StepInfo, sys: &PackContext) {
        if info.loop_idx.is_multiple_of(self.every) {
            self.write_frame(&format!("step {}", info.loop_idx), sys);
        }
    }
}

// ── ProgressHandler ───────────────────────────────────────────────────────────

/// Prints human-readable progress lines to `stderr`.
///
/// Added as a default handler by [`crate::packer::Molpack::new`].
pub struct ProgressHandler {
    start: Option<Instant>,
}

impl ProgressHandler {
    pub fn new() -> Self {
        Self { start: None }
    }
}

impl Default for ProgressHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl Handler for ProgressHandler {
    fn on_start(&mut self, ntotat: usize, ntotmol: usize) {
        self.start = Some(Instant::now());
        eprintln!("Packing {ntotmol} molecules ({ntotat} atoms)...");
    }

    fn on_initial(&mut self, sys: &PackContext) {
        let elapsed = self.start.map(|t| t.elapsed().as_secs_f64()).unwrap_or(0.0);
        eprintln!(
            "  Initializing... done ({:.1}s)  overlap: {:.4e}  constraints: {:.4e}",
            elapsed, sys.fdist, sys.frest
        );
    }

    fn on_phase_start(&mut self, info: &PhaseInfo) {
        let desc = match info.molecule_type {
            Some(itype) => format!("Compacting type {itype}"),
            None => "Optimizing all types together".to_string(),
        };
        eprintln!("  Phase [{}/{}] {desc}", info.phase + 1, info.total_phases,);
    }

    fn on_step(&mut self, info: &StepInfo, _sys: &PackContext) {
        let elapsed = self.start.map(|t| t.elapsed().as_secs_f64()).unwrap_or(0.0);
        eprintln!(
            "    Step [{}/{}]  overlap: {:.2e}  constraints: {:.2e}  improved {:.1}%  ({:.1}s)",
            info.loop_idx + 1,
            info.max_loops,
            info.fdist,
            info.frest,
            info.improvement_pct,
            elapsed,
        );
    }

    fn on_finish(&mut self, sys: &PackContext) {
        let elapsed = self.start.map(|t| t.elapsed().as_secs_f64()).unwrap_or(0.0);
        if sys.fdist < 0.01 && sys.frest < 0.01 {
            eprintln!(
                "  Converged in {:.1}s — overlap: {:.2e}  constraints: {:.2e}",
                elapsed, sys.fdist, sys.frest,
            );
        } else {
            eprintln!(
                "  Did not converge ({:.1}s) — overlap: {:.2e}  constraints: {:.2e}",
                elapsed, sys.fdist, sys.frest,
            );
        }
    }
}

// ── EarlyStopHandler ──────────────────────────────────────────────────────────

/// Requests early termination when improvement stalls.
///
/// Tracks `fdist + frest` total violation. After `warmup` iterations, if the
/// relative improvement drops below `threshold` for `patience` consecutive
/// steps, sets the stop flag.
///
/// Added as a default handler by [`crate::packer::Molpack::new`].
pub struct EarlyStopHandler {
    /// Relative improvement threshold.
    pub threshold: F,
    /// Iterations to skip before tracking. Default: `5`.
    pub warmup: usize,
    /// Consecutive stall iterations before stopping. Default: `3`.
    pub patience: usize,
    prev_violation: F,
    stall_count: usize,
    stop: bool,
}

impl EarlyStopHandler {
    pub fn new(threshold: F) -> Self {
        Self {
            threshold,
            warmup: 5,
            patience: 3,
            prev_violation: F::INFINITY,
            stall_count: 0,
            stop: false,
        }
    }

    pub fn with_warmup(mut self, warmup: usize) -> Self {
        self.warmup = warmup;
        self
    }

    pub fn with_patience(mut self, patience: usize) -> Self {
        self.patience = patience.max(1);
        self
    }
}

impl Default for EarlyStopHandler {
    /// Default is intentionally conservative (effectively disabled),
    /// so Packmol-sized examples are not stopped before convergence.
    fn default() -> Self {
        Self::new(F::NEG_INFINITY)
    }
}

impl Handler for EarlyStopHandler {
    fn on_initial(&mut self, _sys: &PackContext) {
        self.prev_violation = F::INFINITY;
        self.stall_count = 0;
        self.stop = false;
    }

    fn on_phase_start(&mut self, _info: &PhaseInfo) {
        self.prev_violation = F::INFINITY;
        self.stall_count = 0;
        self.stop = false;
    }

    fn on_step(&mut self, info: &StepInfo, _sys: &PackContext) {
        let v = info.fdist + info.frest;
        if info.loop_idx <= self.warmup {
            self.prev_violation = v;
            return;
        }
        let rel_change = if self.prev_violation > 0.0 {
            (self.prev_violation - v) / self.prev_violation
        } else if v < objective_small_floor() {
            1.0 // already converged
        } else {
            F::INFINITY
        };

        if rel_change < self.threshold {
            self.stall_count += 1;
            if self.stall_count >= self.patience {
                log::debug!(
                    "EarlyStop: stalled for {} iters (rel_change={:.2e})",
                    self.stall_count,
                    rel_change
                );
                self.stop = true;
            }
        } else {
            self.stall_count = 0;
        }
        self.prev_violation = v;
    }

    fn should_stop(&self) -> bool {
        self.stop
    }
}
