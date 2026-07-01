//! Force-field-agnostic geometry optimization (energy minimization).
//!
//! The [`LBFGS`] class relaxes a single structure ([`run`](LBFGS::run)) or a
//! homogeneous batch ([`run_batch`](LBFGS::run_batch)) using the shared L-BFGS
//! core in [`lbfgs`]. It consumes only the `(energy, forces = -grad)` contract
//! of [`Potential`], so it works with any force
//! field — MMFF94, the harmonic/LJ kernels, or a user-supplied potential.
//!
//! Coordinates use the same flat layout as the rest of `molrs-ff`:
//! `[x0,y0,z0, x1,y1,z1, ...]` (`3·n_atoms` elements, `f64`). Convergence is on
//! `fmax` — the maximum per-atom force magnitude `max_i ‖F_i‖` (kcal/mol/Å) —
//! matching the ASE / molpy convention.

pub mod lbfgs;

// The contract every potential (force-field kernel or hand-built) implements,
// and the only thing the optimizer needs. Lives here with the optimizer so
// optimization does not depend on the force-field (`ff`) feature.
pub trait Potential: Send + Sync {
    /// Compute energy and forces (= -gradient) in one pass.
    /// Returns `(energy, forces)` where forces has length `coords.len()`.
    fn calc_energy_forces(&self, coords: &[F]) -> (F, Vec<F>);

    /// Compute total potential energy (kcal/mol).
    fn calc_energy(&self, coords: &[F]) -> F {
        self.calc_energy_forces(coords).0
    }

    /// Compute forces (= -gradient), a length-3N vector.
    fn calc_forces(&self, coords: &[F]) -> Vec<F> {
        self.calc_energy_forces(coords).1
    }
}

use lbfgs::{Converge, fmax_from_grad, minimize_core};
use molrs::types::F;

pub use lbfgs::{MinResult, minimize_lbfgs_rms};

/// Convergence and step controls for the [`LBFGS`] optimizer.
#[derive(Clone, Copy, Debug)]
pub struct LbfgsConfig {
    /// Maximum per-atom force magnitude for convergence (kcal/mol/Å).
    /// Optimization stops when `max_i ‖F_i‖ < fmax`. Default `0.05`.
    pub fmax: F,
    /// Outer-iteration cap. Default `500`.
    pub max_steps: usize,
    /// Per-step displacement cap in Å (trust region). Default `0.2`. Use
    /// `f64::INFINITY` to disable the trust region (unbounded line search).
    pub max_step: F,
    /// L-BFGS correction-pair history size. Default `8`.
    pub memory: usize,
}

impl Default for LbfgsConfig {
    fn default() -> Self {
        Self {
            fmax: 0.05,
            max_steps: 500,
            max_step: 0.2,
            memory: 8,
        }
    }
}

/// Outcome of a single minimization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OptReport {
    /// Whether `fmax` convergence was reached within `max_steps`.
    pub converged: bool,
    /// Number of outer L-BFGS iterations performed.
    pub n_steps: usize,
    /// Potential energy at the returned point (kcal/mol).
    pub final_energy: F,
    /// Maximum per-atom force magnitude at the returned point (kcal/mol/Å).
    pub final_fmax: F,
}

/// Limited-memory BFGS geometry optimizer over a molecule-bound [`Potential`].
///
/// Mirrors molpy's `LBFGS(potential, …).run(structure)`: hold the potential and
/// config, then [`run`](LBFGS::run) a single structure or [`run_batch`] a
/// homogeneous set. Coordinates are flat `[x0,y0,z0, …]` and relaxed in place.
///
/// [`run_batch`]: LBFGS::run_batch
pub struct LBFGS<'p> {
    potential: &'p dyn Potential,
    cfg: LbfgsConfig,
}

impl<'p> LBFGS<'p> {
    /// Create an optimizer bound to `potential` with the given config.
    pub fn new(potential: &'p dyn Potential, cfg: LbfgsConfig) -> Self {
        Self { potential, cfg }
    }

    /// Relax a single structure in place. `coords` is the flat `3·n_atoms`
    /// buffer, updated to the located minimizer.
    ///
    /// # Errors
    /// Returns `Err` if `coords.len()` is not a multiple of three.
    pub fn run(&self, coords: &mut [F]) -> Result<OptReport, String> {
        if coords.is_empty() {
            return Ok(OptReport {
                converged: true,
                n_steps: 0,
                final_energy: 0.0,
                final_fmax: 0.0,
            });
        }
        if !coords.len().is_multiple_of(3) {
            return Err(format!(
                "coords length {} is not a multiple of 3 (expected 3·n_atoms)",
                coords.len()
            ));
        }
        Ok(self.run_one(coords))
    }

    /// Relax a homogeneous batch in place, one [`OptReport`] per structure.
    ///
    /// All `n_structs` structures share this potential. `coords` is the
    /// concatenation of `n_structs` flat blocks each of length `3·n_atoms`
    /// (row-major over the conceptual `(B, N, 3)` array). Each block is optimized
    /// independently; with the `rayon` feature the blocks run in parallel.
    ///
    /// # Errors
    /// Returns `Err` if `coords.len() != n_structs · n_atoms · 3`, or a
    /// zero-atom batch.
    pub fn run_batch(
        &self,
        coords: &mut [F],
        n_atoms: usize,
        n_structs: usize,
    ) -> Result<Vec<OptReport>, String> {
        let stride = n_atoms * 3;
        let expected = n_structs * stride;
        if coords.len() != expected {
            return Err(format!(
                "coords length {} != n_structs ({}) · n_atoms ({}) · 3 = {}",
                coords.len(),
                n_structs,
                n_atoms,
                expected
            ));
        }
        if n_structs == 0 {
            return Ok(Vec::new());
        }
        // `chunks_mut(0)` / `par_chunks_mut(0)` panic, so reject a zero-atom
        // batch explicitly rather than letting it reach the split.
        if stride == 0 {
            return Err(format!(
                "n_atoms must be > 0 for a batch of {n_structs} structures"
            ));
        }

        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            Ok(coords
                .par_chunks_mut(stride)
                .map(|block| self.run_one(block))
                .collect())
        }
        #[cfg(not(feature = "rayon"))]
        {
            Ok(coords
                .chunks_mut(stride)
                .map(|block| self.run_one(block))
                .collect())
        }
    }

    /// Single-structure minimization without the length check — the batch path
    /// already guarantees `block.len() == 3·n_atoms`.
    fn run_one(&self, coords: &mut [F]) -> OptReport {
        let (final_energy, grad, n_steps, converged) = minimize_core(
            coords,
            self.cfg.max_steps,
            Converge::Fmax(self.cfg.fmax),
            self.cfg.max_step,
            self.cfg.memory,
            |c| self.potential.calc_energy_forces(c),
        );
        OptReport {
            converged,
            n_steps,
            final_energy,
            final_fmax: fmax_from_grad(&grad),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two atoms joined by a single harmonic bond `E = ½k(r − r0)²` along x.
    /// Force on each atom is along the bond; the system relaxes to `r == r0`.
    struct HarmonicBond {
        k: F,
        r0: F,
    }

    impl Potential for HarmonicBond {
        fn calc_energy_forces(&self, coords: &[F]) -> (F, Vec<F>) {
            // atoms 0 and 1, vector from 0 -> 1.
            let d = [
                coords[3] - coords[0],
                coords[4] - coords[1],
                coords[5] - coords[2],
            ];
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            let e = 0.5 * self.k * (r - self.r0) * (r - self.r0);
            // dE/dr = k(r - r0); force on atom 1 = -dE/dr * d/r, atom 0 opposite.
            let mut f = vec![0.0; 6];
            if r > 1e-12 {
                let coeff = self.k * (r - self.r0) / r; // dE/dr / r
                for i in 0..3 {
                    let fi = coeff * d[i]; // d(E)/d(x1_i) = coeff * d_i ... gradient
                    f[i] = fi; // force on atom 0 = -grad_0 = +coeff*d  (since grad_0 = -coeff*d)
                    f[3 + i] = -fi; // force on atom 1
                }
            }
            (e, f)
        }
    }

    fn opts() -> LbfgsConfig {
        LbfgsConfig::default()
    }

    #[test]
    fn relaxes_harmonic_bond_to_equilibrium() {
        let pot = HarmonicBond { k: 100.0, r0: 1.0 };
        // start stretched to 1.5 Å along x.
        let mut coords = vec![0.0, 0.0, 0.0, 1.5, 0.0, 0.0];
        let report = LBFGS::new(&pot, opts()).run(&mut coords).unwrap();
        assert!(report.converged, "should converge: {report:?}");
        let r = coords[3] - coords[0];
        assert!(
            (r.abs() - 1.0).abs() < 1e-6,
            "bond length should reach r0, got {r}"
        );
        assert!(
            report.final_energy < 1e-9,
            "energy ~0, got {}",
            report.final_energy
        );
        assert!(report.final_fmax <= opts().fmax, "fmax satisfied");
    }

    #[test]
    fn fmax_convergence_semantics() {
        let pot = HarmonicBond { k: 100.0, r0: 1.0 };
        // max_steps = 1 from far away -> not converged, exactly one step.
        let mut coords = vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let o = LbfgsConfig {
            max_steps: 1,
            ..opts()
        };
        let r = LBFGS::new(&pot, o).run(&mut coords).unwrap();
        assert!(!r.converged);
        assert_eq!(r.n_steps, 1);
    }

    #[test]
    fn idempotent_at_minimum() {
        let pot = HarmonicBond { k: 100.0, r0: 1.0 };
        let mut coords = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // already at r0
        let r = LBFGS::new(&pot, opts()).run(&mut coords).unwrap();
        assert!(r.converged);
        assert!(
            r.n_steps <= 1,
            "already-minimized takes <=1 step, took {}",
            r.n_steps
        );
    }

    #[test]
    fn single_atom_converges_immediately() {
        // A lone atom with no forces: fmax is already 0.
        struct Free;
        impl Potential for Free {
            fn calc_energy_forces(&self, coords: &[F]) -> (F, Vec<F>) {
                (0.0, vec![0.0; coords.len()])
            }
        }
        let mut coords = vec![0.3, -0.2, 0.1];
        let r = LBFGS::new(&Free, opts()).run(&mut coords).unwrap();
        assert!(r.converged);
        assert!(r.n_steps <= 1);
    }

    #[test]
    fn rejects_non_multiple_of_three() {
        struct Free;
        impl Potential for Free {
            fn calc_energy_forces(&self, coords: &[F]) -> (F, Vec<F>) {
                (0.0, vec![0.0; coords.len()])
            }
        }
        let mut coords = vec![0.0, 0.0, 0.0, 1.0];
        assert!(LBFGS::new(&Free, opts()).run(&mut coords).is_err());
    }

    #[test]
    fn empty_coords_is_converged_noop() {
        struct Free;
        impl Potential for Free {
            fn calc_energy_forces(&self, coords: &[F]) -> (F, Vec<F>) {
                (0.0, vec![0.0; coords.len()])
            }
        }
        let mut coords: Vec<F> = vec![];
        let r = LBFGS::new(&Free, opts()).run(&mut coords).unwrap();
        assert!(r.converged);
        assert_eq!(r.n_steps, 0);
    }

    #[test]
    fn trust_region_caps_step() {
        // A stiff bond far from equilibrium would take a large step without a
        // trust region; with a tiny max_step every component move is bounded.
        let pot = HarmonicBond { k: 500.0, r0: 1.0 };
        let mut coords = vec![0.0, 0.0, 0.0, 3.0, 0.0, 0.0];
        let before = coords.clone();
        let o = LbfgsConfig {
            max_steps: 1,
            max_step: 0.01,
            ..opts()
        };
        LBFGS::new(&pot, o).run(&mut coords).unwrap();
        for (a, b) in coords.iter().zip(&before) {
            assert!(
                (a - b).abs() <= 0.01 + 1e-12,
                "component moved {} > max_step",
                (a - b).abs()
            );
        }
    }

    #[test]
    fn batch_equals_serial() {
        let pot = HarmonicBond { k: 100.0, r0: 1.0 };
        let single_start = vec![0.0, 0.0, 0.0, 1.4, 0.0, 0.0];

        let mut single = single_start.clone();
        let single_report = LBFGS::new(&pot, opts()).run(&mut single).unwrap();

        // 4 identical copies stacked.
        let b = 4;
        let mut batch: Vec<F> = Vec::new();
        for _ in 0..b {
            batch.extend_from_slice(&single_start);
        }
        let reports = LBFGS::new(&pot, opts())
            .run_batch(&mut batch, 2, b)
            .unwrap();
        assert_eq!(reports.len(), b);
        for (i, rep) in reports.iter().enumerate() {
            assert!((rep.final_energy - single_report.final_energy).abs() < 1e-10);
            assert!((rep.final_fmax - single_report.final_fmax).abs() < 1e-10);
            let block = &batch[i * 6..i * 6 + 6];
            for (a, s) in block.iter().zip(&single) {
                assert!((a - s).abs() < 1e-9, "batch block {i} diverged from serial");
            }
        }
    }

    #[test]
    fn batch_rejects_size_mismatch() {
        let pot = HarmonicBond { k: 100.0, r0: 1.0 };
        let mut coords = vec![0.0; 6 * 3 + 1]; // not 3 structs * 2 atoms * 3
        assert!(
            LBFGS::new(&pot, opts())
                .run_batch(&mut coords, 2, 3)
                .is_err()
        );
    }

    #[test]
    fn batch_zero_structs_is_empty() {
        let pot = HarmonicBond { k: 100.0, r0: 1.0 };
        let mut coords: Vec<F> = vec![];
        let reports = LBFGS::new(&pot, opts())
            .run_batch(&mut coords, 2, 0)
            .unwrap();
        assert!(reports.is_empty());
    }

    #[test]
    fn batch_zero_atoms_errors_not_panics() {
        // A (B, 0, 3) batch: n_atoms == 0 -> stride 0 would panic chunks_mut.
        let pot = HarmonicBond { k: 100.0, r0: 1.0 };
        let mut coords: Vec<F> = vec![];
        assert!(
            LBFGS::new(&pot, opts())
                .run_batch(&mut coords, 0, 3)
                .is_err()
        );
    }
}
