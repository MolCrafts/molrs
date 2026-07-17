//! ETKDGv3 conformer-embedding pipeline.
//!
//! This is a port of RDKit's `EmbedMolecule` orchestration
//! (`$RDBASE/Code/GraphMol/DistGeomHelpers/Embedder.cpp`, BSD-3, Copyright (C)
//! 2004-2025 Greg Landrum / Sereina Riniker and other RDKit contributors)
//! wired onto MolCrafts' own constraint generator (`crate::conformer::distgeom`,
//! ETKDGv3 bounds + experimental torsions + chiral sets) and the MMFF94
//! force field (`molrs::ff::mmff`) for the second-stage cleanup.
//!
//! ## Stages (mapped onto the public `StageKind` variants)
//! 1. `Preprocess`    — optional hydrogen addition.
//! 2. `BuildInitial`  — `build_constraints` → metrization sample → 4D
//!    eigenvalue embedding (`embed4d`).
//! 3. `CoarseOptimize`— first-stage 4D distance/chiral/fourth-dim minimization
//!    then 3D experimental-torsion refinement (`etmin`).
//! 4. `FinalOptimize` — second-stage MMFF94 energy minimization (`molrs::ff`).
//! 5. `StereoCheck`   — chiral-volume sign verification (no inversion).
//!
//! The maxIterations retry loop + `useRandomCoords` fallback live in `retry`.

mod embed4d;
mod etmin;
mod mmff_min;
mod retry;

use std::sync::OnceLock;

use rand::{SeedableRng, random, rngs::StdRng};

use crate::conformer::distgeom::{self, ChiralSign, DgConstraints, EtkdgVersion};
use crate::conformer::options::{ConformerOptions, ForceFieldKind};
use crate::conformer::report::{ConformerReport, ConformerStageReport, StageKind};
use molrs::error::MolRsError;
use molrs::ff::potential::intramolecular_pairs;
use molrs::ff::typifier::mmff::MMFF94Typifier;
use molrs::perceive::hydrogens::add_hydrogens;
use molrs::system::atomistic::Atomistic;

/// Embedding dimension for the first stage (RDKit ETKDG uses 4D).
const EMBED_DIM: usize = 4;
/// Fraction below which a chiral volume is treated as inverted (RDKit
/// `checkChiralCenters` 0.8 threshold).
const CHIRAL_RATIO_TOL: f64 = 0.8;

/// Run the ETKDGv3 embedding pipeline and return the molecule with 3D
/// coordinates plus a stage report.
///
/// `mol` is treated as a connectivity graph; any pre-existing 3D coordinates
/// are used only to seed chiral-volume signs (so a stereochemically-defined
/// input keeps its handedness) and are otherwise overwritten.
pub fn generate_3d_impl(
    mol: &Atomistic,
    opts: &ConformerOptions,
) -> Result<(Atomistic, ConformerReport), MolRsError> {
    if mol.n_atoms() == 0 {
        return Err(MolRsError::validation(
            "cannot generate 3D structure for empty molecule",
        ));
    }

    let mut report = ConformerReport::new(ForceFieldKind::MMFF94);

    let seed = opts.rng_seed.unwrap_or_else(random::<u64>);
    if opts.rng_seed.is_none() {
        report.warnings.push(format!(
            "rng_seed not provided; auto-generated seed={seed} for this run"
        ));
    }

    // --- Preprocess: hydrogen handling -----------------------------------
    let work = if opts.add_hydrogens {
        add_hydrogens(mol)
    } else {
        mol.clone()
    };
    let preprocess_steps = work.n_atoms().saturating_sub(mol.n_atoms());
    report.stages.push(ConformerStageReport {
        stage: StageKind::Preprocess,
        energy_before: None,
        energy_after: None,
        steps: preprocess_steps,
        converged: true,
        elapsed_ms: 0,
    });

    let n = work.n_atoms();

    // Single atom: place at origin (no geometry to solve). RDKit also returns
    // a valid (trivial) conformer.
    if n == 1 {
        let mut out = work;
        place_single_atom(&mut out)?;
        report.stages.push(ConformerStageReport {
            stage: StageKind::BuildInitial,
            energy_before: None,
            energy_after: None,
            steps: 1,
            converged: true,
            elapsed_ms: 0,
        });
        report.final_energy = Some(0.0);
        return Ok((out, report));
    }

    // --- Build ETKDGv3 constraints ---------------------------------------
    // Experimental torsions are assigned through the full CrystalFF
    // three-table set (v2 ++ small-rings ++ macrocycles) matched by the core
    // SMARTS engine (`molrs::perceive::smarts`), reproducing RDKit
    // `getExperimentalTorsions`. See `distgeom::torsion_prefs`.
    let version = EtkdgVersion::Etkdgv3;
    let constraints = distgeom::build_constraints(&work, version)?;

    // --- Retry loop ------------------------------------------------------
    let max_iters = retry::effective_max_iterations(opts.max_iterations_internal(), n);
    let mut best: Option<Vec<f64>> = None;
    let mut last_embed4d_steps = 0usize;
    let mut last_coarse_energy = f64::NAN;
    let mut last_coarse_steps = 0usize;
    let mut last_coarse_conv = false;
    let mut chiral_ok = false;

    for attempt in 0..max_iters {
        let aseed = retry::attempt_seed(seed, attempt);
        let mut rng = StdRng::seed_from_u64(aseed);
        let (coords3d, e4d_steps, coarse_e, coarse_steps, coarse_conv, chiral_pass) =
            try_embed(&constraints, n, &mut rng, false);
        last_embed4d_steps = e4d_steps;
        last_coarse_energy = coarse_e;
        last_coarse_steps = coarse_steps;
        last_coarse_conv = coarse_conv;
        if let Some(c) = coords3d {
            if chiral_pass {
                best = Some(c);
                chiral_ok = true;
                break;
            }
            // Keep a non-chiral-clean candidate as a fallback.
            if best.is_none() {
                best = Some(c);
            }
        }
    }

    // useRandomCoords fallback.
    if (best.is_none() || !chiral_ok) && opts.use_random_coords_fallback_internal() {
        let aseed = retry::attempt_seed(seed, max_iters + 1);
        let mut rng = StdRng::seed_from_u64(aseed);
        let (coords3d, e4d_steps, coarse_e, coarse_steps, coarse_conv, chiral_pass) =
            try_embed(&constraints, n, &mut rng, true);
        last_embed4d_steps = e4d_steps;
        last_coarse_energy = coarse_e;
        last_coarse_steps = coarse_steps;
        last_coarse_conv = coarse_conv;
        if let Some(c) = coords3d {
            if chiral_pass || best.is_none() {
                best = Some(c);
            }
            report
                .warnings
                .push("used random-coordinate fallback embedding".to_string());
        }
    }

    let mut coords3d = match best {
        Some(c) => c,
        None => {
            return Err(MolRsError::validation(
                "ETKDG embedding failed: no consistent conformer after retries",
            ));
        }
    };

    report.stages.push(ConformerStageReport {
        stage: StageKind::BuildInitial,
        energy_before: None,
        energy_after: None,
        steps: last_embed4d_steps,
        converged: true,
        elapsed_ms: 0,
    });
    report.stages.push(ConformerStageReport {
        stage: StageKind::CoarseOptimize,
        energy_before: None,
        energy_after: Some(last_coarse_energy),
        steps: last_coarse_steps,
        converged: last_coarse_conv,
        elapsed_ms: 0,
    });

    // --- Second-stage MMFF94 cleanup -------------------------------------
    let mut final_energy = None;
    if opts.mmff_cleanup_internal() {
        match mmff_cleanup(&work, &mut coords3d) {
            Ok((e, steps, conv)) => {
                final_energy = Some(e);
                report.stages.push(ConformerStageReport {
                    stage: StageKind::FinalOptimize,
                    energy_before: None,
                    energy_after: Some(e),
                    steps,
                    converged: conv,
                    elapsed_ms: 0,
                });
            }
            Err(msg) => {
                report
                    .warnings
                    .push(format!("MMFF94 cleanup skipped: {msg}"));
                report.stages.push(ConformerStageReport {
                    stage: StageKind::FinalOptimize,
                    energy_before: None,
                    energy_after: None,
                    steps: 0,
                    converged: false,
                    elapsed_ms: 0,
                });
            }
        }
    }

    // --- Stereo check ----------------------------------------------------
    let mut stereo_warnings = Vec::new();
    for c in &constraints.chiral {
        if c.sign == ChiralSign::Unknown {
            continue;
        }
        let vol = etmin::calc_chiral_volume(&coords3d, c.neighbors, 3);
        let target_positive = matches!(c.sign, ChiralSign::Positive);
        let got_positive = vol > 0.0;
        if target_positive != got_positive
            || (target_positive && c.volume_lower > 0.0 && vol < c.volume_lower * CHIRAL_RATIO_TOL)
            || (!target_positive && c.volume_upper < 0.0 && vol > c.volume_upper * CHIRAL_RATIO_TOL)
        {
            stereo_warnings.push(format!(
                "tetrahedral-inversion at atom {}: expected {:?} signed volume, got {:.3}",
                c.center, c.sign, vol
            ));
        }
    }
    let stereo_steps = stereo_warnings.len();
    report.warnings.extend(stereo_warnings);
    report.stages.push(ConformerStageReport {
        stage: StageKind::StereoCheck,
        energy_before: None,
        energy_after: None,
        steps: stereo_steps,
        converged: true,
        elapsed_ms: 0,
    });

    // --- Write coordinates back ------------------------------------------
    let mut out = work;
    write_coords(&mut out, &coords3d)?;
    report.final_energy = final_energy.or(Some(last_coarse_energy));
    Ok((out, report))
}

/// One embedding attempt: 4D embed → first-stage minimization → 3D
/// experimental-torsion refinement → chiral check.
///
/// Returns `(coords3d, embed4d_steps, coarse_energy, coarse_steps,
/// coarse_converged, chiral_pass)`. `coords3d` is `None` if the 4D embedding
/// was degenerate.
#[allow(clippy::type_complexity)]
fn try_embed<R: rand::Rng + ?Sized>(
    constraints: &DgConstraints,
    n: usize,
    rng: &mut R,
    use_random_coords: bool,
) -> (Option<Vec<f64>>, usize, f64, usize, bool, bool) {
    let bounds = &constraints.bounds;

    let mut coords4d = if use_random_coords {
        embed4d::compute_random_coords(n, EMBED_DIM, 10.0, rng)
    } else {
        let dist = embed4d::pick_random_dist_mat(bounds, rng);
        match embed4d::compute_initial_coords(&dist, n, EMBED_DIM, rng, true, 1) {
            Some(c) => c,
            None => return (None, 0, f64::NAN, 0, false, false),
        }
    };

    // First minimization: distance + chiral + 4th-dimension (RDKit
    // firstMinimization, weightChiral=1.0, weightFourthDim=0.1).
    let field1 = etmin::FirstStageField::build(bounds, &constraints.chiral, EMBED_DIM, 1.0, 0.1);
    let (e1, _c1, s1) = etmin::minimize(&mut coords4d, 400, 1e-3, |p, g| field1.energy_grad(p, g));
    // Reject obviously-bad first minimizations (RDKit github #971,
    // `MAX_MINIMIZED_E_PER_ATOM`). Random-coords fallback skips this gate.
    if !use_random_coords && e1 / (n as f64) >= etmin::MAX_MINIMIZED_E_PER_ATOM {
        return (None, s1, e1, 0, false, false);
    }

    // Fourth-dimension squeeze (RDKit minimizeFourthDimension, weightChiral=0.2,
    // weightFourthDim=1.0) to collapse 4D → 3D.
    let field1b = etmin::FirstStageField::build(bounds, &constraints.chiral, EMBED_DIM, 0.2, 1.0);
    let _ = etmin::minimize(&mut coords4d, 200, 1e-3, |p, g| field1b.energy_grad(p, g));

    // Project to 3D (drop the 4th component).
    let mut coords3d = vec![0.0; n * 3];
    for i in 0..n {
        for k in 0..3 {
            coords3d[i * 3 + k] = coords4d[i * EMBED_DIM + k];
        }
    }

    // Second stage: 3D experimental-torsion refinement (RDKit
    // minimizeWithExpTorsions / construct3DForceField).
    let field2 = etmin::ExpTorsionField::build(
        bounds,
        &constraints.experimental_torsions,
        &constraints.improper,
    );
    let (e2, c2, s2) = etmin::minimize(&mut coords3d, 300, 1e-3, |p, g| field2.energy_grad(p, g));

    // Chiral check.
    let mut chiral_pass = true;
    for c in &constraints.chiral {
        let vol = etmin::calc_chiral_volume(&coords3d, c.neighbors, 3);
        let lb = c.volume_lower;
        let ub = c.volume_upper;
        if (lb > 0.0 && vol < lb && (vol / lb < CHIRAL_RATIO_TOL || have_opposite_sign(vol, lb)))
            || (ub < 0.0
                && vol > ub
                && (vol / ub < CHIRAL_RATIO_TOL || have_opposite_sign(vol, ub)))
        {
            chiral_pass = false;
            break;
        }
    }

    (Some(coords3d), s1, e2, s2, c2, chiral_pass)
}

/// RDKit `haveOppositeSign`.
fn have_opposite_sign(a: f64, b: f64) -> bool {
    a.is_sign_negative() ^ b.is_sign_negative()
}

/// The MMFF94 typifier, constructed **once for the process**.
///
/// `MMFF94Typifier::new()` parses the embedded MMFF94 parameter set — hundreds of
/// KB of XML — on every call. It is stateless with respect to the molecule, so
/// constructing one per conformer would re-parse that XML on every `generate`,
/// which is pure waste in the one place the conformer pipeline is called in a
/// loop (`Conformer::generate` is invoked once per conformer). It is hoisted here
/// so the parse happens at most once, on the first cleanup that runs.
fn mmff94_typifier() -> &'static MMFF94Typifier {
    static TYPIFIER: OnceLock<MMFF94Typifier> = OnceLock::new();
    TYPIFIER.get_or_init(MMFF94Typifier::new)
}

/// MMFF94 second-stage cleanup minimization. Returns `(energy, steps,
/// converged)`. Errors (as a message) if the molecule has no MMFF typing.
///
/// Runs the standard route — typify → `Frame` → `ForceField::to_potentials` — the
/// same one every other force field in molrs goes through. (It used to call a
/// bespoke `MmffForceField` energy assembly, a second implementation of the seven
/// MMFF terms that `ff::potential::*::mmff` already provides; that layer is gone.)
fn mmff_cleanup(mol: &Atomistic, coords3d: &mut [f64]) -> Result<(f64, usize, bool), String> {
    // Write current coords so MMFF setup that consults geometry sees them.
    let mut staged = mol.clone();
    write_coords(&mut staged, coords3d).map_err(|e| e.to_string())?;

    let typifier = mmff94_typifier();
    let mut frame = typifier.typify(&staged)?.to_frame();
    // The neighbour list is the consumer's to build — and here the consumer is the
    // minimizer. Bonded terms and the 1-2/1-3 exclusions are topological, so this
    // list stays valid across the relaxation.
    frame.insert("pairs", intramolecular_pairs(&frame));
    let potentials = typifier.ff().to_potentials(&frame)?;

    // RDKit's MMFFOptimizeMolecule runs a full BFGS minimization to a
    // gradient-norm tolerance. Mirror that with L-BFGS to an RMS-gradient
    // convergence of 1e-3 kcal/mol/Å (matching RDKit's default
    // `MMFFOptimizeMolecule` grad tol) under a generous iteration cap, so the
    // freshly-embedded geometry is relaxed all the way to the MMFF minimum.
    let (e, _grad_rms, steps, conv) =
        mmff_min::minimize_lbfgs(coords3d, 1000, 1e-3, |p| potentials.calc_energy_forces(p));
    Ok((e, steps, conv))
}

/// Place a single-atom molecule at the origin.
fn place_single_atom(mol: &mut Atomistic) -> Result<(), MolRsError> {
    let id = mol.atoms().map(|(id, _)| id).next();
    if let Some(id) = id {
        mol.set_atom(id, "x", 0.0)?;
        mol.set_atom(id, "y", 0.0)?;
        mol.set_atom(id, "z", 0.0)?;
    }
    Ok(())
}

/// Write a flat `n*3` coordinate buffer back into the molecule (atom-iteration
/// order matches `distgeom`/`MMFF` indexing).
fn write_coords(mol: &mut Atomistic, coords: &[f64]) -> Result<(), MolRsError> {
    let ids: Vec<_> = mol.atoms().map(|(id, _)| id).collect();
    for (i, id) in ids.into_iter().enumerate() {
        mol.set_atom(id, "x", coords[i * 3])?;
        mol.set_atom(id, "y", coords[i * 3 + 1])?;
        mol.set_atom(id, "z", coords[i * 3 + 2])?;
    }
    Ok(())
}
