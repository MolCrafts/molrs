//! End-to-end orchestration of Embed stages.

use rand::{SeedableRng, random, rngs::StdRng};

use super::builder::embed_fragment_rules;
use super::distance_geometry::embed_distance_geometry;
use super::optimizer::{EnergyModel, lbfgs, steepest_descent};
use super::options::{EmbedAlgorithm, ForceFieldKind, EmbedOptions};
use super::report::{EmbedReport, StageKind, StageReport};
use super::rotor_search;
use super::stereo_guard;
use molrs::error::MolRsError;
use molrs::hydrogens::add_hydrogens;
use molrs::molgraph::MolGraph;

pub(crate) fn generate_3d_impl(
    mol: &MolGraph,
    opts: &EmbedOptions,
) -> Result<(MolGraph, EmbedReport), MolRsError> {
    if mol.n_atoms() == 0 {
        return Err(MolRsError::validation(
            "cannot generate 3D structure for empty molecule",
        ));
    }

    let mut preprocess_warnings = Vec::new();
    let forcefield_used = resolve_forcefield(opts.forcefield, &mut preprocess_warnings);
    let mut report = EmbedReport::new(opts.embed_algorithm, forcefield_used);
    report.warnings.extend(preprocess_warnings);

    let seed = opts.rng_seed.unwrap_or_else(random::<u64>);
    let mut rng = StdRng::seed_from_u64(seed);
    if opts.rng_seed.is_none() {
        report.warnings.push(format!(
            "rng_seed not provided; auto-generated seed={} for this run",
            seed
        ));
    }

    let mut work = if opts.add_hydrogens {
        add_hydrogens(mol)
    } else {
        mol.clone()
    };
    let preprocess_steps = if opts.add_hydrogens {
        work.n_atoms().saturating_sub(mol.n_atoms())
    } else {
        0
    };
    report.stages.push(StageReport {
        stage: StageKind::Preprocess,
        energy_before: None,
        energy_after: None,
        steps: preprocess_steps,
        converged: true,
        elapsed_ms: 0,
    });

    let stereo_before = stereo_guard::capture_if_3d(&work);

    let summary = match opts.embed_algorithm {
        EmbedAlgorithm::FragmentRules => embed_fragment_rules(&mut work, &mut rng)?,
        EmbedAlgorithm::DistanceGeometry => embed_distance_geometry(&mut work, &mut rng)?,
    };
    report.warnings.extend(summary.warnings);
    report.stages.push(StageReport {
        stage: StageKind::BuildInitial,
        energy_before: None,
        energy_after: None,
        steps: summary.placed_atoms,
        converged: true,
        elapsed_ms: 0,
    });

    let model = EnergyModel::from_mol(&work).with_chiral_constraints(&work);
    let mut coords = model.read_coords_from_mol(&work);

    let e0 = model.energy(&coords);
    let coarse = steepest_descent(&model, &mut coords, opts.coarse_steps(), 0.02, 1e-3);
    report.stages.push(StageReport {
        stage: StageKind::CoarseOptimize,
        energy_before: Some(e0),
        energy_after: Some(coarse.energy),
        steps: coarse.steps,
        converged: coarse.converged,
        elapsed_ms: 0,
    });

    let rotor = rotor_search::run(&work, &model, &mut coords, opts, &mut rng);
    report.stages.push(StageReport {
        stage: StageKind::RotorSearch,
        energy_before: Some(rotor.energy_before),
        energy_after: Some(rotor.energy_after),
        steps: rotor.attempts,
        converged: rotor.improved,
        elapsed_ms: 0,
    });

    let e1 = model.energy(&coords);
    let final_min = lbfgs(&model, &mut coords, opts.final_steps(), 5e-4);
    report.stages.push(StageReport {
        stage: StageKind::FinalOptimize,
        energy_before: Some(e1),
        energy_after: Some(final_min.energy),
        steps: final_min.steps,
        converged: final_min.converged,
        elapsed_ms: 0,
    });

    model.write_coords_to_mol(&mut work, &coords)?;
    report.final_energy = Some(final_min.energy);

    let stereo_after = stereo_guard::capture_if_3d(&work);
    let mut stereo_warnings =
        stereo_guard::compare_snapshots(stereo_before.as_ref(), stereo_after.as_ref());
    stereo_warnings.extend(stereo_guard::post_generation_warnings(&work));
    let stereo_steps = stereo_warnings.len();
    report.warnings.extend(stereo_warnings);
    report.stages.push(StageReport {
        stage: StageKind::StereoCheck,
        energy_before: None,
        energy_after: None,
        steps: stereo_steps,
        converged: true,
        elapsed_ms: 0,
    });

    Ok((work, report))
}

fn resolve_forcefield(requested: ForceFieldKind, warnings: &mut Vec<String>) -> ForceFieldKind {
    match requested {
        ForceFieldKind::Uff => ForceFieldKind::Uff,
        ForceFieldKind::MMFF94 => {
            warnings.push(
                "MMFF94 backend is not fully parameterized yet; falling back to UFF".to_string(),
            );
            ForceFieldKind::Uff
        }
        ForceFieldKind::Auto => {
            warnings.push(
                "Auto forcefield selected: MMFF94 unavailable for full coverage, using UFF"
                    .to_string(),
            );
            ForceFieldKind::Uff
        }
    }
}
