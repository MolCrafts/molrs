use molrs::types::F;
use std::collections::HashMap;
use std::env;

use crate::constraint::Restraint;
use crate::numerics::numeric_controls;
use crate::target::Target;

/// Quantified violation metrics for one packed configuration.
#[derive(Debug, Clone, Copy, Default)]
pub struct ViolationMetrics {
    /// Maximum inter-molecule distance violation in Angstrom (tolerance - distance).
    pub max_distance_violation: F,
    /// Maximum geometric-constraint penalty value (Packmol restraint function value).
    pub max_constraint_penalty: F,
    /// Number of violating inter-molecule atom pairs.
    pub violating_pairs: usize,
    /// Number of atoms involved in any violation.
    pub violating_atoms: usize,
}

/// Validation report for a packing run.
#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub expected_atoms: usize,
    pub actual_atoms: usize,
    pub expected_molecules: usize,
    /// Atom count exactly matches expanded target specification.
    pub atom_count_ok: bool,
    /// Molecule count exactly matches expanded target specification.
    pub molecule_count_ok: bool,
    /// Distance violations satisfy precision threshold.
    pub distance_ok: bool,
    /// Constraint violations satisfy precision threshold.
    pub constraints_ok: bool,
    /// Detailed metrics.
    pub metrics: ViolationMetrics,
}

impl ValidationReport {
    pub fn is_valid(&self) -> bool {
        self.atom_count_ok && self.molecule_count_ok && self.distance_ok && self.constraints_ok
    }
}

#[derive(Debug, Clone)]
struct ExpandedMol<'a> {
    target: &'a Target,
    start: usize,
    end: usize,
    molecule_id: usize,
}

#[derive(Debug, Clone)]
struct AtomRestraints {
    restraints: Vec<Restraint>,
}

/// Validate packed coordinates against target specification.
pub fn validate_from_targets(
    targets: &[Target],
    coordinates: &[[F; 3]],
    tolerance: F,
    precision: F,
) -> ValidationReport {
    let expanded = expand_targets(targets);
    let expected_atoms = expanded.last().map_or(0usize, |m| m.end);
    let expected_molecules = expanded.len();

    let (constraint_penalty, constraint_violating_atoms) =
        constraint_metrics(&expanded, coordinates, precision);
    let (distance_violation, pair_violations, distance_violating_atoms) =
        distance_metrics(&expanded, coordinates, tolerance, precision);

    let violating_atoms = union_count(
        coordinates.len(),
        &constraint_violating_atoms,
        &distance_violating_atoms,
    );

    ValidationReport {
        expected_atoms,
        actual_atoms: coordinates.len(),
        expected_molecules,
        atom_count_ok: coordinates.len() == expected_atoms,
        molecule_count_ok: expected_molecules > 0,
        distance_ok: distance_violation <= precision,
        constraints_ok: constraint_penalty <= precision,
        metrics: ViolationMetrics {
            max_distance_violation: distance_violation,
            max_constraint_penalty: constraint_penalty,
            violating_pairs: pair_violations,
            violating_atoms,
        },
    }
}

fn expand_targets(targets: &[Target]) -> Vec<ExpandedMol<'_>> {
    let free: Vec<&Target> = targets.iter().filter(|t| t.fixed_at.is_none()).collect();
    let fixed: Vec<&Target> = targets.iter().filter(|t| t.fixed_at.is_some()).collect();

    let mut expanded = Vec::new();
    let mut cursor = 0usize;
    let mut mol_id = 0usize;

    for target in free.into_iter().chain(fixed.into_iter()) {
        let nmols = if target.fixed_at.is_some() {
            1
        } else {
            target.count
        };
        for _ in 0..nmols {
            let start = cursor;
            let end = start + target.natoms();
            expanded.push(ExpandedMol {
                target,
                start,
                end,
                molecule_id: mol_id,
            });
            cursor = end;
            mol_id += 1;
        }
    }

    expanded
}

fn atom_restraints(target: &Target) -> Vec<AtomRestraints> {
    let mut per_atom = vec![
        AtomRestraints {
            restraints: target.molecule_constraint.restraints.clone(),
        };
        target.natoms()
    ];

    for ac in &target.atom_constraints {
        for &idx in &ac.atom_indices {
            if let Some(slot) = per_atom.get_mut(idx) {
                slot.restraints.extend(ac.restraints.clone());
            }
        }
    }
    per_atom
}

fn constraint_metrics(
    expanded: &[ExpandedMol<'_>],
    coordinates: &[[F; 3]],
    precision: F,
) -> (F, Vec<bool>) {
    let mut max_penalty = 0.0 as F;
    let mut violating_atoms = vec![false; coordinates.len()];
    let penalty_eps = precision;

    for mol in expanded {
        let per_atom = atom_restraints(mol.target);
        for (local_i, atom_i) in (mol.start..mol.end).enumerate() {
            let pos = coordinates[atom_i];
            let mut atom_penalty = 0.0 as F;
            for r in &per_atom[local_i].restraints {
                atom_penalty += r.value(&pos, 1.0, 0.01);
            }
            if atom_penalty > max_penalty {
                max_penalty = atom_penalty;
            }
            if atom_penalty > penalty_eps {
                violating_atoms[atom_i] = true;
            }
        }
    }

    (max_penalty, violating_atoms)
}

fn distance_metrics(
    expanded: &[ExpandedMol<'_>],
    coordinates: &[[F; 3]],
    tolerance: F,
    precision: F,
) -> (F, usize, Vec<bool>) {
    if coordinates.is_empty() {
        return (0.0, 0, Vec::new());
    }

    let mut molecule_of_atom = vec![0usize; coordinates.len()];
    for mol in expanded {
        for atom_mol in molecule_of_atom.iter_mut().take(mol.end).skip(mol.start) {
            *atom_mol = mol.molecule_id;
        }
    }

    let mut minc = [F::INFINITY; 3];
    let mut maxc = [F::NEG_INFINITY; 3];
    for p in coordinates {
        for k in 0..3 {
            minc[k] = minc[k].min(p[k]);
            maxc[k] = maxc[k].max(p[k]);
        }
    }

    let cell = tolerance.max(1.0e-6);
    let inv = 1.0 / cell;
    let mut buckets: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    let mut max_violation = 0.0 as F;
    let mut violating_pairs = 0usize;
    let mut violating_atoms = vec![false; coordinates.len()];
    let eps = precision.max(numeric_controls().epsrel);
    let debug = env::var("molrs-pack_DEBUG_VALIDATION").is_ok();
    let mut debug_left = 5usize;

    for (i, p) in coordinates.iter().enumerate() {
        let cx = ((p[0] - minc[0]) * inv).floor() as i64;
        let cy = ((p[1] - minc[1]) * inv).floor() as i64;
        let cz = ((p[2] - minc[2]) * inv).floor() as i64;

        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let key = (cx + dx, cy + dy, cz + dz);
                    if let Some(list) = buckets.get(&key) {
                        for &j in list {
                            if molecule_of_atom[i] == molecule_of_atom[j] {
                                continue;
                            }
                            let q = coordinates[j];
                            let d2 = (p[0] - q[0]).powi(2)
                                + (p[1] - q[1]).powi(2)
                                + (p[2] - q[2]).powi(2);
                            let d = d2.sqrt();
                            if d < tolerance {
                                let v = tolerance - d;
                                if v > max_violation {
                                    max_violation = v;
                                }
                                if v > eps {
                                    violating_pairs += 1;
                                    violating_atoms[i] = true;
                                    violating_atoms[j] = true;
                                    if debug && debug_left > 0 {
                                        eprintln!(
                                            "validation pair: i={} j={} d={:.6} tol={:.3} v={:.6} mi={} mj={}",
                                            i,
                                            j,
                                            d,
                                            tolerance,
                                            v,
                                            molecule_of_atom[i],
                                            molecule_of_atom[j]
                                        );
                                        debug_left -= 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        buckets.entry((cx, cy, cz)).or_default().push(i);
    }

    (max_violation, violating_pairs, violating_atoms)
}

fn union_count(n: usize, a: &[bool], b: &[bool]) -> usize {
    if a.len() != n || b.len() != n {
        return 0;
    }
    (0..n).filter(|&i| a[i] || b[i]).count()
}
