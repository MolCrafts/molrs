//! CL&Pol SAPT-derived Lennard-Jones scaling.

use std::collections::HashMap;
use std::fmt;

use super::forcefield::{ForceField, StyleDefs};

const C0: f64 = 0.254_952;
const C1: f64 = 0.106_906;
const SIGMA_SCALE: f64 = 0.985;

/// Scaling properties for one molecular fragment.
#[derive(Debug, Clone, PartialEq)]
pub struct FragmentScaling {
    pub name: String,
    pub q: f64,
    pub mu: f64,
    pub alpha: f64,
    pub polarizable: bool,
}

/// Atom data needed to assign force-field types and calculate a fragment COM.
#[derive(Debug, Clone, PartialEq)]
pub struct FragmentAtoms {
    pub name: String,
    pub atom_types: Vec<String>,
    pub coords: Vec<[f64; 3]>,
    pub masses: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ScaleLjError {
    InvalidAlpha(String),
    MissingFragment(String),
    Shape(String),
}

impl fmt::Display for ScaleLjError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidAlpha(name) => write!(f, "fragment '{name}' alpha must be positive"),
            Self::MissingFragment(name) => write!(f, "no scaling data for fragment '{name}'"),
            Self::Shape(name) => write!(
                f,
                "fragment '{name}' types, coordinates, and masses must have equal length"
            ),
        }
    }
}

impl std::error::Error for ScaleLjError {}

/// SAPT epsilon-scaling factor for a fragment pair at COM distance `r` (Å).
pub fn compute_k_ij(
    fr_i: &FragmentScaling,
    fr_j: &FragmentScaling,
    r: f64,
) -> Result<f64, ScaleLjError> {
    if fr_i.alpha <= 0.0 {
        return Err(ScaleLjError::InvalidAlpha(fr_i.name.clone()));
    }
    if fr_j.alpha <= 0.0 {
        return Err(ScaleLjError::InvalidAlpha(fr_j.name.clone()));
    }
    let mut denominator = 1.0;
    if !fr_i.polarizable {
        denominator += C0 * r * r * fr_j.q.powi(2) / fr_j.alpha + C1 * fr_j.mu.powi(2) / fr_j.alpha;
    }
    if !fr_j.polarizable {
        denominator += C0 * r * r * fr_i.q.powi(2) / fr_i.alpha + C1 * fr_i.mu.powi(2) / fr_i.alpha;
    }
    Ok(1.0 / denominator)
}

fn center_of_mass(fragment: &FragmentAtoms) -> Result<[f64; 3], ScaleLjError> {
    if fragment.atom_types.len() != fragment.coords.len()
        || fragment.coords.len() != fragment.masses.len()
    {
        return Err(ScaleLjError::Shape(fragment.name.clone()));
    }
    let mut total = 0.0;
    let mut center = [0.0; 3];
    for (coord, mass) in fragment.coords.iter().zip(&fragment.masses) {
        let weight = if *mass > 0.0 { *mass } else { 1.0 };
        total += weight;
        for d in 0..3 {
            center[d] += weight * coord[d];
        }
    }
    if total > 0.0 {
        for value in &mut center {
            *value /= total;
        }
    }
    Ok(center)
}

/// Clone and scale cross-fragment LJ pair parameters without mutating `ff`.
pub fn scale_lj(
    ff: &ForceField,
    fragments: &[FragmentAtoms],
    scaling: &HashMap<String, FragmentScaling>,
    scale_sigma: bool,
) -> Result<ForceField, ScaleLjError> {
    let mut type_to_fragment = HashMap::new();
    let mut centers = HashMap::new();
    for fragment in fragments {
        if !scaling.contains_key(&fragment.name) {
            return Err(ScaleLjError::MissingFragment(fragment.name.clone()));
        }
        centers.insert(fragment.name.clone(), center_of_mass(fragment)?);
        for atom_type in &fragment.atom_types {
            type_to_fragment.insert(atom_type.clone(), fragment.name.clone());
        }
    }

    let mut output = ff.clone();
    for style in output.styles_mut() {
        let StyleDefs::Pair(types) = &mut style.defs else {
            continue;
        };
        for pair in types {
            let (Some(fi), Some(fj)) = (
                type_to_fragment.get(&pair.itom),
                type_to_fragment.get(&pair.jtom),
            ) else {
                continue;
            };
            if fi == fj {
                continue;
            }
            let ci = centers[fi];
            let cj = centers[fj];
            let distance =
                ((ci[0] - cj[0]).powi(2) + (ci[1] - cj[1]).powi(2) + (ci[2] - cj[2]).powi(2))
                    .sqrt();
            let factor = compute_k_ij(&scaling[fi], &scaling[fj], distance)?;
            if let Some(epsilon) = pair.params.get("epsilon") {
                pair.params.set("epsilon", epsilon * factor);
            }
            if scale_sigma && let Some(sigma) = pair.params.get("sigma") {
                pair.params.set("sigma", sigma * SIGMA_SCALE);
            }
        }
    }
    Ok(output)
}

/// CL&Pol fragment parameters compiled from paduagroup/clandpol fragment.ff.
pub fn builtin_fragment_scaling() -> HashMap<String, FragmentScaling> {
    super::params::CLPOL_FRAGMENTS
        .iter()
        .copied()
        .map(|(name, q, mu, alpha, polarizable)| {
            (
                name.to_string(),
                FragmentScaling {
                    name: name.to_string(),
                    q,
                    mu,
                    alpha,
                    polarizable,
                },
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closed_form_keeps_mu_term_independent_of_distance() {
        let a = FragmentScaling {
            name: "a".into(),
            q: -1.0,
            mu: 0.0,
            alpha: 3.0,
            polarizable: false,
        };
        let b = FragmentScaling {
            name: "b".into(),
            q: 1.0,
            mu: 2.0,
            alpha: 8.0,
            polarizable: true,
        };
        let inv1 = 1.0 / compute_k_ij(&a, &b, 3.0).unwrap();
        let inv2 = 1.0 / compute_k_ij(&a, &b, 6.0).unwrap();
        let charge_only = C0 * (3.0_f64.powi(2) - 6.0_f64.powi(2)) / b.alpha;
        assert!((inv1 - inv2 - charge_only).abs() < 1e-12);
    }
}
