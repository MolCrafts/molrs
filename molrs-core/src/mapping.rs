//! AA ↔ CG mapping: coarsen and backmap between [`Atomistic`] and
//! [`CoarseGrain`] molecular graphs.
//!
//! A [`CGMapping`] defines the correspondence between atom groups in the
//! all-atom model and beads in the coarse-grained model.
//!
//! # Examples
//!
//! ```
//! use molrs::atomistic::Atomistic;
//! use molrs::mapping::{CGMapping, WeightScheme};
//!
//! // Build a 3-atom molecule: O-H-H
//! let mut aa = Atomistic::new();
//! let o  = aa.add_atom_xyz("O", 0.0, 0.0, 0.0);
//! let h1 = aa.add_atom_xyz("H", 0.96, 0.0, 0.0);
//! let h2 = aa.add_atom_xyz("H",-0.24, 0.93, 0.0);
//! aa.add_bond(o, h1).unwrap();
//! aa.add_bond(o, h2).unwrap();
//!
//! // Map all 3 atoms → 1 bead of type "W"
//! let mapping = CGMapping::new(
//!     vec![0, 0, 0],
//!     vec!["W".to_string()],
//!     WeightScheme::GeometricCenter,
//! );
//! let cg = mapping.coarsen(&aa).unwrap();
//! assert_eq!(cg.n_atoms(), 1);
//! ```

use std::collections::HashMap;

use crate::atomistic::Atomistic;
use crate::coarsegrain::CoarseGrain;
use crate::error::MolRsError;
use crate::molgraph::AtomId;

/// How bead positions are computed from constituent atoms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightScheme {
    /// Center of mass (requires `"mass"` property on atoms).
    CenterOfMass,
    /// Geometric center (uniform weights).
    GeometricCenter,
}

/// AA ↔ CG mapping definition.
///
/// - `bead_mask[i]` = bead index that atom `i` maps to (0-based).
/// - `bead_types[j]` = bead type string for bead `j`.
/// - `templates[j]` = atomistic sub-graph for bead `j` (populated by `coarsen`).
#[derive(Debug, Clone)]
pub struct CGMapping {
    /// Per-atom bead assignment. Length must equal n_atoms of the AA model.
    pub bead_mask: Vec<usize>,
    /// Bead type names, indexed by bead index.
    pub bead_types: Vec<String>,
    /// Position calculation scheme.
    pub weight_scheme: WeightScheme,
    /// Atomistic templates per bead (populated during coarsen, used for backmap).
    templates: Vec<Atomistic>,
}

impl CGMapping {
    /// Create a new mapping definition.
    ///
    /// - `bead_mask`: atom index → bead index. Bead indices must be contiguous
    ///   starting from 0.
    /// - `bead_types`: one type name per bead.
    /// - `weight_scheme`: how to compute bead positions.
    pub fn new(
        bead_mask: Vec<usize>,
        bead_types: Vec<String>,
        weight_scheme: WeightScheme,
    ) -> Self {
        Self {
            bead_mask,
            bead_types,
            weight_scheme,
            templates: Vec::new(),
        }
    }

    /// Forward mapping: [`Atomistic`] → [`CoarseGrain`].
    ///
    /// 1. Groups atoms by `bead_mask`.
    /// 2. Computes bead positions (COM or geometric center).
    /// 3. Extracts atomistic sub-graphs as templates for backmapping.
    /// 4. Infers CG bonds from cross-boundary AA bonds.
    pub fn coarsen(&self, aa: &Atomistic) -> Result<CoarseGrain, MolRsError> {
        // Validate mask length.
        if self.bead_mask.len() != aa.n_atoms() {
            return Err(MolRsError::validation(format!(
                "bead_mask length ({}) != n_atoms ({})",
                self.bead_mask.len(),
                aa.n_atoms()
            )));
        }

        let n_beads = self.bead_types.len();

        // Validate mask values.
        for (i, &bead_idx) in self.bead_mask.iter().enumerate() {
            if bead_idx >= n_beads {
                return Err(MolRsError::validation(format!(
                    "bead_mask[{}] = {} out of range (n_beads = {})",
                    i, bead_idx, n_beads
                )));
            }
        }

        // Collect atom IDs in insertion order.
        let atom_ids: Vec<AtomId> = aa.atoms().map(|(id, _)| id).collect();

        // Group atom indices by bead.
        let mut bead_groups: Vec<Vec<usize>> = vec![Vec::new(); n_beads];
        for (atom_idx, &bead_idx) in self.bead_mask.iter().enumerate() {
            bead_groups[bead_idx].push(atom_idx);
        }

        // Compute bead positions and build templates.
        let mut cg = CoarseGrain::new();
        let mut bead_ids: Vec<AtomId> = Vec::with_capacity(n_beads);

        for (bead_idx, group) in bead_groups.iter().enumerate() {
            if group.is_empty() {
                return Err(MolRsError::validation(format!(
                    "bead {} has no atoms assigned",
                    bead_idx
                )));
            }

            let (cx, cy, cz) = self.compute_center(aa, &atom_ids, group)?;
            let bead_id = cg.add_bead(&self.bead_types[bead_idx], cx, cy, cz);
            bead_ids.push(bead_id);
        }

        // Infer CG bonds: AA bonds that cross bead boundaries.
        let mut cg_bond_set: Vec<(usize, usize)> = Vec::new();
        for (_, bond) in aa.bonds() {
            let a_idx = atom_ids.iter().position(|&id| id == bond.atoms[0]).unwrap();
            let b_idx = atom_ids.iter().position(|&id| id == bond.atoms[1]).unwrap();
            let ba = self.bead_mask[a_idx];
            let bb = self.bead_mask[b_idx];
            if ba != bb {
                let pair = if ba < bb { (ba, bb) } else { (bb, ba) };
                if !cg_bond_set.contains(&pair) {
                    cg_bond_set.push(pair);
                }
            }
        }
        for (ba, bb) in &cg_bond_set {
            cg.add_bond(bead_ids[*ba], bead_ids[*bb])?;
        }

        Ok(cg)
    }

    /// Coarsen and store templates for later backmapping.
    ///
    /// Same as [`coarsen`](Self::coarsen) but also saves per-bead atomistic
    /// sub-graphs into `self.templates`.
    pub fn coarsen_with_templates(&mut self, aa: &Atomistic) -> Result<CoarseGrain, MolRsError> {
        let cg = self.coarsen(aa)?;
        self.build_templates(aa)?;
        Ok(cg)
    }

    /// Backmapping: [`CoarseGrain`] → [`Atomistic`].
    ///
    /// For each bead, places the corresponding atomistic template centered at
    /// the bead position. Cross-bead bonds are restored.
    ///
    /// Requires templates to have been populated via
    /// [`coarsen_with_templates`](Self::coarsen_with_templates).
    pub fn backmap(&self, cg: &CoarseGrain) -> Result<Atomistic, MolRsError> {
        if self.templates.is_empty() {
            return Err(MolRsError::validation(
                "no templates available; call coarsen_with_templates first",
            ));
        }
        if cg.n_atoms() != self.templates.len() {
            return Err(MolRsError::validation(format!(
                "CG has {} beads but mapping has {} templates",
                cg.n_atoms(),
                self.templates.len()
            )));
        }

        let bead_ids: Vec<AtomId> = cg.atoms().map(|(id, _)| id).collect();
        let mut result = Atomistic::new();

        // Per-bead: translate template to bead position, merge into result.
        for (bead_idx, &bead_id) in bead_ids.iter().enumerate() {
            let bead = cg.get_atom(bead_id)?;
            let bx = bead.get_f64("x").unwrap_or(0.0);
            let by = bead.get_f64("y").unwrap_or(0.0);
            let bz = bead.get_f64("z").unwrap_or(0.0);

            let template = &self.templates[bead_idx];
            let (tcx, tcy, tcz) = geometric_center(template);

            // Clone template and translate to bead position.
            let mut fragment = template.clone();
            let dx = bx - tcx;
            let dy = by - tcy;
            let dz = bz - tcz;
            fragment.translate([dx, dy, dz]);

            result.merge(fragment.into_inner());
        }

        Ok(result)
    }

    // -- private helpers --

    fn compute_center(
        &self,
        aa: &Atomistic,
        atom_ids: &[AtomId],
        group: &[usize],
    ) -> Result<(f64, f64, f64), MolRsError> {
        match self.weight_scheme {
            WeightScheme::GeometricCenter => {
                let mut sx = 0.0;
                let mut sy = 0.0;
                let mut sz = 0.0;
                let n = group.len() as f64;
                for &atom_idx in group {
                    let atom = aa.get_atom(atom_ids[atom_idx])?;
                    sx += atom.get_f64("x").unwrap_or(0.0);
                    sy += atom.get_f64("y").unwrap_or(0.0);
                    sz += atom.get_f64("z").unwrap_or(0.0);
                }
                Ok((sx / n, sy / n, sz / n))
            }
            WeightScheme::CenterOfMass => {
                let mut sx = 0.0;
                let mut sy = 0.0;
                let mut sz = 0.0;
                let mut total_mass = 0.0;
                for &atom_idx in group {
                    let atom = aa.get_atom(atom_ids[atom_idx])?;
                    let mass = atom.get_f64("mass").unwrap_or_else(|| {
                        // Fall back to element mass from symbol.
                        atom.get_str("element")
                            .and_then(crate::element::Element::by_symbol)
                            .map(|e| e.atomic_mass() as f64)
                            .unwrap_or(1.0)
                    });
                    let x = atom.get_f64("x").unwrap_or(0.0);
                    let y = atom.get_f64("y").unwrap_or(0.0);
                    let z = atom.get_f64("z").unwrap_or(0.0);
                    sx += mass * x;
                    sy += mass * y;
                    sz += mass * z;
                    total_mass += mass;
                }
                if total_mass.abs() < 1e-15 {
                    return Err(MolRsError::validation("total mass is zero"));
                }
                Ok((sx / total_mass, sy / total_mass, sz / total_mass))
            }
        }
    }

    fn build_templates(&mut self, aa: &Atomistic) -> Result<(), MolRsError> {
        let atom_ids: Vec<AtomId> = aa.atoms().map(|(id, _)| id).collect();
        let n_beads = self.bead_types.len();

        let mut bead_groups: Vec<Vec<usize>> = vec![Vec::new(); n_beads];
        for (atom_idx, &bead_idx) in self.bead_mask.iter().enumerate() {
            bead_groups[bead_idx].push(atom_idx);
        }

        self.templates.clear();
        for group in &bead_groups {
            let mut template = Atomistic::new();
            let mut id_map: HashMap<AtomId, AtomId> = HashMap::new();

            // Copy atoms in this group.
            for &atom_idx in group {
                let old_id = atom_ids[atom_idx];
                let atom = aa.get_atom(old_id)?.clone();
                let new_id = template.as_molgraph_mut().add_atom(atom);
                id_map.insert(old_id, new_id);
            }

            // Copy intra-group bonds.
            for (_, bond) in aa.bonds() {
                if let (Some(&new_a), Some(&new_b)) =
                    (id_map.get(&bond.atoms[0]), id_map.get(&bond.atoms[1]))
                {
                    let bid = template.add_bond(new_a, new_b)?;
                    // Copy bond properties (e.g. order).
                    for (k, v) in &bond.props {
                        template
                            .get_bond_mut(bid)?
                            .props
                            .insert(k.clone(), v.clone());
                    }
                }
            }

            self.templates.push(template);
        }

        Ok(())
    }
}

/// Geometric center of all atoms with x/y/z coords.
fn geometric_center(mol: &Atomistic) -> (f64, f64, f64) {
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut sz = 0.0;
    let mut n = 0.0;
    for (_, atom) in mol.atoms() {
        sx += atom.get_f64("x").unwrap_or(0.0);
        sy += atom.get_f64("y").unwrap_or(0.0);
        sz += atom.get_f64("z").unwrap_or(0.0);
        n += 1.0;
    }
    if n < 1e-15 {
        return (0.0, 0.0, 0.0);
    }
    (sx / n, sy / n, sz / n)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_water() -> Atomistic {
        let mut aa = Atomistic::new();
        let o = aa.add_atom_xyz("O", 0.0, 0.0, 0.0);
        let h1 = aa.add_atom_xyz("H", 0.96, 0.0, 0.0);
        let h2 = aa.add_atom_xyz("H", -0.24, 0.93, 0.0);
        aa.add_bond(o, h1).unwrap();
        aa.add_bond(o, h2).unwrap();
        aa
    }

    #[test]
    fn test_coarsen_single_bead() {
        let aa = build_water();
        let mapping = CGMapping::new(
            vec![0, 0, 0],
            vec!["W".to_string()],
            WeightScheme::GeometricCenter,
        );
        let cg = mapping.coarsen(&aa).unwrap();

        assert_eq!(cg.n_atoms(), 1);
        assert_eq!(cg.n_bonds(), 0);

        let bead = cg.atoms().next().unwrap().1;
        assert_eq!(bead.get_str("bead_type"), Some("W"));

        // Geometric center of (0,0,0), (0.96,0,0), (-0.24,0.93,0)
        let cx = bead.get_f64("x").unwrap();
        let cy = bead.get_f64("y").unwrap();
        assert!((cx - 0.24).abs() < 1e-10);
        assert!((cy - 0.31).abs() < 1e-10);
    }

    #[test]
    fn test_coarsen_two_beads_with_bond() {
        let mut aa = Atomistic::new();
        let c1 = aa.add_atom_xyz("C", 0.0, 0.0, 0.0);
        let c2 = aa.add_atom_xyz("C", 1.5, 0.0, 0.0);
        let h1 = aa.add_atom_xyz("H", -1.0, 0.0, 0.0);
        let h2 = aa.add_atom_xyz("H", 2.5, 0.0, 0.0);
        aa.add_bond(c1, c2).unwrap(); // cross-bead bond
        aa.add_bond(c1, h1).unwrap();
        aa.add_bond(c2, h2).unwrap();

        // bead 0: C1+H1, bead 1: C2+H2
        let mapping = CGMapping::new(
            vec![0, 1, 0, 1],
            vec!["CH".to_string(), "CH".to_string()],
            WeightScheme::GeometricCenter,
        );
        let cg = mapping.coarsen(&aa).unwrap();

        assert_eq!(cg.n_atoms(), 2);
        assert_eq!(cg.n_bonds(), 1); // C1-C2 bond crosses boundary
    }

    #[test]
    fn test_coarsen_mask_length_mismatch() {
        let aa = build_water();
        let mapping = CGMapping::new(
            vec![0, 0], // wrong length
            vec!["W".to_string()],
            WeightScheme::GeometricCenter,
        );
        assert!(mapping.coarsen(&aa).is_err());
    }

    #[test]
    fn test_round_trip() {
        let aa = build_water();
        let mut mapping = CGMapping::new(
            vec![0, 0, 0],
            vec!["W".to_string()],
            WeightScheme::GeometricCenter,
        );
        let cg = mapping.coarsen_with_templates(&aa).unwrap();
        assert_eq!(cg.n_atoms(), 1);

        let reconstructed = mapping.backmap(&cg).unwrap();
        assert_eq!(reconstructed.n_atoms(), 3);
        assert_eq!(reconstructed.n_bonds(), 2);
    }

    #[test]
    fn test_backmap_without_templates() {
        let mut cg = CoarseGrain::new();
        cg.add_bead("W", 0.0, 0.0, 0.0);

        let mapping = CGMapping::new(
            vec![0],
            vec!["W".to_string()],
            WeightScheme::GeometricCenter,
        );
        // No templates → error
        assert!(mapping.backmap(&cg).is_err());
    }

    #[test]
    fn test_center_of_mass() {
        let mut aa = Atomistic::new();
        // O is ~16x heavier than H
        let o = aa.add_atom_xyz("O", 0.0, 0.0, 0.0);
        let h1 = aa.add_atom_xyz("H", 0.96, 0.0, 0.0);
        let h2 = aa.add_atom_xyz("H", -0.24, 0.93, 0.0);
        aa.add_bond(o, h1).unwrap();
        aa.add_bond(o, h2).unwrap();

        let mapping = CGMapping::new(
            vec![0, 0, 0],
            vec!["W".to_string()],
            WeightScheme::CenterOfMass,
        );
        let cg = mapping.coarsen(&aa).unwrap();
        let bead = cg.atoms().next().unwrap().1;

        // COM should be much closer to O (0,0,0) than geometric center
        let cx = bead.get_f64("x").unwrap();
        assert!(cx.abs() < 0.15); // closer to 0 than the geometric center (0.24)
    }
}
