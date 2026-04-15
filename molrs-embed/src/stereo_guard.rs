//! Stereo sanity checks for Embed.

use std::collections::HashMap;

use molrs::molgraph::{AtomId, BondId, MolGraph};
use molrs::stereo::{
    BondStereo, TetrahedralStereo, assign_bond_stereo_from_3d, assign_stereo_from_3d,
    find_chiral_centers,
};

#[derive(Debug, Clone)]
pub(crate) struct StereoSnapshot {
    tetra: HashMap<AtomId, TetrahedralStereo>,
    bonds: HashMap<BondId, BondStereo>,
}

pub(crate) fn capture_if_3d(mol: &MolGraph) -> Option<StereoSnapshot> {
    let has_all_coords = mol.atoms().all(|(_, atom)| {
        atom.get_f64("x").is_some() && atom.get_f64("y").is_some() && atom.get_f64("z").is_some()
    });
    if !has_all_coords {
        return None;
    }

    Some(StereoSnapshot {
        tetra: assign_stereo_from_3d(mol),
        bonds: assign_bond_stereo_from_3d(mol),
    })
}

pub(crate) fn compare_snapshots(
    before: Option<&StereoSnapshot>,
    after: Option<&StereoSnapshot>,
) -> Vec<String> {
    let mut warnings = Vec::new();
    let (Some(before), Some(after)) = (before, after) else {
        return warnings;
    };

    for (id, prev) in &before.tetra {
        let next = after
            .tetra
            .get(id)
            .copied()
            .unwrap_or(TetrahedralStereo::Unspecified);
        if matches!(*prev, TetrahedralStereo::CW | TetrahedralStereo::CCW)
            && matches!(next, TetrahedralStereo::CW | TetrahedralStereo::CCW)
            && *prev != next
        {
            warnings.push(format!(
                "tetrahedral stereochemistry changed at atom {:?}: {:?} -> {:?}",
                id, prev, next
            ));
        }
    }

    for (id, prev) in &before.bonds {
        let next = after.bonds.get(id).copied().unwrap_or(BondStereo::None);
        if matches!(*prev, BondStereo::E | BondStereo::Z)
            && matches!(next, BondStereo::E | BondStereo::Z)
            && *prev != next
        {
            warnings.push(format!(
                "double-bond stereochemistry changed at bond {:?}: {:?} -> {:?}",
                id, prev, next
            ));
        }
    }

    warnings
}

pub(crate) fn post_generation_warnings(mol: &MolGraph) -> Vec<String> {
    let mut warnings = Vec::new();

    let chiral = find_chiral_centers(mol);
    if !chiral.is_empty() {
        let assigned = assign_stereo_from_3d(mol);
        for id in chiral {
            if matches!(
                assigned
                    .get(&id)
                    .copied()
                    .unwrap_or(TetrahedralStereo::Unspecified),
                TetrahedralStereo::Unspecified
            ) {
                warnings.push(format!(
                    "potential chiral center {:?} remains stereochemically unspecified",
                    id
                ));
            }
        }
    }

    let bond_stereo = assign_bond_stereo_from_3d(mol);
    for (bid, s) in bond_stereo {
        if s == BondStereo::Either {
            warnings.push(format!(
                "double bond {:?} has ambiguous E/Z assignment after generation",
                bid
            ));
        }
    }

    warnings
}
