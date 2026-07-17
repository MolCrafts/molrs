//! Chiral + improper (out-of-plane) constraint assembly.
//!
//! Ported from RDKit (BSD-3, Copyright (C) 2004-2025 Greg Landrum and other
//! RDKit contributors):
//!   * chiral sets: `EmbeddingOps::findChiralSets`
//!     (`$RDBASE/Code/GraphMol/DistGeomHelpers/Embedder.cpp`),
//!   * improper / out-of-plane "basic knowledge" terms:
//!     `getExperimentalTorsions`
//!     (`$RDBASE/Code/GraphMol/ForceFieldHelpers/CrystalFF/TorsionPreferences.cpp`).
//!
//! `molrs::MolGraph` does not carry an RDKit-style `ChiralTag`. For a
//! stereocentre we therefore take the **sign of the signed tetrahedral volume
//! of the supplied 3D coordinates** as the chirality reference: this is what
//! actually distinguishes R from S, and is the quantity the downstream
//! embedder constrains. When the molecule has no coordinates the sign is left
//! `Unknown` and only the volume magnitude bounds are emitted.

use molrs::system::atomistic::{AtomId, Atomistic};

use super::mol_features::{Hybridization, Perceived};

/// Sign of a chiral constraint's signed tetrahedral volume.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChiralSign {
    /// Positive signed volume (RDKit CCW center, `[+5, +100]`).
    Positive,
    /// Negative signed volume (RDKit CW center, `[-100, -5]`).
    Negative,
    /// No 3D reference available — magnitude-only.
    Unknown,
}

/// A chiral-volume constraint on a centre and its four (ordered) neighbours.
#[derive(Clone, Debug)]
pub struct ChiralConstraint {
    /// Centre atom index.
    pub center: usize,
    /// Ordered neighbour indices (length 4; the centre is repeated when the
    /// centre has only three heavy neighbours, mirroring RDKit).
    pub neighbors: [usize; 4],
    /// Lower volume bound (signed).
    pub volume_lower: f64,
    /// Upper volume bound (signed).
    pub volume_upper: f64,
    /// Sign derived from 3D coordinates (or `Unknown`).
    pub sign: ChiralSign,
}

/// An improper (out-of-plane) constraint for a planar sp2 centre with three
/// neighbours (RDKit "basic knowledge" inversion term).
#[derive(Clone, Debug)]
pub struct ImproperConstraint {
    /// `[n0, center, n2, n3]` atom indices.
    pub atoms: [usize; 4],
    /// Atomic number of the central atom.
    pub center_atomic_num: u8,
    /// `true` when the sp2 carbon is bound to an sp2 oxygen.
    pub bound_to_sp2_o: bool,
}

fn coord(mol: &Atomistic, id: AtomId) -> Option<[f64; 3]> {
    let a = mol.get_atom(id).ok()?;
    Some([a.get_f64("x")?, a.get_f64("y")?, a.get_f64("z")?])
}

/// Signed tetrahedral volume of `(p0-p3)·((p1-p3)×(p2-p3))`.
fn signed_volume(p0: [f64; 3], p1: [f64; 3], p2: [f64; 3], p3: [f64; 3]) -> f64 {
    let a = [p0[0] - p3[0], p0[1] - p3[1], p0[2] - p3[2]];
    let b = [p1[0] - p3[0], p1[1] - p3[1], p1[2] - p3[2]];
    let c = [p2[0] - p3[0], p2[1] - p3[1], p2[2] - p3[2]];
    let cross = [
        b[1] * c[2] - b[2] * c[1],
        b[2] * c[0] - b[0] * c[2],
        b[0] * c[1] - b[1] * c[0],
    ];
    a[0] * cross[0] + a[1] * cross[1] + a[2] * cross[2]
}

/// Build chiral constraints (RDKit `findChiralSets`, restricted to tetrahedral
/// C/N centres) using the input 3D coordinates to fix the volume sign.
pub fn build_chiral(mol: &Atomistic, p: &Perceived) -> Vec<ChiralConstraint> {
    let mut out = Vec::new();
    for (i, atom) in p.atoms.iter().enumerate() {
        let z = atom.element.z();
        if z == 1 {
            continue;
        }
        // RDKit treats degree-4 C/N as potential stereocentres.
        let is_candidate = (z == 6 || z == 7) && atom.degree == 4;
        if !is_candidate {
            continue;
        }
        let mut nbrs: Vec<usize> = p.adj[i].clone();
        if nbrs.len() < 3 {
            continue;
        }
        let mut vol_lower = 5.0;
        let vol_upper = 100.0;
        if nbrs.len() < 4 {
            vol_lower = 2.0;
            nbrs.push(i);
        }
        let neighbors = [nbrs[0], nbrs[1], nbrs[2], nbrs[3]];

        // Sign from 3D coordinates if available.
        let sign = match (
            coord(mol, p.atom_ids[neighbors[0]]),
            coord(mol, p.atom_ids[neighbors[1]]),
            coord(mol, p.atom_ids[neighbors[2]]),
            coord(mol, p.atom_ids[neighbors[3]]),
        ) {
            (Some(p0), Some(p1), Some(p2), Some(p3)) => {
                let v = signed_volume(p0, p1, p2, p3);
                if v > 1e-6 {
                    ChiralSign::Positive
                } else if v < -1e-6 {
                    ChiralSign::Negative
                } else {
                    ChiralSign::Unknown
                }
            }
            _ => ChiralSign::Unknown,
        };

        let (lower, upper) = match sign {
            ChiralSign::Positive => (vol_lower, vol_upper),
            ChiralSign::Negative => (-vol_upper, -vol_lower),
            ChiralSign::Unknown => (vol_lower, vol_upper),
        };

        out.push(ChiralConstraint {
            center: i,
            neighbors,
            volume_lower: lower,
            volume_upper: upper,
            sign,
        });
    }
    out
}

/// Build improper (out-of-plane) constraints for sp2 C/N/O centres with three
/// neighbours (RDKit basic-knowledge inversion terms).
pub fn build_improper(p: &Perceived) -> Vec<ImproperConstraint> {
    let mut out = Vec::new();
    for (i, atom) in p.atoms.iter().enumerate() {
        let z = atom.element.z();
        if !(z == 6 || z == 7 || z == 8) {
            continue;
        }
        if atom.hybridization != Hybridization::Sp2 || atom.degree != 3 {
            continue;
        }
        let nbrs = &p.adj[i];
        if nbrs.len() != 3 {
            continue;
        }
        let mut bound_to_sp2_o = false;
        for &nb in nbrs {
            if z == 6
                && p.atoms[nb].element.z() == 8
                && p.atoms[nb].hybridization == Hybridization::Sp2
            {
                bound_to_sp2_o = true;
            }
        }
        // RDKit packs as [n0, center, n2, n3]: position 1 is the centre.
        out.push(ImproperConstraint {
            atoms: [nbrs[0], i, nbrs[1], nbrs[2]],
            center_atomic_num: z,
            bound_to_sp2_o,
        });
    }
    out
}
