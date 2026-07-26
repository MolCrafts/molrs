//! Shared topology snapshot for the MMFF typer / charger.
//!
//! Ported (data-flow only) from the per-atom queries RDKit's
//! `AtomTyper.cpp` makes on an `ROMol`: `getAtomicNum`, `getDegree`,
//! `getTotalDegree`, `getFormalCharge`, `getBondBetweenAtoms`,
//! `getValence(EXPLICIT) + getNumImplicitHs()`, plus the derived
//! Kekulé/SSSR/aromaticity/hybridization flags.
//!
//! Ported from RDKit `Code/GraphMol/ForceFieldHelpers/MMFF/AtomTyper.cpp`
//! and `Code/GraphMol/ConjugHybrid.cpp` (BSD-3, RDKit contributors).
//!
//! This is *data only*: an immutable snapshot built once from a
//! [`MolGraph`]. All higher layers (`aromaticity`, `atomtype`,
//! `charges`) read from it and never mutate the graph.

use molrs::Element;
use molrs::perceive::rings::{RingInfo, find_rings};
use molrs::system::molgraph::PropValue;
use molrs::{AtomId, Atomistic};

/// Bond order (Kekulé): we treat the SDF integer order verbatim.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BondOrder {
    Single,
    Double,
    Triple,
    /// Set only on ring bonds after MMFF aromaticity perception.
    Aromatic,
}

impl BondOrder {
    fn from_order(o: f64) -> Self {
        // RDKit SDF reader stores integer orders; aromatic SDFs are
        // kekulized before writing, so we only see 1/2/3 here.
        if o >= 2.5 {
            BondOrder::Triple
        } else if o >= 1.5 {
            BondOrder::Double
        } else {
            BondOrder::Single
        }
    }
}

/// Immutable per-atom / per-bond snapshot used by all MMFF passes.
#[derive(Debug, Clone)]
pub struct Topo {
    /// Atom ids in stable iteration order (this defines the public index).
    pub atom_ids: Vec<AtomId>,
    /// atom id -> dense index into `atom_ids`.
    idx_of: std::collections::HashMap<AtomId, usize>,
    /// atomic number per atom index.
    pub atno: Vec<u8>,
    /// formal charge per atom index (from `"formal_charge"` prop, default 0).
    pub formal_charge: Vec<i32>,
    /// neighbor atom indices per atom index.
    pub nbrs: Vec<Vec<usize>>,
    /// bond order to each neighbor (parallel to `nbrs`); ring bonds become
    /// `Aromatic` after aromaticity perception.
    pub nbr_order: Vec<Vec<BondOrder>>,
    /// kekulé bond order to each neighbor (parallel to `nbrs`); never rewritten
    /// by aromaticity, so total-bond-order tests stay correct.
    pub nbr_kekule: Vec<Vec<BondOrder>>,
    /// SSSR ring info.
    pub rings: RingInfo,
    /// rings expressed as dense atom indices (parallel to `rings.rings()`).
    pub ring_idx: Vec<Vec<usize>>,
    /// MMFF aromatic flag per atom (set by `aromaticity::set_mmff_aromaticity`).
    pub is_aromatic: Vec<bool>,
    /// MMFF aromatic flag per ring (parallel to `ring_idx`).
    pub ring_aromatic: Vec<bool>,
}

impl Topo {
    /// Build the snapshot from an [`Atomistic`] molecule.
    ///
    /// Returns `Err(symbol)` for an atom whose element symbol is unknown.
    pub fn build(mol: &Atomistic) -> Result<Self, String> {
        let atom_ids: Vec<AtomId> = mol.atoms().map(|(id, _)| id).collect();
        let idx_of: std::collections::HashMap<AtomId, usize> = atom_ids
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i))
            .collect();

        let n = atom_ids.len();
        let mut atno = vec![0u8; n];
        let mut formal_charge = vec![0i32; n];
        let mut nbrs = vec![Vec::new(); n];
        let mut nbr_order = vec![Vec::new(); n];
        let mut nbr_kekule = vec![Vec::new(); n];

        for (i, &id) in atom_ids.iter().enumerate() {
            let atom = mol.get_atom(id).map_err(|e| e.to_string())?;
            let sym = atom.get_str("element").unwrap_or("");
            let el = Element::by_symbol(sym).ok_or_else(|| sym.to_string())?;
            atno[i] = el.z();
            formal_charge[i] = match atom.get("formal_charge") {
                Some(PropValue::F64(v)) => v.round() as i32,
                Some(PropValue::Int(v)) => *v,
                _ => 0,
            };
            for (nbr_id, order) in mol.neighbor_bonds(id) {
                let j = idx_of[&nbr_id];
                let bo = BondOrder::from_order(order);
                nbrs[i].push(j);
                nbr_order[i].push(bo);
                nbr_kekule[i].push(bo);
            }
        }

        let rings = find_rings(mol);
        let ring_idx: Vec<Vec<usize>> = rings
            .rings()
            .iter()
            .map(|r| r.iter().map(|id| idx_of[id]).collect())
            .collect();
        let ring_aromatic = vec![false; ring_idx.len()];

        Ok(Self {
            atom_ids,
            idx_of,
            atno,
            formal_charge,
            nbrs,
            nbr_order,
            nbr_kekule,
            rings,
            ring_idx,
            is_aromatic: vec![false; n],
            ring_aromatic,
        })
    }

    /// Number of atoms.
    pub fn n_atoms(&self) -> usize {
        self.atom_ids.len()
    }

    /// `AtomId` for a dense atom index.
    pub fn id(&self, i: usize) -> AtomId {
        self.atom_ids[i]
    }

    /// RDKit `getDegree()`: number of explicit (heavy + H) neighbors.
    ///
    /// In our pipeline all hydrogens are explicit, so this equals
    /// `getTotalDegree()` as well.
    pub fn degree(&self, i: usize) -> usize {
        self.nbrs[i].len()
    }

    /// RDKit `getTotalDegree()`. With explicit Hs this is `degree`.
    pub fn total_degree(&self, i: usize) -> usize {
        self.degree(i)
    }

    /// Bond order between atoms `i` and `j` if bonded.
    pub fn bond_order(&self, i: usize, j: usize) -> Option<BondOrder> {
        self.nbrs[i]
            .iter()
            .position(|&k| k == j)
            .map(|p| self.nbr_order[i][p])
    }

    /// RDKit `getValence(EXPLICIT) + getNumImplicitHs()` — the total bond
    /// order around the atom. With explicit Hs, `getNumImplicitHs() == 0`,
    /// so this is simply the integer sum of Kekulé bond orders.
    pub fn total_bond_order(&self, i: usize) -> u32 {
        self.nbr_kekule[i]
            .iter()
            .map(|o| match o {
                BondOrder::Single | BondOrder::Aromatic => 1,
                BondOrder::Double => 2,
                BondOrder::Triple => 3,
            })
            .sum()
    }

    /// Number of hydrogen neighbors.
    pub fn n_h_neighbors(&self, i: usize) -> usize {
        self.nbrs[i].iter().filter(|&&j| self.atno[j] == 1).count()
    }

    /// Is atom `i` in a ring of exactly `size` atoms?
    pub fn is_atom_in_ring_of_size(&self, i: usize, size: usize) -> bool {
        self.rings
            .rings_of_size(size)
            .iter()
            .any(|r| r.iter().any(|&id| self.idx_of[&id] == i))
    }

    /// Are all the listed atoms in the same ring of exactly `size`?
    pub fn atoms_in_same_ring_of_size(&self, size: usize, atoms: &[usize]) -> bool {
        self.ring_idx
            .iter()
            .filter(|r| r.len() == size)
            .any(|r| atoms.iter().all(|a| r.contains(a)))
    }

    /// Is atom `i` in an *aromatic* ring of exactly `size` atoms?
    /// (RDKit `RingMembershipSize::isAtomInAromaticRingOfSize`.)
    pub fn is_atom_in_aromatic_ring_of_size(&self, i: usize, size: usize) -> bool {
        if !self.is_aromatic[i] {
            return false;
        }
        self.ring_idx
            .iter()
            .zip(self.ring_aromatic.iter())
            .any(|(r, &arom)| arom && r.len() == size && r.contains(&i))
    }
}
