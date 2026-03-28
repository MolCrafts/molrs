//! Dynamic molecular graph for editing-oriented CRUD operations.
//!
//! [`MolGraph`] stores atoms (or beads), bonds, angles, and dihedrals in
//! generational arenas ([`slotmap::SlotMap`]), providing O(1) insert / remove /
//! lookup with stable handles that survive mutations.
//!
//! Every entity is a property bag ([`Atom`]): coordinates live as `"x"`, `"y"`,
//! `"z"` keys, matching the Python `Entity(UserDict)` convention.
//!
//! # Examples
//!
//! ```
//! use molrs::molgraph::{Atom, MolGraph};
//!
//! let mut g = MolGraph::new();
//!
//! let o = g.add_atom(Atom::xyz("O", 0.0, 0.0, 0.0));
//! let h1 = g.add_atom(Atom::xyz("H", 0.96, 0.0, 0.0));
//! let h2 = g.add_atom(Atom::xyz("H", -0.24, 0.93, 0.0));
//!
//! g.add_bond(o, h1).expect("add bond");
//! g.add_bond(o, h2).expect("add bond");
//! g.add_angle(h1, o, h2).expect("add angle");
//!
//! assert_eq!(g.n_atoms(), 3);
//! assert_eq!(g.n_bonds(), 2);
//! assert_eq!(g.n_angles(), 1);
//!
//! g.translate([1.0, 0.0, 0.0]);
//! assert!((g.get_atom(o).expect("get atom").get_f64("x").unwrap() - 1.0).abs() < 1e-12);
//! ```

use std::collections::HashMap;
use std::ops::{Index, IndexMut};

use slotmap::{SlotMap, new_key_type};

use super::block::Block;
use super::frame::Frame;
use crate::error::MolRsError;
use crate::types::{F, I, U};

// ---------------------------------------------------------------------------
// PropValue
// ---------------------------------------------------------------------------

/// Heterogeneous property value stored in an [`Atom`].
#[derive(Debug, Clone, PartialEq)]
pub enum PropValue {
    F64(f64),
    Str(String),
    Int(I),
}

impl From<f64> for PropValue {
    fn from(v: f64) -> Self {
        PropValue::F64(v)
    }
}
impl From<I> for PropValue {
    fn from(v: I) -> Self {
        PropValue::Int(v)
    }
}
impl From<&str> for PropValue {
    fn from(v: &str) -> Self {
        PropValue::Str(v.to_owned())
    }
}
impl From<String> for PropValue {
    fn from(v: String) -> Self {
        PropValue::Str(v)
    }
}

// ---------------------------------------------------------------------------
// Atom  (dynamic prop bag — also used for beads via `type Bead = Atom`)
// ---------------------------------------------------------------------------

/// A dynamic property bag representing an atom (or bead).
///
/// All data — including coordinates (`"x"`, `"y"`, `"z"`), element symbol,
/// mass, charge, etc. — is stored as key-value pairs.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Atom {
    props: HashMap<String, PropValue>,
}

impl Atom {
    /// Create an empty atom.
    pub fn new() -> Self {
        Self::default()
    }

    /// Convenience: create an atom with symbol + xyz.
    pub fn xyz(symbol: &str, x: f64, y: f64, z: f64) -> Self {
        let mut a = Self::new();
        a.set("element", symbol);
        a.set("x", x);
        a.set("y", y);
        a.set("z", z);
        a
    }

    // ---- dict-like API ----

    /// Insert or update a property.
    pub fn set(&mut self, key: &str, val: impl Into<PropValue>) {
        self.props.insert(key.to_owned(), val.into());
    }

    /// Get a reference to a property value.
    pub fn get(&self, key: &str) -> Option<&PropValue> {
        self.props.get(key)
    }

    /// Get a mutable reference to a property value.
    pub fn get_mut(&mut self, key: &str) -> Option<&mut PropValue> {
        self.props.get_mut(key)
    }

    /// Try to read a property as `f64`.
    pub fn get_f64(&self, key: &str) -> Option<f64> {
        match self.props.get(key)? {
            PropValue::F64(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to read a property as `&str`.
    pub fn get_str(&self, key: &str) -> Option<&str> {
        match self.props.get(key)? {
            PropValue::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Try to read a property as `I`.
    pub fn get_int(&self, key: &str) -> Option<I> {
        match self.props.get(key)? {
            PropValue::Int(v) => Some(*v),
            _ => None,
        }
    }

    /// Check whether a key exists.
    pub fn contains_key(&self, key: &str) -> bool {
        self.props.contains_key(key)
    }

    /// Remove a property, returning its value if present.
    pub fn remove(&mut self, key: &str) -> Option<PropValue> {
        self.props.remove(key)
    }

    /// Iterate over all property keys.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.props.keys().map(|k| k.as_str())
    }

    /// Number of properties.
    pub fn len(&self) -> usize {
        self.props.len()
    }

    /// Whether there are no properties.
    pub fn is_empty(&self) -> bool {
        self.props.is_empty()
    }
}

impl Index<&str> for Atom {
    type Output = PropValue;
    fn index(&self, key: &str) -> &Self::Output {
        self.props
            .get(key)
            .unwrap_or_else(|| panic!("Atom does not contain key '{}'", key))
    }
}

impl IndexMut<&str> for Atom {
    fn index_mut(&mut self, key: &str) -> &mut Self::Output {
        self.props
            .get_mut(key)
            .unwrap_or_else(|| panic!("Atom does not contain key '{}'", key))
    }
}

/// Alias for coarse-grained usage — same type, different prop keys by convention.
pub type Bead = Atom;

// ---------------------------------------------------------------------------
// Key types
// ---------------------------------------------------------------------------

new_key_type! {
    /// Stable handle to an atom in a [`MolGraph`].
    pub struct AtomId;
    /// Stable handle to a bond in a [`MolGraph`].
    pub struct BondId;
    /// Stable handle to an angle in a [`MolGraph`].
    pub struct AngleId;
    /// Stable handle to a dihedral in a [`MolGraph`].
    pub struct DihedralId;
}

// ---------------------------------------------------------------------------
// Topology structs
// ---------------------------------------------------------------------------

/// A bond between two atoms.
#[derive(Debug, Clone)]
pub struct Bond {
    pub atoms: [AtomId; 2],
    pub props: HashMap<String, PropValue>,
}

/// An angle between three atoms (j is the central atom).
#[derive(Debug, Clone)]
pub struct Angle {
    pub atoms: [AtomId; 3],
    pub props: HashMap<String, PropValue>,
}

/// A dihedral between four atoms.
#[derive(Debug, Clone)]
pub struct Dihedral {
    pub atoms: [AtomId; 4],
    pub props: HashMap<String, PropValue>,
}

// ---------------------------------------------------------------------------
// MolGraph
// ---------------------------------------------------------------------------

/// A dynamic molecular graph supporting ergonomic CRUD for atoms, bonds,
/// angles, and dihedrals.
#[derive(Debug, Clone)]
pub struct MolGraph {
    atoms: SlotMap<AtomId, Atom>,
    bonds: SlotMap<BondId, Bond>,
    angles: SlotMap<AngleId, Angle>,
    dihedrals: SlotMap<DihedralId, Dihedral>,
    adjacency: HashMap<AtomId, Vec<BondId>>,
}

impl Default for MolGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl MolGraph {
    /// Create an empty molecular graph.
    pub fn new() -> Self {
        Self {
            atoms: SlotMap::with_key(),
            bonds: SlotMap::with_key(),
            angles: SlotMap::with_key(),
            dihedrals: SlotMap::with_key(),
            adjacency: HashMap::new(),
        }
    }

    // =====================================================================
    // Atom CRUD
    // =====================================================================

    /// Insert an atom, returning its stable handle.
    pub fn add_atom(&mut self, atom: Atom) -> AtomId {
        let id = self.atoms.insert(atom);
        self.adjacency.insert(id, Vec::new());
        id
    }

    /// Remove an atom and all incident bonds / angles / dihedrals.
    pub fn remove_atom(&mut self, id: AtomId) -> Result<Atom, MolRsError> {
        let atom = self
            .atoms
            .remove(id)
            .ok_or_else(|| MolRsError::not_found("atom", format!("AtomId {:?}", id)))?;

        // Collect incident bonds.
        let incident: Vec<BondId> = self.adjacency.remove(&id).unwrap_or_default();
        for bid in incident {
            self.remove_bond_inner(bid, id);
        }

        // Remove angles referencing this atom.
        let doomed_angles: Vec<AngleId> = self
            .angles
            .iter()
            .filter(|(_, a)| a.atoms.contains(&id))
            .map(|(aid, _)| aid)
            .collect();
        for aid in doomed_angles {
            self.angles.remove(aid);
        }

        // Remove dihedrals referencing this atom.
        let doomed_dihedrals: Vec<DihedralId> = self
            .dihedrals
            .iter()
            .filter(|(_, d)| d.atoms.contains(&id))
            .map(|(did, _)| did)
            .collect();
        for did in doomed_dihedrals {
            self.dihedrals.remove(did);
        }

        Ok(atom)
    }

    /// Get a reference to an atom.
    pub fn get_atom(&self, id: AtomId) -> Result<&Atom, MolRsError> {
        self.atoms
            .get(id)
            .ok_or_else(|| MolRsError::not_found("atom", format!("AtomId {:?}", id)))
    }

    /// Get a mutable reference to an atom.
    pub fn get_atom_mut(&mut self, id: AtomId) -> Result<&mut Atom, MolRsError> {
        self.atoms
            .get_mut(id)
            .ok_or_else(|| MolRsError::not_found("atom", format!("AtomId {:?}", id)))
    }

    // =====================================================================
    // Bond CRUD
    // =====================================================================

    /// Add a bond between two existing atoms.
    pub fn add_bond(&mut self, a: AtomId, b: AtomId) -> Result<BondId, MolRsError> {
        if !self.atoms.contains_key(a) {
            return Err(MolRsError::not_found("atom", format!("AtomId {:?}", a)));
        }
        if !self.atoms.contains_key(b) {
            return Err(MolRsError::not_found("atom", format!("AtomId {:?}", b)));
        }
        let bond = Bond {
            atoms: [a, b],
            props: HashMap::new(),
        };
        let bid = self.bonds.insert(bond);
        self.adjacency.entry(a).or_default().push(bid);
        self.adjacency.entry(b).or_default().push(bid);
        Ok(bid)
    }

    /// Remove a bond and update the adjacency index.
    pub fn remove_bond(&mut self, id: BondId) -> Result<Bond, MolRsError> {
        let bond = self
            .bonds
            .remove(id)
            .ok_or_else(|| MolRsError::not_found("bond", format!("BondId {:?}", id)))?;
        for &aid in &bond.atoms {
            if let Some(adj) = self.adjacency.get_mut(&aid) {
                adj.retain(|bid| *bid != id);
            }
        }
        Ok(bond)
    }

    /// Get a reference to a bond.
    pub fn get_bond(&self, id: BondId) -> Result<&Bond, MolRsError> {
        self.bonds
            .get(id)
            .ok_or_else(|| MolRsError::not_found("bond", format!("BondId {:?}", id)))
    }

    /// Get a mutable reference to a bond.
    pub fn get_bond_mut(&mut self, id: BondId) -> Result<&mut Bond, MolRsError> {
        self.bonds
            .get_mut(id)
            .ok_or_else(|| MolRsError::not_found("bond", format!("BondId {:?}", id)))
    }

    // internal: remove bond from adjacency for a specific atom being removed
    fn remove_bond_inner(&mut self, bid: BondId, removed_atom: AtomId) {
        if let Some(bond) = self.bonds.remove(bid) {
            // Update adjacency for the *other* atom (the removed one's list is
            // already being dropped).
            for &aid in &bond.atoms {
                if aid != removed_atom
                    && let Some(adj) = self.adjacency.get_mut(&aid)
                {
                    adj.retain(|b| *b != bid);
                }
            }
        }
    }

    // =====================================================================
    // Angle CRUD
    // =====================================================================

    /// Add an angle (i-j-k, j central).
    pub fn add_angle(&mut self, i: AtomId, j: AtomId, k: AtomId) -> Result<AngleId, MolRsError> {
        for &atom_id in &[i, j, k] {
            if !self.atoms.contains_key(atom_id) {
                return Err(MolRsError::not_found(
                    "atom",
                    format!("AtomId {:?}", atom_id),
                ));
            }
        }
        let angle = Angle {
            atoms: [i, j, k],
            props: HashMap::new(),
        };
        Ok(self.angles.insert(angle))
    }

    /// Remove an angle.
    pub fn remove_angle(&mut self, id: AngleId) -> Result<Angle, MolRsError> {
        self.angles
            .remove(id)
            .ok_or_else(|| MolRsError::not_found("angle", format!("AngleId {:?}", id)))
    }

    /// Get a reference to an angle.
    pub fn get_angle(&self, id: AngleId) -> Result<&Angle, MolRsError> {
        self.angles
            .get(id)
            .ok_or_else(|| MolRsError::not_found("angle", format!("AngleId {:?}", id)))
    }

    // =====================================================================
    // Dihedral CRUD
    // =====================================================================

    /// Add a dihedral (i-j-k-l).
    pub fn add_dihedral(
        &mut self,
        i: AtomId,
        j: AtomId,
        k: AtomId,
        l: AtomId,
    ) -> Result<DihedralId, MolRsError> {
        for &atom_id in &[i, j, k, l] {
            if !self.atoms.contains_key(atom_id) {
                return Err(MolRsError::not_found(
                    "atom",
                    format!("AtomId {:?}", atom_id),
                ));
            }
        }
        let dih = Dihedral {
            atoms: [i, j, k, l],
            props: HashMap::new(),
        };
        Ok(self.dihedrals.insert(dih))
    }

    /// Remove a dihedral.
    pub fn remove_dihedral(&mut self, id: DihedralId) -> Result<Dihedral, MolRsError> {
        self.dihedrals
            .remove(id)
            .ok_or_else(|| MolRsError::not_found("dihedral", format!("DihedralId {:?}", id)))
    }

    /// Get a reference to a dihedral.
    pub fn get_dihedral(&self, id: DihedralId) -> Result<&Dihedral, MolRsError> {
        self.dihedrals
            .get(id)
            .ok_or_else(|| MolRsError::not_found("dihedral", format!("DihedralId {:?}", id)))
    }

    // =====================================================================
    // Iteration & Query
    // =====================================================================

    /// Iterate over all `(AtomId, &Atom)` pairs.
    pub fn atoms(&self) -> impl Iterator<Item = (AtomId, &Atom)> {
        self.atoms.iter()
    }

    /// Iterate over all `(BondId, &Bond)` pairs.
    pub fn bonds(&self) -> impl Iterator<Item = (BondId, &Bond)> {
        self.bonds.iter()
    }

    /// Iterate over all `(AngleId, &Angle)` pairs.
    pub fn angles(&self) -> impl Iterator<Item = (AngleId, &Angle)> {
        self.angles.iter()
    }

    /// Iterate over all `(DihedralId, &Dihedral)` pairs.
    pub fn dihedrals(&self) -> impl Iterator<Item = (DihedralId, &Dihedral)> {
        self.dihedrals.iter()
    }

    /// Number of atoms.
    pub fn n_atoms(&self) -> usize {
        self.atoms.len()
    }
    /// Number of bonds.
    pub fn n_bonds(&self) -> usize {
        self.bonds.len()
    }
    /// Number of angles.
    pub fn n_angles(&self) -> usize {
        self.angles.len()
    }
    /// Number of dihedrals.
    pub fn n_dihedrals(&self) -> usize {
        self.dihedrals.len()
    }

    /// Iterate over neighbor atom IDs of a given atom (via bond connectivity).
    pub fn neighbors(&self, id: AtomId) -> impl Iterator<Item = AtomId> + '_ {
        self.adjacency
            .get(&id)
            .into_iter()
            .flat_map(|bonds| bonds.iter())
            .filter_map(move |&bid| {
                let bond = self.bonds.get(bid)?;
                let other = if bond.atoms[0] == id {
                    bond.atoms[1]
                } else {
                    bond.atoms[0]
                };
                Some(other)
            })
    }

    /// Iterate over `(neighbor_id, bond_order)` for a given atom.
    ///
    /// Bond order is read from the `"order"` property (default 1.0).
    pub fn neighbor_bonds(&self, id: AtomId) -> impl Iterator<Item = (AtomId, f64)> + '_ {
        self.adjacency
            .get(&id)
            .into_iter()
            .flat_map(|bonds| bonds.iter())
            .filter_map(move |&bid| {
                let bond = self.bonds.get(bid)?;
                let other = if bond.atoms[0] == id {
                    bond.atoms[1]
                } else {
                    bond.atoms[0]
                };
                let order = match bond.props.get("order") {
                    Some(PropValue::F64(v)) => *v,
                    _ => 1.0,
                };
                Some((other, order))
            })
    }

    // =====================================================================
    // Spatial transforms
    // =====================================================================

    /// Translate all atoms that have `x`/`y`/`z` props.
    pub fn translate(&mut self, delta: [f64; 3]) {
        for (_, atom) in self.atoms.iter_mut() {
            let keys = ["x", "y", "z"];
            for (i, key) in keys.iter().enumerate() {
                if let Some(PropValue::F64(v)) = atom.get_mut(key) {
                    *v += delta[i];
                }
            }
        }
    }

    /// Rotate all atoms that have `x`/`y`/`z` props around `axis` by `angle`
    /// (radians), optionally about a center point.
    pub fn rotate(&mut self, axis: [f64; 3], angle: f64, about: Option<[f64; 3]>) {
        // Normalize axis
        let len = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        if len < 1e-15 {
            return;
        }
        let k = [axis[0] / len, axis[1] / len, axis[2] / len];
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        let origin = about.unwrap_or([0.0, 0.0, 0.0]);

        for (_, atom) in self.atoms.iter_mut() {
            let (Some(PropValue::F64(x)), Some(PropValue::F64(y)), Some(PropValue::F64(z))) =
                (atom.get("x"), atom.get("y"), atom.get("z"))
            else {
                continue;
            };
            let p = [x - origin[0], y - origin[1], z - origin[2]];

            // Rodrigues' rotation formula: v' = v cos θ + (k × v) sin θ + k (k·v)(1 − cos θ)
            let kdotp = k[0] * p[0] + k[1] * p[1] + k[2] * p[2];
            let cross = [
                k[1] * p[2] - k[2] * p[1],
                k[2] * p[0] - k[0] * p[2],
                k[0] * p[1] - k[1] * p[0],
            ];
            let rotated = [
                p[0] * cos_a + cross[0] * sin_a + k[0] * kdotp * (1.0 - cos_a) + origin[0],
                p[1] * cos_a + cross[1] * sin_a + k[1] * kdotp * (1.0 - cos_a) + origin[1],
                p[2] * cos_a + cross[2] * sin_a + k[2] * kdotp * (1.0 - cos_a) + origin[2],
            ];

            atom.set("x", rotated[0]);
            atom.set("y", rotated[1]);
            atom.set("z", rotated[2]);
        }
    }

    // =====================================================================
    // Composition
    // =====================================================================

    /// Merge another `MolGraph` into `self`, consuming `other`.
    /// All IDs in `other` are remapped to new IDs in `self`.
    pub fn merge(&mut self, other: MolGraph) {
        let mut atom_map: HashMap<AtomId, AtomId> = HashMap::new();

        // Transfer atoms.
        for (old_id, atom) in other.atoms {
            let new_id = self.add_atom(atom);
            atom_map.insert(old_id, new_id);
        }

        // Transfer bonds (remap atom IDs).
        for (_, mut bond) in other.bonds {
            let a = atom_map[&bond.atoms[0]];
            let b = atom_map[&bond.atoms[1]];
            bond.atoms = [a, b];
            let bid = self.bonds.insert(bond);
            self.adjacency.entry(a).or_default().push(bid);
            self.adjacency.entry(b).or_default().push(bid);
        }

        // Transfer angles.
        for (_, mut angle) in other.angles {
            angle.atoms = [
                atom_map[&angle.atoms[0]],
                atom_map[&angle.atoms[1]],
                atom_map[&angle.atoms[2]],
            ];
            self.angles.insert(angle);
        }

        // Transfer dihedrals.
        for (_, mut dih) in other.dihedrals {
            dih.atoms = [
                atom_map[&dih.atoms[0]],
                atom_map[&dih.atoms[1]],
                atom_map[&dih.atoms[2]],
                atom_map[&dih.atoms[3]],
            ];
            self.dihedrals.insert(dih);
        }
    }

    // =====================================================================
    // Frame conversion
    // =====================================================================

    /// Export to a [`Frame`]. Each unique prop key becomes a column in the
    /// `"atoms"` block. Bonds, angles, dihedrals become separate blocks with
    /// 0-based indices referencing atom row order.
    pub fn to_frame(&self) -> Frame {
        use ndarray::Array1;

        let mut frame = Frame::new();

        // Build stable ordering of atom IDs.
        let atom_ids: Vec<AtomId> = self.atoms.keys().collect();
        let n = atom_ids.len();
        let id_to_row: HashMap<AtomId, usize> = atom_ids
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i))
            .collect();

        // Collect all unique prop keys across all atoms.
        let mut all_keys: Vec<String> = {
            let mut set = std::collections::BTreeSet::new();
            for (_, atom) in &self.atoms {
                for k in atom.keys() {
                    set.insert(k.to_owned());
                }
            }
            set.into_iter().collect()
        };
        all_keys.sort();

        // Build atoms block: one column per key.
        let mut atoms_block = Block::new();
        for key in &all_keys {
            // Determine type from first non-None value.
            let first_val = atom_ids
                .iter()
                .filter_map(|&id| self.atoms.get(id).and_then(|a| a.get(key)))
                .next();

            match first_val {
                Some(PropValue::F64(_)) => {
                    let col: Vec<F> = atom_ids
                        .iter()
                        .map(|&id| {
                            self.atoms
                                .get(id)
                                .and_then(|a| a.get_f64(key))
                                .unwrap_or(0.0) as F
                        })
                        .collect();
                    let _ = atoms_block.insert(key.as_str(), Array1::from_vec(col).into_dyn());
                }
                Some(PropValue::Int(_)) => {
                    let col: Vec<I> = atom_ids
                        .iter()
                        .map(|&id| self.atoms.get(id).and_then(|a| a.get_int(key)).unwrap_or(0))
                        .collect();
                    let _ = atoms_block.insert(key.as_str(), Array1::from_vec(col).into_dyn());
                }
                Some(PropValue::Str(_)) => {
                    let col: Vec<String> = atom_ids
                        .iter()
                        .map(|&id| {
                            self.atoms
                                .get(id)
                                .and_then(|a| a.get_str(key))
                                .unwrap_or("")
                                .to_owned()
                        })
                        .collect();
                    let _ = atoms_block.insert(key.as_str(), Array1::from_vec(col).into_dyn());
                }
                _ => {}
            }
        }
        if n > 0 {
            frame.insert("atoms", atoms_block);
        }

        // Bonds block.
        if !self.bonds.is_empty() {
            let mut bonds_block = Block::new();
            let mut col_i: Vec<U> = Vec::with_capacity(self.bonds.len());
            let mut col_j: Vec<U> = Vec::with_capacity(self.bonds.len());
            let mut col_order: Vec<F> = Vec::with_capacity(self.bonds.len());
            for (_, bond) in &self.bonds {
                col_i.push(id_to_row[&bond.atoms[0]] as U);
                col_j.push(id_to_row[&bond.atoms[1]] as U);
                let order = bond
                    .props
                    .get("order")
                    .and_then(|v| {
                        if let PropValue::F64(f) = v {
                            Some(*f as F)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(1.0);
                col_order.push(order);
            }
            let _ = bonds_block.insert("atomi", Array1::from_vec(col_i).into_dyn());
            let _ = bonds_block.insert("atomj", Array1::from_vec(col_j).into_dyn());
            let _ = bonds_block.insert("order", Array1::from_vec(col_order).into_dyn());
            frame.insert("bonds", bonds_block);
        }

        // Angles block.
        if !self.angles.is_empty() {
            let mut angles_block = Block::new();
            let mut ci: Vec<U> = Vec::with_capacity(self.angles.len());
            let mut cj: Vec<U> = Vec::with_capacity(self.angles.len());
            let mut ck: Vec<U> = Vec::with_capacity(self.angles.len());
            for (_, angle) in &self.angles {
                ci.push(id_to_row[&angle.atoms[0]] as U);
                cj.push(id_to_row[&angle.atoms[1]] as U);
                ck.push(id_to_row[&angle.atoms[2]] as U);
            }
            let _ = angles_block.insert("atomi", Array1::from_vec(ci).into_dyn());
            let _ = angles_block.insert("atomj", Array1::from_vec(cj).into_dyn());
            let _ = angles_block.insert("atomk", Array1::from_vec(ck).into_dyn());
            frame.insert("angles", angles_block);
        }

        // Dihedrals block.
        if !self.dihedrals.is_empty() {
            let mut dih_block = Block::new();
            let mut ci: Vec<U> = Vec::with_capacity(self.dihedrals.len());
            let mut cj: Vec<U> = Vec::with_capacity(self.dihedrals.len());
            let mut ck: Vec<U> = Vec::with_capacity(self.dihedrals.len());
            let mut cl: Vec<U> = Vec::with_capacity(self.dihedrals.len());
            for (_, d) in &self.dihedrals {
                ci.push(id_to_row[&d.atoms[0]] as U);
                cj.push(id_to_row[&d.atoms[1]] as U);
                ck.push(id_to_row[&d.atoms[2]] as U);
                cl.push(id_to_row[&d.atoms[3]] as U);
            }
            let _ = dih_block.insert("atomi", Array1::from_vec(ci).into_dyn());
            let _ = dih_block.insert("atomj", Array1::from_vec(cj).into_dyn());
            let _ = dih_block.insert("atomk", Array1::from_vec(ck).into_dyn());
            let _ = dih_block.insert("atoml", Array1::from_vec(cl).into_dyn());
            frame.insert("dihedrals", dih_block);
        }

        frame
    }

    /// Import from a [`Frame`]. The `"atoms"` block columns become props;
    /// `"bonds"` block `atomi`/`atomj` columns become bonds.
    pub fn from_frame(frame: &Frame) -> Result<Self, MolRsError> {
        let mut g = MolGraph::new();

        let atoms_block = frame
            .get("atoms")
            .ok_or_else(|| MolRsError::parse("Frame missing 'atoms' block"))?;

        let nrows = atoms_block.nrows().unwrap_or(0);

        // Collect column keys.
        let col_keys: Vec<String> = atoms_block.keys().map(|k| k.to_owned()).collect();

        // Pre-read columns by type.
        let mut float_cols: Vec<(&str, &ndarray::ArrayD<F>)> = Vec::new();
        let mut i64_cols: Vec<(&str, &ndarray::ArrayD<I>)> = Vec::new();
        let mut str_cols: Vec<(&str, &ndarray::ArrayD<String>)> = Vec::new();

        for key in &col_keys {
            if let Some(arr) = atoms_block.get_float(key) {
                float_cols.push((key.as_str(), arr));
            } else if let Some(arr) = atoms_block.get_int(key) {
                i64_cols.push((key.as_str(), arr));
            } else if let Some(arr) = atoms_block.get_string(key) {
                str_cols.push((key.as_str(), arr));
            }
        }

        // Build atoms row by row.
        let mut atom_ids: Vec<AtomId> = Vec::with_capacity(nrows);
        for row in 0..nrows {
            let mut atom = Atom::new();
            for &(key, arr) in &float_cols {
                #[allow(clippy::unnecessary_cast)]
                atom.set(key, arr[[row]] as f64);
            }
            for &(key, arr) in &i64_cols {
                atom.set(key, PropValue::Int(arr[[row]]));
            }
            for &(key, arr) in &str_cols {
                atom.set(key, PropValue::Str(arr[[row]].clone()));
            }
            atom_ids.push(g.add_atom(atom));
        }

        // Bonds.
        if let Some(bonds_block) = frame.get("bonds") {
            let col_i = bonds_block
                .get_uint("atomi")
                .ok_or_else(|| MolRsError::parse("bonds block missing 'atomi' column"))?;
            let col_j = bonds_block
                .get_uint("atomj")
                .ok_or_else(|| MolRsError::parse("bonds block missing 'atomj' column"))?;

            // Bond order can be stored as float or uint
            let col_order_f = bonds_block.get_float("order");
            let col_order_u = bonds_block.get_uint("order");

            let nb = bonds_block.nrows().unwrap_or(0);
            for row in 0..nb {
                let ai = col_i[[row]] as usize;
                let aj = col_j[[row]] as usize;
                if ai < atom_ids.len() && aj < atom_ids.len() {
                    let bid = g.add_bond(atom_ids[ai], atom_ids[aj])?;
                    #[allow(clippy::unnecessary_cast)]
                    let order = col_order_f
                        .map(|c| c[[row]] as f64)
                        .or_else(|| col_order_u.map(|c| c[[row]] as f64));
                    if let Some(o) = order
                        && let Ok(bond) = g.get_bond_mut(bid)
                    {
                        bond.props.insert("order".to_string(), PropValue::F64(o));
                    }
                }
            }
        }

        // Angles.
        if let Some(angles_block) = frame.get("angles")
            && let (Some(ci), Some(cj), Some(ck)) = (
                angles_block.get_uint("atomi"),
                angles_block.get_uint("atomj"),
                angles_block.get_uint("atomk"),
            )
        {
            let na = angles_block.nrows().unwrap_or(0);
            for row in 0..na {
                let ai = ci[[row]] as usize;
                let aj = cj[[row]] as usize;
                let ak = ck[[row]] as usize;
                if ai < atom_ids.len() && aj < atom_ids.len() && ak < atom_ids.len() {
                    g.add_angle(atom_ids[ai], atom_ids[aj], atom_ids[ak])?;
                }
            }
        }

        // Dihedrals.
        if let Some(dih_block) = frame.get("dihedrals")
            && let (Some(ci), Some(cj), Some(ck), Some(cl)) = (
                dih_block.get_uint("atomi"),
                dih_block.get_uint("atomj"),
                dih_block.get_uint("atomk"),
                dih_block.get_uint("atoml"),
            )
        {
            let nd = dih_block.nrows().unwrap_or(0);
            for row in 0..nd {
                let ai = ci[[row]] as usize;
                let aj = cj[[row]] as usize;
                let ak = ck[[row]] as usize;
                let al = cl[[row]] as usize;
                if ai < atom_ids.len()
                    && aj < atom_ids.len()
                    && ak < atom_ids.len()
                    && al < atom_ids.len()
                {
                    g.add_dihedral(atom_ids[ai], atom_ids[aj], atom_ids[ak], atom_ids[al])?;
                }
            }
        }

        Ok(g)
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ----- PropValue & Atom dict-like API -----

    #[test]
    fn test_propvalue_from() {
        let v: PropValue = std::f64::consts::PI.into();
        assert_eq!(v, PropValue::F64(std::f64::consts::PI));

        let v: PropValue = (42 as I).into();
        assert_eq!(v, PropValue::Int(42 as I));

        let v: PropValue = "H".into();
        assert_eq!(v, PropValue::Str("H".to_owned()));
    }

    #[test]
    fn test_atom_dict_api() {
        let mut a = Atom::new();
        a.set("x", 1.5);
        a.set("element", "C");
        a.set("type_id", PropValue::Int(3 as I));

        assert_eq!(a.get_f64("x"), Some(1.5));
        assert_eq!(a.get_str("element"), Some("C"));
        assert_eq!(a.get_int("type_id"), Some(3 as I));
        assert_eq!(a.get_f64("missing"), None);
        assert!(a.contains_key("x"));
        assert!(!a.contains_key("missing"));
        assert_eq!(a.len(), 3);

        a.remove("type_id");
        assert_eq!(a.len(), 2);
    }

    #[test]
    fn test_atom_index() {
        let mut a = Atom::xyz("O", 1.0, 2.0, 3.0);
        assert_eq!(a["x"], PropValue::F64(1.0));

        // Mutate via IndexMut.
        a["x"] = PropValue::F64(99.0);
        assert_eq!(a.get_f64("x"), Some(99.0));
    }

    #[test]
    fn test_atom_xyz_constructor() {
        let a = Atom::xyz("H", 0.96, 0.0, 0.0);
        assert_eq!(a.get_str("element"), Some("H"));
        assert_eq!(a.get_f64("x"), Some(0.96));
        assert_eq!(a.get_f64("y"), Some(0.0));
        assert_eq!(a.get_f64("z"), Some(0.0));
    }

    // ----- Atom CRUD -----

    #[test]
    fn test_add_remove_atom() {
        let mut g = MolGraph::new();
        assert_eq!(g.n_atoms(), 0);

        let id = g.add_atom(Atom::xyz("C", 0.0, 0.0, 0.0));
        assert_eq!(g.n_atoms(), 1);
        assert_eq!(
            g.get_atom(id).expect("get atom").get_str("element"),
            Some("C")
        );

        g.remove_atom(id).expect("remove atom");
        assert_eq!(g.n_atoms(), 0);
        assert!(g.get_atom(id).is_err());
    }

    #[test]
    fn test_atom_mut() {
        let mut g = MolGraph::new();
        let id = g.add_atom(Atom::xyz("C", 0.0, 0.0, 0.0));

        g.get_atom_mut(id).expect("get atom mut").set("x", 5.0);
        assert_eq!(g.get_atom(id).expect("get atom").get_f64("x"), Some(5.0));
    }

    // ----- Bond CRUD -----

    #[test]
    fn test_add_remove_bond() {
        let mut g = MolGraph::new();
        let a = g.add_atom(Atom::xyz("C", 0.0, 0.0, 0.0));
        let b = g.add_atom(Atom::xyz("O", 1.0, 0.0, 0.0));

        let bid = g.add_bond(a, b).expect("add bond");
        assert_eq!(g.n_bonds(), 1);

        // Neighbors.
        let neigh_a: Vec<AtomId> = g.neighbors(a).collect();
        assert_eq!(neigh_a, vec![b]);
        let neigh_b: Vec<AtomId> = g.neighbors(b).collect();
        assert_eq!(neigh_b, vec![a]);

        // Remove bond.
        g.remove_bond(bid).expect("remove bond");
        assert_eq!(g.n_bonds(), 0);
        assert_eq!(g.neighbors(a).count(), 0);
        assert_eq!(g.neighbors(b).count(), 0);
    }

    #[test]
    fn test_bond_invalid_atoms() {
        let mut g = MolGraph::new();
        let a = g.add_atom(Atom::new());
        let b = g.add_atom(Atom::new());
        // Remove b, then try to bond to it.
        g.remove_atom(b).expect("remove atom b");
        assert!(g.add_bond(a, b).is_err());
    }

    // ----- Cascading deletion -----

    #[test]
    fn test_cascading_deletion() {
        let mut g = MolGraph::new();
        let a = g.add_atom(Atom::xyz("C", 0.0, 0.0, 0.0));
        let b = g.add_atom(Atom::xyz("H", 1.0, 0.0, 0.0));
        let c = g.add_atom(Atom::xyz("H", -1.0, 0.0, 0.0));
        let d = g.add_atom(Atom::xyz("H", 0.0, 1.0, 0.0));

        g.add_bond(a, b).expect("add bond");
        g.add_bond(a, c).expect("add bond");
        g.add_bond(a, d).expect("add bond");
        g.add_angle(b, a, c).expect("add angle");
        g.add_dihedral(b, a, c, d).expect("add dihedral");

        assert_eq!(g.n_bonds(), 3);
        assert_eq!(g.n_angles(), 1);
        assert_eq!(g.n_dihedrals(), 1);

        // Remove the central atom — everything incident must go.
        g.remove_atom(a).expect("remove atom a");
        assert_eq!(g.n_atoms(), 3);
        assert_eq!(g.n_bonds(), 0);
        assert_eq!(g.n_angles(), 0);
        assert_eq!(g.n_dihedrals(), 0);
    }

    // ----- Angle & Dihedral CRUD -----

    #[test]
    fn test_angle_crud() {
        let mut g = MolGraph::new();
        let a = g.add_atom(Atom::new());
        let b = g.add_atom(Atom::new());
        let c = g.add_atom(Atom::new());

        let aid = g.add_angle(a, b, c).expect("add angle");
        assert_eq!(g.n_angles(), 1);
        assert_eq!(g.get_angle(aid).expect("get angle").atoms, [a, b, c]);

        g.remove_angle(aid).expect("remove angle");
        assert_eq!(g.n_angles(), 0);
    }

    #[test]
    fn test_dihedral_crud() {
        let mut g = MolGraph::new();
        let a = g.add_atom(Atom::new());
        let b = g.add_atom(Atom::new());
        let c = g.add_atom(Atom::new());
        let d = g.add_atom(Atom::new());

        let did = g.add_dihedral(a, b, c, d).expect("add dihedral");
        assert_eq!(g.n_dihedrals(), 1);
        assert_eq!(
            g.get_dihedral(did).expect("get dihedral").atoms,
            [a, b, c, d]
        );

        g.remove_dihedral(did).expect("remove dihedral");
        assert_eq!(g.n_dihedrals(), 0);
    }

    // ----- Iteration -----

    #[test]
    fn test_iteration_counts() {
        let mut g = MolGraph::new();
        let a = g.add_atom(Atom::xyz("H", 0.0, 0.0, 0.0));
        let b = g.add_atom(Atom::xyz("O", 1.0, 0.0, 0.0));
        let c = g.add_atom(Atom::xyz("H", 2.0, 0.0, 0.0));

        g.add_bond(a, b).expect("add bond");
        g.add_bond(b, c).expect("add bond");
        g.add_angle(a, b, c).expect("add angle");

        assert_eq!(g.atoms().count(), 3);
        assert_eq!(g.bonds().count(), 2);
        assert_eq!(g.angles().count(), 1);
        assert_eq!(g.dihedrals().count(), 0);
    }

    // ----- Neighbor query -----

    #[test]
    fn test_neighbors() {
        let mut g = MolGraph::new();
        let a = g.add_atom(Atom::new());
        let b = g.add_atom(Atom::new());
        let c = g.add_atom(Atom::new());

        g.add_bond(a, b).expect("add bond");
        g.add_bond(a, c).expect("add bond");

        let mut n: Vec<AtomId> = g.neighbors(a).collect();
        n.sort_by_key(|id| id.0); // deterministic order
        assert_eq!(n.len(), 2);
        assert!(n.contains(&b));
        assert!(n.contains(&c));

        // b's only neighbor is a.
        let nb: Vec<AtomId> = g.neighbors(b).collect();
        assert_eq!(nb, vec![a]);
    }

    // ----- Spatial transforms -----

    #[test]
    fn test_translate() {
        let mut g = MolGraph::new();
        let id = g.add_atom(Atom::xyz("C", 1.0, 2.0, 3.0));

        g.translate([10.0, 20.0, 30.0]);

        let a = g.get_atom(id).expect("get atom");
        assert!((a.get_f64("x").unwrap() - 11.0).abs() < 1e-12);
        assert!((a.get_f64("y").unwrap() - 22.0).abs() < 1e-12);
        assert!((a.get_f64("z").unwrap() - 33.0).abs() < 1e-12);
    }

    #[test]
    fn test_translate_skips_missing_xyz() {
        let mut g = MolGraph::new();
        let id = g.add_atom(Atom::new()); // no x/y/z
        g.translate([1.0, 2.0, 3.0]);
        // Should not panic.
        assert!(g.get_atom(id).expect("get atom").get_f64("x").is_none());
    }

    #[test]
    fn test_rotate_90_deg_z() {
        let mut g = MolGraph::new();
        let id = g.add_atom(Atom::xyz("C", 1.0, 0.0, 0.0));

        let half_pi = std::f64::consts::FRAC_PI_2;
        g.rotate([0.0, 0.0, 1.0], half_pi, None);

        let a = g.get_atom(id).expect("get atom");
        assert!((a.get_f64("x").unwrap()).abs() < 1e-12);
        assert!((a.get_f64("y").unwrap() - 1.0).abs() < 1e-12);
        assert!((a.get_f64("z").unwrap()).abs() < 1e-12);
    }

    // ----- Frame conversion round-trip -----

    #[test]
    fn test_to_from_frame() {
        let mut g = MolGraph::new();
        let o = g.add_atom(Atom::xyz("O", 0.0, 0.0, 0.0));
        let h1 = g.add_atom(Atom::xyz("H", 0.96, 0.0, 0.0));
        let h2 = g.add_atom(Atom::xyz("H", -0.24, 0.93, 0.0));

        g.add_bond(o, h1).expect("add bond");
        g.add_bond(o, h2).expect("add bond");
        g.add_angle(h1, o, h2).expect("add angle");

        let frame = g.to_frame();
        assert!(frame.contains_key("atoms"));
        assert!(frame.contains_key("bonds"));
        assert!(frame.contains_key("angles"));

        assert_eq!(frame["atoms"].nrows(), Some(3));
        assert_eq!(frame["bonds"].nrows(), Some(2));
        assert_eq!(frame["angles"].nrows(), Some(1));

        // Round-trip.
        let g2 = MolGraph::from_frame(&frame).expect("from_frame");
        assert_eq!(g2.n_atoms(), 3);
        assert_eq!(g2.n_bonds(), 2);
        assert_eq!(g2.n_angles(), 1);
    }

    // ----- Merge -----

    #[test]
    fn test_merge() {
        let mut g1 = MolGraph::new();
        let a = g1.add_atom(Atom::xyz("C", 0.0, 0.0, 0.0));
        let b = g1.add_atom(Atom::xyz("O", 1.0, 0.0, 0.0));
        g1.add_bond(a, b).expect("add bond");

        let mut g2 = MolGraph::new();
        let c = g2.add_atom(Atom::xyz("N", 5.0, 0.0, 0.0));
        let d = g2.add_atom(Atom::xyz("H", 6.0, 0.0, 0.0));
        g2.add_bond(c, d).expect("add bond");

        g1.merge(g2);

        assert_eq!(g1.n_atoms(), 4);
        assert_eq!(g1.n_bonds(), 2);
    }

    // ----- Clone independence -----

    #[test]
    fn test_clone_independence() {
        let mut g = MolGraph::new();
        let id = g.add_atom(Atom::xyz("C", 0.0, 0.0, 0.0));
        g.add_atom(Atom::xyz("H", 1.0, 0.0, 0.0));

        let mut g2 = g.clone();
        // Mutate original.
        g.get_atom_mut(id).expect("get atom mut").set("x", 99.0);

        // Clone should still have original value (slotmap keys carry across clone).
        assert_eq!(g2.get_atom(id).expect("get atom").get_f64("x"), Some(0.0));

        // Independent counts.
        assert_eq!(g2.n_atoms(), 2);
        let first_id = {
            let (fid, _) = g2.atoms().next().unwrap();
            fid
        };
        g2.remove_atom(first_id).expect("remove atom");
        assert_eq!(g2.n_atoms(), 1);
        // Original unaffected.
        assert_eq!(g.n_atoms(), 2);
    }

    // ----- Realistic molecule: water -----

    #[test]
    fn test_water_molecule() {
        let mut water = MolGraph::new();
        let o = water.add_atom(Atom::xyz("O", 0.0, 0.0, 0.0));
        let h1 = water.add_atom(Atom::xyz("H", 0.9572, 0.0, 0.0));
        let h2 = water.add_atom(Atom::xyz("H", -0.2399, 0.9266, 0.0));

        water.add_bond(o, h1).expect("add bond");
        water.add_bond(o, h2).expect("add bond");
        water.add_angle(h1, o, h2).expect("add angle");

        assert_eq!(water.n_atoms(), 3);
        assert_eq!(water.n_bonds(), 2);
        assert_eq!(water.n_angles(), 1);

        // O should have 2 neighbors.
        assert_eq!(water.neighbors(o).count(), 2);
        // Each H should have 1 neighbor.
        assert_eq!(water.neighbors(h1).count(), 1);
        assert_eq!(water.neighbors(h2).count(), 1);
    }

    // ----- Coarse-grained usage -----

    #[test]
    fn test_coarse_grained() {
        let mut g = MolGraph::new();

        // Bead is just Atom alias — use different prop keys.
        let mut b1 = Bead::new();
        b1.set("name", "W");
        b1.set("x", 0.0);
        b1.set("y", 0.0);
        b1.set("z", 0.0);
        b1.set("mass", 72.0);

        let mut b2 = Bead::new();
        b2.set("name", "W");
        b2.set("x", 4.7);
        b2.set("y", 0.0);
        b2.set("z", 0.0);
        b2.set("mass", 72.0);

        let id1 = g.add_atom(b1);
        let id2 = g.add_atom(b2);
        g.add_bond(id1, id2).expect("add bond");

        assert_eq!(g.n_atoms(), 2);
        assert_eq!(g.n_bonds(), 1);
        assert_eq!(
            g.get_atom(id1).expect("get atom").get_f64("mass"),
            Some(72.0)
        );
        assert_eq!(
            g.get_atom(id1).expect("get atom").get_str("name"),
            Some("W")
        );
    }
}
