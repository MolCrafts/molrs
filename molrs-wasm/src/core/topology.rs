//! Bond-graph topology — the WASM face of `molrs::core::Topology`.
//!
//! The graph a frame's `bonds` block spells out, with the angles, dihedrals,
//! impropers and connected components derived from it. Ring perception is not
//! here: it is chemistry, and belongs to `Perceive.findRings`.

use molrs::core::Topology as RsTopology;
use wasm_bindgen::prelude::*;

use crate::core::frame::Frame;

/// Graph-based molecular topology with automated detection of angles,
/// dihedrals, impropers and connected components.
///
/// API mirrors igraph / molpy conventions.
///
/// # Example (JavaScript)
///
/// ```js
/// const topo = Topology.fromFrame(frame);
/// console.log(topo.nAtoms, topo.nBonds);
///
/// const angles = topo.angles();       // Uint32Array [i,j,k, ...]
/// const dihedrals = topo.dihedrals(); // Uint32Array [i,j,k,l, ...]
/// const cc = topo.connectedComponents(); // Int32Array per-atom labels
/// ```
#[wasm_bindgen(js_name = Topology)]
pub struct Topology {
    inner: RsTopology,
}

#[wasm_bindgen(js_class = Topology)]
impl Topology {
    /// Create a topology with `n` atoms and no bonds.
    #[wasm_bindgen(constructor)]
    pub fn new(n_atoms: usize) -> Self {
        Self {
            inner: RsTopology::with_atoms(n_atoms),
        }
    }

    /// Build a topology from a Frame's `bonds` block.
    ///
    /// The atom count is the `atoms` block's row count; the edges are the
    /// `bonds` block's `atomi` / `atomj` columns, in row order. A frame with no
    /// `bonds` block (or an empty one) is a graph with no edges.
    ///
    /// # Errors
    ///
    /// Throws if the frame has no `atoms` block, if a non-empty `bonds` block
    /// lacks `atomi` / `atomj`, or if a bond names an atom outside the frame.
    #[wasm_bindgen(js_name = fromFrame)]
    pub fn from_frame(frame: &Frame) -> Result<Topology, JsValue> {
        frame.with_frame(|rs_frame| {
            RsTopology::from_frame(rs_frame)
                .map(|inner| Self { inner })
                .map_err(|e| JsValue::from_str(&format!("Topology.fromFrame: {e}")))
        })
    }

    /// Number of atoms (vertices).
    #[wasm_bindgen(getter, js_name = nAtoms)]
    pub fn n_atoms(&self) -> usize {
        self.inner.n_atoms()
    }

    /// Number of bonds (edges).
    #[wasm_bindgen(getter, js_name = nBonds)]
    pub fn n_bonds(&self) -> usize {
        self.inner.n_bonds()
    }

    /// Number of unique angles.
    #[wasm_bindgen(getter, js_name = nAngles)]
    pub fn n_angles(&self) -> usize {
        self.inner.n_angles()
    }

    /// Number of unique proper dihedrals.
    #[wasm_bindgen(getter, js_name = nDihedrals)]
    pub fn n_dihedrals(&self) -> usize {
        self.inner.n_dihedrals()
    }

    /// Number of connected components.
    #[wasm_bindgen(getter, js_name = nComponents)]
    pub fn n_components(&self) -> usize {
        self.inner.n_components()
    }

    /// All bond pairs as flat `Uint32Array` `[i0,j0, i1,j1, ...]`.
    pub fn bonds(&self) -> Vec<u32> {
        self.inner
            .bonds()
            .iter()
            .flat_map(|b| [b[0] as u32, b[1] as u32])
            .collect()
    }

    /// All angle triplets as flat `Uint32Array` `[i,j,k, ...]`.
    pub fn angles(&self) -> Vec<u32> {
        self.inner
            .angles()
            .iter()
            .flat_map(|a| [a[0] as u32, a[1] as u32, a[2] as u32])
            .collect()
    }

    /// All proper dihedral quartets as flat `Uint32Array` `[i,j,k,l, ...]`.
    pub fn dihedrals(&self) -> Vec<u32> {
        self.inner
            .dihedrals()
            .iter()
            .flat_map(|d| [d[0] as u32, d[1] as u32, d[2] as u32, d[3] as u32])
            .collect()
    }

    /// All improper dihedral quartets as flat `Uint32Array` `[center,i,j,k, ...]`.
    pub fn impropers(&self) -> Vec<u32> {
        self.inner
            .impropers()
            .iter()
            .flat_map(|d| [d[0] as u32, d[1] as u32, d[2] as u32, d[3] as u32])
            .collect()
    }

    /// Per-atom connected component labels as `Int32Array`.
    ///
    /// Labels are 0-based and contiguous. Each atom gets a component ID.
    /// Atoms in the same connected subgraph share the same label.
    #[wasm_bindgen(js_name = connectedComponents)]
    pub fn connected_components(&self) -> Vec<i32> {
        self.inner
            .connected_components()
            .iter()
            .map(|&c| c as i32)
            .collect()
    }

    /// Neighbor atom indices of atom `idx` as `Uint32Array`.
    pub fn neighbors(&self, idx: usize) -> Vec<u32> {
        self.inner
            .neighbors(idx)
            .iter()
            .map(|&n| n as u32)
            .collect()
    }

    /// Degree (number of bonds) of atom `idx`.
    pub fn degree(&self, idx: usize) -> usize {
        self.inner.degree(idx)
    }

    /// Whether atoms `i` and `j` are directly bonded.
    #[wasm_bindgen(js_name = areBonded)]
    pub fn are_bonded(&self, i: usize, j: usize) -> bool {
        self.inner.are_bonded(i, j)
    }

    /// Add a single atom.
    #[wasm_bindgen(js_name = addAtom)]
    pub fn add_atom(&mut self) {
        self.inner.add_atom();
    }

    /// Add a bond between atoms `i` and `j`.
    #[wasm_bindgen(js_name = addBond)]
    pub fn add_bond(&mut self, i: usize, j: usize) {
        self.inner.add_bond(i, j);
    }

    /// Delete an atom by index.
    #[wasm_bindgen(js_name = deleteAtom)]
    pub fn delete_atom(&mut self, idx: usize) {
        self.inner.delete_atom(idx);
    }

    /// Delete a bond by edge index.
    #[wasm_bindgen(js_name = deleteBond)]
    pub fn delete_bond(&mut self, idx: usize) {
        self.inner.delete_bond(idx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::core::{Block, keys};
    use molrs::op::types::Idx;
    use ndarray::Array1;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn from_frame_reads_the_canonical_bond_columns() {
        let mut atoms = Block::new();
        atoms
            .insert("x", Array1::<f64>::zeros(3).into_dyn())
            .unwrap();
        let mut bonds = Block::new();
        bonds
            .insert(keys::ATOMI, Array1::<Idx>::from_vec(vec![0, 1]).into_dyn())
            .unwrap();
        bonds
            .insert(keys::ATOMJ, Array1::<Idx>::from_vec(vec![1, 2]).into_dyn())
            .unwrap();
        let mut rs_frame = molrs::core::Frame::new();
        rs_frame.insert("atoms", atoms);
        rs_frame.insert("bonds", bonds);
        let frame = Frame::from_rs(rs_frame).unwrap();

        let topo = Topology::from_frame(&frame).unwrap();
        assert_eq!(topo.n_atoms(), 3);
        assert_eq!(topo.n_bonds(), 2);
        assert_eq!(topo.n_angles(), 1);
    }
}
