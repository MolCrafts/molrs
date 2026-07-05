//! Coarse-grained molecular graph.
//!
//! [`CoarseGrain`] wraps the domain-agnostic [`MolGraph`] where every node is a
//! bead (a group of atoms). It registers its own `bonds` kind and exposes the
//! bead / CG-bond vocabulary; `MolGraph` itself stays chemistry-agnostic.
//!
//! Generic graph methods (`nodes`, `neighbors`, …) remain available via
//! `Deref`/`DerefMut`.
//!
//! # Examples
//!
//! ```
//! use molrs::system::coarsegrain::CoarseGrain;
//!
//! let mut cg = CoarseGrain::new();
//! let b1 = cg.add_bead("W", 0.0, 0.0, 0.0);
//! let b2 = cg.add_bead("W", 3.0, 0.0, 0.0);
//! cg.add_bond(b1, b2).unwrap();
//!
//! assert_eq!(cg.n_beads(), 2);
//! assert_eq!(cg.n_bonds(), 1);
//! ```

use std::collections::HashMap;
use std::ops::{Deref, DerefMut};

use slotmap::Key;

use crate::error::MolRsError;
use crate::store::frame::Frame;
use crate::system::atomistic::{Bond, BondId};
use crate::system::molgraph::{Atom, KindId, MolGraph, NodeId};

/// Handle to a bead (a graph node).
pub type BeadId = NodeId;

/// Coarse-grained molecular graph.
///
/// Invariant: every node has a `"bead_type"` property.
///
/// A bead additionally owns a **membership**: the set of underlying atom handles
/// it groups. Membership is variable-size directed ownership across worlds (the
/// atoms live in a separate all-atom world), so it is **not** a fixed-arity peer
/// relation and is stored here as opaque atom handles keyed by bead, not as a
/// scalar component. Resolving a handle back to an atom view is the caller's job
/// (it owns the source world); this layer owns only the handle topology.
#[derive(Debug, Clone)]
pub struct CoarseGrain {
    graph: MolGraph,
    bond: KindId,
    members: HashMap<BeadId, Vec<u64>>,
}

impl Deref for CoarseGrain {
    type Target = MolGraph;
    fn deref(&self) -> &MolGraph {
        &self.graph
    }
}

impl DerefMut for CoarseGrain {
    fn deref_mut(&mut self) -> &mut MolGraph {
        &mut self.graph
    }
}

impl Default for CoarseGrain {
    fn default() -> Self {
        Self::new()
    }
}

impl CoarseGrain {
    /// Create an empty coarse-grained molecular graph with the CG `bonds` kind
    /// registered.
    pub fn new() -> Self {
        let mut graph = MolGraph::new();
        let bond = graph.register_kind("bonds", 2);
        Self {
            graph,
            bond,
            members: HashMap::new(),
        }
    }

    /// Add a bead with type name and 3D coordinates.
    pub fn add_bead(&mut self, bead_type: &str, x: f64, y: f64, z: f64) -> BeadId {
        let mut a = Atom::new();
        a.set("bead_type", bead_type);
        a.set("x", x);
        a.set("y", y);
        a.set("z", z);
        self.graph.add_node_with(a)
    }

    /// Add a bead with type name only (no coordinates).
    pub fn add_bead_bare(&mut self, bead_type: &str) -> BeadId {
        let mut a = Atom::new();
        a.set("bead_type", bead_type);
        self.graph.add_node_with(a)
    }

    /// Remove a bead and all incident CG bonds (and its membership).
    pub fn remove_bead(&mut self, id: BeadId) -> Result<Atom, MolRsError> {
        self.members.remove(&id);
        self.graph.remove_node(id)
    }

    // ---- bead → atom membership (opaque foreign atom handles) ----

    /// Set the atom handles a bead groups (replaces any existing membership).
    /// An empty slice clears the membership.
    pub fn set_bead_members(&mut self, bead: BeadId, atoms: Vec<u64>) {
        if atoms.is_empty() {
            self.members.remove(&bead);
        } else {
            self.members.insert(bead, atoms);
        }
    }

    /// The atom handles a bead groups (empty if none recorded).
    pub fn bead_members(&self, bead: BeadId) -> &[u64] {
        self.members.get(&bead).map_or(&[], Vec::as_slice)
    }

    /// Beads whose membership includes `atom`, in bead-handle order.
    pub fn beads_of_atom(&self, atom: u64) -> Vec<BeadId> {
        let mut out: Vec<BeadId> = self
            .members
            .iter()
            .filter(|(_, atoms)| atoms.contains(&atom))
            .map(|(&bead, _)| bead)
            .collect();
        out.sort_by_key(|b| b.data().as_ffi());
        out
    }

    /// Materialize a bead's property bag (owned copy of its set components).
    pub fn get_bead(&self, id: BeadId) -> Result<Atom, MolRsError> {
        self.graph.get_node(id)
    }

    /// Iterate over all `(BeadId, Atom)` pairs (each property bag materialized).
    pub fn beads(&self) -> impl Iterator<Item = (BeadId, Atom)> + '_ {
        self.graph.nodes()
    }

    /// Number of beads.
    pub fn n_beads(&self) -> usize {
        self.graph.n_nodes()
    }

    /// Add a CG bond between two existing beads.
    pub fn add_bond(&mut self, a: BeadId, b: BeadId) -> Result<BondId, MolRsError> {
        self.graph.add_relation(self.bond, &[a, b])
    }

    /// Materialize a CG bond (endpoints + properties).
    pub fn get_bond(&self, id: BondId) -> Result<Bond, MolRsError> {
        self.graph.get_relation(self.bond, id)
    }

    /// Iterate over all `(BondId, Bond)` pairs (each materialized).
    pub fn bonds(&self) -> impl Iterator<Item = (BondId, Bond)> + '_ {
        self.graph.relations(self.bond)
    }

    /// Number of CG bonds.
    pub fn n_bonds(&self) -> usize {
        self.graph.n_relations(self.bond)
    }

    /// Export to a tabular [`Frame`] (`beads` block + a `cgbonds` block).
    ///
    /// CoarseGrain owns its frame conversion because the central [`Frame`] is a
    /// domain interface with CG-specific block requirements: the generic node
    /// block is relabeled `atoms`→`beads` and the bond relation block
    /// `bonds`→`cgbonds` with its `atomi`/`atomj` endpoint columns renamed to
    /// `ibead`/`jbead`. All relabeling is on the already-materialized numpy
    /// columns — no data is copied.
    pub fn to_frame(&self) -> Frame {
        let mut frame = self.graph.to_frame();
        frame.rename_block("atoms", "beads");
        frame.rename_column("bonds", "atomi", "ibead");
        frame.rename_column("bonds", "atomj", "jbead");
        frame.rename_block("bonds", "cgbonds");
        frame
    }

    /// Build from a [`Frame`], registering the CG `bonds` kind first so the
    /// `bonds` block is read back.
    pub fn from_frame(frame: &Frame) -> Result<Self, MolRsError> {
        let mut cg = Self::new();
        cg.graph.read_frame(frame)?;
        Ok(cg)
    }

    /// Promote from a [`MolGraph`], validating all nodes have `"bead_type"`.
    pub fn try_from_molgraph(mol: MolGraph) -> Result<Self, MolRsError> {
        for (id, atom) in mol.nodes() {
            if atom.get_str("bead_type").is_none() {
                return Err(MolRsError::validation(format!(
                    "node {:?} missing 'bead_type' property",
                    id
                )));
            }
        }
        let bond = mol.kind_id("bonds").unwrap_or(KindId(0));
        Ok(Self {
            graph: mol,
            bond,
            members: HashMap::new(),
        })
    }

    /// Unwrap to the inner [`MolGraph`] (zero cost).
    pub fn into_inner(self) -> MolGraph {
        self.graph
    }

    /// Borrow the inner [`MolGraph`].
    pub fn as_molgraph(&self) -> &MolGraph {
        &self.graph
    }

    /// Mutably borrow the inner [`MolGraph`].
    pub fn as_molgraph_mut(&mut self) -> &mut MolGraph {
        &mut self.graph
    }

    // ---- structural graph hash (see [`crate::system::graph_hash`]) ----

    /// Isomorphism-invariant Weisfeiler–Lehman structural hash of the bead graph
    /// (bead-type node labels, bond-order edge labels). Shares the same
    /// [`MolGraph`] primitive that serves the all-atom case.
    pub fn structural_hash(&self) -> u64 {
        crate::system::graph_hash::structural_hash(&self.graph)
    }

    /// Deterministic canonical bead ordering from the WL refinement (see
    /// [`crate::system::graph_hash::canonical_order`]).
    pub fn canonical_order(&self) -> Vec<BeadId> {
        crate::system::graph_hash::canonical_order(&self.graph)
    }

    /// Whether `self` and `other` are isomorphic as labeled bead graphs.
    pub fn is_isomorphic(&self, other: &CoarseGrain) -> bool {
        crate::system::graph_hash::is_isomorphic(&self.graph, &other.graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_and_add_bead() {
        let mut cg = CoarseGrain::new();
        let b1 = cg.add_bead("W", 0.0, 0.0, 0.0);
        let b2 = cg.add_bead("P1", 3.0, 0.0, 0.0);
        cg.add_bond(b1, b2).unwrap();
        assert_eq!(cg.n_beads(), 2);
        assert_eq!(cg.n_bonds(), 1);
    }

    #[test]
    fn test_bead_has_type() {
        let mut cg = CoarseGrain::new();
        let b = cg.add_bead("W", 1.0, 2.0, 3.0);
        let bead = cg.get_bead(b).unwrap();
        assert_eq!(bead.get_str("bead_type"), Some("W"));
        assert_eq!(bead.get_f64("x"), Some(1.0));
    }

    #[test]
    fn test_try_from_molgraph_missing_bead_type() {
        let mut g = MolGraph::new();
        g.register_kind("bonds", 2);
        g.add_node_with(Atom::new());
        assert!(CoarseGrain::try_from_molgraph(g).is_err());
    }

    #[test]
    fn test_into_inner() {
        let mut cg = CoarseGrain::new();
        cg.add_bead("W", 0.0, 0.0, 0.0);
        let g: MolGraph = cg.into_inner();
        assert_eq!(g.n_nodes(), 1);
    }
}
