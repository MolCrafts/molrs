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

/// Result of [`CoarseGrain::extract_subgraph`].
#[derive(Debug, Clone)]
pub struct ExtractedCoarseGrain {
    pub graph: CoarseGrain,
    pub boundary: Vec<BeadId>,
    pub parent_of: HashMap<BeadId, BeadId>,
    pub hops: HashMap<BeadId, i64>,
    pub node_map: HashMap<BeadId, BeadId>,
}

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
    ///
    /// # Panics
    ///
    /// Panics when a value of the bag this builds contradicts the element
    /// type an existing bead column holds for that key (a string `x` where
    /// `x` is an `f64` column). The bag is built here out of typed arguments,
    /// so that is a defect in the graph's own vocabulary and not a data
    /// condition; the generic
    /// [`MolGraph::add_node_with`](crate::system::molgraph::MolGraph::add_node_with)
    /// reached through [`as_molgraph_mut`](Self::as_molgraph_mut) returns the
    /// conflict for callers holding a foreign bag.
    pub fn add_bead(&mut self, bead_type: &str, x: f64, y: f64, z: f64) -> BeadId {
        let mut a = Atom::new();
        a.set("bead_type", bead_type);
        a.set("x", x);
        a.set("y", y);
        a.set("z", z);
        self.graph
            .add_node_with(a)
            .expect("caller-built bead bag contradicts an existing bead column")
    }

    /// Add a bead with type name only (no coordinates).
    ///
    /// # Panics
    ///
    /// Panics when the `bead_type` column already holds a different element
    /// type — see [`add_bead`](Self::add_bead).
    pub fn add_bead_bare(&mut self, bead_type: &str) -> BeadId {
        let mut a = Atom::new();
        a.set("bead_type", bead_type);
        self.graph
            .add_node_with(a)
            .expect("caller-built bead bag contradicts an existing bead column")
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
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] when a bead or bond property contradicts the
    /// dtype the Frame schema declares for its key; the message names the
    /// refused column. The inner [`MolGraph`] accepts any value under a key it
    /// has no column for, so a string written under `"x"` is legal in the
    /// graph and only refused here.
    pub fn to_frame(&self) -> Result<Frame, MolRsError> {
        let mut frame = self.graph.to_frame()?;
        frame.rename_block("atoms", "beads");
        // A CG system with no bonds has no `bonds` block, so the rename is
        // legitimately a no-op there. When the block *is* present the graph
        // just wrote both endpoints, so a failure is a broken invariant rather
        // than a data condition — hence expect, not a swallowed Result.
        if frame.contains_key("bonds") {
            frame
                .rename_column("bonds", "atomi", "ibead")
                .expect("graph-built bonds block carries atomi");
            frame
                .rename_column("bonds", "atomj", "jbead")
                .expect("graph-built bonds block carries atomj");
        }
        frame.rename_block("bonds", "cgbonds");
        Ok(frame)
    }

    /// Build from the CG-shaped [`Frame`] emitted by [`Self::to_frame`].
    ///
    /// [`MolGraph`] owns one canonical frame vocabulary (`atoms` / `bonds` /
    /// `atomi` / `atomj`), so the CG domain labels are reversed on a clone before
    /// delegating.  The caller's frame is never mutated.
    pub fn from_frame(frame: &Frame) -> Result<Self, MolRsError> {
        let mut canonical = frame.clone();
        if !canonical.rename_block("beads", "atoms") {
            return Err(MolRsError::parse("Frame missing 'beads' block"));
        }
        if canonical.contains_key("cgbonds") {
            // The rename is a write into `atomi`/`atomj`, so the schema checks
            // the moved column against the endpoint spec (UInt). A CG frame
            // carrying signed endpoints is rejected here rather than silently
            // losing its topology downstream.
            canonical
                .rename_column("cgbonds", "ibead", "atomi")
                .and_then(|()| canonical.rename_column("cgbonds", "jbead", "atomj"))
                .map_err(|e| {
                    MolRsError::parse(format!("Frame 'cgbonds' endpoints unusable: {e}"))
                })?;
            if !canonical.rename_block("cgbonds", "bonds") {
                return Err(MolRsError::parse(
                    "Frame 'cgbonds' block could not be renamed",
                ));
            }
        }
        let mut cg = Self::new();
        cg.graph.read_frame(&canonical)?;
        Ok(cg)
    }

    /// Promote from a [`MolGraph`], validating all nodes have `"bead_type"`.
    ///
    /// The CG `bonds` kind is re-registered **by name**: an existing `bonds`
    /// kind of arity 2 keeps its id, and a missing one is registered fresh.
    /// Resolving by dense id instead would report a foreign kind's relations as
    /// this system's CG bonds whenever the graph registered something else
    /// first, and the next `add_bond` would write into it.
    ///
    /// # Errors
    ///
    /// Returns [`MolRsError::Validation`] when a node carries no `"bead_type"`,
    /// and when the graph already spells `bonds` at an arity other than 2
    /// (naming the kind and both arities).
    pub fn try_from_molgraph(mut mol: MolGraph) -> Result<Self, MolRsError> {
        for (id, atom) in mol.nodes() {
            if atom.get_str("bead_type").is_none() {
                return Err(MolRsError::validation(format!(
                    "node {:?} missing 'bead_type' property",
                    id
                )));
            }
        }
        let bond = mol.try_register_kind("bonds", 2)?;
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

    /// Translate every bead that has coordinates by `delta`.
    pub fn translate(&mut self, delta: [f64; 3]) {
        crate::spatial::geometry::translate(self.as_molgraph_mut(), delta);
    }

    /// Rotate every bead that has coordinates by `angle` radians about `axis`.
    /// `about` defaults to the origin when `None`.
    pub fn rotate(&mut self, axis: [f64; 3], angle: f64, about: Option<[f64; 3]>) {
        crate::spatial::geometry::rotate(self.as_molgraph_mut(), axis, angle, about);
    }

    // ---- subgraph extraction / composition ----

    /// Induced subgraph on an explicit bead set. Stale handles fail-fast.
    pub fn induced_subgraph(
        &self,
        beads: &[BeadId],
    ) -> Result<(CoarseGrain, HashMap<BeadId, BeadId>), MolRsError> {
        let induced = self.graph.induced_subgraph(beads)?;
        let mut cg = CoarseGrain::try_from_molgraph(induced.graph)?;
        for (&old, &new) in &induced.node_map {
            let mem = self.bead_members(old);
            if !mem.is_empty() {
                cg.set_bead_members(new, mem.to_vec());
            }
        }
        Ok((cg, induced.node_map))
    }

    /// Radius ball around `centers` over CG bonds. Membership (opaque foreign
    /// atom handles) is copied for selected beads unchanged.
    pub fn extract_subgraph(
        &self,
        centers: &[BeadId],
        radius: i64,
    ) -> Result<ExtractedCoarseGrain, MolRsError> {
        let ball = self.graph.extract_ball(
            centers,
            radius,
            self.bond,
            /* copy_higher_order */ true,
            /* whole_groups */ &[],
        )?;
        let mut cg = CoarseGrain::try_from_molgraph(ball.graph)?;
        for (&old, &new) in &ball.node_map {
            let mem = self.bead_members(old);
            if !mem.is_empty() {
                cg.set_bead_members(new, mem.to_vec());
            }
        }
        Ok(ExtractedCoarseGrain {
            graph: cg,
            boundary: ball.boundary,
            parent_of: ball.parent_of,
            hops: ball.hops,
            node_map: ball.node_map,
        })
    }

    /// Structural merge. Returns `handle in other → handle in self`. Remaps bead
    /// membership keys; foreign atom handles inside membership stay as-is.
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] when a bead or relation property of `other`
    /// contradicts the element type `self` holds for that key — see
    /// [`MolGraph::merge`](crate::system::molgraph::MolGraph::merge), whose
    /// partial-write contract this inherits.
    pub fn merge(&mut self, other: CoarseGrain) -> Result<HashMap<BeadId, BeadId>, MolRsError> {
        let node_map = self.graph.merge(other.graph)?;
        for (old_bead, members) in other.members {
            if let Some(&new_bead) = node_map.get(&old_bead)
                && !members.is_empty()
            {
                self.members.insert(new_bead, members);
            }
        }
        Ok(node_map)
    }

    /// Independent deep copy. **Handles are preserved**.
    pub fn copy(&self) -> Self {
        self.clone()
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
    fn frame_round_trip_uses_cg_domain_labels() {
        let mut cg = CoarseGrain::new();
        let a = cg.add_bead("W", 0.0, 0.0, 0.0);
        let b = cg.add_bead("P1", 3.0, 0.0, 0.0);
        cg.add_bond(a, b).unwrap();

        let frame = cg.to_frame().expect("a schema-conforming graph converts");
        assert!(frame.contains_key("beads"));
        assert!(frame.contains_key("cgbonds"));
        assert!(!frame.contains_key("atoms"));
        let restored = CoarseGrain::from_frame(&frame).expect("CG frame round-trip");
        assert_eq!(restored.n_beads(), 2);
        assert_eq!(restored.n_bonds(), 1);
    }

    /// A `CoarseGrain` owns no bead setter of its own, so the door a caller
    /// reaches is the inner graph's `set_node` — and it carries the same
    /// schema opinion there: a string under the float key `charge` never
    /// becomes a bead property, so no bead column can contradict the Frame
    /// schema in the first place.
    #[test]
    fn set_node_through_the_inner_graph_refuses_a_str_under_a_schema_float_key() {
        use crate::store::block::DType;
        let mut cg = CoarseGrain::new();
        let b = cg.add_bead("W", 0.0, 0.0, 0.0);

        let err = cg
            .as_molgraph_mut()
            .set_node(b, "charge", "negative")
            .expect_err("a str cannot be stored at a schema-float key");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
        let msg = err.to_string();
        assert!(
            msg.contains("'charge'"),
            "the error names the key, got {msg}"
        );
        assert!(
            msg.contains(DType::Float.name()) && msg.contains(DType::String.name()),
            "the error names both dtypes, got {msg}"
        );
    }

    #[test]
    fn test_try_from_molgraph_missing_bead_type() {
        let mut g = MolGraph::new();
        g.register_kind("bonds", 2);
        g.add_node_with(Atom::new()).expect("fixture node");
        assert!(CoarseGrain::try_from_molgraph(g).is_err());
    }

    /// A graph whose *first* registered kind is not `bonds` — what a fragment
    /// graph looks like — must not have that foreign kind's relations reported
    /// as its CG bonds, nor be written into by the next `add_bond`.
    #[test]
    fn try_from_molgraph_resolves_bonds_by_name_not_kind_zero() {
        let mut graph = MolGraph::new();
        let ports = graph.register_kind("ports", 2);
        let mut bead = Atom::new();
        bead.set("bead_type", "W");
        let b1 = graph.add_node_with(bead.clone()).expect("fixture node");
        let b2 = graph.add_node_with(bead).expect("fixture node");
        graph.add_relation(ports, &[b1, b2]).unwrap();

        let mut cg = CoarseGrain::try_from_molgraph(graph).expect("bead types are present");
        let ports = cg.kind_id("ports").expect("the foreign kind survives");
        assert_eq!(cg.n_bonds(), 0, "a port relation is not a CG bond");

        cg.add_bond(b1, b2).expect("a bond can still be added");
        assert_eq!(cg.n_bonds(), 1);
        assert_eq!(
            cg.n_relations(ports),
            1,
            "add_bond must not write into the foreign kind"
        );
    }

    /// A caller-supplied graph that spells `bonds` at another arity is a data
    /// condition, so the promotion returns an error instead of aborting the
    /// process inside `register_kind`.
    #[test]
    fn try_from_molgraph_rejects_conflicting_arity() {
        let mut graph = MolGraph::new();
        graph.register_kind("bonds", 3);
        let err = CoarseGrain::try_from_molgraph(graph)
            .expect_err("a 3-ary 'bonds' kind conflicts with the CG bond kind");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
    }

    #[test]
    fn test_into_inner() {
        let mut cg = CoarseGrain::new();
        cg.add_bead("W", 0.0, 0.0, 0.0);
        let g: MolGraph = cg.into_inner();
        assert_eq!(g.n_nodes(), 1);
    }
}
