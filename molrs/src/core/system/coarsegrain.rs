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

use ndarray::Array1;
use slotmap::Key;

use crate::error::MolRsError;
use crate::store::block::Block;
use crate::store::frame::Frame;
use crate::system::atomistic::{Bond, BondId};
use crate::system::molgraph::{Atom, KindId, MolGraph, NodeId};
use crate::types::Idx;

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
    /// Bead membership is written to a `members` block with one row per
    /// (bead, atom) pair: `ibead` (UInt, the bead's row in `beads`) and `atom`
    /// (UInt, the opaque atom handle), grouped by bead in `beads` row order and
    /// keeping each bead's member order. The block is absent when no bead has
    /// members.
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

        // `node_ids` is the row order the graph just wrote `beads` in.
        let mut ibead: Vec<Idx> = Vec::new();
        let mut atom: Vec<Idx> = Vec::new();
        for (row, id) in self.graph.node_ids().enumerate() {
            for &handle in self.bead_members(id) {
                ibead.push(row as Idx);
                atom.push(handle);
            }
        }
        if !ibead.is_empty() {
            let mut members = Block::new();
            members
                .insert("ibead", Array1::from_vec(ibead).into_dyn())
                .and_then(|()| members.insert("atom", Array1::from_vec(atom).into_dyn()))
                .map_err(|e| MolRsError::validation(format!("Frame 'members' block: {e}")))?;
            frame.insert("members", members);
        }
        Ok(frame)
    }

    /// Build from the CG-shaped [`Frame`] emitted by [`Self::to_frame`].
    ///
    /// [`MolGraph`] owns one canonical frame vocabulary (`atoms` / `bonds` /
    /// `atomi` / `atomj`), so the CG domain labels are reversed on a clone before
    /// delegating.  The caller's frame is never mutated.
    ///
    /// A `members` block (see [`Self::to_frame`]) is read back after the
    /// graph: row `ibead` of `beads` gains atom handle `atom`, in block row
    /// order.
    ///
    /// # Errors
    ///
    /// [`MolRsError::Parse`] when the frame has no `beads` block or its
    /// `cgbonds` endpoints cannot be renamed, and [`MolRsError::Validation`]
    /// when a bead carries no `bead_type`, when the `members` block lacks a
    /// UInt `ibead` or `atom` column, when an `ibead` names no bead row, or
    /// when an `(ibead, atom)` row repeats — plus every error of the graph
    /// read itself.
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
        let mut cg = Self::from_canonical_frame(&canonical)?;
        if let Some(members) = frame.get("members") {
            cg.read_members(members)?;
        }
        Ok(cg)
    }

    /// Build from the atomistic vocabulary of a LAMMPS data frame: every
    /// `atoms` row becomes one bead and every `bonds` row one CG bond.
    ///
    /// Where [`Self::from_frame`] reads the `beads` / `cgbonds` vocabulary,
    /// this reads `atoms` / `bonds` as the LAMMPS data reader
    /// (`io::data::lammps_data`) writes them: bond
    /// endpoints `atomi` / `atomj` are 0-based `atoms` rows. Every `atoms`
    /// column is copied onto its bead and every `bonds` column onto its bond.
    /// `bead_type` is taken from the `type_key` column: a `Str` column is
    /// copied verbatim, and a `UInt` / `Int` column (the reader's `type_id`) is
    /// rendered in decimal. The `type_key` column itself is kept on each bead
    /// as a plain property; with `type_key == "bead_type"` that column (which
    /// the schema declares `Str`) is the bead type and nothing is added. Only
    /// `atoms` and `bonds` are read;
    /// `angles`, `dihedrals`, `impropers` and every other block are ignored.
    ///
    /// Resolving LAMMPS "Atom Type Labels" into names is the caller's job:
    /// pass the column that already holds the type spelling you want.
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] when the frame has no `atoms` block, when
    /// `atoms` has no `type_key` column, when that column is neither `Str` nor
    /// `UInt` / `Int`, when one of its rows is null, or when `atoms` already
    /// has a `bead_type` column and `type_key` names another one (the read
    /// would overwrite it) — plus every error of
    /// the graph read itself (e.g. a bond endpoint past the atom rows).
    pub fn from_atom_frame(frame: &Frame, type_key: &str) -> Result<Self, MolRsError> {
        let atoms = frame.get("atoms").ok_or_else(|| {
            MolRsError::validation("Frame has no 'atoms' block to read beads from")
        })?;
        let Some(dtype) = atoms.dtype(type_key) else {
            return Err(MolRsError::validation(format!(
                "Frame 'atoms' block has no '{type_key}' column to take bead_type from"
            )));
        };
        if type_key != "bead_type" && atoms.contains_key("bead_type") {
            return Err(MolRsError::validation(format!(
                "Frame 'atoms' block already has a 'bead_type' column, which taking \
                 bead_type from '{type_key}' would overwrite"
            )));
        }
        if atoms.validity(type_key).is_some() {
            return Err(MolRsError::validation(format!(
                "Frame 'atoms' column '{type_key}' has a null row, so a bead would have no \
                 bead_type"
            )));
        }
        // `None`: the `type_key` column already is a Str `bead_type`.
        let bead_type: Option<Vec<String>> = if let Some(col) = atoms.get_string(type_key) {
            (type_key != "bead_type").then(|| col.iter().cloned().collect())
        } else if let Some(col) = atoms.get_uint(type_key) {
            Some(col.iter().map(ToString::to_string).collect())
        } else if let Some(col) = atoms.get_int(type_key) {
            Some(col.iter().map(ToString::to_string).collect())
        } else {
            return Err(MolRsError::validation(format!(
                "Frame 'atoms' column '{type_key}' is {}, not a Str or UInt/Int bead type",
                dtype.name()
            )));
        };

        let mut beads = atoms.clone();
        if let Some(bead_type) = bead_type {
            beads
                .insert("bead_type", Array1::from_vec(bead_type).into_dyn())
                .map_err(|e| {
                    MolRsError::validation(format!("Frame 'atoms' column '{type_key}': {e}"))
                })?;
        }
        let mut canonical = Frame::new();
        canonical.insert("atoms", beads);
        if let Some(bonds) = frame.get("bonds") {
            canonical.insert("bonds", bonds.clone());
        }
        Self::from_canonical_frame(&canonical)
    }

    /// Read a frame already in [`MolGraph`]'s canonical vocabulary (`atoms` /
    /// `bonds`) into a validated `CoarseGrain`: every bead must carry
    /// `bead_type`, as [`Self::try_from_molgraph`] requires.
    fn from_canonical_frame(canonical: &Frame) -> Result<Self, MolRsError> {
        let mut graph = MolGraph::new();
        graph.register_kind("bonds", 2);
        graph.read_frame(canonical)?;
        Self::try_from_molgraph(graph)
    }

    /// Attach the membership a `members` block states: row `r` gives bead row
    /// `ibead[r]` the atom handle `atom[r]`. Bead rows are the graph's row
    /// order, which a freshly read graph shares with its `beads` block. A
    /// repeated `(ibead, atom)` row is refused, since it would list the atom
    /// twice in its bead.
    fn read_members(&mut self, members: &Block) -> Result<(), MolRsError> {
        let column = |key: &str| {
            members.get_uint(key).ok_or_else(|| {
                MolRsError::validation(format!("Frame 'members' block has no UInt '{key}' column"))
            })
        };
        let ibead = column("ibead")?;
        let atom = column("atom")?;
        let bead_ids: Vec<BeadId> = self.graph.node_ids().collect();
        for (row, (&bead_row, &handle)) in ibead.iter().zip(atom.iter()).enumerate() {
            let Some(&bead) = bead_ids.get(bead_row as usize) else {
                return Err(MolRsError::validation(format!(
                    "Frame 'members' row {row} names bead {bead_row}, past the {} beads",
                    bead_ids.len()
                )));
            };
            let listed = self.members.entry(bead).or_default();
            if listed.contains(&handle) {
                return Err(MolRsError::validation(format!(
                    "Frame 'members' row {row} repeats atom {handle} of bead {bead_row}"
                )));
            }
            listed.push(handle);
        }
        Ok(())
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

    /// Scale every bead that has coordinates by a per-axis `factor` about
    /// `about` (the origin when `None`). Pass `[s, s, s]` for a uniform scale.
    pub fn scale(&mut self, factor: [f64; 3], about: Option<[f64; 3]>) {
        crate::spatial::geometry::scale(self.as_molgraph_mut(), factor, about);
    }

    /// Rotate every bead that has coordinates by `angle` radians about `axis`.
    /// `about` defaults to the origin when `None`.
    ///
    /// # Errors
    ///
    /// The error of [`crate::spatial::geometry::rotate`] — `axis` has no
    /// direction or `angle` is not finite; nothing moves then.
    pub fn rotate(
        &mut self,
        axis: [f64; 3],
        angle: f64,
        about: Option<[f64; 3]>,
    ) -> Result<(), crate::error::MolRsError> {
        crate::spatial::geometry::rotate(self.as_molgraph_mut(), axis, angle, about)
    }

    /// Place `transforms.len()` rigid copies of `template`, copy `c` moved by
    /// `transforms[c]` (rotation, then a translation in Å) and stamped
    /// `frag_id = frag_ids[c]`; returns the new beads copy-major. Column-wise
    /// and atomic — see [`MolGraph::replicate`].
    ///
    /// **No bead membership is copied.** Membership names foreign atoms of a
    /// separate all-atom world, and the copies do not own them; every
    /// replicated bead starts with an empty membership.
    ///
    /// # Errors
    ///
    /// The errors of [`MolGraph::replicate`]; `self` is unchanged then.
    pub fn replicate(
        &mut self,
        template: &CoarseGrain,
        transforms: &[crate::op::rigid::Rigid],
        frag_ids: &[crate::types::I],
    ) -> Result<Vec<BeadId>, MolRsError> {
        self.graph.replicate(&template.graph, transforms, frag_ids)
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

    #[test]
    fn scale_multiplies_offsets_from_the_centre_per_axis() {
        // p' = (p - c) * f + c. Point (1, 2, 3), factor (2, 3, 0.5):
        //   about c = (1, 1, 1) -> (1, 4, 2);  about the origin -> (2, 6, 1.5).
        let factor = [2.0, 3.0, 0.5];
        let cases: [(Option<[f64; 3]>, [f64; 3]); 2] = [
            (Some([1.0, 1.0, 1.0]), [1.0, 4.0, 2.0]),
            (None, [2.0, 6.0, 1.5]),
        ];
        for (about, expected) in cases {
            let mut sys = CoarseGrain::new();
            let id = sys.add_bead("W", 1.0, 2.0, 3.0);
            let fixed = sys.add_bead("W", 1.0, 1.0, 1.0);
            sys.scale(factor, about);
            let moved = sys.get_bead(id).expect("live handle");
            for (key, want) in ["x", "y", "z"].into_iter().zip(expected) {
                let got = moved.get_f64(key).expect("coordinate kept");
                assert!(
                    (got - want).abs() < 1e-12,
                    "{about:?} {key}: {got} != {want}"
                );
            }
            if about.is_some() {
                // The centre itself is a fixed point of the map.
                let centre = sys.get_bead(fixed).expect("live handle");
                for key in ["x", "y", "z"] {
                    let got = centre.get_f64(key).expect("coordinate kept");
                    assert!((got - 1.0).abs() < 1e-12, "centre {key} moved to {got}");
                }
            }
        }
    }

    // ---- from_atom_frame / from_frame validation / membership round trip ----

    use crate::store::block::Block;
    use ndarray::Array1;

    fn float_col(values: &[f64]) -> ndarray::ArrayD<f64> {
        Array1::from_vec(values.to_vec()).into_dyn()
    }

    fn uint_col(values: &[u64]) -> ndarray::ArrayD<u64> {
        Array1::from_vec(values.to_vec()).into_dyn()
    }

    fn str_col(values: &[&str]) -> ndarray::ArrayD<String> {
        Array1::from_vec(values.iter().map(|s| (*s).to_owned()).collect()).into_dyn()
    }

    /// A LAMMPS-style `atoms` block of `n` rows with x/y/z only; the caller
    /// adds the type column under test.
    fn xyz_atoms(x: &[f64], y: &[f64], z: &[f64]) -> Block {
        let mut atoms = Block::new();
        atoms.insert("x", float_col(x)).unwrap();
        atoms.insert("y", float_col(y)).unwrap();
        atoms.insert("z", float_col(z)).unwrap();
        atoms
    }

    #[test]
    fn from_atom_frame_copies_a_str_type_column_verbatim_and_every_other_column() {
        let mut atoms = xyz_atoms(&[0.0, 1.0], &[0.5, 2.0], &[-1.0, 3.0]);
        atoms.insert("type", str_col(&["CT", "OH"])).unwrap();
        atoms.insert("charge", float_col(&[0.25, -0.25])).unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);

        let cg = CoarseGrain::from_atom_frame(&frame, "type").expect("a Str type column");
        assert_eq!(cg.n_beads(), 2);
        let beads: Vec<Atom> = cg.beads().map(|(_, bead)| bead).collect();
        let expected = [("CT", 0.0, 0.5, -1.0, 0.25), ("OH", 1.0, 2.0, 3.0, -0.25)];
        for (bead, (ty, x, y, z, q)) in beads.iter().zip(expected) {
            assert_eq!(bead.get_str("bead_type"), Some(ty));
            assert_eq!(bead.get_f64("x"), Some(x));
            assert_eq!(bead.get_f64("y"), Some(y));
            assert_eq!(bead.get_f64("z"), Some(z));
            assert_eq!(bead.get_f64("charge"), Some(q));
        }
    }

    #[test]
    fn from_atom_frame_renders_a_uint_type_id_in_decimal() {
        let mut atoms = xyz_atoms(&[0.0], &[0.0], &[0.0]);
        atoms.insert("type_id", uint_col(&[2])).unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);

        let cg = CoarseGrain::from_atom_frame(&frame, "type_id").expect("a UInt type_id column");
        let (_, bead) = cg.beads().next().expect("one bead");
        assert_eq!(bead.get_str("bead_type"), Some("2"));
    }

    #[test]
    fn from_atom_frame_turns_bond_rows_into_cg_bonds() {
        let mut atoms = xyz_atoms(&[0.0, 1.0, 2.0], &[0.0; 3], &[0.0; 3]);
        atoms.insert("type", str_col(&["A", "B", "C"])).unwrap();
        // LAMMPS data frames address bond endpoints by 0-based atoms row.
        let mut bonds = Block::new();
        bonds.insert("atomi", uint_col(&[0, 1])).unwrap();
        bonds.insert("atomj", uint_col(&[2, 2])).unwrap();
        bonds.insert("type_id", uint_col(&[1, 1])).unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame.insert("bonds", bonds);

        let cg = CoarseGrain::from_atom_frame(&frame, "type").expect("atoms + bonds");
        assert_eq!(cg.n_bonds(), 2);
        let ids: Vec<BeadId> = cg.node_ids().collect();
        let mut endpoints: Vec<[BeadId; 2]> = cg
            .bonds()
            .map(|(_, bond)| [bond.nodes[0], bond.nodes[1]])
            .collect();
        endpoints.sort_by_key(|[a, _]| ids.iter().position(|id| id == a));
        assert_eq!(endpoints, vec![[ids[0], ids[2]], [ids[1], ids[2]]]);
    }

    #[test]
    fn from_atom_frame_refuses_a_missing_type_key_column() {
        let mut atoms = xyz_atoms(&[0.0], &[0.0], &[0.0]);
        atoms.insert("type_id", uint_col(&[1])).unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);

        let err = CoarseGrain::from_atom_frame(&frame, "type")
            .expect_err("no 'type' column to take bead_type from");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
    }

    #[test]
    fn from_atom_frame_refuses_a_float_type_key_column() {
        let mut atoms = xyz_atoms(&[0.0], &[0.0], &[0.0]);
        atoms.insert("mass", float_col(&[12.011])).unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);

        let err = CoarseGrain::from_atom_frame(&frame, "mass")
            .expect_err("a Float column is not a bead type");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
    }

    #[test]
    fn from_atom_frame_refuses_an_existing_bead_type_under_another_type_key() {
        let mut atoms = xyz_atoms(&[0.0], &[0.0], &[0.0]);
        atoms.insert("type", str_col(&["CT"])).unwrap();
        atoms.insert("bead_type", str_col(&["W"])).unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);

        let err = CoarseGrain::from_atom_frame(&frame, "type")
            .expect_err("taking bead_type from 'type' would overwrite the existing bead_type");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
    }

    #[test]
    fn from_atom_frame_refuses_a_frame_without_an_atoms_block() {
        let frame = Frame::new();
        let err = CoarseGrain::from_atom_frame(&frame, "type")
            .expect_err("no atoms block to read beads from");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
    }

    #[test]
    fn from_frame_refuses_a_bead_without_bead_type() {
        let beads = xyz_atoms(&[0.0, 1.0], &[0.0, 0.0], &[0.0, 0.0]);
        let mut frame = Frame::new();
        frame.insert("beads", beads);

        let err = CoarseGrain::from_frame(&frame)
            .expect_err("every bead carries bead_type, on the way in too");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
    }

    #[test]
    fn bead_membership_survives_to_frame_then_from_frame() {
        let mut cg = CoarseGrain::new();
        let a = cg.add_bead("W", 0.0, 0.0, 0.0);
        let b = cg.add_bead("P1", 1.0, 0.0, 0.0);
        cg.add_bead("P2", 2.0, 0.0, 0.0);
        cg.set_bead_members(a, vec![10, 11]);
        cg.set_bead_members(b, vec![12]);

        let frame = cg.to_frame().expect("a schema-conforming graph converts");
        let restored = CoarseGrain::from_frame(&frame).expect("CG frame round-trip");
        let ids: Vec<BeadId> = restored.node_ids().collect();
        assert_eq!(ids.len(), 3);
        assert_eq!(restored.bead_members(ids[0]), &[10, 11]);
        assert_eq!(restored.bead_members(ids[1]), &[12]);
        assert!(restored.bead_members(ids[2]).is_empty());
    }

    #[test]
    fn from_frame_refuses_a_members_row_past_the_bead_count() {
        let mut beads = xyz_atoms(&[0.0, 1.0], &[0.0, 0.0], &[0.0, 0.0]);
        beads.insert("bead_type", str_col(&["W", "W"])).unwrap();
        let mut members = Block::new();
        members.insert("ibead", uint_col(&[0, 2])).unwrap();
        members.insert("atom", uint_col(&[10, 11])).unwrap();
        let mut frame = Frame::new();
        frame.insert("beads", beads);
        frame.insert("members", members);

        let err =
            CoarseGrain::from_frame(&frame).expect_err("ibead 2 names no bead of a two-bead frame");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
    }

    #[test]
    fn from_frame_refuses_a_repeated_members_row() {
        let mut beads = xyz_atoms(&[0.0, 1.0], &[0.0, 0.0], &[0.0, 0.0]);
        beads.insert("bead_type", str_col(&["W", "W"])).unwrap();
        let mut members = Block::new();
        members.insert("ibead", uint_col(&[0, 0])).unwrap();
        members.insert("atom", uint_col(&[10, 10])).unwrap();
        let mut frame = Frame::new();
        frame.insert("beads", beads);
        frame.insert("members", members);

        let err =
            CoarseGrain::from_frame(&frame).expect_err("the (ibead 0, atom 10) row appears twice");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
    }

    // ---- replicate ----

    /// Membership names foreign atoms the copies do not own, so a replicated
    /// bead carries none even when its template bead does.
    #[test]
    fn replicate_copies_no_bead_membership() {
        use crate::op::rigid::Rigid;

        let mut template = CoarseGrain::new();
        let w = template.add_bead("W", 0.0, 0.0, 0.0);
        let p = template.add_bead("P1", 1.0, 0.0, 0.0);
        template.add_bond(w, p).unwrap();
        template.set_bead_members(w, vec![10, 11]);
        template.set_bead_members(p, vec![12]);

        let shifted = Rigid {
            rotation: Rigid::IDENTITY.rotation,
            translation: [5.0, 0.0, 0.0],
        };
        let mut out = CoarseGrain::new();
        let beads = out
            .replicate(&template, &[Rigid::IDENTITY, shifted], &[0, 1])
            .expect("two copies of a two-bead template replicate");

        assert_eq!(beads.len(), 4);
        assert_eq!(out.n_beads(), 4);
        assert_eq!(out.n_bonds(), 2);
        for &bead in &beads {
            assert!(
                out.bead_members(bead).is_empty(),
                "bead {bead:?} was given membership it does not own"
            );
        }
        assert_eq!(template.bead_members(w), &[10, 11], "template untouched");
    }
}
