//! All-atom molecular graph with element-level chemistry semantics.
//!
//! [`Atomistic`] wraps the domain-agnostic [`MolGraph`] and owns **all** atom /
//! bond / angle / dihedral / improper vocabulary: it registers those relation
//! kinds at construction (caching their [`KindId`]s) and exposes the typed
//! convenience API (`add_bond`, `bonds`, `get_bond`, …). `MolGraph` itself knows
//! nothing of bonds — the chemistry lives here.
//!
//! Generic graph methods (`nodes`, `neighbors`, `translate`, `add_relation`, …)
//! remain available via `Deref`/`DerefMut`.
//!
//! # Examples
//!
//! ```
//! use molrs::system::atomistic::Atomistic;
//!
//! let mut mol = Atomistic::new();
//! let c = mol.add_atom_bare("C");
//! let h = mol.add_atom_bare("H");
//! mol.add_bond(c, h).unwrap();
//!
//! assert_eq!(mol.n_atoms(), 2);
//! assert_eq!(mol.n_bonds(), 1);
//! ```

use std::collections::HashMap;
use std::ops::{Deref, DerefMut};

use crate::error::MolRsError;
use crate::store::frame::Frame;
use crate::store::keys;
use crate::system::bond::{BondNumber, BondType};
use crate::system::molgraph::{Atom, KindId, MolGraph, NodeId, PropValue, Relation, RelationId};

/// Result of [`Atomistic::extract_subgraph`].
#[derive(Debug, Clone)]
pub struct ExtractedAtomistic {
    pub graph: Atomistic,
    /// Selected parent atoms with a bond-neighbor outside the ball.
    pub boundary: Vec<AtomId>,
    /// New atom id → parent atom id.
    pub parent_of: HashMap<AtomId, AtomId>,
    /// Parent atom id → hops from nearest center.
    pub hops: HashMap<AtomId, i64>,
    /// Parent atom id → new atom id.
    pub node_map: HashMap<AtomId, AtomId>,
}

/// Handle to an atom (a graph node).
pub type AtomId = NodeId;
/// Handle to a bond (a relation). Distinct-key semantics for `HashSet<BondId>`.
pub type BondId = RelationId;
/// Handle to an angle (a relation).
pub type AngleId = RelationId;
/// Handle to a dihedral (a relation).
pub type DihedralId = RelationId;
/// Handle to an improper (a relation).
pub type ImproperId = RelationId;

/// A bond — a 2-ary relation.
pub type Bond = Relation;
/// An angle (i-j-k) — a 3-ary relation.
pub type Angle = Relation;
/// A dihedral (i-j-k-l) — a 4-ary relation.
pub type Dihedral = Relation;
/// An improper (i-j-k-l) — a 4-ary relation, distinct from a dihedral by kind.
pub type Improper = Relation;

/// All-atom molecular graph.
///
/// Invariant: every atom carries the canonical [`keys::ELEMENT`] property.
#[derive(Debug, Clone)]
pub struct Atomistic {
    graph: MolGraph,
    bond: KindId,
    angle: KindId,
    dihedral: KindId,
    improper: KindId,
}

impl Deref for Atomistic {
    type Target = MolGraph;
    fn deref(&self) -> &MolGraph {
        &self.graph
    }
}

impl DerefMut for Atomistic {
    fn deref_mut(&mut self) -> &mut MolGraph {
        &mut self.graph
    }
}

impl Default for Atomistic {
    fn default() -> Self {
        Self::new()
    }
}

impl Atomistic {
    /// Create an empty all-atom molecular graph with the bond / angle /
    /// dihedral / improper kinds registered.
    pub fn new() -> Self {
        let mut graph = MolGraph::new();
        let bond = graph.register_kind("bonds", 2);
        let angle = graph.register_kind("angles", 3);
        let dihedral = graph.register_kind("dihedrals", 4);
        let improper = graph.register_kind("impropers", 4);
        Self {
            graph,
            bond,
            angle,
            dihedral,
            improper,
        }
    }

    // ---- atoms (nodes) ----

    /// Add an atom carrying a property bag.
    pub fn add_atom(&mut self, atom: Atom) -> AtomId {
        self.graph.add_node_with(atom)
    }

    /// Add an atom with element symbol and 3D coordinates.
    pub fn add_atom_xyz(&mut self, symbol: &str, x: f64, y: f64, z: f64) -> AtomId {
        self.graph.add_node_with(Atom::xyz(symbol, x, y, z))
    }

    /// Add an atom with element symbol only (no coordinates).
    ///
    /// Writes the chemical identity under the canonical [`keys::ELEMENT`] field
    /// (not a format alias such as `"symbol"`).
    pub fn add_atom_bare(&mut self, symbol: &str) -> AtomId {
        let mut a = Atom::new();
        a.set(keys::ELEMENT, symbol);
        self.graph.add_node_with(a)
    }

    /// Remove an atom and all incident bonds / angles / dihedrals / impropers.
    pub fn remove_atom(&mut self, id: AtomId) -> Result<Atom, MolRsError> {
        self.graph.remove_node(id)
    }

    /// Materialize an atom's property bag (owned copy of its set components).
    pub fn get_atom(&self, id: AtomId) -> Result<Atom, MolRsError> {
        self.graph.get_node(id)
    }

    /// Set a single component on an atom.
    pub fn set_atom(
        &mut self,
        id: AtomId,
        key: &str,
        val: impl Into<PropValue>,
    ) -> Result<(), MolRsError> {
        self.graph.set_node(id, key, val)
    }

    /// Clear a single component on an atom (no-op if absent).
    pub fn clear_atom(&mut self, id: AtomId, key: &str) -> Result<(), MolRsError> {
        self.graph.clear_node(id, key)
    }

    /// Iterate over all `(AtomId, Atom)` pairs (each property bag materialized).
    pub fn atoms(&self) -> impl Iterator<Item = (AtomId, Atom)> + '_ {
        self.graph.nodes()
    }

    /// Number of atoms.
    pub fn n_atoms(&self) -> usize {
        self.graph.n_nodes()
    }

    // ---- bonds ----

    /// Add a bond between two existing atoms (default order 1.0).
    pub fn add_bond(&mut self, a: AtomId, b: AtomId) -> Result<BondId, MolRsError> {
        let bid = self.graph.add_relation(self.bond, &[a, b])?;
        self.set_bond_type(bid, BondType::Single)?;
        Ok(bid)
    }

    /// The bond's chemical class; [`BondType::Unknown`] if it has none.
    pub fn bond_type(&self, id: BondId) -> BondType {
        match self.get_bond(id) {
            Ok(b) => BondType::from_prop(b.props.get(keys::BOND_TYPE)),
            Err(_) => BondType::Unknown,
        }
    }

    /// The bond's localized (Kekulé) bond number; [`BondNumber::Unknown`] if it
    /// has none — an aromatic bond before kekulization, for instance.
    pub fn bond_number(&self, id: BondId) -> BondNumber {
        match self.get_bond(id) {
            Ok(b) => BondNumber::from_prop(b.props.get(keys::BOND_NUMBER)),
            Err(_) => BondNumber::Unknown,
        }
    }

    /// Set both facts about a bond at once.
    ///
    /// They are set together because they are only meaningful together: a class
    /// without a number leaves the bond un-standardized, and a number without a
    /// class leaves a renderer no way to tell aromatic from double.
    pub fn set_bond_class(
        &mut self,
        id: BondId,
        bond_type: BondType,
        bond_number: BondNumber,
    ) -> Result<(), MolRsError> {
        self.set_bond_prop(id, keys::BOND_TYPE, bond_type)?;
        self.set_bond_prop(id, keys::BOND_NUMBER, bond_number)
    }

    /// Set a plain (non-aromatic) bond, whose class implies its number.
    ///
    /// `Aromatic` has no implied number, so it must go through
    /// [`set_bond_class`](Self::set_bond_class) with the number a Kekulé
    /// assignment decided.
    pub fn set_bond_type(&mut self, id: BondId, bond_type: BondType) -> Result<(), MolRsError> {
        let number = bond_type.implied_number().unwrap_or_default();
        self.set_bond_class(id, bond_type, number)
    }

    /// Remove a bond.
    pub fn remove_bond(&mut self, id: BondId) -> Result<Bond, MolRsError> {
        self.graph.remove_relation(self.bond, id)
    }

    /// Materialize a bond (endpoints + properties).
    pub fn get_bond(&self, id: BondId) -> Result<Bond, MolRsError> {
        self.graph.get_relation(self.bond, id)
    }

    /// Set a single property on a bond.
    pub fn set_bond_prop(
        &mut self,
        id: BondId,
        key: &str,
        val: impl Into<PropValue>,
    ) -> Result<(), MolRsError> {
        self.graph.set_relation_prop(self.bond, id, key, val)
    }

    /// Iterate over all `(BondId, Bond)` pairs (each materialized).
    pub fn bonds(&self) -> impl Iterator<Item = (BondId, Bond)> + '_ {
        self.graph.relations(self.bond)
    }

    /// Number of bonds.
    pub fn n_bonds(&self) -> usize {
        self.graph.n_relations(self.bond)
    }

    /// Iterate over `(neighbor_id, bond_id)` for a given atom.
    ///
    /// Yields the bond handle rather than a number: a caller that wants the
    /// localized count asks [`bond_number`](Self::bond_number), and one that
    /// wants to know whether the bond is aromatic asks
    /// [`bond_type`](Self::bond_type). Handing back a single float is what let
    /// those two questions be answered by the same value.
    pub fn neighbor_bonds(&self, id: AtomId) -> impl Iterator<Item = (AtomId, BondId)> + '_ {
        self.graph
            .neighbor_relations(id)
            .filter(move |(kind, _, _)| *kind == self.bond)
            .map(move |(_, rid, other)| (other, rid))
    }

    /// Endpoints `(a, b)` of a bond by id, without materializing its property
    /// map (reads only the relation's endpoint list).
    pub fn bond_endpoints(&self, id: BondId) -> Option<(AtomId, AtomId)> {
        self.graph
            .relation_nodes(self.bond, id)
            .ok()
            .map(|eps| (eps[0], eps[1]))
    }

    /// Iterate `(BondId, neighbor_id)` incident to `id` via the adjacency index
    /// (O(degree)), without materializing each bond's property map. The bond
    /// order, if needed, is looked up separately by the caller.
    pub fn incident_bond_ids(&self, id: AtomId) -> impl Iterator<Item = (BondId, AtomId)> + '_ {
        let bond = self.bond;
        self.graph
            .neighbor_relations(id)
            .filter_map(move |(kind, rid, other)| (kind == bond).then_some((rid, other)))
    }

    // ---- angles ----

    /// Add an angle (i-j-k, j central).
    pub fn add_angle(&mut self, i: AtomId, j: AtomId, k: AtomId) -> Result<AngleId, MolRsError> {
        self.graph.add_relation(self.angle, &[i, j, k])
    }

    /// Remove an angle.
    pub fn remove_angle(&mut self, id: AngleId) -> Result<Angle, MolRsError> {
        self.graph.remove_relation(self.angle, id)
    }

    /// Materialize an angle (endpoints + properties).
    pub fn get_angle(&self, id: AngleId) -> Result<Angle, MolRsError> {
        self.graph.get_relation(self.angle, id)
    }

    /// Set a single property on an angle.
    pub fn set_angle_prop(
        &mut self,
        id: AngleId,
        key: &str,
        val: impl Into<PropValue>,
    ) -> Result<(), MolRsError> {
        self.graph.set_relation_prop(self.angle, id, key, val)
    }

    /// Iterate over all `(AngleId, Angle)` pairs (each materialized).
    pub fn angles(&self) -> impl Iterator<Item = (AngleId, Angle)> + '_ {
        self.graph.relations(self.angle)
    }

    /// Number of angles.
    pub fn n_angles(&self) -> usize {
        self.graph.n_relations(self.angle)
    }

    // ---- dihedrals ----

    /// Add a dihedral (i-j-k-l).
    pub fn add_dihedral(
        &mut self,
        i: AtomId,
        j: AtomId,
        k: AtomId,
        l: AtomId,
    ) -> Result<DihedralId, MolRsError> {
        self.graph.add_relation(self.dihedral, &[i, j, k, l])
    }

    /// Remove a dihedral.
    pub fn remove_dihedral(&mut self, id: DihedralId) -> Result<Dihedral, MolRsError> {
        self.graph.remove_relation(self.dihedral, id)
    }

    /// Materialize a dihedral (endpoints + properties).
    pub fn get_dihedral(&self, id: DihedralId) -> Result<Dihedral, MolRsError> {
        self.graph.get_relation(self.dihedral, id)
    }

    /// Set a single property on a dihedral.
    pub fn set_dihedral_prop(
        &mut self,
        id: DihedralId,
        key: &str,
        val: impl Into<PropValue>,
    ) -> Result<(), MolRsError> {
        self.graph.set_relation_prop(self.dihedral, id, key, val)
    }

    /// Iterate over all `(DihedralId, Dihedral)` pairs (each materialized).
    pub fn dihedrals(&self) -> impl Iterator<Item = (DihedralId, Dihedral)> + '_ {
        self.graph.relations(self.dihedral)
    }

    /// Number of dihedrals.
    pub fn n_dihedrals(&self) -> usize {
        self.graph.n_relations(self.dihedral)
    }

    /// Perceive angle and dihedral relations from the bond graph.
    ///
    /// Builds a [`Topology`](crate::system::topology::Topology) (a native
    /// adjacency snapshot) from the current bonds and reuses its
    /// `angles()` / `dihedrals()` enumeration — angles are 2-edge paths
    /// `i-j-k` (deduplicated `i < k`), proper dihedrals are 3-edge paths
    /// `i-j-k-l` (each central edge once). `Atomistic` is just the domain leaf
    /// that names the graph-theoretic result.
    ///
    /// Idempotent: an angle/dihedral already present (by canonical endpoints)
    /// is not duplicated. With `clear_existing`, all existing angle/dihedral
    /// relations of the requested kinds are removed first. Returns
    /// `(n_angles_added, n_dihedrals_added)`.
    pub fn generate_topology(
        &mut self,
        gen_angle: bool,
        gen_dihedral: bool,
        clear_existing: bool,
    ) -> Result<(usize, usize), MolRsError> {
        use crate::system::topology::Topology;

        if clear_existing {
            if gen_angle {
                let ids: Vec<_> = self.graph.relation_ids(self.angle).collect();
                for id in ids {
                    self.graph.remove_relation(self.angle, id)?;
                }
            }
            if gen_dihedral {
                let ids: Vec<_> = self.graph.relation_ids(self.dihedral).collect();
                for id in ids {
                    self.graph.remove_relation(self.dihedral, id)?;
                }
            }
        }

        // Build the native topology snapshot from bonds (atoms in node-id order).
        let atoms: Vec<AtomId> = self.graph.node_ids().collect();
        let pos: std::collections::HashMap<AtomId, usize> =
            atoms.iter().enumerate().map(|(i, &a)| (a, i)).collect();
        let mut edges: Vec<[usize; 2]> = Vec::new();
        for id in self.graph.relation_ids(self.bond) {
            let n = self.graph.relation_nodes(self.bond, id)?;
            if n.len() == 2 {
                edges.push([pos[&n[0]], pos[&n[1]]]);
            }
        }
        let topo = Topology::from_edges(atoms.len(), &edges);

        let mut n_ang = 0usize;
        let mut n_dih = 0usize;

        if gen_angle {
            let mut seen: std::collections::HashSet<Vec<NodeId>> = std::collections::HashSet::new();
            for id in self.graph.relation_ids(self.angle) {
                seen.insert(canonical_path(&self.graph.relation_nodes(self.angle, id)?));
            }
            for a in topo.angles() {
                let nodes = [atoms[a[0]], atoms[a[1]], atoms[a[2]]];
                if seen.insert(canonical_path(&nodes)) {
                    self.add_angle(nodes[0], nodes[1], nodes[2])?;
                    n_ang += 1;
                }
            }
        }

        if gen_dihedral {
            let mut seen: std::collections::HashSet<Vec<NodeId>> = std::collections::HashSet::new();
            for id in self.graph.relation_ids(self.dihedral) {
                seen.insert(canonical_path(
                    &self.graph.relation_nodes(self.dihedral, id)?,
                ));
            }
            for d in topo.dihedrals() {
                let nodes = [atoms[d[0]], atoms[d[1]], atoms[d[2]], atoms[d[3]]];
                if seen.insert(canonical_path(&nodes)) {
                    self.add_dihedral(nodes[0], nodes[1], nodes[2], nodes[3])?;
                    n_dih += 1;
                }
            }
        }

        Ok((n_ang, n_dih))
    }

    /// BFS shortest-path distances over the bond graph from `source`, as
    /// `(atom_id, hops)` pairs for every atom reachable from `source`
    /// (including `source` at distance 0). Unreachable atoms are omitted; an
    /// unknown `source` yields an empty vector.
    ///
    /// The BFS runs directly on the native `MolGraph` adjacency (relations
    /// filtered to the bond kind), keyed by a `SecondaryMap` (O(1),
    /// slot-indexed) — no `Topology` materialization and no
    /// AtomId→contiguous-index remap.
    pub fn topo_distances(&self, source: AtomId, max_hops: Option<i64>) -> Vec<(AtomId, i64)> {
        use slotmap::SecondaryMap;
        use std::collections::VecDeque;

        if self.graph.get_node(source).is_err() {
            return Vec::new();
        }
        let mut dist: SecondaryMap<AtomId, i64> = SecondaryMap::new();
        dist.insert(source, 0);
        let mut queue = VecDeque::new();
        queue.push_back(source);
        while let Some(cur) = queue.pop_front() {
            let d = dist[cur];
            if max_hops.is_some_and(|limit| d >= limit) {
                continue; // atom is at the boundary; do not expand past the radius
            }
            for (kind, _rid, other) in self.graph.neighbor_relations(cur) {
                if kind == self.bond && !dist.contains_key(other) {
                    dist.insert(other, d + 1);
                    queue.push_back(other);
                }
            }
        }
        dist.into_iter().collect()
    }

    // ---- impropers ----

    /// Add an improper dihedral (i-j-k-l; i conventionally central).
    pub fn add_improper(
        &mut self,
        i: AtomId,
        j: AtomId,
        k: AtomId,
        l: AtomId,
    ) -> Result<ImproperId, MolRsError> {
        self.graph.add_relation(self.improper, &[i, j, k, l])
    }

    /// Remove an improper.
    pub fn remove_improper(&mut self, id: ImproperId) -> Result<Improper, MolRsError> {
        self.graph.remove_relation(self.improper, id)
    }

    /// Materialize an improper (endpoints + properties).
    pub fn get_improper(&self, id: ImproperId) -> Result<Improper, MolRsError> {
        self.graph.get_relation(self.improper, id)
    }

    /// Set a single property on an improper.
    pub fn set_improper_prop(
        &mut self,
        id: ImproperId,
        key: &str,
        val: impl Into<PropValue>,
    ) -> Result<(), MolRsError> {
        self.graph.set_relation_prop(self.improper, id, key, val)
    }

    /// Iterate over all `(ImproperId, Improper)` pairs (each materialized).
    pub fn impropers(&self) -> impl Iterator<Item = (ImproperId, Improper)> + '_ {
        self.graph.relations(self.improper)
    }

    /// Number of impropers.
    pub fn n_impropers(&self) -> usize {
        self.graph.n_relations(self.improper)
    }

    // ---- kind handles (for callers that go through the generic API) ----

    /// The bond relation kind.
    pub fn bond_kind(&self) -> KindId {
        self.bond
    }

    // ---- frame conversion ----

    /// Export to a tabular [`Frame`] (atoms / bonds / angles / dihedrals /
    /// impropers blocks).
    pub fn to_frame(&self) -> Frame {
        self.graph.to_frame()
    }

    /// Build from a [`Frame`], registering the standard kinds first so bond /
    /// angle / dihedral / improper blocks are read back.
    pub fn from_frame(frame: &Frame) -> Result<Self, MolRsError> {
        let mut mol = Self::new();
        mol.graph.read_frame(frame)?;
        Ok(mol)
    }

    // ---- conversions ----

    /// Promote from a [`MolGraph`], validating all atoms have [`keys::ELEMENT`].
    /// The graph's relation kinds are re-registered to the standard set.
    pub fn try_from_molgraph(mol: MolGraph) -> Result<Self, MolRsError> {
        for (id, atom) in mol.nodes() {
            if atom.get_str(keys::ELEMENT).is_none() {
                return Err(MolRsError::validation(format!(
                    "node {:?} missing '{}' property",
                    id,
                    keys::ELEMENT
                )));
            }
        }
        let bond = mol.kind_id("bonds").unwrap_or(KindId(0));
        let angle = mol.kind_id("angles").unwrap_or(KindId(1));
        let dihedral = mol.kind_id("dihedrals").unwrap_or(KindId(2));
        let improper = mol.kind_id("impropers").unwrap_or(KindId(3));
        Ok(Self {
            graph: mol,
            bond,
            angle,
            dihedral,
            improper,
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

    // ---- subgraph extraction (see [`crate::system::extract`]) ----

    /// Induced subgraph on an explicit atom set. Stale handles fail-fast.
    /// Returns `(subgraph, parent→new handle map)`.
    ///
    /// Re-wraps via [`Self::try_from_molgraph`] so the `"element"` invariant is
    /// enforced — chemical identity is the canonical ELEMENT field, never a
    /// format alias such as `"symbol"`.
    pub fn induced_subgraph(
        &self,
        atoms: &[AtomId],
    ) -> Result<(Atomistic, HashMap<AtomId, AtomId>), MolRsError> {
        let induced = self.graph.induced_subgraph(atoms)?;
        let atomistic = Atomistic::try_from_molgraph(induced.graph)?;
        Ok((atomistic, induced.node_map))
    }

    /// Radius ball around `centers` over the bond graph.
    ///
    /// When `regenerate_topology` is true, only bonds are copied from the parent
    /// and angles/dihedrals are perceived on the ball (O(ball)). When false,
    /// higher-order terms fully contained in the ball are copied (may scan the
    /// parent's higher-order tables — small-graph / verbatim path).
    pub fn extract_subgraph(
        &self,
        centers: &[AtomId],
        radius: i64,
        regenerate_topology: bool,
        whole_groups: &[Vec<AtomId>],
    ) -> Result<ExtractedAtomistic, MolRsError> {
        let ball = self.graph.extract_ball(
            centers,
            radius,
            self.bond,
            /* copy_higher_order */ !regenerate_topology,
            whole_groups,
        )?;
        let mut atomistic = Atomistic::try_from_molgraph(ball.graph)?;
        if regenerate_topology {
            atomistic.generate_topology(true, true, false)?;
        }
        Ok(ExtractedAtomistic {
            graph: atomistic,
            boundary: ball.boundary,
            parent_of: ball.parent_of,
            hops: ball.hops,
            node_map: ball.node_map,
        })
    }

    /// Structural merge of `other` into `self`. Returns `handle in other → handle
    /// in self`. Handles are remapped (not identity-preserving).
    pub fn merge(&mut self, other: Atomistic) -> HashMap<AtomId, AtomId> {
        self.graph.merge(other.graph)
    }

    /// Independent deep copy. **Handles are preserved** (same generational keys).
    pub fn copy(&self) -> Self {
        self.clone()
    }

    // ---- structural graph hash (see [`crate::system::graph_hash`]) ----

    /// Isomorphism-invariant Weisfeiler–Lehman structural hash of the molecule
    /// (element / charge / aromatic node labels, bond-order edge labels). A
    /// stable, reproducible dedup key — identical for a node-permuted copy.
    pub fn structural_hash(&self) -> u64 {
        crate::system::graph_hash::structural_hash(&self.graph)
    }

    /// Deterministic canonical atom ordering from the WL refinement, so two
    /// isomorphic molecules line up node-by-node (see
    /// [`crate::system::graph_hash::canonical_order`]).
    pub fn canonical_order(&self) -> Vec<AtomId> {
        crate::system::graph_hash::canonical_order(&self.graph)
    }

    /// Whether `self` and `other` are isomorphic as labeled molecular graphs.
    pub fn is_isomorphic(&self, other: &Atomistic) -> bool {
        crate::system::graph_hash::is_isomorphic(&self.graph, &other.graph)
    }

    // Aromaticity perception is a free-function *system*:
    // [`crate::perceive::aromaticity::perceive_aromaticity`]. No algorithm method here.
}

/// Canonical (orientation-independent) key for an angle/dihedral endpoint
/// sequence: the lexicographically smaller of the sequence and its reverse.
/// Matches the canonicalization in
/// [`MolGraph::paths_of_length`](crate::system::molgraph::MolGraph::paths_of_length).
fn canonical_path(nodes: &[NodeId]) -> Vec<NodeId> {
    let fwd = nodes.to_vec();
    let mut rev = fwd.clone();
    rev.reverse();
    if fwd <= rev { fwd } else { rev }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::molgraph::Atom;
    use std::collections::HashSet;

    #[test]
    fn merge_returns_complete_node_map() {
        let mut a = Atomistic::new();
        let c1 = a.add_atom_bare("C");
        let c2 = a.add_atom_bare("C");
        a.add_bond(c1, c2).unwrap();

        let mut b = Atomistic::new();
        let o = b.add_atom_bare("O");
        let h = b.add_atom_bare("H");
        b.add_bond(o, h).unwrap();

        let map = a.merge(b);
        assert_eq!(map.len(), 2);
        assert_eq!(a.n_atoms(), 4);
        assert_eq!(a.n_bonds(), 2);
        let new_o = map[&o];
        let new_h = map[&h];
        assert!(a.get_atom(new_o).unwrap().get_str("element") == Some("O"));
        assert!(a.get_atom(new_h).unwrap().get_str("element") == Some("H"));
        // remapped bond endpoints resolve
        let mut ends = HashSet::new();
        for (_, bond) in a.bonds() {
            ends.extend(bond.nodes.iter().copied());
        }
        assert!(ends.contains(&new_o) && ends.contains(&new_h));
    }

    #[test]
    fn copy_preserves_handles() {
        let mut mol = Atomistic::new();
        let c = mol.add_atom_bare("C");
        let h = mol.add_atom_bare("H");
        let bid = mol.add_bond(c, h).unwrap();
        let cloned = mol.copy();
        assert!(cloned.get_atom(c).is_ok());
        assert!(cloned.get_atom(h).is_ok());
        assert!(cloned.get_bond(bid).is_ok());
    }

    #[test]
    fn test_new_and_add() {
        let mut mol = Atomistic::new();
        let c = mol.add_atom_bare("C");
        let h = mol.add_atom_bare("H");
        mol.add_bond(c, h).unwrap();
        assert_eq!(mol.n_atoms(), 2);
        assert_eq!(mol.n_bonds(), 1);
        assert_eq!(mol.get_atom(c).unwrap().get_str("element"), Some("C"));
    }

    #[test]
    fn test_add_atom_xyz() {
        let mut mol = Atomistic::new();
        let o = mol.add_atom_xyz("O", 0.0, 0.0, 0.0);
        let atom = mol.get_atom(o).unwrap();
        assert_eq!(atom.get_str("element"), Some("O"));
        assert_eq!(atom.get_f64("x"), Some(0.0));
    }

    /// Ethane (C2H6): C0-C1 plus 3 H on each carbon. 7 bonds.
    fn ethane() -> Atomistic {
        let mut mol = Atomistic::new();
        let atoms: Vec<AtomId> = ["C", "C", "H", "H", "H", "H", "H", "H"]
            .iter()
            .map(|e| mol.add_atom_bare(e))
            .collect();
        for (i, j) in [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (1, 6), (1, 7)] {
            mol.add_bond(atoms[i], atoms[j]).unwrap();
        }
        mol
    }

    #[test]
    fn generate_topology_ethane_counts() {
        let mut mol = ethane();
        let (n_ang, n_dih) = mol.generate_topology(true, true, false).unwrap();
        // Angles: C0 centre C(1,2,3) -> C(4,2,3)... 2 C-centres each with 4
        // neighbours -> 2*C(4,2)=12. Dihedrals across the C0-C1 bond: 3*3 = 9.
        assert_eq!(n_ang, 12, "ethane angles");
        assert_eq!(n_dih, 9, "ethane dihedrals");
        assert_eq!(mol.n_angles(), 12);
        assert_eq!(mol.n_dihedrals(), 9);
    }

    #[test]
    fn generate_topology_is_idempotent() {
        let mut mol = ethane();
        mol.generate_topology(true, true, false).unwrap();
        // Second call adds nothing (already present).
        let (n_ang, n_dih) = mol.generate_topology(true, true, false).unwrap();
        assert_eq!((n_ang, n_dih), (0, 0));
        assert_eq!(mol.n_angles(), 12);
        assert_eq!(mol.n_dihedrals(), 9);
    }

    #[test]
    fn generate_topology_clear_existing_regenerates() {
        let mut mol = ethane();
        mol.generate_topology(true, true, false).unwrap();
        let (n_ang, n_dih) = mol.generate_topology(true, true, true).unwrap();
        // clear_existing wipes then regenerates the identical set.
        assert_eq!((n_ang, n_dih), (12, 9));
        assert_eq!(mol.n_angles(), 12);
        assert_eq!(mol.n_dihedrals(), 9);
    }

    #[test]
    fn generate_topology_selective() {
        let mut mol = ethane();
        let (n_ang, n_dih) = mol.generate_topology(true, false, false).unwrap();
        assert_eq!(n_ang, 12);
        assert_eq!(n_dih, 0);
        assert_eq!(mol.n_dihedrals(), 0);
    }

    #[test]
    fn topo_distances_parity() {
        // Native BFS over the bond graph. Asserts the (atom, hops) set directly
        // against the expected hop multisets.
        let hops = |mut v: Vec<(AtomId, i64)>| -> Vec<i64> {
            v.sort();
            v.into_iter().map(|(_, d)| d).collect::<Vec<i64>>()
        };
        let sorted_hops = |v: Vec<(AtomId, i64)>| {
            let mut d: Vec<i64> = v.into_iter().map(|(_, d)| d).collect();
            d.sort_unstable();
            d
        };

        let eth = ethane();
        let atoms: Vec<AtomId> = eth.node_ids().collect();
        // Ethane: C0(0)-C1(1) with H2,H3,H4 on C0 and H5,H6,H7 on C1.
        // From C0: self 0; C1 1; its own H's 1; far H's 2.
        let expected_per_source: [Vec<i64>; 8] = [
            // C0
            vec![0, 1, 1, 1, 1, 2, 2, 2],
            // C1
            vec![1, 0, 2, 2, 2, 1, 1, 1],
            // H2 (on C0)
            vec![1, 2, 0, 2, 2, 3, 3, 3],
            // H3 (on C0)
            vec![1, 2, 2, 0, 2, 3, 3, 3],
            // H4 (on C0)
            vec![1, 2, 2, 2, 0, 3, 3, 3],
            // H5 (on C1)
            vec![2, 1, 3, 3, 3, 0, 2, 2],
            // H6 (on C1)
            vec![2, 1, 3, 3, 3, 2, 0, 2],
            // H7 (on C1)
            vec![2, 1, 3, 3, 3, 2, 2, 0],
        ];
        for (i, &src) in atoms.iter().enumerate() {
            assert_eq!(
                sorted_hops(eth.topo_distances(src, None)),
                sorted_hops(expected_per_source[i].iter().map(|&d| (src, d)).collect()),
                "ethane source {src:?}"
            );
        }

        // 12-atom linear chain: an endpoint sees every hop count 0..=11 once.
        let mut chain = Atomistic::new();
        let ids: Vec<_> = (0..12).map(|_| chain.add_atom_bare("C")).collect();
        for k in 0..ids.len() - 1 {
            chain.add_bond(ids[k], ids[k + 1]).unwrap();
        }
        assert_eq!(
            sorted_hops(chain.topo_distances(ids[0], None)),
            (0..12).collect::<Vec<_>>(),
        );
        // The mid-chain atom 5 sees a symmetric profile.
        assert_eq!(
            hops(chain.topo_distances(ids[0], None)),
            (0..12).collect::<Vec<_>>(),
            "chain endpoint ordered by atom id"
        );

        // Unknown source → empty vec (error path). Use a genuinely-stale id
        // from the same arena (add then remove bumps the slot generation so
        // the old id no longer resolves) — a foreign arena's id can collide on
        // slot+generation and is not a reliable "absent" probe.
        let mut solo = Atomistic::new();
        let stale = solo.add_atom_bare("C");
        solo.remove_atom(stale).unwrap();
        assert!(solo.topo_distances(stale, None).is_empty());
    }

    #[test]
    fn test_full_topology_and_cascade() {
        let mut mol = Atomistic::new();
        let a = mol.add_atom_bare("C");
        let b = mol.add_atom_bare("C");
        let c = mol.add_atom_bare("C");
        let d = mol.add_atom_bare("C");
        mol.add_bond(a, b).unwrap();
        mol.add_bond(a, c).unwrap();
        mol.add_bond(a, d).unwrap();
        mol.add_angle(b, a, c).unwrap();
        mol.add_dihedral(b, a, c, d).unwrap();
        mol.add_improper(b, a, c, d).unwrap();
        assert_eq!(mol.n_bonds(), 3);
        assert_eq!(mol.n_angles(), 1);
        assert_eq!(mol.n_dihedrals(), 1);
        assert_eq!(mol.n_impropers(), 1);
        // typed BondId usable as a HashSet key
        let ids: std::collections::HashSet<BondId> = mol.bonds().map(|(id, _)| id).collect();
        assert_eq!(ids.len(), 3);
        // cascade on central atom
        mol.remove_atom(a).unwrap();
        assert_eq!(mol.n_bonds(), 0);
        assert_eq!(mol.n_angles(), 0);
        assert_eq!(mol.n_dihedrals(), 0);
        assert_eq!(mol.n_impropers(), 0);
    }

    #[test]
    fn test_neighbors_and_neighbor_bonds() {
        let mut mol = Atomistic::new();
        let o = mol.add_atom_bare("O");
        let h1 = mol.add_atom_bare("H");
        let h2 = mol.add_atom_bare("H");
        mol.add_bond(o, h1).unwrap();
        mol.add_bond(o, h2).unwrap();
        assert_eq!(mol.neighbors(o).count(), 2);
        let nb: Vec<(AtomId, BondId)> = mol.neighbor_bonds(o).collect();
        assert_eq!(nb.len(), 2);
        assert!(
            nb.iter()
                .all(|(_, bid)| mol.bond_number(*bid) == BondNumber::Single)
        );
    }

    #[test]
    fn test_frame_roundtrip() {
        let mut mol = Atomistic::new();
        let a = mol.add_atom_xyz("C", 0.0, 0.0, 0.0);
        let b = mol.add_atom_xyz("C", 1.0, 0.0, 0.0);
        let c = mol.add_atom_xyz("C", 0.0, 1.0, 0.0);
        let d = mol.add_atom_xyz("C", 0.0, 0.0, 1.0);
        mol.add_bond(a, b).unwrap();
        mol.add_angle(a, b, c).unwrap();
        mol.add_improper(a, b, c, d).unwrap();
        let frame = mol.to_frame();
        let mol2 = Atomistic::from_frame(&frame).unwrap();
        assert_eq!(mol2.n_atoms(), 4);
        assert_eq!(mol2.n_bonds(), 1);
        assert_eq!(mol2.n_angles(), 1);
        assert_eq!(mol2.n_impropers(), 1);
    }

    #[test]
    fn test_try_from_molgraph_missing_element() {
        let mut g = MolGraph::new();
        g.register_kind("bonds", 2);
        g.add_node_with(Atom::new()); // no element
        assert!(Atomistic::try_from_molgraph(g).is_err());
    }

    #[test]
    fn test_deref_generic_methods() {
        let mut mol = Atomistic::new();
        let c1 = mol.add_atom_bare("C");
        let c2 = mol.add_atom_bare("C");
        mol.add_bond(c1, c2).unwrap();
        // generic MolGraph methods via Deref
        assert_eq!(mol.n_nodes(), 2);
        assert_eq!(mol.neighbors(c1).count(), 1);
    }
}
